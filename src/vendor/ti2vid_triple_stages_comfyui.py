"""
Three-stage text/image-to-video generation pipeline — ComfyUI workflow port.

Behaviorally mirrors ``scripts/workflow_3mljpp.py`` (the LTX-V 3-stage AV
workflow exported from ComfyUI). This is the byte-for-byte-equivalent
counterpart to ``ti2vid_triple_stages.py``; differences are spelled out
below so future readers can tell which knob came from where.

Stage layout (same as the JSON workflow):

  Stage 1  (small)   height/4 × width/4
           │  8 denoising steps from sigma=1.0  (full-noise start)
           │  manual sigma schedule "1.0, 0.99375, 0.9875, 0.98125,
           │   0.975, 0.909375, 0.725, 0.421875, 0.0"
           │  RF-correct ancestral Euler (eta=1.0, s_noise=1.0)
           │  image conditioning preprocessed at H.264 CRF=18
           ▼
           LTXVLatentUpsampler (×2 spatial)
           │
  Stage 2  (mid)     height/2 × width/2
           │  3 denoising steps from sigma=0.85  (refine)
           │  manual sigma schedule "0.85, 0.7250, 0.4219, 0.0"
           │  image conditioning at native resolution (no preprocess)
           ▼
           LTXVLatentUpsampler (×2 spatial)
           │
  Stage 3  (full)    height × width
                3 denoising steps from sigma=0.85  (final refine)
                manual sigma schedule "0.85, 0.7250, 0.4219, 0.0"
                image conditioning at native resolution (no preprocess)

  Decode:    VAEDecodeTiled (tile=512, overlap=64) + LTXVAudioVAEDecode

Key differences from ``ti2vid_triple_stages.py``:

* ``LTX2Scheduler`` (Karras-style) → manual sigmas pinned to the workflow's
  ``ManualSigmas`` literals.
* Upsample chain runs **twice** (1→2 and 2→3); the original vendor variant
  only upsampled once (1→2) and ran Stage 3 at the same resolution as Stage 2.
* Distilled LoRA at strength 0.5 is applied to **all three** stages (matches
  the single ``LoraLoaderModelOnly`` whose output feeds every CFGGuider in
  the JSON), not just the final stage.
* ``cfg=1`` on every CFGGuider, but **not** the no-op CFG case. ComfyUI's
  cfg++ samplers register their post-CFG hook with
  ``disable_cfg1_optimization=True``, so even at cfg=1 ComfyUI runs **two**
  model forwards per step: a conditional pass on the (empty) positive prompt
  → ``denoised`` (the x0 estimate) and an unconditional pass on the long
  negative prompt → ``uncond_denoised``, which the cfg++ ODE derivative uses.
  We mirror this with two ``SimpleDenoiser`` passes per step (positive and
  negative context) — the negative prompt is load-bearing here.
* Stage 1 image conditioning runs through ``LTXVPreprocess(img_compression=18)``
  (CRF=18 H.264 round-trip); Stages 2 and 3 use the resized image directly
  (CRF=0 → ``preprocess()`` becomes the identity). All three stages use
  ``strength=1.0`` (the ``ImageConditioningInput`` default).
* Samplers come from vendored ComfyUI (``src/vendor/comfy_sampling.py``).
  Stage 1 uses ComfyUI's ``euler_ancestral_cfg_pp`` (eta=1.0, s_noise=1.0);
  Stages 2/3 use ``euler_cfg_pp`` (eta=0.0, s_noise=0.0). The per-step math
  is ``cfgpp_denoising_step`` (verbatim from ComfyUI's
  ``sample_euler_ancestral_cfg_pp`` body). We apply it in a joint AV loop:
  per step the conditional and unconditional denoisers each run one forward
  (matching ComfyUI's ``calc_cond_batch`` with ``disable_cfg1_optimization=True``)
  and the cfg++ step is applied to each modality independently (the step math
  is element-wise).
* Stage 3 ``ModalitySpec`` carries the upscaled latent through as
  ``initial_latent`` (the original variant dropped this and restarted Stage 3
  from pure noise).

Non-behavioral divergences (unavoidable framework gaps, listed for honesty):

* ComfyUI ``VAEDecodeTiled(temporal_overlap=4)`` — ltx-pipelines requires
  ``tile_overlap_in_frames`` divisible by 8. We use 8. The default 241-frame
  request fits in a single 512-frame temporal tile so overlap is moot.
* ComfyUI uses three independent ``RandomNoise`` seeds (one per stage, each
  also seeding that stage's ancestral renoise via
  ``default_noise_sampler(x, seed)``). We derive three seeds from the request
  ``seed`` (``_derive_stage_seeds``) and give each stage its own
  ``torch.Generator`` driving both its initial noise and its renoise — same
  structure as the workflow, just not the workflow's literal seed values
  (which ``scripts/workflow_3mljpp.py`` re-randomises at runtime anyway).
"""

import argparse
import functools
import logging
import tempfile
from collections.abc import Iterator
from dataclasses import replace

import torch
from PIL import Image as _PILImage
from tqdm import tqdm

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer import X0Model
from ltx_core.model.video_vae import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    get_video_chunks_number,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
    post_process_latent,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import Denoiser, ModalitySpec


# ── ComfyUI workflow constants — pinned to the JSON literals ────────────────
# Each block below is a 1:1 mirror of a ComfyUI node from
# scripts/workflow_3mljpp.py. Touching these means the pipeline stops
# matching the workflow.

# ManualSigmas "5002:4984" — Stage 1 schedule, front-loaded.
COMFY_STAGE_1_SIGMAS = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)

# ManualSigmas "5001:4985" — Stage 2 schedule, refining from 0.85.
COMFY_STAGE_2_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)

# ManualSigmas "5012:5006" — Stage 3 schedule (identical to Stage 2).
COMFY_STAGE_3_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)

# LTXVPreprocess("5013:3336") — img_compression=18 → H.264 CRF=18 on Stage 1
# image only. Stages 2/3 take the resized image without preprocessing
# (modeled here as CRF=0 → ltx_pipelines.media_io.preprocess() short-circuits).
COMFY_STAGE_1_IMAGE_CRF = 18

# ResizeImageMaskNode("5016:4990") — scale longer dimension to 1536 with
# Lanczos. The workflow runs this once and feeds the result to BOTH
# LTXVPreprocess (Stage 1) and LTXVImgToVideoConditionOnly (Stages 2/3).
# Without this intermediate hop, combined_image_conditionings would resize
# native → stage h/w directly, which is not numerically equivalent.
COMFY_IMAGE_LONGER_DIM = 1536

# LoraLoaderModelOnly("5025:5023") — strength_model=0.5; output feeds every
# CFGGuider, so the LoRA is live across all three stages.
COMFY_DISTILLED_LORA_STRENGTH = 0.5

# CFGGuider("5002:4828", "5001:4964", "5012:5005") — cfg=1 on every stage.
# Not the no-guidance case: ComfyUI's cfg++ samplers run the negative-prompt
# uncond pass even at cfg=1 (disable_cfg1_optimization). _cfgpp_denoising_loop
# therefore runs two SimpleDenoiser passes (positive + negative) per step.

# VAEDecodeTiled("5027:4851") — tile_size=512, overlap=64, temporal_size=512,
# temporal_overlap=4 (rounded to 8 to satisfy TemporalTilingConfig).
COMFY_TILING_CONFIG = TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=512, tile_overlap_in_frames=8),
)

# Default frame count / fps from the workflow's PrimitiveInt/Float nodes.
COMFY_DEFAULT_NUM_FRAMES = 241
COMFY_DEFAULT_FRAME_RATE = 24.0

# Long stylistic-negative used by the workflow's CLIPTextEncode (negative).
COMFY_DEFAULT_NEGATIVE_PROMPT = (
    "camera zooming out, low resolution, blurry, grainy, pixelated, "
    "wide shot, distant view, shallow focus, motion blur, low detail, "
    "flat lighting, dark scene, noisy image, poor texture, soft edges, "
    "muted colors, overexposed, underexposed, static, silent, no movement, "
    "blurry, low quality, still frame, frames, watermark, overlay, titles, "
    "has subtitles, Deformed, scene cut, scene transition, no movement, "
    "glitching, low resolution, extra hands appearing, extra limbs appearing, "
    "warping, extra body parts"
)


# RandomNoise("5002:4832"=727273229127121, "5001:4967"=200996433497366,
# "5012:5009"=975078551246030) — three independent seeds, one per stage. The
# literals aren't load-bearing (scripts/workflow_3mljpp.py re-randomises them at
# runtime), so we derive three deterministic per-stage seeds from the request
# seed instead — keeps the API single-seed and reproducible. Within a stage that
# seed drives both the initial latent noise (GaussianNoiser) and the ancestral
# renoise (_cfgpp_denoising_loop), mirroring ComfyUI's RandomNoise(seed) +
# default_noise_sampler(x, seed).
_STAGE_SEED_MASK = (1 << 63) - 1
_STAGE_SEED_SALTS = (0, 0x9E3779B97F4A7C15, 0x2545F4914F6CDD1D)


def _derive_stage_seeds(seed: int) -> tuple[int, int, int]:
    """Spread one request seed into three decorrelated per-stage seeds."""
    base = seed & _STAGE_SEED_MASK
    s1, s2, s3 = ((base ^ salt) & _STAGE_SEED_MASK for salt in _STAGE_SEED_SALTS)
    return s1, s2, s3


def _assert_quad_resolution(height: int, width: int) -> None:
    """Assert that height/width support a 4× downscale chain.

    Stage 1 runs at height/4 × width/4, and the LTX VAE quantises to /32, so
    the final dimensions must be divisible by 128 (4×32). Workflow defaults
    (1024×1536) satisfy this; this guard exists for callers that pick custom
    resolutions.
    """
    if height % 128 != 0 or width % 128 != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) must be divisible by 128 for the "
            f"three-stage 4× chain (Stage 1 = height/4 × width/4 must be /32)."
        )


def _resize_images_to_longer_dim(
    images: list[ImageConditioningInput],
    target: int,
    out_dir: str,
) -> list[ImageConditioningInput]:
    """Mirror ResizeImageMaskNode("scale longer dimension", lanczos) once per
    call. Returns new ImageConditioningInputs whose ``path`` points at the
    resized PNG inside ``out_dir`` (callers must keep that dir alive until
    the last conditioning call returns)."""
    out: list[ImageConditioningInput] = []
    for i, img in enumerate(images):
        with _PILImage.open(img.path) as src:
            src = src.convert("RGB")
            w, h = src.size
            longer = max(w, h)
            if longer > target:
                scale = target / longer
                new_size = (int(round(w * scale)), int(round(h * scale)))
                resized = src.resize(new_size, _PILImage.LANCZOS)
            else:
                resized = src.copy()
            out_path = f"{out_dir}/img_{i}.png"
            resized.save(out_path)
        out.append(img._replace(path=out_path))
    return out


def _cfgpp_denoising_loop(  # noqa: PLR0913
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,  # noqa: ARG001 — accepted for loop= signature; unused
    transformer: X0Model,
    denoiser: Denoiser,
    *,
    neg_denoiser: Denoiser,
    generator: torch.Generator,
    eta: float,
    s_noise: float,
) -> tuple[LatentState | None, LatentState | None]:
    """Joint audio-video denoising loop using ComfyUI's CFG++ ancestral Euler.

    Per-step math is delegated to ``src.vendor.comfy_sampling.cfgpp_denoising_step``,
    lifted verbatim from the else-branch of
    ``comfy.k_diffusion.sampling.sample_euler_ancestral_cfg_pp``.

    Choice of ``eta``/``s_noise``:

    * ``(1.0, 1.0)`` reproduces ComfyUI's ``euler_ancestral_cfg_pp`` (Stage 1).
    * ``(0.0, 0.0)`` reproduces ComfyUI's ``euler_cfg_pp`` (Stages 2/3).

    Two denoisers, two forwards per step. ``denoiser`` carries the positive
    context (the workflow's empty prompt) and produces the x0 estimate
    ``denoised``; ``neg_denoiser`` carries the negative context (the long
    quality prompt) and produces ``uncond_denoised``, which the cfg++
    derivative ``d = to_d(x, sigma, alpha_s · uncond_denoised)`` consumes.
    ComfyUI's ``sample_euler_ancestral_cfg_pp`` registers its post-CFG hook
    with ``disable_cfg1_optimization=True``, so it runs both passes even at
    cfg=1 — folding ``uncond_denoised`` into ``denoised`` (the pre-2026-05
    behaviour here) made the cfg++ direction wrong on every step. The cfg++
    step is element-wise, so applying it per modality after one joint forward
    apiece is identical to ComfyUI's single-tensor sampler run per modality.
    At ``max_batch_size=1`` the two passes run sequentially (one B=1 forward
    each) — numerically the same as one batched B=2 forward; batching is a
    latency optimisation, not a correctness requirement.

    ``stepper`` is unused (kept for the ``loop=`` callable signature accepted
    by ``DiffusionStage.run_denoising``).

    Mask handling: ``post_process_latent`` is applied to BOTH the conditional
    and unconditional ``denoised`` outputs before stepping, so LTX
    image-conditioning frames stay pinned to ``clean_latent`` at sampling time
    — matching ComfyUI's ``KSamplerX0Inpaint``, which wraps the model *before*
    the CFG combine and thus inpaints both passes.

    Both noise samplers share this loop's ``generator`` (renoise stream
    advances across video then audio per step). Each stage gets its own
    generator (see ``__call__``); within a stage it also drives the initial
    latent noise — mirroring ComfyUI, where one ``RandomNoise`` seed feeds
    both ``SamplerCustomAdvanced`` and ``default_noise_sampler(x, seed)``.
    ``torch.randn`` in fp32 matches the pre-refactor renoise dtype (the step
    fn casts everything to fp32 anyway).
    """
    from src.vendor.comfy_sampling import cfgpp_denoising_step

    def make_noise_sampler(latent_template: torch.Tensor):
        return lambda sigma, sigma_next: torch.randn(
            latent_template.shape, dtype=torch.float32,
            device=latent_template.device, generator=generator,
        )

    video_noise_sampler = make_noise_sampler(video_state.latent) if video_state is not None else None
    audio_noise_sampler = make_noise_sampler(audio_state.latent) if audio_state is not None else None

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        # Conditional (positive/empty prompt) → x0 estimate; unconditional
        # (negative prompt) → uncond_denoised used by the cfg++ derivative.
        # Mirrors ComfyUI's calc_cond_batch with disable_cfg1_optimization=True.
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        uncond_video, uncond_audio = neg_denoiser(transformer, video_state, audio_state, sigmas, step_idx)

        sigma = sigmas[step_idx].to(torch.float32)
        sigma_next = sigmas[step_idx + 1].to(torch.float32)

        if video_state is not None and denoised_video is not None:
            v_dtype = video_state.latent.dtype
            # ComfyUI applies KSamplerX0Inpaint to the model output *before* the
            # CFG combine, so both passes get the conditioning-frame pin.
            v_cond = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
            v_uncond = post_process_latent(uncond_video, video_state.denoise_mask, video_state.clean_latent)
            v_new = cfgpp_denoising_step(
                video_state.latent.to(torch.float32),
                v_cond.to(torch.float32),
                v_uncond.to(torch.float32),
                sigma, sigma_next,
                eta=eta, s_noise=s_noise,
                noise_sampler=video_noise_sampler,
            )
            video_state = replace(video_state, latent=v_new.to(v_dtype))

        if audio_state is not None and denoised_audio is not None:
            a_dtype = audio_state.latent.dtype
            a_cond = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)
            a_uncond = post_process_latent(uncond_audio, audio_state.denoise_mask, audio_state.clean_latent)
            a_new = cfgpp_denoising_step(
                audio_state.latent.to(torch.float32),
                a_cond.to(torch.float32),
                a_uncond.to(torch.float32),
                sigma, sigma_next,
                eta=eta, s_noise=s_noise,
                noise_sampler=audio_noise_sampler,
            )
            audio_state = replace(audio_state, latent=a_new.to(a_dtype))

    return video_state, audio_state


class TI2VidTripleStagesComfyUIPipeline:
    """Three-stage image-to-video pipeline mirroring scripts/workflow_3mljpp.py.

    Stage 1 generates a small base latent (height/4 × width/4) with image
    conditioning that has been H.264-compressed at CRF=18. The latent is
    upsampled 2× spatially, refined for 3 steps at sigma=0.85 (Stage 2),
    upsampled 2× again, then refined for another 3 steps at sigma=0.85
    (Stage 3) before VAE decode.

    Distilled LoRA at strength 0.5 is loaded once and reused across all
    three stages. Audio runs alongside video the whole way, with the audio
    latent state threaded through each stage's ``ModalitySpec``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16
        self.checkpoint_path = checkpoint_path
        self.registry = registry

        self.prompt_encoder = PromptEncoder(checkpoint_path, gemma_root, self.dtype, self.device, registry=registry)
        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry)
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, self.device, registry=registry)

        # ComfyUI loads the distilled LoRA exactly once (LoraLoaderModelOnly,
        # strength_model=0.5) and feeds the same LoRA-applied model into all
        # three CFGGuiders. Mirror by stacking it on every DiffusionStage.
        all_loras = (*tuple(loras), *tuple(distilled_lora))
        stage_kwargs = dict(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=all_loras,
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.stage_1 = DiffusionStage(**stage_kwargs)
        self.stage_2 = DiffusionStage(**stage_kwargs)
        self.stage_3 = DiffusionStage(**stage_kwargs)

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        _assert_quad_resolution(height=height, width=width)

        # One generator per stage (matches ComfyUI's three RandomNoise nodes).
        # Each stage's generator drives both its initial latent noise
        # (GaussianNoiser, below) and its ancestral renoise (_cfgpp_denoising_loop).
        stage_seeds = _derive_stage_seeds(seed)
        stage_generators = tuple(
            torch.Generator(device=self.device).manual_seed(s) for s in stage_seeds
        )
        dtype = self.dtype

        # ── Image pre-resize (ResizeImageMaskNode "5016:4990") ──────────────
        # Lanczos to longer-dim=1536 once, before any conditioning hop. The
        # temp dir must outlive every combined_image_conditionings() call;
        # by the time we exit the `with` block, conditionings are GPU tensors
        # and the decoded_video iterator no longer references the images.
        with tempfile.TemporaryDirectory(prefix="ltx_triple_stages_") as tmp:
            images = _resize_images_to_longer_dim(images, COMFY_IMAGE_LONGER_DIM, tmp)

            # ── Text encoding (LTXAVTextEncoderLoader + 2× CLIPTextEncode + LTXVConditioning) ──
            # Both the positive (empty) and negative prompts are used: ComfyUI's
            # cfg++ samplers run a negative-prompt uncond pass even at cfg=1
            # (disable_cfg1_optimization), and the cfg++ ODE direction depends
            # on it. See _cfgpp_denoising_loop.
            ctx_p, ctx_n = self.prompt_encoder(
                [prompt, negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

            # ── Sigma schedules — pinned to the workflow's ManualSigmas literals ──
            stage_1_sigmas = torch.tensor(COMFY_STAGE_1_SIGMAS, dtype=torch.float32, device=self.device)
            stage_2_sigmas = torch.tensor(COMFY_STAGE_2_SIGMAS, dtype=torch.float32, device=self.device)
            stage_3_sigmas = torch.tensor(COMFY_STAGE_3_SIGMAS, dtype=torch.float32, device=self.device)

            # ── Stage 1: base resolution (height/4 × width/4) ────────────────
            stage_1_height = height // 4
            stage_1_width = width // 4

            # LTXVPreprocess(img_compression=18) → applies only to Stage 1 inputs.
            # Strength stays at the ImageConditioningInput default (1.0) for all stages.
            stage_1_images = [img._replace(crf=COMFY_STAGE_1_IMAGE_CRF) for img in images]
            stage_1_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=stage_1_images,
                    height=stage_1_height,
                    width=stage_1_width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
            )

            video_state, audio_state = self.stage_1(
                denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
                sigmas=stage_1_sigmas,
                noiser=GaussianNoiser(generator=stage_generators[0]),
                width=stage_1_width,
                height=stage_1_height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=stage_1_conditionings,
                    noise_scale=stage_1_sigmas[0].item(),
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                    noise_scale=stage_1_sigmas[0].item(),
                ),
                # ComfyUI's ``euler_ancestral_cfg_pp`` (eta=1.0, s_noise=1.0) +
                # the negative-prompt uncond pass (disable_cfg1_optimization).
                loop=functools.partial(
                    _cfgpp_denoising_loop,
                    neg_denoiser=SimpleDenoiser(v_context=v_context_n, a_context=a_context_n),
                    generator=stage_generators[0], eta=1.0, s_noise=1.0,
                ),
                max_batch_size=max_batch_size,
            )

            # ── Stage 1 → Stage 2: 2× LTXVLatentUpsampler ───────────────────
            upscaled_video_latent = self.upsampler(video_state.latent[:1])

            # ── Stage 2: half resolution (height/2 × width/2) ────────────────
            stage_2_height = height // 2
            stage_2_width = width // 2

            # Stages 2 & 3 use the resized image WITHOUT the H.264 round-trip
            # (LTXVPreprocess is wired only to the Stage 1 LTXVImgToVideoConditionOnly).
            stages_23_images = [img._replace(crf=0) for img in images]
            stage_2_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=stages_23_images,
                    height=stage_2_height,
                    width=stage_2_width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
            )

            video_state, audio_state = self.stage_2(
                denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
                sigmas=stage_2_sigmas,
                noiser=GaussianNoiser(generator=stage_generators[1]),
                width=stage_2_width,
                height=stage_2_height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=upscaled_video_latent,
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=audio_state.latent,
                ),
                # ComfyUI's ``euler_cfg_pp`` (eta=0.0, s_noise=0.0) +
                # the negative-prompt uncond pass (disable_cfg1_optimization).
                loop=functools.partial(
                    _cfgpp_denoising_loop,
                    neg_denoiser=SimpleDenoiser(v_context=v_context_n, a_context=a_context_n),
                    generator=stage_generators[1], eta=0.0, s_noise=0.0,
                ),
                max_batch_size=max_batch_size,
            )

            # ── Stage 2 → Stage 3: 2× LTXVLatentUpsampler ───────────────────
            upscaled_video_latent_2 = self.upsampler(video_state.latent[:1])

            # ── Stage 3: full resolution (height × width) ────────────────────
            stage_3_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=stages_23_images,
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
            )

            video_state, audio_state = self.stage_3(
                denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
                sigmas=stage_3_sigmas,
                noiser=GaussianNoiser(generator=stage_generators[2]),
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=stage_3_conditionings,
                    noise_scale=stage_3_sigmas[0].item(),
                    initial_latent=upscaled_video_latent_2,
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                    noise_scale=stage_3_sigmas[0].item(),
                    initial_latent=audio_state.latent,
                ),
                # ComfyUI's ``euler_cfg_pp`` (eta=0.0, s_noise=0.0) +
                # the negative-prompt uncond pass (disable_cfg1_optimization).
                loop=functools.partial(
                    _cfgpp_denoising_loop,
                    neg_denoiser=SimpleDenoiser(v_context=v_context_n, a_context=a_context_n),
                    generator=stage_generators[2], eta=0.0, s_noise=0.0,
                ),
                max_batch_size=max_batch_size,
            )

            # ── Decode (VAEDecodeTiled + LTXVAudioVAEDecode) ────────────────
            decoded_video = self.video_decoder(
                video_state.latent,
                tiling_config or COMFY_TILING_CONFIG,
                stage_generators[2],
            )
            decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Three-stage AV image-to-video — ComfyUI workflow port. "
            "Behaviorally mirrors scripts/workflow_3mljpp.py."
        )
    )
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to LTX-2.3 checkpoint (ltx-2.3-22b-dev.safetensors).")
    parser.add_argument("--distilled-lora", type=str, action="append", dest="distilled_loras",
                        default=[], help="Distilled LoRA path "
                                         "(default strength=0.5; matches LoraLoaderModelOnly).")
    parser.add_argument("--spatial-upsampler-path", type=str, required=True,
                        help="Path to spatial upsampler "
                             "(ltx-2.3-spatial-upscaler-x2-1.0.safetensors).")
    parser.add_argument("--gemma-root", type=str, required=True,
                        help="Path to Gemma text encoder root.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Positive text prompt. The workflow ships an empty string here.")
    parser.add_argument("--negative-prompt", type=str, default=COMFY_DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt (drives the cfg++ uncond pass; "
                             "ComfyUI runs it even at cfg=1).")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output video path (.mp4).")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed; three per-stage seeds are derived "
                             "from it (ComfyUI uses three independent seeds).")
    parser.add_argument("--width", type=int, default=896,
                        help="Final output width. Must be divisible by 128. "
                             "Default 896 mirrors workflow's hardcoded "
                             "EmptyLTXVLatentVideo(width=224) × 4× chain.")
    parser.add_argument("--height", type=int, default=1280,
                        help="Final output height. Must be divisible by 128. "
                             "Default 1280 mirrors workflow's hardcoded "
                             "EmptyLTXVLatentVideo(height=320) × 4× chain.")
    parser.add_argument("--num-frames", type=int, default=COMFY_DEFAULT_NUM_FRAMES,
                        help="Number of frames (must satisfy (8*K)+1).")
    parser.add_argument("--frame-rate", type=float, default=COMFY_DEFAULT_FRAME_RATE,
                        help="Frame rate.")
    parser.add_argument("--lora", action="append", dest="loras", default=[],
                        help="Additional LoRA (path [strength]). Stacked under the distilled LoRA.")
    parser.add_argument("--image", action="append", dest="images", default=[],
                        help="Image conditioning: 'PATH FRAME_IDX'. "
                             "strength is fixed at 1.0 to match the workflow.")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Max batch size per transformer forward pass.")
    parser.add_argument("--quantization", type=str, choices=["fp8-cast", "fp8-scaled-mm"],
                        help="Quantization policy.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile.")
    return parser


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = build_arg_parser()
    args = parser.parse_args()

    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy

    # Distilled LoRA — default strength 0.5 to match LoraLoaderModelOnly.
    distilled_loras = []
    for d in args.distilled_loras:
        parts = d.split()
        path = parts[0]
        strength = float(parts[1]) if len(parts) > 1 else COMFY_DISTILLED_LORA_STRENGTH
        distilled_loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))

    loras = []
    for lora_arg in args.loras:
        parts = lora_arg.split()
        path = parts[0]
        strength = float(parts[1]) if len(parts) > 1 else 1.0
        loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))

    # Image conditioning — strength fixed at 1.0 to match
    # LTXVImgToVideoConditionOnly(strength=1.0). CRF is set per-stage inside
    # __call__; the user-supplied value (default in ImageConditioningInput) is
    # overwritten there.
    images = []
    for img_arg in args.images:
        parts = img_arg.split()
        path, frame_idx = parts[0], int(parts[1])
        images.append(ImageConditioningInput(path=path, frame_idx=frame_idx, strength=1.0))

    quantization = None
    if args.quantization == "fp8-cast":
        quantization = QuantizationPolicy.fp8_cast()
    elif args.quantization == "fp8-scaled-mm":
        quantization = QuantizationPolicy.fp8_scaled_mm()

    pipeline = TI2VidTripleStagesComfyUIPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_loras,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=loras,
        quantization=quantization,
        torch_compile=args.compile,
    )

    video_chunks_number = get_video_chunks_number(args.num_frames, COMFY_TILING_CONFIG)

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=images,
        tiling_config=COMFY_TILING_CONFIG,
        max_batch_size=args.max_batch_size,
    )

    encode_video(
        video=video,
        fps=int(args.frame_rate),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
