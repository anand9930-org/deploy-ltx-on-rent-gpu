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
* ``SimpleDenoiser`` everywhere — every CFGGuider in the workflow has
  ``cfg=1`` which is the no-op CFG case, so guidance machinery is bypassed.
* Stage 1 image conditioning runs through ``LTXVPreprocess(img_compression=18)``
  (CRF=18 H.264 round-trip); Stages 2 and 3 use the resized image directly
  (CRF=0 → ``preprocess()`` becomes the identity). All three stages use
  ``strength=1.0`` (the ``ImageConditioningInput`` default).
* Stage 1 uses an RF-correct ancestral Euler loop (port of ComfyUI's
  ``sample_euler_ancestral_RF``, ``eta=1.0``, ``s_noise=1.0``); Stages 2/3
  fall through to the default non-ancestral Euler in ``DiffusionStage``.
* Stage 3 ``ModalitySpec`` carries the upscaled latent through as
  ``initial_latent`` (the original variant dropped this and restarted Stage 3
  from pure noise).

Non-behavioral divergences (unavoidable framework gaps, listed for honesty):

* ComfyUI uses ``euler_ancestral_cfg_pp`` for Stage 1 and ``euler_cfg_pp`` for
  Stages 2+3. The ``cfg_pp`` (post-projection guidance) hook is a no-op when
  ``cfg=1`` (every CFGGuider in the workflow), so ``euler_ancestral_cfg_pp``
  collapses to ``euler_ancestral`` for our inputs. We port that — the RF
  branch — verbatim in ``_ancestral_euler_denoising_loop``. Stages 2/3
  collapse to plain Euler likewise.
* ComfyUI ``VAEDecodeTiled(temporal_overlap=4)`` — ltx-pipelines requires
  ``tile_overlap_in_frames`` divisible by 8. We use 8. The default 241-frame
  request fits in a single 512-frame temporal tile so overlap is moot.
* ComfyUI uses three independent ``RandomNoise`` seeds (one per stage); we
  thread one ``torch.Generator`` through the noiser AND the Stage 1
  ancestral renoise. Random draws still differ per stage because the
  generator's state advances between calls.
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
# cfg=1 is the no-guidance case; SimpleDenoiser is the equivalent denoiser.

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


def _ancestral_euler_denoising_loop(  # noqa: PLR0913
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
    *,
    generator: torch.Generator,
) -> tuple[LatentState | None, LatentState | None]:
    """Rectified-flow ancestral Euler — verbatim port of ComfyUI's
    ``sample_euler_ancestral_RF`` (``comfy/k_diffusion/sampling.py``)
    with ``eta=1.0`` and ``s_noise=1.0`` hardcoded to match the workflow's
    ``KSamplerSelect(sampler_name="euler_ancestral_cfg_pp")`` defaults.

    LTX is a rectified-flow model (``x_t = (1-σ)·x_data + σ·noise``); ComfyUI
    dispatches ``sample_euler_ancestral`` → ``sample_euler_ancestral_RF`` for
    ``model_sampling.CONST`` checkpoints. The RF variant differs from the
    plain k-diffusion (variance-exploding) form in two load-bearing places:

      * ``sigma_down`` is a fraction of ``sigma_next`` (controlled by ``eta``),
        not derived from a ``sigma**2 - sigma_next**2`` variance budget.
      * After the deterministic Euler step lands at ``sigma_down``, the
        latent is rescaled by ``alpha_ip1 / alpha_down`` (i.e.
        ``(1-σ_next) / (1-σ_down)``). This pulls the signal coefficient
        back to ``1 - σ_next``. Without that rescale, the signal coefficient
        compounds across steps (≈ 1.99×, 1.50×, 1.33× ... against the
        Stage-1 schedule, ~14× over 7 steps) → saturated output.

    The deterministic step is delegated to the existing
    ``EulerDiffusionStep``: passing ``[sigma, sigma_down]`` and
    ``step_idx=0`` produces ``(σ_down/σ)·x + (1 − σ_down/σ)·denoised'``,
    exactly the linear-interp form ComfyUI uses inline.

    Mask handling matches ``_step_state`` in upstream
    ``ltx_pipelines/utils/samplers.py``: ``post_process_latent`` is applied
    to ``denoised`` (collapsing it onto the clean conditioning latent
    weighted by ``denoise_mask``); the renoise term is added uniformly.
    The algebra closes — at conditioned tokens the post-step latent is
    ``(1 − σ_next)·clean + σ_next·noise`` for any mask value.

    Audio runs alongside video by symmetry, sharing the per-pipeline
    ``generator`` so the noise-stream advances deterministically.
    """
    eta = 1.0
    s_noise = 1.0

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)

        sigma = sigmas[step_idx].to(torch.float32)
        sigma_next = sigmas[step_idx + 1].to(torch.float32)

        if sigma_next.item() == 0.0:
            # Final step: collapse to the (post-processed) clean prediction.
            # ComfyUI's RF loop branches with ``if sigmas[i+1] == 0: x = denoised``.
            if video_state is not None and denoised_video is not None:
                pp = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
                video_state = replace(video_state, latent=pp.to(video_state.latent.dtype))
            if audio_state is not None and denoised_audio is not None:
                pp = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)
                audio_state = replace(audio_state, latent=pp.to(audio_state.latent.dtype))
            continue

        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_ip1 = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2).clamp_min(0.0).sqrt()
        alpha_ratio = alpha_ip1 / alpha_down

        # 2-element sigma slice → linear-interp Euler step to sigma_down.
        step_sigmas = torch.stack([sigma, sigma_down])

        if video_state is not None and denoised_video is not None:
            v_dtype = video_state.latent.dtype
            v_pp = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
            v_step = stepper.step(video_state.latent, v_pp, step_sigmas, 0)
            v_noise = torch.randn(
                v_step.shape, dtype=torch.float32, device=v_step.device, generator=generator,
            )
            v_renoised = alpha_ratio * v_step.to(torch.float32) + v_noise * (s_noise * renoise_coeff)
            video_state = replace(video_state, latent=v_renoised.to(v_dtype))

        if audio_state is not None and denoised_audio is not None:
            a_dtype = audio_state.latent.dtype
            a_pp = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)
            a_step = stepper.step(audio_state.latent, a_pp, step_sigmas, 0)
            a_noise = torch.randn(
                a_step.shape, dtype=torch.float32, device=a_step.device, generator=generator,
            )
            a_renoised = alpha_ratio * a_step.to(torch.float32) + a_noise * (s_noise * renoise_coeff)
            audio_state = replace(audio_state, latent=a_renoised.to(a_dtype))

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

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = self.dtype

        # ── Image pre-resize (ResizeImageMaskNode "5016:4990") ──────────────
        # Lanczos to longer-dim=1536 once, before any conditioning hop. The
        # temp dir must outlive every combined_image_conditionings() call;
        # by the time we exit the `with` block, conditionings are GPU tensors
        # and the decoded_video iterator no longer references the images.
        with tempfile.TemporaryDirectory(prefix="ltx_triple_stages_") as tmp:
            images = _resize_images_to_longer_dim(images, COMFY_IMAGE_LONGER_DIM, tmp)

            # ── Text encoding (LTXAVTextEncoderLoader + 2× CLIPTextEncode + LTXVConditioning) ──
            # The negative prompt is encoded but unused: cfg=1 on every CFGGuider
            # collapses to SimpleDenoiser, which doesn't consume negative context.
            # Encoding it anyway mirrors the workflow's CLIPTextEncode-Negative node
            # (and matches its boot-time work).
            ctx_p, _ctx_n = self.prompt_encoder(
                [prompt, negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding

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
                noiser=noiser,
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
                # ComfyUI uses ``euler_ancestral_cfg_pp`` for Stage 1 only;
                # Stages 2/3 use the default non-ancestral Euler.
                loop=functools.partial(_ancestral_euler_denoising_loop, generator=generator),
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
                noiser=noiser,
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
                noiser=noiser,
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
                max_batch_size=max_batch_size,
            )

            # ── Decode (VAEDecodeTiled + LTXVAudioVAEDecode) ────────────────
            decoded_video = self.video_decoder(
                video_state.latent,
                tiling_config or COMFY_TILING_CONFIG,
                generator,
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
                        help="Negative prompt (encoded but unused under cfg=1).")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output video path (.mp4).")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed (single seed; ComfyUI uses three).")
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
