"""Three-stage image-to-video pipeline — runs the real ComfyUI node graph.

This is a hand-written, readable cascade over the actual **ComfyUI core** node
classes that the workflow ``~/Downloads/3mljpp-api.json`` (transcribed in
``scripts/workflow_3mljpp.py``) uses — NOT a reimplementation on ``ltx_pipelines``
blocks (that was the old ``ti2vid_triple_stages_comfyui.py``, which kept diverging)
and NOT ComfyUI's graph executor (too much caching/lazy-eval churn). Every operation
is one node call; the literals below are pinned to the workflow's node inputs.

We use ComfyUI **core** nodes only — no ComfyUI-LTXVideo. The workflow's two
LTXVideo-specific nodes have core equivalents: ``LTXVImgToVideoConditionOnly`` →
core's ``LTXVImgToVideoInplace`` (byte-equivalent conditioning logic; the LTXVideo
node was upstreamed — core even keeps a ``generate = execute`` alias), and
``LTXFloatToInt`` → ``round(frame_rate)`` inline (the node's body is literally
``round(a)``). The other LTX nodes (``LTXAVTextEncoderLoader``, ``LTXVAudioVAELoader``,
``LTXVEmptyLatentAudio``, ``LTXVAudioVAEDecode``, ``LTXVConditioning``,
``LTXVPreprocess``, ``LTXVConcatAVLatent``, ``LTXVSeparateAVLatent``,
``LTXVLatentUpsampler``) are all in ComfyUI core's ``comfy_extras/nodes_lt*.py``.
This matters because core nodes use absolute imports — they can be imported and
called directly with no ComfyUI runtime beyond the three-line bootstrap in
``src/comfyui_runtime.py`` (no ``nodes.init_extra_nodes()`` / asyncio / ``PromptServer``);
ComfyUI-LTXVideo's modules use relative imports (``from .nodes_registry import …``)
and would need the full runtime, which we deliberately avoid.

Trade-off (accepted): this path uses ComfyUI's own model loading, so it does NOT
get the repo's FA3 / FP8-scaled_mm / torch.compile patches against ``ltx_core``
that the other pipelines rely on — it's BF16 + ComfyUI's defaults, ~1.3–2× slower
on H100. See ``src/comfyui_runtime.py`` for the bootstrap, the Dockerfile for the
pinned ComfyUI SHA.

Module import is ComfyUI-free (the node imports live inside ``__init__``), so this
module — and the ``DEFAULT_NEGATIVE_PROMPT`` constant ``service.py`` imports — can
be imported on a box without ComfyUI installed (e.g. the test suite).

Stage layout (mirrors the JSON):

  Stage 1  (height/4 × width/4)  euler_ancestral_cfg_pp, 8 steps from sigma 1.0,
           │                     image conditioning preprocessed at H.264 CRF 18
           ▼ LTXVLatentUpsampler ×2 spatial
  Stage 2  (height/2 × width/2)  euler_cfg_pp, 3 steps from sigma 0.85,
           │                     image re-conditioned at native res (no preprocess)
           ▼ LTXVLatentUpsampler ×2 spatial
  Stage 3  (height × width)       euler_cfg_pp, 3 steps from sigma 0.85,
                                  image re-conditioned at native res (no preprocess)
  Decode:  VAEDecodeTiled (512/64/512/4) + LTXVAudioVAEDecode
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from typing import Any, NamedTuple

from src import comfyui_runtime


class ImageInput(NamedTuple):
    path: str
    frame_idx: int
    strength: float

logger = logging.getLogger(__name__)


# ── Workflow literals — pinned to the 3mljpp-api.json node inputs ───────────
# (node-id references are the JSON's, e.g. "5002:4984"; scripts/workflow_3mljpp.py
#  has the full transcription.)

# CheckpointLoaderSimple "5025:5022" / LTXVAudioVAELoader "5025:5024" /
# LTXAVTextEncoderLoader "5025:5017" ckpt_name.
COMFY_CKPT_NAME = "ltx-2.3-22b-dev-fp8.safetensors"
# LoraLoaderModelOnly "5025:5023" — the JSON names "...-384.safetensors"; the
# image ships the "-384-1.1" build (see src/download_models.py). strength 0.5.
COMFY_DISTILLED_LORA_NAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
COMFY_DISTILLED_LORA_STRENGTH = 0.5
# LTXAVTextEncoderLoader "5025:5017" text_encoder — the JSON's literal, a
# consolidated single-file Gemma. src/download_models.py fetches it from
# Comfy-Org/ltx-2 (split_files/text_encoders/gemma_3_12B_it.safetensors) and
# comfyui_runtime registers <model_dir>/split_files/text_encoders/ (and a plain
# <model_dir>/text_encoders/) as "text_encoders" dirs, so ComfyUI resolves this name.
COMFY_TEXT_ENCODER_NAME = "gemma_3_12B_it.safetensors"
# LatentUpscaleModelLoader "5025:5021" — JSON names "...-x2-1.0"; image ships "-1.1".
COMFY_UPSCALER_NAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# ResizeImageMaskNode "5016:4990" — scale longer dimension to 1536, lanczos.
COMFY_IMAGE_LONGER_DIM = 1536
# LTXVPreprocess "5013:3336" — H.264 CRF on the Stage-1 conditioning image only.
COMFY_STAGE_1_IMAGE_CRF = 18
# LTXVImgToVideoConditionOnly strength (all 3 stages).
COMFY_IMG_COND_STRENGTH = 1.0
# CFGGuider "5002:4828" / "5001:4964" / "5012:5005" — cfg=1 on every stage.
# (FLOAT input — keep it a float so it round-trips through the schema as the
# workflow's literal does.)
COMFY_CFG = 1.0
# KSamplerSelect: "5002:4831" (Stage 1) vs "5001:4976" / "5012:5003" (Stages 2/3).
COMFY_STAGE_1_SAMPLER = "euler_ancestral_cfg_pp"
COMFY_STAGE_23_SAMPLER = "euler_cfg_pp"
# ManualSigmas: "5002:4984" (Stage 1) vs "5001:4985" / "5012:5006" (Stages 2/3).
COMFY_STAGE_1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
COMFY_STAGE_23_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
# VAEDecodeTiled "5027:4851".
COMFY_DECODE_TILING = {"tile_size": 512, "overlap": 64, "temporal_size": 512, "temporal_overlap": 4}
# EmptyLTXVLatentVideo "5013:3059" is at final/4 (224×320 → 896×1280 via the 2×+2× chain).
COMFY_LATENT_DOWNSCALE = 4
# PrimitiveInt "5026:4988" / PrimitiveFloat "5026:4989" defaults.
COMFY_DEFAULT_NUM_FRAMES = 241
COMFY_DEFAULT_FRAME_RATE = 24.0
# PrimitiveBoolean "5026:4987" — LTXVImgToVideoConditionOnly.bypass; always False.
COMFY_BYPASS_I2V = False

# CLIPTextEncode-Negative "5026:5019" literal.
DEFAULT_NEGATIVE_PROMPT = (
    "camera zooming out, low resolution, blurry, grainy, pixelated, "
    "wide shot, distant view, shallow focus, motion blur, low detail, "
    "flat lighting, dark scene, noisy image, poor texture, soft edges, "
    "muted colors, overexposed, underexposed, static, silent, no movement, "
    "blurry, low quality, still frame, frames, watermark, overlay, titles, "
    "has subtitles, Deformed, scene cut, scene transition, no movement, "
    "glitching, low resolution, extra hands appearing, extra limbs appearing, "
    "warping, extra body parts"
)

# RandomNoise "5002:4832" / "5001:4967" / "5012:5009" — the JSON has three
# literal seeds (re-randomised at runtime by workflow_3mljpp.py, so not
# load-bearing); we derive three deterministic per-stage seeds from the request
# seed instead so the API stays single-seed and reproducible. Within a stage that
# seed drives both the initial latent noise and the ancestral renoise (matching
# ComfyUI's RandomNoise → SamplerCustomAdvanced → default_noise_sampler(x, seed)).
_STAGE_SEED_MASK = (1 << 63) - 1
_STAGE_SEED_SALTS = (0, 0x9E3779B97F4A7C15, 0x2545F4914F6CDD1D)


def _derive_stage_seeds(seed: int) -> tuple[int, int, int]:
    """Spread one request seed into three decorrelated per-stage seeds."""
    base = seed & _STAGE_SEED_MASK
    s1, s2, s3 = ((base ^ salt) & _STAGE_SEED_MASK for salt in _STAGE_SEED_SALTS)
    return s1, s2, s3


def _center_crop_frames(frames: Any, out_h: int, out_w: int) -> Any:
    """Center-crop a ComfyUI IMAGE tensor ``(N, H, W, 3)`` to ``(N, out_h, out_w, 3)``.

    Used to convert the 128-grid generation size (e.g. 1920x1152) to the
    canonical 1080p output (1920x1080) — the cascade can't natively produce
    1080 on a /128 grid, so we generate the smallest /128 size that contains
    the target and slice the symmetric margins off in pixel space.
    """
    _, h, w, _ = frames.shape
    if h == out_h and w == out_w:
        return frames
    off_h = (h - out_h) // 2
    off_w = (w - out_w) // 2
    return frames[:, off_h : off_h + out_h, off_w : off_w + out_w, :]


def _invoke(node_cls: Any, **kwargs: Any) -> tuple:
    """Call a ComfyUI node class (V1 ``FUNCTION``-style or V3 ``io.ComfyNode``)
    and return its outputs as a plain tuple.

    V3 nodes (``comfy_extras.nodes_lt`` etc.) expose a classmethod
    ``EXECUTE_NORMALIZED`` returning an ``io.NodeOutput`` (``.args`` is the output
    tuple). V1 nodes (``LTXVLatentUpsampler``, the ``nodes.py`` loaders
    ``CheckpointLoaderSimple``/``LoraLoaderModelOnly``/``LoadImage``/``CLIPTextEncode``/
    ``VAEDecodeTiled``, …) have ``FUNCTION = "<method>"`` returning a tuple.

    Note: ``EXECUTE_NORMALIZED`` forwards ``**kwargs`` verbatim to ``execute`` — it
    does NOT reassemble ComfyUI's graph-executor sugar. A V3 ``DynamicCombo`` input
    (here only ``ResizeImageMaskNode.resize_type``) must be passed pre-built as a
    dict ``{"<combo-name>": <value>, "<subwidget>": <value>}``, not as the API-JSON's
    dotted ``"<combo-name>.<subwidget>"`` key.
    """
    if hasattr(node_cls, "EXECUTE_NORMALIZED"):
        return tuple(node_cls.EXECUTE_NORMALIZED(**kwargs).args)
    return tuple(getattr(node_cls(), node_cls.FUNCTION)(**kwargs))


class TripleStagesComfyUIGraphPipeline:
    """Runs the 3mljpp 3-stage AV image-to-video workflow via real ComfyUI nodes.

    ``__init__`` bootstraps the ComfyUI runtime (idempotent) and runs the
    loader nodes once (checkpoint → MODEL+VAE, distilled LoRA, Gemma text
    encoder, audio VAE, spatial upscaler). ``__call__`` runs the per-request
    cascade and returns ``(frames_tensor, audio_dict)`` — ComfyUI's IMAGE
    tensor ``(N, H, W, 3)`` float [0,1] and its AUDIO dict
    ``{"waveform": ..., "sample_rate": ...}`` — for the caller to encode.
    """

    def __init__(self, model_dir: str, comfyui_path: str):
        comfyui_runtime.bootstrap_once(comfyui_path, model_dir)

        # Node-class imports — only valid after bootstrap put ComfyUI on
        # sys.path. Keep these inside __init__ so the module is importable
        # without ComfyUI (the test suite + service.py's constant import).
        from nodes import (  # ComfyUI core (V1)
            CLIPTextEncode,
            CheckpointLoaderSimple,
            LoadImage,
            LoraLoaderModelOnly,
            VAEDecodeTiled,
        )
        from comfy_extras.nodes_custom_sampler import (  # V3
            CFGGuider,
            KSamplerSelect,
            ManualSigmas,
            RandomNoise,
            SamplerCustomAdvanced,
        )
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader  # V3
        from comfy_extras.nodes_lt import (  # V3
            EmptyLTXVLatentVideo,
            LTXVConcatAVLatent,
            LTXVConditioning,
            LTXVImgToVideoInplace,  # == ComfyUI-LTXVideo's LTXVImgToVideoConditionOnly (upstreamed)
            LTXVPreprocess,
            LTXVSeparateAVLatent,
        )
        from comfy_extras.nodes_lt_audio import (  # V3
            LTXAVTextEncoderLoader,
            LTXVAudioVAEDecode,
            LTXVAudioVAELoader,
            LTXVEmptyLatentAudio,
        )
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler  # V1
        from comfy_extras.nodes_post_processing import ResizeImageMaskNode  # V1/V3

        self._n = {
            "CheckpointLoaderSimple": CheckpointLoaderSimple,
            "LoraLoaderModelOnly": LoraLoaderModelOnly,
            "LoadImage": LoadImage,
            "VAEDecodeTiled": VAEDecodeTiled,
            "CLIPTextEncode": CLIPTextEncode,
            "KSamplerSelect": KSamplerSelect,
            "ManualSigmas": ManualSigmas,
            "RandomNoise": RandomNoise,
            "CFGGuider": CFGGuider,
            "SamplerCustomAdvanced": SamplerCustomAdvanced,
            "LatentUpscaleModelLoader": LatentUpscaleModelLoader,
            "EmptyLTXVLatentVideo": EmptyLTXVLatentVideo,
            "LTXVConditioning": LTXVConditioning,
            "LTXVPreprocess": LTXVPreprocess,
            "LTXVConcatAVLatent": LTXVConcatAVLatent,
            "LTXVSeparateAVLatent": LTXVSeparateAVLatent,
            "LTXVAudioVAELoader": LTXVAudioVAELoader,
            "LTXVAudioVAEDecode": LTXVAudioVAEDecode,
            "LTXVEmptyLatentAudio": LTXVEmptyLatentAudio,
            "LTXAVTextEncoderLoader": LTXAVTextEncoderLoader,
            "LTXVLatentUpsampler": LTXVLatentUpsampler,
            "ResizeImageMaskNode": ResizeImageMaskNode,
            "LTXVImgToVideoInplace": LTXVImgToVideoInplace,
        }

        # ── Loaders (once) ──────────────────────────────────────────────────
        (upscaler,) = _invoke(LatentUpscaleModelLoader, model_name=COMFY_UPSCALER_NAME)
        model, _clip_unused, vae = _invoke(CheckpointLoaderSimple, ckpt_name=COMFY_CKPT_NAME)
        (lora_model,) = _invoke(
            LoraLoaderModelOnly,
            lora_name=COMFY_DISTILLED_LORA_NAME,
            strength_model=COMFY_DISTILLED_LORA_STRENGTH,
            model=model,
        )
        (audio_vae,) = _invoke(LTXVAudioVAELoader, ckpt_name=COMFY_CKPT_NAME)
        (clip,) = _invoke(
            LTXAVTextEncoderLoader,
            text_encoder=COMFY_TEXT_ENCODER_NAME,
            ckpt_name=COMFY_CKPT_NAME,
            device="default",
        )
        self._model = lora_model
        self._vae = vae
        self._clip = clip
        self._audio_vae = audio_vae
        self._upscaler = upscaler
        logger.info("TripleStagesComfyUIGraphPipeline ready (BF16, ComfyUI node path)")

    # ── per-stage denoise (the SamplerCustomAdvanced + KSamplerSelect + ManualSigmas
    #     + RandomNoise + LTXVConcatAVLatent + LTXVSeparateAVLatent quintet) ─────
    def _denoise_stage(
        self, *, video_cond_latent: Any, audio_latent: Any, guider: Any,
        sampler_name: str, sigmas: str, seed: int,
    ) -> tuple[Any, Any]:
        (av_latent,) = _invoke(
            self._n["LTXVConcatAVLatent"], video_latent=video_cond_latent, audio_latent=audio_latent,
        )
        (sampler,) = _invoke(self._n["KSamplerSelect"], sampler_name=sampler_name)
        (sigmas_t,) = _invoke(self._n["ManualSigmas"], sigmas=sigmas)
        (noise,) = _invoke(self._n["RandomNoise"], noise_seed=seed)
        out = _invoke(
            self._n["SamplerCustomAdvanced"],
            noise=noise, guider=guider, sampler=sampler, sigmas=sigmas_t, latent_image=av_latent,
        )[0]  # SamplerCustomAdvanced → (output, denoised_output); take output
        video_latent, audio_latent_out = _invoke(self._n["LTXVSeparateAVLatent"], av_latent=out)
        return video_latent, audio_latent_out

    def _img_cond(self, *, image: Any, latent: Any) -> Any:
        # Core's LTXVImgToVideoInplace is the upstreamed LTXVImgToVideoConditionOnly:
        # vae.encode(image) → samples[:, :, :t.shape[2]] = t → noise_mask[:, :, :t.shape[2]]
        # = 1.0 - strength → {"samples", "noise_mask"}. Same inputs (vae, image, latent,
        # strength, bypass), so the workflow's LTXVImgToVideoConditionOnly node maps here.
        (cond,) = _invoke(
            self._n["LTXVImgToVideoInplace"],
            vae=self._vae, image=image, latent=latent,
            bypass=COMFY_BYPASS_I2V, strength=COMFY_IMG_COND_STRENGTH,
        )
        return cond

    def _upscale(self, video_latent: Any) -> Any:
        (up,) = _invoke(
            self._n["LTXVLatentUpsampler"],
            samples=video_latent, upscale_model=self._upscaler, vae=self._vae,
        )
        return up

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageInput],
        tiling_config: object | None = None,  # noqa: ARG002 — baked into COMFY_DECODE_TILING
        enhance_prompt: bool = False,
        max_batch_size: int = 1,  # noqa: ARG002 — ComfyUI manages batching
    ) -> tuple[Iterator | Any, Any]:
        if enhance_prompt:
            logger.warning("triple_stages_comfyui: enhance_prompt=True ignored "
                           "(the ComfyUI workflow has no prompt-enhancer node)")
        if width % (COMFY_LATENT_DOWNSCALE * 32) or height % (COMFY_LATENT_DOWNSCALE * 32):
            raise ValueError(
                f"Resolution {width}x{height} must be divisible by {COMFY_LATENT_DOWNSCALE * 32} "
                "for the 4×-chain (Stage 1 = final/4 must be a multiple of the VAE's 32)."
            )

        # CUDA-synchronized clock for per-stage timing. CUDA kernel launches return
        # immediately to Python; without sync, perf_counter would credit kernel work
        # to whichever section followed it. Sync inserts a single GPU-side fence per
        # boundary — negligible cost on this cascade since stages are strictly
        # sequential (no overlap to forfeit).
        import torch

        def _now() -> float:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        timings: dict[str, float] = {}

        seed_1, seed_2, seed_3 = _derive_stage_seeds(seed)

        # ── Image: stage into ComfyUI's input dir, LoadImage → ResizeImageMaskNode.
        # T2V (no image) → a neutral-gray placeholder (the workflow is fundamentally
        # I2V; LoadImage → ResizeImageMaskNode → LTXVImgToVideoConditionOnly always run).
        _t = _now()
        image_name = self._stage_input_image(images, width, height)
        (loaded_image, _mask) = _invoke(self._n["LoadImage"], image=image_name)
        # ResizeImageMaskNode is V3 with a DynamicCombo `resize_type` — execute() wants
        # it as a dict {"resize_type": <combo value>, "<subwidget>": <value>}. The API-JSON's
        # dotted "resize_type.longer_size" is graph-executor sugar; calling EXECUTE_NORMALIZED
        # directly bypasses that (see _invoke), so build the dict here.
        (resized_image,) = _invoke(
            self._n["ResizeImageMaskNode"],
            input=loaded_image,
            scale_method="lanczos",
            resize_type={"resize_type": "scale longer dimension", "longer_size": COMFY_IMAGE_LONGER_DIM},
        )
        timings["image_prep"] = _now() - _t

        # ── Text conditioning.
        _t = _now()
        (pos_cond,) = _invoke(self._n["CLIPTextEncode"], text=prompt, clip=self._clip)
        (neg_cond,) = _invoke(self._n["CLIPTextEncode"], text=negative_prompt, clip=self._clip)
        ltxv_pos, ltxv_neg = _invoke(
            self._n["LTXVConditioning"], frame_rate=float(frame_rate), positive=pos_cond, negative=neg_cond,
        )
        (guider,) = _invoke(
            self._n["CFGGuider"], cfg=COMFY_CFG, model=self._model, positive=ltxv_pos, negative=ltxv_neg,
        )
        timings["text_encode"] = _now() - _t

        # ── Audio (empty latent, threaded through all stages). The workflow uses
        # LTXVideo's LTXFloatToInt(a) — its body is literally `round(a)`; inlined
        # here so we don't pull in ComfyUI-LTXVideo for one trivial node.
        _t = _now()
        frame_rate_int = round(frame_rate)
        (empty_audio,) = _invoke(
            self._n["LTXVEmptyLatentAudio"],
            frames_number=num_frames, frame_rate=frame_rate_int, batch_size=1, audio_vae=self._audio_vae,
        )
        timings["audio_init"] = _now() - _t

        # ── Stage 1 — final/4, euler_ancestral_cfg_pp, CRF-18 image preprocess.
        _t = _now()
        (preprocessed,) = _invoke(self._n["LTXVPreprocess"], img_compression=COMFY_STAGE_1_IMAGE_CRF, image=resized_image)
        (empty_latent,) = _invoke(
            self._n["EmptyLTXVLatentVideo"],
            width=width // COMFY_LATENT_DOWNSCALE, height=height // COMFY_LATENT_DOWNSCALE,
            length=num_frames, batch_size=1,
        )
        cond_1 = self._img_cond(image=preprocessed, latent=empty_latent)
        video_1, audio_1 = self._denoise_stage(
            video_cond_latent=cond_1, audio_latent=empty_audio, guider=guider,
            sampler_name=COMFY_STAGE_1_SAMPLER, sigmas=COMFY_STAGE_1_SIGMAS, seed=seed_1,
        )
        timings["stage_1"] = _now() - _t

        # ── Stage 2 — final/2, euler_cfg_pp, image re-conditioned (no preprocess).
        _t = _now()
        cond_2 = self._img_cond(image=resized_image, latent=self._upscale(video_1))
        video_2, audio_2 = self._denoise_stage(
            video_cond_latent=cond_2, audio_latent=audio_1, guider=guider,
            sampler_name=COMFY_STAGE_23_SAMPLER, sigmas=COMFY_STAGE_23_SIGMAS, seed=seed_2,
        )
        timings["stage_2"] = _now() - _t

        # ── Stage 3 — final res, euler_cfg_pp, image re-conditioned (no preprocess).
        _t = _now()
        cond_3 = self._img_cond(image=resized_image, latent=self._upscale(video_2))
        video_3, audio_3 = self._denoise_stage(
            video_cond_latent=cond_3, audio_latent=audio_2, guider=guider,
            sampler_name=COMFY_STAGE_23_SAMPLER, sigmas=COMFY_STAGE_23_SIGMAS, seed=seed_3,
        )
        timings["stage_3"] = _now() - _t

        # ── Decode.
        _t = _now()
        (frames,) = _invoke(
            self._n["VAEDecodeTiled"], samples=video_3, vae=self._vae, **COMFY_DECODE_TILING,
        )
        timings["vae_decode"] = _now() - _t

        _t = _now()
        (audio,) = _invoke(self._n["LTXVAudioVAEDecode"], samples=audio_3, audio_vae=self._audio_vae)
        timings["audio_decode"] = _now() - _t

        # Structured per-stage timing log for Round 3 diagnostics. Sum approximates
        # the pipeline call's wall time (the wrapper in triple_stages_comfyui.py
        # reports the total separately).
        logger.info(
            "stage_timings=%s sum=%.2fs",
            {k: round(v, 2) for k, v in timings.items()},
            sum(timings.values()),
        )
        return frames, audio

    def encode_to_mp4(
        self,
        *,
        frames: Any,
        audio: Any,
        output_path: str,
        fps: int,
        out_w: int | None = None,
        out_h: int | None = None,
    ) -> None:
        """Assemble the decoded frames + audio into an mp4 via ComfyUI's
        ``CreateVideo`` (the workflow's final node — same fps/container as a real
        ComfyUI run) and write it to ``output_path``. ``CreateVideo`` consumes
        ComfyUI's IMAGE ``(N,H,W,3)`` float[0,1] + AUDIO dict directly (no
        conversion); ``VideoFromComponents.save_to`` with AUTO container/codec →
        mp4/h264 for a ``.mp4`` path.

        When ``out_w``/``out_h`` are supplied and differ from the decoded frame
        size, center-crop the IMAGE tensor before encoding. This is how the
        pipeline turns its /128 generation size (e.g. 1920x1152) into the
        public-facing 1920x1080 output."""
        from comfy_extras.nodes_video import CreateVideo

        if out_w is not None and out_h is not None:
            frames = _center_crop_frames(frames, out_h=out_h, out_w=out_w)
        video_obj = _invoke(CreateVideo, images=frames, fps=float(fps), audio=audio)[0]
        video_obj.save_to(output_path)

    def _stage_input_image(self, images: list[ImageInput], width: int, height: int) -> str:
        """Copy the conditioning image into ComfyUI's input dir and return its
        basename (for ``LoadImage``). T2V (empty ``images``) → a neutral-gray PNG."""
        in_dir = comfyui_runtime.input_dir()
        name = f"comfyin_{uuid.uuid4().hex[:12]}.png"
        dest = os.path.join(in_dir, name)
        if images:
            import shutil

            shutil.copyfile(images[0].path, dest)
        else:
            from PIL import Image

            # Neutral-gray placeholder at the request AR (ResizeImageMaskNode +
            # LTXVImgToVideoConditionOnly will resize/crop it anyway).
            Image.new("RGB", (max(1, width // COMFY_LATENT_DOWNSCALE), max(1, height // COMFY_LATENT_DOWNSCALE)),
                      (128, 128, 128)).save(dest)
        return name
