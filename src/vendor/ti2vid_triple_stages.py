"""
Three-stage text/image-to-video generation pipeline.

Stage 1 generates at half latent resolution, then upsamples to full resolution
for Stage 2 and Stage 3 refinement — based on the ComfyUI three-stage workflow
that produces significantly better i2v quality than the standard two-stage pipeline.

Key differences from two-stage:
- Stage 1 at half resolution (latent stage_1_H × stage_1_W) with 9 custom sigma steps
- Image resized to 1536px on longer edge before conditioning
- cfg=1 (minimal CFG, image conditioning drives output)
- Two full-resolution refinement stages with 4 sigma steps each
"""

import argparse
import logging
from collections.abc import Iterator

import einops
import torch

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput, detect_checkpoint_path
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec


class TI2VidTripleStagesPipeline:
    """
    Three-stage image-to-video pipeline.

    Stage 1: Generate at half resolution (latent stage_1_H × stage_1_W) with
              image conditioning and 9 custom sigma steps.
    Stage 2: Upsample to full resolution, apply image conditioning again with
              4 sigma steps.
    Stage 3: Another 4-sigma refinement pass at full resolution.
    Decode:  Final video + audio decoding.
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

        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.stage_3 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
        stage1_steps: int = 16,
        stage2_steps: int = 8,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        ctx_p, ctx_n = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # ── Stage 1: half resolution generation ──────────────────────────────────
        stage_1_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )

        # Image conditioning — same as two-stage pipeline (no H.264 preprocessing)
        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_1_shape.height,
                width=stage_1_shape.width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                preprocessed_images=None,
            )
        )

        # Compute sigmas using Karras-style schedule
        # Stage 1: start at 1.0 (full noise); Stage 2/3: start at 0.85 (partial denoise)
        stage_1_sigmas = LTX2Scheduler().execute(steps=stage1_steps).to(dtype=torch.float32, device=self.device)
        stage_2_sigmas = LTX2Scheduler().execute(steps=stage2_steps).to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_1(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_shape.width,
            height=stage_1_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
                noise_scale=1.0,
            ),
            audio=ModalitySpec(context=a_context_p),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        # ── Upsample to full resolution ─────────────────────────────────────
        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        # ── Stage 2: full resolution ────────────────────────────────────────
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
                preprocessed_images=None,  # encode at correct full-res dims via load_image_and_preprocess
            )
        )


        video_state, audio_state = self.stage_2(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
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
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        # ── Stage 3: another full-resolution refinement pass ─────────────────
        video_state, audio_state = self.stage_3(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_2_conditionings),
            audio=ModalitySpec(context=a_context_p),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Three-stage image-to-video generation (ComfyUI workflow).")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to LTX-2 model checkpoint.")
    parser.add_argument("--distilled-lora", type=str, action="append", dest="distilled_loras",
                        default=[], help="Distilled LoRA (can be specified multiple times).")
    parser.add_argument("--spatial-upsampler-path", type=str, required=True,
                        help="Path to spatial upsampler.")
    parser.add_argument("--gemma-root", type=str, required=True,
                        help="Path to Gemma text encoder root.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for video generation.")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output video path.")
    parser.add_argument("--seed", type=int, default=10,
                        help="Random seed.")
    parser.add_argument("--width", type=int, default=1536,
                        help="Final output width (multiple of 64).")
    parser.add_argument("--height", type=int, default=1024,
                        help="Final output height (multiple of 64).")
    parser.add_argument("--num-frames", type=int, default=121,
                        help="Number of frames (must satisfy (8*K)+1).")
    parser.add_argument("--frame-rate", type=float, default=24.0,
                        help="Frame rate.")
    parser.add_argument("--video-cfg-scale", type=float, default=1.0,
                        help="Video CFG scale (default: 1.0, low to rely on image conditioning).")
    parser.add_argument("--video-stg-scale", type=float, default=0.0,
                        help="Video STG scale (default: 0.0).")
    parser.add_argument("--video-rescale-scale", type=float, default=0.0,
                        help="Video rescale scale (default: 0.0).")
    parser.add_argument("--video-stg-blocks", type=int, nargs="*", default=[],
                        help="STG blocks for video (default: []).")
    parser.add_argument("--audio-cfg-scale", type=float, default=1.0,
                        help="Audio CFG scale (default: 1.0).")
    parser.add_argument("--audio-stg-scale", type=float, default=0.0,
                        help="Audio STG scale (default: 0.0).")
    parser.add_argument("--audio-rescale-scale", type=float, default=0.0,
                        help="Audio rescale scale (default: 0.0).")
    parser.add_argument("--audio-stg-blocks", type=int, nargs="*", default=[],
                        help="STG blocks for audio (default: []).")
    parser.add_argument("--lora", action="append", dest="loras", default=[],
                        help="Additional LoRA (path [strength]).")
    parser.add_argument("--image", action="append", dest="images", default=[],
                        help="Image conditioning: PATH FRAME_IDX STRENGTH [CRF].")
    parser.add_argument("--streaming-prefetch-count", type=int, default=None,
                        help="Layer streaming prefetch count.")
    parser.add_argument("--max-batch-size", type=int, default=1,
                        help="Max batch size per transformer forward pass.")
    parser.add_argument("--stage1-steps", type=int, default=16,
                        help="Stage 1 denoising steps (default: 16). ComfyUI uses 8.")
    parser.add_argument("--stage2-steps", type=int, default=8,
                        help="Stage 2 & 3 denoising steps (default: 8). ComfyUI uses 3.")
    parser.add_argument("--image-strength", type=float, default=1.0,
                        help="Image conditioning strength (default: 1.0). ComfyUI uses 0.7 for stage 1.")
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

    distilled_loras = []
    for d in args.distilled_loras:
        parts = d.split()
        path = parts[0]
        strength = float(parts[1]) if len(parts) > 1 else 0.8
        distilled_loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))

    loras = []
    for l in args.loras:
        parts = l.split()
        path = parts[0]
        strength = float(parts[1]) if len(parts) > 1 else 0.8
        loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))

    images = []
    for img_arg in args.images:
        parts = img_arg.split()
        path, frame_idx, strength = parts[0], int(parts[1]), args.image_strength
        crf = int(parts[3]) if len(parts) > 3 else 33
        images.append(ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength, crf=crf))

    quantization = None
    if args.quantization == "fp8-cast":
        quantization = QuantizationPolicy.fp8_cast()
    elif args.quantization == "fp8-scaled-mm":
        quantization = QuantizationPolicy.fp8_scaled_mm()

    pipeline = TI2VidTripleStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=distilled_loras,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=loras,
        quantization=quantization,
        torch_compile=args.compile,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_scale,
        stg_scale=args.video_stg_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=1.0,
        skip_step=0,
        stg_blocks=args.video_stg_blocks,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_scale,
        stg_scale=args.audio_stg_scale,
        rescale_scale=args.audio_rescale_scale,
        modality_scale=1.0,
        skip_step=0,
        stg_blocks=args.audio_stg_blocks,
    )

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=images,
        tiling_config=tiling_config,
        streaming_prefetch_count=args.streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
