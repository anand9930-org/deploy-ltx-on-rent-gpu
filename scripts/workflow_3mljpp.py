# Auto-generated from /Users/anand/Downloads/3mljpp-api.json by
# pydn/ComfyUI-to-Python-Extension. Regenerate via:
#
#   COMFYUI_PATH=/Users/anand/Documents/ComfyUI-tools/ComfyUI \
#     ~/Documents/ComfyUI/.venv/bin/python \
#     /Users/anand/Documents/ComfyUI-tools/ComfyUI-to-Python-Extension/comfyui_to_python.py \
#     -f <api.json> -o scripts/workflow_3mljpp.py -q 1
#
# Note: four nodes leaked from a UI-only subgraph and had no class_type in the
# API export. Inferred from input signatures — verify before running:
#   5013:3159, 5001:4970, 5012:5008  → LTXVImgToVideoConditionOnly (strength=1.0)
#   5013:4983                        → LTXFloatToInt
import json
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Return a sequence or mapping result item by index."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def get_comfyui_path() -> str:
    """Return the configured ComfyUI path, preferring COMFYUI_PATH when set."""
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if comfyui_path:
        return comfyui_path
    return find_path("ComfyUI")


def find_path(name: str, path: str = None) -> str:
    """Recursively search parent folders until the named entry is found."""
    if path is None:
        path = os.getcwd()

    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None

    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """Add the ComfyUI checkout to sys.path."""
    comfyui_path = get_comfyui_path()
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        if comfyui_path in sys.path:
            sys.path.remove(comfyui_path)
        sys.path.insert(0, comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """Load ComfyUI extra model paths configuration when available."""
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def bootstrap_comfyui_runtime() -> None:
    """Mirror the allocator-related ComfyUI startup steps before torch import."""
    add_comfyui_directory_to_sys_path()

    import comfy.options

    comfy.options.enable_args_parsing()

    from comfy.cli_args import args

    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ",".join(map(str, devices))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        os.environ["HIP_VISIBLE_DEVICES"] = str(devices)

    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["HIP_VISIBLE_DEVICES"] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)

    if args.oneapi_device_selector is not None:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = args.oneapi_device_selector

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import cuda_malloc

    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ["OCL_SET_SVM_SIZE"] = "262144"


def cleanup_comfyui_runtime(unload_models: bool | None = None) -> None:
    """Best-effort cleanup for embedded or repeated generated-script execution."""
    import gc

    def run_cleanup_hook(name: str, should_run: bool = True) -> None:
        if not should_run or not hasattr(model_management, name):
            return
        cleanup_fn = getattr(model_management, name)
        try:
            cleanup_fn()
        except Exception as exc:
            warnings.warn(
                f"ComfyUI cleanup hook {name} failed during teardown: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    should_unload = unload_models
    if should_unload is None:
        should_unload = os.environ.get(
            "COMFYUI_TOPYTHON_UNLOAD_MODELS", ""
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    try:
        import comfy.model_management as model_management
    except ModuleNotFoundError:
        gc.collect()
        return

    run_cleanup_hook("cleanup_models_gc")
    run_cleanup_hook("unload_all_models", should_run=should_unload)
    run_cleanup_hook("soft_empty_cache")
    gc.collect()


def import_custom_nodes() -> None:
    """Initialize ComfyUI custom nodes in the exporter runtime."""
    comfyui_path = get_comfyui_path()
    if comfyui_path and comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    import asyncio
    import execution
    from nodes import init_extra_nodes

    if comfyui_path in sys.path:
        sys.path.remove(comfyui_path)
    sys.path.insert(0, comfyui_path)

    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())


# Workflow data
def build_workflow() -> dict[str, Any]:
    return {
        "4852": {
            "inputs": {
                "filename_prefix": "output",
                "format": "auto",
                "codec": "auto",
                "video": ["5027:4849", 0],
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"},
        },
        "5025:5021": {
            "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"},
            "class_type": "LatentUpscaleModelLoader",
            "_meta": {"title": "Load Latent Upscale Model"},
        },
        "5025:5022": {
            "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"},
        },
        "5025:5023": {
            "inputs": {
                "lora_name": "ltx-2.3-22b-distilled-lora-384.safetensors",
                "strength_model": 0.5,
                "model": ["5025:5022", 0],
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "Load LoRA"},
        },
        "5025:5024": {
            "inputs": {"ckpt_name": "ltx-2.3-22b-dev.safetensors"},
            "class_type": "LTXVAudioVAELoader",
            "_meta": {"title": "LTXV Audio VAE Loader"},
        },
        "5025:5017": {
            "inputs": {
                "text_encoder": "gemma_3_12B_it.safetensors",
                "ckpt_name": "ltx-2.3-22b-dev.safetensors",
                "device": "default",
            },
            "class_type": "LTXAVTextEncoderLoader",
            "_meta": {"title": "LTXV Audio Text Encoder Loader"},
        },
        "5016:2004": {
            "inputs": {"image": "8QJE3YYRN196A1CD5D6MSJ0AW0.jpeg"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        "5016:4990": {
            "inputs": {
                "resize_type": "scale longer dimension",
                "resize_type.longer_size": 1536,
                "scale_method": "lanczos",
                "input": ["5016:2004", 0],
            },
            "class_type": "ResizeImageMaskNode",
            "_meta": {"title": "Resize Image/Mask"},
        },
        "5026:4989": {
            "inputs": {"value": 24},
            "class_type": "PrimitiveFloat",
            "_meta": {"title": "fps"},
        },
        "5026:4988": {
            "inputs": {"value": 241},
            "class_type": "PrimitiveInt",
            "_meta": {"title": "number of frames"},
        },
        "5026:4987": {
            "inputs": {"value": False},
            "class_type": "PrimitiveBoolean",
            "_meta": {"title": "bypass_i2v"},
        },
        "5026:5018": {
            "inputs": {"text": "", "clip": ["5025:5017", 0]},
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"},
        },
        "5026:5019": {
            "inputs": {
                "text": "camera zooming out, low resolution, blurry, "
                "grainy, pixelated, wide shot, distant view, "
                "shallow focus, motion blur, low detail, "
                "flat lighting, dark scene, noisy image, "
                "poor texture, soft edges, muted colors, "
                "overexposed, underexposed, static, silent, "
                "no movement, blurry, low quality, still "
                "frame, frames, watermark, overlay, titles, "
                "has subtitles, Deformed, scene cut, scene "
                "transition, no movement, glitching, low "
                "resolution, extra hands appearing, extra "
                "limbs appearing, warping, extra body parts",
                "clip": ["5025:5017", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative Prompt)"},
        },
        "5026:5020": {
            "inputs": {
                "frame_rate": ["5026:4989", 0],
                "positive": ["5026:5018", 0],
                "negative": ["5026:5019", 0],
            },
            "class_type": "LTXVConditioning",
            "_meta": {"title": "LTXVConditioning"},
        },
        "5013:3980": {
            "inputs": {
                "frames_number": ["5026:4988", 0],
                "frame_rate": ["5013:4983", 0],
                "batch_size": 1,
                "audio_vae": ["5025:5024", 0],
            },
            "class_type": "LTXVEmptyLatentAudio",
            "_meta": {"title": "LTXV Empty Latent Audio"},
        },
        "5013:4528": {
            "inputs": {
                "video_latent": ["5013:3159", 0],
                "audio_latent": ["5013:3980", 0],
            },
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "LTXVConcatAVLatent"},
        },
        "5013:3159": {
            "inputs": {
                "vae": ["5025:5022", 2],
                "image": ["5013:3336", 0],
                "latent": ["5013:3059", 0],
                "bypass": ["5026:4987", 0],
                "strength": 1.0,
            },
            "_meta": {},
            "class_type": "LTXVImgToVideoConditionOnly",
        },
        "5013:3336": {
            "inputs": {"img_compression": 18, "image": ["5016:4990", 0]},
            "class_type": "LTXVPreprocess",
            "_meta": {"title": "LTXVPreprocess"},
        },
        "5013:4983": {
            "inputs": {"a": ["5026:4989", 0]},
            "_meta": {},
            "class_type": "LTXFloatToInt",
        },
        "5013:3059": {
            "inputs": {
                "width": 224,
                "height": 320,
                "length": ["5026:4988", 0],
                "batch_size": 1,
            },
            "class_type": "EmptyLTXVLatentVideo",
            "_meta": {"title": "EmptyLTXVLatentVideo"},
        },
        "5002:4828": {
            "inputs": {
                "cfg": 1,
                "model": ["5025:5023", 0],
                "positive": ["5026:5020", 0],
                "negative": ["5026:5020", 1],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFGGuider"},
        },
        "5002:4831": {
            "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "5002:4829": {
            "inputs": {
                "noise": ["5002:4832", 0],
                "guider": ["5002:4828", 0],
                "sampler": ["5002:4831", 0],
                "sigmas": ["5002:4984", 0],
                "latent_image": ["5013:4528", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "5002:4984": {
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                "0.909375, 0.725, 0.421875, 0.0"
            },
            "class_type": "ManualSigmas",
            "_meta": {"title": "ManualSigmas"},
        },
        "5002:4845": {
            "inputs": {"av_latent": ["5002:4829", 0]},
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "LTXVSeparateAVLatent"},
        },
        "5002:4832": {
            "inputs": {"noise_seed": 727273229127121},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "5001:4976": {
            "inputs": {"sampler_name": "euler_cfg_pp"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "5001:4971": {
            "inputs": {
                "noise": ["5001:4967", 0],
                "guider": ["5001:4964", 0],
                "sampler": ["5001:4976", 0],
                "sigmas": ["5001:4985", 0],
                "latent_image": ["5001:4969", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "5001:4985": {
            "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"},
            "class_type": "ManualSigmas",
            "_meta": {"title": "ManualSigmas"},
        },
        "5001:4969": {
            "inputs": {
                "video_latent": ["5001:4970", 0],
                "audio_latent": ["5002:4845", 1],
            },
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "LTXVConcatAVLatent"},
        },
        "5001:4975": {
            "inputs": {
                "samples": ["5002:4845", 0],
                "upscale_model": ["5025:5021", 0],
                "vae": ["5025:5022", 2],
            },
            "class_type": "LTXVLatentUpsampler",
            "_meta": {"title": "LTXVLatentUpsampler"},
        },
        "5001:4967": {
            "inputs": {"noise_seed": 200996433497366},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "5001:4970": {
            "inputs": {
                "vae": ["5025:5022", 2],
                "image": ["5016:4990", 0],
                "latent": ["5001:4975", 0],
                "bypass": ["5026:4987", 0],
                "strength": 1.0,
            },
            "_meta": {},
            "class_type": "LTXVImgToVideoConditionOnly",
        },
        "5001:4973": {
            "inputs": {"av_latent": ["5001:4971", 0]},
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "LTXVSeparateAVLatent"},
        },
        "5001:4964": {
            "inputs": {
                "cfg": 1,
                "model": ["5025:5023", 0],
                "positive": ["5026:5020", 0],
                "negative": ["5026:5020", 1],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFGGuider"},
        },
        "5012:5003": {
            "inputs": {"sampler_name": "euler_cfg_pp"},
            "class_type": "KSamplerSelect",
            "_meta": {"title": "KSamplerSelect"},
        },
        "5012:5004": {
            "inputs": {
                "noise": ["5012:5009", 0],
                "guider": ["5012:5005", 0],
                "sampler": ["5012:5003", 0],
                "sigmas": ["5012:5006", 0],
                "latent_image": ["5012:5007", 0],
            },
            "class_type": "SamplerCustomAdvanced",
            "_meta": {"title": "SamplerCustomAdvanced"},
        },
        "5012:5005": {
            "inputs": {
                "cfg": 1,
                "model": ["5025:5023", 0],
                "positive": ["5026:5020", 0],
                "negative": ["5026:5020", 1],
            },
            "class_type": "CFGGuider",
            "_meta": {"title": "CFGGuider"},
        },
        "5012:5006": {
            "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"},
            "class_type": "ManualSigmas",
            "_meta": {"title": "ManualSigmas"},
        },
        "5012:5007": {
            "inputs": {
                "video_latent": ["5012:5008", 0],
                "audio_latent": ["5001:4973", 1],
            },
            "class_type": "LTXVConcatAVLatent",
            "_meta": {"title": "LTXVConcatAVLatent"},
        },
        "5012:5008": {
            "inputs": {
                "vae": ["5025:5022", 2],
                "image": ["5016:4990", 0],
                "latent": ["5012:5010", 0],
                "bypass": ["5026:4987", 0],
                "strength": 1.0,
            },
            "_meta": {},
            "class_type": "LTXVImgToVideoConditionOnly",
        },
        "5012:5009": {
            "inputs": {"noise_seed": 975078551246030},
            "class_type": "RandomNoise",
            "_meta": {"title": "RandomNoise"},
        },
        "5012:5010": {
            "inputs": {
                "samples": ["5001:4973", 0],
                "upscale_model": ["5025:5021", 0],
                "vae": ["5025:5022", 2],
            },
            "class_type": "LTXVLatentUpsampler",
            "_meta": {"title": "LTXVLatentUpsampler"},
        },
        "5012:5011": {
            "inputs": {"av_latent": ["5012:5004", 0]},
            "class_type": "LTXVSeparateAVLatent",
            "_meta": {"title": "LTXVSeparateAVLatent"},
        },
        "5027:4849": {
            "inputs": {
                "fps": ["5026:4989", 0],
                "images": ["5027:4851", 0],
                "audio": ["5027:4848", 0],
            },
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"},
        },
        "5027:4851": {
            "inputs": {
                "tile_size": 512,
                "overlap": 64,
                "temporal_size": 512,
                "temporal_overlap": 4,
                "samples": ["5012:5011", 0],
                "vae": ["5025:5022", 2],
            },
            "class_type": "VAEDecodeTiled",
            "_meta": {"title": "VAE Decode (Tiled)"},
        },
        "5027:4848": {
            "inputs": {"samples": ["5012:5011", 1], "audio_vae": ["5025:5024", 0]},
            "class_type": "LTXVAudioVAEDecode",
            "_meta": {"title": "LTXV Audio VAE Decode"},
        },
    }


def build_extra_pnginfo() -> dict[str, Any] | None:
    return None


workflow = build_workflow()
prompt = json.loads(json.dumps(workflow))
extra_pnginfo = build_extra_pnginfo()


# Workflow execution
def main(unload_models: bool | None = None):
    bootstrap_comfyui_runtime()
    add_extra_model_paths()
    import_custom_nodes()

    # Node imports
    from nodes import (
        CLIPTextEncode,
        CheckpointLoaderSimple,
        LoadImage,
        LoraLoaderModelOnly,
        NODE_CLASS_MAPPINGS,
        VAEDecodeTiled,
    )

    import torch

    try:
        with torch.inference_mode():
            latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
            latentupscalemodelloader_5025_5021 = (
                latentupscalemodelloader.EXECUTE_NORMALIZED(
                    model_name="ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
                )
            )
            checkpointloadersimple = CheckpointLoaderSimple()
            checkpointloadersimple_5025_5022 = checkpointloadersimple.load_checkpoint(
                ckpt_name="ltx-2.3-22b-dev.safetensors"
            )
            loraloadermodelonly = LoraLoaderModelOnly()
            loraloadermodelonly_5025_5023 = loraloadermodelonly.load_lora_model_only(
                lora_name="ltx-2.3-22b-distilled-lora-384.safetensors",
                strength_model=0.5,
                model=get_value_at_index(checkpointloadersimple_5025_5022, 0),
            )
            ltxvaudiovaeloader = NODE_CLASS_MAPPINGS["LTXVAudioVAELoader"]()
            ltxvaudiovaeloader_5025_5024 = ltxvaudiovaeloader.EXECUTE_NORMALIZED(
                ckpt_name="ltx-2.3-22b-dev.safetensors"
            )
            ltxavtextencoderloader = NODE_CLASS_MAPPINGS["LTXAVTextEncoderLoader"]()
            ltxavtextencoderloader_5025_5017 = (
                ltxavtextencoderloader.EXECUTE_NORMALIZED(
                    text_encoder="gemma_3_12B_it.safetensors",
                    ckpt_name="ltx-2.3-22b-dev.safetensors",
                    device="default",
                )
            )
            loadimage = LoadImage()
            loadimage_5016_2004 = loadimage.load_image(
                image="8QJE3YYRN196A1CD5D6MSJ0AW0.jpeg"
            )
            primitivefloat = NODE_CLASS_MAPPINGS["PrimitiveFloat"]()
            primitivefloat_5026_4989 = primitivefloat.EXECUTE_NORMALIZED(value=24)
            primitiveint = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
            primitiveint_5026_4988 = primitiveint.EXECUTE_NORMALIZED(value=241)
            primitiveboolean = NODE_CLASS_MAPPINGS["PrimitiveBoolean"]()
            primitiveboolean_5026_4987 = primitiveboolean.EXECUTE_NORMALIZED(
                value=False
            )
            cliptextencode = CLIPTextEncode()
            cliptextencode_5026_5018 = cliptextencode.encode(
                text="", clip=get_value_at_index(ltxavtextencoderloader_5025_5017, 0)
            )
            cliptextencode_5026_5019 = cliptextencode.encode(
                text="camera zooming out, low resolution, blurry, grainy, pixelated, wide shot, distant view, shallow focus, motion blur, low detail, flat lighting, dark scene, noisy image, poor texture, soft edges, muted colors, overexposed, underexposed, static, silent, no movement, blurry, low quality, still frame, frames, watermark, overlay, titles, has subtitles, Deformed, scene cut, scene transition, no movement, glitching, low resolution, extra hands appearing, extra limbs appearing, warping, extra body parts",
                clip=get_value_at_index(ltxavtextencoderloader_5025_5017, 0),
            )
            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            ksamplerselect_5002_4831 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="euler_ancestral_cfg_pp"
            )
            manualsigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]()
            manualsigmas_5002_4984 = manualsigmas.EXECUTE_NORMALIZED(
                sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
            )
            randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
            node_5002_4832_noise_seed = prompt["5002:4832"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_5002_4832 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_5002_4832_noise_seed
            )
            ksamplerselect_5001_4976 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="euler_cfg_pp"
            )
            manualsigmas_5001_4985 = manualsigmas.EXECUTE_NORMALIZED(
                sigmas="0.85, 0.7250, 0.4219, 0.0"
            )
            node_5001_4967_noise_seed = prompt["5001:4967"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_5001_4967 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_5001_4967_noise_seed
            )
            ksamplerselect_5012_5003 = ksamplerselect.EXECUTE_NORMALIZED(
                sampler_name="euler_cfg_pp"
            )
            manualsigmas_5012_5006 = manualsigmas.EXECUTE_NORMALIZED(
                sigmas="0.85, 0.7250, 0.4219, 0.0"
            )
            node_5012_5009_noise_seed = prompt["5012:5009"]["inputs"]["noise_seed"] = (
                random.randint(1, 2**64)
            )
            randomnoise_5012_5009 = randomnoise.EXECUTE_NORMALIZED(
                noise_seed=node_5012_5009_noise_seed
            )
            ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
            resizeimagemasknode = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ltxvpreprocess = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
            ltxvimgtovideoconditiononly = NODE_CLASS_MAPPINGS[
                "LTXVImgToVideoConditionOnly"
            ]()
            ltxfloattoint = NODE_CLASS_MAPPINGS["LTXFloatToInt"]()
            ltxvemptylatentaudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
            ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
            samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
            ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
            vaedecodetiled = VAEDecodeTiled()
            ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
            savevideo = NODE_CLASS_MAPPINGS["SaveVideo"]()
            for q in range(1):
                ltxvconditioning_5026_5020 = ltxvconditioning.EXECUTE_NORMALIZED(
                    frame_rate=get_value_at_index(primitivefloat_5026_4989, 0),
                    positive=get_value_at_index(cliptextencode_5026_5018, 0),
                    negative=get_value_at_index(cliptextencode_5026_5019, 0),
                )
                cfgguider_5012_5005 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(loraloadermodelonly_5025_5023, 0),
                    positive=get_value_at_index(ltxvconditioning_5026_5020, 0),
                    negative=get_value_at_index(ltxvconditioning_5026_5020, 1),
                )
                resizeimagemasknode_5016_4990 = resizeimagemasknode.EXECUTE_NORMALIZED(
                    resize_type="scale longer dimension",
                    **{"resize_type.longer_size": 1536},
                    scale_method="lanczos",
                    input=get_value_at_index(loadimage_5016_2004, 0),
                )
                cfgguider_5001_4964 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(loraloadermodelonly_5025_5023, 0),
                    positive=get_value_at_index(ltxvconditioning_5026_5020, 0),
                    negative=get_value_at_index(ltxvconditioning_5026_5020, 1),
                )
                cfgguider_5002_4828 = cfgguider.EXECUTE_NORMALIZED(
                    cfg=1,
                    model=get_value_at_index(loraloadermodelonly_5025_5023, 0),
                    positive=get_value_at_index(ltxvconditioning_5026_5020, 0),
                    negative=get_value_at_index(ltxvconditioning_5026_5020, 1),
                )
                ltxvpreprocess_5013_3336 = ltxvpreprocess.EXECUTE_NORMALIZED(
                    img_compression=18,
                    image=get_value_at_index(resizeimagemasknode_5016_4990, 0),
                )
                emptyltxvlatentvideo_5013_3059 = (
                    emptyltxvlatentvideo.EXECUTE_NORMALIZED(
                        width=224,
                        height=320,
                        length=get_value_at_index(primitiveint_5026_4988, 0),
                        batch_size=1,
                    )
                )
                ltxvimgtovideoconditiononly_5013_3159 = (
                    ltxvimgtovideoconditiononly.generate(
                        vae=get_value_at_index(checkpointloadersimple_5025_5022, 2),
                        image=get_value_at_index(ltxvpreprocess_5013_3336, 0),
                        latent=get_value_at_index(emptyltxvlatentvideo_5013_3059, 0),
                        bypass=get_value_at_index(primitiveboolean_5026_4987, 0),
                        strength=1.0,
                    )
                )
                ltxfloattoint_5013_4983 = ltxfloattoint.op(
                    a=get_value_at_index(primitivefloat_5026_4989, 0)
                )
                ltxvemptylatentaudio_5013_3980 = (
                    ltxvemptylatentaudio.EXECUTE_NORMALIZED(
                        frames_number=get_value_at_index(primitiveint_5026_4988, 0),
                        frame_rate=get_value_at_index(ltxfloattoint_5013_4983, 0),
                        batch_size=1,
                        audio_vae=get_value_at_index(ltxvaudiovaeloader_5025_5024, 0),
                    )
                )
                ltxvconcatavlatent_5013_4528 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                    video_latent=get_value_at_index(
                        ltxvimgtovideoconditiononly_5013_3159, 0
                    ),
                    audio_latent=get_value_at_index(ltxvemptylatentaudio_5013_3980, 0),
                )
                samplercustomadvanced_5002_4829 = (
                    samplercustomadvanced.EXECUTE_NORMALIZED(
                        noise=get_value_at_index(randomnoise_5002_4832, 0),
                        guider=get_value_at_index(cfgguider_5002_4828, 0),
                        sampler=get_value_at_index(ksamplerselect_5002_4831, 0),
                        sigmas=get_value_at_index(manualsigmas_5002_4984, 0),
                        latent_image=get_value_at_index(
                            ltxvconcatavlatent_5013_4528, 0
                        ),
                    )
                )
                ltxvseparateavlatent_5002_4845 = (
                    ltxvseparateavlatent.EXECUTE_NORMALIZED(
                        av_latent=get_value_at_index(samplercustomadvanced_5002_4829, 0)
                    )
                )
                ltxvlatentupsampler_5001_4975 = ltxvlatentupsampler.upsample_latent(
                    samples=get_value_at_index(ltxvseparateavlatent_5002_4845, 0),
                    upscale_model=get_value_at_index(
                        latentupscalemodelloader_5025_5021, 0
                    ),
                    vae=get_value_at_index(checkpointloadersimple_5025_5022, 2),
                )
                ltxvimgtovideoconditiononly_5001_4970 = (
                    ltxvimgtovideoconditiononly.generate(
                        vae=get_value_at_index(checkpointloadersimple_5025_5022, 2),
                        image=get_value_at_index(resizeimagemasknode_5016_4990, 0),
                        latent=get_value_at_index(ltxvlatentupsampler_5001_4975, 0),
                        bypass=get_value_at_index(primitiveboolean_5026_4987, 0),
                        strength=1.0,
                    )
                )
                ltxvconcatavlatent_5001_4969 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                    video_latent=get_value_at_index(
                        ltxvimgtovideoconditiononly_5001_4970, 0
                    ),
                    audio_latent=get_value_at_index(ltxvseparateavlatent_5002_4845, 1),
                )
                samplercustomadvanced_5001_4971 = (
                    samplercustomadvanced.EXECUTE_NORMALIZED(
                        noise=get_value_at_index(randomnoise_5001_4967, 0),
                        guider=get_value_at_index(cfgguider_5001_4964, 0),
                        sampler=get_value_at_index(ksamplerselect_5001_4976, 0),
                        sigmas=get_value_at_index(manualsigmas_5001_4985, 0),
                        latent_image=get_value_at_index(
                            ltxvconcatavlatent_5001_4969, 0
                        ),
                    )
                )
                ltxvseparateavlatent_5001_4973 = (
                    ltxvseparateavlatent.EXECUTE_NORMALIZED(
                        av_latent=get_value_at_index(samplercustomadvanced_5001_4971, 0)
                    )
                )
                ltxvlatentupsampler_5012_5010 = ltxvlatentupsampler.upsample_latent(
                    samples=get_value_at_index(ltxvseparateavlatent_5001_4973, 0),
                    upscale_model=get_value_at_index(
                        latentupscalemodelloader_5025_5021, 0
                    ),
                    vae=get_value_at_index(checkpointloadersimple_5025_5022, 2),
                )
                ltxvimgtovideoconditiononly_5012_5008 = (
                    ltxvimgtovideoconditiononly.generate(
                        vae=get_value_at_index(checkpointloadersimple_5025_5022, 2),
                        image=get_value_at_index(resizeimagemasknode_5016_4990, 0),
                        latent=get_value_at_index(ltxvlatentupsampler_5012_5010, 0),
                        bypass=get_value_at_index(primitiveboolean_5026_4987, 0),
                        strength=1.0,
                    )
                )
                ltxvconcatavlatent_5012_5007 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                    video_latent=get_value_at_index(
                        ltxvimgtovideoconditiononly_5012_5008, 0
                    ),
                    audio_latent=get_value_at_index(ltxvseparateavlatent_5001_4973, 1),
                )
                samplercustomadvanced_5012_5004 = (
                    samplercustomadvanced.EXECUTE_NORMALIZED(
                        noise=get_value_at_index(randomnoise_5012_5009, 0),
                        guider=get_value_at_index(cfgguider_5012_5005, 0),
                        sampler=get_value_at_index(ksamplerselect_5012_5003, 0),
                        sigmas=get_value_at_index(manualsigmas_5012_5006, 0),
                        latent_image=get_value_at_index(
                            ltxvconcatavlatent_5012_5007, 0
                        ),
                    )
                )
                ltxvseparateavlatent_5012_5011 = (
                    ltxvseparateavlatent.EXECUTE_NORMALIZED(
                        av_latent=get_value_at_index(samplercustomadvanced_5012_5004, 0)
                    )
                )
                vaedecodetiled_5027_4851 = vaedecodetiled.decode(
                    tile_size=512,
                    overlap=64,
                    temporal_size=512,
                    temporal_overlap=4,
                    samples=get_value_at_index(ltxvseparateavlatent_5012_5011, 0),
                    vae=get_value_at_index(checkpointloadersimple_5025_5022, 2),
                )
                ltxvaudiovaedecode_5027_4848 = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
                    samples=get_value_at_index(ltxvseparateavlatent_5012_5011, 1),
                    audio_vae=get_value_at_index(ltxvaudiovaeloader_5025_5024, 0),
                )
                createvideo_5027_4849 = createvideo.EXECUTE_NORMALIZED(
                    fps=get_value_at_index(primitivefloat_5026_4989, 0),
                    images=get_value_at_index(vaedecodetiled_5027_4851, 0),
                    audio=get_value_at_index(ltxvaudiovaedecode_5027_4848, 0),
                )
                savevideo_4852 = savevideo.EXECUTE_NORMALIZED(
                    filename_prefix="output",
                    format="auto",
                    codec="auto",
                    video=get_value_at_index(createvideo_5027_4849, 0),
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                )
    finally:
        cleanup_comfyui_runtime(unload_models=unload_models)


# Entrypoint
if __name__ == "__main__":
    main()
