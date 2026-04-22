# LTX 2.3 + Sage Attention — Resource Links

## Official repos & docs
- ComfyUI-LTXVideo (Lightricks): https://github.com/Lightricks/ComfyUI-LTXVideo
- LTX-2.3 ComfyUI workflow examples: https://docs.comfy.org/tutorials/video/ltx/ltx-2-3
- LTX-2 official docs (ComfyUI integration): https://docs.ltx.video/open-source-model/integration-tools/comfy-ui
- SageAttention (thu-ml): https://github.com/thu-ml/SageAttention

## Confirmed working setups & benchmarks
- HuggingFace discussion — LTX-2.3 NVFP4 + SageAttention benchmarks (FP8/NVFP4 with & without Sage, RTX 5090): https://huggingface.co/Lightricks/LTX-2.3-nvfp4/discussions/2
- Kijai LTXV2_comfy — "LTX2 Mem Eff Sage Attention Patch" workflow notes: https://huggingface.co/Kijai/LTXV2_comfy/discussions/41
- RuneXX LTX-2 test workflow (with Sage patch nodes): https://huggingface.co/RuneXX/LTX-2-Workflows

## Nodes needed for the Sage patch
- ComfyUI-KJNodes (provides "Patch Sage Attention KJ" node): https://github.com/kijai/ComfyUI-KJNodes
- ComfyUI-Sage-EasyInstall (Windows-friendly Sage + Triton installer): https://github.com/mickmumpitz/ComfyUI-Sage-EasyInstall

## Guides & tutorials
- LTX blog — ComfyUI workflow guide for LTX-2.3: https://ltx.io/model/model-blog/comfyui-workflow-guide
- YouTube — Faster LTX & Wan generations with SageAttention + Triton: https://www.youtube.com/watch?v=YZ3quLdLMqk
- AIStudyNow — LTX 2.3 three-stage workflow writeup: https://aistudynow.com/how-i-fixed-ltx-2-3-video-artifacts-comfyui-3-stage-workflow/

## Known issues / caveats to check before setup
- GGUF + Sage dtype error (fp16/bf16 required): https://github.com/city96/ComfyUI-GGUF/issues/403
- Feature request tracking native Sage + torch.compile + FP16 for LTX-2: https://github.com/Lightricks/ComfyUI-LTXVideo/issues/421
- Earlier LTX-Video + `--use-sage-attention` static-image bug (LTX-Video 13B, context for regressions): https://github.com/Lightricks/LTX-Video/issues/170
