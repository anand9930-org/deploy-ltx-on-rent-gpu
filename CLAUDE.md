# CLAUDE.md

LTX-2.3 22B BentoML video-generation service. NVIDIA CUDA + ComfyUI graph pipeline.

> **Branch `feature/RTX-6000-pro-deployment`**: retargets to RTX PRO 6000 Blackwell Server Edition (96 GB, **sm_120** — verified live; earlier docs said sm_122 but the actual `torch.cuda.get_device_capability()` returns `(12, 0)`) on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` + stable PyTorch **2.8.0 (cu128)**. cu128 is the highest CUDA minor version supported by RunPod's current driver fleet (570.195.03 = CUDA 12.8 max, on both Community AND Secure Cloud as of 2026-05-14). LTX-2.3 FP8 weights are dequantized to BF16 at every matmul on this stack; Phase 1.6d tried recovering the FP8 fast path via ComfyUI's `--fast fp8_matrix_mult` flag (`torch._scaled_mm`), rolled back in Phase 1.6g after it regressed I2V image conditioning at frame counts ≥ 241 (per-tensor FP8 activation cast saturates cross-attention). ComfyUI runs with `--highvram --reserve-vram 2`. Future-work: when RunPod's fleet moves to driver 580+, upgrade to cu130 + torch 2.10.x to unlock ComfyUI's `comfy_kitchen` CUDA backend (FP8 path gated at `torch.version.cuda >= (13,)` in `comfy/quant_ops.py`) — that path has per-module enable lists that skip cross-attention. H100/Hopper / NGC PyTorch path lives on `main`.

## Project layout

```
.
├── Dockerfile                    NGC base + ComfyUI clone + node-contract fail-fast
├── service.py                    BentoML entrypoint: /generate_sync, /generate/submit|status|get
├── start.sh                      Pod boot: download_models → bentoml serve
├── setup.sh                      Local dev bootstrap
├── bentofile.yaml                BentoML build manifest
├── pyproject.toml                Project deps (cp312, anyio>=4.9)
├── test_input.json               Sample request body
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                 pydantic-settings: Settings class + get_settings()
│   ├── comfyui_runtime.py        Minimal ComfyUI bootstrap (sys.path + model dirs)
│   ├── pipeline/                 LTXVideoGenerator package
│   │   ├── __init__.py           re-exports (LTXVideoGenerator, DEFAULT_NEGATIVE_PROMPT, _round_to, _round_frames)
│   │   ├── core.py               LTXVideoGenerator class (__init__, _ensure_mode, generate)
│   │   ├── triple_stages_comfyui.py      TripleStagesComfyUIMixin — build + generate
│   │   ├── triple_stages_comfyui_graph.py  ComfyUI node cascade (loaders + 3-stage denoise + decode)
│   │   └── inputs/               Request-time input materialisers (URL/b64 → tempfile)
│   │       ├── __init__.py       re-exports materialize_image, derive_orientation
│   │       └── image.py          I2V image input (PIL validation, orientation derivation)
│   ├── download_models.py        First-boot HF model download (~63 GB)
│   └── storage.py                Supabase upload + signed URL
│
├── tests/                        pytest suite (mocked — no GPU/ComfyUI needed)
│   ├── conftest.py               MockGenerator fixture
│   ├── test_service.py           BentoML endpoints
│   ├── test_pipeline.py          _round_frames_8k1, LANDSCAPE/PORTRAIT_BUCKET, _resolve_aspect_ratio
│   ├── test_pipeline_triple_stages_comfyui.py   ComfyUI variant contract
│   ├── test_triple_stages_comfyui_graph.py      workflow literals + seed derivation + center-crop
│   ├── test_pipeline_inputs_image.py
│   ├── test_comfyui_runtime.py
│   ├── test_download_models.py
│   ├── test_storage.py
│   └── e2e/                      live-pod integration tests (not run by pytest; manual)
│       └── maya_aspect_ratio/    Maya I2V x 4 aspect_ratio scenarios (run_tests.py + inputs/outputs/payloads/results)
│
├── deploy/
│   ├── runpod/{deploy.sh, README.md, CLAUDE.md}
│   └── vast/{deploy.sh, README.md, CLAUDE.md}
│
├── docs/                         Reference docs
├── scripts/                      workflow_3mljpp.py (reference ComfyUI workflow)
├── .github/workflows/docker-build.yml   GHCR build on main + workflow_dispatch
└── LTX-2-ref/                    Read-only checkout of upstream for local reference (not shipped)
```

## Boot order

`start.sh` → `python -m src.download_models` → `bentoml serve service:LTXVideoService` → `service.py` imports `src.pipeline` → `LTXVideoGenerator.__init__` calls `_ensure_mode("triple_stages_comfyui")` → `_build_triple_stages_comfyui()` → bootstraps ComfyUI runtime + loads model weights via ComfyUI nodes.

## Pipeline architecture

Single pipeline: `TripleStagesComfyUIGraphPipeline` in `triple_stages_comfyui_graph.py`. Runs real ComfyUI **core** node classes (not ComfyUI-LTXVideo, not ltx_pipelines). The module is a hand-written cascade over the workflow `scripts/workflow_3mljpp.py`:

- Stage 1: height/4 × width/4, euler_ancestral_cfg_pp, 8 steps from sigma 1.0
- Stage 2: height/2 × width/2, euler_cfg_pp, 3 steps from sigma 0.85
- Stage 3: full resolution, euler_cfg_pp, 3 steps from sigma 0.85
- Decode: VAEDecodeTiled + LTXVAudioVAEDecode

Key constants are pinned to the workflow's node inputs in `triple_stages_comfyui_graph.py`. The cascade generates on a `/128` grid (forced by this workflow's choice of Stage 1 = final/4 — LTX-2.3 itself only requires `/32`, and a Stage-1-at-final/2 cascade would need only `/64`). The public API exposes only two buckets — `aspect_ratio="16:9"` (gen 1920×1152, out 1920×1080) and `"9:16"` (gen 1152×1920, out 1080×1920) — and `encode_to_mp4` center-crops the decoded IMAGE tensor to the 1080p output before encoding.

Models downloaded at boot (~63 GB total):
- `ltx-2.3-22b-dev-fp8.safetensors` (~30 GB) — ComfyUI handles FP8 natively
- `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (~7.6 GB) — strength 0.5, all stages
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (~1 GB)
- `gemma_3_12B_it.safetensors` (~24 GB) — consolidated Gemma text encoder

---

# Python coding practices for this repo

## Imports & module structure

- No wildcard imports. No `import *` re-exports.
- Standard import order: stdlib, third-party, local — separated by blank lines. Type-only imports under `if TYPE_CHECKING:`.
- ComfyUI node-class imports live inside `__init__` or functions, not at module top (keeps modules importable without ComfyUI).

## Functions & types

- Type-hint all public functions. Use `from __future__ import annotations` for forward refs in type-heavy modules.
- Prefer `dataclass` or `NamedTuple` over ad-hoc dicts for structured state.
- Return early; flatten nesting. Max ~3 levels of indentation in any function.
- Functions ≤50 lines; if longer, the split usually surfaces a real seam.

## Errors & boundaries

- **Never bare `except:`.** Catch the specific exception you can actually handle.
- **Never `except Exception` silently** — log with `exc_info=True` or re-raise.
- Validate at system boundaries only (HTTP request bodies, file inputs, env vars). Trust internal calls.
- Fail loud at boot, not deep in a generation. The Dockerfile's ComfyUI node-contract check is the canonical pattern.

## State & side effects

- Module-level code does imports and constants only — no I/O, no GPU calls.
- Mutable default args are forbidden (`def f(x=[])`). Use `None` + initialize inside.

## Logging

- Use `logger = logging.getLogger(__name__)` per module — never `print` outside `__main__` blocks.
- Use `%`-style lazy interpolation: `logger.info("foo=%s", val)`, not f-strings.
- Log decisions and state transitions, not chatter.

## Dependencies

- Only add a dep if it crosses a real boundary (HTTP, video I/O, infra SDK).
- Pin floors with reason comments in `pyproject.toml`.

## Tests

- Tests run without GPU and without ComfyUI installed. Mock at the `LTXVideoGenerator` boundary (see `tests/conftest.py::MockGenerator`).
- One test file per source module. Test names describe the behaviour, not the function.
- No network, no Supabase, no HF Hub in unit tests. Use `pytest` fixtures + `monkeypatch`.

## What NOT to do

- Don't add abstractions for hypothetical futures (Protocols, ABCs, factories) until a second concrete case exists.
- Don't widen exception handling to "make a test pass."
- Don't suppress `TORCH_LOGS` or warnings to silence noise — fix the root cause.

## Refactor checklist (apply per PR)

1. `ruff check src/ service.py tests/` clean.
2. `pytest tests/` green.
3. `docker build .` reaches the ComfyUI node-contract check.
4. End-to-end I2V on a test pod: visual quality + latency sanity check.
5. `git diff` reads as one focused change. If it doesn't, split the PR.

---

# Configuration: single source of truth

Every env var the service reads is declared as a field on `Settings` in **`src/config.py`**. `python-dotenv` populates `os.environ` from `.env` at the two entrypoints; pydantic-settings types it everywhere else.

```
                                  src/config.py
                                  ─────────────
                                  class Settings(BaseSettings):
                                      log_level: str = "INFO"
.env (local dev only) ───┐         model_dir: str = "/models"
                         │  load   hf_token: str | None = None
                         ▼  dotenv comfyui_path: str = "/app/ComfyUI"
                       os.environ  supabase_url: str | None = None
                         ▲          supabase_service_key: str | None = None
RunPod pod template ─────┘          supabase_bucket: str = "ltx-videos"
GH Actions secrets                  supabase_url_expiry_seconds: int = 604800

                                  def get_settings() -> Settings:
                                      return Settings()   # fresh, no cache
```

Entrypoints that call `load_dotenv()` (only these two — never library modules):

- `service.py` — top of file, before `from src.pipeline import ...`.
- `src/download_models.py` — inside `if __name__ == "__main__":`.

## Rules

- **No `os.getenv` / `os.environ.get` outside `src/config.py`.** Add a field to `Settings`, then `get_settings().<field>`.
- **`load_dotenv()` lives only in entrypoints.** Library modules import `get_settings`, never `dotenv`.
- **Production env always wins over `.env`.** The orchestrator sets `os.environ` before our process starts; `load_dotenv()` does not override existing env.
- **`get_settings()` is not cached.** Every call constructs a fresh `Settings()` so `patch.dict(os.environ, ...)` tests work without fixture changes.
- **Secrets** (`hf_token`, `supabase_service_key`) are `str | None` today. If they ever appear in logs, switch to `pydantic.SecretStr`.

---

# Async / blocking-call audit

BentoML's `@bentoml.api` and `@bentoml.task` route sync handlers to a thread pool, so today's synchronous code does not technically block the event loop.

| Site | Call | Type | Severity | Recommendation |
|---|---|---|---|---|
| `src/pipeline/inputs/image.py` (`_fetch_url`) | `httpx.get(image_url, ...)` | sync HTTP, up to 50 MB / 30 s | **High** | switch to `httpx.AsyncClient` + `async def materialize_image` |
| `src/storage.py` | sync Supabase SDK | **Medium** | wrap in `asyncio.to_thread(...)` |
| `service.py` | sync handlers | Medium | convert to `async def` + `asyncio.to_thread` |

## Rules

- **Handlers (`@bentoml.api`, `@bentoml.task`) are `async def`.** CPU/GPU-heavy calls go through `await asyncio.to_thread(fn, *args)`.
- **HTTP egress uses `httpx.AsyncClient`**, never `httpx.get` / `requests.get` from inside a request path.
- **`time.sleep` is banned in async paths.** Use `asyncio.sleep`.
