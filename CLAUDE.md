# CLAUDE.md

LTX-2.3 22B BentoML video-generation service. NGC PyTorch 25.06 + FA3 + FP8 scaled_mm + optional TeaCache. Targets H100/H200 RunPod pods.

## Project layout

```
.
├── Dockerfile                    NGC base + LTX-2 install + FA3 wheel + torchaudio stub + ACL fail-fast
├── service.py                    BentoML entrypoint: /generate_sync, /generate/submit|status|get
├── start.sh                      Pod boot: download_models → bentoml serve
├── setup.sh                      Local dev bootstrap
├── bentofile.yaml                BentoML build manifest
├── pyproject.toml                Project deps (cp312, anyio>=4.9, no torchaudio)
├── test_input.json               Sample T2V request body
├── README.md
│
├── src/                          Application code (≈2.9 kLOC)
│   ├── __init__.py
│   ├── upstream.py               ★ ACL — single chokepoint for ltx_core/ltx_pipelines symbols
│   ├── pipeline/                 LTXVideoGenerator package — split by scenario
│   │   ├── __init__.py           re-exports public surface (LTXVideoGenerator, DEFAULT_NEGATIVE_PROMPT, _round_to, _round_frames)
│   │   ├── core.py               module-level FP8 handoff, shared helpers (probe / scaled_mm policy / audit / stage-2 cleanup), LTXVideoGenerator class (__init__, _ensure_mode, _extras_for, _log_*, generate dispatcher)
│   │   ├── t2v.py                T2VMixin — _build_t2v + _t2v_generate (TI2VidTwoStagesPipeline path)
│   │   ├── i2v.py                I2VMixin — _build_unified + _unified_generate + _run_i2v (ICLoraPipeline lifecycle, shared with V2V)
│   │   ├── v2v.py                V2VMixin — _run_v2v only (build inherited from I2VMixin)
│   │   └── inputs/               Request-time input materialisers (URL/b64 → tempfile)
│   │       ├── __init__.py       re-exports materialize_image, derive_dims_from_image, materialize_video
│   │       ├── image.py          I2V image input (PIL validation, auto-AR dim derivation)
│   │       └── video.py          V2V reference-video input (lighter validation, ICLora cracks the container)
│   ├── service.py (../)          (BentoML service, see top-level service.py)
│   ├── attention_override.py     FA3 enable + AttentionFunction singleton (recompile fix)
│   ├── compile_override.py       torch.compile config shim for NGC 25.06 (missing flag hasattr-gate)
│   ├── teacache.py               Opt-in step caching (ENABLE_TEACACHE=1) — DO NOT enable on I2V
│   ├── download_models.py        First-boot HF model download (~64 GB)
│   └── storage.py                Supabase upload + signed URL
│
├── tests/                        pytest suite (mocked — no GPU/upstream needed)
│   ├── conftest.py
│   ├── test_service.py           BentoML endpoints (MockGenerator)
│   ├── test_pipeline.py
│   ├── test_pipeline_inputs_image.py
│   ├── test_download_models.py
│   └── test_storage.py
│
├── deploy/
│   ├── runpod/{deploy.sh, README.md, CLAUDE.md}
│   └── vast/{deploy.sh, README.md, CLAUDE.md}
│
├── docs/                         Reference docs (FP8 spec, FA3 guide, latency, caching, A/B results)
├── scripts/                      build_fa3_wheel.sh, runpod_build_fa3_wheel.sh
├── .github/workflows/docker-build.yml   GHCR build on main + workflow_dispatch
└── LTX-2-ref/                    Read-only checkout of upstream for local reference (not shipped)
```

## Boot order (load-bearing)

`start.sh` → `python -m src.download_models` → `bentoml serve service:LTXVideoGenerator` → `service.py` imports `src.pipeline` → `LTXVideoGenerator.__init__` calls (in order):

1. `enable_attention_callable_singleton()` — must run before any transformer build
2. `enable_flash_attention_3()` — patches configurator + mask fallback
3. `enable_compile_config_shim()` — replaces `compile_transformer` + rebinds `COMPILE_TRANSFORMER` on `ltx_pipelines.utils.blocks`
4. `enable_teacache(...)` if `ENABLE_TEACACHE=1` (T2V only)

Patch sites must be the only consumers of upstream module references; any new patch goes through `src/upstream.py`.

## Upstream contract (`src/upstream.py`)

Three import styles, picked per symbol — see file docstring. **Rule**: every `from ltx_core...` / `from ltx_pipelines...` lives here and nowhere else. Module-level patch targets (`compile_transformer`, `COMPILE_TRANSFORMER`) MUST come in via `import X as ltx_X` (Style 2), not `from X import Y`. Class-method patch targets are safe as direct re-exports (Style 1). Optional symbols use `HAS_*` flags (Style 3).

Bump SHA: edit `ARG LTX2_UPSTREAM_SHA` in `Dockerfile`. Build-time `python -c "import src.upstream"` fails fast on rename.

---

# Python coding practices for this repo

These are the rules to apply during refactoring. Each one exists because of a specific failure mode in this codebase or its upstream coupling — not generic style preference.

## Imports & module structure

- **Every upstream symbol goes through `src/upstream.py`.** No direct `from ltx_core...` or `from ltx_pipelines...` outside that file.
- **Lazy-import overrides inside the function that uses them**, not at module top. Keeps boot order explicit and avoids triggering `torch._inductor` init at import time.
- **`from X import Y` binds `Y` at import time.** If `X.Y` will be reassigned later (monkey-patch), use `import X as alias` and read `alias.Y` lazily.
- No wildcard imports. No `import *` re-exports.
- Standard import order: stdlib, third-party, local — separated by blank lines. Type-only imports under `if TYPE_CHECKING:`.

## Functions & types

- Type-hint all public functions. Use `from __future__ import annotations` for forward refs in type-heavy modules.
- Prefer `dataclass` (or `@dataclass(frozen=True, slots=True)` where appropriate) over ad-hoc dicts for structured state.
- Return early; flatten nesting. Max ~3 levels of indentation in any function.
- Functions ≤50 lines; if longer, the split usually surfaces a real seam.
- One responsibility per function. If the docstring needs "and", split it.

## Errors & boundaries

- **Never bare `except:`.** Catch the specific exception you can actually handle.
- **Never `except Exception` silently** — log with `exc_info=True` or re-raise. Diagnostic-only catches must be commented `# noqa: BLE001 — diagnostic, never fail the build` (see `compile_override.py:89`).
- Validate at system boundaries only (HTTP request bodies, file inputs, env vars). Trust internal calls.
- Fail loud at boot, not deep in a generation. The Dockerfile's `import src.upstream` check is the canonical pattern.

## State & side effects

- Module-level code does imports and constants only — no I/O, no GPU calls, no upstream class instantiation.
- Idempotent overrides: `_applied` / `_singleton_applied` guard re-entry (see `attention_override.py:39-40`).
- Mutable default args are forbidden (`def f(x=[])`). Use `None` + initialize inside.

## Logging

- Use `logger = logging.getLogger(__name__)` per module — never `print` outside `__main__` blocks.
- Use `%`-style lazy interpolation: `logger.info("foo=%s", val)`, not f-strings (avoids formatting cost when level is filtered).
- Log decisions and state transitions, not chatter. Include `id(...)` / shape / dtype where it actually helps debug a recompile or a mask path.

## Comments & docstrings

- Module docstring: what the module owns + why it exists (1 paragraph + key references). See `src/upstream.py` and `src/attention_override.py` as the bar.
- Function/class docstrings: contract — args, returns, side effects, idempotency. No restating the code.
- Inline comments explain WHY, never WHAT. If the comment paraphrases the next line, delete it.
- Reference upstream commits/PRs/file paths when a workaround is keyed off them (see `compile_override.py:5-8`).

## Dependencies

- Only add a dep if it crosses a real boundary (HTTP, video I/O, infra SDK). No utility libs for things stdlib does.
- Pin floors with reason comments in `pyproject.toml` (see the `anyio>=4.9` block — load-bearing for httpx_ws).
- Never let `uv` resolve a dep that pulls a torch pin (torchaudio is the cautionary tale).

## Tests

- Tests run without GPU and without upstream installed. Mock at the `LTXVideoGenerator` boundary (see `tests/test_service.py::MockGenerator`).
- One test file per source module. Test names describe the behaviour, not the function: `test_submit_returns_pending_then_completed`.
- No network, no Supabase, no HF Hub in unit tests. Use `pytest` fixtures + `monkeypatch`.

## Concurrency & I/O

- BentoML async endpoints are `async def`; CPU/GPU heavy work goes through `asyncio.to_thread` or BentoML's worker pool. Never block the event loop.
- Long-poll endpoints (`/generate/status`) must be cheap and idempotent.

## What NOT to do

- Don't add abstractions for hypothetical futures (Protocols, ABCs, factories) until a second concrete case exists.
- Don't widen exception handling to "make a test pass."
- Don't suppress `TORCH_LOGS` or warnings to silence noise — fix the root cause.
- Don't refactor `src/upstream.py` to "clean up" Style 2 module references into Style 1 imports. They are load-bearing (see file docstring §3).
- Don't enable TeaCache on I2V (memory: `teacache_i2v_quality.md`).

## Refactor checklist (apply per PR)

1. No new `from ltx_core` / `from ltx_pipelines` outside `src/upstream.py`.
2. `pytest tests/` green.
3. `docker build .` reaches the `import src.upstream` check.
4. End-to-end Maya I2V on a test pod: SHA256 + latency within ±2 s of pre-refactor baseline.
5. `git diff` reads as one focused change. If it doesn't, split the PR.

---

# Configuration: single source of truth

## Shape (as of commit on `feature/fp8-h100-refactor`)

Every env var the service reads is declared as a field on `Settings` in **`src/config.py`** (~125 lines). `python-dotenv` populates `os.environ` from `.env` at the two entrypoints; pydantic-settings types it everywhere else.

```
                                  src/config.py
                                  ─────────────
                                  class Settings(BaseSettings):
                                      log_level: str = "INFO"
.env (local dev only) ───┐         model_dir: str = "/models"
                         │  load   hf_token: str | None = None
                         ▼  dotenv ltx_fp8_mode: str = ""
                       os.environ  ltx_attention_type: str = ""
                         ▲          ltx_default_mode: str = "i2v"
RunPod pod template ─────┘          enable_torch_compile: bool = True
GH Actions secrets                  enable_teacache: bool = False
                                    teacache_threshold: float = 0.03
                                    teacache_stages: tuple[str, ...] = ("stage_1",)
                                    supabase_url: str | None = None
                                    supabase_service_key: str | None = None
                                    supabase_bucket: str = "ltx-videos"
                                    supabase_url_expiry_seconds: int = 604800

                                  def get_settings() -> Settings:
                                      return Settings()   # fresh, no cache
```

Entrypoints that call `load_dotenv()` (only these two — never library modules):

- `service.py` — top of file, before `from src.pipeline import ...` (load-bearing because `src/pipeline/core.py` runs a module-level FP8-mode check at import).
- `src/download_models.py` — inside `if __name__ == "__main__":`.

Consumers:
- 14 fields, all routed through `from src.config import get_settings` (9 `get_settings()` call sites in `src/`, 2 in `service.py`).
- Only writer to `os.environ` left in the codebase: `src/pipeline/core.py`'s module-level `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` — that's a torch-allocator handoff, not config reading.

## Rules

- **No `os.getenv` / `os.environ.get` outside `src/config.py`.** Add a field to `Settings`, then `get_settings().<field>`.
- **`load_dotenv()` lives only in entrypoints.** Library modules import `get_settings`, never `dotenv`.
- **Production env always wins over `.env`.** The orchestrator (RunPod template, GitHub Actions secrets) sets `os.environ` before our process starts; `load_dotenv()` does not override existing env. In production `.env` is absent and `load_dotenv()` is a silent no-op.
- **`.env` is gitignored.** `.env.example` is committed and mirrors the `Settings` field order so a copy-paste produces a runnable config.
- **`get_settings()` is not cached.** Every call constructs a fresh `Settings()` so existing `patch.dict(os.environ, ...)` tests work without fixture changes. Cost is ~200 µs and not on any hot path.
- **Lenient validators preserve operability.** `field_validator(mode="before")` on `teacache_threshold` and `teacache_stages` warns and falls back to defaults rather than crashing pod boot on a typo. Add the same pattern for any future env where a malformed value should not down the service.
- **Booleans go through pydantic.** No hand-rolled `value.strip().lower() in {"1","true","yes"}` parsing — pydantic accepts `1/0/true/false/yes/no/on/off` case-insensitively for `bool` fields.
- **Secrets** (`hf_token`, `supabase_service_key`) are `str | None` today. If they ever appear in logs or repr, switch them to `pydantic.SecretStr` and call `.get_secret_value()` at the consumer site.

---

# Async / blocking-call audit

BentoML's `@bentoml.api` and `@bentoml.task` route sync handlers to a thread pool, so today's synchronous code does not technically block the event loop. But it caps concurrency at the worker pool size and makes I/O-bound steps (HTTP downloads, Supabase uploads) run serially per request — even though `traffic.max_concurrency=3`.

## Findings

| Site | Call | Type | Severity | Recommendation |
|---|---|---|---|---|
| `src/pipeline/inputs/image.py` (`_fetch_url`) | `httpx.get(image_url, ...)` | sync HTTP, up to 50 MB / 30 s | **High** | switch to `httpx.AsyncClient` + `async def materialize_image` |
| `src/pipeline/inputs/video.py` (`_fetch_url`) | `httpx.get(reference_video_url, ...)` | sync HTTP, up to 200 MB / 60 s | **High** | switch to `httpx.AsyncClient` + `async def materialize_video` |
| `src/storage.py:53-58` | `open(...)` + `client.storage.from_(b).upload(f)` | sync file read + sync Supabase SDK | **Medium** | wrap upload in `asyncio.to_thread(...)` (Supabase Python SDK is sync); use `aiofiles` for the read or pass bytes directly |
| `src/storage.py:66-69` | `client.storage.create_signed_url(...)` | sync Supabase SDK | **Medium** | same — `asyncio.to_thread` |
| `src/pipeline/core.py` (`_probe_fp8_exclusions`) | `safe_open(path, framework="pt")` (model load) | sync, multi-GB | Low (boot-time only) | leave — runs once in `__init__` |
| `src/pipeline/inputs/image.py` (`_validate_image_bytes`, `derive_dims_from_image`) | `Image.open(...)` | sync, ms | Low | leave |
| `src/pipeline/{t2v,i2v}.py` (generation) | torch CUDA work | GPU-bound | Low | leave — wrap the whole `generator.generate(...)` call in `asyncio.to_thread` from the handler instead of trying to make CUDA async |
| `service.py:75, 131` | `def generate(...)`, `def generate_sync(...)` | sync handler | Medium | convert to `async def`, call `result = await asyncio.to_thread(self.generator.generate, **kwargs)` so HTTP downloads + GPU run can interleave with other requests |
| `service.py:122` | `os.remove(result["output_path"])` | sync, ms | Low | leave |

`time.sleep` is not used anywhere. Good.

## Rules

- **Handlers (`@bentoml.api`, `@bentoml.task`) are `async def`.** CPU/GPU-heavy calls go through `await asyncio.to_thread(fn, *args)`.
- **HTTP egress uses `httpx.AsyncClient`**, never `httpx.get` / `requests.get` from inside a request path. One shared `AsyncClient` per service instance, created in `__init__` and reused (HTTP/2 + connection pooling).
- **No sync SDK call inside an `async def`** without `asyncio.to_thread`. The Supabase Python SDK is sync — wrap, don't await it directly.
- **No `open()` inside `async def`** for files that aren't tiny constants. Use `aiofiles` or `asyncio.to_thread`.
- **`time.sleep` is banned in async paths.** Use `asyncio.sleep`.
- Module-load-time I/O (model weights, FA3 wheel) stays sync — those run once in `__init__`, not on the request path.

## Suggested order if/when we refactor this

0. ✓ **Shipped on `feature/fp8-h100-image-input-support`** — `src/config.py` + `python-dotenv` replaced every `os.getenv` site; `load_dotenv()` lives only in the two entrypoints.
1. Convert `service.py` handlers to `async def` + `asyncio.to_thread` for `generator.generate(...)` and `storage.upload_video(...)`.
2. Convert `src/pipeline/inputs/image.py` / `video.py` to `httpx.AsyncClient` and `async def materialize_*`. Update callers.
3. Re-run pytest (mock the async client) and the Maya I2V end-to-end on a pod. SHA256 should match within transcoder noise.
