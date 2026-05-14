"""Single source of truth for runtime configuration.

``python-dotenv`` (entrypoints only — ``service.py`` and
``src/download_models.py``'s ``__main__``) populates ``os.environ`` from
``.env``; :func:`get_settings` reads ``os.environ`` exclusively. Production
env always wins because the orchestrator sets it before our code runs.
Settings does NOT read ``.env`` directly so tests can't pick up a dev's
local ``.env``.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of every environment variable the service consumes."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Model storage ───────────────────────────────────────────────────
    model_dir: str = "/models"
    hf_token: str | None = None

    # ── ComfyUI graph runtime ───────────────────────────────────────────
    comfyui_path: str = "/app/ComfyUI"

    # ── SageAttention (opt-in) ──────────────────────────────────────────
    # Master switch. Default OFF; flipping SAGE_ATTENTION=1 in the pod env
    # makes bootstrap_once monkey-patch comfy.ldm.modules.attention to use
    # an explicit Blackwell-safe SageAttention CUDA kernel (avoids the
    # auto-dispatcher's Triton path implicated in Comfy-Org/ComfyUI#11583).
    sage_attention: bool = False
    # Selects which CUDA kernel the monkey-patch installs.
    #   "int8" → sageattn_qk_int8_pv_fp16_cuda  (conservative; best match
    #            against the Phase 1.6d failure mode — per-block INT8 with
    #            K-smoothing, ~10× FP8 E4M3 headroom in QK).
    #   "fp8"  → sageattn_qk_int8_pv_fp8_cuda   (faster, but FP8 PV stage
    #            carries residual cross-attention risk on long sequences).
    sage_attention_kernel: Literal["int8", "fp8"] = "int8"

    # ── Supabase Storage ────────────────────────────────────────────────
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_bucket: str = "ltx-videos"
    supabase_url_expiry_seconds: int = 604800


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` (no cache — tests rely on
    ``patch.dict(os.environ, ...)`` taking effect per call; ~200 µs, off
    every hot path)."""
    return Settings()
