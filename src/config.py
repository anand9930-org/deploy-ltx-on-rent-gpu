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
    # VRAM mode passed to ComfyUI at boot. `gpu-only` (default): everything
    # GPU-resident, no offload. `normalvram`: dynamic offload; unlocks
    # PR #13618's LTX block prefetch in ComfyUI >= 783782d5d7. `highvram`:
    # same VRAM state as gpu-only but doesn't pin text encoders. Toggled via
    # COMFYUI_VRAM_MODE env var for A/B testing without code edits.
    comfyui_vram_mode: Literal["gpu-only", "normalvram", "highvram"] = "gpu-only"

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
