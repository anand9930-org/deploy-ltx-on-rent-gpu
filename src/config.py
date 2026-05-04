"""Single source of truth for runtime configuration.

``python-dotenv`` (entrypoints only — ``service.py`` and
``src/download_models.py``'s ``__main__``) populates ``os.environ`` from
``.env``; :func:`get_settings` reads ``os.environ`` exclusively. Production
env always wins because the orchestrator sets it before our code runs.
Settings does NOT read ``.env`` directly so tests can't pick up a dev's
local ``.env``. Rules + add-a-field workflow: see CLAUDE.md `Configuration`.
"""

from __future__ import annotations

import warnings

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed view of every environment variable the service consumes."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        # Ignore unrelated env vars (PATH, HOME, BENTOML_*, …) so they do
        # not collide with our field names or trip "extra fields" errors.
        extra="ignore",
        # ``model_dir`` collides with pydantic's reserved ``model_*``
        # namespace and would otherwise emit a UserWarning at every
        # Settings() instantiation. We rely on the field name matching
        # the env var ``MODEL_DIR``, so opt out of the namespace check.
        protected_namespaces=(),
    )

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Model storage ───────────────────────────────────────────────────
    model_dir: str = "/models"
    hf_token: str | None = None

    # ── FP8 / attention / pipeline mode ─────────────────────────────────
    # Empty string is the documented "absent" sentinel — downstream code
    # branches on ``ltx_fp8_mode in ("scaled_mm", "cast")`` so an unset
    # var must compare unequal to both, which "" does.
    ltx_fp8_mode: str = ""              # "", "scaled_mm", "cast"
    ltx_attention_type: str = ""        # "", "flash_attention_3", "sdpa"
    ltx_default_mode: str = "i2v"       # "t2v" or "i2v"
    enable_torch_compile: bool = True

    # ── TeaCache ────────────────────────────────────────────────────────
    enable_teacache: bool = False
    teacache_threshold: float = 0.03
    teacache_stages: tuple[str, ...] = ("stage_1",)

    # ── Supabase Storage ────────────────────────────────────────────────
    supabase_url: str | None = None
    supabase_service_key: str | None = None
    supabase_bucket: str = "ltx-videos"
    supabase_url_expiry_seconds: int = 604800

    # ── Lenient validators ──────────────────────────────────────────────
    # Preserve the pre-config defensive parsing semantics from
    # ``src.teacache.teacache_config_from_env``: warn and fall back to
    # the default rather than crashing the whole pod boot when an
    # operator typos the env var.

    @field_validator("teacache_threshold", mode="before")
    @classmethod
    def _coerce_teacache_threshold(cls, v: object) -> float:
        try:
            f = v if isinstance(v, float) else float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            warnings.warn(
                f"TEACACHE_THRESHOLD={v!r} is not a float; falling back to 0.03",
                stacklevel=2,
            )
            return 0.03
        if not (0.0 < f <= 1.0):
            warnings.warn(
                f"TEACACHE_THRESHOLD={f} is out of (0, 1]; falling back to 0.03",
                stacklevel=2,
            )
            return 0.03
        return f

    @field_validator("teacache_stages", mode="before")
    @classmethod
    def _split_teacache_stages(cls, v: object) -> object:
        # Accept "stage_1,stage_2" CSV from env. An all-whitespace or empty
        # value falls back to the default tuple. Non-string values
        # (already a tuple/list, e.g. constructed in tests) pass through.
        if isinstance(v, str):
            parts = tuple(s.strip() for s in v.split(",") if s.strip())
            return parts or ("stage_1",)
        return v


def get_settings() -> Settings:
    """Return a fresh :class:`Settings` (no cache — tests rely on
    ``patch.dict(os.environ, ...)`` taking effect per call; ~200 µs, off
    every hot path)."""
    return Settings()
