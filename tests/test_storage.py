"""Tests for storage module (no Supabase credentials needed)."""

import os
from unittest.mock import patch

from src.storage import is_configured


class TestIsConfigured:
    def test_not_configured_when_env_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_configured() is False

    def test_not_configured_when_partial(self):
        with patch.dict(os.environ, {"SUPABASE_URL": "https://x.supabase.co"}, clear=True):
            assert is_configured() is False

    def test_configured_when_both_set(self):
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://x.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-key",
        }, clear=True):
            assert is_configured() is True
