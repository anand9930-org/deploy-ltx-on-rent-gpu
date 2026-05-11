"""Tests for the vendored ComfyUI cfg++ sampler math (``src/vendor/comfy_sampling.py``).

This module only depends on ``torch``, so unlike the triple-stages pipeline
it can be imported and exercised without ltx_core / ltx_pipelines. The fixes
landed on ``feature/fp8-h100-video-triple-stages-TI`` (restore the negative-
prompt uncond pass) hinge on ``cfgpp_denoising_step`` actually consuming its
``uncond_denoised`` argument — pinned below — plus the RF/CONST edge cases.
"""

import math

import pytest
import torch

from src.vendor.comfy_sampling import (
    cfgpp_denoising_step,
    get_ancestral_step,
    to_d,
    _rf_lambda_fn,
)


class TestRfLambdaFn:
    """``_rf_lambda_fn`` = the CONST/RF branch of ComfyUI's ``sigma_to_half_log_snr``."""

    def test_logit_neg_at_half_is_zero(self):
        assert float(_rf_lambda_fn(torch.tensor(0.5))) == pytest.approx(0.0, abs=1e-6)

    def test_logit_neg_value(self):
        # -logit(0.25) = log((1-0.25)/0.25) = log(3)
        assert float(_rf_lambda_fn(torch.tensor(0.25))) == pytest.approx(math.log(3.0), abs=1e-5)

    def test_alpha_equals_one_minus_sigma(self):
        # CONST identity: alpha = sigma * exp(lambda_fn(sigma)) == 1 - sigma
        for s in (0.1, 0.421875, 0.7, 0.99375):
            sig = torch.tensor(s)
            assert float(sig * _rf_lambda_fn(sig).exp()) == pytest.approx(1.0 - s, abs=1e-5)


class TestGetAncestralStep:
    def test_eta_zero_is_deterministic(self):
        assert get_ancestral_step(0.7, 0.5, eta=0.0) == (0.5, 0.0)

    def test_eta_one_finite(self):
        sigma_down, sigma_up = get_ancestral_step(0.7, 0.5, eta=1.0)
        assert math.isfinite(sigma_down) and math.isfinite(sigma_up)
        assert 0.0 < sigma_down < 0.5
        assert sigma_up > 0.0

    def test_infinite_sigma_from_collapses_cleanly(self):
        # σ=1.0 on a CONST/RF model → alpha_s=0 → sigma_from=inf. min(finite, nan)
        # keeps the finite arg, so sigma_down=0 and sigma_up=sigma_to.
        assert get_ancestral_step(float("inf"), 0.5, eta=1.0) == (0.0, 0.5)


class TestToD:
    def test_scalar_sigma_broadcasts(self):
        x = torch.ones(2, 3)
        denoised = torch.zeros(2, 3)
        assert torch.allclose(to_d(x, torch.tensor(0.5), denoised), torch.full((2, 3), 2.0))

    def test_higher_dim_sigma_broadcasts(self):
        x = torch.ones(2, 3, 4)
        denoised = torch.zeros(2, 3, 4)
        sigma = torch.full((2,), 0.5)  # ndim=1 < x.ndim=3
        assert torch.allclose(to_d(x, sigma, denoised), torch.full((2, 3, 4), 2.0))


class TestCfgppDenoisingStep:
    def _fixed_tensors(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(2, 3, 4, generator=g)
        denoised = torch.randn(2, 3, 4, generator=g)
        uncond = torch.randn(2, 3, 4, generator=g)
        return x, denoised, uncond

    def test_sigma_next_zero_returns_denoised_verbatim(self):
        x, denoised, uncond = self._fixed_tensors()
        out = cfgpp_denoising_step(
            x, denoised, uncond, torch.tensor(0.4219), torch.tensor(0.0),
            eta=1.0, s_noise=1.0, noise_sampler=None,
        )
        assert out is denoised

    def test_eta_zero_does_not_touch_noise_sampler(self):
        x, denoised, uncond = self._fixed_tensors()

        def boom(_sigma, _sigma_next):
            raise AssertionError("noise_sampler must not be called at eta=0")

        out = cfgpp_denoising_step(
            x, denoised, uncond, torch.tensor(0.85), torch.tensor(0.4219),
            eta=0.0, s_noise=0.0, noise_sampler=boom,
        )
        assert torch.isfinite(out).all()

    def test_uncond_denoised_changes_the_output(self):
        # The whole point of restoring the negative-prompt pass: the cfg++
        # derivative d = to_d(x, sigma, alpha_s * uncond_denoised) depends on
        # uncond_denoised, so cond==uncond and cond!=uncond must diverge.
        x, denoised, uncond = self._fixed_tensors()

        def boom(_sigma, _sigma_next):
            raise AssertionError("noise_sampler must not be called at eta=0")

        same = cfgpp_denoising_step(
            x, denoised, denoised, torch.tensor(0.85), torch.tensor(0.4219),
            eta=0.0, s_noise=0.0, noise_sampler=boom,
        )
        differ = cfgpp_denoising_step(
            x, denoised, uncond, torch.tensor(0.85), torch.tensor(0.4219),
            eta=0.0, s_noise=0.0, noise_sampler=boom,
        )
        assert not torch.allclose(same, differ)

    def test_sigma_one_edge_is_finite_and_matches_rf_renoise(self):
        # At σ=1.0 (CONST/RF): alpha_s=0 → d term drops (sigma_down=0), so the
        # step reduces to x = (1-σ_next)·denoised + σ_next·noise.
        x, denoised, uncond = self._fixed_tensors()
        g1 = torch.Generator().manual_seed(123)
        out = cfgpp_denoising_step(
            x, denoised, uncond, torch.tensor(1.0), torch.tensor(0.99375),
            eta=1.0, s_noise=1.0,
            noise_sampler=lambda _s, _sn: torch.randn(x.shape, generator=g1),
        )
        assert torch.isfinite(out).all()
        g2 = torch.Generator().manual_seed(123)
        noise = torch.randn(x.shape, generator=g2)
        expected = (1.0 - 0.99375) * denoised + 0.99375 * noise
        assert torch.allclose(out, expected, atol=1e-5)
