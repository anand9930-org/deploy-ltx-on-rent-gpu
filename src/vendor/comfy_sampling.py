"""ComfyUI k-diffusion samplers — vendored (cfg++ Euler per-step math only).

Source: https://github.com/comfyanonymous/ComfyUI (also mirrored at
        https://github.com/Comfy-Org/ComfyUI)
Path:   comfy/k_diffusion/sampling.py
Commit: f505cb4070d197f8fc783938319cf49015548e80

Functions copied / derived (all in that file at that SHA):

* ``to_d``, ``get_ancestral_step`` — copied verbatim.
* ``sigma_to_half_log_snr`` — the CONST/RF branch only → ``_rf_lambda_fn`` here.
* ``sample_euler_ancestral_cfg_pp`` / ``sample_euler_cfg_pp`` — the per-step
  (else-branch) body → ``cfgpp_denoising_step`` here. ``cfg_pp`` is the
  ancestral variant with ``eta=0, s_noise=0``.

Why vendored, not imported from a ComfyUI install:

The full ComfyUI app (graph executor, ``folder_paths``, ``model_management``,
custom-node registry) is too heavy for this service and would bypass our
FA3 + FP8 scaled_mm + torch.compile patches against ``ltx_core`` (which the
ComfyUI-LTXVideo nodes do NOT use; they call a different model path). We only
need the pure-Python sampler math, which has no dependency on ComfyUI's
runtime beyond two hooks the caller is responsible for:

1. ``model.inner_model.model_patcher.get_model_object("model_sampling")`` +
   ``partial(sigma_to_half_log_snr, model_sampling=...)`` — upstream uses this
   to pick the noise-schedule branch (CONST/RF vs ε-prediction). LTX-2 is a
   CONST/RF model; we hardcode that branch as ``_rf_lambda_fn`` (verbatim
   equivalent of the ``isinstance(..., CONST)`` branch: ``sigma.logit().neg()``).

2. ``comfy.model_patcher.set_model_options_post_cfg_function`` registers a
   callback that captures ``uncond_denoised`` after the CFG combine, and is
   installed with ``disable_cfg1_optimization=True`` — so ComfyUI runs the
   unconditional (negative-prompt) forward EVEN at cfg=1. ``denoised`` is the
   conditional prediction (the workflow's empty positive prompt);
   ``uncond_denoised`` is the negative-prompt prediction, and the cfg++
   derivative ``d = to_d(x, sigma, alpha_s · uncond_denoised)`` depends on it.
   We don't reproduce that callback machinery — the caller supplies both:
   ``ti2vid_triple_stages_comfyui._cfgpp_denoising_loop`` runs two
   ``SimpleDenoiser`` passes (positive + negative context) per step and feeds
   the post-processed predictions in as ``denoised`` / ``uncond_denoised``.

How to refresh from upstream:

    curl -fsSL https://raw.githubusercontent.com/comfyanonymous/ComfyUI/<SHA>/comfy/k_diffusion/sampling.py > /tmp/sampling.py
    # Re-sync to_d / get_ancestral_step / the cfg_pp else-branch body. Keep the
    # §1 (lambda_fn) and §2 (uncond pass) notes above accurate. Update the SHA
    # at the top of this module + in src/vendor/README.md. Run pytest tests/.

Public surface:

* ``cfgpp_denoising_step(x, denoised, uncond_denoised, sigma, sigma_next, eta,
  s_noise, noise_sampler, lambda_fn=_rf_lambda_fn)`` — one step of ComfyUI's
  ``sample_euler_ancestral_cfg_pp`` (the else-branch body, verbatim). Supply
  both the conditional prediction (``denoised``) and the unconditional /
  negative-prompt prediction (``uncond_denoised``); at ``eta=0, s_noise=0`` it
  reduces to ``euler_cfg_pp``.
* ``to_d`` / ``get_ancestral_step`` / ``_rf_lambda_fn`` — supporting helpers,
  copied verbatim from upstream.
"""

import torch


# ── Verbatim helpers ────────────────────────────────────────────────────────
# Copied from comfy/k_diffusion/sampling.py @ f505cb4. The only deviation
# from upstream is to_d's broadcast: upstream uses comfy.k_diffusion.utils.
# append_dims; we inline the equivalent to avoid importing the whole module.


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative.

    Equivalent to upstream ``return (x - denoised) / utils.append_dims(sigma, x.ndim)``.
    ``append_dims`` pads a 0-dim sigma with trailing singleton axes so it
    broadcasts against ``x``; torch's broadcasting already does this for
    0-dim sigma, so we inline only the higher-dim case.
    """
    if isinstance(sigma, torch.Tensor) and sigma.ndim and sigma.ndim < x.ndim:
        sigma = sigma.view(*sigma.shape, *([1] * (x.ndim - sigma.ndim)))
    return (x - denoised) / sigma


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step.

    NaN-handling note: when ``sigma_from`` is ``inf`` (happens at σ=1.0 on
    CONST/RF models where ``α_s = 0``), the inner expression evaluates to
    NaN. Python's built-in ``min(finite, nan)`` keeps the finite first
    argument (NaN comparisons all return False, so ``finite < nan`` is False
    and the iteration keeps ``finite``). Then ``sigma_down = sqrt(σ_to^2 −
    σ_to^2) = 0``. The downstream cfg_pp step closes cleanly:
    ``x_new = α_t · denoised + α_t · σ_up · noise``. Verified by hand.
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def _rf_lambda_fn(sigma):
    """``sigma_to_half_log_snr`` for the CONST/RF noise schedule.

    Returns ``-logit(σ) = log((1-σ)/σ)``. Verbatim equivalent of the CONST
    branch in upstream ``comfy.k_diffusion.sampling.sigma_to_half_log_snr``:

        if isinstance(model_sampling, comfy.model_sampling.CONST):
            return sigma.logit().neg()

    LTX-2 is a CONST/RF model, so this is the only branch we need. Note
    ``α_s = σ · exp(_rf_lambda_fn(σ)) = σ · (1−σ)/σ = 1−σ`` algebraically,
    but we preserve the exp/logit form for byte-equivalence with ComfyUI's
    floating-point evaluation order.
    """
    return sigma.logit().neg()


# ── Factored per-step math ──────────────────────────────────────────────────


def cfgpp_denoising_step(
    x, denoised, uncond_denoised, sigma, sigma_next, eta, s_noise,
    noise_sampler, lambda_fn=_rf_lambda_fn,
):
    """One step of ``sample_euler_ancestral_cfg_pp`` (the inner for-loop body).

    Lifted verbatim from the else-branch body in upstream
    ``comfy.k_diffusion.sampling.sample_euler_ancestral_cfg_pp``. The
    ``if sigmas[i+1] == 0`` short-circuit is preserved.

    ``denoised`` is the conditional (positive-prompt) prediction; ``uncond_denoised``
    is the unconditional (negative-prompt) prediction — ComfyUI runs both even at
    cfg=1 (``disable_cfg1_optimization=True``), see module docstring §2. Factored
    out so a joint audio-video loop can apply identical CFG++ ancestral-Euler math
    to each modality independently (the step is element-wise); see
    ``ti2vid_triple_stages_comfyui._cfgpp_denoising_loop``.

    Returns the post-step ``x`` (matches upstream's ``x = ...`` assignments).
    """
    if sigma_next == 0:
        # Upstream: `if sigmas[i + 1] == 0: x = denoised`
        return denoised

    alpha_s = sigma * lambda_fn(sigma).exp()
    alpha_t = sigma_next * lambda_fn(sigma_next).exp()
    d = to_d(x, sigma, alpha_s * uncond_denoised)   # to noise

    # DDIM stochastic sampling
    sigma_down, sigma_up = get_ancestral_step(sigma / alpha_s, sigma_next / alpha_t, eta=eta)
    sigma_down = alpha_t * sigma_down

    # Euler method
    x = alpha_t * denoised + sigma_down * d
    if eta > 0 and s_noise > 0:
        x = x + alpha_t * noise_sampler(sigma, sigma_next) * s_noise * sigma_up
    return x
