"""ComfyUI k-diffusion samplers — vendored (denoising loops only).

Source: https://github.com/comfyanonymous/ComfyUI (also mirrored at
        https://github.com/Comfy-Org/ComfyUI)
Path:   comfy/k_diffusion/sampling.py
Commit: f505cb4070d197f8fc783938319cf49015548e80

Per-function permalinks (pin the SHA above when bumping):

* to_d:
    https://github.com/comfyanonymous/ComfyUI/blob/f505cb4070d197f8fc783938319cf49015548e80/comfy/k_diffusion/sampling.py
* get_ancestral_step:                ^^^ same file ^^^
* default_noise_sampler:             ^^^ same file ^^^
* sample_euler_ancestral_cfg_pp:     ^^^ same file ^^^
* sample_euler_cfg_pp:               ^^^ same file ^^^

Why vendored, not imported from a ComfyUI install:

The full ComfyUI app (graph executor, ``folder_paths``, ``model_management``,
custom-node registry) is too heavy for this service and would bypass our
FA3 + FP8 scaled_mm + torch.compile patches against ``ltx_core`` (which the
ComfyUI-LTXVideo nodes do NOT use; they call a different model path). We
only need the pure-Python sampler math, which is ~30 lines per function and
has no dependency on ComfyUI's runtime beyond two stub-able hooks:

1. ``model.inner_model.model_patcher.get_model_object("model_sampling")`` +
   ``partial(sigma_to_half_log_snr, model_sampling=...)`` — upstream uses this
   to pick the noise-schedule branch (CONST/RF vs ε-prediction). LTX-2 is a
   CONST/RF model; we hardcode that branch as ``_rf_lambda_fn`` (verbatim
   equivalent of ``comfy.k_diffusion.sampling.sigma_to_half_log_snr`` at the
   ``isinstance(..., CONST)`` branch: ``sigma.logit().neg()``).

2. ``comfy.model_patcher.set_model_options_post_cfg_function`` registers a
   callback that captures ``uncond_denoised`` after the CFG combine. Our
   workflow has cfg=1 on every CFGGuider, which means ``uncond_denoised ==
   denoised`` always (no separate unconditional pass). We bypass the callback
   and set ``uncond_denoised = denoised`` inline. This matches what ComfyUI
   would compute at cfg=1 exactly.

How to refresh from upstream:

    curl -fsSL https://raw.githubusercontent.com/comfyanonymous/ComfyUI/<SHA>/comfy/k_diffusion/sampling.py \\
        > /tmp/sampling.py
    # Copy the relevant function bodies into this file. Keep the two stubs
    # documented above. Update the SHA at the top of this module + in
    # src/vendor/README.md. Run pytest tests/.

Public surface:

* ``cfgpp_denoising_step(...)`` — per-step math factored out of the body of
  ``sample_euler_ancestral_cfg_pp``. Use this when you already have a
  ``(x, denoised)`` pair (e.g. from a joint AV denoiser) and want to apply
  the CFG++ ancestral Euler step without going through the ComfyUI-shaped
  model-callable API.
* ``sample_euler_ancestral_cfg_pp(model, x, sigmas, ...)`` — the verbatim
  upstream function. Delegates per-step math to ``cfgpp_denoising_step``;
  the two functions can never drift.
* ``sample_euler_cfg_pp(model, x, sigmas, ...)`` — verbatim upstream
  wrapper: ``sample_euler_ancestral_cfg_pp(..., eta=0.0, s_noise=0.0)``.
"""

import torch
from tqdm.auto import trange


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


def default_noise_sampler(x, seed=None):
    if seed is not None:
        if x.device == torch.device("cpu"):
            seed += 1

        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
    else:
        generator = None

    return lambda sigma, sigma_next: torch.randn(x.size(), dtype=x.dtype, layout=x.layout, device=x.device, generator=generator)


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

    Factored out so a joint audio-video denoising loop can call ``denoiser``
    once per step (one transformer forward) and then apply identical CFG++
    ancestral-Euler math to each modality independently, sharing the
    ``denoiser``'s output. The two-call sampler-driver alternative would
    double the transformer work per step.

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


# ── Verbatim sampler entry points ───────────────────────────────────────────
# Body delegates per-step math to ``cfgpp_denoising_step`` so the two cannot
# drift. Upstream signature preserved.


@torch.no_grad()
def sample_euler_ancestral_cfg_pp(
    model, x, sigmas, extra_args=None, callback=None, disable=None,
    eta=1., s_noise=1., noise_sampler=None, lambda_fn=_rf_lambda_fn,
):
    """Ancestral sampling with Euler method steps (CFG++).

    Verbatim from upstream ``comfy.k_diffusion.sampling.sample_euler_ancestral_cfg_pp``
    with two stubs (see module docstring §1, §2 for the rationale):

    * ``model_sampling`` lookup replaced by ``lambda_fn`` kwarg (default = CONST/RF).
    * ``set_model_options_post_cfg_function`` skipped; we set
      ``uncond_denoised = denoised`` inline (cfg=1 contract).
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        # cfg=1 contract: see module docstring §2. At cfg=1 the CFGGuider's
        # combined output equals the unconditional output (no separate uncond
        # pass), so uncond_denoised == denoised.
        uncond_denoised = denoised
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        x = cfgpp_denoising_step(
            x, denoised, uncond_denoised, sigmas[i], sigmas[i + 1],
            eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, lambda_fn=lambda_fn,
        )
    return x


@torch.no_grad()
def sample_euler_cfg_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Euler method steps (CFG++).

    Verbatim from upstream: 1-line wrapper around
    ``sample_euler_ancestral_cfg_pp`` with ``eta=0.0, s_noise=0.0``.
    """
    return sample_euler_ancestral_cfg_pp(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=None,
    )
