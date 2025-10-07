# src/models/augmentation_utils.py

import torch
from typing import Tuple

# ----------------------- GPU-side noise/mask helpers -----------------------

@torch.no_grad()
def _rand_cauchy(shape, *, device, generator=None, dtype=torch.float32):
    # Inverse-CDF: tan(pi*(U-0.5))
    u = torch.rand(shape, device=device, generator=generator, dtype=dtype)
    return torch.tan(torch.pi * (u - 0.5))

@torch.no_grad()
def _sample_truncated_folded_cauchy(mu: float, sigma: float, threshold: float,
                                    shape, *, device, generator=None, max_iters: int = 10):
    """
    Samples |Cauchy(0,1)| scaled and shifted: mu + sigma * |Cauchy|
    If threshold > 0, resamples values > threshold (up to max_iters).
    Returns tensor of shape, dtype float32, device=device.
    """
    out = mu + sigma * torch.abs(_rand_cauchy(shape, device=device, generator=generator))
    if threshold > 0:
        mask = out > threshold
        it = 0
        while mask.any() and it < max_iters:
            resample = mu + sigma * torch.abs(_rand_cauchy(mask.sum().item(),
                                                           device=device, generator=generator))
            out[mask] = resample
            mask = out > threshold
            it += 1
    return out

@torch.no_grad()
def _apply_noise_and_mask_on_device(
    x: torch.Tensor,
    *,
    gen: torch.Generator,
    use_cauchy_gauss: bool = True,
    cauchy_mu: float = 0.003,
    cauchy_sigma: float = 0.0032,
    cauchy_threshold: float = 0.07,
    add_gauss_sigma: float = 0.0,
    mask_frac: float = 0.0,
    mask_mode: str = "sample",
):
    device = x.device

    # ---- Noise ----
    if use_cauchy_gauss:
        shape_rms = (x.shape[0],) + (1,) * (x.ndim - 1)
        rms = _sample_truncated_folded_cauchy(
            cauchy_mu, cauchy_sigma, cauchy_threshold,
            shape_rms, device=device, generator=gen
        ).to(torch.float32)

        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * rms
        if add_gauss_sigma > 0.0:
            noise = noise + torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        
        # Apply noise only where signal is non-zero
        signal_mask = (x != 0).to(x.dtype)
        x = x + (noise.to(x.dtype) * signal_mask)

    elif add_gauss_sigma > 0.0:
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        signal_mask = (x != 0).to(x.dtype)
        x = x + (noise.to(x.dtype) * signal_mask)

    # ---- Mask ----
    if mask_frac > 0.0:
        if mask_mode == "sample":
            b = x.shape[0]
            k = int(round(mask_frac * b))
            if k > 0:
                perm = torch.randperm(b, generator=gen, device=device)
                drop_idx = perm[:k]
                m = torch.ones((b,) + (1,) * (x.ndim - 1), device=device, dtype=x.dtype)
                m[drop_idx] = 0
                x = x * m
        elif mask_mode == "element":
            keep_prob = 1.0 - mask_frac
            m = (torch.rand(x.shape, device=device, generator=gen) < keep_prob).to(x.dtype)
            x = x * m
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

    return x