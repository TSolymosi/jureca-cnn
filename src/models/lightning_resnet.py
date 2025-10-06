import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import torch.distributed as dist
from typing import Optional, Dict, List, Tuple
from contextlib import nullcontext
import math

from ResNet3D import generate_2d_model  # keep your import


def _gather_cat_cpu(t) -> Optional[torch.Tensor]:
    if t is None:
        obj = None
    elif isinstance(t, list):
        if len(t) == 0:
            obj = None
        else:
            obj = torch.cat(t, dim=0)
    elif torch.is_tensor(t):
        obj = t
    else:
        obj = None

    if dist.is_available() and dist.is_initialized():
        objs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(objs, obj)
        if dist.get_rank() == 0:
            tensors = [o for o in objs if isinstance(o, torch.Tensor) and o.numel() > 0]
            if len(tensors) == 0:
                return None
            return torch.cat(tensors, dim=0)
        else:
            return None
    else:
        return obj


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
        # Per-sample RMS -> (B,1,1,1,...) broadcast
        shape_rms = (x.shape[0],) + (1,) * (x.ndim - 1)
        rms = _sample_truncated_folded_cauchy(
            cauchy_mu, cauchy_sigma, cauchy_threshold,
            shape_rms, device=device, generator=gen
        ).to(torch.float32)

        # NOTE: torch.randn_like(...) doesn't accept generator => use torch.randn(...)
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * rms
        if add_gauss_sigma > 0.0:
            noise = noise + torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        # >>> ONLY add noise where signal is non-zero <<<
        signal_mask = (x != 0).to(x.dtype)
        x = x + (noise.to(x.dtype) * signal_mask)

    elif add_gauss_sigma > 0.0:
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        # >>> ONLY add noise where signal is non-zero <<<
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
            # NOTE: torch.rand_like(...) doesn't accept generator => use torch.rand(...)
            m = (torch.rand(x.shape, device=device, generator=gen) < keep_prob).to(x.dtype)
            x = x * m
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

    return x

# --------------------------------------------------------------------------


class LitResNetMDN(LightningModule):
    """
    Ports the data checks + MDN/non-MDN loss behavior into Lightning.
    Adds fast GPU-side Cauchy-noise + masking (sample- or element-wise).
    """
    def __init__(
        self, model_cfg, optim_cfg, training_cfg,
        std_weights=None,
        dump_bad_batch: bool = True,
        dump_dir: str = "./",
        # ---- New GPU-noise/masking controls (safe defaults) ----
        apply_aug_train_only: bool = False,   # apply noise/mask during training only
        use_cauchy_gauss: bool = True,       # replicate old CPU pipeline (Gaussian with Cauchy-drawn RMS)
        cauchy_mu: float = 0.003,
        cauchy_sigma: float = 0.0032,
        cauchy_threshold: float = 0.07,
        add_gauss_sigma: float = 0.0,        # extra plain Gaussian noise
        mask_frac: float = 0.0,              # 0.6 => drop 60%
        mask_mode: str = "sample",           # "sample" (drop files) or "element"
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # ----------------------- Model -----------------------
        self.config_name = self._get_config_name(model_cfg.model_depth)
        self.model = generate_2d_model(
            config_name=self.config_name,
            use_batchnorm=model_cfg.use_batchnorm,
            TARGET_PARAMETERS=model_cfg.target_params,
            n_outputs=len(model_cfg.target_params),
            use_attention_heads=model_cfg.use_attention_heads,
            attention_latent_dim=model_cfg.attention_latent_dim,
            use_mdn=model_cfg.use_mdn,
            covariance_type=model_cfg.covariance_type,
            num_mixtures=model_cfg.num_mixtures
        )
        self.model_params: List[str] = model_cfg.target_params
        self.use_mdn: bool = model_cfg.use_mdn

        # --------------------- Loss setup --------------------
        self.criterion = nn.MSELoss(reduction='mean') if not self.use_mdn else None
        self.sigma_head_outputs_positive = getattr(model_cfg, "sigma_head_outputs_positive", True)
        self.sigma_floor = getattr(model_cfg, "sigma_floor", 1e-4)

        # Optional inverse-variance weighting (unused in current NLL)
        self.std_weights = std_weights
        self.dump_bad_batch = dump_bad_batch
        self.dump_dir = dump_dir

        # cache for validation epoch end (rank0 only)
        self._val_dataset = None
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_sigmas: Optional[List[torch.Tensor]] = [] if self.use_mdn else None
        self._val_cache: Dict[str, Optional[torch.Tensor]] = {}

        # cache for TEST epoch end (rank0 only)
        self._test_dataset = None
        self._test_preds:   List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._test_sigmas:  Optional[List[torch.Tensor]] = [] if self.use_mdn else None
        self._test_Ls:      Optional[List[torch.Tensor]] = [] if (self.use_mdn and model_cfg.covariance_type == "full") else None
        self._test_cache:   Dict[str, Optional[torch.Tensor]] = {}


        # ---- Store augmentation config ----
        self.apply_aug_train_only = apply_aug_train_only
        self.aug_use_cauchy_gauss = use_cauchy_gauss
        self.aug_cauchy_mu = cauchy_mu
        self.aug_cauchy_sigma = cauchy_sigma
        self.aug_cauchy_threshold = cauchy_threshold
        self.aug_add_gauss_sigma = add_gauss_sigma
        self.aug_mask_frac = mask_frac
        self.aug_mask_mode = mask_mode  # "sample" or "element"

        # Covariance type for MDN: "full" or "diagonal"
        self.covariance_type = model_cfg.covariance_type

    # ------------------- Lightning lifecycle -------------------

    def on_fit_start(self):
        # Pull stds from the datamodule when available
        if self.std_weights is None and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "dataset_ref", None) is not None:
                sw = dm.dataset_ref.scaler_stds
                try:
                    sw = sw.detach().cpu()
                except Exception:
                    pass
                self.std_weights = sw
                if self.trainer.is_global_zero:
                    self.print(f"[on_fit_start] std_weights set from datamodule: {self.std_weights}")

        if self.trainer.is_global_zero:
            dm = self.trainer.datamodule
            ds = getattr(dm, "dataset_ref", None)
            model_params = list(self.model_params)
            data_params  = list(getattr(dm, "model_params", []))  # from data_cfg
            ds_params    = list(getattr(ds, "model_params", [])) if ds is not None else []

            print("[CHECK] model_params:", model_params)
            print("[CHECK] data_cfg.model_params:", data_params)
            print("[CHECK] dataset.model_params:", ds_params)
            assert model_params == data_params == ds_params, (
                "Parameter order/name mismatch between model and datamodule/dataset! "
                "This will misapply std_weights and corrupt per-param NLL."
            )

            means = getattr(ds, "scaler_means", None)
            stds  = getattr(ds, "scaler_stds", None)
            if means is not None and stds is not None:
                for i, name in enumerate(model_params):
                    print(f"[SCALE] {name}: mean={float(means[i]):.4g}, std={float(stds[i]):.4g}")

            print(f"[INFO] Cauchy + Gaussian noise: use_cauchy_gauss={self.aug_use_cauchy_gauss}", flush=True)
            
    def on_validation_epoch_start(self):
        self._val_dataset = getattr(self.trainer.datamodule, "dataset_ref", None)
        self._val_preds = []
        self._val_targets = []
        self._val_sigmas = [] if self.use_mdn else None
        self._val_Ls = [] if (self.use_mdn and self.covariance_type == "full") else None

    def on_test_epoch_start(self):
        self._test_dataset = getattr(self.trainer.datamodule, "dataset_ref", None)
        self._test_preds = []
        self._test_targets = []
        self._test_sigmas = [] if self.use_mdn else None
        self._test_Ls = [] if (self.use_mdn and self.covariance_type == "full") else None
        self._test_cache = {}


    # ------------------------- Helpers -------------------------

    def _get_config_name(self, model_depth: int) -> str:
        mapping = {
            10: "resnet10_2d_equivalent",
            18: "resnet18_2d_equivalent",
            34: "resnet34_2d",
            50: "resnet50_2d"
        }
        if model_depth not in mapping:
            raise ValueError(f"Unsupported model depth: {model_depth}. Supported depths: 10,18,34,50.")
        return mapping[model_depth]

    def forward(self, x: torch.Tensor):
        return self.model(x)

    @staticmethod
    def _finite_mask_per_sample(x: torch.Tensor) -> torch.Tensor:
        return ~torch.isfinite(x.view(x.shape[0], -1).sum(dim=1))

    def _sanity_checks(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        if not torch.isfinite(x).all():
            nan_mask = self._finite_mask_per_sample(x)
            bad_indices = nan_mask.nonzero(as_tuple=True)[0]
            if self.trainer.is_global_zero:
                self.print(f"\nNaNs/Inf in input at batch {batch_idx}, sample indices: {bad_indices.tolist()}")
                self.print("Corresponding target values for bad inputs:")
                for idx in bad_indices.tolist():
                    self.print(f"  Sample {idx}: {y[idx].detach().cpu().numpy()}")
                if self.dump_bad_batch:
                    os.makedirs(self.dump_dir, exist_ok=True)
                    torch.save({"data": x.detach().cpu(), "target": y.detach().cpu()},
                               os.path.join(self.dump_dir, f"nan_input_batch{batch_idx}.pt"))
            raise ValueError("NaN/Inf input encountered — likely due to bad parameter combination")

        if torch.isnan(x).any():
            raise ValueError("NaN in input")

        if x.shape[2] == 0:
            raise ValueError("Empty spectral dimension (D=0) in input")

        if not torch.isfinite(y).all():
            raise ValueError("Target contains NaNs/Inf")

    def _split_mdn(self, output) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        mu_list, sigma_list = [], []
        for i, _ in enumerate(self.model_params):
            mu    = output[i * 2]
            sigma = output[i * 2 + 1]
            mu    = mu.view(mu.shape[0], 1)
            sigma = sigma.view(sigma.shape[0], 1)
            if not self.sigma_head_outputs_positive:
                sigma = F.softplus(sigma)
            mu_list.append(mu)
            sigma_list.append(sigma)
        return mu_list, sigma_list

    def _split_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, noise_rms = batch
        else:
            x, y = batch
            noise_rms = None
        return x, y, noise_rms

    def _compute_loss(self, output, target, return_per_param: bool = False):
        if not self.use_mdn:
            if output.numel() == 0:
                raise RuntimeError(f"Empty output tensor! Shape: {output.shape}")
            if target.numel() == 0:
                raise RuntimeError(f"Empty target tensor! Shape: {target.shape}")
            if output.shape != target.shape:
                raise RuntimeError(f"Output-target shape mismatch: {output.shape} vs {target.shape}")
            if (not torch.isfinite(output).all()) or (not torch.isfinite(target).all()):
                raise ValueError("Non-finite values in output/target")
            loss = self.criterion(output, target)
            return (loss, {}) if return_per_param else loss

        per_param = {}
        losses = []
        reg_term = 0.0
        sigma_reg = False
        sigma_reg_weights = {
            "D": 0.05, "L": 0.05, "rr": 0.05, "ro": 0.05,
            "p": 0.05, "Tlow": 0.05, "plummer_shape": 0.05, "NCH3CN": 0.05
        }

        #with torch.autocast(device_type="cuda", enabled=False):
        ac = torch.amp.autocast('cuda', enabled=False) if torch.cuda.is_available() else nullcontext()
        with ac:
            for i, name in enumerate(self.model_params):
                mu    = output[i*2].squeeze(-1).float()
                sigma = output[i*2+1].squeeze(-1).float()
                var = sigma.mul(sigma).add(1e-6)
                y_i = target[:, i].float()
                li = F.gaussian_nll_loss(mu, y_i, var, full=True, reduction="none")
                per_param[name] = li.mean()
                losses.append(li)
                if sigma_reg:
                    w = sigma_reg_weights.get(name, 0.0)
                    if w > 0:
                        reg_term += w * torch.mean(torch.log(var))
        loss = torch.stack(losses, dim=1).mean() + reg_term
        return (loss, per_param) if return_per_param else loss


    # Covariance helpers
    @staticmethod
    def _is_fullcov_out(out) -> bool:
        """True when model returns a dict with full-covariance pieces."""
        return isinstance(out, dict) and ("mu" in out) and ("L" in out)

    @staticmethod
    def _split_mdn_tuple(output, n_params: int):
        """Takes flat tuple (mu1, sigma1, mu2, sigma2, ...) -> (mus[B,d], sigmas[B,d])"""
        mus, sigmas = [], []
        for i in range(n_params):
            mu_i    = output[i * 2].squeeze(-1)
            sigma_i = output[i * 2 + 1].squeeze(-1)
            mus.append(mu_i)
            sigmas.append(sigma_i)
        return torch.stack(mus, dim=1) if mus[0].ndim == 1 else torch.cat([m.unsqueeze(1) for m in mus], dim=1), \
            torch.stack(sigmas, dim=1) if sigmas[0].ndim == 1 else torch.cat([s.unsqueeze(1) for s in sigmas], dim=1)


    def _mvn_nll_from_cholesky(self, mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        AMP-safe MVN NLL with Σ = L L^T (L lower-triangular).
        We do the triangular solve + logdet in float32 for numerical stability.
        A tiny jitter is added on the diagonal to avoid singularities early in training.
        """
        if mu.ndim != 2:
            mu = mu.view(mu.shape[0], -1)
        if y.ndim != 2:
            y = y.view(y.shape[0], -1)

        ac = nullcontext()
        # if autocast is active, disable for the linear algebra
        try:
            ac = torch.amp.autocast('cuda', enabled=False) if mu.is_cuda else nullcontext()
        except Exception:
            ac = nullcontext()

        with ac:
            mu32, L32, y32 = mu.float(), L.float(), y.float()
            # ensure lower-triangular and positive diag
            L32 = torch.tril(L32)
            
            eye = torch.eye(L32.shape[-1], device=L32.device, dtype=L32.dtype).unsqueeze(0)
            L32 = L32 + 1e-6 * eye  # jitter
            # triangular solve & logdet in fp32
            diff = (y32 - mu32).unsqueeze(-1)
            z = torch.linalg.solve_triangular(L32, diff, upper=False)
            maha = z.square().sum(dim=(-2, -1))
            diag = torch.diagonal(L32, dim1=-2, dim2=-1)
            logdet = 2.0 * torch.log(diag).sum(dim=-1)
            const = mu32.shape[-1] * math.log(2.0 * math.pi)
            nll = 0.5 * (maha + logdet + const)
            loss = nll.mean()
        #return loss
        return loss.to(mu.dtype)

    # --- mixture of Gaussians (mog) NLLs + marginals ---
    def _mog_nll_diag(self, pi_logits, mu, sigma, y):
        """
        Diagonal mixture NLL.
        Shapes:
        pi_logits: (B,K)
        mu:        (B,K,d)
        sigma:     (B,K,d)  (positive)
        y:         (B,d)
        """
        log_pi = F.log_softmax(pi_logits, dim=-1)         # (B,K)
        y_exp = y.unsqueeze(1)                            # (B,1,d)
        var   = (sigma * sigma).clamp_min(1e-12)          # (B,K,d)
        const = mu.size(-1) * math.log(2.0 * math.pi)
        log_det = var.log().sum(-1)                       # (B,K)
        maha = ((y_exp - mu).square() / var).sum(-1)      # (B,K)
        log_prob = -0.5 * (const + log_det + maha)        # (B,K)
        return -(log_pi + log_prob).logsumexp(dim=-1).mean()

    def _mog_nll_full(self, pi_logits, mu, L, y):
        """
        Full-cov mixture NLL using Cholesky L.
        Shapes:
        pi_logits: (B,K)
        mu:        (B,K,d)
        L:         (B,K,d,d)  (lower-tri; diag made positive here)
        y:         (B,d)
        """
        B, K, d = mu.shape
        log_pi = F.log_softmax(pi_logits, dim=-1)         # (B,K)

        # enforce lower-tri + positive diag
        L = torch.tril(L)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_pos = F.softplus(diag) + 1e-3
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)

        ac = torch.amp.autocast('cuda', enabled=False) if y.is_cuda else nullcontext()
        with ac:
            y_mu = (y.unsqueeze(1) - mu).unsqueeze(-1)    # (B,K,d,1)
            z = torch.linalg.solve_triangular(L.float(), y_mu.float(), upper=False)
            maha = z.square().sum(dim=(-2, -1))           # (B,K)
            logdet = 2.0 * diag_pos.float().log().sum(-1) # (B,K)
            const = d * math.log(2.0 * math.pi)
            log_prob = -0.5 * (maha + logdet + const)     # (B,K)

        return -(log_pi + log_prob).logsumexp(dim=-1).mean().to(mu.dtype)

    @staticmethod
    def _mixture_marginals_diag(pi_logits, mu, sigma):
        """Return (mu_mix[B,d], sig_mix[B,d]) for diagonal mixtures."""
        pi = F.softmax(pi_logits, dim=-1)                 # (B,K)
        mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)       # (B,d)
        var_k = (sigma * sigma)                           # (B,K,d)
        second = (pi.unsqueeze(-1) * (var_k + mu * mu)).sum(dim=1)  # (B,d)
        var_mix = (second - mu_mix * mu_mix).clamp_min(1e-12)
        return mu_mix, var_mix.sqrt()

    @staticmethod
    def _mixture_marginals_full(pi_logits, mu, L):
        """Return (mu_mix[B,d], sig_mix[B,d]) using diag Σ_k = sum_j L_ij^2."""
        pi = F.softmax(pi_logits, dim=-1)                 # (B,K)
        mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)       # (B,d)
        diagSigma_k = (L * L).sum(dim=-1)                 # (B,K,d)
        second = (pi.unsqueeze(-1) * (diagSigma_k + mu * mu)).sum(dim=1)
        var_mix = (second - mu_mix * mu_mix).clamp_min(1e-12)
        return mu_mix, var_mix.sqrt()


    # --------------------- training / val / test ---------------------

    def _maybe_augment_on_device(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_aug_train_only and not self.training:
            return x
        # Per-rank, per-step generator (avoid identical noise across DDP ranks)
        gen = torch.Generator(device=self.device)
        seed = (
            int(self.global_step) * 1000003
            + int(self.current_epoch) * 9176
            + int(getattr(self, "global_rank", 0))
        )
        gen.manual_seed(seed)
        x = _apply_noise_and_mask_on_device(
            x, gen=gen,
            use_cauchy_gauss=self.aug_use_cauchy_gauss,
            cauchy_mu=self.aug_cauchy_mu,
            cauchy_sigma=self.aug_cauchy_sigma,
            cauchy_threshold=self.aug_cauchy_threshold,
            add_gauss_sigma=self.aug_add_gauss_sigma,
            mask_frac=self.aug_mask_frac,
            mask_mode=self.aug_mask_mode,
        )
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # (optional) on-GPU augments exactly as before
        x = self._maybe_augment_on_device(x)

        self._sanity_checks(x, y, batch_idx)

        use_amp = x.is_cuda and isinstance(getattr(self.trainer, "precision", ""), str) and "mixed" in self.trainer.precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x)

        if isinstance(out, dict) and ("pi_logits" in out):
            # ---- MIXTURE PATH (K>1) ----
            if "sigma" in out:   # diagonal mixture
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
            else:                 # full-cov mixture (has 'L')
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])

                # optional: keep a heatmap source for mixtures (top-weight component)
                if "L" in out:
                    pi = F.softmax(out["pi_logits"], dim=-1)
                    top = pi.argmax(dim=-1)                                  # (B,)
                    L_top = out["L"][torch.arange(out["L"].size(0)), top]    # (B,d,d)
                    cache = getattr(self, "_val_cache", {})
                    cache.setdefault("Ls", []).append(L_top.detach().cpu())
                    self._val_cache = cache

            self.log(f"train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        # ---- FULL COVARIANCE PATH ----
        if self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)

            # --- gentle diag-σ calibration in standardized space ---
            # config-driven targets & weights (defaults below)
            targets = getattr(self.hparams.model_cfg, "cov_diag_targets", {})
            weights = getattr(self.hparams.model_cfg, "cov_diag_weights", {})
            use_log = bool(getattr(self.hparams.model_cfg, "cov_diag_use_log", False))

            if targets or weights:
                with torch.no_grad():
                    sigma_diag = torch.sqrt((L.float().square()).sum(dim=-1))  # (B, d)
                # map param name -> column index
                name2idx = {n: i for i, n in enumerate(self.model_params)}
                reg_terms = []
                for name, w in weights.items():
                    if w <= 0 or name not in name2idx:
                        continue
                    i = name2idx[name]
                    tgt = float(targets.get(name, 1.0))
                    if use_log:
                        reg = (torch.log(sigma_diag[:, i].clamp_min(1e-6)) - math.log(tgt))**2
                    else:
                        reg = (sigma_diag[:, i] - tgt)**2
                    reg_terms.append(w * reg.mean())
                if reg_terms:
                    loss = loss + torch.stack(reg_terms).sum()
            # ------------------------------------------------------------

            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        # ---- DIAGONAL (old) PATH ----
        if self.use_mdn:
            # original tuple output (mu_i, sigma_i, ...)
            loss, per_param = self._compute_loss(out, y, return_per_param=True)

            if batch_idx == 0 and self.trainer.is_global_zero:
                for i, name in enumerate(self.model_params):
                    mu_i    = out[i*2].squeeze(-1)
                    sigma_i = out[i*2+1].squeeze(-1)
                    if not self.sigma_head_outputs_positive:
                        sigma_i = F.softplus(sigma_i)
                    sigma_i = torch.clamp(sigma_i, min=self.sigma_floor)
                    y_i = y[:, i]
                    nll_i = F.gaussian_nll_loss(mu_i.float(), y_i.float(), (sigma_i*sigma_i).float(),
                                                full=True, reduction="none").mean()
                    print(f"[Lightning][{name}] y(mean±std)={y_i.mean():.3g}±{y_i.std():.3g} "
                        f"mu={mu_i.mean():.3g}±{mu_i.std():.3g} sigma={sigma_i.mean():.3g} "
                        f"NLL={nll_i.item():.6f}")

            for k, v in per_param.items():
                self.log(f"train/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)  # not quite sure why this is needed
            return loss

        # ---- Plain regression (no MDN) ----
        if out.shape != y.shape:
            raise RuntimeError(f"Output-target shape mismatch: {out.shape} vs {y.shape}")
        loss = self.criterion(out, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        if not self.apply_aug_train_only:
            x = self._maybe_augment_on_device(x)

        self._sanity_checks(x, y, batch_idx)

        use_amp = x.is_cuda and isinstance(getattr(self.trainer, "precision", ""), str) and "mixed" in self.trainer.precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x)

        if isinstance(out, dict) and ("pi_logits" in out):
            # ---- MIXTURE PATH (K>1) ----
            if "sigma" in out:   # diagonal mixture
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
            else:                 # full-cov mixture (has 'L')
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])

            # cache for plots
            self._val_preds.append(mu_mix.detach().cpu())
            self._val_targets.append(y.detach().cpu())
            self._val_sigmas = (self._val_sigmas or [])
            self._val_sigmas.append(sig_mix.detach().cpu())

            # cache per-component tensors for visualization
            cache = getattr(self, "_val_cache", {})
            cache.setdefault("pi_logits_all", []).append(out["pi_logits"].detach().cpu())  # (B,K)
            cache.setdefault("mu_all", []).append(out["mu"].detach().cpu())                # (B,K,d)
            if "sigma" in out:
                cache.setdefault("sigma_all", []).append(out["sigma"].detach().cpu())      # (B,K,d)
            if "L" in out:
                cache.setdefault("L_all", []).append(out["L"].detach().cpu())              # (B,K,d,d)
            self._val_cache = cache

            self.log(f"val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return {"val/loss": loss}


        # FULL COV
        if self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)
            sigma_diag = torch.sqrt((L.square()).sum(dim=-1))  # (B,d)

            if not hasattr(self, "_val_Ls") or self._val_Ls is None:
                self._val_Ls = []
            self._val_Ls.append(L.detach().cpu())

            self._val_preds.append(mu.detach().cpu())
            self._val_targets.append(y.detach().cpu())
            self._val_sigmas = (self._val_sigmas or [])
            self._val_sigmas.append(sigma_diag.detach().cpu())

            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            #self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)    # not quite sure why this is needed
            return {"val/loss": loss}

        # DIAGONAL (old)
        if self.use_mdn:
            mus, sigmas = self._split_mdn_tuple(out, len(self.model_params))
            # enforce positivity if your head doesn't
            if not self.sigma_head_outputs_positive:
                sigmas = F.softplus(sigmas)
            sigmas = torch.clamp(sigmas, min=self.sigma_floor)

            self._val_preds.append(mus.detach().cpu())
            self._val_targets.append(y.detach().cpu())
            self._val_sigmas.append(sigmas.detach().cpu())

            loss, per_param = self._compute_loss(out, y, return_per_param=True)
            for k, v in per_param.items():
                self.log(f"val/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return {"val/loss": loss}

        # no-MDN
        if out.shape != y.shape:
            raise RuntimeError(f"Output-target shape mismatch: {out.shape} vs {y.shape}")
        loss = self.criterion(out, y)
        self._val_preds.append(out.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val/loss": loss}


    
    def on_test_start(self):
        # init cache + keep a ref to the raw (unwrapped) dataset for inverse transforms
        self._test_cache = {"preds": [], "targets": [], "sigmas": [], "mu": [], "L": [], "filenames": []}
        dl = self.trainer.datamodule.test_dataloader()
        ds = getattr(dl, "dataset", dl)
        while hasattr(ds, "dataset"):  # unwrap Subset/Concat etc.
            ds = ds.dataset
        self._test_dataset = ds

    def test_step(self, batch, batch_idx):
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # match validation: only augment when not "train only"
        if not self.apply_aug_train_only:
            x = self._maybe_augment_on_device(x)

        self._sanity_checks(x, y, batch_idx)

        use_amp = x.is_cuda and isinstance(getattr(self.trainer, "precision", ""), str) and "mixed" in self.trainer.precision
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x)

        # ---------------- MIXTURE (K>1) ----------------
        if isinstance(out, dict) and ("pi_logits" in out):
            if "sigma" in out:   # diagonal mixture
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
            else:                 # full-cov mixture (has 'L')
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])

            # cache (mirror val)
            self._test_preds.append(mu_mix.detach().cpu())
            self._test_targets.append(y.detach().cpu())
            self._test_sigmas = (self._test_sigmas or [])
            self._test_sigmas.append(sig_mix.detach().cpu())

            cache = getattr(self, "_test_cache", {}) or {}
            cache.setdefault("pi_logits_all", []).append(out["pi_logits"].detach().cpu())  # (B,K)
            cache.setdefault("mu_all", []).append(out["mu"].detach().cpu())                # (B,K,d)
            if "sigma" in out:
                cache.setdefault("sigma_all", []).append(out["sigma"].detach().cpu())      # (B,K,d)
            if "L" in out:
                cache.setdefault("L_all", []).append(out["L"].detach().cpu())              # (B,K,d,d)
            self._test_cache = cache

            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return {"test/loss": loss}

        # ---------------- SINGLE-GAUSSIAN FULL-COV ----------------
        if self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)

            sigma_diag = torch.sqrt((L.square()).sum(dim=-1))  # (B,d)  ← used for error bars

            self._test_preds.append(mu.detach().cpu())
            self._test_targets.append(y.detach().cpu())
            self._test_sigmas = (self._test_sigmas or [])
            self._test_sigmas.append(sigma_diag.detach().cpu())
            if not hasattr(self, "_test_Ls") or self._test_Ls is None:
                self._test_Ls = []
            self._test_Ls.append(L.detach().cpu())

            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return {"test/loss": loss}

        # ---------------- DIAGONAL (old) MDN ----------------
        if self.use_mdn:
            mus, sigmas = self._split_mdn_tuple(out, len(self.model_params))
            if not self.sigma_head_outputs_positive:
                sigmas = F.softplus(sigmas)
            sigmas = torch.clamp(sigmas, min=self.sigma_floor)

            self._test_preds.append(mus.detach().cpu())
            self._test_targets.append(y.detach().cpu())
            self._test_sigmas = (self._test_sigmas or [])
            self._test_sigmas.append(sigmas.detach().cpu())

            loss, per_param = self._compute_loss(out, y, return_per_param=True)
            for k, v in per_param.items():
                self.log(f"test/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return {"test/loss": loss}

        # ---------------- PLAIN REGRESSION ----------------
        if out.shape != y.shape:
            raise RuntimeError(f"Output-target shape mismatch: {out.shape} vs {y.shape}")
        loss = self.criterion(out, y)
        self._test_preds.append(out.detach().cpu())
        self._test_targets.append(y.detach().cpu())
        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"test/loss": loss}


    def on_validation_epoch_end(self):
        preds_all = _gather_cat_cpu(self._val_preds)
        targets_all = _gather_cat_cpu(self._val_targets)
        sigmas_all = _gather_cat_cpu(self._val_sigmas) if self.use_mdn else None
        Ls_all = _gather_cat_cpu(self._val_Ls) if (self._val_Ls is not None) else None

        cache = getattr(self, "_val_cache", {})
        for key in ["pi_logits_all", "mu_all", "sigma_all", "L_all"]:
            if key in cache and isinstance(cache[key], list):
                cache[key] = _gather_cat_cpu(cache[key])

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            self._val_cache = {
                "preds": preds_all,
                "targets": targets_all,
                "sigmas": sigmas_all,
                "Ls": Ls_all,
                "pi_logits_all": cache.get("pi_logits_all", None),
                "mu_all": cache.get("mu_all", None),
                "sigma_all": cache.get("sigma_all", None),
                "L_all": cache.get("L_all", None),
            }
        else:
            self._val_cache = {}

        self._val_preds = []
        self._val_targets = []
        self._val_sigmas = None
        self._val_Ls = None
        

    def on_test_epoch_end(self):
        preds_all   = _gather_cat_cpu(self._test_preds)
        targets_all = _gather_cat_cpu(self._test_targets)
        sigmas_all  = _gather_cat_cpu(self._test_sigmas) if self.use_mdn else None
        Ls_all      = _gather_cat_cpu(self._test_Ls) if (self._test_Ls is not None) else None

        cache = getattr(self, "_test_cache", {}) or {}
        for key in ["pi_logits_all", "mu_all", "sigma_all", "L_all"]:
            if key in cache and isinstance(cache[key], list):
                cache[key] = _gather_cat_cpu(cache[key])

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            self._test_cache = {
                "preds": preds_all,
                "targets": targets_all,
                "sigmas": sigmas_all,
                "Ls": Ls_all,
                "pi_logits_all": cache.get("pi_logits_all", None),
                "mu_all": cache.get("mu_all", None),
                "sigma_all": cache.get("sigma_all", None),
                "L_all": cache.get("L_all", None),
            }
        else:
            self._test_cache = {}

        # clear buffers
        self._test_preds = []
        self._test_targets = []
        self._test_sigmas = None
        self._test_Ls = None


    # ------------------- Optim / Scheduler -------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optim_cfg.lr,
            weight_decay=self.hparams.optim_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1
            }
        }
