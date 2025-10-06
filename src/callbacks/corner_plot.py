import os, math, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from pathlib import Path

# --- NEW: use the 'corner' library like setup #2 ---
try:
    import corner
except ImportError as e:
    raise ImportError("The 'corner' library is required. Install with `pip install corner`.") from e


def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _draw_samples(mu, L=None, sigma=None, n=3000, seed=0, device="cpu", dtype=torch.float32):
    mu = torch.as_tensor(mu, dtype=dtype, device=device).flatten()
    d = mu.numel()
    gen = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(n, d, generator=gen, device=device, dtype=dtype)
    if L is not None:
        L = torch.as_tensor(L, dtype=dtype, device=device)
        s = mu + eps @ L.T
    elif sigma is not None:
        sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
        s = mu + eps * sigma
    else:
        raise ValueError("Need L or sigma.")
    return s.detach().cpu().numpy()

def _build_scale_jacobian(unscaled_mu, stds, log_idx):
    """
    Local linearization to map covariance from standardized->original units, including log10 dims.

    For non-log dims: x = μ + σ z  => J_ii = std_i
    For   log10 dims: y = 10^x, x = μ_unscaled + ...  => dy ≈ (ln10 * 10^μ_unscaled) dx
                      so J_ii = std_i * ln(10) * 10^(μ_unscaled)
    """
    unscaled_mu = torch.as_tensor(unscaled_mu, dtype=torch.float32)
    stds = torch.as_tensor(stds, dtype=torch.float32)
    J = torch.diag(stds.clone())
    if len(log_idx) > 0:
        ln10 = math.log(10.0)
        power = torch.pow(10.0, unscaled_mu[log_idx])
        J[log_idx, log_idx] = stds[log_idx] * ln10 * power
    return J


class CornerPlotPerSampleCallback(Callback):
    """
    Corner-plot callback that:
      • reads the same cache keys as before (mu, L/sigma, filenames; targets optional)
      • transforms to ORIGINAL units (with correct Jacobian for log10 dims)
      • draws corner plots with 'corner', showing truths in red, and 16/50/84% quantiles
      • runs every N epochs and saves files like:  <output_dir>/epoch_{E}_sample_{i}.png

    Expected pl_module caches in _val_cache / _test_cache:
      'mu': (N,D) ; 'L': (N,D,D) OR 'sigma': (N,D) ; 'targets' (N,D) optional ; 'filenames' optional
      (Also accepts 'preds'/'Ls'/'sigmas' fallbacks or single-mixture 'mu_all'/'L_all')
    """
    def __init__(
        self,
        param_names,
        output_dir="./corner_plots",
        plot_every_n_epochs=5,
        num_samples_to_plot=10,
        num_posterior_samples=5000,
        min_rel_span=1e-3,      # require ≥0.1% relative spread
        min_abs_span=1e-13,      # require ≥1e-13 absolute spread
        stage=("val","test"),           # keep stage selection
        seed=0
    ):
        super().__init__()
        self.param_names = list(param_names)
        #outputdir is outputdir + cornerplot folder
        self.output_dir = Path(output_dir) / "corner_plots"
        self.plot_every_n_epochs = int(plot_every_n_epochs)
        self.num_samples_to_plot = int(num_samples_to_plot)
        self.num_posterior_samples = int(num_posterior_samples)
        self.stage = tuple(stage) if isinstance(stage, (list, tuple)) else (stage,)
        self.seed = int(seed)
        self.min_rel_span = float(min_rel_span)
        self.min_abs_span = float(min_abs_span)

    # ---------------- internal: cache plumbing (kept compatible) ----------------
    def _fetch_cache(self, pl_module, which):
        print(f"[CornerPlot] _fetch_cache: stage={which}", flush=True)
        key = "_val_cache" if which == "val" else "_test_cache"
        cache = getattr(pl_module, key, None)
        if not isinstance(cache, dict):
            return None

        def _stack(v):
            if v is None: return None
            if isinstance(v, list):
                v = [t.detach().cpu() if torch.is_tensor(t) else torch.as_tensor(t) for t in v]
                return torch.cat(v, dim=0)
            return v

        mu    = _stack(cache.get("mu", None)) or _stack(cache.get("preds", None))      # (N,D)
        L     = _stack(cache.get("L", None))  or _stack(cache.get("Ls", None))         # (N,D,D)
        sigma = _stack(cache.get("sigma", None)) or _stack(cache.get("sigmas", None))  # (N,D)
        y     = _stack(cache.get("targets", None))                                      # (N,D) optional

        # Single-mixture fallback (K=1)
        if mu is None and cache.get("mu_all", None) is not None:
            mu_all = _stack(cache["mu_all"])  # (N,K,D)
            if mu_all is not None and mu_all.ndim == 3 and mu_all.shape[1] == 1:
                mu = mu_all[:, 0, :]
        if L is None and cache.get("L_all", None) is not None:
            L_all = _stack(cache["L_all"])    # (N,K,D,D)
            if L_all is not None and L_all.ndim == 4 and L_all.shape[1] == 1:
                L = L_all[:, 0, :, :]

        if mu is None or (L is None and sigma is None):
            return None

        return {
            "mu": mu, "L": L, "sigma": sigma, "targets": y,
            "filenames": cache.get("filenames", None)
        }

    # ---------------- internal: one stage run ----------------
    @rank_zero_only
    def _run_stage(self, trainer, pl_module, which):
        print(f"[CornerPlot] _run_stage: stage={which}", flush=True)
        stuff = self._fetch_cache(pl_module, which)
        if stuff is None:
            print(f"[CornerPlot] No cache for {which}. Skipping.", flush=True); return

        mu, L, sigma, y_scaled, filenames = stuff["mu"], stuff["L"], stuff["sigma"], stuff["targets"], stuff["filenames"]
        N, D = mu.shape
        labels = self.param_names[:D]

        # Dataset reference for scalers + which params are in log10 space
        dm = getattr(trainer, "datamodule", None)
        ds = getattr(dm, "dataset_ref", None) if dm is not None else None
        if ds is None:
            print("[CornerPlot] No dataset_ref; cannot inverse-transform. Skipping.", flush=True)
            return
        means = getattr(ds, "scaler_means", None)
        stds  = getattr(ds, "scaler_stds", None)
        log_idx = getattr(ds, "param_indices_to_log", [])

        if means is None or stds is None:
            print("[CornerPlot] Missing scaler_means/stds; cannot inverse-transform. Skipping.", flush=True)
            return

        # Output directory
        _ensure_dir(self.output_dir)

        # How many samples to plot
        n_to_plot = min(self.num_samples_to_plot, N)
        print(f"[CornerPlot] {which}: N={N}, D={D}, n_to_plot={n_to_plot}, draws={self.num_posterior_samples}", flush=True)

        means32 = torch.as_tensor(means, dtype=torch.float32)
        stds32  = torch.as_tensor(stds,  dtype=torch.float32)

        for i in range(n_to_plot):
            # base filename (prefer original file name if present)
            base = None
            if isinstance(filenames, (list, tuple)) and i < len(filenames):
                base = os.path.splitext(os.path.basename(str(filenames[i])))[0]
            if base is None:
                base = f"sample_{i:05d}"

            # ---- μ/L/σ in ORIGINAL units (with proper Jacobian for log dims) ----
            mu_i = mu[i].detach().cpu().to(torch.float32)

            if L is not None:
                L_i = L[i].detach().cpu().to(torch.float32)
                unscaled_mu = mu_i * stds32 + means32
                J = _build_scale_jacobian(unscaled_mu, stds32, log_idx).to(torch.float32)
                L_orig = J @ L_i
                mu_orig = unscaled_mu.clone()
                if len(log_idx) > 0:
                    mu_orig[log_idx] = torch.pow(10.0, mu_orig[log_idx])
                samples = _draw_samples(mu_orig, L=L_orig, n=self.num_posterior_samples, seed=self.seed + i)

            else:
                sigma_i = sigma[i].detach().cpu().to(torch.float32)
                unscaled_mu = mu_i * stds32 + means32
                J = _build_scale_jacobian(unscaled_mu, stds32, log_idx).to(torch.float32)
                # Promote diagonal σ into a Cholesky via J @ diag(σ_scaled)
                L_diag = torch.diag(sigma_i)
                L_orig = J @ L_diag
                mu_orig = unscaled_mu.clone()
                if len(log_idx) > 0:
                    mu_orig[log_idx] = torch.pow(10.0, mu_orig[log_idx])
                samples = _draw_samples(mu_orig, L=L_orig, n=self.num_posterior_samples, seed=self.seed + i)

            # # Skip if there’s no dynamic range (matches setup #2 guard)
            # stds_np = np.std(samples, axis=0)
            # if np.any(stds_np < 1e-9):
            #     bad = [labels[j] for j, s in enumerate(stds_np) if s < 1e-9]
            #     print(f"[CornerPlot] WARNING: Sample {i}: no dynamic range for {bad}. Skipping.", flush=True)
            #     continue

            # Drop dims with too little dynamic range
            s = np.asarray(samples)
            keep_idx, drop_names = [], []
            for j, name in enumerate(labels):
                col = s[:, j]
                span = float(np.ptp(col))
                ref = float(abs(np.median(col)))
                rel = span / max(ref, 1e-13)
                if (span >= self.min_abs_span) or (rel >= self.min_rel_span):
                    keep_idx.append(j)
                else:
                    drop_names.append(name)
                
            if drop_names:
                print(f"[CornerPlot] Sample {i}: dropping {len(drop_names)} params with too little range: {drop_names}", flush=True)
            if not keep_idx:
                # nothing meaningful to plot
                print(f"[CornerPlot] WARNING: Sample {i}: no params left after dropping. Skipping.", flush=True)
                continue
            # keep only non-flat dims
            samples = s[:, keep_idx]
            labels  = [labels[j] for j in keep_idx]
            
            # ---- Truths in ORIGINAL units (if targets present) ----
            truths_np = None
            if y_scaled is not None and i < y_scaled.shape[0]:
                # ds.inverse_transform_labels handles both stds and log10->linear for us
                y_i_scaled = y_scaled[i].detach().cpu()
                y_i_orig   = ds.inverse_transform_labels(y_i_scaled).detach().cpu().numpy()
                truths_np  = y_i_orig

            # ---- Corner plot (match setup #2 styling) ----
            fig = corner.corner(
                samples,
                labels=labels,
                truths=truths_np,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                truth_color="red"
            )
            epoch = getattr(trainer, "current_epoch", 0)
            fig.suptitle(f"Corner Plot for {base} — Epoch {epoch + 1}", fontsize=16)

            out_path = os.path.join(self.output_dir, f"epoch_{epoch + 1}_sample_{i}.png")
            try:
                fig.savefig(out_path, dpi=120)
                plt.close(fig)
                print(f"[CornerPlot] saved: {out_path}", flush=True)
            except Exception as e:
                print(f"[CornerPlot] Failed to save {out_path}: {e}", flush=True)

    # ---------------- Lightning hooks ----------------
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # match setup #2: run every N epochs, skip sanity checking
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_n_epochs != 0:
            return
        if "val" in self.stage:
            self._run_stage(trainer, pl_module, "val")

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        if "test" in self.stage:
            print(f"[CornerPlot] on_test_end", flush=True)
            self._run_stage(trainer, pl_module, "test")
