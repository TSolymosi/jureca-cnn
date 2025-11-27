import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

# Try to import the corner library and provide a helpful error if it's not installed.
try:
    import corner
except ImportError:
    raise ImportError(
        "The 'corner' library is required for the HybridCornerPlotCallback. "
        "Please install it with 'pip install corner'"
    )

# --- Helper Functions (from your colleague's robust implementation) ---

def _draw_samples(mu, L=None, sigma=None, n=3000, seed=0, device="cpu", dtype=torch.float32):
    """Draws N samples from a multivariate Gaussian distribution."""
    mu = torch.as_tensor(mu, dtype=dtype, device=device).flatten()
    d = mu.numel()
    gen = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(n, d, generator=gen, device=device, dtype=dtype)
    if L is not None:
        L = torch.as_tensor(L, dtype=dtype, device=device)
        # Add small jitter to diagonal for stability before sampling
        L = L + torch.eye(d, device=device, dtype=dtype) * 1e-6
        s = mu + eps @ L.T
    elif sigma is not None:
        sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
        s = mu + eps * sigma
    else:
        raise ValueError("Either L (Cholesky factor) or sigma (diagonal stds) must be provided.")
    return s.cpu().numpy()

def _build_scale_jacobian(unscaled_mu, stds, log_idx):
    """
    Builds the Jacobian of the inverse transformation from scaled -> original space.
    This is crucial for correctly propagating covariance.
    """
    unscaled_mu = torch.as_tensor(unscaled_mu, dtype=torch.float32)
    stds = torch.as_tensor(stds, dtype=torch.float32)
    J = torch.diag(stds.clone())
    if len(log_idx) > 0:
        ln10 = math.log(10.0)
        power = torch.pow(10.0, unscaled_mu[log_idx])
        J[log_idx, log_idx] = stds[log_idx] * ln10 * power
    return J


class HybridCornerPlotCallback(Callback):
    """
    A robust callback to generate corner plots for validation samples periodically.

    This callback combines several best practices:
    - Uses a mathematically correct Jacobian transformation to propagate uncertainty
      from the model's scaled/log space back to the physical space.
    - Generates plots periodically during training (e.g., every N epochs).
    - Uses the standard `corner` library for high-quality plots.
    - Includes a dynamic range check to prevent crashes in early training.
    - Flexibly fetches prediction data from the LightningModule's cache.
    """
    def __init__(self,
                 param_names: list,
                 log_scale_params: list,
                 output_dir: str = "./corner_plots",
                 plot_every_n_epochs: int = 5,
                 num_samples_to_plot: int = 10,
                 num_posterior_samples: int = 5000,
                 seed: int = 42):
        super().__init__()
        self.param_names = list(param_names)
        self.log_scale_params = list(log_scale_params)
        self.output_dir = output_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        self.num_samples_to_plot = num_samples_to_plot
        self.num_posterior_samples = num_posterior_samples
        self.seed = int(seed)

    def _fetch_cache(self, pl_module):
        """Flexibly retrieves prediction data from the LightningModule's validation cache."""
        cache = getattr(pl_module, "_val_cache", None)
        if not isinstance(cache, dict):
            return None

        # Prioritize 'preds' and 'Ls' as these are the keys in your LightningModule
        mu = cache.get("preds", None)
        L = cache.get("Ls", None)
        targets = cache.get("targets", None)
        
        if mu is None or L is None or targets is None:
            print("[HybridCornerPlot] Could not find required 'preds', 'Ls', or 'targets' in cache. Skipping.")
            return None
            
        return {"mu": mu, "L": L, "targets": targets}

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Hook to run after every validation epoch."""
        epoch = trainer.current_epoch
        
        # Check if it's time to plot based on the configured frequency.
        if (epoch + 1) % self.plot_every_n_epochs != 0:
            return

        # --- 1. Fetch data and check for validity ---
        stuff = self._fetch_cache(pl_module)
        if stuff is None:
            return

        print(f"[HybridCornerPlot] Generating corner plots for epoch {epoch + 1}...")
        mu_scaled, L_scaled, targets_scaled = stuff["mu"], stuff["L"], stuff["targets"]
        N, D = mu_scaled.shape
        labels = self.param_names[:D]

        # --- 2. Get dataset reference for unscaling ---
        dm = getattr(trainer, "datamodule", None)
        ds = getattr(dm, "dataset_ref", None) if dm is not None else None
        if ds is None:
            print("[HybridCornerPlot] Could not get dataset_ref from datamodule. Cannot unscale. Skipping.")
            return
            
        means = getattr(ds, "scaler_means", None)
        stds = getattr(ds, "scaler_stds", None)
        log_indices = [i for i, name in enumerate(self.param_names) if name in self.log_scale_params]

        if means is None or stds is None:
            print("[HybridCornerPlot] scaler_means or scaler_stds not found in dataset. Cannot unscale. Skipping.")
            return

        # --- 3. Create plots for a subset of samples ---
        output_dir_epoch = os.path.join(self.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir_epoch, exist_ok=True)
        
        n_to_plot = min(N, self.num_samples_to_plot)

        for i in range(n_to_plot):
            mu_i_scaled = mu_scaled[i].cpu().to(torch.float32)
            L_i_scaled = L_scaled[i].cpu().to(torch.float32)

            # --- 4. Unscale predictions using the Jacobian Transformation ---
            stds_t = torch.as_tensor(stds, dtype=torch.float32)
            means_t = torch.as_tensor(means, dtype=torch.float32)
            
            # First, unscale the mean to the "pre-log" space
            mu_i_unscaled = mu_i_scaled * stds_t + means_t
            
            # Build the Jacobian and transform the Cholesky factor
            J = _build_scale_jacobian(mu_i_unscaled, stds_t, log_indices)
            L_i_physical = J @ L_i_scaled
            
            # Now, apply the final non-linear (log) transformation to the mean
            mu_i_physical = mu_i_unscaled.clone()
            if len(log_indices) > 0:
                mu_i_physical[log_indices] = torch.pow(10.0, mu_i_physical[log_indices])
            
            # Draw posterior samples in the final physical space
            samples = _draw_samples(mu_i_physical, L=L_i_physical, n=self.num_posterior_samples, seed=self.seed + i)
            
            # --- 5. Robustness Check: Dynamic Range ---
            sample_stds = np.std(samples, axis=0)
            if np.any(sample_stds < 1e-9):
                bad_params = [labels[j] for j, s in enumerate(sample_stds) if s < 1e-9]
                print(f"[HybridCornerPlot] WARNING: Epoch {epoch+1}, Sample {i}: No dynamic range for {bad_params}. Skipping plot.")
                continue

            # --- 6. Unscale the true target values for plotting ---
            target_i_unscaled = ds.inverse_transform_labels(targets_scaled[i].unsqueeze(0)).squeeze().cpu().numpy()
            
            # --- 7. Generate and Save the Plot ---
            fig = corner.corner(
                samples,
                labels=labels,
                truths=target_i_unscaled,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                truth_color="red"
            )
            fig.suptitle(f"Corner Plot for Sample {i} - Epoch {epoch + 1}", fontsize=16)

            plot_filename = os.path.join(output_dir_epoch, f"sample_{i}.png")
            fig.savefig(plot_filename, dpi=120)
            plt.close(fig)

        print(f"[HybridCornerPlot] Finished saving {n_to_plot} corner plots to {output_dir_epoch}")

