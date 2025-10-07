import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

try:
    import corner
except ImportError:
    raise ImportError(
        "The 'corner' library is required for the HybridCornerPlotCallback. "
        "Please install it with 'pip install corner'"
    )

# --- Helper Functions (Unchanged) ---

def _draw_samples(mu, L=None, n=3000, seed=0, device="cpu", dtype=torch.float32):
    """Draws N samples from a single multivariate Gaussian distribution."""
    mu = torch.as_tensor(mu, dtype=dtype, device=device).flatten()
    d = mu.numel()
    gen = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(n, d, generator=gen, device=device, dtype=dtype)
    L = torch.as_tensor(L, dtype=dtype, device=device)
    L = L + torch.eye(d, device=device, dtype=dtype) * 1e-6
    s = mu + eps @ L.T
    return s.cpu().numpy()

def _build_scale_jacobian(unscaled_mu, stds, log_idx):
    """Builds the Jacobian of the inverse transformation from scaled -> original space."""
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
    A robust callback to generate corner plots for validation samples,
    with automatic support for both single-Gaussian and Mixture of Gaussians (MoG) models.
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
        """Flexibly retrieves data from the cache, detecting if it's a mixture or single Gaussian."""
        cache = getattr(pl_module, "_val_cache", None)
        if not isinstance(cache, dict): return None

        targets = cache.get("targets", None)
        if targets is None: return None

        # --- NEW: Check for mixture data first ---
        pi_logits = cache.get("pi_logits_all", None)
        mu_all = cache.get("mu_all", None)
        L_all = cache.get("L_all", None)

        if pi_logits is not None and mu_all is not None and L_all is not None:
            return {
                "is_mixture": True,
                "pi_logits": pi_logits,
                "mu_all": mu_all,
                "L_all": L_all,
                "targets": targets,
            }

        # --- Fallback to single-Gaussian data ---
        mu = cache.get("preds", None)
        L = cache.get("Ls", None)
        if mu is not None and L is not None:
            return {"is_mixture": False, "mu": mu, "L": L, "targets": targets}

        return None # No valid data found

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_n_epochs != 0:
            return

        stuff = self._fetch_cache(pl_module)
        if stuff is None:
            print("[HybridCornerPlot] No valid single-Gaussian or mixture data found in cache. Skipping.")
            return

        print(f"[HybridCornerPlot] Generating corner plots for epoch {epoch + 1}...")
        
        # --- Get dataset reference for unscaling ---
        dm = getattr(trainer, "datamodule", None)
        ds = getattr(dm, "dataset_ref", None)
        if ds is None or not all(hasattr(ds, attr) for attr in ["scaler_means", "scaler_stds"]):
             print("[HybridCornerPlot] Dataset reference or scalers not found. Cannot unscale. Skipping.")
             return

        means_t = torch.as_tensor(ds.scaler_means, dtype=torch.float32)
        stds_t = torch.as_tensor(ds.scaler_stds, dtype=torch.float32)
        log_indices = [i for i, name in enumerate(self.param_names) if name in self.log_scale_params]
        
        # --- Create plots for a subset of samples ---
        output_dir_epoch = os.path.join(self.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir_epoch, exist_ok=True)
        
        targets_scaled = stuff["targets"]
        n_to_plot = min(targets_scaled.shape[0], self.num_samples_to_plot)

        for i in range(n_to_plot):
            final_samples = None # This will hold the samples for the corner plot

            # --- SINGLE GAUSSIAN LOGIC ---
            if not stuff["is_mixture"]:
                mu_i_scaled = stuff["mu"][i].cpu().to(torch.float32)
                L_i_scaled = stuff["L"][i].cpu().to(torch.float32)
                
                mu_i_unscaled = mu_i_scaled * stds_t + means_t
                J = _build_scale_jacobian(mu_i_unscaled, stds_t, log_indices)
                L_i_physical = J @ L_i_scaled
                
                mu_i_physical = mu_i_unscaled.clone()
                if len(log_indices) > 0:
                    mu_i_physical[log_indices] = torch.pow(10.0, mu_i_physical[log_indices])
                
                final_samples = _draw_samples(mu_i_physical, L=L_i_physical, n=self.num_posterior_samples, seed=self.seed + i)

            # --- MIXTURE OF GAUSSIANS LOGIC ---
            else:
                pi_logits_i = stuff["pi_logits"][i].cpu().to(torch.float32)
                mu_all_i = stuff["mu_all"][i].cpu().to(torch.float32) # Shape: [K, D]
                L_all_i = stuff["L_all"][i].cpu().to(torch.float32)   # Shape: [K, D, D]
                K = pi_logits_i.shape[0]

                # Determine how many samples to draw from each component
                pi_i = torch.softmax(pi_logits_i, dim=-1)
                counts = (pi_i * self.num_posterior_samples).round().int()
                # Adjust for rounding errors to ensure the total is correct
                counts_sum_diff = self.num_posterior_samples - counts.sum()
                counts[torch.argmax(pi_i)] += counts_sum_diff

                mixture_samples_list = []
                for k in range(K):
                    if counts[k] == 0: continue

                    # Unscale each component individually
                    mu_k_scaled = mu_all_i[k]
                    L_k_scaled = L_all_i[k]
                    
                    mu_k_unscaled = mu_k_scaled * stds_t + means_t
                    J_k = _build_scale_jacobian(mu_k_unscaled, stds_t, log_indices)
                    L_k_physical = J_k @ L_k_scaled

                    mu_k_physical = mu_k_unscaled.clone()
                    if len(log_indices) > 0:
                        mu_k_physical[log_indices] = torch.pow(10.0, mu_k_physical[log_indices])
                    
                    # Draw samples for this component
                    samples_k = _draw_samples(mu_k_physical, L=L_k_physical, n=counts[k].item(), seed=self.seed + i*K + k)
                    mixture_samples_list.append(samples_k)
                
                if not mixture_samples_list: continue # Skip if no samples were drawn
                final_samples = np.concatenate(mixture_samples_list, axis=0)

            if final_samples is None: continue

            # --- Robustness Check & Plotting (Same for both cases) ---
            sample_stds = np.std(final_samples, axis=0)
            if np.any(sample_stds < 1e-9):
                bad_params = [self.param_names[j] for j, s in enumerate(sample_stds) if s < 1e-9]
                print(f"[HybridCornerPlot] WARNING: Epoch {epoch+1}, Sample {i}: No dynamic range for {bad_params}. Skipping plot.")
                continue

            target_i_physical = ds.inverse_transform_labels(targets_scaled[i].unsqueeze(0)).squeeze().cpu().numpy()
            
            fig = corner.corner(
                final_samples,
                labels=self.param_names,
                truths=target_i_physical,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                truth_color="red"
            )
            fig.suptitle(f"Corner Plot for Sample {i} - Epoch {epoch + 1}", fontsize=16)

            plot_filename = os.path.join(output_dir_epoch, f"sample_{i}.png")
            fig.savefig(plot_filename, dpi=120)
            plt.close(fig)

        print(f"[HybridCornerPlot] Finished saving plots for epoch {epoch + 1} to {output_dir_epoch}")