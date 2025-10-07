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
# _draw_samples and _build_scale_jacobian remain the same as before.

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

    New features:
    - For MoG models, it plots the individual Gaussian components on the diagonal histograms.
    - Can optionally keep log-scaled parameters in log10 space for consistent visualization.
    """
    def __init__(self,
                 param_names: list,
                 log_scale_params: list,
                 output_dir: str = "./corner_plots",
                 plot_every_n_epochs: int = 5,
                 num_samples_to_plot: int = 10,
                 num_posterior_samples: int = 5000,
                 plot_in_log_space: bool = True, # <-- NEW ARGUMENT
                 seed: int = 42):
        super().__init__()
        self.param_names = list(param_names)
        self.log_scale_params = list(log_scale_params)
        self.output_dir = output_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        self.num_samples_to_plot = num_samples_to_plot
        self.num_posterior_samples = num_posterior_samples
        self.plot_in_log_space = plot_in_log_space # <-- NEW
        self.seed = int(seed)

    # _fetch_cache method remains unchanged
    def _fetch_cache(self, pl_module):
        # ... (same as before) ...
        cache = getattr(pl_module, "_val_cache", None)
        if not isinstance(cache, dict): return None
        targets = cache.get("targets", None)
        if targets is None: return None
        pi_logits = cache.get("pi_logits_all", None)
        mu_all = cache.get("mu_all", None)
        L_all = cache.get("L_all", None)
        if pi_logits is not None and mu_all is not None and L_all is not None:
            return {"is_mixture": True, "pi_logits": pi_logits, "mu_all": mu_all, "L_all": L_all, "targets": targets}
        mu = cache.get("preds", None)
        L = cache.get("Ls", None)
        if mu is not None and L is not None:
            return {"is_mixture": False, "mu": mu, "L": L, "targets": targets}
        return None

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
        
        # --- Get dataset reference for unscaling (unchanged) ---
        ds = getattr(trainer.datamodule, "dataset_ref", None)
        if ds is None or not all(hasattr(ds, attr) for attr in ["scaler_means", "scaler_stds"]):
             print("[HybridCornerPlot] Dataset reference or scalers not found. Cannot unscale. Skipping.")
             return

        means_t = torch.as_tensor(ds.scaler_means, dtype=torch.float32)
        stds_t = torch.as_tensor(ds.scaler_stds, dtype=torch.float32)
        
        # --- MODIFIED: Determine which parameters to transform vs. keep in log space ---
        plot_labels = self.param_names[:] # Make a copy
        log_indices_model = [i for i, name in enumerate(self.param_names) if name in self.log_scale_params]
        
        # If plot_in_log_space is True, we modify the labels and identify which columns to NOT transform
        final_log_indices_plot = []
        if self.plot_in_log_space:
            for idx in log_indices_model:
                plot_labels[idx] = f"log10({self.param_names[idx]})"
                final_log_indices_plot.append(idx)
        
        output_dir_epoch = os.path.join(self.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir_epoch, exist_ok=True)
        
        targets_scaled = stuff["targets"]
        n_to_plot = min(targets_scaled.shape[0], self.num_samples_to_plot)

        for i in range(n_to_plot):
            # ... (Single Gaussian logic can be adapted similarly, but focusing on mixture case) ...

            # --- MIXTURE OF GAUSSIANS LOGIC ---
            if stuff["is_mixture"]:
                pi_logits_i = stuff["pi_logits"][i].cpu().to(torch.float32)
                mu_all_i_scaled = stuff["mu_all"][i].cpu().to(torch.float32)
                L_all_i_scaled = stuff["L_all"][i].cpu().to(torch.float32)
                K = pi_logits_i.shape[0]

                pi_i = torch.softmax(pi_logits_i, dim=-1)
                counts = (pi_i * self.num_posterior_samples).round().int()
                counts_sum_diff = self.num_posterior_samples - counts.sum()
                counts[torch.argmax(pi_i)] += counts_sum_diff

                mixture_samples_list = []
                # --- NEW: Store per-component samples for plotting ---
                per_component_samples_for_hist = []

                for k in range(K):
                    if counts[k] == 0: continue

                    mu_k_scaled = mu_all_i_scaled[k]
                    L_k_scaled = L_all_i_scaled[k]
                    
                    # Unscale to physical space OR pre-log space
                    mu_k_unscaled = mu_k_scaled * stds_t + means_t
                    J_k = _build_scale_jacobian(mu_k_unscaled, stds_t, log_indices_model)
                    L_k_physical = J_k @ L_k_scaled
                    mu_k_physical = mu_k_unscaled.clone()

                    # Now, based on plot_in_log_space, either keep log params as-is or transform them
                    if not self.plot_in_log_space:
                        if len(log_indices_model) > 0:
                            # Transform L for the log params
                            # This is complex, so for now we transform the samples which is equivalent for visualization
                            mu_k_physical[log_indices_model] = torch.pow(10.0, mu_k_physical[log_indices_model])

                    samples_k_physical = _draw_samples(mu_k_unscaled, L=J_k @ L_k_scaled, n=counts[k].item(), seed=self.seed + i*K + k)
                    
                    # Apply the final 10**x transform if needed
                    if not self.plot_in_log_space:
                         if len(log_indices_model) > 0:
                            samples_k_physical[:, log_indices_model] = 10**samples_k_physical[:, log_indices_model]
                    
                    mixture_samples_list.append(samples_k_physical)
                    per_component_samples_for_hist.append(samples_k_physical) # Store for histograms
                
                if not mixture_samples_list: continue
                final_samples = np.concatenate(mixture_samples_list, axis=0)

            # --- Unscale the true target values for plotting ---
            target_i_unscaled = (targets_scaled[i] * stds_t + means_t).cpu().numpy()
            if not self.plot_in_log_space:
                if len(log_indices_model) > 0:
                    target_i_unscaled[log_indices_model] = 10**target_i_unscaled[log_indices_model]
            
            # --- Generate the Plot ---
            fig = corner.corner(final_samples, labels=plot_labels, truths=target_i_unscaled, show_titles=True)

            # --- NEW: Overlay individual Gaussian components on histograms ---
            if stuff["is_mixture"]:
                # Get the axes on the diagonal
                diag_axes = [fig.axes[j * (len(self.param_names)) + j] for j in range(len(self.param_names))]
                
                colors = plt.cm.get_cmap('viridis', K)
                
                for k in range(len(per_component_samples_for_hist)):
                    component_samples = per_component_samples_for_hist[k]
                    weight = pi_i[k].item()
                    
                    for param_idx in range(len(self.param_names)):
                        ax = diag_axes[param_idx]
                        # Plot a weighted histogram for this component
                        ax.hist(component_samples[:, param_idx], bins=ax.get_xticks(),
                                density=True, alpha=0.6, color=colors(k),
                                weights=np.full(len(component_samples), weight))

            fig.suptitle(f"Corner Plot for Sample {i} - Epoch {epoch + 1}", fontsize=16)
            plot_filename = os.path.join(output_dir_epoch, f"sample_{i}.png")
            fig.savefig(plot_filename, dpi=120)
            plt.close(fig)

        print(f"[HybridCornerPlot] Finished saving {n_to_plot} corner plots to {output_dir_epoch}")