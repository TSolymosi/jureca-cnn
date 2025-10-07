import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    
    Features:
    - For MoG models, plots individual Gaussian components on diagonal histograms
    - Keeps log-scaled parameters in log10 space for consistent visualization
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
        cache = getattr(pl_module, "_val_cache", None)
        if not isinstance(cache, dict):
            return None
        targets = cache.get("targets", None)
        if targets is None:
            return None
        
        # Check for mixture model outputs
        pi_logits = cache.get("pi_logits_all", None)
        mu_all = cache.get("mu_all", None)
        L_all = cache.get("L_all", None)
        if pi_logits is not None and mu_all is not None and L_all is not None:
            return {
                "is_mixture": True,
                "pi_logits": pi_logits,
                "mu_all": mu_all,
                "L_all": L_all,
                "targets": targets
            }
        
        # Check for single Gaussian outputs
        mu = cache.get("preds", None)
        L = cache.get("Ls", None)
        if mu is not None and L is not None:
            return {
                "is_mixture": False,
                "mu": mu,
                "L": L,
                "targets": targets
            }
        return None

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_n_epochs != 0:
            return

        stuff = self._fetch_cache(pl_module)
        if stuff is None:
            print("[HybridCornerPlot] No valid data found in cache. Skipping.")
            return

        print(f"[HybridCornerPlot] Generating corner plots for epoch {epoch + 1}...")
        
        # Get dataset reference for unscaling
        ds = getattr(trainer.datamodule, "dataset_ref", None)
        if ds is None or not all(hasattr(ds, attr) for attr in ["scaler_means", "scaler_stds"]):
            print("[HybridCornerPlot] Dataset reference or scalers not found. Cannot unscale. Skipping.")
            return

        means_t = torch.as_tensor(ds.scaler_means, dtype=torch.float32)
        stds_t = torch.as_tensor(ds.scaler_stds, dtype=torch.float32)
        
        # Determine log indices and create plot labels
        log_indices = [i for i, name in enumerate(self.param_names) if name in self.log_scale_params]
        plot_labels = []
        for i, name in enumerate(self.param_names):
            if i in log_indices:
                plot_labels.append(f"log10({name})")
            else:
                plot_labels.append(name)
        
        output_dir_epoch = os.path.join(self.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir_epoch, exist_ok=True)
        
        targets_scaled = stuff["targets"]
        n_to_plot = min(targets_scaled.shape[0], self.num_samples_to_plot)

        for i in range(n_to_plot):
            if stuff["is_mixture"]:
                final_samples, component_samples, weights = self._process_mixture(
                    stuff, i, means_t, stds_t, log_indices
                )
            else:
                final_samples = self._process_single_gaussian(
                    stuff, i, means_t, stds_t, log_indices
                )
                component_samples = None
                weights = None
            
            if final_samples is None:
                continue
            
            # Unscale target (keep in log space for log parameters)
            target_i_unscaled = (targets_scaled[i] * stds_t + means_t).cpu().numpy()
            
            # Generate corner plot
            fig = corner.corner(
                final_samples,
                labels=plot_labels,
                truths=target_i_unscaled,
                show_titles=True,
                quantiles=[0.16, 0.5, 0.84],
                title_fmt='.2f'
            )
            
            # Overlay individual components on diagonal
            if component_samples is not None and weights is not None:
                self._overlay_components(fig, component_samples, weights, len(self.param_names))
            
            fig.suptitle(f"Corner Plot for Sample {i} - Epoch {epoch + 1}", fontsize=16, y=1.0)
            plot_filename = os.path.join(output_dir_epoch, f"sample_{i}.png")
            fig.savefig(plot_filename, dpi=120, bbox_inches='tight')
            plt.close(fig)

        print(f"[HybridCornerPlot] Finished saving {n_to_plot} corner plots to {output_dir_epoch}")

    def _process_mixture(self, stuff, i, means_t, stds_t, log_indices):
        """Process mixture of Gaussians prediction."""
        pi_logits_i = stuff["pi_logits"][i].cpu().to(torch.float32)
        mu_all_i_scaled = stuff["mu_all"][i].cpu().to(torch.float32)
        L_all_i_scaled = stuff["L_all"][i].cpu().to(torch.float32)
        K = pi_logits_i.shape[0]

        # Calculate mixing weights
        pi_i = torch.softmax(pi_logits_i, dim=-1)
        
        # Distribute samples among components
        counts = (pi_i * self.num_posterior_samples).round().int()
        counts_sum_diff = self.num_posterior_samples - counts.sum()
        counts[torch.argmax(pi_i)] += counts_sum_diff

        mixture_samples_list = []
        component_samples_list = []
        active_weights = []

        for k in range(K):
            if counts[k] == 0:
                continue

            mu_k_scaled = mu_all_i_scaled[k]
            L_k_scaled = L_all_i_scaled[k]
            
            # Unscale mean to pre-log space (NOT transformed to linear)
            mu_k_unscaled = mu_k_scaled * stds_t + means_t
            
            # Build Jacobian for covariance transformation
            J_k = _build_scale_jacobian(mu_k_unscaled, stds_t, log_indices)
            L_k_unscaled = J_k @ L_k_scaled
            
            # Draw samples in the unscaled (log) space
            samples_k = _draw_samples(
                mu_k_unscaled,
                L=L_k_unscaled,
                n=counts[k].item(),
                seed=self.seed + i * K + k
            )
            
            mixture_samples_list.append(samples_k)
            component_samples_list.append(samples_k)
            active_weights.append(pi_i[k].item())
        
        if not mixture_samples_list:
            return None, None, None
        
        final_samples = np.concatenate(mixture_samples_list, axis=0)
        
        return final_samples, component_samples_list, active_weights

    def _process_single_gaussian(self, stuff, i, means_t, stds_t, log_indices):
        """Process single Gaussian prediction."""
        mu_i_scaled = stuff["mu"][i].cpu().to(torch.float32)
        L_i_scaled = stuff["L"][i].cpu().to(torch.float32)
        
        # Unscale to pre-log space
        mu_i_unscaled = mu_i_scaled * stds_t + means_t
        
        # Build Jacobian
        J_i = _build_scale_jacobian(mu_i_unscaled, stds_t, log_indices)
        L_i_unscaled = J_i @ L_i_scaled
        
        # Draw samples
        samples = _draw_samples(
            mu_i_unscaled,
            L=L_i_unscaled,
            n=self.num_posterior_samples,
            seed=self.seed + i
        )
        
        return samples

    def _overlay_components(self, fig, component_samples, weights, n_params):
        """Overlay individual Gaussian components on diagonal histograms."""
        # Get diagonal axes
        diag_axes = [fig.axes[j * n_params + j] for j in range(n_params)]
        
        # Use distinct colors for each component
        colors = plt.cm.tab10(np.linspace(0, 1, len(component_samples)))
        
        for k, (samples, weight) in enumerate(zip(component_samples, weights)):
            for param_idx in range(n_params):
                ax = diag_axes[param_idx]
                
                # Get current x-limits from the corner plot
                xlim = ax.get_xlim()
                
                # Create histogram for this component
                counts, bins = np.histogram(
                    samples[:, param_idx],
                    bins=30,
                    range=xlim,
                    density=True
                )
                
                # Scale by component weight
                counts_scaled = counts * weight
                
                # Plot as a filled curve
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.fill_between(
                    bin_centers,
                    0,
                    counts_scaled,
                    alpha=0.3,
                    color=colors[k],
                    label=f'Component {k+1} ({weight:.2%})',
                    step='mid'
                )
        
        # Add legend to the last diagonal plot
        if len(component_samples) > 1:
            diag_axes[-1].legend(fontsize=8, loc='upper right')