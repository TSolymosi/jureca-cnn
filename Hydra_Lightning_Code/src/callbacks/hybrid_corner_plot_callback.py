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


def _smart_format(value, param_name, log_scale_params):
    """
    Smart formatting based on parameter type and value magnitude.
    
    Returns a string formatted appropriately for the parameter:
    - Log-scale params in log space: Use regular decimals (e.g., "10.23")
    - Linear params: Use appropriate precision based on magnitude
    """
    if param_name in log_scale_params:
        # These are log10(X), so values typically in range [-2, 12]
        # Use 2 decimal places
        return f"{value:.2f}"
    else:
        # Linear space parameters
        abs_val = abs(value)
        
        if abs_val == 0:
            return "0.00"
        elif abs_val >= 1000:
            # Large values: no decimals
            return f"{value:.0f}"
        elif abs_val >= 100:
            # Hundreds: 1 decimal
            return f"{value:.1f}"
        elif abs_val >= 1:
            # Between 1 and 100: 2 decimals
            return f"{value:.2f}"
        elif abs_val >= 0.01:
            # Small but reasonable: 3 decimals
            return f"{value:.3f}"
        else:
            # Very small: use scientific notation
            return f"{value:.2e}"


class HybridCornerPlotCallback(Callback):
    """
    A callback to generate corner plots for validation samples.
    
    Can display either:
    - Scaled space (mean ≈ 0, std ≈ 1) - good for checking model internals
    - Physical space (original units) - good for scientific interpretation
    
    Features:
    - Plots mixture components with different colors on diagonal histograms
    - Works for both single Gaussian and Mixture of Gaussians models
    - Smart formatting: normal notation instead of scientific notation for readability
    """
    def __init__(self,
                 param_names: list,
                 log_scale_params: list,
                 output_dir: str = "./corner_plots",
                 plot_every_n_epochs: int = 5,
                 num_samples_to_plot: int = 10,
                 num_posterior_samples: int = 5000,
                 plot_in_physical_space: bool = True,
                 seed: int = 42):
        super().__init__()
        self.param_names = list(param_names)
        self.log_scale_params = list(log_scale_params)
        self.output_dir = output_dir
        self.plot_every_n_epochs = plot_every_n_epochs
        self.num_samples_to_plot = num_samples_to_plot
        self.num_posterior_samples = num_posterior_samples
        self.plot_in_physical_space = plot_in_physical_space
        self.seed = int(seed)

    def _fetch_cache(self, pl_module):
        """Fetch cached predictions and targets from the Lightning module."""
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
        """Main callback function - called after each validation epoch."""
        epoch = trainer.current_epoch
        if (epoch + 1) % self.plot_every_n_epochs != 0:
            return

        stuff = self._fetch_cache(pl_module)
        if stuff is None:
            print("[HybridCornerPlot] No valid data found in cache. Skipping.")
            return

        print(f"[HybridCornerPlot] Generating corner plots for epoch {epoch + 1}...")
        
        # Get dataset reference for scaling parameters
        ds = getattr(trainer.datamodule, "dataset_ref", None)
        if ds is None or not all(hasattr(ds, attr) for attr in ["scaler_means", "scaler_stds"]):
            print("[HybridCornerPlot] Dataset reference or scalers not found. Cannot transform. Skipping.")
            return

        means_t = torch.as_tensor(ds.scaler_means, dtype=torch.float32)
        stds_t = torch.as_tensor(ds.scaler_stds, dtype=torch.float32)
        log_indices = [i for i, name in enumerate(self.param_names) if name in self.log_scale_params]
        
        # Create labels based on plotting mode
        if self.plot_in_physical_space:
            plot_labels = []
            for i, name in enumerate(self.param_names):
                if i in log_indices:
                    plot_labels.append(f"log$_{{10}}$({name})")  # Proper LaTeX formatting
                else:
                    plot_labels.append(name)
        else:
            # Scaled space labels
            plot_labels = []
            for name in self.param_names:
                if name in self.log_scale_params:
                    plot_labels.append(f"{name} (scaled log)")
                else:
                    plot_labels.append(f"{name} (scaled)")
        
        output_dir_epoch = os.path.join(self.output_dir, f"epoch_{epoch+1}")
        os.makedirs(output_dir_epoch, exist_ok=True)
        
        targets_scaled = stuff["targets"]
        n_to_plot = min(targets_scaled.shape[0], self.num_samples_to_plot)

        for i in range(n_to_plot):
            if stuff["is_mixture"]:
                final_samples, component_samples, weights = self._process_mixture_scaled(
                    stuff, i
                )
            else:
                final_samples = self._process_single_gaussian_scaled(
                    stuff, i
                )
                component_samples = None
                weights = None
            
            if final_samples is None:
                continue
            
            # Transform samples and targets to physical space if requested
            if self.plot_in_physical_space:
                final_samples = self._transform_to_physical(final_samples, means_t, stds_t, log_indices)
                if component_samples is not None:
                    component_samples = [
                        self._transform_to_physical(comp, means_t, stds_t, log_indices)
                        for comp in component_samples
                    ]
                target_i = self._transform_to_physical(
                    targets_scaled[i].cpu().numpy().reshape(1, -1), 
                    means_t, stds_t, log_indices
                )[0]
            else:
                target_i = targets_scaled[i].cpu().numpy()
            
            # Debugging output for first sample
            if i == 0:
                space_name = "PHYSICAL SPACE" if self.plot_in_physical_space else "SCALED SPACE"
                print(f"\n[DEBUG] Sample {i} ({space_name}):")
                print(f"  Target (ground truth):")
                for j, (label, val) in enumerate(zip(plot_labels, target_i)):
                    formatted = _smart_format(val, self.param_names[j], self.log_scale_params)
                    print(f"    {label}: {formatted}")
                
                print(f"  Posterior statistics:")
                for j, label in enumerate(plot_labels):
                    p16, p50, p84 = np.percentile(final_samples[:, j], [16, 50, 84])
                    p50_fmt = _smart_format(p50, self.param_names[j], self.log_scale_params)
                    upper_fmt = _smart_format(p84 - p50, self.param_names[j], self.log_scale_params)
                    lower_fmt = _smart_format(p50 - p16, self.param_names[j], self.log_scale_params)
                    print(f"    {label}: {p50_fmt} +{upper_fmt} -{lower_fmt}")
            
            # Create custom title formatting function
            def title_fmt_func(param_idx):
                """Returns a formatting function for a specific parameter."""
                param_name = self.param_names[param_idx]
                def fmt(value):
                    return _smart_format(value, param_name, self.log_scale_params)
                return fmt
            
            # Generate corner plot with custom formatting
            # Corner doesn't support per-parameter formatting directly,
            # so we need to create a custom titles list
            
            # Calculate quantiles for titles
            quantiles = [0.16, 0.5, 0.84]
            titles = []
            for j in range(len(self.param_names)):
                q_values = np.percentile(final_samples[:, j], [16, 50, 84])
                median = q_values[1]
                upper = q_values[2] - median
                lower = median - q_values[0]
                
                # Format using smart formatting
                med_str = _smart_format(median, self.param_names[j], self.log_scale_params)
                up_str = _smart_format(upper, self.param_names[j], self.log_scale_params)
                low_str = _smart_format(lower, self.param_names[j], self.log_scale_params)
                
                title = f"${med_str}^{{+{up_str}}}_{{-{low_str}}}$"
                titles.append(title)
            
            # Generate corner plot
            fig = corner.corner(
                final_samples,
                labels=plot_labels,
                truths=target_i,
                show_titles=True,
                quantiles=quantiles,
                titles=titles,  # Use our custom formatted titles
                title_kwargs={"fontsize": 10}
            )
            
            # Overlay individual components on diagonal
            if component_samples is not None and weights is not None:
                self._overlay_components(fig, component_samples, weights, 
                                        len(self.param_names), self.param_names, 
                                        self.log_scale_params)
            
            space_label = "Physical Space" if self.plot_in_physical_space else "Scaled Space"
            fig.suptitle(f"Corner Plot ({space_label}) - Sample {i} - Epoch {epoch + 1}", 
                        fontsize=14, y=1.0)
            plot_filename = os.path.join(output_dir_epoch, f"sample_{i}.png")
            fig.savefig(plot_filename, dpi=120, bbox_inches='tight')
            plt.close(fig)

        print(f"[HybridCornerPlot] Finished saving {n_to_plot} corner plots to {output_dir_epoch}")

    def _transform_to_physical(self, samples_scaled, means_t, stds_t, log_indices):
        """
        Transform samples from scaled space to physical space.
        
        For log-scale parameters, keeps them as log10(X) for plotting.
        """
        samples_t = torch.as_tensor(samples_scaled, dtype=torch.float32)
        samples_unscaled = samples_t * stds_t + means_t
        return samples_unscaled.cpu().numpy()

    def _process_mixture_scaled(self, stuff, i):
        """Process mixture of Gaussians prediction in scaled space."""
        pi_logits_i = stuff["pi_logits"][i].cpu().to(torch.float32)
        mu_all_i_scaled = stuff["mu_all"][i].cpu().to(torch.float32)
        L_all_i_scaled = stuff["L_all"][i].cpu().to(torch.float32)
        K = pi_logits_i.shape[0]

        pi_i = torch.softmax(pi_logits_i, dim=-1)
        
        counts = (pi_i * self.num_posterior_samples).round().int()
        counts_sum_diff = self.num_posterior_samples - counts.sum()
        counts[torch.argmax(pi_i)] += counts_sum_diff

        mixture_samples_list = []
        component_samples_list = []
        active_weights = []

        for k in range(K):
            if counts[k] == 0:
                continue

            mu_k = mu_all_i_scaled[k]
            L_k = L_all_i_scaled[k]
            
            samples_k = _draw_samples(
                mu_k,
                L=L_k,
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

    def _process_single_gaussian_scaled(self, stuff, i):
        """Process single Gaussian prediction in scaled space."""
        mu_i = stuff["mu"][i].cpu().to(torch.float32)
        L_i = stuff["L"][i].cpu().to(torch.float32)
        
        samples = _draw_samples(
            mu_i,
            L=L_i,
            n=self.num_posterior_samples,
            seed=self.seed + i
        )
        
        return samples

    def _overlay_components(self, fig, component_samples, weights, n_params, 
                           param_names, log_scale_params):
        """Overlay individual Gaussian components on diagonal histograms with smart formatting."""
        diag_axes = [fig.axes[j * n_params + j] for j in range(n_params)]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(component_samples)))
        
        # Calculate stats for each component and parameter
        component_stats = []
        for param_idx in range(n_params):
            param_stats = []
            for k, samples in enumerate(component_samples):
                mu_k = samples[:, param_idx].mean()
                sigma_k = samples[:, param_idx].std()
                param_stats.append((mu_k, sigma_k))
            component_stats.append(param_stats)
        
        # Plot filled histograms
        for k, (samples, weight) in enumerate(zip(component_samples, weights)):
            for param_idx in range(n_params):
                ax = diag_axes[param_idx]
                
                xlim = ax.get_xlim()
                
                counts, bins = np.histogram(
                    samples[:, param_idx],
                    bins=30,
                    range=xlim,
                    density=True
                )
                
                counts_scaled = counts * weight
                
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.fill_between(
                    bin_centers,
                    0,
                    counts_scaled,
                    alpha=0.3,
                    color=colors[k],
                    step='mid'
                )
        
        # Add legends with parameter-specific stats (using smart formatting)
        if len(component_samples) > 1:
            from matplotlib.patches import Patch
            
            for param_idx, ax in enumerate(diag_axes):
                param_name = param_names[param_idx]
                legend_elements = []
                
                for k in range(len(component_samples)):
                    mu_k, sigma_k = component_stats[param_idx][k]
                    
                    # Use smart formatting for legend
                    mu_str = _smart_format(mu_k, param_name, log_scale_params)
                    sigma_str = _smart_format(sigma_k, param_name, log_scale_params)
                    
                    label = f'C{k+1}: μ={mu_str}, σ={sigma_str}'
                    legend_elements.append(
                        Patch(facecolor=colors[k], alpha=0.5, label=label)
                    )
                
                ax.legend(
                    handles=legend_elements, 
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),
                    fontsize=6,
                    framealpha=0.95,
                    edgecolor='gray',
                    fancybox=False
                )
            
            # Add global mixture weights
            ax_last = diag_axes[-1]
            weight_text = "Mixture Weights:\n" + "\n".join([
                f"Comp {k+1}: {weights[k]:.1%}" for k in range(len(weights))
            ])
            ax_last.text(
                1.02, 0.5, weight_text,
                transform=ax_last.transAxes,
                fontsize=7,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray')
            )