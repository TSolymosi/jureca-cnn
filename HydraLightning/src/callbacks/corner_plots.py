import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from torch.distributions.multivariate_normal import MultivariateNormal

# Attempt to import the corner library and provide a helpful error if it's not installed.
try:
    import corner
except ImportError:
    raise ImportError(
        "The 'corner' library is required for CornerPlotCallback. "
        "Please install it with 'pip install corner'"
    )


class CornerPlotCallback(Callback):
    """
    A PyTorch Lightning Callback to generate and save corner plots for a subset of validation samples.

    This callback is designed to work with a model that predicts a full multivariate Gaussian
    distribution (i.e., a mean vector and a Cholesky factor 'L' for the covariance matrix).
    It performs the following steps at the end of a validation epoch:
    1. Checks if the current epoch is one designated for plotting.
    2. Retrieves the cached predictions (mu, L) and true targets.
    3. Inverse transforms the scaled data back to its original physical scale.
    4. Handles the conversion of log-scaled parameters back to linear scale.
    5. For each specified sample, it generates a corner plot from the predicted posterior
       distribution and saves it to a file.
    """
    def __init__(self,
                 model_params: list,
                 log_scale_params: list,
                 plot_every_n_epochs: int = 5,
                 num_samples_to_plot: int = 10,
                 num_posterior_samples: int = 5000,
                 output_dir: str = "./corner_plots"):
        """
        Args:
            model_params (list): A list of strings with the names of the model's output parameters.
            log_scale_params (list): A sub-list of model_params indicating which parameters are predicted in log10 scale.
            plot_every_n_epochs (int): The frequency at which to generate plots.
            num_samples_to_plot (int): The number of validation samples to plot (from the beginning of the set).
            num_posterior_samples (int): The number of samples to draw from the predicted distribution for the plot.
            output_dir (str): The directory where the corner plot images will be saved.
        """
        super().__init__()
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.plot_every_n_epochs = plot_every_n_epochs
        self.num_samples_to_plot = num_samples_to_plot
        self.num_posterior_samples = num_posterior_samples
        self.output_dir = output_dir

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """This hook is executed on the main process after the validation epoch finishes."""

        # Skip execution during sanity checking or if it's not a plotting epoch.
        epoch = trainer.current_epoch
        if trainer.sanity_checking or (epoch + 1) % self.plot_every_n_epochs != 0:
            return

        # --- 1. Retrieve cached data and dataset reference ---
        cache = getattr(pl_module, "_val_cache", {})
        ds = getattr(pl_module, "_val_dataset", None)

        # For corner plots, we need the mean vectors ('preds') and Cholesky factors ('Ls').
        all_mus_scaled = cache.get("preds", None)
        all_Ls_scaled = cache.get("Ls", None)
        all_targets_scaled = cache.get("targets", None)

        # --- 2. Perform Sanity Checks ---
        if ds is None:
            print("[CornerPlotCallback] Could not find dataset reference. Skipping.")
            return
        if all_mus_scaled is None or all_Ls_scaled is None or all_targets_scaled is None:
            print("[CornerPlotCallback] Missing required 'preds', 'Ls', or 'targets' in cache. Skipping.")
            return
        # Ensure the model is in the correct mode to produce L matrices.
        if pl_module.hparams.model_cfg.covariance_type != "full":
            print(f"[CornerPlotCallback] Model `covariance_type` is not 'full'. Cannot generate corner plots. Skipping.")
            return

        print(f"[CornerPlotCallback] Generating corner plots for epoch {epoch + 1}...")
        os.makedirs(self.output_dir, exist_ok=True)

        # --- 3. Subset and Inverse Transform the Data ---
        num_to_plot = min(self.num_samples_to_plot, len(all_mus_scaled))
        if num_to_plot == 0:
            print("[CornerPlotCallback] No samples to plot. Skipping.")
            return

        mus_subset = all_mus_scaled[:num_to_plot]
        Ls_subset = all_Ls_scaled[:num_to_plot]
        targets_subset = all_targets_scaled[:num_to_plot]

        # Inverse transform the means and targets using the dataset's scaler.
        mus_original_scale = ds.inverse_transform_labels(mus_subset)
        targets_original_scale = ds.inverse_transform_labels(targets_subset)

        # Manually inverse transform the covariance matrix.
        # If x_scaled = (x_orig - mean) / std, then Cov(x_orig) = D_std @ Cov(x_scaled) @ D_std.
        # For Cholesky factors, this simplifies to L_orig = D_std @ L_scaled.
        scaler_stds = ds.scaler_stds.to(Ls_subset.device, dtype=Ls_subset.dtype)
        D_std = torch.diag(scaler_stds).unsqueeze(0)  # Shape: [1, d, d] for broadcasting
        Ls_original_scale = D_std @ Ls_subset

        # --- 4. Loop Through Samples and Generate Plots ---
        for i in range(num_to_plot):
            mu = mus_original_scale[i]
            L = Ls_original_scale[i]
            target = targets_original_scale[i]

            # Reconstruct the full covariance matrix from its Cholesky factor.
            # Add a small jitter for numerical stability.
            sigma = L @ L.T + torch.eye(L.shape[0], device=L.device) * 1e-6

            # Create a multivariate normal distribution from the predicted parameters.
            try:
                distribution = MultivariateNormal(loc=mu, covariance_matrix=sigma)
                # Draw samples from the posterior distribution.
                posterior_samples = distribution.sample((self.num_posterior_samples,))
            except Exception as e:
                print(f"[CornerPlotCallback] Could not create distribution for sample {i}. Error: {e}. Skipping sample.")
                continue

            # --- 5. Handle Log-Scaled Parameters ---
            # Check for dynamic range before plotting.
            # The standard deviation of the samples is a good measure of dynamic range.
            stds = torch.std(posterior_samples, dim=0)
            
            # Check if any parameter has a standard deviation that is effectively zero.
            if torch.any(stds < 1e-9):
                # Find which parameters have no range.
                bad_params = [self.model_params[j] for j, s in enumerate(stds) if s < 1e-9]
                print(f"[CornerPlotCallback] WARNING: Epoch {epoch + 1}, Sample {i}: No dynamic range for parameters {bad_params}. "
                      "This is common in early training. Skipping this corner plot.")
                continue # Skip to the next sample.

            # We must convert the log-scaled samples and targets back to linear scale for plotting.
            plot_samples = posterior_samples.clone().cpu().numpy()
            plot_targets = target.clone().cpu().numpy()

            param_indices_to_transform = [
                idx for idx, name in enumerate(self.model_params) if name in self.log_scale_params
            ]

            for idx in param_indices_to_transform:
                plot_samples[:, idx] = 10**plot_samples[:, idx]
                plot_targets[idx] = 10**plot_targets[idx]

            # --- 6. Create and Save the Corner Plot ---
            fig = corner.corner(
                plot_samples,
                labels=self.model_params,
                truths=plot_targets,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12},
                truth_color="red"
            )
            fig.suptitle(f"Corner Plot for Sample {i} - Epoch {epoch + 1}", fontsize=16)

            plot_filename = os.path.join(self.output_dir, f"epoch_{epoch+1}_sample_{i}.png")
            try:
                fig.savefig(plot_filename, dpi=120)
                plt.close(fig)  # Close the figure to free memory
            except Exception as e:
                print(f"[CornerPlotCallback] Failed to save plot {plot_filename}. Error: {e}")

        print(f"[CornerPlotCallback] Finished saving {num_to_plot} corner plots to {self.output_dir}.")