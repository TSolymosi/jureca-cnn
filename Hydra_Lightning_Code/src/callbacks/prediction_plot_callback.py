import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class PredictionPlotCallback(Callback):
    """
    A PyTorch Lightning Callback that generates diagnostic plots colored by SNR.
    
    NEW: Can generate additional diagnostic plots for specific parameters,
    colored by other parameters to explore pockets of parameter space.
    """
    def __init__(self, 
                 model_params, 
                 log_scale_params, 
                 output_dir="./pred_plots",
                 enable_parameter_space_diagnostics=False,
                 diagnostic_target_params=None,
                 diagnostic_plot_every_n_epochs=10):
        """
        Args:
            model_params: List of parameter names
            log_scale_params: List of parameters that use log scale
            output_dir: Base output directory
            enable_parameter_space_diagnostics: Toggle for parameter-colored plots
            diagnostic_target_params: List of params to diagnose (e.g., ['Tlow', 'p'])
                                     If None, diagnoses all parameters
            diagnostic_plot_every_n_epochs: How often to generate diagnostic plots
        """
        super().__init__()
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.output_dir = output_dir
        self._final_artifacts_done = False
        
        # NEW: Parameter space diagnostic options
        self.enable_param_diagnostics = enable_parameter_space_diagnostics
        self.diagnostic_target_params = diagnostic_target_params or model_params
        self.diagnostic_plot_every_n = diagnostic_plot_every_n_epochs

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate SNR-colored prediction plots and optional parameter-colored diagnostics."""
        if trainer.sanity_checking:
            return
            
        pl_module.eval()

        # Fetch cached data
        cache = getattr(pl_module, "_val_cache", {})
        ds = getattr(pl_module, "_val_dataset", None)

        all_preds = cache.get("preds", None)
        all_targets = cache.get("targets", None)
        all_sigmas = cache.get("sigmas", None)
        all_snr = cache.get("snr", None)

        if ds is None or all_preds is None or all_targets is None or all_preds.numel() == 0:
            print("[PredictionPlotCallback] No dataset or cached tensors found. Skipping.")
            return

        epoch = trainer.current_epoch

        # Setup output directories
        os.makedirs(self.output_dir, exist_ok=True)
        Epoch_Plots = os.path.join(self.output_dir, "Epoch_Plots")
        ResNet_Plots = os.path.join(self.output_dir, "ResNet_Plots")
        Final_Results = os.path.join(self.output_dir, "Final_Results")
        Diagnostic_Plots = os.path.join(self.output_dir, "Diagnostic_Plots")  # NEW
        for d in (Epoch_Plots, ResNet_Plots, Final_Results, Diagnostic_Plots):
            os.makedirs(d, exist_ok=True)

        # Keep log-space versions for proper error bar calculation
        means_t = ds.scaler_means
        stds_t = ds.scaler_stds
        
        preds_logspace = all_preds * stds_t + means_t
        targets_logspace = all_targets * stds_t + means_t
        sigmas_logspace = all_sigmas * stds_t if all_sigmas is not None else None
        
        # Inverse transform to original/physical scale
        targets_original = ds.inverse_transform_labels(all_targets)
        
        if pl_module.use_mdn and all_sigmas is not None:
            preds_original, sigmas_original = ds.inverse_transform_labels_with_uncertainty(
                all_preds, all_sigmas
            )
        else:
            preds_original = ds.inverse_transform_labels(all_preds)
            sigmas_original = None

        # Convert to numpy
        all_predictions_original = preds_original.cpu().numpy()
        all_targets_original = targets_original.cpu().numpy()
        all_sigmas_original = sigmas_original.cpu().numpy() if sigmas_original is not None else None
        snr_values = all_snr.cpu().numpy() if all_snr is not None else None
        
        # Log-space versions (numpy)
        preds_logspace_np = preds_logspace.cpu().numpy()
        targets_logspace_np = targets_logspace.cpu().numpy()
        sigmas_logspace_np = sigmas_logspace.cpu().numpy() if sigmas_logspace is not None else None

        # Generate standard SNR-colored plots for each parameter
        self._plot_standard_predictions(
            all_predictions_original, all_targets_original, all_sigmas_original,
            preds_logspace_np, targets_logspace_np, sigmas_logspace_np,
            snr_values, epoch, Epoch_Plots, pl_module.use_mdn
        )

        # NEW: Generate parameter-colored diagnostic plots
        if self.enable_param_diagnostics and (epoch + 1) % self.diagnostic_plot_every_n == 0:
            print(f"[PredictionPlotCallback] Generating parameter space diagnostic plots...")
            self._plot_parameter_space_diagnostics(
                all_predictions_original, all_targets_original, all_sigmas_original,
                preds_logspace_np, targets_logspace_np, sigmas_logspace_np,
                epoch, Diagnostic_Plots, pl_module.use_mdn
            )

        # Final artifacts
        is_final_epoch = (epoch + 1 == trainer.max_epochs)
        is_early_stop = bool(getattr(trainer, "should_stop", False))
        
        if (is_final_epoch or is_early_stop) and not self._final_artifacts_done:
            self._final_artifacts_done = True
            print("[PredictionPlotCallback] Final epoch. Generating final artifacts...")
            self._save_final_csv(
                preds_original, targets_original, sigmas_original, 
                snr_values, Final_Results
            )

    def _plot_standard_predictions(self, all_predictions_original, all_targets_original, 
                                   all_sigmas_original, preds_logspace_np, targets_logspace_np,
                                   sigmas_logspace_np, snr_values, epoch, output_dir, use_mdn):
        """Generate standard SNR-colored prediction plots."""
        for i, param in enumerate(self.model_params):
            plt.figure(figsize=(10, 10))

            scale_type = ""
            is_log_param = param in self.log_scale_params
            
            # Configure scale and get appropriate data
            if is_log_param:
                scale_type = "Log-Log Scale"
                plt.xscale("log")
                plt.yscale("log")
                plt.title(f"Epoch {epoch+1} - {param} ({scale_type})")
                
                # Use log-space data for log-params
                y_true_log = targets_logspace_np[:, i]
                y_pred_log = preds_logspace_np[:, i]
                y_err_log = sigmas_logspace_np[:, i] if sigmas_logspace_np is not None else None
                
                # Convert to physical space for plotting on log axes
                y_true = 10 ** y_true_log
                y_pred = 10 ** y_pred_log
                
                # For error bars: compute asymmetric errors in physical space
                if y_err_log is not None:
                    y_pred_upper_log = y_pred_log + y_err_log
                    y_pred_lower_log = y_pred_log - y_err_log
                    
                    y_pred_upper = 10 ** y_pred_upper_log
                    y_pred_lower = 10 ** y_pred_lower_log
                    
                    yerr_lower = y_pred - y_pred_lower
                    yerr_upper = y_pred_upper - y_pred
                    
                    y_err = np.array([yerr_lower, yerr_upper])
                else:
                    y_err = None
                
                mask = (y_true > 0) & (y_pred > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
                
            else:
                # Linear parameters: use original physical space
                scale_type = "Linear Scale"
                plt.title(f"Epoch {epoch+1} - {param} ({scale_type})")
                
                y_true = all_targets_original[:, i]
                y_pred = all_predictions_original[:, i]
                y_err = all_sigmas_original[:, i] if all_sigmas_original is not None else None
                
                mask = np.isfinite(y_true) & np.isfinite(y_pred)

            if np.any(mask):
                yt, yp = y_true[mask], y_pred[mask]
                
                minval = min(yt.min(), yp.min())
                maxval = max(yt.max(), yp.max())
                if minval <= 0 and is_log_param:
                    minval = np.nextafter(0.0, 1.0)

                # Setup SNR colormap
                if snr_values is not None:
                    snr_masked = snr_values[mask]
                    vmin, vmax = 8, 100
                    norm = Normalize(vmin=np.log10(vmin), vmax=np.log10(vmax))
                    cmap = plt.cm.viridis
                    snr_log = np.log10(np.clip(snr_masked, vmin, vmax))
                    colors = cmap(norm(snr_log))
                else:
                    colors = 'blue'

                # Plot with error bars and SNR colors
                if use_mdn and y_err is not None:
                    ye = y_err[:, mask] if is_log_param else y_err[mask]
                    
                    if is_log_param:
                        ye[0] = np.minimum(ye[0], yp - minval * 1.001)
                        ye[0] = np.maximum(ye[0], 0.0)
                        ye[1] = np.maximum(ye[1], 0.0)
                        too_close_to_min = (yp < minval * 1.1)
                        ye[0][too_close_to_min] = 0.0
                        yerr_plot = ye
                    else:
                        lower = np.maximum(yp - ye, -np.inf)
                        upper = yp + ye
                        yerr_lower = yp - lower
                        yerr_upper = upper - yp
                        yerr_lower = np.maximum(yerr_lower, 0.0)
                        yerr_upper = np.maximum(yerr_upper, 0.0)
                        yerr_plot = np.vstack([yerr_lower, yerr_upper])
                    
                    if snr_values is not None:
                        for j in range(len(yt)):
                            plt.errorbar(
                                yt[j], yp[j], 
                                yerr=[[yerr_plot[0, j]], [yerr_plot[1, j]]],
                                fmt='o', color=colors[j], alpha=0.6, 
                                markersize=4, capsize=2, elinewidth=1
                            )
                    else:
                        plt.errorbar(yt, yp, yerr=yerr_plot,
                                fmt="o", alpha=0.3, markersize=4, capsize=2, elinewidth=1)
                else:
                    plt.scatter(yt, yp, c=colors if snr_values is not None else 'blue', 
                            alpha=0.6, s=20, cmap=cmap if snr_values is not None else None)

                # Add colorbar for SNR
                if snr_values is not None:
                    sm = ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=plt.gca(), label='SNR')
                    cbar_ticks = [8, 15, 30, 50, 100]
                    cbar.set_ticks(np.log10(cbar_ticks))
                    cbar.set_ticklabels(cbar_ticks)

                plt.plot([minval, maxval], [minval, maxval], "r--", label="y = x", linewidth=2)
                plt.xlim(minval, maxval)
                plt.ylim(minval, maxval)
            else:
                print(f"[PredictionPlotCallback] No valid data to plot for parameter {param}")

            plt.xlabel(f"True {param}")
            plt.ylabel(f"Predicted {param}")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()

            fname = os.path.join(output_dir, f"E{epoch+1}_Test_{param}.png")
            plt.savefig(fname, dpi=150)
            plt.close()

    def _plot_parameter_space_diagnostics(self, all_predictions_original, all_targets_original,
                                          all_sigmas_original, preds_logspace_np, targets_logspace_np,
                                          sigmas_logspace_np, epoch, output_dir, use_mdn):
        """
        Generate diagnostic plots for each target parameter, colored by all other parameters.
        This helps identify pockets of parameter space where the model performs differently.
        """
        # For each target parameter to diagnose
        for target_param in self.diagnostic_target_params:
            if target_param not in self.model_params:
                print(f"[WARNING] Diagnostic target '{target_param}' not in model_params")
                continue
            
            target_idx = self.model_params.index(target_param)
            target_is_log = target_param in self.log_scale_params
            
            # Create subdirectory for this target parameter
            target_dir = os.path.join(output_dir, f"E{epoch+1}_{target_param}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Get target parameter data (using same logic as standard plots)
            if target_is_log:
                y_true_target_log = targets_logspace_np[:, target_idx]
                y_pred_target_log = preds_logspace_np[:, target_idx]
                y_err_target_log = sigmas_logspace_np[:, target_idx] if sigmas_logspace_np is not None else None
                
                y_true_target = 10 ** y_true_target_log
                y_pred_target = 10 ** y_pred_target_log
                
                if y_err_target_log is not None:
                    y_pred_upper_log = y_pred_target_log + y_err_target_log
                    y_pred_lower_log = y_pred_target_log - y_err_target_log
                    y_pred_upper = 10 ** y_pred_upper_log
                    y_pred_lower = 10 ** y_pred_lower_log
                    yerr_lower = y_pred_target - y_pred_lower
                    yerr_upper = y_pred_upper - y_pred_target
                    y_err_target = np.array([yerr_lower, yerr_upper])
                else:
                    y_err_target = None
            else:
                y_true_target = all_targets_original[:, target_idx]
                y_pred_target = all_predictions_original[:, target_idx]
                y_err_target = all_sigmas_original[:, target_idx] if all_sigmas_original is not None else None
            
            # For each other parameter (to use as color)
            for color_param in self.model_params:
                if color_param == target_param:
                    continue  # Skip self-coloring
                
                color_idx = self.model_params.index(color_param)
                color_is_log = color_param in self.log_scale_params
                
                # Get color values (TRUE values in appropriate space)
                if color_is_log:
                    # Use log-space values for log parameters
                    color_values = targets_logspace_np[:, color_idx]  # This is log10(X)
                    color_values_physical = 10 ** color_values  # For colorbar labels
                else:
                    color_values = all_targets_original[:, color_idx]
                    color_values_physical = color_values
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Create mask
                if target_is_log:
                    mask = ((y_true_target > 0) & (y_pred_target > 0) & 
                           np.isfinite(y_true_target) & np.isfinite(y_pred_target) &
                           np.isfinite(color_values))
                else:
                    mask = (np.isfinite(y_true_target) & np.isfinite(y_pred_target) &
                           np.isfinite(color_values))
                
                if not np.any(mask):
                    plt.close(fig)
                    continue
                
                # Apply mask
                yt = y_true_target[mask]
                yp = y_pred_target[mask]
                ye = y_err_target[:, mask] if (y_err_target is not None and target_is_log) else (y_err_target[mask] if y_err_target is not None else None)
                color_vals = color_values[mask]
                color_vals_phys = color_values_physical[mask]
                
                # Determine plot limits for target
                minval = min(yt.min(), yp.min())
                maxval = max(yt.max(), yp.max())
                if minval <= 0 and target_is_log:
                    minval = np.nextafter(0.0, 1.0)
                
                # Setup colormap for the coloring parameter
                if color_is_log:
                    # Use log-space for colormap (already in log form)
                    vmin_color = np.percentile(color_vals, 1)
                    vmax_color = np.percentile(color_vals, 99)
                    color_vals_plot = color_vals
                    color_label = f'log₁₀({color_param})'
                else:
                    # Linear scale coloring
                    vmin_color = np.percentile(color_vals, 1)
                    vmax_color = np.percentile(color_vals, 99)
                    color_vals_plot = color_vals
                    color_label = color_param
                
                norm = Normalize(vmin=vmin_color, vmax=vmax_color)
                cmap_param = plt.cm.plasma  # Different colormap than SNR
                
                # Configure axes
                if target_is_log:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    scale_type = "Log-Log"
                else:
                    scale_type = "Linear"
                
                # Plot with error bars and parameter coloring
                if use_mdn and ye is not None:
                    if target_is_log:
                        # ye is [2, N] array with asymmetric errors
                        ye[0] = np.minimum(ye[0], yp - minval * 1.001)
                        ye[0] = np.maximum(ye[0], 0.0)
                        ye[1] = np.maximum(ye[1], 0.0)
                        too_close_to_min = (yp < minval * 1.1)
                        ye[0][too_close_to_min] = 0.0
                        yerr_plot = ye
                    else:
                        lower = np.maximum(yp - ye, -np.inf)
                        upper = yp + ye
                        yerr_lower = yp - lower
                        yerr_upper = upper - yp
                        yerr_lower = np.maximum(yerr_lower, 0.0)
                        yerr_upper = np.maximum(yerr_upper, 0.0)
                        yerr_plot = np.vstack([yerr_lower, yerr_upper])
                    
                    # Plot each point individually with its color
                    for j in range(len(yt)):
                        color = cmap_param(norm(color_vals_plot[j]))
                        ax.errorbar(
                            yt[j], yp[j], 
                            yerr=[[yerr_plot[0, j]], [yerr_plot[1, j]]],
                            fmt='o', color=color, alpha=0.6, 
                            markersize=4, capsize=2, elinewidth=1
                        )
                else:
                    # Simple scatter plot
                    scatter = ax.scatter(yt, yp, c=color_vals_plot, cmap=cmap_param, norm=norm,
                                       alpha=0.6, s=20, edgecolors='none')
                
                # Add colorbar
                sm = ScalarMappable(cmap=cmap_param, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label=color_label)
                
                # If log scale, show actual physical values on colorbar
                if color_is_log:
                    tick_locs = cbar.get_ticks()
                    # Convert from log-space to physical space
                    tick_labels = [f'{10**loc:.2g}' for loc in tick_locs]
                    cbar.set_ticklabels(tick_labels)
                
                # Plot y=x line
                ax.plot([minval, maxval], [minval, maxval], "r--", label="y = x", linewidth=2)
                ax.set_xlim(minval, maxval)
                ax.set_ylim(minval, maxval)
                
                # Labels and title
                ax.set_xlabel(f"True {target_param}")
                ax.set_ylabel(f"Predicted {target_param}")
                ax.set_title(f"Epoch {epoch+1} - {target_param} ({scale_type})\nColored by {color_param}")
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
                ax.legend()
                
                plt.tight_layout()
                
                # Save
                fname = os.path.join(target_dir, f"{target_param}_colored_by_{color_param}.png")
                plt.savefig(fname, dpi=150)
                plt.close(fig)
            
            print(f"Saved diagnostic plots for {target_param} in {target_dir}")

    def _save_final_csv(self, preds_original, targets_original, sigmas_original, 
                       snr_values, output_dir):
        """Save final predictions to CSV."""
        df = pd.DataFrame()
        preds_np = preds_original.cpu().numpy()
        targets_np = targets_original.cpu().numpy()
        sigmas_np = sigmas_original.cpu().numpy() if sigmas_original is not None else None

        for i, param_name in enumerate(self.model_params):
            df[f"true_{param_name}"] = targets_np[:, i]
            df[f"pred_{param_name}"] = preds_np[:, i]
            if sigmas_original is not None:
                df[f"sigma_{param_name}"] = sigmas_np[:, i]
        
        if snr_values is not None:
            df["snr"] = snr_values

        csv_path = os.path.join(output_dir, "final_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved full predictions CSV (with SNR) at: {csv_path}")


class CacheResetCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        cache = getattr(pl_module, "_val_cache", None)
        if isinstance(cache, dict):
            for k in list(cache.keys()):
                v = cache[k]
                if isinstance(v, list):
                    v.clear()
            cache.clear()
        
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CacheResetCallback] Cleared validation cache and freed memory.")


        