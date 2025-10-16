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
    """
    def __init__(self, model_params, log_scale_params, output_dir="./pred_plots"):
        super().__init__()
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.output_dir = output_dir
        self._final_artifacts_done = False

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """Generate SNR-colored prediction plots."""
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
        for d in (Epoch_Plots, ResNet_Plots, Final_Results):
            os.makedirs(d, exist_ok=True)

        # ===== NEW: Keep log-space versions for proper error bar calculation =====
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

        # Generate SNR-colored plots for each parameter
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
                
                # ===== CRITICAL FIX: Use log-space data for log-params =====
                # Get data in log-space (log10(X))
                y_true_log = targets_logspace_np[:, i]
                y_pred_log = preds_logspace_np[:, i]
                y_err_log = sigmas_logspace_np[:, i] if sigmas_logspace_np is not None else None
                
                # Convert to physical space for plotting on log axes
                y_true = 10 ** y_true_log
                y_pred = 10 ** y_pred_log
                
                # For error bars: compute asymmetric errors in physical space
                if y_err_log is not None:
                    # Upper and lower bounds in log-space (symmetric)
                    y_pred_upper_log = y_pred_log + y_err_log
                    y_pred_lower_log = y_pred_log - y_err_log
                    
                    # Convert to physical space (now asymmetric)
                    y_pred_upper = 10 ** y_pred_upper_log
                    y_pred_lower = 10 ** y_pred_lower_log
                    
                    # Calculate asymmetric error bar distances
                    yerr_lower = y_pred - y_pred_lower  # Distance down
                    yerr_upper = y_pred_upper - y_pred   # Distance up
                    
                    y_err = np.array([yerr_lower, yerr_upper])  # Shape: [2, N]
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
                
                # Determine plot limits
                minval = min(yt.min(), yp.min())
                maxval = max(yt.max(), yp.max())
                if minval <= 0 and is_log_param:
                    minval = np.nextafter(0.0, 1.0)

                # --- Setup SNR colormap ---
                if snr_values is not None:
                    snr_masked = snr_values[mask]
                    
                    # Define SNR range for colormap (10 to 50, log scale)
                    vmin, vmax = 10, 50
                    norm = Normalize(vmin=np.log10(vmin), vmax=np.log10(vmax))
                    cmap = plt.cm.viridis
                    
                    # Map SNR to colors (log scale)
                    snr_log = np.log10(np.clip(snr_masked, vmin, vmax))
                    colors = cmap(norm(snr_log))
                else:
                    colors = 'blue'  # Fallback if no SNR data

                # Plot with error bars and SNR colors
                if pl_module.use_mdn and y_err is not None:
                    ye = y_err[:, mask] if is_log_param else y_err[mask]
                    
                    if is_log_param:
                        # ye is [2, N] array with asymmetric errors
                        # Clamp lower errors to keep values positive
                        ye[0] = np.minimum(ye[0], yp - minval * 1.001)
                        yerr_plot = ye
                    else:
                        # Symmetric errors for linear params
                        lower = np.maximum(yp - ye, -np.inf)
                        upper = yp + ye
                        yerr_plot = np.vstack([yp - lower, upper - yp])
                    
                    # Plot each point individually with its SNR color
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
                    # Simple scatter plot
                    plt.scatter(yt, yp, c=colors if snr_values is not None else 'blue', 
                            alpha=0.6, s=20, cmap=cmap if snr_values is not None else None)

                # Add colorbar for SNR
                if snr_values is not None:
                    sm = ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=plt.gca(), label='SNR (dB scale)')
                    # Set colorbar ticks to actual SNR values
                    cbar_ticks = [10, 15, 20, 30, 50]
                    cbar.set_ticks(np.log10(cbar_ticks))
                    cbar.set_ticklabels(cbar_ticks)

                # Plot y=x line
                plt.plot([minval, maxval], [minval, maxval], "r--", label="y = x", linewidth=2)
                plt.xlim(minval, maxval)
                plt.ylim(minval, maxval)
            else:
                print(f"[PredictionPlotCallback] No valid data to plot for parameter {param}")

            plt.xlabel(f"True {param}")
            plt.ylabel(f"Predicted {param}")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()

            fname = os.path.join(Epoch_Plots, f"E{epoch+1}_Test_{param}.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            
            print(f"Saved SNR-colored Epoch Plot: {fname} ({scale_type})", flush=True)

        # --- Final artifacts (CSV with SNR) ---
        is_final_epoch = (epoch + 1 == trainer.max_epochs)
        is_early_stop = bool(getattr(trainer, "should_stop", False))
        
        if (is_final_epoch or is_early_stop) and not self._final_artifacts_done:
            self._final_artifacts_done = True
            print("[PredictionPlotCallback] Final epoch. Generating final artifacts...")

            # Save CSV with SNR values
            df = pd.DataFrame()
            preds_np = preds_original.cpu().numpy()
            targets_np = targets_original.cpu().numpy()
            sigmas_np = sigmas_original.cpu().numpy() if sigmas_original is not None else None

            for i, param_name in enumerate(self.model_params):
                df[f"true_{param_name}"] = targets_np[:, i]
                df[f"pred_{param_name}"] = preds_np[:, i]
                if sigmas_original is not None:
                    df[f"sigma_{param_name}"] = sigmas_np[:, i]
            
            # Add SNR column
            if snr_values is not None:
                df["snr"] = snr_values

            csv_path = os.path.join(Final_Results, "final_predictions.csv")
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