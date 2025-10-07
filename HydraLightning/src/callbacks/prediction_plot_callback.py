import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class PredictionPlotCallback(Callback):
    """
    A PyTorch Lightning Callback that generates several diagnostic plots and artifacts
    at the end of each validation epoch.

    Primary functions:
    1.  Generates scatter plots of predicted vs. true values for each parameter.
    2.  Handles both linear and log-scaled parameters appropriately.
    3.  At the end of the final training epoch, it generates additional artifacts:
        - Residual vs. Sigma plots to analyze uncertainty quality.
        - Calculates 1-sigma coverage ratios.
        - Saves all predictions, uncertainties, and true values to a final CSV file.
    """
    def __init__(self, model_params, log_scale_params, output_dir="./pred_plots"):
        super().__init__()
        # List of all parameter names the model is predicting.
        self.model_params = model_params
        # A sub-list of parameters that are handled in log10 space.
        self.log_scale_params = log_scale_params
        # The base directory where all output plots and files will be saved.
        self.output_dir = output_dir
        # A flag to ensure that final artifacts (CSV, etc.) are only generated once.
        self._final_artifacts_done = False

    @rank_zero_only  # Ensures this hook only runs on the main process (rank 0).
    def on_validation_epoch_end(self, trainer, pl_module):
        """This hook is executed automatically by Lightning after the validation loop finishes."""
        # Skip this callback during Lightning's initial sanity checking runs.
        if trainer.sanity_checking:
            return
            
        pl_module.eval() # Set the model to evaluation mode.

        # --- Section 1: Fetching the necessary data ---
        # Retrieve the dictionary of cached tensors from the LightningModule.
        cache = getattr(pl_module, "_val_cache", {})
        # Retrieve the reference to the dataset object to access its unscaling methods.
        ds = getattr(pl_module, "_val_dataset", None)

        # Get the aggregated predictions, targets, and uncertainties from the cache.
        all_preds   = cache.get("preds", None)
        all_targets = cache.get("targets", None)
        all_sigmas  = cache.get("sigmas", None)

        # If any of the essential data is missing, skip the rest of the callback.
        if ds is None or all_preds is None or all_targets is None or all_preds.numel() == 0:
            print("[PredictionPlotCallback] No dataset or cached tensors found in LightningModule. Skipping plotting.")
            return

        # --- Section 2: Inverse transform data back to original physical scale ---
        # Use the dataset's methods to un-normalize the targets.
        targets_original = ds.inverse_transform_labels(all_targets)
        
        # If the model is an MDN (predicting uncertainty), un-normalize both predictions and sigmas.
        if pl_module.use_mdn and all_sigmas is not None:
            preds_original, sigmas_original = ds.inverse_transform_labels_with_uncertainty(
                all_preds, all_sigmas
            )
        else: # Otherwise, just un-normalize the predictions.
            preds_original = ds.inverse_transform_labels(all_preds)
            sigmas_original = None

        epoch = trainer.current_epoch

        # --- Section 3: Setup output directories ---
        os.makedirs(self.output_dir, exist_ok=True)
        Epoch_Plots = os.path.join(self.output_dir, "Epoch_Plots")
        ResNet_Plots = os.path.join(self.output_dir, "ResNet_Plots")
        Final_Results = os.path.join(self.output_dir, "Final_Results")
        for d in (Epoch_Plots, ResNet_Plots, Final_Results):
            os.makedirs(d, exist_ok=True)

        # Convert tensors to NumPy arrays for plotting and analysis.
        all_predictions_original = preds_original.cpu().numpy()
        all_targets_original = targets_original.cpu().numpy()
        all_sigmas_original = sigmas_original.cpu().numpy() if sigmas_original is not None else None

        # --- Section 4: Generate Predicted vs. True scatter plots for each parameter ---
        for i, param in enumerate(self.model_params):
            y_true = all_targets_original[:, i]
            y_pred = all_predictions_original[:, i]
            y_err  = all_sigmas_original[:, i] if all_sigmas_original is not None else None

            plt.figure(figsize=(10, 10)) # Increased figure size for better readability

            scale_type = "" # Variable to hold the scale type for the print statement.
            
            # If the parameter is in the log_scale_params list, configure a log-log plot.
            if param in self.log_scale_params:
                scale_type = "Log-Log Scale"
                plt.xscale("log"); plt.yscale("log")
                plt.title(f"Epoch {epoch+1} - {param} ({scale_type})")
                # Create a mask to filter out non-positive values, which are invalid for log plots.
                mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
            else: # Otherwise, use a standard linear scale.
                scale_type = "Linear Scale"
                plt.title(f"Epoch {epoch+1} - {param} ({scale_type})")
                mask = np.isfinite(y_true) & np.isfinite(y_pred)

            # Only proceed with plotting if there is valid data after masking.
            if np.any(mask):
                yt, yp = y_true[mask], y_pred[mask]

                # Determine plot limits to make the plot square and centered on the y=x line.
                minval = min(yt.min(), yp.min())
                maxval = max(yt.max(), yp.max())
                # For log plots, ensure the minimum value is a small positive number.
                if minval <= 0 and param in self.log_scale_params:
                    minval = np.nextafter(0.0, 1.0)

                # If uncertainties are available, plot with error bars.
                if pl_module.use_mdn and y_err is not None:
                    ye = y_err[mask]
                    # This logic prevents error bars from going below zero on a log plot, which is invalid.
                    lower = np.maximum(yp - ye, minval * 0.999 if param in self.log_scale_params else -np.inf)
                    upper = yp + ye
                    yerr_plot = np.vstack([yp - lower, upper - yp])
                    plt.errorbar(yt, yp, yerr=yerr_plot,
                                 fmt="o", alpha=0.3, markersize=4, capsize=2, elinewidth=1)
                else: # Otherwise, create a simple scatter plot.
                    plt.scatter(yt, yp, alpha=0.3, s=10)

                # Plot the red y=x diagonal line for reference.
                plt.plot([minval, maxval], [minval, maxval], "r--", label="y = x")
                plt.xlim(minval, maxval)
                plt.ylim(minval, maxval)
            else:
                # Handle cases where no valid data is available for plotting.
                print(f"[PredictionPlotCallback] No valid data to plot for parameter {param}")
                # ... (error text on plot) ...

            plt.xlabel(f"True {param}")
            plt.ylabel(f"Predicted {param}")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()

            fname = os.path.join(Epoch_Plots, f"E{epoch+1}_Test_{param}.png")
            plt.savefig(fname, dpi=150)
            plt.close()
            
            # --- MODIFIED PRINT STATEMENT ---
            print(f"Saved Epoch Plot: {fname} ({scale_type})", flush=True)

        # --- Section 5: Generate Final Artifacts (only at the very end of training) ---
        is_final_epoch = (epoch + 1 == trainer.max_epochs)
        is_early_stop = bool(getattr(trainer, "should_stop", False))
        
        if (is_final_epoch or is_early_stop) and not self._final_artifacts_done:
            self._final_artifacts_done = True # Ensure this block runs only once.
            print("[PredictionPlotCallback] Final epoch reached. Generating final artifacts...")

            # --- Residual vs. Sigma Plots ---
            residual = preds_original - targets_original
            coverage_ratios = []

            for i, name in enumerate(self.model_params):
                plt.figure(figsize=(6, 5))
                plt.scatter(sigmas_original[:, i], residual[:, i], alpha=0.3, s=10)
                plt.xlabel('Predicted σ'); plt.ylabel('Residual (μ - y)')
                plt.title(f'Residual vs σ: {name}')
                plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(ResNet_Plots, f"{name}_residual.png"))
                plt.close()

                within_1sigma = ((preds_original > targets_original - sigmas_original) &
                                (preds_original < targets_original + sigmas_original)).float().mean().item()
                coverage_ratios.append((name, within_1sigma))

            # --- Coverage Ratio Calculation ---
            # ... (logic for calculating and printing coverage) ...

            print("\nCoverage within ±1σ:")
            for name, ratio in coverage_ratios:
                print(f"  {name}: {ratio * 100:.2f}%")

            # if sigmas_original is not None:
            #     print("Evaluating calibration and plotting curves...")
            #     _ = evaluate_calibration(
            #         mu=preds_original,
            #         sigma=sigmas_original,
            #         y_true=targets_original,
            #         param_names=self.model_params,
            #         epoch=epoch,
            #         folder_name=self.output_dir
            #     )

            # --- Final CSV Export ---
            # ... (logic for creating and saving the final predictions DataFrame) ...
            df = pd.DataFrame()
            preds_np = preds_original.cpu().numpy()
            targets_np = targets_original.cpu().numpy()
            sigmas_np = None

            if sigmas_original is not None:
                sigmas_np = sigmas_original.cpu().numpy()

            for i, param_name in enumerate(self.model_params):
                df[f"true_{param_name}"] = targets_np[:, i]
                df[f"pred_{param_name}"] = preds_np[:, i]
                if sigmas_original is not None:
                    df[f"sigma_{param_name}"] = sigmas_np[:, i]

            csv_path = os.path.join(Final_Results, "final_predictions.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved full predictions CSV at: {csv_path}")
            
class CacheResetCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        cache = getattr(pl_module, "_val_cache", None)
        if isinstance(cache, dict):
            for k in list(cache.keys()):
                v = cache[k]
                if isinstance(v, list):
                    v.clear()
            cache.clear()
        # optional, cheap:
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CacheResetCallback] Cleared validation cache and freed memory.")