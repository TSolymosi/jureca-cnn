import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightning.pytorch.callbacks import Callback
from src.evaluation.calibration import evaluate_calibration
from pathlib import Path

import math
from lightning.pytorch.utilities import rank_zero_only

class PredictionPlotCallback(Callback):
    def __init__(self, model_params, log_scale_params, output_dir="Epoch_Plots"):
        super().__init__()
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.output_dir = output_dir
        self._final_artifacts_done = False


    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        #print(f"[Rank {trainer.global_rank}] -> Entered on_validation_epoch_end Prediction Plot")
        pl_module.eval()

        cache = getattr(pl_module, "_val_cache", {})
        ds = getattr(pl_module, "_val_dataset", None)

        all_preds   = cache.get("preds", None)
        all_targets = cache.get("targets", None)
        all_sigmas  = cache.get("sigmas", None)

        if ds is None:
            print("[PredictionPlotCallback] No dataset reference found. Skipping.")
            return
        if all_preds is None or all_targets is None or all_preds.numel() == 0:
            print("[PredictionPlotCallback] No cached validation tensors on rank 0. Skipping.")
            return

        # Inverse transforms using your dataset helpers
        targets_original = ds.inverse_transform_labels(all_targets)

        print(f"label scaling: \n means:{ds.scaler_means}, \n stds:{ds.scaler_stds}")

        if pl_module.use_mdn and all_sigmas is not None:
            preds_original, sigmas_original = ds.inverse_transform_labels_with_uncertainty(
                all_preds, all_sigmas
            )
        else:
            preds_original = ds.inverse_transform_labels(all_preds)
            sigmas_original = None

        epoch = trainer.current_epoch #+ 1
        is_final_epoch = (epoch == trainer.max_epochs)
        is_early_stop = bool(
            getattr(trainer, "should_stop", False) or
            getattr(getattr(trainer, "fit_loop", None), "should_stop", False)
        )

        os.makedirs(self.output_dir, exist_ok=True)
        Epoch_Plots = os.path.join(self.output_dir, "Epoch_Plots")
        ResNet_Plots = os.path.join(self.output_dir, "ResNet_Plots")
        Final_Results = os.path.join(self.output_dir, "Final_Results")
        os.makedirs(Epoch_Plots, exist_ok=True)
        os.makedirs(ResNet_Plots, exist_ok=True)
        os.makedirs(Final_Results, exist_ok=True)

        records = []

        # Convert tensors to numpy
        all_predictions_original = preds_original.cpu().numpy()
        all_targets_original = targets_original.cpu().numpy()
        all_sigmas_original = sigmas_original.cpu().numpy() if sigmas_original is not None else None

        for i, param in enumerate(self.model_params):
            y_true = all_targets_original[:, i]
            y_pred = all_predictions_original[:, i]
            y_err = all_sigmas_original[:, i] if all_sigmas_original is not None else None
            residuals = y_pred - y_true

            if (is_final_epoch or is_early_stop) and trainer.is_global_zero and not getattr(self, "_final_artifacts_done", False):
                for j in range(len(y_true)):
                    records.append({
                        "param": param,
                        "true": y_true[j],
                        "pred": y_pred[j],
                        "residual": residuals[j],
                        "sigma": y_err[j] if y_err is not None else np.nan
                    })

            # --- plotting (unchanged, just using y_true/y_pred/y_err) ---
            plt.figure(figsize=(7, 7))
            if param in self.log_scale_params:
                plt.xscale('log'); plt.yscale('log')
                plt.title(f"Epoch {epoch} - {param} (Log-Log Scale)")
                # Filter valid data
                mask = (y_true > 0) & (y_pred > 0)

                if np.any(mask):
                    if pl_module.use_mdn:
                        plt.errorbar(y_true[mask], y_pred[mask], yerr=y_err[mask], fmt='o', alpha=0.3, markersize=4, capsize=2)
                        minval = min(y_true[mask].min(), y_pred[mask].min())
                        maxval = max(y_true[mask].max(), y_pred[mask].max())
                        plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
                    else:
                        plt.scatter(y_true[mask], y_pred[mask], alpha=0.3, s=10)
                        minval = min(y_true[mask].min(), y_pred[mask].min())
                        maxval = max(y_true[mask].max(), y_pred[mask].max())
                        plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
                    plt.xlim(minval, maxval)
                    plt.ylim(minval, maxval)
                else:
                    print(f"Warning: No positive data for log-log plot of {param}")
                    plt.text(0.5, 0.5, 'No valid data', transform=plt.gca().transAxes,
                            ha='center', va='center')
                    plt.xlim(1e-9, 1)
                    plt.ylim(1e-9, 1)
            else:
                # Linear plot
                plt.title(f"Epoch {epoch+1} - {param} (Linear Scale)")
                if pl_module.use_mdn:
                    plt.errorbar(y_true, y_pred, yerr=y_err, fmt='o', alpha=0.3, markersize=4, capsize=2)
                    minval = min(y_true.min(), y_pred.min())
                    maxval = max(y_true.max(), y_pred.max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
                else:
                    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
                    minval = min(y_true.min(), y_pred.min())
                    maxval = max(y_true.max(), y_pred.max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')


            plt.xlabel(f"True {param}")
            plt.ylabel(f"Predicted {param}")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()

            plot_filename = os.path.join(Epoch_Plots, f"E{epoch}_Test_{param}.png")
            print("Saving plot:", plot_filename)
            plt.savefig(plot_filename, dpi=150)
            plt.close()

        # Final artifacts once
        if (is_final_epoch or is_early_stop) and trainer.is_global_zero and not getattr(self, "_final_artifacts_done", False):
            self._final_artifacts_done = True

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

            print("\nCoverage within ±1σ:")
            for name, ratio in coverage_ratios:
                print(f"  {name}: {ratio * 100:.2f}%")

            if sigmas_original is not None:
                print("Evaluating calibration and plotting curves...")
                _ = evaluate_calibration(
                    mu=preds_original,
                    sigma=sigmas_original,
                    y_true=targets_original,
                    param_names=self.model_params,
                    epoch=epoch,
                    folder_name=self.output_dir
                )

            # CSV
            df = pd.DataFrame()
            preds_np = preds_original.cpu().numpy()
            targets_np = targets_original.cpu().numpy()
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
