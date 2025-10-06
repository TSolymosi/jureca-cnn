# src/callbacks/prediction_plot_callback.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class PredictionPlotCallback(Callback):
    def __init__(self, model_params, log_scale_params, output_dir="./pred_plots"):
        super().__init__()
        self.model_params = list(model_params)
        self.log_scale_params = set(log_scale_params)
        self.output_dir = output_dir
        self._final_artifacts_done = False

    # ---------- helpers ----------
    def _to_tensor(self, x):
        """Accept list[Tensor] or Tensor; return concatenated Tensor (or None)."""
        if x is None:
            return None
        if isinstance(x, list):
            if len(x) == 0:
                return None
            if torch.is_tensor(x[0]):
                return torch.cat(x, dim=0)
            return None
        return x

    def _get_cache_and_ds(self, split, pl_module, trainer):
        if split == "Val":
            cache = getattr(pl_module, "_val_cache", {})
            ds = getattr(pl_module, "_val_dataset", None)
        else:  # "Test"
            cache = getattr(pl_module, "_test_cache", {})
            ds = getattr(pl_module, "_test_dataset", None)
            if ds is None:
                # fall back to datamodule.dataset_ref if module didn't set *_dataset
                dm = getattr(trainer, "datamodule", None)
                ds = getattr(dm, "dataset_ref", None) if dm is not None else None
        return cache, ds

    def _maybe_get_sigmas_from_L(self, all_sigmas, cache):
        """If per-dim sigma missing, derive diag(Σ)=sum(L^2) from L or Ls."""
        if all_sigmas is not None:
            return all_sigmas
        L = self._to_tensor(cache.get("L", None)) or self._to_tensor(cache.get("Ls", None)) \
            or self._to_tensor(cache.get("L_all", None))
        if L is None:
            return None
        return torch.sqrt((L ** 2).sum(dim=-1).clamp_min(1e-12))

    def _run_split(self, *, split, trainer, pl_module):
        pl_module.eval()

        cache, ds = self._get_cache_and_ds(split, pl_module, trainer)

        all_preds   = self._to_tensor(cache.get("preds", None))
        all_targets = self._to_tensor(cache.get("targets", None))
        all_sigmas  = self._to_tensor(cache.get("sigmas", None))
        all_sigmas  = self._maybe_get_sigmas_from_L(all_sigmas, cache)

        if (ds is None) or (all_preds is None) or (all_targets is None):
            print(f"[PredictionPlotCallback] No dataset or cached tensors for {split}. Skipping.")
            return
        if hasattr(all_preds, "numel") and all_preds.numel() == 0:
            print(f"[PredictionPlotCallback] Empty tensors for {split}. Skipping.")
            return

        # inverse transform (dataset helpers handle log10 params for you)
        targets_original = ds.inverse_transform_labels(all_targets)
        if getattr(pl_module, "use_mdn", False) and (all_sigmas is not None):
            preds_original, sigmas_original = ds.inverse_transform_labels_with_uncertainty(
                all_preds, all_sigmas
            )
        else:
            preds_original = ds.inverse_transform_labels(all_preds)
            sigmas_original = None

        epoch = trainer.current_epoch

        # dirs
        os.makedirs(self.output_dir, exist_ok=True)
        epoch_dir = os.path.join(self.output_dir, "Epoch_Plots")
        resid_dir = os.path.join(self.output_dir, "ResNet_Plots")
        final_dir = os.path.join(self.output_dir, "Final_Results")
        for d in (epoch_dir, resid_dir, final_dir):
            os.makedirs(d, exist_ok=True)

        # numpy arrays for plotting
        y_pred_np = preds_original.detach().cpu().numpy()
        y_true_np = targets_original.detach().cpu().numpy()
        y_sig_np  = sigmas_original.detach().cpu().numpy() if sigmas_original is not None else None

        # --- per-parameter scatter ---
        for i, param in enumerate(self.model_params):
            yt = y_true_np[:, i]
            yp = y_pred_np[:, i]
            ye = y_sig_np[:, i] if y_sig_np is not None else None

            plt.figure(figsize=(7, 7))
            loglog = param in self.log_scale_params
            if loglog:
                plt.xscale("log"); plt.yscale("log")
                plt.title(f"Epoch {epoch+1} - {param} ({split}, Log-Log)")
                mask = np.isfinite(yt) & np.isfinite(yp) & (yt > 0) & (yp > 0)
            else:
                plt.title(f"Epoch {epoch+1} - {param} ({split})")
                mask = np.isfinite(yt) & np.isfinite(yp)

            if np.any(mask):
                yt, yp = yt[mask], yp[mask]
                minval = min(yt.min(), yp.min())
                maxval = max(yt.max(), yp.max())
                if loglog and minval <= 0:
                    minval = np.nextafter(0.0, 1.0)

                if (getattr(pl_module, "use_mdn", False)) and (ye is not None):
                    ye = ye[mask]
                    lower = np.maximum(yp - ye, minval * 0.999 if loglog else -np.inf)
                    upper = yp + ye
                    yerr_plot = np.vstack([yp - lower, upper - yp])
                    plt.errorbar(yt, yp, yerr=yerr_plot, fmt="o", alpha=0.3, markersize=4, capsize=2)
                else:
                    plt.scatter(yt, yp, alpha=0.3, s=10)

                plt.plot([minval, maxval], [minval, maxval], "r--", label="y = x")
                plt.xlim(minval, maxval); plt.ylim(minval, maxval)
            else:
                plt.text(0.5, 0.5, "No valid data", transform=plt.gca().transAxes,
                         ha="center", va="center")
                if loglog:
                    plt.xlim(1e-9, 1); plt.ylim(1e-9, 1)

            plt.xlabel(f"True {param}"); plt.ylabel(f"Predicted {param}")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            out_png = os.path.join(epoch_dir, f"E{epoch+1}_{split}_{param}.png")
            plt.savefig(out_png, dpi=150); plt.close()
            print("Saved plot:", out_png, flush=True)

        # --- final artifacts ---
        is_final_epoch = (epoch + 1 == getattr(trainer, "max_epochs", epoch + 1))
        is_early_stop = bool(getattr(trainer, "should_stop", False)
                             or getattr(getattr(trainer, "fit_loop", None), "should_stop", False))
        force_final = (split == "Test")

        if (force_final or is_final_epoch or is_early_stop) and trainer.is_global_zero and not self._final_artifacts_done:
            self._final_artifacts_done = True

            # Residuals
            residual = preds_original - targets_original

            if sigmas_original is not None:
                for i, name in enumerate(self.model_params):
                    plt.figure(figsize=(6, 5))
                    plt.scatter(sigmas_original[:, i].cpu().numpy(),
                                residual[:, i].cpu().numpy(), alpha=0.3, s=10)
                    plt.xlabel('Predicted σ'); plt.ylabel('Residual (μ - y)')
                    plt.title(f'Residual vs σ: {name} ({split})')
                    plt.grid(True); plt.tight_layout()
                    plt.savefig(os.path.join(resid_dir, f"{name}_residual_{split}.png"))
                    plt.close()

                # Coverage printout
                within_1sigma = ((preds_original > targets_original - sigmas_original) &
                                 (preds_original < targets_original + sigmas_original)).float().mean().item()
                print(f"[PredictionPlotCallback] Overall ±1σ coverage ({split}): {within_1sigma*100:.2f}%")

            # CSV (wide)
            df = pd.DataFrame()
            preds_np   = preds_original.cpu().numpy()
            targets_np = targets_original.cpu().numpy()
            sigmas_np  = sigmas_original.cpu().numpy() if sigmas_original is not None else None

            for i, param_name in enumerate(self.model_params):
                df[f"true_{param_name}"] = targets_np[:, i]
                df[f"pred_{param_name}"] = preds_np[:, i]
                if sigmas_np is not None:
                    df[f"sigma_{param_name}"] = sigmas_np[:, i]

            csv_path = os.path.join(final_dir, "final_predictions.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved full predictions CSV at: {csv_path}")

    # ---------- Lightning hooks ----------
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._run_split(split="Val", trainer=trainer, pl_module=pl_module)

    @rank_zero_only
    def on_test_end(self, trainer, pl_module):
        # Produce final artifacts after model has finished caching
        self._run_split(split="Test", trainer=trainer, pl_module=pl_module)


class CacheResetCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        cache = getattr(pl_module, "_val_cache", None)
        if isinstance(cache, dict):
            for k, v in list(cache.items()):
                if isinstance(v, list):
                    v.clear()
            cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CacheResetCallback] Cleared validation cache and freed memory.")

    def on_test_end(self, trainer, pl_module):
        # Optional: free test cache too after artifacts are written
        cache = getattr(pl_module, "_test_cache", None)
        if isinstance(cache, dict):
            for k, v in list(cache.items()):
                if isinstance(v, list):
                    v.clear()
            cache.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CacheResetCallback] Cleared test cache and freed memory.")
