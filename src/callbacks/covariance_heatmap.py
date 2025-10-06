import os, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

class CovarianceHeatmapCallback(Callback):
    def __init__(self, param_names, output_dir="./pred_plots"):
        super().__init__()
        self.param_names = list(param_names)
        self.output_dir = output_dir

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # plot every 5 epochs
        if (trainer.current_epoch + 1) % 5 != 0:
            return
            
        cache = getattr(pl_module, "_val_cache", {})
        Ls = cache.get("Ls", None)
        if Ls is None or (not torch.is_tensor(Ls)) or Ls.numel() == 0:
            return

        # Ls shape: (N, d, d)
        with torch.no_grad():
            L = Ls.float()
            Sigma = torch.matmul(L, L.transpose(-1, -2))  # (N, d, d)
            Sigma_mean = Sigma.mean(dim=0).cpu().numpy()  # (d, d)
            d = Sigma_mean.shape[0]
            std_z = np.sqrt(np.clip(np.diag(Sigma_mean), 1e-12, None))
            denom = np.outer(std_z, std_z)
            Corr = Sigma_mean / denom
            Corr = np.clip(Corr, -1.0, 1.0)

        outdir = os.path.join(self.output_dir, "Covariance_Matrices")
        os.makedirs(outdir, exist_ok=True)
        epoch = trainer.current_epoch + 1

        def _plot(mat, title, fname, vmin=None, vmax=None, cmap="coolwarm"):
            plt.figure(figsize=(1.1*d, 1.0*d))
            im = plt.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xticks(range(d), self.param_names, rotation=45, ha="right")
            plt.yticks(range(d), self.param_names)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=160)
            plt.close()

        print(f"Saving covariance/correlation matrices to {outdir}", flush=True)

        # 1) Covariance (standardized space)
        vmax = np.max(np.abs(Sigma_mean))
        _plot(Sigma_mean, f"Mean covariance (z-space) — Epoch {epoch}",
              f"E{epoch}_covariance.png", vmin=-vmax, vmax=vmax)

        # 2) Correlation
        _plot(Corr, f"Correlation matrix — Epoch {epoch}",
              f"E{epoch}_correlation.png", vmin=-1.0, vmax=1.0)

        # ---- EXTRA 1: save std_z and top-|corr| pairs ----
        # std_z CSV: one row, columns are param_names (in order)
        std_csv = os.path.join(outdir, f"E{epoch}_std_z.csv")
        np.savetxt(std_csv, std_z.reshape(1, -1), delimiter=",",
                   header=",".join(self.param_names), comments="")
        # top-|corr| pairs TXT
        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                pairs.append((abs(Corr[i, j]), i, j, Corr[i, j]))
        pairs.sort(reverse=True)
        top_path = os.path.join(outdir, f"E{epoch}_top_corr.txt")
        with open(top_path, "w") as f:
            for _, i, j, c in pairs[:10]:
                f.write(f"{self.param_names[i]:>14s} ↔ {self.param_names[j]:<14s}: {c:+.3f}\n")
        # quick console summary
        top3 = [(self.param_names[i], self.param_names[j], float(c)) for _, i, j, c in pairs[:3]]
        print(f"[CovHeatmap] std_z: "
              + ", ".join([f"{n}={v:.3f}" for n, v in zip(self.param_names, std_z)]), flush=True)
        print(f"[CovHeatmap] top corr: "
              + ", ".join([f"{a}↔{b}:{c:+.3f}" for a,b,c in top3]), flush=True)

        # ---- EXTRA 2: physical-unit covariance using label stds ----
        # Try to get stds from the LightningModule (set in on_fit_start)
        std_weights = None
        try:
            sw = getattr(pl_module, "std_weights", None)
            if sw is not None:
                sw = torch.as_tensor(sw).detach().cpu().numpy().astype(np.float64)
                if sw.shape[0] == d:
                    std_weights = sw
        except Exception:
            std_weights = None

        # Fallback: try datamodule.dataset_ref.scaler_stds
        if std_weights is None:
            try:
                ds = getattr(trainer.datamodule, "dataset_ref", None)
                sw = getattr(ds, "scaler_stds", None)
                if sw is not None:
                    sw = torch.as_tensor(sw).detach().cpu().numpy().astype(np.float64)
                    if sw.shape[0] == d:
                        std_weights = sw
            except Exception:
                std_weights = None

        # if std_weights is not None:
        #     S = np.diag(std_weights)                  # diag of label stds (physical units)
        #     Sigma_phys = S @ Sigma_mean @ S           # mean covariance in physical units
        #     vmax_p = np.max(np.abs(Sigma_phys))
        #     _plot(Sigma_phys, f"Mean covariance (physical units) — Epoch {epoch}",
        #           f"E{epoch}_covariance_phys.png", vmin=-vmax_p, vmax=vmax_p)
        # else:
        #     print("[CovHeatmap] WARN: could not retrieve label stds; skipping physical-unit covariance.", flush=True)
