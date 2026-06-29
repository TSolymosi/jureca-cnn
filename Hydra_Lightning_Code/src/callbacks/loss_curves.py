from typing import Dict, List
from pathlib import Path
import re
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class LossCurvesCallback(Callback):
    def __init__(
        self,
        output_dir,
        fname_overall="loss_curves.png",
        fname_perparam="val_per_param_loss.png",
    ):
        self.output_dir = Path(output_dir)
        self.fname_overall = fname_overall
        self.fname_perparam = fname_perparam

        self.history = {
            "epoch": [],
            "train/loss": [],
            "val/loss": [],
        }
        self.per_param: Dict[str, List[float]] = {}

        self.loss_file: Path | None = None

    # ------------------------------------------------------------------
    # FIT START: resume-safe loading
    # ------------------------------------------------------------------
    
    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Available hparams:", pl_module.hparams)


        write_job_id = getattr(pl_module.hparams, "job_id", None)
        load_job_id  = getattr(pl_module.hparams, "load_id", None)

        if write_job_id is None:
            raise RuntimeError("job_id must be provided via Hydra (+job_id=...)")

        self.loss_file = self.output_dir / f"loss_history_job_{write_job_id}.npz"

        # ------------------------------------------------------------
        # Case 1: continuing from a *different* job
        # ------------------------------------------------------------
        from pathlib import Path
        import shutil

        if load_job_id and load_job_id != write_job_id:

            # Resolve directories robustly
            parent_dir = self.output_dir.resolve().parent
            src_dir = parent_dir / str(load_job_id)
            src = src_dir / f"loss_history_job_{load_job_id}.npz"

            if not src.is_file():
                raise RuntimeError(
                    f"Loss history for job {load_job_id} not found at: {src}"
                )

            # Ensure destination directory exists
            self.loss_file.parent.mkdir(parents=True, exist_ok=True)

            # Only copy if destination does not already exist
            if not self.loss_file.exists():
                shutil.copy2(src, self.loss_file)



        # ------------------------------------------------------------
        # Case 2: resume or fresh start
        # ------------------------------------------------------------
        if self.loss_file.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = self.loss_file.with_suffix(f".bak_{ts}.npz")
            backup.write_bytes(self.loss_file.read_bytes())

            data = np.load(self.loss_file, allow_pickle=True)

            self.history["epoch"] = data["epoch"].tolist()
            self.history["train/loss"] = data["train_loss"].tolist()
            self.history["val/loss"] = data["val_loss"].tolist()

            for k in data.files:
                if k.startswith("val_param_"):
                    pname = k.replace("val_param_", "")
                    self.per_param[pname] = data[k].tolist()


    # ------------------------------------------------------------------
    # VALIDATION EPOCH END: update + persist
    # ------------------------------------------------------------------
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        epoch = int(trainer.current_epoch)

        train_loss = (
            m.get("train/loss")
            or m.get("train/loss_epoch")
            or m.get("train_loss")
            or m.get("train_loss_epoch")
        )
        val_loss = (
            m.get("val/loss")
            or m.get("val/loss_epoch")
            or m.get("val_loss")
            or m.get("val_loss_epoch")
        )

        # overwrite-or-append logic
        if epoch in self.history["epoch"]:
            i = self.history["epoch"].index(epoch)
            self.history["train/loss"][i] = float(train_loss) if train_loss is not None else np.nan
            self.history["val/loss"][i] = float(val_loss) if val_loss is not None else np.nan
        else:
            self.history["epoch"].append(epoch)
            self.history["train/loss"].append(float(train_loss) if train_loss is not None else np.nan)
            self.history["val/loss"].append(float(val_loss) if val_loss is not None else np.nan)

        # --------------------------------------------------------------
        # per-parameter validation loss
        # --------------------------------------------------------------
        pat_nll = re.compile(r"^val/([A-Za-z0-9_]+)_nll$")
        pat_loss = re.compile(r"^val/([A-Za-z0-9_]+)_loss$")

        found = {}

        for k, v in m.items():
            mobj = pat_nll.match(k) or pat_loss.match(k)
            if not mobj:
                continue
            pname = mobj.group(1)
            found[pname] = float(v)

        for pname, value in found.items():
            if pname not in self.per_param:
                self.per_param[pname] = [np.nan] * len(self.history["epoch"])
            idx = self.history["epoch"].index(epoch)
            self.per_param[pname][idx] = value

        # pad missing params
        for pname in self.per_param:
            if len(self.per_param[pname]) < len(self.history["epoch"]):
                self.per_param[pname].append(np.nan)

        self._write_npz()

    # ------------------------------------------------------------------
    # FILE WRITE (atomic)
    # ------------------------------------------------------------------
    def _write_npz(self):
        tmp = self.loss_file.with_suffix(".tmp.npz")
        arrays = {
            "epoch": np.asarray(self.history["epoch"]),
            "train_loss": np.asarray(self.history["train/loss"]),
            "val_loss": np.asarray(self.history["val/loss"]),
        }
        for pname, series in self.per_param.items():
            arrays[f"val_param_{pname}"] = np.asarray(series)

        np.savez(tmp, **arrays)
        tmp.replace(self.loss_file)

    # ------------------------------------------------------------------
    # FIT END: plotting from disk-backed state
    # ------------------------------------------------------------------
    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        if not self.history["epoch"]:
            return

        epochs = np.asarray(self.history["epoch"])
        train = np.asarray(self.history["train/loss"])
        val = np.asarray(self.history["val/loss"])

        # overall
        plt.figure()
        plt.plot(epochs, np.clip(train, None, 10), label="train")
        plt.plot(epochs, np.clip(val, None, 10), label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / self.fname_overall, dpi=180)
        plt.close()

        # per-parameter
        if self.per_param:
            plt.figure()
            for pname, series in sorted(self.per_param.items()):
                plt.plot(epochs, series, label=pname)
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.legend(ncol=2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / self.fname_perparam, dpi=180)
            plt.close()
