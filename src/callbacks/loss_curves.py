from typing import Dict, List
from pathlib import Path
import re
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
import os
from lightning.pytorch.utilities import rank_zero_only

class LossCurvesCallback(Callback):
    def __init__(self, fname_overall="loss_curves.png", fname_perparam="val_per_param_loss.png", output_dir=None):
        self.fname_overall = fname_overall
        self.fname_perparam = fname_perparam
        self.output_dir = output_dir
        self.history = {
            "epoch": [],
            "train/loss": [],
            "val/loss": [],
        }
        self.per_param: Dict[str, List[float]] = {}  # e.g. {"D": [..], "L":[..]}

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        epoch = trainer.current_epoch

        # be flexible about key names
        train_loss = m.get("train/loss") or m.get("train/loss_epoch") or m.get("train_loss") or m.get("train_loss_epoch")
        val_loss   = m.get("val/loss")   or m.get("val/loss_epoch")   or m.get("val_loss")   or m.get("val_loss_epoch")

        # always append epoch; fill missing with NaN
        self.history["epoch"].append(int(epoch))
        self.history["train/loss"].append(float(train_loss) if train_loss is not None else float("nan"))
        self.history["val/loss"].append(float(val_loss) if val_loss is not None else float("nan"))

        # per-parameter (unchanged, but keeps padding logic)
        import re
        pat = re.compile(r"^val/([A-Za-z0-9_]+)_loss$")
        for k, v in m.items():
            mobj = pat.match(k)
            if not mobj:
                continue
            pname = mobj.group(1)
            if pname not in self.per_param:
                self.per_param[pname] = [float("nan")] * (len(self.history["epoch"]) - 1)
            self.per_param[pname].append(float(v))
        for pname in self.per_param:
            if len(self.per_param[pname]) < len(self.history["epoch"]):
                self.per_param[pname].append(float("nan"))

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        # resolve output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # overall train vs val
        print(self.history)
        if len(self.history["epoch"]) > 0:
            plt.figure()
            plt.plot(self.history["epoch"], self.history["train/loss"], label="train/loss")
            plt.plot(self.history["epoch"], self.history["val/loss"], label="val/loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train vs Val Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            print("Saving overall loss curve to:", os.path.join(self.output_dir, self.fname_overall))
            plt.savefig(os.path.join(self.output_dir, self.fname_overall), dpi=180)
            plt.close()

        # validation loss per parameter
        if self.per_param:
            plt.figure()
            for pname, series in sorted(self.per_param.items()):
                plt.plot(self.history["epoch"], series, label=f"{pname}")
            plt.xlabel("Epoch")
            plt.ylabel("Val Loss")
            plt.title("Validation Loss per Parameter")
            plt.legend(ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            print("Saving per-parameter loss curves to:", os.path.join(self.output_dir, self.fname_perparam))
            plt.savefig(os.path.join(self.output_dir, self.fname_perparam), dpi=180)
            plt.close()


        

