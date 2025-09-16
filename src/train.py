import hydra
from hydra.utils import instantiate

import torch
torch.set_float32_matmul_precision("high")  # or "medium"
# optional, often good too:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method was already set (safe to ignore)
    pass

import os, pathlib, tempfile
os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(os.getenv("TMPDIR", tempfile.gettempdir())) / "mpl"))
import matplotlib
matplotlib.use("Agg")

from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs)

from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from src.models.lightning_resnet import LitResNetMDN
from src.data.fits_datamodule import FitsDataModule
from src.callbacks.prediction_plot_callback import PredictionPlotCallback





@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from hydra.core.plugins import Plugins
    from hydra.plugins.sweeper import Sweeper

    print("Available sweepers:", Plugins.instance().discover(Sweeper), flush=True)
    # --- Reproducibility ---
    seed_everything(42, workers=True)

    # --- Instantiate datamodule ---
    datamodule = FitsDataModule(cfg.data)
    #datamodule.setup()

    # --- Pull dataset reference for label scalers ---
    #dataset_ref = datamodule.get_dataset_reference()
    #std_weights = dataset_ref.scaler_stds  # For MDN loss weighting
    std_weights = None

    # --- Instantiate model ---
    model = LitResNetMDN(
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        training_cfg=cfg.trainer,
        std_weights=std_weights
    )

    # --- Callbacks ---
    # Instantiate callbacks via Hydra, injecting missing arguments manually
    callback_list = []
    for name, cb_cfg in cfg.callbacks.items():
        target = cb_cfg.get("_target_", "")
        
        print_rank0(f"Instantiating callback: {name} with target {target}")

        if "PredictionPlotCallback" in target:
            cb = instantiate(
                cb_cfg,
                model_params=cfg.data.model_params,
                log_scale_params=cfg.data.log_scale_params
            )
        else:
            cb = instantiate(cb_cfg)

        callback_list.append(cb)


    rank_zero_info(f"Using callbacks: {[type(cb).__name__ for cb in callback_list]}")

    trainer = Trainer(
        callbacks=callback_list,
        **cfg.trainer,
        num_sanity_val_steps=0,
    )

    # --- right before calling trainer.fit(...) ---
    ckpt_path = cfg.get("ckpt_path", None)        # path to a .ckpt file
    mode      = cfg.get("mode", "train")          # "train" | "validate" | "test"

    if mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.validate(model, datamodule=datamodule)  # optional
    elif mode == "validate":
        trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif mode == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # --- Fit ---
    trainer.fit(model, datamodule=datamodule)

    # Optional: test if defined
    trainer.validate(model, datamodule=datamodule)
    return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    main()
