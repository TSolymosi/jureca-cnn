import sys
import os
import inspect
import re

# --- Force Python to find the 'src' package correctly ---
# Get the absolute path of the current script (train.py)
# e.g., /p/scratch/westai0043/CNN_HL_tobias/src/train.py
script_path = os.path.abspath(__file__)

# Get the directory containing the script (the 'src' folder)
# e.g., /p/scratch/westai0043/CNN_HL_tobias/src
src_dir = os.path.dirname(script_path)

# Get the project's root directory (one level up from 'src')
# e.g., /p/scratch/westai0043/CNN_HL_tobias
project_root = os.path.dirname(src_dir)

# Insert the project root at the beginning of the Python path.
# This ensures that when 'import src.callbacks' is called,
# it finds the 'src' folder within our project first.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"--- INFO: Added project root to Python path: {project_root} ---")

import hydra
from hydra.utils import instantiate

import torch
# These settings can significantly speed up training on modern NVIDIA GPUs (Ampere architecture and newer)
# by using the TensorFloat-32 (TF32) format for matrix multiplications.
torch.set_float32_matmul_precision("high")  # or "medium"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.multiprocessing as mp
try:
    # 'spawn' is a safe way to start new processes for data loading (num_workers > 0).
    # It's often necessary in CUDA environments to avoid deadlocks or initialization errors.
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # This will be raised if the start method has already been set, which is safe to ignore.
    pass

import os, pathlib, tempfile
# This is a workaround for environments where the default Matplotlib configuration directory
# might not be writable (e.g., in some cluster or container environments).
# It redirects Matplotlib's config to a temporary directory.
os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path(os.getenv("TMPDIR", tempfile.gettempdir())) / "mpl"))
import matplotlib
# Sets the Matplotlib backend to "Agg". This is a non-interactive backend that saves plots
# to files instead of trying to display them on a screen. It's essential for running on servers without a GUI.
matplotlib.use("Agg")



from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def print_rank0(*args, **kwargs):
    """A helper function to ensure that print statements are only executed on the main
    process (rank 0) in a multi-GPU setup, preventing duplicate logging."""
    print(*args, **kwargs)

from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_info




# --- Import the custom modules from your project's source directory ---
from src.models.lightning_resnet import LitResNetMDN
from src.data.fits_datamodule import FitsDataModule
from src.callbacks.prediction_plot_callback import PredictionPlotCallback

#from src.callbacks.two_stage_callback import TwoStageMDNTraining


# The `@hydra.main` decorator turns this function into a Hydra-configurable application.
# - `config_path`: Specifies the directory where your .yaml configuration files are located.
# - `config_name`: Specifies the main configuration file to load.
# - `version_base`: Manages compatibility with Hydra's evolving configuration standards.
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """
    The main training function, orchestrated by Hydra.

    Args:
        cfg (DictConfig): A dictionary-like object created by Hydra, containing the
                          entire configuration from your .yaml files.
    """
    from hydra.core.plugins import Plugins
    from hydra.plugins.sweeper import Sweeper
    
    # This line is for debugging and shows which Hydra "sweeper" plugins (for hyperparameter optimization) are available.
    print("Available sweepers:", Plugins.instance().discover(Sweeper), flush=True)

    # --- Reproducibility ---
    # `seed_everything` is a PyTorch Lightning utility that sets random seeds for Python,
    # NumPy, and PyTorch to ensure that runs are reproducible.
    seed_everything(42, workers=True)

    # --- Instantiate DataModule ---
    # The `FitsDataModule` is instantiated using the configuration found in `cfg.data`.
    # Hydra automatically maps the key-value pairs in the YAML to the arguments of the class `__init__`.
    datamodule = FitsDataModule(cfg.data)

    # --- Get dataset properties for the model ---
    # This is handled automatically by the model's `on_fit_start` hook now.
    std_weights = None

    # --- Instantiate Model ---
    # The `LitResNetMDN` LightningModule is instantiated.
    # It receives separate configuration objects for the model architecture (`cfg.model`),
    # the optimizer (`cfg.optim`), and the trainer settings (`cfg.trainer`).
    model = LitResNetMDN(
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        training_cfg=cfg.trainer,
        strategy_cfg=cfg.training_strategy,
        std_weights=std_weights,
        job_id=cfg.get("job_id", None),
        load_id=cfg.get("load_id", None),
    )

    # --- START OF NEW DIAGNOSTIC BLOCK ---
    @rank_zero_only
    def run_diagnostics(model_object):
        print("\n" + "="*80)
        print(" " * 25 + "RUNTIME OBJECT DIAGNOSTICS")
        print("="*80)
        
        try:
            class_file = inspect.getfile(model_object.__class__)
            print(f"[INFO] The 'LitResNetMDN' class was loaded from file:")
            print(f"       -> {class_file}")
        except Exception as e:
            print(f"[ERROR] Could not determine the source file: {e}")
        
        print("-" * 80)
        
        try:
            hook_method = model_object.on_validation_epoch_end
            signature = inspect.getfullargspec(hook_method)
            print("[INFO] Signature of the 'on_validation_epoch_end' method FOUND:")
            print(f"       -> args = {signature.args}")
        except Exception as e:
            print(f"[ERROR] Could not inspect the hook method: {e}")

        print("="*80 + "\n", flush=True)

    run_diagnostics(model)
    # --- END OF NEW DIAGNOSTIC BLOCK ---

    # --- Instantiate Callbacks ---
    callback_list = []

    

    # Callbacks are objects that can perform actions at various stages of training (e.g., at the end of an epoch).
    # This loop dynamically instantiates all callbacks defined in the `cfg.callbacks` section of the config.
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

    # --- Instantiate Trainer ---
    # The PyTorch Lightning `Trainer` orchestrates the entire training process.
    trainer = Trainer(
        callbacks=callback_list,
        **cfg.trainer,
        num_sanity_val_steps=0,
        sync_batchnorm=True,
    )

    # --- Run Training, Validation, or Testing ---
    ckpt_path = cfg.get("ckpt_path", None)
    mode = cfg.get("mode", "train")
    load_id = cfg.get("load_id", None)
    load_option = cfg.get("load_option", "best")  # default: best

    folder_name = cfg.get("folder_name", "default_folder")
    ckpt_dir = os.path.join("outputs", folder_name, str(load_id), "checkpoints")


    

    if ckpt_path is None and load_id is not None:
        

        if load_option == "last":
            last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
                rank_zero_info(f"Loading last checkpoint: {ckpt_path}")
            else:
                rank_zero_info(f"last.ckpt not found for load_id={load_id}. Starting from scratch.")
        
        elif load_option == "best":
            best_ckpt = None
            best_loss = float("inf")

            if os.path.isdir(ckpt_dir):
                for fname in os.listdir(ckpt_dir):
                    m = re.match(r"epoch_\d+-val_loss_([-+]?[0-9]*\.?[0-9]+)\.ckpt$", fname)
                    if m:
                        loss = float(m.group(1))
                        if loss < best_loss:
                            best_loss = loss
                            best_ckpt = os.path.join(ckpt_dir, fname)

            if best_ckpt:
                ckpt_path = best_ckpt
                rank_zero_info(f"Loading best checkpoint: {ckpt_path} (val_loss={best_loss})")
            else:
                last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
                if os.path.exists(last_ckpt):
                    ckpt_path = last_ckpt
                    rank_zero_info(f"No best checkpoint found. Using last: {ckpt_path}")
                else:
                    rank_zero_info(f"No checkpoint found. Starting from scratch.")

    else:
        # load_option is interpreted as explicit filename
        explicit = os.path.join(ckpt_dir, load_option)
        if os.path.exists(explicit):
            ckpt_path = explicit
            rank_zero_info(f"Loading explicit checkpoint: {ckpt_path}")
        else:
            rank_zero_info(f"Explicit checkpoint '{explicit}' not found. Starting from scratch.")

    


    if mode == "train":
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        
        rank_zero_info("\n" + "="*80)
        rank_zero_info(" " * 20 + "TRAINING COMPLETE. RUNNING FINAL EVALUATION...")
        rank_zero_info("="*80 + "\n")
        
        best_ckpt_path = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_ckpt_path = cb.best_model_path
                break
        
        if best_ckpt_path and os.path.exists(best_ckpt_path):
            rank_zero_info(f"Found best checkpoint at: {best_ckpt_path}")
            trainer.test(model, dataloaders=datamodule.val_dataloader(), ckpt_path=best_ckpt_path)
        else:
            rank_zero_info("WARNING: Could not find best checkpoint path. Skipping final test.")
        
        if trainer.is_global_zero:
            if hasattr(model, "_test_cache") and model._test_cache:
                
                final_results = model._test_cache
                dataset_ref = datamodule.get_dataset_reference()
                
                targets_original = dataset_ref.inverse_transform_labels(final_results["targets"])
                
                df_data = {}
                param_names = dataset_ref.model_params
                
                if "snr" in final_results:
                    df_data['snr'] = final_results["snr"].numpy()
                
                for i, name in enumerate(param_names):
                    df_data[f"true_{name}"] = targets_original[:, i].numpy()

                if "sigmas" in final_results and final_results["sigmas"] is not None:
                    preds_original, sigmas_original = dataset_ref.inverse_transform_labels_with_uncertainty(
                        final_results["preds"], final_results["sigmas"]
                    )
                    for i, name in enumerate(param_names):
                        df_data[f"pred_{name}"] = preds_original[:, i].numpy()
                        df_data[f"sigma_{name}"] = sigmas_original[:, i].numpy()
                else:
                    preds_original = dataset_ref.inverse_transform_labels(final_results["preds"])
                    for i, name in enumerate(param_names):
                        df_data[f"pred_{name}"] = preds_original[:, i].numpy()
                
                import pandas as pd
                df = pd.DataFrame(df_data)

                output_path = "final_predictions.csv"
                df.to_csv(output_path, index=False)
                rank_zero_info(f"Saved final predictions (including SNR) to: {os.getcwd()}/{output_path}")

            else:
                rank_zero_info("WARNING: Final evaluation did not produce a test cache. Skipping CSV save.")

    elif mode == "validate":
        trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif mode == "test":
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    return trainer.callback_metrics.get("val/loss", -1.0)

if __name__ == "__main__":
    # This is the standard entry point for a Python script.
    # It calls the `main` function, which is decorated by and controlled by Hydra.
    main()