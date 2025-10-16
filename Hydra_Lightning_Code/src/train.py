import sys
import os
import inspect

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
    #datamodule.setup() # This is typically called automatically by the Trainer, so manual calls are often not needed.

    # --- Get dataset properties for the model ---
    # This section was likely used to pass dataset-specific properties (like standard deviation for weighting)
    # to the model. It's currently disabled, and the model now fetches this information internally
    # from the datamodule in the `on_fit_start` hook.
    #dataset_ref = datamodule.get_dataset_reference()
    #std_weights = dataset_ref.scaler_stds
    std_weights = None

    # --- Instantiate Model ---
    # The `LitResNetMDN` LightningModule is instantiated.
    # It receives separate configuration objects for the model architecture (`cfg.model`),
    # the optimizer (`cfg.optim`), and the trainer settings (`cfg.trainer`).
    model = LitResNetMDN(
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        training_cfg=cfg.trainer,
        std_weights=std_weights
    )

    # --- START OF NEW DIAGNOSTIC BLOCK ---
    # We will add this check to get the ground truth about the model object.
    

    @rank_zero_only
    def run_diagnostics(model_object):
        print("\n" + "="*80)
        print(" " * 25 + "RUNTIME OBJECT DIAGNOSTICS")
        print("="*80)
        
        # 1. Find out which file this class was actually loaded from.
        try:
            class_file = inspect.getfile(model_object.__class__)
            print(f"[INFO] The 'LitResNetMDN' class was loaded from file:")
            print(f"       -> {class_file}")
        except Exception as e:
            print(f"[ERROR] Could not determine the source file: {e}")
        
        print("-" * 80)
        
        # 2. Inspect the signature of the on_validation_epoch_end method on the live object.
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
    # Callbacks are objects that can perform actions at various stages of training (e.g., at the end of an epoch).
    # This loop dynamically instantiates all callbacks defined in the `cfg.callbacks` section of the config.
    callback_list = []
    for name, cb_cfg in cfg.callbacks.items():
        # `_target_` is a special key in Hydra configs that specifies the Python class to instantiate.
        target = cb_cfg.get("_target_", "")
        print_rank0(f"Instantiating callback: {name} with target {target}")

        # This is a special case. The PredictionPlotCallback needs access to `model_params` and `log_scale_params`
        # which are defined in the data section of the config. This code block manually injects them during
        # instantiation, as the callback doesn't have direct access to `cfg.data`.
        if "PredictionPlotCallback" in target:
            cb = instantiate(
                cb_cfg,
                model_params=cfg.data.model_params,
                log_scale_params=cfg.data.log_scale_params
            )
        else:
            # For all other callbacks, instantiate them directly from their config.
            cb = instantiate(cb_cfg)
        callback_list.append(cb)

    # Log the list of active callbacks on the main process.
    rank_zero_info(f"Using callbacks: {[type(cb).__name__ for cb in callback_list]}")

    # --- Instantiate Trainer ---
    # The PyTorch Lightning `Trainer` orchestrates the entire training process.
    # It is configured using the instantiated callbacks and the settings from `cfg.trainer`.
    # `num_sanity_val_steps=0` disables the initial validation sanity check.
    trainer = Trainer(
        callbacks=callback_list,
        **cfg.trainer,
        num_sanity_val_steps=0,
    )

    # --- Run Training, Validation, or Testing ---
    # This logic allows you to control the trainer's behavior from the command line or config file.
    ckpt_path = cfg.get("ckpt_path", None)  # Path to a specific checkpoint to load.
    mode = cfg.get("mode", "train")          # The desired mode: "train", "validate", or "test".
    load_id = cfg.get("load_id", None)     # Get the load_id from the config

    if ckpt_path is None and load_id is not None:
        # If a load_id is provided but not a direct ckpt_path, construct the path.
        # This assumes your outputs are saved in a standard Hydra output directory.
        # We look for the 'last.ckpt' file, which is saved by the ModelCheckpoint callback.
        
        # The path depends on your hydra output structure. It's often 'outputs/job_name/job_id/'.
        # Assuming folder_name is used, the structure would be 'outputs/folder_name/load_id/'.
        folder_name = cfg.get("folder_name", "default_folder")
        
        potential_path = os.path.join("outputs", folder_name, str(load_id), "checkpoints", "last.ckpt")
        
        if os.path.exists(potential_path):
            ckpt_path = potential_path
            rank_zero_info(f"Found checkpoint for load_id '{load_id}' at: {ckpt_path}")
        else:
            rank_zero_info(f"WARNING: load_id '{load_id}' was provided, but no checkpoint found at '{potential_path}'. Starting from scratch.")

    if mode == "train":
        # Starts the main training and validation loop. Can resume from `ckpt_path` if provided.
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        # You could optionally run a final validation on the best saved checkpoint after training.
        # trainer.validate(model, datamodule=datamodule, ckpt_path="best")
    elif mode == "validate":
        # Runs a single validation epoch using the provided checkpoint.
        trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif mode == "test":
        # Runs a single test epoch using the provided checkpoint.
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # The return value is often used by Hydra's hyperparameter optimization plugins (sweepers).
    # It reports the final validation loss as the metric for evaluating the success of a run.
    return trainer.callback_metrics["val/loss"].item()

if __name__ == "__main__":
    # This is the standard entry point for a Python script.
    # It calls the `main` function, which is decorated by and controlled by Hydra.
    main()