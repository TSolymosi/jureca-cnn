from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities import rank_zero_only
from dataloader import create_dataloaders
import os

from contextlib import suppress
import torch.distributed as dist

def _ddp_barrier():
    with suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

class FitsDataModule(LightningDataModule):
    """
    DDP-safe DataModule:
      - prepare_data(): rank-0 triggers PREP (writes artifacts only, no loaders)
      - setup(): all ranks wait, then LOAD those artifacts and build loaders
    """
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self._is_setup = False

        self.train_loader = None
        self.val_loader = None
        self.dataset_ref = None

        # Convenience handles from cfg (edit names if your cfg differs)
        self.fits_dir             = data_cfg.data_dir
        self.scaling_params_path  = data_cfg.scaling_params_path   # REQUIRED path
        self.wavelength_stride    = data_cfg.get("wavelength_stride", 1)
        self.use_local_nvme       = data_cfg.get("use_local_nvme", False)
        self.load_preprocessed    = data_cfg.get("load_preprocessed", False)
        self.preprocessed_dir     = data_cfg.get("preprocessed_dir", None)
        self.batch_size           = data_cfg.get("batch_size", 32)
        self.num_workers          = data_cfg.get("num_workers", 8)
        self.model_params         = data_cfg.model_params
        self.log_scale_params     = data_cfg.log_scale_params
        self.data_subset_fraction = data_cfg.get("data_subset_fraction", 1.0)
        self.seed                 = data_cfg.get("seed", 42)

    @rank_zero_only
    def prepare_data(self):
        """
        Rank-0 only: run PREP once to write:
          - label_scaling.pt
          - split_indices.pt
          - subset_indices.pt
          - file_list.json
        """
        # Ensure parent directory exists (robust when the path is new)
        os.makedirs(os.path.dirname(self.scaling_params_path), exist_ok=True)

        # Trigger PREP (returns None by design)
        create_dataloaders(
            fits_dir=self.fits_dir,
            scaling_params_path=self.scaling_params_path,
            wavelength_stride=self.wavelength_stride,
            load_preprocessed=self.load_preprocessed,
            preprocessed_dir=self.preprocessed_dir,
            use_local_nvme=self.use_local_nvme,
            batch_size=self.batch_size,        # not used in PREP; fine to pass
            num_workers=self.num_workers,      # not used in PREP; fine to pass
            model_params=self.model_params,
            log_scale_params=self.log_scale_params,
            data_subset_fraction=self.data_subset_fraction,
            seed=self.seed,
            prep_mode="prepare",
        )

    def setup(self, stage=None):
        #print(f"[Rank {dist.get_rank()}] -> Entered setup")
        if self._is_setup:
            return

        # wait for rank-0 prep before reading artifacts
        _ddp_barrier()

        # Now LOAD the artifacts and build real loaders
        self.train_loader, self.val_loader, self.dataset_ref = create_dataloaders(
            fits_dir=self.fits_dir,
            scaling_params_path=self.scaling_params_path,
            wavelength_stride=self.wavelength_stride,
            load_preprocessed=self.load_preprocessed,
            preprocessed_dir=self.preprocessed_dir,
            use_local_nvme=self.use_local_nvme,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            model_params=self.model_params,
            log_scale_params=self.log_scale_params,
            data_subset_fraction=self.data_subset_fraction,
            seed=self.seed,
            prep_mode="load",
        )
        self._is_setup = True

    # Lightning expects these accessors
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    # Optional: for callbacks needing inverse transforms / metadata
    def get_dataset_reference(self):
        return self.dataset_ref