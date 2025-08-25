import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from astropy.io import fits
import os
import re
import numpy as np
import shutil
import time
import glob
import random

from contextlib import suppress
import torch.distributed as dist

# Override the default print function to flush the output immediately
import functools
print = functools.partial(print, flush=True)

seed=42
random.seed(seed)  # For reproducibility
np.random.seed(seed)

from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs)

# Git version (test)

# Helper function for Hydra setup
def ddp_barrier():
    with suppress(Exception):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

def normalize(data, method='minmax'):
    """Normalize 3D data using specified method"""
    if method == 'minmax':
        data = np.nan_to_num(data)
        lower = np.percentile(data, 1)
        upper = np.percentile(data, 99)
        return np.clip((data - lower) / (upper - lower + 1e-6), 0, 1)
    elif method == 'zscore':
        data = np.nan_to_num(data)
        return (data - np.mean(data)) / (np.std(data) + 1e-6)
    elif method == "root":
        data = np.nan_to_num(data)
        return np.cbrt(data)
    elif method == "log":
        data = np.nan_to_num(data)
        return np.log10(data + 1e-6)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class FitsDataset(data.Dataset):
    #print("+++ FitsDataset class body executed +++")
    def __init__(
        self,
        fits_dir=None,
        file_list=None,
        wavelength_stride=1,
        use_local_nvme=True,
        load_preprocessed=False,
        preprocessed_dir='/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data',
        model_params=["D", "L", "rr", "p"],
        log_scale_params=["D", "L"],
    ):
        self.wavelength_stride = wavelength_stride
        self.load_preprocessed = load_preprocessed
        self.file_list = file_list
        print("=== FitsDataset constructor entered ===")
        

        self.scaler_means = None
        self.scaler_stds = None
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.param_indices_to_log = [self.model_params.index(p) for p in self.log_scale_params if p in self.model_params]


        if self.load_preprocessed:
            # -------- Load from preprocessed .npy files --------
            print(f"Loading preprocessed .npy data from: {preprocessed_dir}")
            self.data_dir = os.path.join(preprocessed_dir, "data_100")
            self.label_dir = os.path.join(preprocessed_dir, "labels_100")

            self.data_files = sorted(glob.glob(os.path.join(self.data_dir, "data_*.npy")))
            self.labels = np.load(os.path.join(self.label_dir, "labels.npy"))
            self.label_min = np.load(os.path.join(self.label_dir, "label_min.npy"))
            self.label_max = np.load(os.path.join(self.label_dir, "label_max.npy"))
            self.resolved_file_list = list(getattr(self, "data_files", []))
            assert len(self.data_files) == len(self.labels), "Mismatch in data and label count."

        else:
            # -------- Load from FITS files --------
            self.original_fits_dir = fits_dir
            self.use_local_nvme = use_local_nvme

            # Copy to NVMe if requested
            if use_local_nvme and os.path.exists("/local/nvme"):
                slurm_id = os.environ.get("SLURM_JOB_ID", "nojob")
                #self.fits_dir = f"/local/nvme/{os.environ['USER']}/{slurm_id}"
                self.fits_dir = f"/local/nvme/{slurm_id}_fits_data"
                if not os.path.exists(self.fits_dir):
                    print_rank0(f"Copying FITS files to local storage: {self.fits_dir}")
                    os.makedirs(self.fits_dir, exist_ok=True)
                    if dist.get_rank() == 0:
                        if file_list:
                            self._copy_selected_files(file_list)
                        else:
                            shutil.copytree(fits_dir, self.fits_dir, dirs_exist_ok=True)
                    dist.barrier()
                else:
                    print_rank0(f"Local NVMe path already exists: {self.fits_dir}")
            else:
                self.fits_dir = fits_dir

            # Find valid FITS files
            if file_list is not None:
                self.fits_files = file_list
            else:
                print_rank0(f"Searching for FITS files in: {self.fits_dir}")
                self.fits_files = [
                    os.path.join(root, f)
                    for root, _, files in os.walk(self.fits_dir)
                    for f in files if f.endswith("arcsec.fits") and self._is_valid_fits(os.path.join(root, f))
                ]
                print_rank0(f"Found {len(self.fits_files)} valid FITS files in {self.fits_dir}")

            if len(self.fits_files) == 0:
                raise RuntimeError("ERROR: No valid FITS files found.")

            self.resolved_file_list = list(getattr(self, "fits_files", []))

            print_rank0("Extracting labels from FITS filenames...")
            self.labels = [self.extract_label(os.path.basename(f)) for f in self.fits_files]
            self.labels = np.array(self.labels)
            #self.label_min = self.labels.min(axis=0)
            #self.label_max = self.labels.max(axis=0)
            
        
            os.makedirs("Parameters", exist_ok=True)
            #np.save("Parameters/label_min.npy", self.label_min)
            #np.save("Parameters/label_max.npy", self.label_max)

    def _is_valid_fits(self, file_path):
        #try:
        #    with fits.open(file_path, memmap=True) as hdul:
        #        return hdul[0].data is not None
        #except Exception:
        #    return False
        return True
        
    def set_scaling_params(self, means, stds):
        self.scaler_means = means
        self.scaler_stds = stds


    def _copy_selected_files(self, file_list):
        for src_path in file_list:
            rel_path = os.path.relpath(src_path, self.original_fits_dir)
            dst_path = os.path.join(self.fits_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

    def extract_label(self, filename):
        pattern = r'([A-Za-z0-9_]+)=([-\d.eE+]+)'
        matches = re.findall(pattern, filename)

        label_dict = {}
        for k, v in matches:
            # Do not strip! Only normalize well-defined prefix
            key = k
            key = k.strip("_")  # remove any accidental leading underscores
            if key.startswith("LTE_"):
                key = key[len("LTE_"):]
            label_dict[key] = float(v)

        # DEBUG
        #print(f"[extract_label] {filename}")
        #print(f"  → Extracted: {label_dict}")
        #print(f"  → Requested params: {self.model_params}")

        label = []
        if "incl" in self.model_params and "phi" in self.model_params:
            if "plummer_shape" in self.model_params:
                raise ValueError("Cannot use 'plummer_shape' with 'incl' and 'phi' parameters together. Choose one set.")
            incl_rad = np.radians(label_dict["incl"])
            phi_rad = np.radians(label_dict["phi"])
            trig_map = {
                "incl": [np.sin(incl_rad), np.cos(incl_rad)],
                "phi":  [np.sin(phi_rad),  np.cos(phi_rad)],
            }
            for param in self.model_params:
                if param in trig_map:
                    label.extend(trig_map[param])
                else:
                    if param not in label_dict:
                        raise KeyError(f"❌ Missing parameter '{param}' in filename:\n{filename}\nExtracted: {label_dict}")
                    label.append(label_dict[param])
        else:
            for param in self.model_params:
                if param == "plummer_shape":
                    # This is our special derived parameter.
                    # Get its components from the dictionary.
                    p_val = label_dict.get('p')
                    rr_val = label_dict.get('rr')

                    # Check that the components were found in the filename.
                    if p_val is None or rr_val is None:
                        raise KeyError(f"To compute 'plummer_shape', both 'p' and 'rr' must be in the filename's labels. File: {filename}")

                    # Perform a safety check for the log operation.
                    if rr_val <= 0:
                        raise ValueError(f"Cannot compute log of non-positive radius={rr_val} for 'plummer_shape'. File: {filename}")

                    # Calculate the derived value and append it. We use log10 for consistency.
                    derived_value = p_val * np.log10(rr_val)
                    label.append(derived_value)
                if param not in label_dict and param != "plummer_shape":
                    raise KeyError(f"❌ Missing parameter '{param}' in filename:\n{filename}\nExtracted: {label_dict}")
                if param != "plummer_shape":
                    # Normal parameters
                    label.append(label_dict[param])

        return label
    



    
    def inverse_transform_labels(self, scaled_labels):
        """
        Inverse transform standardized + log-scaled labels back to original physical units.

        Args:
            scaled_labels (torch.Tensor): Model outputs, shape (N,) or (B, N)

        Returns:
            torch.Tensor: Labels in original (physical) scale
        """

        if self.scaler_means is None or self.scaler_stds is None:
            print("[WARNING] inverse_transform_labels called without scaler parameters.")
            return scaled_labels

        means = self.scaler_means
        stds = self.scaler_stds

        # Ensure input is tensor and on same device as means/stds
        if not isinstance(scaled_labels, torch.Tensor):
            scaled_labels = torch.tensor(scaled_labels)
        scaled_labels = scaled_labels.to(means.device)

        # 1. Undo standardization
        unscaled = scaled_labels * stds + means

        # 2. Undo log10 for specified parameters
        original = unscaled.clone()
        for idx in self.param_indices_to_log:
            if original.ndim == 1:
                original[idx] = torch.pow(10.0, original[idx])
            else:
                original[:, idx] = torch.pow(10.0, original[:, idx])

        return original

    def inverse_transform_labels_with_uncertainty(self, scaled_mu, scaled_sigma):
        """
        Inverse transform predicted means and uncertainties from scaled space
        back to original physical units.

        Args:
            scaled_mu (torch.Tensor): Scaled predicted means, shape (B, N)
            scaled_sigma (torch.Tensor): Scaled predicted stddevs, shape (B, N)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mu_original, sigma_original)
        """
        if self.scaler_means is None or self.scaler_stds is None:
            print("[WARNING] inverse_transform_labels_with_uncertainty called without scaler parameters.")
            return scaled_mu, scaled_sigma

        means = self.scaler_means
        stds = self.scaler_stds

        if not isinstance(scaled_mu, torch.Tensor):
            scaled_mu = torch.from_numpy(scaled_mu)
        if not isinstance(scaled_sigma, torch.Tensor):
            scaled_sigma = torch.from_numpy(scaled_sigma)

        scaled_mu = scaled_mu.to(means.device)
        scaled_sigma = scaled_sigma.to(means.device)

        unscaled_mu = scaled_mu * stds + means
        unscaled_sigma = scaled_sigma * stds

        mu_orig = unscaled_mu.clone()
        sigma_orig = unscaled_sigma.clone()

        for idx in self.param_indices_to_log:
            mu_log = unscaled_mu[:, idx]
            sigma_log = unscaled_sigma[:, idx]

            # Convert mu ± sigma from log space to linear space
            upper = torch.pow(10.0, mu_log + sigma_log)
            lower = torch.pow(10.0, mu_log - sigma_log)

            mu_orig[:, idx] = torch.pow(10.0, mu_log)
            sigma_orig[:, idx] = (upper - lower) / 2.0

        return mu_orig, sigma_orig



    def __len__(self):
        return len(self.data_files if self.load_preprocessed else self.fits_files)

    def __getitem__(self, idx):
        #print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] __getitem__ called with idx: {idx}")
        try:
            if self.load_preprocessed:
                data = np.load(self.data_files[idx])
                label = self.labels[idx]
            else:
                fits_path = self.fits_files[idx]
                with fits.open(fits_path, memmap=True) as hdul:
                    data = hdul[0].data
                if data is None:
                    raise ValueError(f"Data is None in FITS file: {fits_path}")
                if self.wavelength_stride > 1:
                    # Downsample data
                    data = data[::self.wavelength_stride]
                    
                # Normalize data
                #data = normalize(data, method="zscore")
                label = self.labels[idx]
            
            # --- NaN check & repair ---
            if np.isnan(data).any():
                nan_indices = np.argwhere(np.isnan(data))  # array of [d, h, w]
                
                log_file = "/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/NanFits/runtime_nan_log.txt"
                with open(log_file, "a") as log:
                    log.write(f"\n[NaN Repair] File: {fits_path}, Shape: {data.shape}, NaN count: {len(nan_indices)}\n")
                    #for (d, h, w) in nan_indices:
                    #    log.write(f"   -> NaN at (channel={d}, y={h}, x={w})\n")
                for d in range(data.shape[0]):
                    slice_nan = np.isnan(data[d])
                    if np.any(slice_nan):
                        mean_val = np.nanmean(data[d])
                        if np.isnan(mean_val):
                            mean_val = 0.0  # or np.nanmin(data) or any safe default
                            print(f"[WARN] Fully-NaN channel {d} in file {fits_path} — filled with {mean_val}")
                        data[d][slice_nan] = mean_val


            # Normalize label
            #label = (np.array(label) - self.label_min) / (self.label_max - self.label_min + 1e-8)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            # Log scale certain parameters
            for i in self.param_indices_to_log:
                label_tensor[i] = torch.log10(label_tensor[i].clamp(min=1e-12))

            # Standardize using means and stds
            if self.scaler_means is not None and self.scaler_stds is not None:
                label_tensor = (label_tensor - self.scaler_means) / self.scaler_stds
            else:
                print("[WARNING] No scaling parameters provided. Using raw labels.")

            # Return as tensors
            data = torch.tensor(data.astype(np.float32, copy=False), dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            print(f"[ERROR] Failed on index {idx}, file: {self.file_list[idx]}")
            raise e
        return data, label_tensor

# def calculate_label_scaling(full_dataset, indices): # `full_dataset` could be a Subset
#     labels = []
    
#     # Access the underlying dataset if we're dealing with a Subset
#     #original_dataset = full_dataset.dataset if isinstance(full_dataset, torch.utils.data.Subset) else full_dataset
#     # Map to original dataset + original indices
#     if isinstance(full_dataset, torch.utils.data.Subset):
#         subset = full_dataset
#         original_dataset = subset.dataset
#         subset_to_orig = subset.indices
#         orig_indices = [subset_to_orig[i] for i in indices]   # <-- map!
#     else:
#         original_dataset = full_dataset
#         orig_indices = indices
    
#     # Use the attributes from the original dataset
#     param_indices_to_log = [original_dataset.model_params.index(p) for p in original_dataset.log_scale_params if p in original_dataset.model_params]
    
#     for idx in orig_indices:
#         # Get the label from the original dataset's full label list
#         label = original_dataset.labels[idx] 
#         label_tensor = torch.tensor(label, dtype=torch.float32)
#         for i in param_indices_to_log:
#             label_tensor[i] = torch.log10(label_tensor[i].clamp(min=1e-11))
#         labels.append(label_tensor)

#     stacked = torch.stack(labels)
#     means = stacked.mean(dim=0)
#     stds = stacked.std(dim=0)
#     stds[stds == 0] = 1.0
#     return means, stds

def calculate_label_scaling(full_dataset, indices):
    labels = []

    # Map subset indices -> original indices
    if isinstance(full_dataset, torch.utils.data.Subset):
        subset = full_dataset
        original_dataset = subset.dataset
        orig_indices = [subset.indices[i] for i in indices]
    else:
        original_dataset = full_dataset
        orig_indices = indices

    param_indices_to_log = [original_dataset.model_params.index(p)
                            for p in original_dataset.log_scale_params
                            if p in original_dataset.model_params]

    for idx in orig_indices:
        label = original_dataset.labels[idx]
        t = torch.tensor(label, dtype=torch.float32)
        for i in param_indices_to_log:
            t[i] = torch.log10(t[i].clamp(min=1e-11))
        labels.append(t)

    stacked = torch.stack(labels)
    means = stacked.mean(dim=0)
    stds  = stacked.std(dim=0, unbiased=False).clamp_min(1e-12)
    return means, stds


import os, json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Literal


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _derive_aux_paths(scaling_params_path: str):
    """
    Use the directory of scaling_params_path to store other prep artifacts.
    """
    base_dir = os.path.dirname(scaling_params_path)
    split_path = os.path.join(base_dir, "split_indices.pt")
    subset_path = os.path.join(base_dir, "subset_indices.pt")
    filelist_path = os.path.join(base_dir, "file_list.json")
    return split_path, subset_path, filelist_path



def create_dataloaders(
    fits_dir,
    original_file_list_path=None,
    scaling_params_path=None,  # REQUIRED for prep/load handoff
    wavelength_stride=1,
    load_preprocessed=False,
    preprocessed_dir=None,
    use_local_nvme=False,
    batch_size=16,
    num_workers=32,
    train_sampler=None,
    test_sampler=None,
    model_params=("D","L","ro","rr","p","Tlow","NCH3CN","plummer_shape"),
    log_scale_params=("D","L","NCH3CN"),
    data_subset_fraction=1.0,
    seed: int = 42,
    prep_mode: Literal["prepare","load"] = "load",
):
    """
    Two-phase behavior:
      - prepare: rank-0 only. Scan files, make optional subset, deterministic split,
                 compute scaling on train set, save artifacts. Returns None.
      - load:    all ranks. Load artifacts, build datasets/dataloaders, return them.
    """

    os.makedirs(os.path.dirname(scaling_params_path), exist_ok=True)
    assert scaling_params_path is not None, "scaling_params_path must be provided"
    _ensure_dir(scaling_params_path)
    split_path, subset_path, filelist_path = _derive_aux_paths(scaling_params_path)

    # ---------- Resolve input file list ----------
    file_list = None
    if original_file_list_path and os.path.exists(original_file_list_path):
        with open(original_file_list_path, "r") as f:
            file_list = [line.strip() for line in f if line.strip()]

    # ---------- PREPARE PHASE (rank 0 writes, all ranks wait) ----------
    if prep_mode == "prepare":
        rank_zero = False
        if dist.get_rank() == 0:
            rank_zero = True
        print_rank0("=== PREPARE: scanning/creating subset/split & computing scaling (once) ===")

        # Build a full dataset to discover files & labels
        ds_full = FitsDataset(
            fits_dir=fits_dir,
            file_list=file_list,
            wavelength_stride=wavelength_stride,
            use_local_nvme=use_local_nvme,
            load_preprocessed=load_preprocessed,
            preprocessed_dir=preprocessed_dir or fits_dir,
            model_params=list(model_params),
            log_scale_params=list(log_scale_params),
        )
        num_total = len(ds_full)
        
        print_rank0(f"[PREP] Found {num_total} samples")

        rng = np.random.default_rng(seed)
        if data_subset_fraction < 1.0:
            n_keep = int(num_total * data_subset_fraction)
            subset_indices = np.sort(rng.choice(num_total, size=n_keep, replace=False))
            
            print_rank0(f"[PREP] Using subset: {n_keep}/{num_total} samples")
        else:
            subset_indices = np.arange(num_total)

        n_subset = len(subset_indices)
        n_train = int(0.8 * n_subset)
        perm = rng.permutation(n_subset)
        train_idx = subset_indices[perm[:n_train]]
        val_idx   = subset_indices[perm[n_train:]]

        print("[PREP] Computing scaling on train split...")
        means, stds = calculate_label_scaling(ds_full, train_idx.tolist())
        ds_full.set_scaling_params(means, stds)

        # --- rank-0 does all writes ---
        if rank_zero:
            torch.save({"means": means, "stds": stds}, scaling_params_path)
            torch.save({"train_indices": train_idx, "val_indices": val_idx}, split_path)
            torch.save({"subset_indices": subset_indices}, subset_path)
            to_write = file_list if file_list is not None else ds_full.resolved_file_list
            with open(filelist_path, "w") as f:
                json.dump(to_write, f)
            print_rank0(f"[PREP] Saved scaling → {scaling_params_path}")
            print_rank0(f"[PREP] Saved split   → {split_path}")
            print_rank0(f"[PREP] Saved subset  → {subset_path}")
            print_rank0(f"[PREP] Saved file list → {filelist_path}")
            return


    # ---------- LOAD PHASE (all ranks) ----------
    # Read artifacts produced in prepare

    def _wait_for(path, tries=50, delay=0.1):
        for _ in range(tries):
            if os.path.exists(path):
                return True
            time.sleep(delay)
        return os.path.exists(path)

    for p in (scaling_params_path, split_path, filelist_path):
        _wait_for(p)

    assert os.path.exists(scaling_params_path), f"Missing {scaling_params_path}. Run prep first."
    assert os.path.exists(split_path), f"Missing split indices at {split_path}. Run prep first."
    assert os.path.exists(filelist_path), f"Missing file list at {filelist_path}. Run prep first."

    scaling = torch.load(scaling_params_path, weights_only=True)
    split = torch.load(split_path)

    with open(filelist_path, "r") as f:
        file_list_loaded = json.load(f)

    # Rebuild the dataset with the EXACT same file order as in prepare
    print("Creating dataset (LOAD)")
    ds_full = FitsDataset(
        fits_dir=fits_dir,
        file_list=file_list_loaded,
        wavelength_stride=wavelength_stride,
        use_local_nvme=use_local_nvme,
        load_preprocessed=load_preprocessed,
        preprocessed_dir=preprocessed_dir or fits_dir,
        model_params=list(model_params),
        log_scale_params=list(log_scale_params),
    )
    # Apply scaling
    ds_full.set_scaling_params(scaling["means"], scaling["stds"])

    # Wrap with Subset for the same subset as in prepare
    subset_indices_obj = torch.load(subset_path)["subset_indices"]
    # Then use the saved train/val split indices
    train_idx = split["train_indices"]
    val_idx   = split["val_indices"]

    # Map train/val into Subsets of ds_full
    train_dataset = torch.utils.data.Subset(ds_full, train_idx)
    val_dataset   = torch.utils.data.Subset(ds_full, val_idx)

    # Build loaders (note: persistent_workers only if workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False
    )

    # For callbacks needing a reference to the base dataset (for metadata, inverses, etc.)
    dataset_ref = ds_full
    return train_loader, val_loader, dataset_ref


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    #parser.add_argument("--original-file-list", type=str, default=None)
    parser.add_argument("--scaling-params-path", type=str, default=None)
    parser.add_argument("--wavelength-stride", type=int, default=1)
    parser.add_argument("--load-preprocessed", type=bool, default=False)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--use-local-nvme", type=bool, default=False)
    parser.add_argument('--job_id', type=str, default=None, help='Job ID for logging purposes.')
    parser.add_argument('--model_params', type=str, nargs='+', default=["Dens", "Lum", "radius", "prho"], help='List of all model parameters to be trained.')
    parser.add_argument('--log_scale_params', type=str, nargs='+', default=["Dens", "Lum"], help='List of parameters to be log-scaled.')
    args = parser.parse_args()

    train_loader, test_loader, dataset = create_dataloaders(
        fits_dir=args.data_dir,
        #original_file_list_path=args.original_file_list,
        #scaling_params_path=args.scaling_params_path,
        wavelength_stride=args.wavelength_stride,
        load_preprocessed=args.load_preprocessed,
        preprocessed_dir=args.preprocessed_dir,
        use_local_nvme=args.use_local_nvme,
        batch_size=args.batch_size,
        model_params=args.model_params,
        log_scale_params=args.log_scale_params,
    )

    print(f"Dataset size: {len(dataset)}")
    for batch in train_loader:
        print("Sample batch:", batch[0].shape, batch[1].shape)
        break
