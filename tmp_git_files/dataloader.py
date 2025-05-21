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

def is_valid_fits(file_path):
    try:
        with fits.open(file_path, memmap=True) as hdul:
            return hdul[0].data is not None
    except Exception:
        return False

import os
import glob
from multiprocessing import Pool, cpu_count

def valid_fitsfiles_path(fits_dir, output_file="valid_fits_files.txt", num_workers=None):
    print(f"Scanning for FITS files in {fits_dir}...")
    all_files = glob.glob(os.path.join(fits_dir, "**", "*arcsec.fits"), recursive=True)
    print(f"Found {len(all_files)} FITS candidates. Validating...")

    # Use all available cores if not specified
    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # Cap at 32 unless you want to go higher

    # Parallel validation
    with Pool(processes=num_workers) as pool:
        valid_flags = pool.map(is_valid_fits, all_files)

    valid_files = [f for f, is_valid in zip(all_files, valid_flags) if is_valid]

    # Save result
    with open(output_file, "w") as f:
        for file in valid_files:
            f.write(f"{file}\n")

    print(f"Valid FITS files saved to: {output_file} ({len(valid_files)} valid files)")


def extract_label(filename):
    pattern = r'([A-Za-z0-9_]+)=([-\d.eE+]+)'
    matches = re.findall(pattern, filename)
    label_dict = {k.lstrip('_'): float(v) for k, v in matches}
    expected_keys = ["Dens", "Lum", "radius", "prho"]#, "NCH3CN", "incl", "phi"]
    return [label_dict.get(key, 0.0) for key in expected_keys]


def calculate_label_scaling(dataset, indices):
    labels = []
    param_indices_to_log = [dataset.model_params.index(p) for p in dataset.log_scale_params if p in dataset.model_params]
    for idx in indices:
        label = dataset.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        for i in param_indices_to_log:
            label_tensor[i] = torch.log10(label_tensor[i].clamp(min=1e-9))
        labels.append(label_tensor)

    stacked = torch.stack(labels)
    means = stacked.mean(dim=0)
    stds = stacked.std(dim=0)
    stds[stds == 0] = 1.0
    return means, stds


class FitsDataset(data.Dataset):
    def __init__(
        self,
        fits_dir=None,
        file_list=None,
        wavelength_stride=1,
        labels=None,
        model_params=["Dens", "Lum", "radius", "prho"],
        log_scale_params=["Dens", "Lum"],
        label_min=None,
        label_max=None,
        normalization_method="zscore",
    ):
        self.wavelength_stride = wavelength_stride
        self.label_min = label_min
        self.label_max = label_max
        self.labels = labels
        self.scaler_mean = None
        self.scaler_std = None
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.param_indices_to_log = [self.model_params.index(p) for p in self.log_scale_params if p in self.model_params]
        self.normalization_method = normalization_method

        # -------- Load from FITS files --------
        self.original_fits_dir = fits_dir
        self.fits_dir = fits_dir

        # Find valid FITS files
        if file_list is not None:
            self.fits_files = file_list
        else:
            self.fits_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(self.fits_dir)
                for f in files if f.endswith("arcsec.fits") and is_valid_fits(os.path.join(root, f))
            ]

        if len(self.fits_files) == 0:
            raise RuntimeError("No valid FITS files found.")

    def set_scaling_params(self, means, stds):
        self.scaler_means = means
        self.scaler_stds = stds

    def _copy_selected_files(self, file_list):
        for src_path in file_list:
            rel_path = os.path.relpath(src_path, self.original_fits_dir)
            dst_path = os.path.join(self.fits_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

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

    def __len__(self):
        return len(self.fits_files)

    def __getitem__(self, idx):
        try:
            fits_path = self.fits_files[idx]
            with fits.open(fits_path, memmap=True) as hdul:
                data = hdul[0].data
            if data is None:
                raise ValueError(f"Data is None in FITS file: {fits_path}")
            if self.wavelength_stride > 1:
                # Downsample data
                data = data[::self.wavelength_stride]
                
            # Normalize data
            normalization_method = self.normalization_method
            data = normalize(data, method=normalization_method)
            #label = extract_label(os.path.basename(fits_path))
            label = self.labels[idx]

            # Normalize label
            #label = (np.array(label) - self.label_min) / (self.label_max - self.label_min + 1e-8)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            # Log scale certain parameters
            for i in self.param_indices_to_log:
                label_tensor[i] = torch.log10(label_tensor[i].clamp(min=1e-9))

            # Standardize using means and stds
            if self.scaler_means is not None and self.scaler_stds is not None:
                label_tensor = (label_tensor - self.scaler_means) / self.scaler_stds

            # Return as tensors
            #data = torch.tensor(data.astype(np.float32, copy=False), dtype=torch.float32).unsqueeze(0)
            data = torch.from_numpy(data.astype(np.float32, copy=False)).unsqueeze(0)
        except Exception as e:
            print(f"[ERROR] Failed on index {idx}, file: {self.file_list[idx]}")
            raise e
        return data, label_tensor

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def create_dataloaders(
    fits_dir,
    file_list_path,
    scaling_params_path=None,
    wavelength_stride=1,
    load_preprocessed=False,
    preprocessed_dir=None,
    use_local_nvme=False,
    batch_size=16,
    num_workers=32,
    test_split_ratio=0.2,
    random_seed=42,
    train_sampler=None,
    test_sampler=None,
    model_params=["Dens", "Lum", "radius", "prho"],
    log_scale_params=["Dens", "Lum"],
    job_id=None,
    normalization_method="zscore",
):
    regenerate_file_list = False
    if not os.path.exists(file_list_path):
        print(f"File list not found: {file_list_path}. Generating...")
        regenerate_file_list = True
    else:
        with open(file_list_path, "r") as f:
            # Check if the path of the first file in the list is valid
            first_file = f.readline().strip()
            if not os.path.exists(first_file):
                print(f"File list is invalid. Generating new file list...")
                regenerate_file_list = True

    if regenerate_file_list:
        valid_fitsfiles_path(fits_dir, file_list_path, num_workers=num_workers)

    # Load valid file list from disk
    with open(file_list_path, "r") as f:
        all_files = [line.strip() for line in f if line.strip()]

    all_labels = np.array([extract_label(f) for f in all_files])
    label_min = all_labels.min(axis=0)
    label_max = all_labels.max(axis=0)

    # Train-test split at file list level
    train_files, test_files = train_test_split(
        all_files, test_size=test_split_ratio, random_state=random_seed
    )

    train_labels = np.array([extract_label(f) for f in train_files])
    train_label_min = train_labels.min(axis=0)
    train_label_max = train_labels.max(axis=0)

    test_labels = np.array([extract_label(f) for f in test_files])
    test_label_min = test_labels.min(axis=0)
    test_label_max = test_labels.max(axis=0)

    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    # Create separate dataset objects
    train_dataset = FitsDataset(
        fits_dir=fits_dir,
        file_list=train_files,
        wavelength_stride=wavelength_stride,
        labels = train_labels,
        label_max=train_label_max,
        label_min=train_label_min,
        model_params=model_params,
        log_scale_params=log_scale_params,
    )

    test_dataset = FitsDataset(
        fits_dir=fits_dir,
        file_list=test_files,
        wavelength_stride=wavelength_stride,
        labels=test_labels,
        label_max=test_label_max,
        label_min=test_label_min,
        model_params=model_params,
        log_scale_params=log_scale_params,
    )

    # Optional: apply label scaling (shared across both)
    if scaling_params_path and os.path.exists(scaling_params_path):
        scaling = torch.load(scaling_params_path)
        for ds in (train_dataset, test_dataset):
            ds.label_min = scaling["label_min"]
            ds.label_max = scaling["label_max"]
    else:
        # Compute scaling parameters if not provided
        print("Calculating scaling parameters...")
        # Extract training indices (required for scaling)
        train_indices = train_dataset.indices if hasattr(train_dataset, 'indices') else list(range(len(train_dataset)))


        # Compute scaling parameters
        means, stds = calculate_label_scaling(train_dataset, train_indices)

        # Set scaling inside the dataset (apply to both train and test)
        for ds in (train_dataset, test_dataset):
            ds.set_scaling_params(means, stds)

        # Print scaling parameters with param name
        print("Scaling parameters (means and stds):")
        for i, param in enumerate(train_dataset.model_params):
            print(f"{param}: mean = {means[i]:.4f}, std = {stds[i]:.4f}")

        # Save scaling parameters
        scaling_dict = {"means": means, "stds": stds}
        slurm_id = os.environ.get("SLURM_JOB_ID", "nojob")
        print(f"Saving scaling parameters to: Parameters/{slurm_id}/label_scaling.pt")
        os.makedirs(f"Parameters/{slurm_id}/", exist_ok=True)
        torch.save(scaling_dict, f"Parameters/{slurm_id}/label_scaling.pt")

        
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=(test_sampler is None),
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    print(f"Train loader size: {len(train_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    return train_loader, test_loader, (train_dataset, test_dataset)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    #parser.add_argument("--original-file-list", type=str, default=None)
    #parser.add_argument("--scaling-params-path", type=str, default=None)
    parser.add_argument("--wavelength-stride", type=int, default=1)
    parser.add_argument("--load-preprocessed", type=bool, default=False)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use-local-nvme", type=bool, default=False)
    args = parser.parse_args()

    train_loader, test_loader, dataset = create_dataloaders(
        fits_dir=args.data_dir,
        #original_file_list_path=args.original_file_list,
        #scaling_params_path=args.scaling_params_path,
        wavelength_stride=args.wavelength_stride,
        load_preprocessed=args.load_preprocessed,
        preprocessed_dir=args.preprocessed_dir,
        use_local_nvme=args.use_local_nvme,
        batch_size=args.batch_size
    )

    print(f"Dataset size: {len(dataset)}")
    for batch in train_loader:
        print("Sample batch:", batch[0].shape, batch[1].shape)
        break
