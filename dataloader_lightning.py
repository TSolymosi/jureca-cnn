import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset # Added Subset
from astropy.io import fits
import os
import re
import numpy as np
import shutil
# import time # Not used in this section
import glob
# import random # Not used in this section

# Git version (test)

# Normalize function (kept as is, but not explicitly used in FitsDataset __getitem__ for data, only labels)
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
        return np.log10(data + 1e-6) # Added 1e-6 for stability with potential zeros
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class FitsDataset(data.Dataset):
    def __init__(
        self,
        fits_dir=None,
        file_list=None,
        wavelength_stride=1,
        use_local_nvme=True, # Defaulted to True as in your original
        load_preprocessed=False,
        preprocessed_dir='/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data', # Defaulted
        model_params=None, # ["Dens", "Lum", "radius", "prho"], # Made None, expect from args
        log_scale_params=None, # ["Dens", "Lum"], # Made None, expect from args
    ):
        if model_params is None:
            model_params = ["Dens", "Lum", "radius", "prho"] # Default if not provided
        if log_scale_params is None:
            log_scale_params = ["Dens", "Lum"] # Default if not provided

        self.wavelength_stride = wavelength_stride
        self.load_preprocessed = load_preprocessed
        self.fits_dir_original_arg = fits_dir # Store original fits_dir arg for potential copy

        self.scaler_means = None
        self.scaler_stds = None
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        # Ensure param_indices_to_log is calculated correctly even if a log_scale_param is not in model_params
        self.param_indices_to_log = []
        for p_log in self.log_scale_params:
            if p_log in self.model_params:
                self.param_indices_to_log.append(self.model_params.index(p_log))


        if self.load_preprocessed:
            print(f"Loading preprocessed .npy data from: {preprocessed_dir}")
            self.data_dir_resolved = os.path.join(preprocessed_dir, "data_100") # data_100 or similar
            self.label_dir_resolved = os.path.join(preprocessed_dir, "labels_100") # labels_100 or similar

            self.data_files = sorted(glob.glob(os.path.join(self.data_dir_resolved, "data_*.npy")))
            # Labels are loaded directly from the source 'labels.npy' which contains all labels
            # The split into train/val/test will use indices into this array via Subset
            self.all_labels_raw = np.load(os.path.join(self.label_dir_resolved, "labels.npy"))
            # Min/max are not directly used for transformation here, but for reference or other methods
            self.label_min_ref = np.load(os.path.join(self.label_dir_resolved, "label_min.npy"))
            self.label_max_ref = np.load(os.path.join(self.label_dir_resolved, "label_max.npy"))

            if not self.data_files:
                raise RuntimeError(f"No data_*.npy files found in {self.data_dir_resolved}")
            if len(self.data_files) != len(self.all_labels_raw):
                 print(f"Warning: Mismatch in preprocessed data files ({len(self.data_files)}) and all_labels_raw ({len(self.all_labels_raw)}). Using min length.")
                 min_len = min(len(self.data_files), len(self.all_labels_raw))
                 self.data_files = self.data_files[:min_len]
                 self.all_labels_raw = self.all_labels_raw[:min_len]


        else: # Load from FITS
            self.original_fits_dir_for_copy = fits_dir # Actual source for FITS files
            self.use_local_nvme = use_local_nvme

            if use_local_nvme and os.path.exists("/local/nvme") and self.original_fits_dir_for_copy:
                slurm_id = os.environ.get("SLURM_JOB_ID", "nojobid")
                self.fits_dir_resolved = f"/local/nvme/{slurm_id}_fits_data_{os.path.basename(self.original_fits_dir_for_copy)}"
                if not os.path.exists(self.fits_dir_resolved):
                    print(f"Copying FITS files from {self.original_fits_dir_for_copy} to local storage: {self.fits_dir_resolved}")
                    os.makedirs(self.fits_dir_resolved, exist_ok=True)
                    # If file_list is provided, it should contain absolute paths to original files
                    if file_list:
                        self._copy_selected_files(file_list, self.original_fits_dir_for_copy, self.fits_dir_resolved)
                    else: # Copy entire directory
                        for item in os.listdir(self.original_fits_dir_for_copy):
                            s = os.path.join(self.original_fits_dir_for_copy, item)
                            d = os.path.join(self.fits_dir_resolved, item)
                            if os.path.isdir(s):
                                shutil.copytree(s, d, dirs_exist_ok=True)
                            else:
                                shutil.copy2(s, d)
                else:
                    print(f"Local NVMe path already exists: {self.fits_dir_resolved}")
            elif self.original_fits_dir_for_copy:
                self.fits_dir_resolved = self.original_fits_dir_for_copy
            else: # No FITS dir provided, and not loading preprocessed
                raise ValueError("fits_dir must be provided if not loading preprocessed data.")


            if file_list is not None:
                # If using NVMe, file_list paths need to be relative to the new NVMe dir
                if use_local_nvme and self.original_fits_dir_for_copy:
                    self.fits_files_resolved = [os.path.join(self.fits_dir_resolved, os.path.relpath(f, self.original_fits_dir_for_copy)) for f in file_list]
                else:
                    self.fits_files_resolved = file_list
            else: # Scan directory
                self.fits_files_resolved = [
                    os.path.join(root, f)
                    for root, _, files_in_dir in os.walk(self.fits_dir_resolved)
                    for f in files_in_dir if f.endswith("arcsec.fits") # Assuming _is_valid_fits is implicitly True
                ]

            if not self.fits_files_resolved:
                raise RuntimeError(f"No valid FITS files found in {self.fits_dir_resolved}.")

            self.all_labels_raw = np.array([self.extract_label(os.path.basename(f)) for f in self.fits_files_resolved])

            # Min/max not used for transformation, but can be saved for reference
            # self.label_min_ref = self.all_labels_raw.min(axis=0)
            # self.label_max_ref = self.all_labels_raw.max(axis=0)
            # os.makedirs("Parameters_ref", exist_ok=True)
            # np.save("Parameters_ref/label_min_ref.npy", self.label_min_ref)
            # np.save("Parameters_ref/label_max_ref.npy", self.label_max_ref)

    # _is_valid_fits not strictly needed if just checking extension
    # def _is_valid_fits(self, file_path): return True

    def set_scaling_params(self, means, stds):
        if not isinstance(means, torch.Tensor): means = torch.tensor(means, dtype=torch.float32)
        if not isinstance(stds, torch.Tensor): stds = torch.tensor(stds, dtype=torch.float32)
        self.scaler_means = means
        self.scaler_stds = stds
        print("Scaling parameters set in dataset instance.")
        # for i, param in enumerate(self.model_params):
        #     print(f"  Dataset scaler for {param}: mean = {self.scaler_means[i]:.4f}, std = {self.scaler_stds[i]:.4f}")


    def _copy_selected_files(self, file_list_abs_paths, original_base_dir, target_nvme_dir):
        for src_abs_path in file_list_abs_paths:
            if not os.path.exists(src_abs_path):
                print(f"Warning: Source file for copy not found: {src_abs_path}")
                continue
            rel_path = os.path.relpath(src_abs_path, original_base_dir)
            dst_path = os.path.join(target_nvme_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_abs_path, dst_path)

    def extract_label(self, filename):
        pattern = r'([A-Za-z0-9_]+)=([-\d.eE+]+)'
        matches = re.findall(pattern, filename)
        label_dict = {k.lstrip('_'): float(v) for k, v in matches}
        return [label_dict.get(key, 0.0) for key in self.model_params] # Ensure all model_params are covered

    def inverse_transform_labels(self, scaled_labels_tensor):
        if self.scaler_means is None or self.scaler_stds is None:
            print("[WARNING] inverse_transform_labels called without scaler parameters set on dataset. Returning input as is.")
            return scaled_labels_tensor

        means_dev = self.scaler_means.to(scaled_labels_tensor.device)
        stds_dev = self.scaler_stds.to(scaled_labels_tensor.device)

        unscaled = scaled_labels_tensor * stds_dev + means_dev
        original = unscaled.clone()
        for idx in self.param_indices_to_log:
            if original.ndim == 1: # single sample
                original[idx] = torch.pow(10.0, original[idx])
            else: # batch of samples
                original[:, idx] = torch.pow(10.0, original[:, idx])
        return original

    def __len__(self):
        return len(self.data_files if self.load_preprocessed else self.fits_files_resolved)

    def __getitem__(self, idx):
        # This method now fetches the raw data and the *untransformed* raw label for that index.
        # The transformation (log, then standardize) happens *after* this method,
        # typically once when calculating scaling stats or on-the-fly if not pre-standardizing.
        # For this integration, we'll do the label transformation here to match your original flow.

        if self.load_preprocessed:
            # data_path should be self.data_files[idx]
            data_np = np.load(self.data_files[idx])
            # label_raw is from self.all_labels_raw (which contains all labels)
            label_raw_np = self.all_labels_raw[idx] # This idx is the direct index into all_labels_raw
        else: # Load from FITS
            fits_path = self.fits_files_resolved[idx]
            try:
                with fits.open(fits_path, memmap=True) as hdul:
                    data_np = hdul[0].data
                if data_np is None: raise ValueError(f"Data is None in FITS file: {fits_path}")
            except Exception as e:
                print(f"Error opening FITS file {fits_path} at index {idx}: {e}")
                # Return a dummy valid sample or raise error. For now, let's raise.
                raise RuntimeError(f"Failed to load FITS data for index {idx}, path {fits_path}") from e


            if self.wavelength_stride > 1:
                data_np = data_np[::self.wavelength_stride]
            label_raw_np = self.all_labels_raw[idx]

        # --- Label Transformation (log + standardize) ---
        label_tensor_transformed = torch.tensor(label_raw_np, dtype=torch.float32)
        for i_log in self.param_indices_to_log:
            label_tensor_transformed[i_log] = torch.log10(label_tensor_transformed[i_log].clamp(min=1e-9))

        if self.scaler_means is not None and self.scaler_stds is not None:
            label_tensor_transformed = (label_tensor_transformed - self.scaler_means.cpu()) / self.scaler_stds.cpu() # Perform on CPU
        else:
            # This case should ideally not happen if create_dataloaders calculates/sets them.
            # If it does, it means scaling is not applied, which could be an issue.
            print(f"[WARNING] Item {idx}: Scaler means/stds not set in dataset. Labels not standardized.")

        # Data processing
        # Assuming data_np is (D, H, W) or (num_wavelengths, height, width)
        data_tensor = torch.tensor(data_np.astype(np.float32, copy=False), dtype=torch.float32)
        # Your model expects (N, 1, D, H, W), so add channel dim here.
        # The unsqueeze(0) in your original script likely created (1, D, H, W) for a single sample.
        # If data_np is already (D,H,W), this should be fine.
        # If it's (H,W,D) or other, transpose first.
        # Assuming data_np is (D, H, W)
        # For 3D conv input (N, C_in, D, H, W), C_in is 1
        if data_tensor.ndim == 3: # (D, H, W)
             pass # Keep as is, DataLoader will batch it to (N, D, H, W)
                  # Then the model's forward will unsqueeze to (N, 1, D, H, W)
        # However, your Spectral2DResNet expects (N,1,D,H,W) and you removed the unsqueeze in model.
        # So, the dataset should yield (1, D, H, W) for each sample.
        if data_tensor.ndim == 3: # (D, H, W)
            data_tensor = data_tensor.unsqueeze(0) # -> (1, D, H, W)
        elif data_tensor.ndim == 2: # (H, W) with D=1 squeezed out by FITS? Unlikely for hyperspectral.
            # Handle this case if it occurs, e.g., add D and C dimensions.
            # data_tensor = data_tensor.unsqueeze(0).unsqueeze(0) # -> (1, 1, H, W)
            print(f"Warning: Data for item {idx} is 2D. This might be an issue for spectral CNNs.")


        # Final check for data_tensor shape: should be (C, D, H, W) where C=1 for your model.
        # Model's Spectral2DResNet expects (N, 1, D, H, W)
        # DataLoader will batch, so __getitem__ should return (1, D, H, W)
        if data_tensor.shape[0] != 1 and data_tensor.ndim == 4 : # If it's (C,D,H,W) but C > 1
             print(f"Warning: Data tensor has unexpected channel dimension {data_tensor.shape[0]} for item {idx}")
        elif data_tensor.ndim != 4 or data_tensor.shape[0] != 1: # If not (1,D,H,W)
             print(f"Warning: Unexpected data tensor shape {data_tensor.shape} for item {idx}. Expected (1, D, H, W).")


        return data_tensor, label_tensor_transformed


def calculate_label_scaling_stats(full_dataset_instance, indices_for_stats):
    """
    Calculates mean and std for label scaling, applying log transform first.
    Args:
        full_dataset_instance (FitsDataset): The main dataset instance.
        indices_for_stats (list or np.array): Indices of the samples to use for calculation (e.g., training set indices).
    Returns:
        tuple: (means, stds) as torch.Tensors.
    """
    labels_for_scaling = []
    # Iterate through the *specified indices* of the *raw, untransformed labels*
    for idx in indices_for_stats:
        label_raw = full_dataset_instance.all_labels_raw[idx] # Access raw labels
        label_tensor = torch.tensor(label_raw, dtype=torch.float32)
        # Apply log scaling to specified parameters
        for i_log in full_dataset_instance.param_indices_to_log:
            label_tensor[i_log] = torch.log10(label_tensor[i_log].clamp(min=1e-9))
        labels_for_scaling.append(label_tensor)

    if not labels_for_scaling:
        raise ValueError("No labels found for calculating scaling statistics. Check indices.")

    stacked_labels = torch.stack(labels_for_scaling)
    means = stacked_labels.mean(dim=0)
    stds = stacked_labels.std(dim=0)
    stds[stds == 0] = 1.0 # Avoid division by zero if a parameter has no variance
    return means, stds


def create_dataloaders(
    fits_dir, # Original base directory for FITS if not preprocessed
    original_file_list_path=None, # Path to a file containing list of FITS files to use
    scaling_params_path=None, # Path to pre-calculated scaling params (.pt file)
    wavelength_stride=1,
    load_preprocessed=False,
    preprocessed_dir=None, # Base directory for preprocessed .npy data and labels
    use_local_nvme=False,
    batch_size=16,
    num_workers=4, # num_workers for DataLoader
    model_params=None, # List of strings: parameter names
    log_scale_params=None, # List of strings: params to log_scale
    val_split_ratio=0.2, # Ratio for validation set from the 80% train data
    test_split_ratio=0.2, # Ratio for test set from the initial data (e.g., 0.2 means 20% test, 80% train+val)
    return_datasets_for_lightning=False, # New flag for Lightning
    seed=42 # For reproducible splits
):
    if model_params is None: model_params = ["Dens", "Lum", "radius", "prho"]
    if log_scale_params is None: log_scale_params = ["Dens", "Lum"]

    file_list_abs = None
    if original_file_list_path and os.path.exists(original_file_list_path):
        with open(original_file_list_path, 'r') as f:
            file_list_abs = [line.strip() for line in f if line.strip() and os.path.isabs(line.strip())]
            if not file_list_abs: # If paths are relative, try to make them absolute
                base_dir_for_relative_paths = fits_dir if fits_dir else os.path.dirname(original_file_list_path)
                with open(original_file_list_path, 'r') as f:
                    file_list_abs = [os.path.join(base_dir_for_relative_paths, line.strip()) for line in f if line.strip()]


    # This 'dataset_main_instance' is the single source of all data and label transformations.
    # It holds all_labels_raw. Subsets will just point to indices of this.
    dataset_main_instance = FitsDataset(
        fits_dir=fits_dir, # Original FITS dir for potential copy
        file_list=file_list_abs, # Absolute paths if provided
        wavelength_stride=wavelength_stride,
        use_local_nvme=use_local_nvme,
        load_preprocessed=load_preprocessed,
        preprocessed_dir=preprocessed_dir, # Actual dir for .npy if load_preprocessed
        model_params=model_params,
        log_scale_params=log_scale_params
    )

    dataset_size = len(dataset_main_instance)
    indices = list(range(dataset_size))
    np.random.seed(seed) # for reproducibility of split
    np.random.shuffle(indices)

    # Split: Test set first, then Train and Validation from the remainder
    test_split_idx = int(np.floor(test_split_ratio * dataset_size))
    train_val_indices, test_indices = indices[test_split_idx:], indices[:test_split_idx]

    val_split_idx_from_train_val = int(np.floor(val_split_ratio * len(train_val_indices)))
    # Corrected split: val from train_val, train is the rest of train_val
    train_indices, val_indices = train_val_indices[val_split_idx_from_train_val:], train_val_indices[:val_split_idx_from_train_val]


    # Calculate or load scaling parameters using ONLY training indices
    if scaling_params_path and os.path.exists(scaling_params_path):
        print(f"Loading scaling parameters from: {scaling_params_path}")
        scaling_loaded = torch.load(scaling_params_path)
        means_to_set = scaling_loaded['means']
        stds_to_set = scaling_loaded['stds']
    else:
        print("Calculating scaling parameters using the training set split...")
        means_to_set, stds_to_set = calculate_label_scaling_stats(dataset_main_instance, train_indices)
        
        print("Scaling parameters (means and stds) calculated from training data:")
        for i, param in enumerate(dataset_main_instance.model_params):
            print(f"  {param}: mean = {means_to_set[i]:.4f}, std = {stds_to_set[i]:.4f}")

        scaling_dict_to_save = {"means": means_to_set, "stds": stds_to_set, "model_params": model_params}
        slurm_id = os.environ.get("SLURM_JOB_ID", "nojobid")
        save_dir = f"Parameters_calc/{slurm_id}/" # Changed dir name
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "label_scaling_calculated.pt")
        torch.save(scaling_dict_to_save, save_path)
        print(f"Saved calculated scaling parameters to: {save_path}")

    # Set the calculated/loaded scaling parameters to the main dataset instance
    # This ensures all Subsets created from it will use these same parameters for __getitem__ transformation.
    dataset_main_instance.set_scaling_params(means_to_set, stds_to_set)

    # Create Subset instances for train, validation, and test
    train_dataset_subset = Subset(dataset_main_instance, train_indices)
    val_dataset_subset = Subset(dataset_main_instance, val_indices)
    test_dataset_subset = Subset(dataset_main_instance, test_indices)

    if return_datasets_for_lightning:
        # For Lightning, we return the dataset instances.
        # The DataModule will create DataLoaders from these.
        # The dataset_main_instance is returned for accessing inverse_transform_labels.
        return train_dataset_subset, val_dataset_subset, test_dataset_subset, dataset_main_instance
    else:
        # Original behavior: return DataLoaders
        train_loader = DataLoader(train_dataset_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
        val_loader = DataLoader(val_dataset_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
        # If you need a test_loader separate from val_loader for the original script:
        test_loader_orig = DataLoader(test_dataset_subset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0)
        # Your original script seemed to use test_loader as val_loader, and 'dataset' for inverse_transform
        return train_loader, val_loader, dataset_main_instance # val_loader used as test_loader in original main.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FitsDataset Dataloader Test Script")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing FITS files or preprocessed_dir base.")
    parser.add_argument("--original-file-list", type=str, default=None, help="Path to a text file listing FITS files to use.")
    parser.add_argument("--scaling-params-path", type=str, default=None, help="Path to pre-calculated label_scaling.pt file.")
    parser.add_argument("--wavelength-stride", type=int, default=1)
    parser.add_argument('--load-preprocessed', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Base directory for preprocessed data if --load-preprocessed is True.")
    parser.add_argument("--batch-size", type=int, default=2) # Small for testing
    parser.add_argument("--num-workers", type=int, default=0) # 0 for easier debugging
    parser.add_argument("--use-local-nvme", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--job_id', type=str, default="dataloader_test", help='Job ID for logging purposes.')
    parser.add_argument('--model_params', type=str, nargs='+', default=["Dens", "Lum", "radius", "prho"])
    parser.add_argument('--log_scale_params', type=str, nargs='+', default=["Dens", "Lum"])
    args = parser.parse_args()

    print("Testing create_dataloaders...")
    # Test the Lightning-compatible return
    train_ds, val_ds, test_ds, main_ds_ref = create_dataloaders(
        fits_dir=args.data_dir,
        original_file_list_path=args.original_file_list,
        scaling_params_path=args.scaling_params_path,
        wavelength_stride=args.wavelength_stride,
        load_preprocessed=args.load_preprocessed,
        preprocessed_dir=args.preprocessed_dir,
        use_local_nvme=args.use_local_nvme,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_params=args.model_params,
        log_scale_params=args.log_scale_params,
        return_datasets_for_lightning=True # Key change for testing this path
    )
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Main dataset ref for inverse_transform: {type(main_ds_ref)}")

    # Create DataLoaders from these datasets for a quick check
    train_loader_test = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Check one batch
    if len(train_loader_test) > 0:
        sample_data, sample_labels = next(iter(train_loader_test))
        print(f"\nSample batch from train_loader:")
        print(f"  Data shape: {sample_data.shape}")    # Expected: (batch_size, 1, D, H, W)
        print(f"  Labels shape: {sample_labels.shape}")  # Expected: (batch_size, num_params)
        print(f"  Sample data type: {sample_data.dtype}")
        print(f"  Sample labels type: {sample_labels.dtype}")
        print(f"  Sample labels (first sample, transformed): {sample_labels[0]}")

        if main_ds_ref and hasattr(main_ds_ref, 'inverse_transform_labels'):
            original_labels = main_ds_ref.inverse_transform_labels(sample_labels[0])
            print(f"  Sample labels (first sample, inverse_transformed): {original_labels}")
        else:
            print("  Could not perform inverse transform (main_ds_ref or method missing).")
    else:
        print("Train loader is empty, cannot fetch a sample batch.")

    print("\nTest with original return signature (returning DataLoaders):")
    train_loader_orig, val_loader_orig, main_ds_ref_orig = create_dataloaders(
        fits_dir=args.data_dir,
        original_file_list_path=args.original_file_list,
        scaling_params_path=args.scaling_params_path,
        wavelength_stride=args.wavelength_stride,
        load_preprocessed=args.load_preprocessed,
        preprocessed_dir=args.preprocessed_dir,
        use_local_nvme=args.use_local_nvme,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_params=args.model_params,
        log_scale_params=args.log_scale_params,
        return_datasets_for_lightning=False
    )
    print(f"Train loader original: {type(train_loader_orig)}")
    print(f"Val (Test) loader original: {type(val_loader_orig)}")
    if len(train_loader_orig) > 0:
        sample_data_orig, sample_labels_orig = next(iter(train_loader_orig))
        print(f"\nSample batch from original train_loader:")
        print(f"  Data shape: {sample_data_orig.shape}")
        print(f"  Labels shape: {sample_labels_orig.shape}")