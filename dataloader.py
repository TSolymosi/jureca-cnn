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

# Git version (test)

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
    def __init__(
        self,
        fits_dir=None,
        file_list=None,
        wavelength_stride=1,
        use_local_nvme=True,
        load_preprocessed=False,
        preprocessed_dir='/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data',
        model_params=["Dens", "Lum", "radius", "prho"],
        log_scale_params=["Dens", "Lum"],
    ):
        self.wavelength_stride = wavelength_stride
        self.load_preprocessed = load_preprocessed

        self.scaler_means = None
        self.scaler_stds = None
        #self.expected_keys = ["Dens", "Lum", "radius", "prho"]#, "NCH3CN", "incl", "phi"]
        #self.log_scale_params = ["Dens", "Lum"]  # Parameters to be log-scaled
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
                    print(f"Copying FITS files to local storage: {self.fits_dir}")
                    os.makedirs(self.fits_dir, exist_ok=True)
                    if file_list:
                        self._copy_selected_files(file_list)
                    else:
                        shutil.copytree(fits_dir, self.fits_dir, dirs_exist_ok=True)
                else:
                    print(f"Local NVMe path already exists: {self.fits_dir}")
            else:
                self.fits_dir = fits_dir

            # Find valid FITS files
            if file_list is not None:
                self.fits_files = file_list
            else:
                self.fits_files = [
                    os.path.join(root, f)
                    for root, _, files in os.walk(self.fits_dir)
                    for f in files if f.endswith("arcsec.fits")
                ]

            if len(self.fits_files) == 0:
                raise RuntimeError("No valid FITS files found.")

            self.labels = [self.extract_label(os.path.basename(f)) for f in self.fits_files]
            self.labels = np.array(self.labels)
            #self.label_min = self.labels.min(axis=0)
            #self.label_max = self.labels.max(axis=0)
            

            os.makedirs("Parameters", exist_ok=True)
            #np.save("Parameters/label_min.npy", self.label_min)
            #np.save("Parameters/label_max.npy", self.label_max)

        
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

        # Fix: strip known prefixes like 'LTE_' from keys
        label_dict = {}
        for k, v in matches:
            key = k.lstrip('_')
            if key.startswith("LTE_"):
                key = key.replace("LTE_", "")
            label_dict[key] = float(v)

        if "incl" in self.model_params and "phi" in self.model_params:
            incl_rad = np.radians(label_dict.get("incl", 0.0))
            phi_rad = np.radians(label_dict.get("phi", 0.0))
            trig_map = {
                "incl": [np.sin(incl_rad), np.cos(incl_rad)],
                "phi":  [np.sin(phi_rad),  np.cos(phi_rad)],
            }
            label = []
            for param in self.model_params:
                if param in trig_map:
                    label.extend(trig_map[param])  # adds 2 values
                else:
                    label.append(label_dict.get(param, 0.0))
            return label
        else:
            return [label_dict.get(key, 0.0) for key in self.model_params]

    
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
        return len(self.data_files if self.load_preprocessed else self.fits_files)

    def __getitem__(self, idx):
        # ------------------ Load Data ------------------
        if self.load_preprocessed:
            data = np.load(self.data_files[idx])
            raw_label = self.labels[idx]
        else:
            fits_path = self.fits_files[idx]
            with fits.open(fits_path, memmap=True) as hdul:
                data = hdul[0].data
            if data is None:
                raise ValueError(f"Data is None in FITS file: {fits_path}")
            if self.wavelength_stride > 1:
                data = data[::self.wavelength_stride]

            # Normalize data
            # data = normalize(data, method="zscore")
            raw_label = self.labels[idx]

        # --- NaN check & repair ---
        if np.isnan(data).any():
            nan_indices = np.argwhere(np.isnan(data))  # array of [d, h, w]
            
            log_file = "/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/FileList/runtime_nan_log.txt"
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

        # ------------------ Process Labels ------------------
        processed_label = []
        i = 0
        for param in self.model_params:
            if param in ("incl", "phi"):
                # Already in sin/cos from extract_label
                processed_label.append(raw_label[i])
                processed_label.append(raw_label[i+1])
                i += 2
            else:
                val = raw_label[i]
                if param in self.log_scale_params:
                    val = np.log10(val if val > 0 else 1e-12)
                processed_label.append(val)
                i += 1

        label_tensor = torch.tensor(processed_label, dtype=torch.float32)

        # Standardize if applicable
        if self.scaler_means is not None and self.scaler_stds is not None:
            label_tensor = label_tensor.clone()
            j = 0  # index into scaler_means/stds
            i = 0  # index into label_tensor
            while i < len(label_tensor):
                param = self.model_params[j]
                if param in ("incl", "phi"):
                    i += 2  # skip sin and cos
                else:
                    label_tensor[i] = (label_tensor[i] - self.scaler_means[j]) / self.scaler_stds[j]
                    i += 1
                j += 1

        else:
            print("[WARNING] No scaling parameters provided. Using raw labels.")

        # ------------------ Convert Data ------------------
        data = torch.tensor(data.astype(np.float32, copy=False), dtype=torch.float32).unsqueeze(0)


        return data, label_tensor

    
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

def get_scalar_label_tensor(label, model_params, log_scale_params):
    scalar_values = []
    for i, param in enumerate(model_params):
        if param in ("incl", "phi"):
            continue
        val = label[i]
        if param in log_scale_params:
            val = np.log10(max(val, 1e-9))
        scalar_values.append(val)
    return torch.tensor(scalar_values, dtype=torch.float32)

# Training and validation file grouping to split the training and validation datasets with respect to the same thermal models
from collections import defaultdict

def training_val_split(data_dir, output_dir="Parameters/", job_id=None):
    def extract_thermal_key(filename):
        pattern = r'([A-Za-z0-9_]+)=([-\d.eE+]+)'
        matches = re.findall(pattern, filename)
        param_dict = {k.lstrip('_'): v for k, v in matches}
        thermal_keys = [param_dict.get(k, "NA") for k in ["D", "L", "ri", "ro", "rr", "p", "np", "edr", "rvar", "phivar"]]
        return tuple(thermal_keys)
    
    output_dir = os.path.join(output_dir, job_id) if job_id else output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_fits = glob.glob(os.path.join(data_dir, "*arcsec.fits"))
    grouped_files = defaultdict(list)
    for path in all_fits:
        fname = os.path.basename(path)
        key = extract_thermal_key(fname)
        grouped_files[key].append(path)

    all_keys = list(grouped_files.keys())
    #random.seed(42)
    random.shuffle(all_keys)
    train_keys = all_keys[:int(0.8 * len(all_keys))]
    val_keys = all_keys[int(0.8 * len(all_keys)):]

    train_files = [f for k in train_keys for f in grouped_files[k]]
    val_files = [f for k in val_keys for f in grouped_files[k]]

    train_list_path = os.path.join(output_dir, "train_file_list.txt")
    val_list_path = os.path.join(output_dir, "val_file_list.txt")

    with open(train_list_path, "w") as f:
        for line in train_files:
            f.write(line + "\n")

    with open(val_list_path, "w") as f:
        for line in val_files:
            f.write(line + "\n")

    return train_list_path, val_list_path

def load_file_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_dataloaders(
    fits_dir,
    original_file_list_path=None,
    scaling_params_path=None,
    wavelength_stride=1,
    load_preprocessed=False,
    preprocessed_dir=None,
    use_local_nvme=False,
    batch_size=16,
    num_workers=32,
    train_sampler=None,
    test_sampler=None,
    model_params=["Dens", "Lum", "radius", "prho"],
    log_scale_params=["Dens", "Lum"],
    job_id=None
):
    # Load file list if given
    file_list = None
    if original_file_list_path and os.path.exists(original_file_list_path):
        with open(original_file_list_path, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]

    dataset = FitsDataset(
        fits_dir=fits_dir,
        file_list=file_list,
        wavelength_stride=wavelength_stride,
        use_local_nvme=use_local_nvme,
        load_preprocessed=load_preprocessed,
        preprocessed_dir=preprocessed_dir or fits_dir,
        model_params=model_params,
        log_scale_params=log_scale_params,
    )

    train_list_path, val_list_path = training_val_split(
        fits_dir,
        output_dir="Parameters/",
        job_id=job_id
    )

    # If using a file list, load it
    train_files, val_files = load_file_list(train_list_path), load_file_list(val_list_path)

    # Split
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Optionally load scaling params
    if scaling_params_path and os.path.exists(scaling_params_path):
        print(f"Loading scaling parameters from: {scaling_params_path}")
        scaling = torch.load(scaling_params_path)
        dataset.set_scaling_params(scaling['means'], scaling['stds'])
    else:
        # Compute scaling parameters if not provided
        print("Calculating scaling parameters...")
        # Extract training indices (required for scaling)
        train_indices = train_dataset.indices if hasattr(train_dataset, 'indices') else list(range(train_size))

        # Compute scaling parameters
        means, stds = calculate_label_scaling(dataset, train_indices)

        # Set scaling inside the dataset
        dataset.set_scaling_params(means, stds)
        print("Parameter means:", means, "stds:", stds)
        # Print scaling parameters with param name
        print("Scaling parameters (means and stds):")
        
        for i, param in enumerate(dataset.model_params):
            print(f"{param}: mean = {means[i]:.4f}, std = {stds[i]:.4f}")

        # Save scaling parameters
        scaling_dict = {"means": means, "stds": stds}
        slurm_id = os.environ.get("SLURM_JOB_ID", "nojob")
        print(f"Saving scaling parameters to: Parameters/{slurm_id}/label_scaling.pt")
        os.makedirs(f"Parameters/{slurm_id}/", exist_ok=True)
        torch.save(scaling_dict, f"Parameters/{slurm_id}/label_scaling.pt")



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

    return train_loader, test_loader, dataset



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
