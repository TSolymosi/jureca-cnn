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
                    for f in files if f.endswith("arcsec.fits") and self._is_valid_fits(os.path.join(root, f))
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
        label_dict = {k.lstrip('_'): float(v) for k, v in matches}
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

            # Normalize label
            #label = (np.array(label) - self.label_min) / (self.label_max - self.label_min + 1e-8)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            # Log scale certain parameters
            for i in self.param_indices_to_log:
                label_tensor[i] = torch.log10(label_tensor[i].clamp(min=1e-9))

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



"""
# Runtime config
fits_dir = '/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data'
wavelength_stride = 1
use_local_nvme = False  # Set to False to use original directory

preprocessed = True  # Set to True to load preprocessed data

if preprocessed == False:
    fits_dir = '/p/scratch/pasta/production_run/24.03.25/firstCNNtest'
    test_mode = False  # set to False for full dataset
    max_subset_files = 100
    # Load file list, optionally limit for test
    all_files = glob.glob(os.path.join(fits_dir, "**", "*arcsec.fits"), recursive=True)
    valid_files = [f for f in all_files if FitsDataset._is_valid_fits(None, f)]

    if test_mode:
        selected_files = random.sample(valid_files, k=min(max_subset_files, len(valid_files)))
    else:
        selected_files = valid_files

    # Initialize Dataset 
    print("Initializing dataset...")
    dataset = FitsDataset(fits_dir, file_list=selected_files, wavelength_stride=wavelength_stride, use_local_nvme=use_local_nvme)
else:
    dataset = dataset = FitsDataset(wavelength_stride=wavelength_stride, use_local_nvme=use_local_nvme, load_preprocessed=True, preprocessed_dir=fits_dir)

# Split and Load 
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 16
start_time = time.time()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=128, pin_memory=False, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=128, pin_memory=False, persistent_workers=True)

end_time = time.time()
print(f"DataLoader creation took {end_time - start_time:.2f} seconds")


def get_dataloaders(batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=128, pin_memory=False, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=128, pin_memory=False, persistent_workers=True)
    return train_loader, test_loader
"""
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
