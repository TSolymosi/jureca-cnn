import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
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
import functools
import json
from typing import Literal
from astropy.io import fits as _fits

# Flush prints immediately
print = functools.partial(print, flush=True)

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

from lightning.pytorch.utilities import rank_zero_only

@rank_zero_only
def print_rank0(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ------------------ Helpers ------------------ #
# Helper function for distributed training to wait for rank 0 to finish a task
def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

# [REMOVED: not used anywhere now]
# def normalize(data, method='minmax'):
#     """Normalize 3D data using specified method"""
#     data = np.nan_to_num(data)
#     if method == 'minmax':
#         lower, upper = np.percentile(data, [1, 99])
#         return np.clip((data - lower) / (upper - lower + 1e-6), 0, 1)
#     elif method == 'zscore':
#         return (data - np.mean(data)) / (np.std(data) + 1e-6)
#     elif method == "root":
#         return np.cbrt(data)
#     elif method == "log":
#         return np.log10(data + 1e-6)
#     else:
#         raise ValueError(f"Unknown normalization method: {method}")

# ------------------ Dataset ------------------ #
class FitsDataset(data.Dataset):
    def __init__(
        self,
        fits_dir=None,
        file_list=None,
        wavelength_stride=1,
        use_local_nvme=True,
        load_preprocessed=False,
        preprocessed_dir='/p/scratch/pasta/CNN/Processed_Data/processed_data',
        model_params=["D", "L", "ro", "rr", "p", "Tlow", "NCH3CN", "plummer_shape"],
        log_scale_params=["D", "L", "NCH3CN"],
        # ------------- CPU-SIDE NOISE/MASKING — DEPRECATED -------------
        # use_cauchy_noise=False,
        # cauchy_mu=0.003,
        # cauchy_sigma=0.0032,
        # cauchy_threshold=0.07,
        # add_noise_level=0.0,
        # snr_threshold=5.0,
        # ---------------------------------------------------------------
        mask_13co=True,
    ):
        self.wavelength_stride = wavelength_stride
        self.load_preprocessed = load_preprocessed
        self.file_list = file_list
        self.model_params = model_params
        self.log_scale_params = log_scale_params
        self.model_params = list(model_params)
        self.log_scale_params = list(log_scale_params)

        self.param_indices_to_log = [self.model_params.index(p) for p in self.log_scale_params if p in self.model_params]

        # --- Add a flag to ensure info is logged only once ---
        #self._info_logged = False
        # --- Call the new diagnostic logging method ---
        #self.log_parameter_info()

        # [REMOVED: CPU-side noise/masking fields — handled on GPU now]
        # self.use_cauchy_noise = use_cauchy_noise
        # self.cauchy_mu = cauchy_mu
        # self.cauchy_sigma = cauchy_sigma
        # self.cauchy_threshold = cauchy_threshold
        # self.add_noise_level = add_noise_level
        # self.snr_threshold = snr_threshold

        self.mask_13co = mask_13co

        if self.load_preprocessed:
            self.data_dir = os.path.join(preprocessed_dir, "data_100")
            self.label_dir = os.path.join(preprocessed_dir, "labels_100")
            self.data_files = sorted(glob.glob(os.path.join(self.data_dir, "data_*.npy")))
            self.labels = np.load(os.path.join(self.label_dir, "labels.npy"))
            self.resolved_file_list = list(self.data_files)
        else:
            self.original_fits_dir = fits_dir
            self.use_local_nvme = use_local_nvme

            if use_local_nvme and os.path.exists("/local/nvme"):
                slurm_id = os.environ.get("SLURM_JOB_ID", "nojob")
                self.fits_dir = f"/local/nvme/{slurm_id}_fits_data"
                if not os.path.exists(self.fits_dir):
                    print_rank0(f"Copying FITS files to local storage: {self.fits_dir}")
                    os.makedirs(self.fits_dir, exist_ok=True)
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        if file_list:
                            self._copy_selected_files(file_list)
                        else:
                            shutil.copytree(fits_dir, self.fits_dir, dirs_exist_ok=True)
                    ddp_barrier()
                else:
                    print_rank0(f"Local NVMe path already exists: {self.fits_dir}")
            else:
                self.fits_dir = fits_dir

            if file_list is not None:
                self.fits_files = file_list
            else:
                print_rank0(f"Searching for FITS files in: {self.fits_dir}")
                self.fits_files = [
                    os.path.join(root, f)
                    for root, _, files in os.walk(self.fits_dir)
                    for f in files if f.endswith("arcsec.fits")
                ]
                print_rank0(f"Found {len(self.fits_files)} valid FITS files in {self.fits_dir}")

            if len(self.fits_files) == 0:
                raise RuntimeError("No FITS files found.")

            self.resolved_file_list = list(self.fits_files)
            # Precompute 13CO mask if needed (deterministic; cheap)
            self._chan_mask = None
            if self.mask_13co and not self.load_preprocessed:
                hdr = _fits.getheader(self.resolved_file_list[0], memmap=True)
                n_ch = int(hdr['NAXIS3']); crval, cdelt, crpix = hdr['CRVAL3'], hdr['CDELT3'], hdr['CRPIX3']
                idx = np.arange(0, n_ch, self.wavelength_stride, dtype=np.float64)
                freq = crval + (idx - (crpix - 1.0)) * cdelt

                FREQ_13CO_HZ, C_MS, vel_ms = 220.4039006e9, 299_792_458.0, 40_000.0
                half_w = FREQ_13CO_HZ * (vel_ms / C_MS)
                self._chan_mask = (freq >= FREQ_13CO_HZ - half_w) & (freq <= FREQ_13CO_HZ + half_w)

            print_rank0("Extracting labels from FITS filenames...")
            self.labels = np.array([self.extract_label(os.path.basename(f)) for f in self.fits_files])

    def _copy_selected_files(self, file_list):
        for src_path in file_list:
            rel_path = os.path.relpath(src_path, self.original_fits_dir)
            dst_path = os.path.join(self.fits_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
    
    def log_parameter_info(self):
        """
        Prints a clear, one-time summary of the parameter configuration.
        This is designed to be called from the __init__ method.
        """
        # First, check if this is the main process and if we haven't logged before.
        if (not dist.is_initialized() or dist.get_rank() == 0) and not getattr(self, '_info_logged', False):
            print("\n" + "="*80)
            print(" " * 25 + "FitsDataset Parameter Configuration")
            print("="*80)
            
            # 1. Model Parameters
            print(f"1. Model Parameters (in order):\n   {self.model_params}")
            print("-" * 80)

            # 2. Log-Scale Parameters
            print(f"2. Log-Scale Parameters (from config):\n   {self.log_scale_params}")
            print("-" * 80)
            
            # 3. Final Calculated Indices
            print(f"3. Final Log-Scale Indices (used internally):\n   {self.param_indices_to_log}")
            
            # Specific check for NCH3CN
            if 'NCH3CN' in self.log_scale_params:
                if 'NCH3CN' in self.model_params:
                    nch_index = self.model_params.index('NCH3CN')
                    if nch_index in self.param_indices_to_log:
                        print("\n[SUCCESS] 'NCH3CN' is correctly identified as a log-scale parameter.")
                    else:
                        print("\n[ERROR] 'NCH3CN' is in log_scale_params, but its index was not found.")
                else:
                    print("\n[WARNING] 'NCH3CN' is in log_scale_params, but not in the main model_params list.")
            else:
                 print("\n[WARNING] 'NCH3CN' is NOT configured as a log-scale parameter.")

            print("="*80 + "\n", flush=True)
            
            # Set the flag so this message doesn't print again.
            self._info_logged = True
        
        # --- THE FIX ---
        # The barrier is now called by ALL processes, regardless of rank.
        # This ensures that processes 1, 2, 3, etc., will wait here until
        # process 0 is finished with its printing block above.
        # This prevents them from racing ahead and ensures synchronized startup.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def extract_label_old(self, filename):
        pattern = r'([A-Za-z0-9_]+)=([-\d.eE+]+)'
        matches = re.findall(pattern, filename)
        label_dict = {k.strip("_").replace("LTE_", ""): float(v) for k, v in matches}
        label = []
        for param in self.model_params:
            if param == "plummer_shape":
                p_val, rr_val = label_dict["p"], label_dict["rr"]
                label.append(p_val * np.log10(rr_val))
            else:
                label.append(label_dict[param])
        return label
    
    def extract_label(self, filename):
        """
        Extract labels from FITS header instead of filename.
        
        Maps FITS header keys to model parameter names.
        """
        # Mapping from model parameter names to FITS header keys
        header_key_map = {
            'M': 'mass',           # New parameter!
            'D': 'dens',
            'L': 'lum',
            'ro': 'ro',
            'p': 'prho',
            'Tlow': 'Tlow',
            'NCH3CN': 'abunch3cn',
            'plummer_shape': None  # Computed from p and ro (not rr anymore!)
        }
        
        # Get full path to FITS file
        fits_path = filename if os.path.isabs(filename) else os.path.join(self.fits_dir, filename)
        
        # Read FITS header
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                header = hdul[0].header
                
                # Extract thermal_params from COMMENT section
                # The header stores parameters as JSON in COMMENT cards
                thermal_params = self._extract_thermal_params_from_header(header)
                
        except Exception as e:
            print(f"[ERROR] Failed to read header from {filename}: {e}")
            raise
        
        # Build label array
        label = []
        for param in self.model_params:
            if param == "plummer_shape":
                # Compute plummer_shape = p * log10(ro)
                # Note: Changed from p * log10(rr) to p * log10(ro)
                p_val = thermal_params.get('prho', thermal_params.get('p'))
                ro_val = thermal_params.get('ro')
                if p_val is None or ro_val is None:
                    raise ValueError(f"Cannot compute plummer_shape: missing p or ro in {filename}")
                label.append(p_val * np.log10(ro_val))
            else:
                # Look up parameter in thermal_params
                header_key = header_key_map.get(param)
                if header_key is None:
                    raise ValueError(f"Unknown parameter: {param}")
                
                value = thermal_params.get(header_key)
                if value is None:
                    raise ValueError(f"Parameter {param} (header key: {header_key}) not found in {filename}")
                
                label.append(float(value))
        
        return label

    def _extract_thermal_params_from_header(self, header):
        """
        Extract parameters from FITS header COMMENT section.
        
        The header contains a JSON-like structure in COMMENT cards.
        We need to parse it to extract thermal_params.
        """
        # Collect all COMMENT cards
        comments = []
        for card in header.cards:
            if card.keyword == 'COMMENT':
                comments.append(card.value)
        
        # Join into single string
        comment_text = ' '.join(comments)
        
        # Try to parse as JSON
        # The header has: COMMENT { ... COMMENT     "mass": 10.01467, ... COMMENT }
        try:
            import json
            import re
            
            # Extract the JSON block between outermost braces
            # Remove "COMMENT" prefixes that might be embedded
            json_text = re.search(r'\{.*\}', comment_text, re.DOTALL)
            if json_text:
                json_str = json_text.group(0)
                # Parse JSON
                params = json.loads(json_str)
                
                # If thermal_params is nested, extract it
                if 'thermal_params' in params:
                    return params['thermal_params']
                else:
                    return params
            else:
                raise ValueError("Could not find JSON structure in COMMENT")
        
        except Exception as e:
            print(f"[WARNING] Failed to parse thermal_params from header: {e}")
            
            # Fallback: Parse individual lines
            # Look for patterns like: COMMENT     "mass": 10.01467,
            params = {}
            for comment in comments:
                match = re.search(r'"(\w+)":\s*([\d.e+-]+)', comment)
                if match:
                    key = match.group(1)
                    value = float(match.group(2))
                    params[key] = value
            
            return params

    def set_scaling_params(self, means, stds):
        self.scaler_means, self.scaler_stds = means, stds

    def inverse_transform_labels(self, scaled_labels):
        """
        Inverse transform standardized + log-scaled labels back to original physical units.
        """
        if self.scaler_means is None or self.scaler_stds is None:
            print("[WARNING] inverse_transform_labels called without scaler parameters.")
            return scaled_labels

        means, stds = self.scaler_means, self.scaler_stds

        if not isinstance(scaled_labels, torch.Tensor):
            scaled_labels = torch.tensor(scaled_labels)
        scaled_labels = scaled_labels.to(means.device)

        unscaled = scaled_labels * stds + means
        original = unscaled.clone()

        for idx in self.param_indices_to_log:
            if original.ndim == 1:
                original[idx] = torch.pow(10.0, original[idx])
            else:
                original[:, idx] = torch.pow(10.0, original[:, idx])

        return original

    def inverse_transform_labels_with_uncertainty_logspace(self, scaled_mu, scaled_sigma):
        """
        Returns log-scale parameters in LOG SPACE for z-score calculations.
        This is the ONLY correct way to calculate z-scores.
        """
        if self.scaler_means is None or self.scaler_stds is None:
            return scaled_mu, scaled_sigma

        means, stds = self.scaler_means, self.scaler_stds

        if not isinstance(scaled_mu, torch.Tensor):
            scaled_mu = torch.from_numpy(scaled_mu)
        if not isinstance(scaled_sigma, torch.Tensor):
            scaled_sigma = torch.from_numpy(scaled_sigma)

        scaled_mu, scaled_sigma = scaled_mu.to(means.device), scaled_sigma.to(means.device)

        # Simply undo z-score normalization - stay in log space
        unscaled_mu = scaled_mu * stds + means
        unscaled_sigma = scaled_sigma * stds
        
        return unscaled_mu, unscaled_sigma

    def inverse_transform_labels_with_uncertainty(self, scaled_mu, scaled_sigma):
        if self.scaler_means is None or self.scaler_stds is None:
            return scaled_mu, scaled_sigma

        means, stds = self.scaler_means, self.scaler_stds
        
        if not isinstance(scaled_mu, torch.Tensor):
            scaled_mu = torch.from_numpy(scaled_mu)
        if not isinstance(scaled_sigma, torch.Tensor):
            scaled_sigma = torch.from_numpy(scaled_sigma)

        scaled_mu, scaled_sigma = scaled_mu.to(means.device), scaled_sigma.to(means.device)

        # First undo z-score normalization
        unscaled_mu = scaled_mu * stds + means
        unscaled_sigma = scaled_sigma * stds  # This is correct

        mu_orig, sigma_orig = unscaled_mu.clone(), unscaled_sigma.clone()

        # FIXED: Correct uncertainty propagation for log-transformed parameters
        for idx in self.param_indices_to_log:
            mu_log = unscaled_mu[:, idx]      # This is log10(value)
            sigma_log = unscaled_sigma[:, idx] # This is σ in log-space
            
            # Convert log-space mean to linear space
            mu_linear = torch.pow(10.0, mu_log)
            
            # Correct uncertainty propagation:
            # For y = 10^x, dy/dx = y * ln(10)
            # So σ_linear ≈ |y * ln(10)| * σ_log
            sigma_linear = torch.abs(mu_linear * np.log(10)) * sigma_log
            
            mu_orig[:, idx] = mu_linear
            sigma_orig[:, idx] = sigma_linear

        return mu_orig, sigma_orig
    
    def get_frequency_axis(self):
        """
        Computes and returns the frequency axis for the dataset.
        Returns a numpy array of the frequency axis in GHz, or None if it cannot be determined.
        """
        if self.load_preprocessed or not hasattr(self, "fits_files") or not self.fits_files:
            return None

        try:
            first_fits_path = self.fits_files[0]
            with fits.open(first_fits_path, memmap=True) as hdul:
                header = hdul[0].header

            n_chans = header['NAXIS3']
            crval = header['CRVAL3']
            cdelt = header['CDELT3']
            crpix = header['CRPIX3']

            freq_axis_hz = crval + (np.arange(n_chans) - (crpix - 1)) * cdelt
            if self.wavelength_stride > 1:
                freq_axis_hz = freq_axis_hz[::self.wavelength_stride]
            return freq_axis_hz / 1e9
        except Exception as e:
            print(f"[WARNING] Could not determine frequency axis. Error: {e}")
            return None

    def __len__(self):
        return len(self.data_files if self.load_preprocessed else self.fits_files)

    def __getitem__(self, idx):
        # print(f"[DEBUG] __getitem__ idx={idx}", flush=True)

        if self.load_preprocessed:
            data = np.load(self.data_files[idx])
            label = self.labels[idx]
            # fits_header = None
        else:
            fits_path = self.fits_files[idx]
            # print(f"[DEBUG] Opening FITS: {fits_path}", flush=True)

            with fits.open(fits_path, memmap=True) as hdul:
                mm = hdul[0].data  # memmap
                if self.wavelength_stride > 1:
                    data = np.array(mm[::self.wavelength_stride, :, :], dtype=np.float32, copy=True)
                else:
                    data = np.array(mm, dtype=np.float32, copy=True)

            label = self.labels[idx]
            # print(f"[DEBUG] Extracted label: {label}", flush=True)

        # [REMOVED: alternate frequency-axis path + NaN masking using freq_axis]
        # if self.mask_13co and freq_axis_hz is not None:
        #     ...

        # Deterministic 13CO channel mask (cheap; keep)
        if self.mask_13co and getattr(self, "_chan_mask", None) is not None:
            data[self._chan_mask, :, :] = 0.0  # zero out masked channels; avoid NaNs

        # ---------------- CPU-SIDE NOISE — REMOVED ----------------
        # rng = np.random
        # noise_rms = 0.0
        # if self.use_cauchy_noise:
        #     rms = self.cauchy_mu + self.cauchy_sigma * np.abs(rng.standard_cauchy(size=1))
        #     while rms > self.cauchy_threshold:
        #         rms = self.cauchy_mu + self.cauchy_sigma * np.abs(rng.standard_cauchy(size=1))
        #     noise_rms = float(rms)
        #     data = data + rng.normal(0.0, noise_rms, size=data.shape).astype(np.float32)
        # elif self.add_noise_level > 0.0:
        #     noise_rms = float(self.add_noise_level)
        #     data = data + rng.normal(0.0, noise_rms, size=data.shape).astype(np.float32)
        # ----------------------------------------------------------

        # NaN repair (keep: rare but helpful; vectorized and cheap)
        repair_nans = True
        if np.isnan(data).any() and repair_nans:
            finite = np.isfinite(data)
            count = finite.sum(axis=(1, 2), keepdims=True)
            summ = np.where(finite, data, 0.0).sum(axis=(1, 2), keepdims=True)
            means = np.divide(summ, count, out=np.zeros_like(summ), where=count > 0)
            data[~finite] = np.broadcast_to(means, data.shape)[~finite]

        # --- Labels ---
        label_t = torch.as_tensor(label, dtype=torch.float32)
        for i in self.param_indices_to_log:
            label_t[i] = torch.log10(label_t[i].clamp(min=1e-12))
        if hasattr(self, "scaler_means") and self.scaler_means is not None:
            label_t = (label_t - self.scaler_means) / self.scaler_stds

        x = torch.from_numpy(data.astype(np.float32, copy=False)).unsqueeze(0)
        # [CHANGED RETURN]: (x, label_t) only; noise_rms removed
        return x, label_t

# ------------------ Scaling ------------------ #
def calculate_label_scaling(full_dataset, indices):
    labels = []
    if isinstance(full_dataset, torch.utils.data.Subset):
        subset, original_dataset = full_dataset, full_dataset.dataset
        orig_indices = [subset.indices[i] for i in indices]
    else:
        original_dataset, orig_indices = full_dataset, indices

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
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=True).clamp_min(1e-12)

# ------------------ Dataloaders ------------------ #
def _ensure_dir(path): os.makedirs(os.path.dirname(path), exist_ok=True)
def _derive_aux_paths_old(scaling_params_path: str):
    base_dir = os.path.dirname(scaling_params_path)
    return (os.path.join(base_dir, "split_indices.pt"),
            os.path.join(base_dir, "subset_indices.pt"),
            os.path.join(base_dir, "file_list.json"))

def _derive_aux_paths(base_path: str, fraction: float):
    """Derives paths for split, subset, and filelist files from a base path."""
    base_dir = os.path.dirname(base_path)
    
    # Define a suffix for filenames if a fraction is used.
    # e.g., for fraction=0.2, suffix becomes "_frac0p2"
    suffix = ""
    if 0 < fraction < 1.0:
        # Replace dot with 'p' for cleaner filenames.
        suffix = f"_frac{fraction:.2f}".replace('.', 'p')

    # Create unique filenames using the suffix.
    split_path = os.path.join(base_dir, f"split{suffix}.pt")
    subset_path = os.path.join(base_dir, f"subset{suffix}.pt")
    
    # The file list is independent of the subset fraction.
    filelist_path = os.path.join(base_dir, "file_list.json")
    
    return split_path, subset_path, filelist_path

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
    model_params=("D","L","ro","rr","p","Tlow","NCH3CN","plummer_shape"),
    log_scale_params=("D","L","NCH3CN"),
    data_subset_fraction=1.0,
    seed: int = 42,
    prep_mode: Literal["prepare","load"] = "load",
    mask_13co: bool = True,
):
    """
    Creates and configures the training and validation dataloaders.

    This function operates in two modes:
    1. 'prepare': A one-time setup step. It determines the data subset,
       splits it into training/validation sets, calculates normalization
       scalers, and saves all this information to disk. This ensures that
       all subsequent runs are perfectly reproducible.
    2. 'load': The standard mode for training runs. It loads the pre-computed
       subset, split, and scaler files from disk to construct the dataloaders.
    """
    print(f"[DATALOADER] Starting in '{prep_mode}' mode.")
    assert scaling_params_path is not None, "A `scaling_params_path` must be provided to store/load dataset info."

    # --- Generate unique, fraction-aware paths for index files ---
    # This is the core fix: filenames will now include the fraction (e.g., "split_frac0p2.pt").
    split_path, subset_path, filelist_path = _derive_aux_paths(scaling_params_path, data_subset_fraction)

    # Load the initial list of FITS files to use.
    file_list = None
    if original_file_list_path and os.path.exists(original_file_list_path):
        with open(original_file_list_path, "r") as f:
            file_list = [line.strip() for line in f if line.strip()]

    # ========================================================================
    #  PREPARE BRANCH: Run this once to set up the dataset subset and split.
    # ========================================================================
    if prep_mode == "prepare":
        print(f"[DATALOADER] Preparing dataset indices for subset fraction: {data_subset_fraction}")

        # Before doing anything, check if the final output files for this configuration
        # already exist. If they do, this preparation step has already been completed.
        if os.path.exists(split_path) and os.path.exists(subset_path) and os.path.exists(scaling_params_path):
            print(f"[DATALOADER] PREPARE MODE: Found existing metadata files. Preparation is already complete. Skipping.")
            return # Exit the function immediately.
        
        
        
        
        # This expensive setup should only be run on the main process in a distributed setup.
        rank_zero = (not dist.is_initialized()) or (dist.get_rank() == 0)

        # Create a temporary full dataset instance to get the total number of files.
        ds_full = FitsDataset(
            fits_dir=fits_dir, file_list=file_list,
            wavelength_stride=wavelength_stride,
            use_local_nvme=use_local_nvme,
            load_preprocessed=load_preprocessed,
            preprocessed_dir=preprocessed_dir or fits_dir,
            model_params=list(model_params),
            log_scale_params=list(log_scale_params),
            mask_13co=mask_13co,
            # [REMOVED: pass-through of CPU noise args]
            # use_cauchy_noise=use_cauchy_noise,
            # cauchy_mu=cauchy_mu, cauchy_sigma=cauchy_sigma, cauchy_threshold=cauchy_threshold,
            # add_noise_level=add_noise_level, snr_threshold=snr_threshold,
        )
        num_total = len(ds_full)
        
        # Use a seeded random number generator for reproducibility.
        rng = np.random.default_rng(seed)

        # Check if the correct, fraction-specific index files already exist.
        if os.path.exists(split_path) and os.path.exists(subset_path):
            print(f"[DATALOADER] Loading existing indices from {subset_path} and {split_path}")
            subset_indices = torch.load(subset_path)["subset_indices"]
            split = torch.load(split_path)
            train_idx, val_idx = split["train_indices"], split["val_indices"]
        else:
            print("[DATALOADER] Generating new subset and split indices...")
            # If files don't exist, create the deterministic subset.
            if data_subset_fraction >= 1.0:
                subset_indices = np.arange(num_total)
            else:
                num_to_select = int(num_total * data_subset_fraction)
                subset_indices = np.sort(rng.choice(num_total, num_to_select, replace=False))
            
            # Now, create a deterministic train/validation split from that subset.
            n_train = int(0.8 * len(subset_indices))
            perm = rng.permutation(len(subset_indices))
            train_idx = subset_indices[perm[:n_train]]
            val_idx = subset_indices[perm[n_train:]]

        # Calculate normalization scalers using ONLY the training indices to prevent data leakage.
        means, stds = calculate_label_scaling(ds_full, train_idx.tolist())
        ds_full.set_scaling_params(means, stds)

        # Save all the generated information to disk on the main process.
        if rank_zero:
            print(f"[DATALOADER] Saving indices and scalers for fraction {data_subset_fraction}")
            torch.save({"means": means, "stds": stds, "params": list(model_params)}, scaling_params_path)
            torch.save({"train_indices": train_idx, "val_indices": val_idx}, split_path)
            torch.save({"subset_indices": subset_indices}, subset_path)
            
            # Save the list of files that were actually discovered and used.
            with open(filelist_path, "w") as f:
                json.dump(ds_full.resolved_file_list, f)

        print("[DATALOADER] 'prepare' mode finished.")
        return # Exit after preparation is done.

    # ========================================================================
    #  LOAD BRANCH: This is the standard path for all training runs.
    # ========================================================================
    print(f"[DATALOADER] Loading dataset with subset fraction: {data_subset_fraction}")

    # Load the pre-computed information.
    scaling = torch.load(scaling_params_path)
    split = torch.load(split_path) # Loads the correct, fraction-specific file.
    with open(filelist_path, "r") as f:
        file_list_loaded = json.load(f)

    # Create the full dataset object. It will hold all data in memory, but we will
    # only access the specified indices via torch.utils.data.Subset.
    ds_full = FitsDataset(
        fits_dir=fits_dir, file_list=file_list_loaded,
        wavelength_stride=wavelength_stride,
        use_local_nvme=use_local_nvme,
        load_preprocessed=load_preprocessed,
        preprocessed_dir=preprocessed_dir or fits_dir,
        model_params=list(model_params),
        log_scale_params=list(log_scale_params),
        mask_13co=mask_13co,
    )
    ds_full.set_scaling_params(scaling["means"], scaling["stds"])

    train_idx, val_idx = split["train_indices"], split["val_indices"]
    
    # Use torch.utils.data.Subset to create lightweight "views" of the full dataset
    # containing only the desired training and validation samples.
    train_dataset = Subset(ds_full, train_idx)
    val_dataset = Subset(ds_full, val_idx)
    
    print(f"[DATALOADER] Full dataset size: {len(ds_full)}")
    print(f"[DATALOADER] Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create the final DataLoader objects.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None), # Shuffle if no custom sampler is provided.
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # Never shuffle validation data.
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print("[DATALOADER] Dataloaders created successfully.")
    return train_loader, val_loader, ds_full

# ------------------ CLI ------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--scaling-params-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    train_loader, val_loader, dataset = create_dataloaders(
        fits_dir=args.data_dir,
        scaling_params_path=args.scaling_params_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Dataset size: {len(dataset)}")
    for batch in train_loader:
        print("Sample batch:", batch[0].shape, batch[1].shape)
        break
