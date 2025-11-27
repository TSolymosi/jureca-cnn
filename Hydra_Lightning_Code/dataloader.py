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
from multiprocessing import Pool
from functools import partial

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
        model_params=["M", "D", "L", "ro", "p", "Tlow", "NCH3CN"],
        log_scale_params=["M", "D", "L", "NCH3CN"],
        # ------------- CPU-SIDE NOISE/MASKING — DEPRECATED -------------
        # use_cauchy_noise=False,
        # cauchy_mu=0.003,
        # cauchy_sigma=0.0032,
        # cauchy_threshold=0.07,
        # add_noise_level=0.0,
        # snr_threshold=5.0,
        # ---------------------------------------------------------------
        mask_13co=True,
        labels_cache_dir=None,  # NEW: Where to save/load label cache
        num_workers_extract=32,  # NEW: Parallel extraction workers
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
        self.labels_cache_dir = labels_cache_dir
        self.num_workers_extract = num_workers_extract

        # Header key mapping
        self.header_key_map = {
            'M': 'mass',
            'D': 'dens',
            'L': 'lum',
            'ro': 'ro',
            'p': 'prho',
            'Tlow': 'Tlow',
            'NCH3CN': 'abunch3cn',
            'plummer_shape': None  # Computed
        }

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

                # Setup label cache directory on local NVMe
                if labels_cache_dir is None:
                    self.labels_cache_dir = f"/local/nvme/{slurm_id}_labels_cache"
                else:
                    self.labels_cache_dir = labels_cache_dir

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
                if labels_cache_dir is None:
                    self.labels_cache_dir = os.path.join(fits_dir, ".labels_cache")
                else:
                    self.labels_cache_dir = labels_cache_dir

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
            #self.labels = np.array([self.extract_label(os.path.basename(f)) for f in self.fits_files])
            self.labels = self._extract_labels_with_cache()

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
    
    

    def _extract_labels_with_cache(self):
        """
        Extract labels with intelligent caching.
        
        SIMPLIFIED: Only rank 0 does extraction, others just wait and load.
        """
        cache_path = self._get_cache_path()
        
        # Rank 0: Extract or load cache
        if not dist.is_initialized() or dist.get_rank() == 0:
            if os.path.exists(cache_path):
                print_rank0(f"Loading cached labels from: {cache_path}")
                try:
                    labels = np.load(cache_path)
                    
                    # Verify cache integrity
                    expected_shape = (len(self.fits_files), len(self.model_params))
                    if labels.shape == expected_shape:
                        print_rank0(f"Successfully loaded {len(labels)} labels from cache")
                        # Don't call barrier here - we're in multiprocessing context
                        return labels
                    else:
                        print_rank0(f"[WARNING] Cache shape mismatch: {labels.shape} vs {expected_shape}. Re-extracting.")
                except Exception as e:
                    print_rank0(f"[WARNING] Failed to load cache: {e}. Re-extracting.")
            
            # Cache doesn't exist or is invalid: extract labels
            print_rank0(f"Extracting labels from {len(self.fits_files)} FITS files...")
            print_rank0(f"Using {self.num_workers_extract} parallel workers")
            
            labels = self._extract_labels_parallel()
            
            # Save cache
            os.makedirs(self.labels_cache_dir, exist_ok=True)
            np.save(cache_path, labels)
            print_rank0(f"Saved label cache to: {cache_path}")
            
            return labels
        
        else:
            # Other ranks: Wait for rank 0 to finish, then load
            print(f"[Rank {dist.get_rank()}] Waiting for rank 0 to extract/cache labels...")
            
            # Poll until cache file exists (created by rank 0)
            import time
            max_wait_seconds = 3600  # 1 hour timeout
            wait_interval = 5  # Check every 5 seconds
            elapsed = 0
            
            while not os.path.exists(cache_path):
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                if elapsed >= max_wait_seconds:
                    raise RuntimeError(f"Rank {dist.get_rank()} timed out waiting for label cache")
                
                if elapsed % 60 == 0:  # Print every minute
                    print(f"[Rank {dist.get_rank()}] Still waiting for labels... ({elapsed}s elapsed)")
            
            # Cache file exists, load it
            print(f"[Rank {dist.get_rank()}] Loading labels from cache...")
            labels = np.load(cache_path)
            print(f"[Rank {dist.get_rank()}] Loaded {len(labels)} labels")
            
            return labels

    def _get_cache_path(self):
        """
        Generate unique cache filename based on dataset configuration.
        
        This ensures different parameter combinations get different caches.
        """
        import hashlib
        
        # Create a unique identifier from the configuration
        config_str = (
            f"params={','.join(self.model_params)}_"
            f"log={','.join(self.log_scale_params)}_"
            f"nfiles={len(self.fits_files)}"
        )
        
        # Hash the file list to detect changes
        file_list_str = ''.join(sorted([os.path.basename(f) for f in self.fits_files]))
        file_hash = hashlib.md5(file_list_str.encode()).hexdigest()[:8]
        
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        cache_filename = f"labels_{config_hash}_{file_hash}.npy"
        cache_path = os.path.join(self.labels_cache_dir, cache_filename)
        
        return cache_path

    def _extract_labels_parallel(self):
        """Extract labels using parallel processing."""
        from multiprocessing import Pool
        from functools import partial
        
        # Prepare extraction function
        extract_func = partial(
            self._extract_one_label_static,
            fits_dir=self.fits_dir,
            model_params=self.model_params,
            header_key_map=self.header_key_map
        )
        
        # Use progress bar if tqdm is available
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
        
        labels = []
        with Pool(self.num_workers_extract) as pool:
            if use_tqdm:
                labels = list(tqdm(
                    pool.imap(extract_func, self.fits_files),
                    total=len(self.fits_files),
                    desc="Extracting labels"
                ))
            else:
                for i, label in enumerate(pool.imap(extract_func, self.fits_files)):
                    labels.append(label)
                    if (i + 1) % 1000 == 0:
                        print_rank0(f"Processed {i+1}/{len(self.fits_files)} files")
        
        return np.array(labels)
    
    @staticmethod
    def _extract_one_label_static(filename, fits_dir, model_params, header_key_map):
        """
        Static method for extracting one label (for parallel execution).
        """
        fits_path = filename if os.path.isabs(filename) else os.path.join(fits_dir, filename)
        
        try:
            with _fits.open(fits_path, memmap=True) as hdul:
                header = hdul[0].header
                thermal_params = FitsDataset._extract_thermal_params_from_header_static(header)
            
            # Build label array
            label = []
            for param in model_params:
                if param == "plummer_shape":
                    # Try multiple possible names for p
                    p_val = thermal_params.get('prho') or thermal_params.get('p')
                    ro_val = thermal_params.get('ro') or thermal_params.get('r_out')
                    
                    if p_val is None or ro_val is None:
                        raise ValueError(
                            f"Cannot compute plummer_shape: p={p_val}, ro={ro_val}. "
                            f"Available keys: {list(thermal_params.keys())}"
                        )
                    label.append(float(p_val) * np.log10(float(ro_val)))
                else:
                    # Look up the header key
                    header_key = header_key_map.get(param)
                    if header_key is None:
                        raise ValueError(f"Unknown parameter: {param}")
                    
                    # Try to find the value (with fallbacks for name variations)
                    value = thermal_params.get(header_key)
                    
                    # If not found, try alternative names
                    if value is None:
                        # Define fallback names
                        fallback_names = {
                            'mass': ['mass', 'M', 'stellar_mass'],
                            'dens': ['dens', 'D', 'density'],
                            'lum': ['lum', 'L', 'luminosity'],
                            'ro': ['ro', 'r_out', 'outer_radius'],
                            'prho': ['prho', 'p', 'rho_power'],
                            'Tlow': ['Tlow', 'T_low', 'lower_temp'],
                            'abunch3cn': ['abunch3cn', 'NCH3CN', 'ch3cn_abundance'],
                        }
                        
                        if header_key in fallback_names:
                            for alt_name in fallback_names[header_key]:
                                value = thermal_params.get(alt_name)
                                if value is not None:
                                    break
                    
                    if value is None:
                        raise ValueError(
                            f"Parameter {param} (header key: {header_key}) not found. "
                            f"Available keys: {list(thermal_params.keys())}"
                        )
                    
                    label.append(float(value))
            
            return label
        
        except Exception as e:
            print(f"[ERROR] Failed to extract label from {os.path.basename(filename)}: {e}")
            # Return NaNs to indicate failure
            return [np.nan] * len(model_params)
    
    @staticmethod
    def _extract_thermal_params_from_header_static(header):
        """
        Extract thermal parameters from FITS header.
        
        Extracts from TOP-LEVEL JSON, not from nested thermal_params.
        """
        import re
        import json
        
        # Collect all COMMENT cards
        comments = []
        for card in header.cards:
            if card.keyword == 'COMMENT':
                comments.append(card.value)
        
        # Join into single string
        comment_text = ' '.join(comments)
        
        # Try to parse as JSON
        try:
            # Find the JSON block
            json_match = re.search(r'\{.*\}', comment_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Parse JSON
                full_params = json.loads(json_str)
                
                # CRITICAL FIX: Extract ALL top-level parameters
                # Don't go into thermal_params - that doesn't have dens!
                params = {}
                for key, value in full_params.items():
                    # Skip nested dicts and metadata fields
                    if isinstance(value, dict):
                        continue
                    # Skip non-numeric metadata
                    if key in ['finished', 'error', 'full_finished', 'full_error']:
                        continue
                    
                    # Add numeric parameters
                    if isinstance(value, (int, float)):
                        params[key] = value
                    elif isinstance(value, str):
                        # Try to parse as float if it's a number string
                        try:
                            params[key] = float(value)
                        except ValueError:
                            pass  # Skip non-numeric strings
                
                return params
                
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parse failed: {e}")
        except Exception as e:
            print(f"[WARNING] Unexpected error during JSON parse: {e}")
        
        # Fallback: Parse individual lines
        print("[WARNING] Falling back to line-by-line parameter extraction")
        params = {}
        
        for comment in comments:
            # Match patterns like: COMMENT     "dens": 511269597.8625401,
            match = re.search(r'"(\w+)":\s*([-+]?[\d.e+-]+)', comment)
            if match:
                key = match.group(1)
                value_str = match.group(2)
                try:
                    value = float(value_str)
                    params[key] = value
                except ValueError:
                    pass
        
        if not params:
            raise ValueError("No parameters could be extracted from FITS header!")
        
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
    model_params=("M","D","L","ro","p","Tlow","NCH3CN"),
    log_scale_params=("M","D","L","NCH3CN"),
    data_subset_fraction=1.0,
    seed: int = 42,
    prep_mode: Literal["prepare","load"] = "load",
    mask_13co: bool = True,
    labels_cache_dir=None,  # NEW: Cache directory for labels
    num_workers_extract=32,  # NEW: Parallel workers for extraction
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
    # Determine cache directory
    if labels_cache_dir is None and scaling_params_path is not None:
        # Put cache next to scaling params
        labels_cache_dir = os.path.join(os.path.dirname(scaling_params_path), ".labels_cache")
    

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
            labels_cache_dir=labels_cache_dir,  # NEW
            num_workers_extract=num_workers_extract,  # NEW
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
