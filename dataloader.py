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
def ddp_barrier():
    with suppress(Exception):
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
        model_params=["D", "L", "rr", "p"],
        log_scale_params=["D", "L"],
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
        self.param_indices_to_log = [self.model_params.index(p) for p in self.log_scale_params if p in self.model_params]

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

    def extract_label(self, filename):
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

    def inverse_transform_labels_with_uncertainty(self, scaled_mu, scaled_sigma):
        """
        Inverse transform predicted means and uncertainties back to original units.
        """
        if self.scaler_means is None or self.scaler_stds is None:
            print("[WARNING] inverse_transform_labels_with_uncertainty called without scaler parameters.")
            return scaled_mu, scaled_sigma

        means, stds = self.scaler_means, self.scaler_stds

        if not isinstance(scaled_mu, torch.Tensor):
            scaled_mu = torch.from_numpy(scaled_mu)
        if not isinstance(scaled_sigma, torch.Tensor):
            scaled_sigma = torch.from_numpy(scaled_sigma)

        scaled_mu, scaled_sigma = scaled_mu.to(means.device), scaled_sigma.to(means.device)

        unscaled_mu = scaled_mu * stds + means
        unscaled_sigma = scaled_sigma * stds

        mu_orig, sigma_orig = unscaled_mu.clone(), unscaled_sigma.clone()

        for idx in self.param_indices_to_log:
            mu_log, sigma_log = unscaled_mu[:, idx], unscaled_sigma[:, idx]
            upper, lower = torch.pow(10.0, mu_log + sigma_log), torch.pow(10.0, mu_log - sigma_log)
            mu_orig[:, idx] = torch.pow(10.0, mu_log)
            sigma_orig[:, idx] = (upper - lower) / 2.0

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
def _derive_aux_paths(scaling_params_path: str):
    base_dir = os.path.dirname(scaling_params_path)
    return (os.path.join(base_dir, "split_indices.pt"),
            os.path.join(base_dir, "subset_indices.pt"),
            os.path.join(base_dir, "file_list.json"))

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
    # ---------------- DEPRECATED (CPU) ----------------
    # use_cauchy_noise: bool = True,
    # cauchy_mu: float = 0.003,
    # cauchy_sigma: float = 0.0032,
    # cauchy_threshold: float = 0.07,
    # add_noise_level: float = 0.0,
    # snr_threshold: float = 5.0
    # --------------------------------------------------
):
    print("[DEBUG] Entering create_dataloaders", flush=True)

    assert scaling_params_path is not None, "scaling_params_path must be provided"

    print("[DEBUG] scaling_params_path:", scaling_params_path, flush=True)

    _ensure_dir(scaling_params_path)
    split_path, subset_path, filelist_path = _derive_aux_paths(scaling_params_path)
    print("[DEBUG] Derived aux paths:", split_path, subset_path, filelist_path, flush=True)

    file_list = None
    if filelist_path and os.path.exists(filelist_path):
        with open(filelist_path, "r") as f:
            file_list = json.load(f)
        print(f"[DEBUG] Loaded existing file_list ({len(file_list)} files) from {filelist_path}", flush=True)
    else:
        if original_file_list_path and os.path.exists(original_file_list_path):
            with open(original_file_list_path, "r") as f:
                file_list = [line.strip() for line in f if line.strip()]
            print(f"[DEBUG] Loaded file_list ({len(file_list)} files)", flush=True)

    if prep_mode == "prepare":
        print("[DEBUG] Prep mode = prepare", flush=True)

        rank_zero = (not dist.is_initialized()) or (dist.get_rank() == 0)

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
        rng = np.random.default_rng(seed)

        if os.path.exists(split_path) and os.path.exists(subset_path):
            subset_indices = torch.load(subset_path)["subset_indices"]
            split = torch.load(split_path)
            train_idx, val_idx = split["train_indices"], split["val_indices"]
        else:
            subset_indices = np.arange(num_total) if data_subset_fraction >= 1.0 \
                else np.sort(rng.choice(num_total, int(num_total * data_subset_fraction), replace=False))

            n_train = int(0.8 * len(subset_indices))
            perm = rng.permutation(len(subset_indices))
            train_idx, val_idx = subset_indices[perm[:n_train]], subset_indices[perm[n_train:]]

        # Compute scalers from *current* train split
        means, stds = calculate_label_scaling(ds_full, train_idx.tolist())
        ds_full.set_scaling_params(means, stds)

        if rank_zero:
            scaling_dict = {"means": means, "stds": stds, "params": list(model_params)}
            torch.save(scaling_dict, scaling_params_path)
            torch.save({"train_indices": train_idx, "val_indices": val_idx}, split_path)
            torch.save({"subset_indices": subset_indices}, subset_path)
            with open(filelist_path, "w") as f:
                json.dump(file_list or ds_full.resolved_file_list, f)

        print("[DEBUG] Finished prepare branch", flush=True)
        return

    # --------------------------
    # LOAD branch
    # --------------------------
    print("[DEBUG] Prep mode = load", flush=True)

    rank_zero = (not dist.is_initialized()) or (dist.get_rank() == 0)

    scaling = torch.load(scaling_params_path)
    print("[DEBUG] Loaded scaling params", flush=True)

    split = torch.load(split_path)
    print("[DEBUG] Loaded split indices", flush=True)

    with open(filelist_path, "r") as f:
        file_list_loaded = json.load(f)
    print(f"[DEBUG] Loaded {len(file_list_loaded)} file paths from json", flush=True)

    ds_full = FitsDataset(
        fits_dir=fits_dir, file_list=file_list_loaded,
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
    print("[DEBUG] Constructed FitsDataset", flush=True)

    # --- PARAM-ORDER-AWARE SCALER LOADING ---
    loaded_params = scaling.get("params", None)
    current_params = list(model_params)
    train_idx, val_idx = split["train_indices"], split["val_indices"]

    @rank_zero_only
    def _print_scalers(tag, means_t, stds_t):
        means_np = means_t.cpu().numpy()
        stds_np = stds_t.cpu().numpy()
        for i, p in enumerate(current_params):
            print(f"[SCALE {tag}] {p}: mean={means_np[i]:.3f}, std={stds_np[i]:.4g}", flush=True)

    if (loaded_params is None) or (loaded_params != current_params):
        print("[WARN] Scaling params param-order mismatch or missing. Recomputing on current train split…", flush=True)
        means, stds = calculate_label_scaling(ds_full, list(map(int, train_idx)))
        ds_full.set_scaling_params(means, stds)
        if rank_zero:
            torch.save({"means": means, "stds": stds, "params": current_params}, scaling_params_path)
        _print_scalers("RECOMP", means, stds)
    else:
        ds_full.set_scaling_params(scaling["means"], scaling["stds"])
        _print_scalers("LOADED", scaling["means"], scaling["stds"])

    # Sampler/shuffle parity
    shuffle_train = (train_sampler is None)
    train_dataset = torch.utils.data.Subset(ds_full, train_idx)
    val_dataset   = torch.utils.data.Subset(ds_full, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=1,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=1,
        drop_last=False
    )

    print("[DEBUG] Finished create_dataloaders", flush=True)
    return train_loader, val_loader, ds_full

 # ------------------ Test-only: load ALL files ------------------ #
def create_test_loader(
    fits_dir,
    scaling_params_path,
    wavelength_stride=1,
    load_preprocessed=False,
    preprocessed_dir=None,
    use_local_nvme=False,
    batch_size=16,
    num_workers=8,
    model_params=("D","L","ro","rr","p","Tlow","NCH3CN","plummer_shape"),
    log_scale_params=("D","L","NCH3CN"),
    mask_13co=True,
    test_sampler=None,
):
    """
    Build a DataLoader over *all* files in `fits_dir` (no subset, no split),
    using the label scalers saved during training.
    """
    # Load scalers saved by training
    scaling = torch.load(scaling_params_path)
    loaded_params  = scaling.get("params", None)
    current_params = list(model_params)
    if (loaded_params is None) or (loaded_params != current_params):
        raise ValueError(
            "[TEST] The parameter order in your data.model_params does not "
            "match the scalers file. Please use the same order as training.\n"
            f"scalers params: {loaded_params}\ncurrent: {current_params}"
        )

    # Build a dataset from *all* files found under fits_dir
    ds_full = FitsDataset(
        fits_dir=fits_dir, file_list=None,   # None -> enumerate everything
        wavelength_stride=wavelength_stride,
        use_local_nvme=use_local_nvme,
        load_preprocessed=load_preprocessed,
        preprocessed_dir=preprocessed_dir or fits_dir,
        model_params=current_params,
        log_scale_params=list(log_scale_params),
        mask_13co=mask_13co,
    )
    # Use training scalers exactly
    ds_full.set_scaling_params(scaling["means"], scaling["stds"])

    # Non-shuffling DataLoader over the full dataset
    test_loader = DataLoader(
        ds_full,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,           # let Lightning inject DistributedSampler if needed
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=1,
        drop_last=False,
    )
    return test_loader, ds_full


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
