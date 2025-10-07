import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import torch.distributed as dist
from typing import Optional, Dict, List, Tuple
from contextlib import nullcontext
import math
import inspect

# Assuming ResNet3D.py contains the model definition.
# This import brings in the function responsible for creating the ResNet model architecture.
from ResNet3D import generate_2d_model


def _gather_cat_cpu(t: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Gathers tensors from all distributed processes, concatenates them on the CPU.
    This is useful for collecting validation or test results at the end of an epoch.

    Args:
        t (torch.Tensor): A tensor or a list of tensors from the current process.

    Returns:
        Optional[torch.Tensor]: A single concatenated tensor on the CPU on the main process (rank 0),
                                or None on other processes or if the input is empty.
    """
    # Handle cases where the input might be None, an empty list, or already a tensor.
    if t is None:
        obj = None
    elif isinstance(t, list):
        if len(t) == 0:
            obj = None
        else:
            # If it's a list of tensors, concatenate them.
            obj = torch.cat(t, dim=0)
    elif torch.is_tensor(t):
        obj = t
    else:
        obj = None

    # Check if distributed training is active.
    if dist.is_available() and dist.is_initialized():
        # Prepare a list to hold objects from all processes.
        world_size = dist.get_world_size()
        objs = [None for _ in range(world_size)]
        
        # `all_gather_object` collects a pickled object from each process.
        dist.all_gather_object(objs, obj)
        
        # The main process (rank 0) is responsible for concatenating the results.
        if dist.get_rank() == 0:
            # Filter out any None or empty tensors that might have been gathered.
            tensors = [o for o in objs if isinstance(o, torch.Tensor) and o.numel() > 0]
            if len(tensors) == 0:
                return None
            return torch.cat(tensors, dim=0)
        else:
            # Other processes don't need the final concatenated tensor.
            return None
    else:
        # If not in a distributed setting, just return the processed object.
        return obj


# ----------------------- GPU-side noise/mask helpers -----------------------
# These functions are designed to run efficiently on the GPU to augment data
# on-the-fly during training, avoiding CPU bottlenecks in the data loading pipeline.

@torch.no_grad()
def _rand_cauchy(shape: Tuple, *, device, generator: torch.Generator = None, dtype: torch.float32 = torch.float32) -> torch.Tensor:
    """
    Generates random numbers from a standard Cauchy distribution using the inverse transform sampling method.

    Args:
        shape (Tuple): The desired shape of the output tensor.
        device: The device to create the tensor on (e.g., 'cuda:0').
        generator (torch.Generator): A PyTorch random number generator for reproducibility.
        dtype (torch.dtype): The data type of the output tensor.

    Returns:
        torch.Tensor: A tensor of random numbers from a Cauchy(0, 1) distribution.
    """
    # Generate uniform random numbers in [0, 1).
    u = torch.rand(shape, device=device, generator=generator, dtype=dtype)
    # Apply the inverse CDF of the Cauchy distribution: F^-1(u) = tan(pi * (u - 0.5)).
    return torch.tan(torch.pi * (u - 0.5))

@torch.no_grad()
def _sample_truncated_folded_cauchy(mu: float, sigma: float, threshold: float,
                                    shape: Tuple, *, device, generator: torch.Generator = None, max_iters: int = 10) -> torch.Tensor:
    """
    Samples from a 'folded' (absolute value) Cauchy distribution, which is then scaled and shifted.
    It optionally resamples any values that exceed a specified threshold.

    The distribution is `mu + sigma * |Cauchy(0,1)|`.

    Args:
        mu (float): The location parameter (shift) of the distribution.
        sigma (float): The scale parameter of the distribution.
        threshold (float): If > 0, values exceeding this will be resampled up to `max_iters` times.
        shape (Tuple): The desired shape of the output tensor.
        device: The device to create the tensor on.
        generator (torch.Generator): A PyTorch random number generator.
        max_iters (int): The maximum number of resampling iterations.

    Returns:
        torch.Tensor: A tensor of sampled values.
    """
    # Initial sample from the scaled and shifted folded Cauchy distribution.
    out = mu + sigma * torch.abs(_rand_cauchy(shape, device=device, generator=generator))
    
    # If a truncation threshold is set, identify and resample values that are too large.
    if threshold > 0:
        mask = out > threshold
        it = 0
        # Loop until no values exceed the threshold or max iterations are reached.
        while mask.any() and it < max_iters:
            # Generate new samples only for the elements that need to be resampled.
            resample = mu + sigma * torch.abs(_rand_cauchy(mask.sum().item(),
                                                           device=device, generator=generator))
            out[mask] = resample
            # Update the mask to see if any of the new samples also exceed the threshold.
            mask = out > threshold
            it += 1
    return out

@torch.no_grad()
def _apply_noise_and_mask_on_device(
    x: torch.Tensor,
    *,
    gen: torch.Generator,
    use_cauchy_gauss: bool = True,
    cauchy_mu: float = 0.003,
    cauchy_sigma: float = 0.0032,
    cauchy_threshold: float = 0.07,
    add_gauss_sigma: float = 0.0,
    mask_frac: float = 0.0,
    mask_mode: str = "sample",
) -> torch.Tensor:
    """
    Applies noise and/or masking to a batch of data directly on the GPU.

    Args:
        x (torch.Tensor): The input batch of data.
        gen (torch.Generator): A PyTorch random number generator.
        use_cauchy_gauss (bool): If True, adds Gaussian noise where the standard deviation
                                 for each sample is drawn from a Cauchy distribution.
        cauchy_mu (float): Location parameter for the Cauchy distribution.
        cauchy_sigma (float): Scale parameter for the Cauchy distribution.
        cauchy_threshold (float): Truncation threshold for the Cauchy-drawn RMS values.
        add_gauss_sigma (float): Standard deviation for additional fixed Gaussian noise.
        mask_frac (float): The fraction of data to mask (e.g., 0.1 for 10%).
        mask_mode (str): 'sample' to drop entire samples from the batch, or 'element' to
                         drop individual elements within the tensors.

    Returns:
        torch.Tensor: The augmented data tensor.
    """
    device = x.device

    # --- Section 1: Apply Noise ---
    if use_cauchy_gauss:
        # For each sample in the batch, draw a single Root Mean Square (RMS) value
        # from the truncated folded Cauchy distribution. This will be the standard deviation
        # of the Gaussian noise applied to that sample.
        shape_rms = (x.shape[0],) + (1,) * (x.ndim - 1)  # Shape for broadcasting (e.g., [B, 1, 1, 1])
        rms = _sample_truncated_folded_cauchy(
            cauchy_mu, cauchy_sigma, cauchy_threshold,
            shape_rms, device=device, generator=gen
        ).to(torch.float32)

        # Generate Gaussian noise (N(0,1)) and scale it by the per-sample RMS value.
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * rms
        
        # Optionally, add another layer of fixed Gaussian noise.
        if add_gauss_sigma > 0.0:
            noise = noise + torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        
        # Add the final noise to the input tensor, but only where the signal is non-zero.
        # Create a mask where the original signal is non-zero.
        signal_mask = (x != 0).to(x.dtype)
        # Apply noise only to the non-masked regions.
        x = x + (noise.to(x.dtype) * signal_mask)
        

    elif add_gauss_sigma > 0.0:
        # If not using the Cauchy-Gauss method, just apply simple Gaussian noise.
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        # Create a mask where the original signal is non-zero.
        signal_mask = (x != 0).to(x.dtype)
        # Apply noise only to the non-masked regions.
        x = x + (noise.to(x.dtype) * signal_mask)

    # --- Section 2: Apply Masking ---
    if mask_frac > 0.0:
        if mask_mode == "sample":
            # This mode randomly drops entire samples (e.g., images) from the batch.
            b = x.shape[0]  # Batch size
            k = int(round(mask_frac * b)) # Number of samples to drop
            if k > 0:
                # Get a random permutation of indices to select which samples to drop.
                perm = torch.randperm(b, generator=gen, device=device)
                drop_idx = perm[:k]
                # Create a mask that is 0 for dropped samples and 1 otherwise.
                m = torch.ones((b,) + (1,) * (x.ndim - 1), device=device, dtype=x.dtype)
                m[drop_idx] = 0
                # Apply the mask.
                x = x * m
        elif mask_mode == "element":
            # This mode randomly sets individual elements of the tensors to zero.
            keep_prob = 1.0 - mask_frac
            # Create a binary mask where each element is kept with probability `keep_prob`.
            m = (torch.rand(x.shape, device=device, generator=gen) < keep_prob).to(x.dtype)
            x = x * m
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

    return x

# --------------------------------------------------------------------------


class LitResNetMDN(LightningModule):
    """
    A PyTorch Lightning module for a ResNet model that can function as a standard
    regressor or a Mixture Density Network (MDN).

    This class encapsulates the model, loss functions, optimization logic, and
    training/validation/testing steps. It supports several prediction modes:
    1.  Standard Regression (predicting a single value per target, using MSE loss).
    2.  MDN with Diagonal Covariance (predicting a mean and variance for each target independently).
    3.  MDN with Full Covariance (predicting a mean vector and a full covariance matrix for all targets).
    4.  Mixture of Gaussians (MoG) (predicting a weighted mixture of several Gaussian distributions).

    It also integrates the fast, on-GPU data augmentation (noise and masking) helpers defined above.
    """
    def __init__(
        self, model_cfg, optim_cfg, training_cfg,
        std_weights=None,
        dump_bad_batch: bool = True,
        dump_dir: str = "./",
        # --- GPU-side noise/masking controls ---
        apply_aug_train_only: bool = False,
        use_cauchy_gauss: bool = True,
        cauchy_mu: float = 0.003,
        cauchy_sigma: float = 0.0032,
        cauchy_threshold: float = 0.07,
        add_gauss_sigma: float = 0.0,
        mask_frac: float = 0.0,
        mask_mode: str = "sample",
    ):
        super().__init__()
        # `save_hyperparameters` stores the arguments to __init__ in self.hparams,
        # which is useful for logging and model checkpointing.
        self.save_hyperparameters(logger=False)

        # We use a try-except because global_rank is only available after the Trainer starts.
        try:
            rank = self.global_rank
        except Exception:
            rank = "N/A (pre-init)"
            
        print("\n" + "="*80)
        print(f"DIAGNOSTICS FOR PROCESS RANK: {rank}")
        
        # 1. Find out which file this class was actually loaded from.
        class_file = inspect.getfile(self.__class__)
        print(f"[INFO] The 'LitResNetMDN' class was loaded from file:\n       -> {class_file}")
        
        # 2. Inspect the signature of the on_validation_epoch_end method.
        hook_method = self.on_validation_epoch_end
        signature = inspect.getfullargspec(hook_method)
        print(f"[INFO] Signature of 'on_validation_epoch_end' found:\n       -> args = {signature.args}")
        print("="*80 + "\n", flush=True)
        

        # --- Section 1: Model Setup ---
        self.config_name = self._get_config_name(model_cfg.model_depth)
        self.model = generate_2d_model(
            config_name=self.config_name,
            use_batchnorm=model_cfg.use_batchnorm,
            TARGET_PARAMETERS=model_cfg.target_params,
            n_outputs=len(model_cfg.target_params),
            use_attention_heads=model_cfg.use_attention_heads,
            attention_latent_dim=model_cfg.attention_latent_dim,
            use_mdn=model_cfg.use_mdn,
            covariance_type=model_cfg.covariance_type,
            num_mixtures=model_cfg.num_mixtures
        )
        self.model_params: List[str] = model_cfg.target_params
        self.use_mdn: bool = model_cfg.use_mdn
        self.covariance_type = model_cfg.covariance_type # "diagonal" or "full"

        # --- Section 2: Loss Function Setup ---
        # If not using an MDN, the model is a simple regressor, so use Mean Squared Error loss.
        # Otherwise, the loss is a custom Negative Log-Likelihood (NLL) calculated in the training step.
        self.criterion = nn.MSELoss(reduction='mean') if not self.use_mdn else None
        
        # MDN-specific settings for the sigma (standard deviation) outputs.
        # `sigma_head_outputs_positive`: Does the model head guarantee positive sigmas (e.g., via softplus)?
        # `sigma_floor`: A small value to clamp sigmas to, preventing numerical instability (e.g., log(0)).
        self.sigma_head_outputs_positive = getattr(model_cfg, "sigma_head_outputs_positive", True)
        self.sigma_floor = getattr(model_cfg, "sigma_floor", 1e-4)

        # --- Section 3: Miscellaneous Setup ---
        self.std_weights = std_weights # Optional weights for the loss function.
        self.dump_bad_batch = dump_bad_batch # If True, save batches that cause errors (e.g., contain NaNs).
        self.dump_dir = dump_dir

        # Caches to store outputs during a validation epoch.
        # These are gathered from all GPUs at the end of the epoch for metric calculation.
        self._val_dataset = None
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_sigmas: Optional[List[torch.Tensor]] = [] if self.use_mdn else None
        # A dictionary to hold all aggregated validation results at epoch end.
        self._val_cache: Dict[str, Optional[torch.Tensor]] = {}

        # --- Section 4: Store Augmentation Config ---
        # Copies the augmentation settings into class attributes for easy access.
        self.apply_aug_train_only = apply_aug_train_only
        self.aug_use_cauchy_gauss = use_cauchy_gauss
        self.aug_cauchy_mu = cauchy_mu
        self.aug_cauchy_sigma = cauchy_sigma
        self.aug_cauchy_threshold = cauchy_threshold
        self.aug_add_gauss_sigma = add_gauss_sigma
        self.aug_mask_frac = mask_frac
        self.aug_mask_mode = mask_mode


    # ------------------- Lightning Lifecycle Hooks -------------------

    def on_fit_start(self):
        """
        Called once at the very beginning of the training process (`trainer.fit()`).
        Used for setup tasks like fetching dataset properties and performing sanity checks.
        """
        # If standard deviation weights for the loss weren't provided at initialization,
        # try to get them from the Lightning DataModule.
        if self.std_weights is None and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "dataset_ref", None) is not None:
                self.std_weights = dm.dataset_ref.scaler_stds.detach().cpu()
                if self.trainer.is_global_zero: # Log only on the main process.
                    self.print(f"[on_fit_start] std_weights set from datamodule: {self.std_weights}")

        # On the main process, perform a critical sanity check.
        if self.trainer.is_global_zero:
            dm = self.trainer.datamodule
            ds = getattr(dm, "dataset_ref", None)
            model_params = list(self.model_params)
            data_params  = list(getattr(dm, "model_params", []))
            ds_params    = list(getattr(ds, "model_params", [])) if ds is not None else []

            print("[CHECK] model_params:", model_params)
            print("[CHECK] data_cfg.model_params:", data_params)
            print("[CHECK] dataset.model_params:", ds_params)
            # Ensure the order and names of the target parameters are consistent across the
            # model, data configuration, and dataset. A mismatch would lead to incorrect
            # loss calculations and evaluation.
            assert model_params == data_params == ds_params, (
                "Parameter order/name mismatch between model and datamodule/dataset!"
            )
            
            # Print the scaling values (mean and std) used for each target parameter.
            means = getattr(ds, "scaler_means", None)
            stds  = getattr(ds, "scaler_stds", None)
            if means is not None and stds is not None:
                for i, name in enumerate(model_params):
                    print(f"[SCALE] {name}: mean={float(means[i]):.4g}, std={float(stds[i]):.4g}")

            print(f"[INFO] Using on-GPU augmentations: use_cauchy_gauss={self.aug_use_cauchy_gauss}", flush=True)
            print(f"[INFO] Using data augmentation (noise) only on training data: {self.apply_aug_train_only}", flush=True)

    
    def training_step(self, batch, batch_idx):
        """
        The main training loop for a single batch.
        """
        # 1. Unpack data and move to the correct device.
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # 2. Apply on-the-fly GPU augmentations.
        x = self._maybe_augment_on_device(x)

        # 3. Perform sanity checks on the data.
        self._sanity_checks(x, y, batch_idx)

        # 4. Forward pass, using mixed precision if enabled.
        use_amp = x.is_cuda and "mixed" in str(getattr(self.trainer, "precision", ""))
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x)

        # 5. Calculate loss based on the model's output structure.
        # This block handles the different MDN modes.

        # --- PATH 1: Mixture of Gaussians (MoG) ---
        if isinstance(out, dict) and ("pi_logits" in out):
            if "sigma" in out:   # Diagonal covariance mixture
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
            else:                # Full covariance mixture
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                with torch.no_grad():
                    mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])

                # optional: keep a heatmap source for mixtures (top-weight component)
                # if "L" in out:
                #     pi = F.softmax(out["pi_logits"], dim=-1)
                #     top = pi.argmax(dim=-1)                                  # (B,)
                #     L_top = out["L"][torch.arange(out["L"].size(0)), top]    # (B,d,d)
                #     cache = getattr(self, "_val_cache", {})
                #     cache.setdefault("Ls", []).append(L_top.detach().cpu())
                #     self._val_cache = cache

            self.log(f"train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        # --- PATH 2: Single Full-Covariance Gaussian ---
        if self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)
            
            # --- Optional: Add a regularization term to guide predicted variances ---
            # This can help stabilize training by encouraging the predicted standard deviations
            # (derived from the diagonal of the covariance matrix) towards target values.
            targets = getattr(self.hparams.model_cfg, "cov_diag_targets", {})
            weights = getattr(self.hparams.model_cfg, "cov_diag_weights", {})
            use_log = bool(getattr(self.hparams.model_cfg, "cov_diag_use_log", False))

            if targets or weights:
                # Calculate the predicted standard deviations from the Cholesky factor.
                sigma_diag = torch.sqrt((L.float().square()).sum(dim=-1))
                name2idx = {n: i for i, n in enumerate(self.model_params)}
                reg_terms = []
                for name, w in weights.items():
                    if w > 0 and name in name2idx:
                        i = name2idx[name]
                        tgt = float(targets.get(name, 1.0))
                        # The regularization loss is the squared difference between predicted and target sigmas.
                        if use_log:
                            reg = (torch.log(sigma_diag[:, i].clamp_min(1e-6)) - math.log(tgt))**2
                        else:
                            reg = (sigma_diag[:, i] - tgt)**2
                        reg_terms.append(w * reg.mean())
                if reg_terms:
                    loss = loss + torch.stack(reg_terms).sum()
            
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        # --- PATH 3: Legacy Diagonal MDN ---
        if self.use_mdn:
            loss, per_param = self._compute_loss(out, y, return_per_param=True)
            
            # Log the per-parameter NLL and other stats for the first batch for debugging.
            if batch_idx == 0 and self.trainer.is_global_zero:
                for i, name in enumerate(self.model_params):
                    mu_i, sigma_i = out[i*2].squeeze(-1), out[i*2+1].squeeze(-1)
                    if not self.sigma_head_outputs_positive:
                        sigma_i = F.softplus(sigma_i)
                    sigma_i = torch.clamp(sigma_i, min=self.sigma_floor)
                    y_i = y[:, i]
                    nll_i = F.gaussian_nll_loss(mu_i.float(), y_i.float(), (sigma_i*sigma_i).float(),
                                                full=True, reduction="none").mean()
                    print(f"[Train Check][{name}] y(mean±std)={y_i.mean():.3g}±{y_i.std():.3g} "
                        f"mu={mu_i.mean():.3g}±{mu_i.std():.3g} sigma={sigma_i.mean():.3g} "
                        f"NLL={nll_i.item():.6f}")

            # Log losses.
            for k, v in per_param.items():
                self.log(f"train/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss

        # --- PATH 4: Standard Regression (MSE) ---
        loss = self.criterion(out, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def on_validation_epoch_start(self):
        """
        Called at the beginning of every validation epoch.
        """
        self._val_dataset = getattr(self.trainer.datamodule, "dataset_ref", None)
        self.validation_step_outputs = []
        # We no longer need to initialize self._val_preds, self._val_cache, etc., here.
        # The new on_validation_epoch_end handles it.

    # def on_validation_start(self):
    #     """Called at the beginning of validation."""
    #     self.validation_step_outputs = []
    
    def validation_step(self, batch, batch_idx):
        # 1. Boilerplate setup
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if not self.apply_aug_train_only:
            x = self._maybe_augment_on_device(x)
        self._sanity_checks(x, y, batch_idx)

        # 2. Forward pass
        use_amp = x.is_cuda and "mixed" in str(getattr(self.trainer, "precision", ""))
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x)
        
        # 3. Initialize output_dict and loss
        output_dict = {"targets": y}
        loss = None # Initialize loss to None

        # 5. Calculate loss and populate the output dictionary based on the model's mode.
        
        if isinstance(out, dict) and ("pi_logits" in out):
            # ... (MoG logic as before) ...
            # Creates mu_mix, sig_mix, and populates output_dict
            if "sigma" in out:
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
                output_dict["sigma_all"] = out["sigma"]
            else:
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])
                output_dict["L_all"] = out["L"]

            output_dict.update({
                "preds": mu_mix, "sigmas": sig_mix, "pi_logits_all": out["pi_logits"], "mu_all": out["mu"]
            })

        elif self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)
            sigma_diag = torch.sqrt((L.square()).sum(dim=-1))
            output_dict.update({"preds": mu, "Ls": L, "sigmas": sigma_diag})

        elif self.use_mdn:
            mus, sigmas_raw = self._split_mdn_tuple(out, len(self.model_params))
            sigmas = F.softplus(sigmas_raw) if not self.sigma_head_outputs_positive else sigmas_raw
            sigmas = torch.clamp(sigmas, min=self.sigma_floor)
            output_dict.update({"preds": mus, "sigmas": sigmas})
            loss, per_param = self._compute_loss(out, y, return_per_param=True)
            for k, v in per_param.items():
                self.log(f"val/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)

        else: # Standard Regression
            loss = self.criterion(out, y)
            output_dict["preds"] = out

        # 5. Final check and return
        if loss is None:
            # This case should never happen if logic is correct, but it's a safeguard.
            # You might want to raise an error here or handle it gracefully.
            print(f"WARNING: No loss calculated in validation_step for batch {batch_idx}")
            return None # Explicitly return None if something went wrong

        # Store the output
        self.validation_step_outputs.append(output_dict)
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return output_dict
    
    def on_validation_epoch_end(self):
        """
        Receives a list of the dictionaries returned from every validation_step.
        This hook is now responsible for aggregating these into the final cache.
        """
        # # On non-main processes, we don't need to do any aggregation.
        # if self.trainer.global_rank != 0:
        #     return

        # The 'outputs' list is now the instance attribute we created.
        outputs = self.validation_step_outputs

        # On non-main processes, we only need to clear the list and exit.
        if self.trainer.global_rank != 0:
            outputs.clear() # Free memory on non-primary ranks
            return

        # --- Aggregation Logic (runs only on Rank 0) ---
        if not outputs: # Check if the outputs list is empty
            self._val_cache = {} # Ensure cache is empty for callbacks
            outputs.clear() # Clear the list for the next epoch
            return

        # --- NEW AGGREGATION LOGIC ---
        # Keys that we expect to aggregate from the batch outputs
        keys_to_aggregate = ["preds", "targets", "sigmas", "Ls", "pi_logits_all", "mu_all", "L_all", "sigma_all"]
        
        # Initialize a new, clean cache on Rank 0
        final_cache = {}
        
        for key in keys_to_aggregate:
            # Create a list of all tensors for the current key from all batches
            tensor_list = [batch_output[key] for batch_output in outputs if key in batch_output]
            
            # If we found any tensors for this key, concatenate them
            if tensor_list:
                final_cache[key] = torch.cat([t.detach().cpu() for t in tensor_list], dim=0)

        # The final, complete cache is now ready for the callbacks.
        # Assign it to the instance attribute that the callbacks will read.
        self._val_cache = final_cache

        self.validation_step_outputs.clear()
        
    # ------------------------- Helper Methods -------------------------

    def _get_config_name(self, model_depth: int) -> str:
        """Maps a ResNet depth number (e.g., 18, 50) to a predefined configuration name."""
        mapping = {
            10: "resnet10_2d_equivalent",
            18: "resnet18_2d_equivalent",
            34: "resnet34_2d",
            50: "resnet50_2d"
        }
        if model_depth not in mapping:
            raise ValueError(f"Unsupported model depth: {model_depth}. Supported depths: 10, 18, 34, 50.")
        return mapping[model_depth]

    def forward(self, x: torch.Tensor):
        """The standard forward pass for the model."""
        return self.model(x)

    @staticmethod
    def _finite_mask_per_sample(x: torch.Tensor) -> torch.Tensor:
        """
        Checks for non-finite values (NaN or Inf) in a tensor on a per-sample basis.

        Returns:
            A boolean tensor of shape (batch_size,) where True indicates a sample
            contains at least one non-finite value.
        """
        # Reshape to (batch_size, -1), sum across all features, and check for non-finites.
        return ~torch.isfinite(x.view(x.shape[0], -1).sum(dim=1))

    def _sanity_checks(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        """
        Performs checks on input and target tensors to catch common data issues early.
        Raises a ValueError if an issue is found.
        """
        # Check for NaNs or Infs in the input data.
        if not torch.isfinite(x).all():
            nan_mask = self._finite_mask_per_sample(x)
            bad_indices = nan_mask.nonzero(as_tuple=True)[0]
            # Log detailed information on the main process.
            if self.trainer.is_global_zero:
                self.print(f"\nNaNs/Inf in input at batch {batch_idx}, sample indices: {bad_indices.tolist()}")
                self.print("Corresponding target values for bad inputs:")
                for idx in bad_indices.tolist():
                    self.print(f"  Sample {idx}: {y[idx].detach().cpu().numpy()}")
                # Optionally save the entire problematic batch for offline debugging.
                if self.dump_bad_batch:
                    os.makedirs(self.dump_dir, exist_ok=True)
                    torch.save({"data": x.detach().cpu(), "target": y.detach().cpu()},
                               os.path.join(self.dump_dir, f"nan_input_batch{batch_idx}.pt"))
            raise ValueError("NaN/Inf input encountered — likely due to a bad parameter combination in the data generation.")

        # Check for other potential issues.
        if torch.isnan(x).any():
            raise ValueError("NaN in input")
        if x.shape[2] == 0:
            raise ValueError("Empty spectral dimension (D=0) in input")
        if not torch.isfinite(y).all():
            raise ValueError("Target contains NaNs/Inf")

    def _split_mdn(self, output) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        mu_list, sigma_list = [], []
        for i, _ in enumerate(self.model_params):
            mu    = output[i * 2]
            sigma = output[i * 2 + 1]
            mu    = mu.view(mu.shape[0], 1)
            sigma = sigma.view(sigma.shape[0], 1)
            if not self.sigma_head_outputs_positive:
                sigma = F.softplus(sigma)
            mu_list.append(mu)
            sigma_list.append(sigma)
        return mu_list, sigma_list
    def _split_batch(self, batch):
        """A simple helper to unpack the batch, which may or may not contain `noise_rms`."""
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, noise_rms = batch
        else:
            x, y = batch
            noise_rms = None
        return x, y, noise_rms

    def _compute_loss(self, output, target, return_per_param: bool = False):
        """
        (Legacy) Computes the loss for the diagonal MDN case.
        Calculates the Gaussian NLL for each parameter and averages them.
        """
        # --- Path for standard regression (non-MDN) ---
        if not self.use_mdn:
            if output.shape != target.shape:
                raise RuntimeError(f"Output-target shape mismatch: {output.shape} vs {target.shape}")
            loss = self.criterion(output, target)
            return (loss, {}) if return_per_param else loss

        # --- Path for diagonal MDN ---
        per_param = {}
        losses = []
        # The model output is a flat tuple: (mu_1, sigma_1, mu_2, sigma_2, ...)
        with torch.amp.autocast('cuda', enabled=False): # Disable AMP for stability
            for i, name in enumerate(self.model_params):
                mu    = output[i*2].squeeze(-1).float()
                sigma = output[i*2+1].squeeze(-1).float()
                var = sigma.mul(sigma).add(1e-6) # Variance = sigma^2, add epsilon for stability
                y_i = target[:, i].float()
                
                # Calculate Gaussian Negative Log-Likelihood loss.
                li = F.gaussian_nll_loss(mu, y_i, var, full=True, reduction="none")
                per_param[name] = li.mean()
                losses.append(li)

        # Average the NLL across all parameters and samples.
        loss = torch.stack(losses, dim=1).mean()
        return (loss, per_param) if return_per_param else loss

    # ------------------- Covariance and MDN Helper Methods -------------------

    @staticmethod
    def _is_fullcov_out(out) -> bool:
        """Checks if the model output format corresponds to a full-covariance MDN."""
        return isinstance(out, dict) and ("mu" in out) and ("L" in out)

    @staticmethod
    def _split_mdn_tuple(output, n_params: int):
        """
        Converts the flat tuple output (mu1, sigma1, mu2, sigma2, ...) from the legacy
        diagonal MDN model into two tensors: one for all means and one for all sigmas.
        """
        mus, sigmas = [], []
        for i in range(n_params):
            mu_i    = output[i * 2].squeeze(-1)
            sigma_i = output[i * 2 + 1].squeeze(-1)
            mus.append(mu_i)
            sigmas.append(sigma_i)
        # Stack the lists of tensors into single [Batch, n_params] tensors.
        return torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)

    def _mvn_nll_from_cholesky(self, mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Multivariate Normal (MVN) Negative Log-Likelihood loss
        for a single Gaussian distribution with a full covariance matrix.
        The covariance matrix Σ is represented by its Cholesky factor L, where Σ = L * L^T.

        This method is numerically more stable than working with Σ directly.

        Args:
            mu (torch.Tensor): Mean vector, shape [B, d].
            L (torch.Tensor): Lower-triangular Cholesky factor, shape [B, d, d].
            y (torch.Tensor): Target vector, shape [B, d].

        Returns:
            torch.Tensor: The mean NLL loss over the batch.
        """
        if mu.ndim != 2:
            mu = mu.view(mu.shape[0], -1)
        if y.ndim != 2:
            y = y.view(y.shape[0], -1)
        # Disable automatic mixed precision for these linear algebra operations
        # as they are sensitive and benefit from float32 precision.
        with torch.amp.autocast('cuda', enabled=False):
            mu32, L32, y32 = mu.float(), L.float(), y.float()
            
            # Ensure L is strictly lower-triangular.
            L32 = torch.tril(L32)
            
            # Add a small "jitter" to the diagonal of L. This ensures L is invertible
            # and prevents numerical issues like log(0) early in training when diagonals can be near zero.
            eye = torch.eye(L32.shape[-1], device=L32.device, dtype=L32.dtype).unsqueeze(0)
            L32 = L32 + 1e-6 * eye
            
            # The NLL formula is: 0.5 * ( (y-mu)^T * Σ^-1 * (y-mu) + log(det(Σ)) + d*log(2π) )
            # We compute this efficiently using the Cholesky factor L.
            
            # 1. Compute the Mahanalobis distance term: (y-mu)^T * Σ^-1 * (y-mu)
            # This is equivalent to ||z||^2 where L*z = (y-mu). We solve for z.
            diff = (y32 - mu32).unsqueeze(-1)
            z = torch.linalg.solve_triangular(L32, diff, upper=False) # Efficient solve for lower-triangular systems
            maha = z.square().sum(dim=(-2, -1))
            
            # 2. Compute the log-determinant term: log(det(Σ))
            # Since det(Σ) = det(L*L^T) = det(L)^2, and det(L) is the product of its diagonal elements,
            # log(det(Σ)) = 2 * sum(log(diag(L))).
            diag = torch.diagonal(L32, dim1=-2, dim2=-1)
            logdet = 2.0 * torch.log(diag).sum(dim=-1)
            
            # 3. The constant term.
            const = mu32.shape[-1] * math.log(2.0 * math.pi)
            
            # Combine terms to get the NLL for each sample in the batch.
            nll = 0.5 * (maha + logdet + const)
            loss = nll.mean() # Average loss over the batch.
            
        return loss.to(mu.dtype) # Cast back to the original dtype.

    # --- Mixture of Gaussians (MoG) NLL and Marginal Calculation Helpers ---
    
    def _mog_nll_diag(self, pi_logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Negative Log-Likelihood for a Mixture of Gaussians with diagonal covariance.

        Args:
            pi_logits (torch.Tensor): Logits for the mixture weights, shape [B, K].
            mu (torch.Tensor): Means of the K components, shape [B, K, d].
            sigma (torch.Tensor): Standard deviations of the K components, shape [B, K, d].
            y (torch.Tensor): Target vector, shape [B, d].

        Returns:
            torch.Tensor: The mean NLL loss over the batch.
        """
        # log(pi_k) - probabilities of choosing each mixture component k.
        log_pi = F.log_softmax(pi_logits, dim=-1)         # Shape: [B, K]
        
        # Reshape y for broadcasting against the K components.
        y_exp = y.unsqueeze(1)                            # Shape: [B, 1, d]
        var   = (sigma * sigma).clamp_min(1e-12)          # Shape: [B, K, d]

        # Calculate the log probability of y for each component k: log( N(y | mu_k, var_k) )
        const = mu.size(-1) * math.log(2.0 * math.pi)
        log_det = var.log().sum(-1)                       # Shape: [B, K]
        maha = ((y_exp - mu).square() / var).sum(-1)      # Shape: [B, K]
        log_prob = -0.5 * (const + log_det + maha)        # Shape: [B, K]
        
        # The total log probability is log( sum_k( pi_k * N(y | mu_k, var_k) ) ).
        # This is computed stably using logsumexp: logsumexp( log(pi_k) + log(N_k) )
        total_log_prob = (log_pi + log_prob).logsumexp(dim=-1)
        
        # The NLL is the negative of the mean total log probability.
        return -total_log_prob.mean()

    def _mog_nll_full(self, pi_logits: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the NLL for a Mixture of Gaussians with full covariance (using Cholesky factor L).

        Args:
            pi_logits (torch.Tensor): Logits for mixture weights, shape [B, K].
            mu (torch.Tensor): Means of the K components, shape [B, K, d].
            L (torch.Tensor): Cholesky factors of the K components, shape [B, K, d, d].
            y (torch.Tensor): Target vector, shape [B, d].

        Returns:
            torch.Tensor: The mean NLL loss over the batch.
        """
        B, K, d = mu.shape
        log_pi = F.log_softmax(pi_logits, dim=-1)         # Shape: [B, K]

        # Ensure L is lower-triangular and has a positive diagonal.
        L = torch.tril(L)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_pos = F.softplus(diag) + 1e-3 # Use softplus to enforce positivity + add jitter.
        L = L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)

        # Calculate log probability within a float32 context for stability.
        with torch.amp.autocast('cuda', enabled=False):
            # Same logic as _mvn_nll_from_cholesky, but calculated for K components in parallel.
            y_mu = (y.unsqueeze(1) - mu).unsqueeze(-1)    # Shape: [B, K, d, 1]
            z = torch.linalg.solve_triangular(L.float(), y_mu.float(), upper=False)
            maha = z.square().sum(dim=(-2, -1))           # Shape: [B, K]
            logdet = 2.0 * diag_pos.float().log().sum(-1) # Shape: [B, K]
            const = d * math.log(2.0 * math.pi)
            log_prob = -0.5 * (maha + logdet + const)     # Shape: [B, K]

        # Use logsumexp for the final stable calculation.
        total_log_prob = (log_pi + log_prob).logsumexp(dim=-1)
        return -total_log_prob.mean().to(mu.dtype)

    @staticmethod
    def _mixture_marginals_diag(pi_logits, mu, sigma):
        """
        Calculates the mean and standard deviation of the entire mixture distribution
        (not of the individual components). This is used for validation and inference.
        """
        pi = F.softmax(pi_logits, dim=-1)                 # Mixture weights, shape: [B, K]
        
        # E[x] = sum_k( pi_k * E_k[x] ) = sum_k( pi_k * mu_k )
        mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)       # Shape: [B, d]
        
        # Var[x] = E[x^2] - (E[x])^2
        # E[x^2] = sum_k( pi_k * E_k[x^2] ) = sum_k( pi_k * (Var_k[x] + E_k[x]^2) )
        var_k = (sigma * sigma)                           # Shape: [B, K, d]
        second_moment = (pi.unsqueeze(-1) * (var_k + mu * mu)).sum(dim=1)  # Shape: [B, d]
        var_mix = (second_moment - mu_mix * mu_mix).clamp_min(1e-12)
        
        return mu_mix, var_mix.sqrt()

    @staticmethod
    def _mixture_marginals_full(pi_logits, mu, L):
        """
        Calculates the marginal mean and standard deviation for a full-covariance mixture.
        It uses the diagonal of the covariance matrix for the variance calculation.
        """
        pi = F.softmax(pi_logits, dim=-1)                 # Mixture weights, shape: [B, K]
        
        # Mean is calculated the same way.
        mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)       # Shape: [B, d]
        
        # The diagonal of the covariance matrix Σ_k is the sum of squares of the rows of L_k.
        diag_Sigma_k = (L * L).sum(dim=-1)                 # Shape: [B, K, d]
        
        # Variance calculation proceeds as in the diagonal case.
        second_moment = (pi.unsqueeze(-1) * (diag_Sigma_k + mu * mu)).sum(dim=1)
        var_mix = (second_moment - mu_mix * mu_mix).clamp_min(1e-12)
        
        return mu_mix, var_mix.sqrt()


    # --------------------- Training / Validation / Test Steps ---------------------

    def _maybe_augment_on_device(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies on-GPU augmentations if conditions are met (e.g., during training).
        """
        if self.apply_aug_train_only and not self.training:
            return x
        
        # Create a new random generator for each step and each device.
        # This is crucial in distributed training to ensure that different GPUs
        # apply different random augmentations to their slice of the batch.
        gen = torch.Generator(device=self.device)
        seed = (
            int(self.global_step) * 1000003
            + int(self.current_epoch) * 9176
            + int(getattr(self, "global_rank", 0)) # Unique rank for each process
        )
        gen.manual_seed(seed)
        
        # Call the main augmentation function with the configured parameters.
        x = _apply_noise_and_mask_on_device(
            x, gen=gen,
            use_cauchy_gauss=self.aug_use_cauchy_gauss,
            cauchy_mu=self.aug_cauchy_mu,
            cauchy_sigma=self.aug_cauchy_sigma,
            cauchy_threshold=self.aug_cauchy_threshold,
            add_gauss_sigma=self.aug_add_gauss_sigma,
            mask_frac=self.aug_mask_frac,
            mask_mode=self.aug_mask_mode,
        )
        return x


    
    def on_test_start(self):
        """Called at the beginning of testing."""
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        # This method should be refactored to look almost identical to the new validation_step
        # It should return a dictionary of outputs.
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        self._sanity_checks(x, y, batch_idx)

        use_amp = x.is_cuda and isinstance(getattr(self.trainer, "precision", ""), str)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x)
        
        output_dict = {"targets": y}
        loss = None
        
        stage = "test"
        # --- PATH 1: Mixture of Gaussians (MoG) ---
        
        if isinstance(out, dict) and ("pi_logits" in out):
            # ... (MoG logic as before) ...
            # Creates mu_mix, sig_mix, and populates output_dict
            if "sigma" in out:
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
                output_dict["sigma_all"] = out["sigma"]
            else:
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])
                output_dict["L_all"] = out["L"]

            output_dict.update({
                "preds": mu_mix, "sigmas": sig_mix, "pi_logits_all": out["pi_logits"], "mu_all": out["mu"]
            })

        elif self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)
            sigma_diag = torch.sqrt((L.square()).sum(dim=-1))
            output_dict.update({"preds": mu, "Ls": L, "sigmas": sigma_diag})

        elif self.use_mdn:
            mus, sigmas_raw = self._split_mdn_tuple(out, len(self.model_params))
            sigmas = F.softplus(sigmas_raw) if not self.sigma_head_outputs_positive else sigmas_raw
            sigmas = torch.clamp(sigmas, min=self.sigma_floor)
            output_dict.update({"preds": mus, "sigmas": sigmas})
            loss, per_param = self._compute_loss(out, y, return_per_param=True)
            for k, v in per_param.items():
                self.log(f"val/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)

        else: # Standard Regression
            loss = self.criterion(out, y)
            output_dict["preds"] = out

        self.test_step_outputs.append(output_dict)

        if loss is not None:
            self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            
        return output_dict

    def on_test_epoch_end(self):
        # This logic is very similar to on_validation_epoch_end
        # It's where you would create a final cache for test-time callbacks if you had any.

        # The 'outputs' list is now the instance attribute we created.
        outputs = self.test_step_outputs

        # On non-main processes, we only need to clear the list and exit.
        if self.trainer.global_rank != 0:
            outputs.clear() # Free memory on non-primary ranks
            return

        # --- Aggregation Logic (runs only on Rank 0) ---
        if not outputs: # Check if the outputs list is empty
            self._val_cache = {} # Ensure cache is empty for callbacks
            outputs.clear() # Clear the list for the next epoch
            return


        keys_to_aggregate = ["preds", "targets", "sigmas", "Ls", "pi_logits_all", "mu_all", "L_all", "sigma_all"]
        final_cache = {}
        for key in keys_to_aggregate:
            tensor_list = [batch_output[key] for batch_output in outputs if key in batch_output]
            if tensor_list:
                final_cache[key] = torch.cat([t.detach().cpu() for t in tensor_list], dim=0)
        
        # You would typically save this cache or use it for final report generation.
        self._test_cache = final_cache

    # ------------------- Optimizer and Scheduler Configuration -------------------

    def configure_optimizers(self):
        """
        Sets up the optimizer and an optional learning rate scheduler.
        """
        # AdamW is a common and effective choice for deep learning models.
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optim_cfg.lr,
            weight_decay=self.hparams.optim_cfg.weight_decay,
        )
        
        # ReduceLROnPlateau automatically reduces the learning rate when a metric
        # (in this case, validation loss) has stopped improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduce LR when the metric stops decreasing
            factor=0.2,      # new_lr = lr * factor
            patience=5,      # Number of epochs with no improvement to wait before reducing LR
            min_lr=1e-6      # Lower bound on the learning rate
        )
        
        # This dictionary structure is required by PyTorch Lightning.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss", # The metric to monitor for the scheduler
                "frequency": 1         # Check the metric every validation epoch
            }
        }