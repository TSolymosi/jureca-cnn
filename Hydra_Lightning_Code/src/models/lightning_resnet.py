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
import numpy as np

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
    return_rms: bool = False,  # NEW: Optional flag to return RMS values
) -> torch.Tensor:
    """
    Applies noise and/or masking to a batch of data directly on the GPU.
    
    NEW: Can optionally return the RMS values used for noise generation.
    """
    device = x.device
    rms_values = None  # Will store RMS if requested

    # --- Section 1: Apply Noise ---
    if use_cauchy_gauss:
        # For each sample in the batch, draw a single RMS value
        shape_rms = (x.shape[0],) + (1,) * (x.ndim - 1)
        rms = _sample_truncated_folded_cauchy(
            cauchy_mu, cauchy_sigma, cauchy_threshold,
            shape_rms, device=device, generator=gen
        ).to(torch.float32)
        
        # Store RMS if requested (flatten to [B])
        if return_rms:
            rms_values = rms.view(x.shape[0])

        # Generate Gaussian noise and scale by RMS
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * rms
        
        # Optionally add fixed Gaussian noise
        if add_gauss_sigma > 0.0:
            noise = noise + torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        
        # Apply noise only where signal is non-zero
        signal_mask = (x != 0).to(x.dtype)
        x = x + (noise.to(x.dtype) * signal_mask)

    elif add_gauss_sigma > 0.0:
        # Simple Gaussian noise case
        noise = torch.randn(x.shape, device=device, dtype=torch.float32, generator=gen) * float(add_gauss_sigma)
        signal_mask = (x != 0).to(x.dtype)
        x = x + (noise.to(x.dtype) * signal_mask)
        
        # Store constant RMS if requested
        if return_rms:
            rms_values = torch.full((x.shape[0],), add_gauss_sigma, device=device)

    # --- Section 2: Apply Masking ---
    if mask_frac > 0.0:
        if mask_mode == "sample":
            b = x.shape[0]
            k = int(round(mask_frac * b))
            if k > 0:
                perm = torch.randperm(b, generator=gen, device=device)
                drop_idx = perm[:k]
                m = torch.ones((b,) + (1,) * (x.ndim - 1), device=device, dtype=x.dtype)
                m[drop_idx] = 0
                x = x * m
        elif mask_mode == "element":
            keep_prob = 1.0 - mask_frac
            m = (torch.rand(x.shape, device=device, generator=gen) < keep_prob).to(x.dtype)
            x = x * m
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

    if return_rms:
        return x, rms_values
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
        strategy_cfg,
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
        # Add job ids for the current job and optionally a a previous job to continue
        job_id: Optional[str] = None,
        load_id: Optional[str] = None,
    ):
        super().__init__()
        # `save_hyperparameters` stores the arguments to __init__ in self.hparams,
        # which is useful for logging and model checkpointing.
        self.save_hyperparameters(logger=False)
        self.is_stage_1 = True # Default state

        # We use a try-except because global_rank is only available after the Trainer starts.
        try:
            rank = self.global_rank
        except Exception:
            rank = "N/A (pre-init)"
            
        # print("\n" + "="*80)
        # print(f"DIAGNOSTICS FOR PROCESS RANK: {rank}")
        
        # # 1. Find out which file this class was actually loaded from.
        # class_file = inspect.getfile(self.__class__)
        # print(f"[INFO] The 'LitResNetMDN' class was loaded from file:\n       -> {class_file}")
        
        # # 2. Inspect the signature of the on_validation_epoch_end method.
        # hook_method = self.on_validation_epoch_end
        # signature = inspect.getfullargspec(hook_method)
        # print(f"[INFO] Signature of 'on_validation_epoch_end' found:\n       -> args = {signature.args}")
        # print("="*80 + "\n", flush=True)
        

        # --- Section 1: Model Setup ---
        self.config_name = self._get_config_name(model_cfg.model_depth)

        # --- Handle fixed mixture weights ---
        self.fixed_mixture_weights = model_cfg.get("fixed_mixture_weights", None)
        
        self.model = generate_2d_model(
            config_name=self.config_name,
            use_batchnorm=model_cfg.use_batchnorm,
            TARGET_PARAMETERS=model_cfg.target_params,
            n_outputs=len(model_cfg.target_params),
            use_attention_heads=model_cfg.use_attention_heads,
            attention_latent_dim=model_cfg.attention_latent_dim,
            use_mdn=model_cfg.use_mdn,
            covariance_type=model_cfg.covariance_type,
            num_mixtures=model_cfg.num_mixtures,
            covariance_mode=model_cfg.get('covariance_mode', 'cholesky_symmetric'),
            fixed_mixture_weights=self.fixed_mixture_weights,
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
        self.sigma_floor = getattr(model_cfg, "sigma_floor", 1e-6)

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

        # --- Section 5: Two-Stage Training Setup ---
        self.is_stage_1 = True # Default to Stage 1

        

    
    


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

    
    # In LitResNetMDN.py

    def training_step(self, batch, batch_idx):
            """
            The main training loop for a single batch.
            FIXED: Proper component-wise regularization for mixtures.
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
            
            # --- START: TWO-STAGE LOSS LOGIC ---
            # Get the configured switch epoch from hparams, default to a large number if not specified
            # Read from the new hparams location
            switch_epoch = self.hparams.strategy_cfg.get("two_stage_training", {}).get("switch_epoch", 9999)

            if self.is_stage_1:
                # --- STAGE 1: Train means and mixture weights only ---
                
                # Find the "best" component for each data point using the mixture weights

                pi_weights = F.softmax(out["pi_logits"], dim=-1)  # Shape: [B, K]

                # Expand y to match mu's shape for broadcasting
                y_expanded = y.unsqueeze(1) # Shape: [B, 1, d]

                # Calculate the MSE for each component against the target
                per_component_mse = F.mse_loss(out["mu"], y_expanded, reduction='none').mean(dim=-1) # Shape: [B, K]

                # Weight the MSE of each component by its mixture probability and sum them up
                soft_mse_loss = (pi_weights * per_component_mse).sum(dim=-1).mean()

                # Initialize as a zero tensor on the correct device to ensure it exists for logging
                mixture_penalty = torch.tensor(0.0, device=self.device)
                num_mixtures = out["pi_logits"].shape[1]
                if num_mixtures > 1 and not self.fixed_mixture_weights:
                    # Get the softmax (probabilities) and log_softmax of the mixture logits
                    pi_weights = torch.softmax(out["pi_logits"], dim=-1)  # Shape: [B, K]
                    log_pi = F.log_softmax(out["pi_logits"], dim=-1)

                    # Calculate the negative entropy. The optimizer will try to make this
                    # value less negative (i.e., maximize entropy), encouraging balanced weights.
                    negative_entropy = (pi_weights * log_pi).sum(dim=-1).mean()

                    # Define the strength of the penalty. This is a new hyperparameter you can tune.
                    # Start with a small value like 0.01.
                    mixture_penalty_weight = 0.1

                    # Calculate the final penalty term to be added to the loss
                    mixture_penalty = mixture_penalty_weight * negative_entropy

                total_loss = soft_mse_loss + mixture_penalty
                
                # --- STAGE 1 DIAGNOSTIC LOGGING ---
                if batch_idx == 0 and self.trainer.is_global_zero:
                    print(f"\n==================== [STAGE 1 - Epoch {self.current_epoch}] ====================")
                    print(f"    Proxy Loss (MSE on best mean): {total_loss.item():.4f}")
                    print(f"    Mixture Entropy Penalty:        {mixture_penalty.item():.4f}")
                    
                    # Print mixture weight distribution
                    print("\n[Mixture Weight Distribution (Mean across batch)]")
                    for k in range(pi_weights.shape[1]):
                        avg_weight = pi_weights[:, k].mean().item()
                        print(f"    Component {k+1}: {avg_weight:.2%}")
                    print(f"=======================================================================\n", flush=True)

            else:
                # --- STAGE 2 & 3: Train with the full MDN NLL loss ---
                # The optimizer configuration determines WHAT is being trained (cov_head only or everything).
                
                # 5. Calculate loss based on the model's output structure.

                # --- PATH 1 and 2: Mixture of Gaussians (MoG) with Full Covariance or Only Sigmas ---
                if isinstance(out, dict) and ("pi_logits" in out or self.fixed_mixture_weights):
                    
                    if "L" in out:
                        # Full covariance mixture
                        loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                        # --- CRITICAL FIX: Regularize COMPONENT sigmas, not marginal ---
                        L_all = out["L"]  # Shape: [B, K, d, d]
                        K = L_all.shape[1]

                        # Calculate sigma for each component: [B, K, d]
                        sigma_components = torch.sqrt((L_all.float().square()).sum(dim=-1))

                        # --- SECTION 1: GENTLE PULL TOWARDS A HEALTHY MEAN ---
                        # Re-enable this with a small weight. It provides a constant, gentle pressure
                        # to keep all sigmas in a reasonable range and prevents them from drifting too high.
                        target_sigma = 0.01
                        soft_reg_weight = 0.0001  # A small value like 0.001 is a good start
                        sigma_reg_loss = soft_reg_weight * ((sigma_components - target_sigma) ** 2).mean()

                        

                        # --- SECTION 2: STRONG, TARGETED ANTI-COLLAPSE PENALTY ---
                        # This is your most important tool. We will set higher floors for the problem parameters.
                        min_sigma_thresholds = torch.ones(len(self.model_params), device=self.device) * 0.001 # Base floor
                        name2idx = {n: i for i, n in enumerate(self.model_params)}

                        # SET SPECIFIC, HIGHER THRESHOLDS FOR PROBLEM PARAMETERS:
                        # if 'ro' in name2idx:
                        #     min_sigma_thresholds[name2idx['ro']] = 0.01  
                        # if 'NCH3CN' in name2idx:
                        #     min_sigma_thresholds[name2idx['NCH3CN']] = 0.01 
                        # if 'L' in name2idx:
                        #     min_sigma_thresholds[name2idx['L']] = 0.01 

                        min_sigma_thresholds = min_sigma_thresholds.view(1, 1, -1)
                        violation = torch.clamp(min_sigma_thresholds - sigma_components, min=0.0)

                        # CRITICAL FIX: Use L1 penalty instead of L2. 
                        # L2 gradients vanish near the threshold. L1 provides a constant, strong push.
                        min_sigma_penalty = 100.0 * violation.mean()

                    

                        # --- SECTION 3: (OPTIONAL) PENALTY FOR EXCESSIVE UNCERTAINTY ---
                        # Keeping this is fine. It prevents the model from becoming too uncertain, but it's not
                        # the solution to your current problem of *too little* uncertainty.
                        max_sigma_threshold = 2.0
                        violation_max = torch.clamp(sigma_components - max_sigma_threshold, min=0.0)
                        max_sigma_penalty = 0.1 * (violation_max ** 2).mean()

                        # --- FINAL LOSS CALCULATION ---
                        total_loss = loss + sigma_reg_loss + min_sigma_penalty + max_sigma_penalty
                        
                        # --- START: ENTROPY PENALTY TO PREVENT MIXTURE COLLAPSE ---
                        # Initialize as a zero tensor on the correct device to ensure it exists for logging
                        mixture_penalty = torch.tensor(0.0, device=self.device)
                        num_mixtures = out["pi_logits"].shape[1]
                        if num_mixtures > 1 and not self.fixed_mixture_weights:
                            # Get the softmax (probabilities) and log_softmax of the mixture logits
                            pi_weights = torch.softmax(out["pi_logits"], dim=-1)  # Shape: [B, K]
                            log_pi = F.log_softmax(out["pi_logits"], dim=-1)

                            # Calculate the negative entropy. The optimizer will try to make this
                            # value less negative (i.e., maximize entropy), encouraging balanced weights.
                            negative_entropy = (pi_weights * log_pi).sum(dim=-1).mean()

                            # Base weight for epoch 1 and 2. We will decay this in Stage 3 to allow the model to express high confidence if needed.
                            mixture_penalty_weight = 1.0
                            
                            # Decay the penalty late in training (Stage 3)
                            s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3
                            if self.current_epoch >= s3_epoch + 40:
                                # 40 epochs into Stage 3, the penalty is no longer needed and can be turned off to allow confident predictions if the data supports it.
                                mixture_penalty_weight = 0.0
                            elif self.current_epoch >= s3_epoch + 20:
                                # Gradually reduce it further in the middle of Stage 3
                                mixture_penalty_weight = 0.01 
                            elif self.current_epoch >= s3_epoch:
                                # Gradually reduce it at the start of Stage 3
                                mixture_penalty_weight = 0.1 

                            

                            # Calculate the final penalty term to be added to the loss
                            mixture_penalty = mixture_penalty_weight * negative_entropy

                            # Add the penalty to the total loss
                            total_loss = total_loss + mixture_penalty
                        # --- END: ENTROPY PENALTY ---
                    
                        
                        # --- UPDATED DIAGNOSTIC LOGGING BLOCK ---
                        # This block should replace your existing print statements for batch 0
                        if batch_idx == 0 and self.trainer.is_global_zero:
                            s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3
                            stage_label = "STAGE 3 (Fine-tuning)" if self.current_epoch >= s3_epoch else "STAGE 2 (Cov-Head Warmup)"
                            print(f"\n==================== [{stage_label} - Epoch {self.current_epoch}] ====================")
                            
                            # 1. Print sigma statistics (as before)
                            print("[Sigma Statistics (Component-wise)]")
                            for i, name in enumerate(self.model_params):
                                avg_sigma = sigma_components[:, :, i].mean().item()
                                min_sigma = sigma_components[:, :, i].min().item()
                                max_sigma = sigma_components[:, :, i].max().item()
                                print(f"  {name}: avg={avg_sigma:.3f}, min={min_sigma:.3f}, max={max_sigma:.3f}")

                            # 2. Print mixture weight distribution (crucial for diagnosing collapse)
                            if num_mixtures > 1:
                                pi = torch.softmax(out["pi_logits"], dim=-1)  # Re-calculate for clarity
                                print("\n[Mixture Weight Distribution (Mean across batch)]")
                                for k in range(pi.shape[1]):
                                    avg_weight = pi[:, k].mean().item()
                                    print(f"    Component {k+1}: {avg_weight:.2%}")
                            
                            # 3. Print a detailed breakdown of the loss components
                            print("\n[Loss Breakdown]")
                            print(f"    NLL Loss:                {loss.item():.4f}")
                            print(f"    Sigma Reg Loss:          {sigma_reg_loss.item():.4f}") # Uncomment if you use it
                            print(f"    Min Sigma Penalty:       {min_sigma_penalty.item():.4f}") # Uncomment if you use it
                            print(f"    Max Sigma Penalty:       {max_sigma_penalty.item():.4f}") # Uncomment if you use it
                            print(f"    Mixture Entropy Penalty:   {mixture_penalty.item():.4f}")
                            print(f"    ---------------------------------")
                            print(f"    Total Loss:              {total_loss.item():.4f}")
                            print(f"=======================================================================\n", flush=True)
                        
                        # Log individual components for Stage 2
                        self.log("train/nll", loss, on_step=False, on_epoch=True, sync_dist=True)
                        self.log("train/sigma_reg", sigma_reg_loss, on_step=False, on_epoch=True, sync_dist=True)
                        self.log("train/min_sigma_penalty", min_sigma_penalty, on_step=False, on_epoch=True, sync_dist=True)

                    else: # "sigma" in out
                        # Diagonal covariance mixture
                        loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)

                        # Similar regularization for diagonal case
                        sigma_all = out["sigma"]  # Shape: [B, K, d]
                        target_sigma = 1.0
                        reg_weight = 1.0
                        
                        # Per-parameter weighting
                        param_weights = torch.ones(len(self.model_params), device=self.device)
                        name2idx = {n: i for i, n in enumerate(self.model_params)}
                        if 'D' in name2idx:
                            param_weights[name2idx['D']] = 3.0
                        if 'L' in name2idx:
                            param_weights[name2idx['L']] = 2.0
                        
                        sigma_diff = (sigma_all - target_sigma) ** 2
                        weighted_diff = sigma_diff * param_weights.view(1, 1, -1)
                        sigma_reg_loss = reg_weight * weighted_diff.mean()
                        
                        # Minimum sigma constraint
                        min_sigma_threshold = 0.01
                        violation = torch.clamp(min_sigma_threshold - sigma_all, min=0.0)
                        min_sigma_penalty = 10.0 * (violation ** 2).mean()
                        
                        total_loss = loss + sigma_reg_loss + min_sigma_penalty
                        
                        if batch_idx == 0 and self.trainer.is_global_zero:
                            print(f"\n[Batch 0 Check - Epoch {self.current_epoch}]")
                            for i, name in enumerate(self.model_params):
                                avg_sigma = sigma_all[:, :, i].mean().item()
                                min_sigma = sigma_all[:, :, i].min().item()
                                max_sigma = sigma_all[:, :, i].max().item()
                                print(f"  {name}: avg_sigma={avg_sigma:.3f}, min={min_sigma:.3f}, max={max_sigma:.3f}")

                # --- PATH 3: Single Full-Covariance Gaussian (Non-Mixture) ---
                elif self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
                    mu, L = out["mu"], out["L"]
                    loss = self._mvn_nll_from_cholesky(mu, L, y)
                    
                    # Add sigma regularization for single Gaussian too
                    sigma_diag = torch.sqrt((L.float().square()).sum(dim=-1))
                    target_sigma = 1.0
                    reg_weight = 0.1  # Lighter for single Gaussian
                    
                    sigma_reg_loss = reg_weight * ((sigma_diag - target_sigma) ** 2).mean()
                    total_loss = loss + sigma_reg_loss
                    
                    if batch_idx == 0 and self.trainer.is_global_zero:
                        print(f"\n[Batch 0 Check - Epoch {self.current_epoch}]")
                        for i, name in enumerate(self.model_params):
                            avg_sigma = sigma_diag[:, i].mean().item()
                            print(f"  {name}: avg_sigma={avg_sigma:.3f}")

                # --- PATH 4: Legacy Diagonal MDN (kept for backward compatibility) ---
                elif self.use_mdn:
                    total_loss, per_param = self._compute_loss(out, y, return_per_param=True)
                    
                    # Log the per-parameter NLL for debugging
                    for k, v in per_param.items():
                        self.log(f"train/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)

                # --- PATH 5: Standard Regression (MSE) ---
                else:
                    total_loss = self.criterion(out, y)

            # --- END: TWO-STAGE LOSS LOGIC ---

            # Log the final total loss and return it
            self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return total_loss


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
        """
        Validation step with SNR-based filtering using consistent noise application.
        Uses the SAME noise generation logic as training.
        """
        # 1. Boilerplate setup
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        batch_size = x.size(0)
        
        # --- SNR Filtering with Integrated Noise Application ---
        min_snr = 10.0
        max_retries = 20
        
        # Calculate signal for each sample (max intensity)
        signal = x.view(x.size(0), -1).max(dim=1)[0]  # [B]
        
        # Create generator for reproducible validation
        gen = torch.Generator(device=self.device)
        seed = int(self.current_epoch) * 100000 + batch_idx
        gen.manual_seed(seed)
        
        # Initialize storage
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        snr_values = torch.zeros(batch_size, device=self.device)
        x_noisy = torch.zeros_like(x)
        
        #x_augmented_list = []
        #snr_list = []
        rms_list = []
        
        # Process each sample individually with retry logic
        for sample_idx in range(batch_size):
            x_sample = x[sample_idx:sample_idx+1]  # Keep batch dim: [1, ...]
            
            for attempt in range(max_retries):
                # Apply noise and GET the RMS used
                if not self.apply_aug_train_only:
                    x_sample_noisy, rms = _apply_noise_and_mask_on_device(
                        x_sample.clone(),
                        gen=gen,
                        use_cauchy_gauss=self.aug_use_cauchy_gauss,
                        cauchy_mu=self.aug_cauchy_mu,
                        cauchy_sigma=self.aug_cauchy_sigma,
                        cauchy_threshold=self.aug_cauchy_threshold,
                        add_gauss_sigma=self.aug_add_gauss_sigma,
                        mask_frac=self.aug_mask_frac,
                        mask_mode=self.aug_mask_mode,
                        return_rms=True  # NEW: Get the RMS back
                    )
                    rms_value = rms[0].item()
                else:
                    # No augmentation
                    x_noisy = x_sample
                    rms_value = 0.001  # Dummy small value
                
                # Calculate SNR using the ACTUAL RMS from augmentation
                snr = signal[sample_idx].item() / rms_value if rms_value > 0 else float('inf')
                
                # Check if SNR meets threshold
                if snr >= min_snr:
                    valid_mask[sample_idx] = True
                    #x_augmented_list.append(x_noisy)
                    x_noisy[sample_idx] = x_sample_noisy[0]
                    snr_values[sample_idx] = snr
                    #snr_list.append(snr)
                    rms_list.append(rms_value)
                    break
            
            # If no valid noise found, sample is excluded
            if not valid_mask[sample_idx]:
                # Use the last attempted values for logging
                x_noisy[sample_idx] = x_sample_noisy[0]
                snr_values[sample_idx] = snr
                #pass
        
        # Count exclusions
        num_original = batch_size
        num_valid = valid_mask.sum().item()
        num_excluded = num_original - num_valid
        
        if not hasattr(self, '_val_excluded_count'):
            self._val_excluded_count = 0
        self._val_excluded_count += num_excluded
        
        # If no valid samples, return None
        #if num_valid == 0:
        #    return None
        
        # Stack augmented samples
        # x = torch.cat(x_augmented_list, dim=0)
        # y = y[valid_mask]
        # snr_values = torch.tensor(snr_list, device=self.device)
        
        # # Sanity checks
        # self._sanity_checks(x, y, batch_idx)

        #Forward pass
        use_amp = x.is_cuda and "mixed" in str(getattr(self.trainer, "precision", ""))
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x_noisy)
        
        # Initialize output_dict
        output_dict = {"targets": y, "snr": snr_values, "valid_mask": valid_mask}
        loss = None

        # Handle mixture model outputs
        if isinstance(out, dict) and ("pi_logits" in out or self.fixed_mixture_weights):
            
            
            if "sigma" in out:
                # Diagonal covariance mixture
                loss = self._mog_nll_diag(out["pi_logits"], out["mu"], out["sigma"], y)
                mu_mix, sig_mix = self._mixture_marginals_diag(out["pi_logits"], out["mu"], out["sigma"])
                
                # CRITICAL: Store ALL mixture data for corner plots
                output_dict.update({
                    "val_loss": loss.detach(),
                    "preds": mu_mix,
                    "sigmas": sig_mix,
                    "snr": snr_values,
                    "valid_mask": valid_mask,
                    "mu_all": out["mu"],                 # [B, K, d]
                    "sigma_all": out["sigma"],         # [B, K, d]
                    "targets": y
                })
                
            elif "L" in out:
                # Full covariance mixture
                loss = self._mog_nll_full(out["pi_logits"], out["mu"], out["L"], y)
                mu_mix, sig_mix = self._mixture_marginals_full(out["pi_logits"], out["mu"], out["L"])
                
                # CRITICAL: Store ALL mixture data for corner plots
                output_dict.update({
                    "val_loss": loss.detach(),
                    "preds": mu_mix,
                    "sigmas": sig_mix,
                    "snr": snr_values,
                    "valid_mask": valid_mask,
                    "mu_all": out["mu"],                 # [B, K, d]
                    "L_all": out["L"],                   # [B, K, d, d]
                    "targets": y,
                    "pi_logits_all": out["pi_logits"]  # [B, K]
                })

        elif self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            # Single Gaussian full covariance
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)
            sigma_diag = torch.sqrt((L.square()).sum(dim=-1))
            
            output_dict.update({
                "val_loss": loss.detach(),
                "preds": mu,
                "Ls": L,
                "sigmas": sigma_diag,
                "snr": snr_values,
                "valid_mask": valid_mask,
                "targets": y
            })
            

        # elif self.use_mdn:
        #     mus, sigmas_raw = self._split_mdn_tuple(out, len(self.model_params))
        #     sigmas = F.softplus(sigmas_raw) if not self.sigma_head_outputs_positive else sigmas_raw
        #     sigmas = torch.clamp(sigmas, min=self.sigma_floor)
        #     output_dict.update({"preds": mus, "sigmas": sigmas})
        #     loss, per_param = self._compute_loss(out, y, return_per_param=True)
        #     for k, v in per_param.items():
        #         self.log(f"val/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)

        # else:  # Standard Regression
        #     loss = self.criterion(out, y)
        #     output_dict["preds"] = out

        if loss is None:
            print(f"WARNING: No loss calculated in validation_step for batch {batch_idx}")
            return None

        # Store the output
        self.validation_step_outputs.append(output_dict)
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return output_dict


    def on_validation_epoch_end(self):
        """
        Aggregates validation results and logs SNR filtering statistics.
        (Efficient and Corrected Refactor)
        """
        outputs = self.validation_step_outputs
        if not outputs:
            self._val_cache = {}
            return # No outputs to process

        # --- Step 1: Aggregate tensors ON DEVICE for each rank ---
        aggregated_on_device = {}
        keys_to_aggregate = ["preds", "targets", "sigmas", "Ls", "pi_logits_all",
                            "mu_all", "L_all", "sigma_all", "snr", "valid_mask", "val_loss"]

        for key in keys_to_aggregate:
            tensor_list = [batch_output[key] for batch_output in outputs
                        if key in batch_output and batch_output[key] is not None]
            
            if tensor_list:
                # --- START: THE FIX ---
                # Check if the tensors are scalars (like the loss value)
                if tensor_list[0].ndim == 0:
                    # For scalars, STACK them to create a new dimension (a 1D tensor of losses)
                    aggregated_on_device[key] = torch.stack([t.detach() for t in tensor_list])
                else:
                    # For all other multi-dimensional tensors, CONCATENATE along the batch dimension
                    aggregated_on_device[key] = torch.cat([t.detach() for t in tensor_list], dim=0)
                # --- END: THE FIX ---

        # --- Step 2: Gather aggregated tensors from all ranks (no changes here) ---
        if self.trainer.world_size > 1:
            gathered_tensors = self.all_gather(aggregated_on_device)
        else:
            gathered_tensors = aggregated_on_device

        # --- Step 3: Final processing on Rank 0 (no changes here) ---
        if self.trainer.is_global_zero:
            final_cache = {}
            for key, tensors in gathered_tensors.items():
                if self.trainer.world_size > 1:
                    # This reshape now correctly handles the stacked losses as well as the concatenated tensors
                    final_cache[key] = tensors.reshape(-1, *tensors.shape[2:]).cpu()
                else:
                    final_cache[key] = tensors.cpu()

            # --- The rest of your logic remains the same ---
            # Log statistics BEFORE filtering
            total_samples = len(final_cache.get("targets", []))
            if "valid_mask" in final_cache:
                valid_samples = final_cache["valid_mask"].sum().item()
                invalid_samples = total_samples - valid_samples
                
                print(f"\n[Validation Summary]")
                print(f"  Total samples processed: {total_samples}")
                print(f"  Valid samples (SNR >= minimum): {valid_samples}")
                print(f"  Invalid samples (SNR < minimum): {invalid_samples}")
            
            # Apply SNR filtering for the final cache
            if "valid_mask" in final_cache and final_cache["valid_mask"].sum() > 0:
                valid_mask = final_cache["valid_mask"].bool()
                
                filtered_cache = {}
                keys_to_filter = ["preds", "targets", "sigmas", "snr", "pi_logits_all", 
                                "mu_all", "L_all", "sigma_all", "Ls"]
                
                for key in keys_to_filter:
                    if key in final_cache and final_cache[key] is not None:
                        filtered_cache[key] = final_cache[key][valid_mask]
                
                self._val_cache = filtered_cache
                print(f"  Samples in cache (after SNR filter): {len(self._val_cache['targets'])}")

            else:
                self._val_cache = final_cache
                print(f"  No SNR filtering applied or no valid samples")

                        # --- START: MEAN RELATIVE ERROR CALCULATION ---
            # This block calculates (pred - true) / true in physical units.
            print("\n[Mean Relative Error (post-SNR filter)]")
            
            # Ensure the cache has the necessary data before proceeding
            if "preds" in self._val_cache and "targets" in self._val_cache:
                dataset_ref = self.trainer.datamodule.dataset_ref
                
                # Step 1: Inverse transform predictions and targets to their original physical scale
                preds_orig = dataset_ref.inverse_transform_labels(self._val_cache['preds'])
                targets_orig = dataset_ref.inverse_transform_labels(self._val_cache['targets'])
                
                # Step 2: Loop through each parameter and calculate the metric
                for i, name in enumerate(self.model_params):
                    pred_param = preds_orig[:, i]
                    target_param = targets_orig[:, i]
                    
                    # Create a mask to avoid division by zero
                    mask = torch.abs(target_param) > 1e-9
                    
                    if mask.sum() > 0:
                        # Calculate relative error only on non-zero true values
                        relative_error = (pred_param[mask] - target_param[mask]) / target_param[mask]
                        mean_relative_error = relative_error.mean().item()
                        # Print as a percentage for readability
                        print(f"  {name+':':<10} {mean_relative_error * 100: >7.2f}%")
                    else:
                        print(f"  {name+':':<10} N/A (all true values are zero)")
            else:
                print("  Could not calculate: 'preds' or 'targets' not found in cache.")
            # --- END: MEAN RELATIVE ERROR CALCULATION ---
            
            # Compute and log average loss from the gathered and finalized loss tensor
            avg_loss = final_cache["val_loss"].mean()
            self.log("val/loss", avg_loss, prog_bar=True, on_epoch=True, sync_dist=False)
            
            if hasattr(self, '_val_excluded_count'):
                print(f"  Samples excluded during retry attempts: {self._val_excluded_count}")
                self._val_excluded_count = 0

        # Finally, clear the step outputs on all ranks
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

    def _mog_nll_full_old(self, pi_logits: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    
    def _mog_nll_full(self, pi_logits: torch.Tensor, mu: torch.Tensor, L: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates NLL for Mixture of Gaussians with Full Covariance.
        
        UPDATED: Assumes 'L' is already a valid Cholesky factor (positive diagonal),
        so we do NOT add jitter or re-apply softplus here.
        """
        B, K, d = mu.shape
        
        # 1. Log-Probabilities of mixing coefficients
        log_pi = F.log_softmax(pi_logits, dim=-1)   # [B, K]

        # 2. Gaussian Log-Likelihood per component
        # Use float32 for linear algebra stability
        with torch.amp.autocast('cuda', enabled=False):
            L_f32 = L.float()
            mu_f32 = mu.float()
            y_f32 = y.float()
            
            # --- Mahalanobis Distance: (y - μ)ᵀ Σ⁻¹ (y - μ) ---
            # We solve Lz = (y - μ), then distance is ||z||²
            
            # Expand y to match K components: [B, K, d, 1]
            y_mu = (y_f32.unsqueeze(1) - mu_f32).unsqueeze(-1)
            
            # Solve triangular system. 
            # Note: L is guaranteed lower triangular by the Head.
            z = torch.linalg.solve_triangular(L_f32, y_mu, upper=False)
            
            # Sum of squares: [B, K]
            maha = z.square().sum(dim=(-2, -1))
            
            # --- Log Determinant: log|Σ| = 2 * log|L| ---
            # Extract diagonal: [B, K, d]
            diag = torch.diagonal(L_f32, dim1=-2, dim2=-1)
            
            # CHANGE: Removed "+ 1e-6" jitter. 
            # The Head already clamps to self.min_sigma. Adding more here breaks 
            # the gradient flow for very small sigmas.
            logdet = 2.0 * torch.log(diag).sum(-1)
            
            # --- Constant Term ---
            const = d * math.log(2.0 * math.pi)
            
            # Log-prob per component
            log_prob = -0.5 * (maha + logdet + const)

        # 3. Log-Sum-Exp for the Mixture
        # log( sum( pi_k * N_k ) )  ->  logsumexp( log_pi + log_N )
        total_log_prob = (log_pi + log_prob).logsumexp(dim=-1)
        
        return -total_log_prob.mean()

    @staticmethod
    def _mixture_marginals_diag(pi_logits, mu, sigma):
        """
        Calculates the mean and standard deviation of the entire mixture distribution.
        
        FIXED: Uses numerically stable formula to prevent catastrophic cancellation.
        """
        pi = F.softmax(pi_logits, dim=-1) # [B, K]
        
        # Mean: E[X] = sum_k pi_k * mu_k
        mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)  # [B, d]
        
        # Variance: Var[X] = sum_k pi_k * Var_k + sum_k pi_k * (mu_k - mu_mix)^2
        # This formula guarantees positivity and avoids floating point cancellation.
        var_k = sigma * sigma  # [B, K, d]
        var_within = (pi.unsqueeze(-1) * var_k).sum(dim=1)  # [B, d]
        
        diff = mu - mu_mix.unsqueeze(1)  # [B, K, d]
        var_between = (pi.unsqueeze(-1) * (diff * diff)).sum(dim=1)  # [B, d]
        
        var_mix = var_within + var_between
        sigma_mix = torch.clamp(var_mix.sqrt(), min=1e-6)
        
        return mu_mix, sigma_mix

    @staticmethod
    def _mixture_marginals_full(pi_logits, mu, L):
        """
        Calculates the marginal mean and standard deviation for a full-covariance mixture.
        
        FIXED: Uses numerically stable formula to prevent catastrophic cancellation.
        """
        pi = F.softmax(pi_logits, dim=-1)
        # Mean: E[X] = sum_k pi_k * mu_k
        mu_mix = (pi.unsqueeze(-1) * mu).sum(dim=1)  # [B, d]
        
        # Diagonal of covariance matrix for each component
        diag_Sigma_k = torch.einsum('...ij,...ij->...i', L, L)  # [B, K, d]
        
        # Variance: Var[X] = sum_k pi_k * Var_k + sum_k pi_k * (mu_k - mu_mix)^2
        # This formula guarantees positivity and avoids floating point cancellation.
        var_within = (pi.unsqueeze(-1) * diag_Sigma_k).sum(dim=1)  # [B, d]
        
        diff = mu - mu_mix.unsqueeze(1)  # [B, K, d]
        var_between = (pi.unsqueeze(-1) * (diff * diff)).sum(dim=1)  # [B, d]
        
        var_mix = var_within + var_between
        sigma_mix = torch.clamp(var_mix.sqrt(), min=1e-6)
        
        return mu_mix, sigma_mix


    # --------------------- Training / Validation / Test Steps ---------------------

    def _maybe_augment_on_device(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies on-GPU augmentations if conditions are met (e.g., during training).
        """
        if self.apply_aug_train_only and not self.training:
            return x
        
        # Create a new random generator for each step
        gen = torch.Generator(device=self.device)
        seed = (
            int(self.global_step) * 1000003
            + int(self.current_epoch) * 9176
            + int(getattr(self, "global_rank", 0))
        )
        gen.manual_seed(seed)
        
        # Call augmentation WITHOUT requesting RMS (backward compatible)
        x = _apply_noise_and_mask_on_device(
            x, gen=gen,
            use_cauchy_gauss=self.aug_use_cauchy_gauss,
            cauchy_mu=self.aug_cauchy_mu,
            cauchy_sigma=self.aug_cauchy_sigma,
            cauchy_threshold=self.aug_cauchy_threshold,
            add_gauss_sigma=self.aug_add_gauss_sigma,
            mask_frac=self.aug_mask_frac,
            mask_mode=self.aug_mask_mode,
            return_rms=False  # Training doesn't need RMS
        )
        return x


    
    def on_test_start(self):
        """Called at the beginning of testing."""
        self.test_step_outputs = []
        # --- Add counter for excluded samples ---
        self._test_excluded_count = 0

    def test_step(self, batch, batch_idx):
        """
        Test step with SNR-based filtering, consistent with validation.
        """
        # --- START: Copied and adapted logic from validation_step ---
        # 1. Boilerplate setup
        x, y, _ = self._split_batch(batch)
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        batch_size = x.size(0)
        
        # SNR Filtering with Integrated Noise Application
        min_snr = 10.0
        max_retries = 20
        
        signal = x.view(x.size(0), -1).max(dim=1)[0]
        
        # Use a fixed seed for the entire test set for perfect reproducibility
        gen = torch.Generator(device=self.device)
        gen.manual_seed(42) # A fixed seed
        
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        snr_values = torch.zeros(batch_size, device=self.device)
        x_noisy = torch.zeros_like(x)
        
        for sample_idx in range(batch_size):
            x_sample = x[sample_idx:sample_idx+1]
            
            for attempt in range(max_retries):
                x_sample_noisy, rms = _apply_noise_and_mask_on_device(
                    x_sample.clone(), gen=gen,
                    use_cauchy_gauss=self.aug_use_cauchy_gauss,
                    cauchy_mu=self.aug_cauchy_mu, cauchy_sigma=self.aug_cauchy_sigma,
                    cauchy_threshold=self.aug_cauchy_threshold,
                    add_gauss_sigma=self.aug_add_gauss_sigma,
                    mask_frac=0.0, # Typically no masking/dropout in testing
                    return_rms=True
                )
                rms_value = rms[0].item()
                
                snr = signal[sample_idx].item() / rms_value if rms_value > 0 else float('inf')
                
                if snr >= min_snr:
                    valid_mask[sample_idx] = True
                    x_noisy[sample_idx] = x_sample_noisy[0]
                    snr_values[sample_idx] = snr
                    break
            
            if not valid_mask[sample_idx]:
                x_noisy[sample_idx] = x_sample_noisy[0]
                snr_values[sample_idx] = snr
        
        self._test_excluded_count += (batch_size - valid_mask.sum().item())
        
        # 2. Forward pass
        use_amp = x.is_cuda and "mixed" in str(getattr(self.trainer, "precision", ""))
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = self(x_noisy)
        # --- END: Copied and adapted logic ---
        
        output_dict = {
            "targets": y, 
            "snr": snr_values, 
            "valid_mask": valid_mask
        }
        loss = None
        
        stage = "test"
        # Handle mixture model outputs
        if isinstance(out, dict) and ("pi_logits" in out or self.fixed_mixture_weights):
            # --- Determine mixture weights pi ---
            if "pi_logits" in out:
                pi = F.softmax(out["pi_logits"], dim=-1)
                output_dict["pi_logits_all"] = out["pi_logits"] # Still save original logits
            else:
                pi = self.fixed_pi.unsqueeze(0).expand(out["mu"].shape[0], -1)

            
            if "sigma" in out:
                # Diagonal covariance mixture
                loss = self._mog_nll_diag(pi, out["mu"], out["sigma"], y)
                mu_mix, sig_mix = self._mixture_marginals_diag(pi, out["mu"], out["sigma"])
                
                # CRITICAL: Store ALL mixture data for corner plots
                output_dict.update({
                    "val_loss": loss.detach(),
                    "preds": mu_mix,
                    "sigmas": sig_mix,
                    "snr": snr_values,
                    "valid_mask": valid_mask,
                    "mu_all": out["mu"],                 # [B, K, d]
                    "sigma_all": out["sigma"],           # [B, K, d]
                    "targets": y
                })
                
            elif "L" in out:
                # Full covariance mixture
                loss = self._mog_nll_full(pi, out["mu"], out["L"], y)
                mu_mix, sig_mix = self._mixture_marginals_full(pi, out["mu"], out["L"])
                
                # CRITICAL: Store ALL mixture data for corner plots
                output_dict.update({
                    "val_loss": loss.detach(),
                    "preds": mu_mix,
                    "sigmas": sig_mix,
                    "snr": snr_values,
                    "valid_mask": valid_mask,
                    "mu_all": out["mu"],                 # [B, K, d]
                    "L_all": out["L"],                   # [B, K, d, d]
                    "targets": y
                })

        elif self.use_mdn and self.covariance_type == "full" and self._is_fullcov_out(out):
            # Single Gaussian full covariance
            mu, L = out["mu"], out["L"]
            loss = self._mvn_nll_from_cholesky(mu, L, y)
            sigma_diag = torch.sqrt((L.square()).sum(dim=-1))
            
            output_dict.update({
                "val_loss": loss.detach(),
                "preds": mu,
                "Ls": L,
                "sigmas": sigma_diag,
                "snr": snr_values,
                "valid_mask": valid_mask,
                "targets": y
            })

        self.test_step_outputs.append(output_dict)

        if loss is not None:
            self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            
        return output_dict
    
    def on_test_epoch_end(self):
        """
        Aggregates test results, handling DDP and SNR filtering.
        """
        outputs = self.test_step_outputs
        
        # --- START: Copied and adapted logic from on_validation_epoch_end ---
        if not outputs:
            self._test_cache = {}
            outputs.clear()
            return

        # Aggregate ALL samples
        keys_to_aggregate = ["preds", "targets", "sigmas", "snr", "valid_mask"] # Add other keys if needed
        local_aggregated = {}
        for key in keys_to_aggregate:
            tensor_list = [o[key] for o in outputs if key in o and o[key] is not None]
            if tensor_list:
                local_aggregated[key] = torch.cat([t.detach().cpu() for t in tensor_list], dim=0)

        # Gather across GPUs
        final_cache = {}
        if self.trainer.world_size > 1:
            gathered = self.all_gather(local_aggregated)
            if self.trainer.is_global_zero:
                # Concatenate results from all GPUs
                for key, tensor_list in gathered.items():
                    final_cache[key] = torch.cat(list(tensor_list), dim=0)
            else:
                outputs.clear()
                return
        else:
            final_cache = local_aggregated
        
        # On rank 0, perform filtering and logging
        if self.trainer.is_global_zero:
            total_samples = len(final_cache.get("targets", []))
            valid_samples = final_cache["valid_mask"].sum().item()
            
            print("\n[Test Summary]")
            print(f"  Total samples processed: {total_samples}")
            print(f"  Valid samples (SNR >= minimum): {valid_samples}")
            print(f"  Excluded samples: {total_samples - valid_samples}")

            # Filter results for final cache
            valid_mask = final_cache["valid_mask"].bool()
            self._test_cache = {}
            for key in ["preds", "targets", "sigmas", "snr"]:
                if key in final_cache and final_cache[key] is not None:
                    self._test_cache[key] = final_cache[key][valid_mask]
            print(f"  Samples in final test cache: {len(self._test_cache.get('targets', []))}")
        
        outputs.clear()
        # --- END: Copied and adapted logic ---

    

    # ------------------- Optimizer and Scheduler Configuration -------------------

    def configure_optimizers(self):
        """
        Sets up the optimizer with ALL parameters initially.
        """
        # Important: Do not filter by requires_grad=True here. 
        # Pass ALL model parameters so the optimizer tracks them even while frozen.
        optimizer = torch.optim.AdamW(
            self.model.parameters(), # Pass full model.parameters()
            lr=self.hparams.optim_cfg.lr,
            weight_decay=self.hparams.optim_cfg.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "frequency": 1
            }
        }
    
    # Gradient debugging hook for DDP training

    def on_before_optimizer_step(self, optimizer):
        # --- DDP GRADIENT DEBUGGER ---
        # This hook is guaranteed to run after .backward() and before .step().
        # We check on batch_idx 0 of every epoch.
        plot_gradient_debug_print = True  # Set to True to enable gradient debugging prints
        if self.trainer.global_step % self.trainer.num_training_batches == 0 and self.trainer.is_global_zero and plot_gradient_debug_print:
            print(f"\n--- [GRADIENT DEBUGGER - Epoch {self.trainer.current_epoch}, Stage {'1' if self.is_stage_1 else '2'}] ---")
            found_culprit = False
            # Use a color library if available for readability, otherwise use plain text
            try:
                from colorama import Fore, Style
                RED = Fore.RED
                GREEN = Fore.GREEN
                YELLOW = Fore.YELLOW
                RESET = Style.RESET_ALL
            except ImportError:
                RED = GREEN = YELLOW = RESET = ""

            for name, param in self.named_parameters():
                # Check for the culprit: a parameter that REQUIRES a gradient but did NOT receive one.
                if param.requires_grad and param.grad is None:
                    print(f"{RED}[CULPRIT] Param '{name}' requires a gradient but did not receive one.{RESET}")
                    found_culprit = True
            
            if not found_culprit:
                print(f"{GREEN}All parameters that require gradients received them. No culprits found.{RESET}")
            print(f"--- [END DEBUGGER] ---\n", flush=True)

    

    # --- START: NEW 3-STAGE TRAINING HOOKS ---

    def _set_trainable_layers(self, backbone_is_trainable: bool, cov_head_is_trainable: bool, pi_head_is_trainable: bool=True):
        """A helper function to set the requires_grad status for model parts."""
        # Freeze/unfreeze the main model backbone (everything except cov_head)
        for name, param in self.model.named_parameters():
            if 'cov_head' not in name:
                param.requires_grad = backbone_is_trainable
            if 'pi_head' in name:
                param.requires_grad = pi_head_is_trainable
        
        # Freeze/unfreeze the covariance head
        for param in self.model.cov_head.parameters():
            param.requires_grad = cov_head_is_trainable

    def setup(self, stage: str):
        """
        Called at the beginning of fit. Always sets the initial state to Stage 1.
        """
        if stage == 'fit':
            is_three_stage = self.hparams.strategy_cfg.get("two_stage_training", {}).get("enabled", False)
            if is_three_stage:
                self.print(f"\n{'='*80}")
                self.print(f"SETUP: 3-Stage training enabled. Starting in Stage 1 (backbone trainable, cov_head frozen).")
                self.print(f"{'='*80}\n", flush=True)
                self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=False)
                self.is_stage_1 = True
            else:
                self.is_stage_1 = False # Standard single-stage training

    # In lightning_resnet.py, replace your on_train_start with this simpler version

    def on_train_start(self):
        """
        Ensures the correct layers are frozen/unfrozen based on the epoch 
        when resuming from a checkpoint.
        """
        if not self.hparams.strategy_cfg.get("two_stage_training", {}).get("enabled", False):
            return

        s2_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage2
        s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3
        optimizer = self.trainer.optimizers[0]
        
        self.print(f"ON_TRAIN_START: Configuring layers for epoch {self.current_epoch}.")

        # Just set the frozen status; do NOT touch the optimizer here.
        if self.current_epoch >= s3_epoch:
            self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=True, pi_head_is_trainable=True)
            # Update LR for fine-tuning IN PLACE
            fine_tune_lr = self.hparams.optim_cfg.lr / 100.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = fine_tune_lr
                
            self.print(f"Optimizer LR updated to {fine_tune_lr} (State preserved)")
            self.is_stage_1 = False
        elif self.current_epoch >= s2_epoch:
            self._set_trainable_layers(backbone_is_trainable=False, cov_head_is_trainable=True)
            self.is_stage_1 = False
        else:
            self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=False)
            self.is_stage_1 = True

    

    # def on_train_start(self):
    #     """
    #     Called after checkpoint loading. Its main job now is to ensure the
    #     optimizer is correctly configured for the CURRENT state of the model.
    #     """
    #     is_three_stage = self.hparams.strategy_cfg.get("two_stage_training", {}).get("enabled", False)
    #     if not is_three_stage:
    #         return

    #     s2_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage2
    #     s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3
        
    #     self.print(f"ON_TRAIN_START: Configuring optimizer for epoch {self.current_epoch}.")

    #     if self.current_epoch >= s3_epoch:
    #         # Configure optimizer for Stage 3 (fine-tuning all params)
    #         new_lr = self.hparams.optim_cfg.lr / 100.0
    #         self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=True, pi_head_is_trainable=True)
    #         # self.parameters() will now return all trainable params, which is the entire model in Stage 3.
    #         optimizer = torch.optim.AdamW(self.parameters(), lr=new_lr, weight_decay=self.hparams.optim_cfg.weight_decay)
    #         self.trainer.optimizers = [optimizer]

    #     elif self.current_epoch >= s2_epoch:
    #         # Configure optimizer for Stage 2 (cov_head only)
    #         optimizer = torch.optim.AdamW(self.model.cov_head.parameters(), lr=self.hparams.optim_cfg.lr, weight_decay=self.hparams.optim_cfg.weight_decay)
    #         self.trainer.optimizers = [optimizer]
            
    #     else:
    #         # Configure optimizer for Stage 1 (backbone only)
    #         # self.parameters() will correctly return only the trainable params.
    #         optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.optim_cfg.lr, weight_decay=self.hparams.optim_cfg.weight_decay)
    #         self.trainer.optimizers = [optimizer]

    # def on_train_epoch_start(self):
    #     """
    #     Handles the live transitions between stages during a continuous run.
    #     """
    #     is_three_stage = self.hparams.strategy_cfg.get("two_stage_training", {}).get("enabled", False)
    #     if not is_three_stage: return

    #     s2_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage2
    #     s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3

    #     # --- Transition to Stage 2 ---
    #     if self.is_stage_1 and self.current_epoch == s2_epoch:
    #         self.print(f"\n{'='*80}")
    #         self.print(f"TRANSITIONING to STAGE 2 at epoch {self.current_epoch}: Training cov_head ONLY.")
    #         self.print(f"{'='*80}\n", flush=True)
    #         self._set_trainable_layers(backbone_is_trainable=False, cov_head_is_trainable=True)
    #         self.is_stage_1 = False # We are no longer in Stage 1
    #         # Reconfigure optimizer for only cov_head
    #         optimizer = torch.optim.AdamW(self.model.cov_head.parameters(), lr=self.hparams.optim_cfg.lr, weight_decay=self.hparams.optim_cfg.weight_decay)
    #         self.trainer.optimizers = [optimizer]

    #     # --- Transition to Stage 3 ---
    #     # The condition `not self.is_stage_1` ensures this only runs after Stage 2 has started.
    #     elif not self.is_stage_1 and self.current_epoch == s3_epoch:
    #         self.print(f"\n{'='*80}")
    #         self.print(f"TRANSITIONING to STAGE 3 at epoch {self.current_epoch}: Fine-tuning ALL parameters.")
    #         self.print(f"{'='*80}\n", flush=True)
            
    #         # Reconfigure optimizer for all parameters with a tiny learning rate
    #         self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=True, pi_head_is_trainable=True)

    #         # --- Set up differential learning rates ---
    #         fine_tune_lr = self.hparams.optim_cfg.lr / 100.0 # Tiny LR for the backbone
    #         head_lr = self.hparams.optim_cfg.lr / 20.0      # A slightly larger LR for the heads
    #         pi_lr = self.hparams.optim_cfg.lr / 50.0        # Pi head is not frozen, so set a moderate LR for it
    #         # Create parameter groups
    #         backbone_params = [
    #             p for n, p in self.model.named_parameters() 
    #             if 'cov_head' not in n and 'pi_head' not in n and 'output_heads' not in n
    #         ]
    #         head_params = [
    #             p for n, p in self.model.named_parameters() 
    #             if 'cov_head' in n or 'output_heads' in n # Exclude pi_head from the head_params to keep it frozen in Stage 3
    #         ]
    #         pi_head_params = [
    #             p for n, p in self.model.named_parameters() 
    #             if 'pi_head' in n
    #         ]

    #         param_groups = [
    #             {"params": backbone_params, "lr": fine_tune_lr},
    #             {"params": head_params, "lr": head_lr},
    #             {"params": pi_head_params, "lr": pi_lr}
    #         ]

    #         optimizer = torch.optim.AdamW(
    #             param_groups, # Pass the groups instead of self.parameters()
    #             # The main 'lr' is now just a default, the group LRs will be used
    #             lr=fine_tune_lr, 
    #             weight_decay=self.hparams.optim_cfg.weight_decay,
    #         )
    #         trainable_params = [p for p in self.parameters() if p.requires_grad]

    #         # Debug print to verify which parameters are trainable at the start of Stage 3
    #         # for n, p in self.named_parameters():
    #         #     if p.requires_grad:
    #         #         print("TRAINABLE:", n)

    #         # optimizer = torch.optim.AdamW(
    #         #     trainable_params, # Only include parameters that are currently trainable
    #         #     lr=fine_tune_lr, 
    #         #     weight_decay=self.hparams.optim_cfg.weight_decay,
    #         # )
    #         self.trainer.optimizers = [optimizer]

    def on_train_epoch_start(self):
        """
        Handles transitions between stages by modifying parameter groups 
        IN PLACE, without resetting optimizer state.
        """
        if not self.hparams.strategy_cfg.get("two_stage_training", {}).get("enabled", False):
            return

        s2_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage2
        s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3
        optimizer = self.trainer.optimizers[0]

        # --- Transition to Stage 2 ---
        if self.is_stage_1 and self.current_epoch == s2_epoch:
            self.print(f"\n{'='*80}")
            self.print(f"TRANSITIONING to STAGE 2 at epoch {self.current_epoch}: Training cov_head ONLY.")
            self.print(f"{'='*80}\n", flush=True)
            
            self._set_trainable_layers(backbone_is_trainable=False, cov_head_is_trainable=True)
            self.is_stage_1 = False 

            # Optional: Reset LR to base value if needed, but keeping momentum is key
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.hparams.optim_cfg.lr

        # --- Transition to Stage 3 ---
        # Check explicitly if we are entering Stage 3 or strictly inside it
        elif self.current_epoch == s3_epoch:
            # Check if we have already transitioned to avoid spamming/resetting LR every epoch
            # (Using a flag or just checking if backbone is frozen would work, 
            # but simpler to just run this logic once at the specific epoch)
            
            self.print(f"\n{'='*80}")
            self.print(f"TRANSITIONING to STAGE 3 at epoch {self.current_epoch}: Fine-tuning ALL parameters.")
            self.print(f"{'='*80}\n", flush=True)
            
            # Unfreeze
            self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=True, pi_head_is_trainable=False)
            
            # Update LR for fine-tuning IN PLACE
            fine_tune_lr = self.hparams.optim_cfg.lr / 100.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = fine_tune_lr
                
            self.print(f"Optimizer LR updated to {fine_tune_lr} (State preserved)")

   
   

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Robust Checkpoint Surgery (Fixed for Smooth Continuation).
        """
        state_dict = checkpoint["state_dict"]
        current_model_dict = self.model.state_dict()
        
        lightning_prefix = "model."       
        module_identifier = "cov_head"    
        
        # 1. Identify keys belonging to cov_head in checkpoint and current model
        ckpt_cov_keys = [k for k in state_dict.keys() if f"{lightning_prefix}{module_identifier}" in k]
        curr_cov_keys = [k for k in current_model_dict.keys() if k.startswith(module_identifier)]
        
        # Determine if surgery is needed by checking if shapes mismatch
        needs_surgery = False
        
        if len(ckpt_cov_keys) != len(curr_cov_keys):
            needs_surgery = True
        else:
            for local_key in curr_cov_keys:
                global_key = f"{lightning_prefix}{local_key}"
                # If key is missing or shapes don't match, we need surgery
                if global_key not in state_dict or state_dict[global_key].shape != current_model_dict[local_key].shape:
                    needs_surgery = True
                    break

        print(f"\n{'='*80}")
        print(f"[CHECKPOINT LOAD] Inspecting for architecture compatibility...")

        if needs_surgery:
            print(f" -> DETECTED ARCHITECTURE MISMATCH in '{module_identifier}'. Performing surgery.")
            
            # Step 1: Remove ALL old covariance keys from the checkpoint
            if ckpt_cov_keys:
                for k in ckpt_cov_keys:
                    del state_dict[k]
                print(f" -> Removed {len(ckpt_cov_keys)} incompatible parameters from checkpoint.")
            
            # Step 2: Inject FRESH initialized weights
            print(f" -> Injecting fresh initialization for {len(curr_cov_keys)} parameters.")
            for local_key in curr_cov_keys:
                global_key = f"{lightning_prefix}{local_key}"
                state_dict[global_key] = current_model_dict[local_key].clone().cpu()
            
            # Step 3: Wipe optimizer to prevent state mismatch crashes
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []
            print(" -> Wiped optimizer state for fresh start.")
            
            # Step 4: Force Stage 2
            print(" -> FORCING STAGE 2: Freezing backbone, training new cov_head.")
            self.is_stage_1 = False
            self._set_trainable_layers(backbone_is_trainable=False, cov_head_is_trainable=True)
            
        else:
            print(f" -> Perfect match for '{module_identifier}'. Loading weights smoothly without surgery.")
            
            # Normal stage logic based on the resumed epoch
            epoch = checkpoint.get("epoch", 0)
            s2_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage2
            s3_epoch = self.hparams.strategy_cfg.two_stage_training.switch_epoch_stage3
            
            if epoch >= s3_epoch:
                self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=True)
                self.is_stage_1 = False
            elif epoch >= s2_epoch:
                self._set_trainable_layers(backbone_is_trainable=False, cov_head_is_trainable=True)
                self.is_stage_1 = False
            else:
                self._set_trainable_layers(backbone_is_trainable=True, cov_head_is_trainable=False)
                self.is_stage_1 = True

        print(f"{'='*80}\n", flush=True)

