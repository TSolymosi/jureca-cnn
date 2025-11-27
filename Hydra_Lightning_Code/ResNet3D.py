import torch
import torch.nn as nn
import torch.nn.functional as F

# Enables cuDNN auto-tuner to find the best algorithm for the hardware.
torch.backends.cudnn.benchmark = True

import sys
# Allows importing modules from the specified project directory.
sys.path.insert(0, "/p/scratch/westai0043/CNN_HL_tobias/")



import dataloader
from lightning.pytorch.utilities import rank_zero_only


@rank_zero_only
def print_rank0(*args, **kwargs):
    """A wrapper for the print function that only executes on the main process (rank 0)
    in a distributed training setup. This prevents duplicate print statements from every GPU."""
    print(*args, **kwargs)
print_rank0(">>> Using dataloader from:", dataloader.__file__)


import numpy as np


# Override the default print function to ensure output is written immediately,
# which is useful for logging in environments with buffered output.
import functools
print = functools.partial(print, flush=True)

def get_base_dataset(loader):
    """
    Finds the underlying dataset object, navigating through PyTorch's
    wrapper datasets like Subset and DataLoader.
    """
    dataset = loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset

def get_inplanes():
    """Returns a list of standard channel sizes for ResNet architectures."""
    return [64, 128, 256, 512]

# These are helper functions to create 3D convolutions that only operate on a specific axis,
# simulating 1D or 2D convolutions on 3D data.

def spectral_conv1d(in_planes, out_planes, kernel_size=3, stride=1):
    """Creates a 3D convolution that acts only on the first (spectral) dimension."""
    padding = kernel_size // 2
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(kernel_size, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(padding, 0, 0),
                     bias=False)

def spatial_conv2d(in_planes, out_planes, kernel_size=3, stride=1):
    """Creates a 3D convolution that acts only on the last two (spatial) dimensions."""
    padding = kernel_size // 2
    return nn.Conv3d(in_planes, out_planes,
                     kernel_size=(1, kernel_size, kernel_size),
                     stride=(1, stride, stride),
                     padding=(0, padding, padding),
                     bias=False)

def get_num_groups(num_channels):
    """
    Determines an appropriate number of groups for Group Normalization.
    It tries to divide the channels into groups of 32, 16, etc., for stability.
    """
    if num_channels == 0: return 1 # Avoid division by zero.
    for g in [32, 16, 8, 4, 2]:
        if num_channels % g == 0:
            return g
    return 1 # Fallback to 1 group (equivalent to Layer Normalization).

# ----------------- Custom Model Heads -----------------

class MDNOutputHead(nn.Module):
    """
    A head for a Mixture Density Network (MDN) that predicts parameters for a Gaussian distribution.
    Specifically designed for the diagonal covariance case (predicting mu and sigma for each output parameter).
    """
    def __init__(self, input_dim, predict_sigma=True, num_components: int = 1, min_sigma: float = 1e-6):
        super().__init__()
        self.K = int(num_components)
        self.min_sigma = float(min_sigma)
        self.predict_sigma = predict_sigma

        # A simple linear layer to predict the mean (mu) of the Gaussian(s).
        self.mu_head = nn.Linear(input_dim, self.K)
        
        # A more complex sub-network to predict the standard deviation (sigma).
        # This gives the model more capacity to learn the uncertainty.
        if self.predict_sigma:
            self.sigma_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.ReLU(),
                nn.Linear(input_dim // 4, self.K)
            )
        
    def forward(self, x):
        """Takes a feature tensor and returns a dictionary of distribution parameters."""
        mu = self.mu_head(x)
        if self.predict_sigma:
            sigma_out = self.sigma_head(x)
            # Use softplus to ensure sigma is always positive, and add a floor for numerical stability.
            sigma = F.softplus(sigma_out) + self.min_sigma
            return {'mu': mu, 'sigma': sigma}
        return {'mu': mu}
    
class CovarianceHead(nn.Module):
    """
    Predicts full covariance matrix via constrained Cholesky factorization.
    
    GRADIENT-SAFE: No in-place operations anywhere.
    """
    
    def __init__(self, input_dim: int, d: int, num_components: int = 1, 
                 min_sigma: float = 0.0001, init_sigma_scale: float = 0.4,
                 per_param_scales: dict = None,
                 per_param_min_sigma: dict = None):
        """
        Args:
            input_dim: Dimension of input feature vector
            d: Number of target parameters
            num_components: Number of mixture components (K)
            min_sigma: Minimum allowed standard deviation
            init_sigma_scale: Initial scale for sigma predictions
            per_param_scales: Dict mapping parameter index to initial scale
        """
        super().__init__()
        self.d = d
        self.K = int(num_components)
        self.min_sigma = float(min_sigma)
        self.per_param_scales = per_param_scales or {}
        self.per_param_min_sigma = per_param_min_sigma or {} 
        
        # Head 1: Predict log(sigma) for each parameter
        out_dim_sigma = d if self.K == 1 else self.K * d
        self.log_sigma_head = nn.Linear(input_dim, out_dim_sigma)
        
        # Head 2: Predict lower-triangular correlation elements
        n_lower = (d * (d - 1)) // 2
        out_dim_lower = n_lower if self.K == 1 else self.K * n_lower
        self.lower_head = nn.Linear(input_dim, out_dim_lower)
        
        self._initialize_weights(init_sigma_scale)
    
    # In ResNet3D.py, inside the CovarianceHead class

    def _initialize_weights(self, init_sigma_scale):
            """
            Initialize weights for a TWO-STAGE training start.
            The goal is to start Stage 2 from a neutral, reasonable sigma baseline.
            """
            # --- NEW: Define a neutral starting point for all sigmas ---
            # A value of 0.3-0.4 is a sensible default uncertainty in the scaled space.
            neutral_target_sigma = 0.4

            with torch.no_grad():
                # The new goal is simple: set the initial bias so that the initial
                # sigma prediction is exactly our neutral target.
                # We no longer need different initializations for different components.
                # log(exp(target)) = target
                initial_bias = np.log(neutral_target_sigma)
                
                # Set the bias for the log_sigma_head to this value.
                self.log_sigma_head.bias.fill_(initial_bias)

                # We still want the weights to be small, so the bias dominates at the start.
                nn.init.normal_(self.log_sigma_head.weight, mean=0.0, std=0.001)
                
                # The correlation head initialization can remain the same.
                # Starting with zero correlation (an identity matrix) is a good neutral default.
                nn.init.zeros_(self.lower_head.bias)
                nn.init.normal_(self.lower_head.weight, mean=0.0, std=0.01)
    
    def forward(self, feats: torch.Tensor):
        """Forward pass."""
        B = feats.size(0)
        d, K = self.d, self.K
        
        # Predict standard deviations
        log_sigma = self.log_sigma_head(feats)
        
        if K == 1:
            sigma_raw = torch.exp(log_sigma)  # [B, d]
        
            # Apply per-parameter minimum WITHOUT in-place operations
            # Create a tensor of minimums and use torch.maximum
            min_vals = torch.tensor(
                [self.per_param_min_sigma.get(i, self.min_sigma) for i in range(d)],
                device=feats.device,
                dtype=feats.dtype
            ).unsqueeze(0)  # [1, d]
            
            sigma = torch.maximum(sigma_raw, min_vals)  # [B, d] 
        else:
            sigma_raw = torch.exp(log_sigma.view(B, K, d))  # [B, K, d]
        
            # Apply per-parameter minimum WITHOUT in-place operations
            min_vals = torch.tensor(
                [self.per_param_min_sigma.get(i, self.min_sigma) for i in range(d)],
                device=feats.device,
                dtype=feats.dtype
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, d]
            
            sigma = torch.maximum(sigma_raw, min_vals)  # [B, K, d] 
        
        # Predict lower-triangular correlation elements
        lower_raw = self.lower_head(feats)
        
        if K == 1:
            lower_vals = lower_raw
        else:
            n_lower = (d * (d - 1)) // 2
            lower_vals = lower_raw.view(B, K, n_lower)
        
        # Build Cholesky factor L
        if K == 1:
            L = self._build_cholesky_single(sigma, lower_vals, B, d)
            sigma_diag = torch.sqrt((L * L).sum(dim=-1))
            return {"L": L, "sigma_diag": sigma_diag}
        else:
            L = self._build_cholesky_mixture(sigma, lower_vals, B, K, d)
            sigma_diag = torch.sqrt((L * L).sum(dim=-1))
            return {"L": L, "sigma_diag": sigma_diag}
    
    def _build_cholesky_single(self, sigma, lower_vals, B, d):
        """
        Build Cholesky factor for single Gaussian.
        ROBUST: Prevents diagonal collapse through careful constraint enforcement.
        """
        device = sigma.device
        dtype = sigma.dtype
        
        # Work in float32 for numerical stability
        sigma_f32 = sigma.float()
        lower_vals_f32 = lower_vals.float()
        
        # Strategy: Build L_unit with explicit constraint that diagonal >= min_diag
        # We do this by limiting the magnitude of off-diagonal elements
        
        MIN_DIAG = 0.01  # Minimum diagonal value (ensures reasonable uncertainty)
        MAX_OFF_DIAG_FRACTION = 0.95  # Max fraction of "budget" used by off-diagonals
        
        L_unit_elements = []
        
        idx = 0
        for i in range(d):
            row_elements = []
            
            if i == 0:
                # First row: [1, 0, 0, ..., 0]
                for j in range(d):
                    if j == 0:
                        row_elements.append(torch.ones(B, device=device, dtype=torch.float32))
                    else:
                        row_elements.append(torch.zeros(B, device=device, dtype=torch.float32))
            else:
                # For row i: compute off-diagonal elements first
                off_diag_vals = []
                for j in range(i):
                    # Raw correlation value
                    raw_val = torch.tanh(lower_vals_f32[:, idx]) * 0.95
                    off_diag_vals.append(raw_val)
                    idx += 1
                
                # Check if off-diagonals would leave room for minimum diagonal
                off_diag_tensor = torch.stack(off_diag_vals, dim=1)  # [B, i]
                off_diag_sq_sum = (off_diag_tensor ** 2).sum(dim=1)  # [B]
                
                # Maximum allowed sum of squares to preserve MIN_DIAG
                max_allowed_sq_sum = 1.0 - (MIN_DIAG ** 2)
                
                # Scale down if needed
                needs_scaling = off_diag_sq_sum > max_allowed_sq_sum
                if needs_scaling.any():
                    scale = torch.ones(B, device=device, dtype=torch.float32)
                    scale[needs_scaling] = torch.sqrt(max_allowed_sq_sum / off_diag_sq_sum[needs_scaling])
                    scale = scale.unsqueeze(1)  # [B, 1]
                    off_diag_tensor = off_diag_tensor * scale
                    off_diag_sq_sum = (off_diag_tensor ** 2).sum(dim=1)
                
                # Now compute diagonal
                diag_sq = torch.clamp(1.0 - off_diag_sq_sum, min=MIN_DIAG ** 2)
                diag_val = torch.sqrt(diag_sq)
                
                # Build the full row
                for j in range(d):
                    if j < i:
                        row_elements.append(off_diag_tensor[:, j])
                    elif j == i:
                        row_elements.append(diag_val)
                    else:
                        row_elements.append(torch.zeros(B, device=device, dtype=torch.float32))
            
            # Add this row's elements to the main list
            L_unit_elements.extend(row_elements)
        
        # Stack all elements
        L_unit_flat = torch.stack(L_unit_elements, dim=1)  # [B, d*d]
        L_unit = L_unit_flat.view(B, d, d)
        
        # Scale by diagonal matrix D = diag(σ)
        D = torch.diag_embed(sigma_f32)  # [B, d, d]
        L = torch.bmm(D, L_unit)  # [B, d, d]
        
        # Convert back to original dtype
        return L.to(dtype)
    
    def _build_cholesky_mixture(self, sigma, lower_vals, B, K, d):
        """
        Build Cholesky factors for mixture of Gaussians.
        
        Args:
            sigma: Standard deviations [B, K, d]
            lower_vals: Raw predictions for lower triangle [B, K, n_lower]
            B: Batch size
            K: Number of components
            d: Number of parameters
        
        Returns:
            L: Cholesky factors [B, K, d, d]
        """
        # Build each component separately and stack
        L_components = []
        
        for k in range(K):
            sigma_k = sigma[:, k, :]  # [B, d]
            lower_k = lower_vals[:, k, :]  # [B, n_lower]
            
            L_k = self._build_cholesky_single(sigma_k, lower_k, B, d)
            L_components.append(L_k)
        
        # Stack along mixture dimension
        L = torch.stack(L_components, dim=1)  # [B, K, d, d]
        
        return L



class CovarianceHead_old(nn.Module):
    """
    A head designed to predict a full covariance matrix by predicting its Cholesky factor, L.
    This is the key component for enabling corner plots.
    For a d-dimensional output, it predicts the d*(d+1)/2 unique elements of the lower-triangular L matrix.
    
    FIXED: Now initializes with proper diagonal scale to prevent suppressed uncertainties.
    """
    def __init__(self, input_dim: int, d: int, num_components: int = 1, jitter: float = 1e-6, 
                 init_diag_scale: float = 1.0, per_param_scales: dict = None):
        super().__init__()
        self.d = d # The number of target parameters (dimensionality).
        self.K = int(num_components)
        self.jitter = float(jitter) # A small value to add to the diagonal for stability.
        self.init_diag_scale = float(init_diag_scale)
        self.per_param_scales = per_param_scales or {}  # Dict mapping param index to scale

        # Calculate the number of elements in a lower-triangular d x d matrix.
        packed_elements = d * (d + 1) // 2
        
        # The output dimension of the linear layer is the number of elements to predict.
        out_dim = packed_elements if self.K == 1 else self.K * packed_elements
        self.L_params = nn.Linear(input_dim, out_dim)
        
        # CRITICAL FIX: Initialize the bias to encourage reasonable initial uncertainties
        # For diagonal elements (which control the scale), initialize to positive values
        with torch.no_grad():
            # Create indices for diagonal element positions in the packed lower-triangular format
            # For a 7x7 matrix (your 7 parameters), diagonal elements are at positions:
            # i=0: pos 0, i=1: pos 2, i=2: pos 5, i=3: pos 9, i=4: pos 14, i=5: pos 20, i=6: pos 27
            # Pattern: position = i*(i+1)/2 + i = i*(i+3)/2
            diag_indices = []
            idx = 0
            for i in range(d):
                diag_indices.append(idx)
                idx += (i + 2)  # Move to next diagonal: skip (i+1) off-diagonal elements + 1 for next row start
            
            # For each mixture component, initialize diagonal biases
            if self.K == 1:
                for param_idx, diag_idx in enumerate(diag_indices):
                    # Use per-parameter scale if provided, otherwise use default
                    scale = self.per_param_scales.get(param_idx, self.init_diag_scale)
                    self.L_params.bias[diag_idx] = scale
            else:
                for k in range(self.K):
                    offset = k * packed_elements
                    for param_idx, diag_idx in enumerate(diag_indices):
                        scale = self.per_param_scales.get(param_idx, self.init_diag_scale)
                        # Scale up uncertainty for non-primary components
                        if k > 0:
                            scale *= (1.5 + 0.3 * k)
                        self.L_params.bias[offset + diag_idx] = scale
        
        self.softplus = nn.Softplus()

    def forward(self, feats: torch.Tensor):
        """Takes a feature tensor and returns a dictionary containing the Cholesky matrix L."""
        B, d, K = feats.size(0), self.d, self.K
        
        # Predict the flat vector of lower-triangular elements.
        lvec = self.L_params(feats)
        
        # --- Handle single Gaussian component case (K=1) ---
        if K == 1:
            # Create a zero-filled matrix to be populated.
            L = feats.new_zeros(B, d, d)
            # Get the indices of the lower-triangular part of a matrix.
            i, j = torch.tril_indices(d, d, device=feats.device)
            # Fill the lower-triangular part with the predicted elements.
            L[:, i, j] = lvec
            
            # Ensure the diagonal of L is positive, which is required for a valid Cholesky factor.
            diag = torch.diagonal(L, dim1=-2, dim2=-1)
            diag_pos = self.softplus(diag) + self.jitter
            
            # Create a new L with the positive diagonal (in-place modification can cause issues).
            L_final = torch.tril(L) - torch.diag_embed(diag) + torch.diag_embed(diag_pos)

            # For convenience, also calculate the marginal standard deviations.
            sigma_diag = torch.sqrt((L_final.square()).sum(dim=-1))
            return {"L": L_final, "sigma_diag": sigma_diag}

        # --- Handle mixture of Gaussians case (K > 1) ---
        P = d * (d + 1) // 2
        lvec = lvec.view(B, K, P)
        L = feats.new_zeros(B, K, d, d)
        i, j = torch.tril_indices(d, d, device=feats.device)
        L[:, :, i, j] = lvec
        
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_pos = self.softplus(diag) + self.jitter
        L_final = torch.tril(L) - torch.diag_embed(diag) + torch.diag_embed(diag_pos)
        
        sigma_diag = torch.sqrt((L_final.square()).sum(dim=-1))
        return {"L": L_final, "sigma_diag": sigma_diag}


# ----------------- Standard ResNet Building Blocks -----------------
class BasicBlock2D(nn.Module):
    """A standard 2D ResNet 'BasicBlock' used in ResNet-18 and ResNet-34."""
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if use_batchnorm else nn.GroupNorm(get_num_groups(planes), planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else nn.GroupNorm(get_num_groups(planes), planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck2D(nn.Module):
    """A standard 2D ResNet 'Bottleneck' block used in deeper models like ResNet-50."""
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, use_batchnorm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes) if use_batchnorm else nn.GroupNorm(get_num_groups(planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else nn.GroupNorm(get_num_groups(planes), planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if use_batchnorm else nn.GroupNorm(get_num_groups(planes * self.expansion), planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ----------------- Main Model Architecture -----------------
class Spectral2DResNet(nn.Module):
    """
    The main model class. It processes 3D spectral cubes by first applying a series
    of 1D spectral convolutions, then reshaping the result into a 2D-like feature map,
    and finally passing it through a standard 2D ResNet backbone.
    """
    def __init__(self,
                 block_2d, spatial_layers_config, block_inplanes_spectral=64,
                 num_wavelengths_in=2000, initial_proj_channels=64, n_outputs=4,
                 dropout_prob=0.2, use_batchnorm=True, fc_hidden_dim=512,
                 target_params_list=None,
                 use_attention_heads=False, attention_latent_dim=128, use_mdn = True, covariance_type: str = "diagonal", num_mixtures: int = 1,
                 covariance_mode='cholesky_constrained',
                 fixed_mixture_weights: list = None,):
        super().__init__()
        # Store configuration parameters.
        self.use_mdn = use_mdn
        self.covariance_type = covariance_type
        # If fixed weights are given, they determine the number of mixtures
        if fixed_mixture_weights is not None:
            self.num_mixtures = len(fixed_mixture_weights)
            print_rank0(f"[MODEL INFO] Using fixed mixture weights. Number of components set to {self.num_mixtures}.")
            # The "static logits" trick: convert fixed probabilities to stable logits.
            # We take the log of the probabilities. This is the inverse of softmax's exp().
            # When F.log_softmax is applied later, it will recover the correct log probabilities.
            pi = torch.tensor(fixed_mixture_weights, dtype=torch.float32)
            static_logits = torch.log(pi.clamp(min=1e-9))
            
            # Register as a buffer. This ensures the tensor is moved to the correct
            # device (e.g., GPU) with the model, but is not considered a trainable parameter.
            self.register_buffer("static_pi_logits", static_logits)
        else:
            self.num_mixtures = num_mixtures
        self.target_params_list = target_params_list if target_params_list is not None else [f"p_{i}" for i in range(n_outputs)]
        self.covariance_mode = covariance_mode    

        # --- Part 1: Spectral Convolution Frontend ---
        # A series of 1D-like convolutions to reduce the spectral dimension and extract features.
        spectral_kernel_size1, spectral_stride1 = 11, 4; spec_padding1 = spectral_kernel_size1 // 2
        d_out1 = (num_wavelengths_in - spectral_kernel_size1 + 2 * spec_padding1) // spectral_stride1 + 1
        self.spectral_conv_s1 = nn.Sequential(
            spectral_conv1d(1, 16, kernel_size=spectral_kernel_size1, stride=spectral_stride1),
            nn.BatchNorm3d(16) if use_batchnorm else nn.GroupNorm(get_num_groups(16), 16),
            nn.ReLU(inplace=True))
        spectral_kernel_size2, spectral_stride2 = 11, 4; spec_padding2 = spectral_kernel_size2 // 2
        d_out2 = (d_out1 - spectral_kernel_size2 + 2 * spec_padding2) // spectral_stride2 + 1
        self.spectral_conv_s2 = nn.Sequential(
            spectral_conv1d(16, 32, kernel_size=spectral_kernel_size2, stride=spectral_stride2),
            nn.BatchNorm3d(32) if use_batchnorm else nn.GroupNorm(get_num_groups(32), 32),
            nn.ReLU(inplace=True))
        spectral_kernel_size3, spectral_stride3 = 5, 2; spec_padding3 = spectral_kernel_size3 // 2
        d_out3 = (d_out2 - spectral_kernel_size3 + 2 * spec_padding3) // spectral_stride3 + 1
        self.spectral_conv_s3 = nn.Sequential(
            spectral_conv1d(32, block_inplanes_spectral, kernel_size=spectral_kernel_size3, stride=spectral_stride3),
            nn.BatchNorm3d(block_inplanes_spectral) if use_batchnorm else nn.GroupNorm(get_num_groups(block_inplanes_spectral), block_inplanes_spectral),
            nn.ReLU(inplace=True))

        self.final_spectral_dim = d_out3

        # Calculate the size of the spectral dimension after the convolutions.
        # This is needed to know how to reshape the tensor later.
        # ... (calculation of self.final_spectral_dim) ...
        
        reshaped_channels = block_inplanes_spectral * self.final_spectral_dim

        # --- Part 2: 2D ResNet Backbone ---
        # A 1x1 convolution to project the reshaped channels to the desired input size for the ResNet.
        self.entry_projection_2d = nn.Sequential(
            nn.Conv2d(reshaped_channels, initial_proj_channels, kernel_size=1),
            nn.BatchNorm2d(initial_proj_channels) if use_batchnorm else nn.GroupNorm(get_num_groups(initial_proj_channels), initial_proj_channels),
            nn.ReLU(inplace=True))

        # Build the spatial ResNet layers based on the provided configuration.
        self.in_planes = initial_proj_channels
        self.spatial_layers = nn.ModuleList()
        for i, (planes_out, stride, num_blocks) in enumerate(spatial_layers_config):
            self.spatial_layers.append(
                self._make_spatial_layer_2d(block_2d, planes_out, num_blocks, stride, use_batchnorm))

        # --- Part 3: Feature Extraction and Output Heads ---
        self.avgpool_2d = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling.
        final_spatial_channels = spatial_layers_config[-1][0] * block_2d.expansion
        
        # A shared fully connected (FC) block to create a final feature vector.
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_spatial_channels, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob))

        # --- Logic for creating the final output heads based on configuration ---
        self.output_heads = nn.ModuleDict()
        
        # Determine if separate sigma predictions are needed. They are NOT needed for the full covariance case
        # because the sigmas are derived from the diagonal of the predicted covariance matrix.
        predict_sigma = not (self.use_mdn and self.covariance_type == "full")
        
        # Create a separate head for each target parameter.
        for param_name in self.target_params_list:
            if use_mdn:
                # If using MDN, each parameter gets an MDN head to predict its distribution.
                self.output_heads[param_name] = MDNOutputHead(
                    fc_hidden_dim, predict_sigma=predict_sigma, num_components=self.num_mixtures
                )
            else:
                # For simple regression, each parameter gets a standard linear head.
                self.output_heads[param_name] = nn.Linear(fc_hidden_dim, 1)

        # If using full covariance MDN, create the dedicated covariance head.
        if self.use_mdn and self.covariance_type == "full":
            self.cov_head = CovarianceHead(
                fc_hidden_dim, 
                n_outputs, 
                num_components=self.num_mixtures,
                per_param_scales={
                    0: 1.0,   # First parameter (M or D depending on your order)
                    1: 1.0,   # Second parameter (D or L) - CRITICAL: needs larger initial scale
                    2: 1.0,   # ro
                    3: 1.0,   # rr or other param
                    4: 1.0,   # p
                    5: 1.0,   # Tlow
                    6: 1.0,   # NCH3CN
                },
                per_param_min_sigma={  # NEW: parameter-specific minimums
                    0: 0.0001,   # M 
                    1: 0.0001,   # D 
                    2: 0.0001,   # L
                    3: 0.0001,   # ro
                    4: 0.0001,   # p
                    5: 0.0001,   # Tlow
                    6: 0.0001,   # NCH3CN
                },
                
            )

        # If using a mixture of Gaussians (K>1) AND weights are not fixed, 
        # create a head to predict the mixture weights (pi).
        if self.use_mdn and self.num_mixtures > 1 and not fixed_mixture_weights:
            self.pi_head = nn.Linear(fc_hidden_dim, self.num_mixtures)
            
        self._initialize_weights()

    def _make_spatial_layer_2d(self, block_2d, planes, num_blocks, stride, use_batchnorm):
        """Helper function to create one stage of the ResNet backbone."""
        downsample = None
        # A downsample block is needed if changing spatial dimensions (stride!=1) or channel depth.
        if stride != 1 or self.in_planes != planes * block_2d.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block_2d.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_2d.expansion) if use_batchnorm else nn.GroupNorm(get_num_groups(planes * block_2d.expansion), planes * block_2d.expansion))
        
        layers = []
        layers.append(block_2d(self.in_planes, planes, stride, downsample, use_batchnorm=use_batchnorm))
        self.in_planes = planes * block_2d.expansion
        for _ in range(1, num_blocks):
            layers.append(block_2d(self.in_planes, planes, use_batchnorm=use_batchnorm))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initializes model weights using standard practices (Kaiming for conv, Xavier for linear)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """The main forward pass of the model."""
        # --- Spectral Frontend ---
        x = self.spectral_conv_s1(x)
        x = self.spectral_conv_s2(x)
        x = self.spectral_conv_s3(x)

        # --- Reshape for 2D Backbone ---
        N, C_spec, D_red, H, W = x.shape
        # Combine the spectral channels and the reduced spectral dimension into one channel dimension.
        x = x.view(N, C_spec * D_red, H, W)
        x = self.entry_projection_2d(x)

        # --- 2D ResNet Backbone ---
        for i, layer in enumerate(self.spatial_layers):
            x = layer(x)

        # --- Feature Extraction & Head ---
        x = self.avgpool_2d(x)
        shared_features = self.shared_fc(x)

        # --- Head Logic: Assemble the final output based on the model configuration ---

        # Collect mu and sigma predictions from the per-parameter heads.
        mus_list, sigmas_list = [], []
        for pname in self.target_params_list:
            out_p = self.output_heads[pname](shared_features)
            mus_list.append(out_p['mu'])
            # Only collect sigmas if in diagonal mode.
            if self.use_mdn and self.covariance_type == "diagonal":
                sigmas_list.append(out_p['sigma'])

        K = int(self.num_mixtures)
        # Stack the means into a single tensor.
        mu = torch.stack(mus_list, dim=-1) # Shape: [B, K, d]
        if K == 1: mu = mu.squeeze(1)      # Shape: [B, d]

        # --- Full Covariance Path ---
        if self.use_mdn and self.covariance_type == "full":
            cov = self.cov_head(shared_features) # Get the Cholesky matrix L.
            if K == 1:
                # For a single Gaussian, return the mean vector and the L matrix. This is the path for corner plots.
                return {"mu": mu, "L": cov["L"]}
            else:
                output_dict = {"mu": mu, "L": cov["L"]}
                if hasattr(self, 'pi_head'):
                    # Predict logits dynamically
                    output_dict["pi_logits"] = self.pi_head(shared_features)
                else:
                    # Provide the static logits, expanded to the batch size
                    output_dict["pi_logits"] = self.static_pi_logits.unsqueeze(0).expand(x.shape[0], -1)
                return output_dict

        # --- Diagonal Covariance Path ---
        if self.use_mdn and self.covariance_type == "diagonal":
            sigma = torch.stack(sigmas_list, dim=-1)
            if K == 1:
                # For compatibility with the old lightning module, return a flat tuple: (mu1, sigma1, mu2, sigma2, ...)
                flat_output = []
                for j in range(mu.shape[1]): # Iterate over parameters
                    flat_output.append(mu[:, j:j+1])
                    flat_output.append(sigma[:, j:j+1].squeeze(1))
                return tuple(flat_output)
            else:
                # For a diagonal mixture, return mixture weights, means, and sigmas.
                output_dict = {"mu": mu, "sigma": sigma}
                if hasattr(self, 'pi_head'):
                    # Predict logits dynamically
                    output_dict["pi_logits"] = self.pi_head(shared_features)
                else:
                    # Provide the static logits, expanded to the batch size
                    output_dict["pi_logits"] = self.static_pi_logits.unsqueeze(0).expand(x.shape[0], -1)
                return output_dict
        
        # --- Simple Regression Path (No MDN) ---
        # If not using MDN, the `mus_list` actually contains the final parameter predictions.
        output = mu 
        return output
    

# ----------------- Model Factory Function -----------------

def generate_2d_model(config_name="resnet18_2d_equivalent", TARGET_PARAMETERS=None, **kwargs):
    """
    A factory function that constructs and returns a Spectral2DResNet model
    based on a simple configuration name (e.g., "resnet18_2d_equivalent").
    """
    if TARGET_PARAMETERS is None:
        TARGET_PARAMETERS = ["p1", "p2", "p3", "p4"]

    # Common settings for all model configurations.
    default_kwargs = {
        'num_wavelengths_in': 2000,
        'block_inplanes_spectral': 64,
        'initial_proj_channels': 64,
        'n_outputs': len(TARGET_PARAMETERS),
        'dropout_prob': 0.2,
        'use_batchnorm': False,
        'fc_hidden_dim': 512,
        'target_params_list': TARGET_PARAMETERS,
        'covariance_type': "diagonal",
        'num_mixtures': 1,
        'covariance_mode': 'cholesky_constrained',
        'fixed_mixture_weights': None,
    }
    
    # Define the layer structure for different ResNet depths.
    # Each tuple is (output_channels, stride, num_blocks_in_stage).
    if config_name == "resnet10_2d_equivalent":
        spatial_config = [(64, 1, 1), (128, 2, 1), (256, 2, 1), (512, 2, 1)]
        block_type = BasicBlock2D
    elif config_name == "resnet18_2d_equivalent":
        spatial_config = [(64, 1, 2), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
        block_type = BasicBlock2D
    elif config_name == "resnet34_2d":
        spatial_config = [(64, 1, 3), (128, 2, 4), (256, 2, 6), (512, 2, 3)]
        block_type = BasicBlock2D
    elif config_name == "resnet50_2d":
        spatial_config = [(64, 1, 3), (128, 2, 4), (256, 2, 6), (512, 2, 3)]
        block_type = Bottleneck2D # Deeper models use the more efficient Bottleneck block.
    # ... other configurations ...
    else:
        raise ValueError(f"Unknown config_name: {config_name}")

    # Override default settings with any user-provided keyword arguments.
    final_kwargs = {**default_kwargs, **kwargs}

    # Instantiate and return the model.
    model = Spectral2DResNet(block_2d=block_type,
                             spatial_layers_config=spatial_config,
                             **final_kwargs)
    return model


