import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import torchvision.models as models
from torch.profiler import profile, record_function, schedule, tensorboard_trace_handler

# Enables cuDNN auto-tuner to find the best algorithm for the hardware.
torch.backends.cudnn.benchmark = True

import sys
# Allows importing modules from the specified project directory.
sys.path.insert(0, "/p/scratch/westai0043/CNN_HL_tobias/")

#from sklearn.metrics import mean_absolute_error
from dataloader import create_dataloaders
import dataloader
from lightning.pytorch.utilities import rank_zero_only

from src.models.augmentation_utils import _apply_noise_and_mask_on_device
#*(If the import fails due to your specific project structure, you may need to adjust the path, e.g., `from models.lightning_resnet import ...` or using `sys.path.insert` if necessary.)*


@rank_zero_only
def print_rank0(*args, **kwargs):
    """A wrapper for the print function that only executes on the main process (rank 0)
    in a distributed training setup. This prevents duplicate print statements from every GPU."""
    print(*args, **kwargs)
print_rank0(">>> Using dataloader from:", dataloader.__file__)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
import pandas as pd

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
                 use_attention_heads=False, attention_latent_dim=128, use_mdn = True, covariance_type: str = "diagonal", num_mixtures: int = 1):
        super().__init__()
        # Store configuration parameters.
        self.use_mdn = use_mdn
        self.covariance_type = covariance_type
        self.num_mixtures = num_mixtures
        self.target_params_list = target_params_list if target_params_list is not None else [f"p_{i}" for i in range(n_outputs)]

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
                fc_hidden_dim, n_outputs, 
                num_components=self.num_mixtures,
                init_diag_scale=1.0,
                per_param_scales = {
                    0: 2.0,   # D
                    1: 3.0,   # L - CRITICAL: needs larger initial scale
                    2: 1.0,   # ro
                    3: 1.0,   # rr
                    4: 1.0,   # p
                    5: 1.0,   # Tlow
                    6: 2.0,   # NCH3CN
                }
            )

        # If using a mixture of Gaussians (K>1), create a head to predict the mixture weights (pi).
        if self.use_mdn and self.num_mixtures > 1:
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
                # For a mixture, also predict and return the mixture weights.
                pi_logits = self.pi_head(shared_features)
                L = cov["L"]
                return {"pi_logits": pi_logits, "mu": mu, "L": L}

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
                pi_logits = self.pi_head(shared_features)
                return {"pi_logits": pi_logits, "mu": mu, "sigma": sigma}
        
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
        'num_mixtures': 1
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


# ----------------- Checkpointing, Evaluation, and Training Script -----------------
# The functions below this point (save_checkpoint, load_checkpoint, plot_residual_vs_sigma,
# evaluate_calibration, train, test, etc.) are part of a standalone training script,
# not part of the model definition itself. They are used to run the training process
# when this file is executed directly.

CHECKPOINT_FILENAME = f"model_checkpoint.pth"

def save_checkpoint(epoch, model, optimizer, scaler, train_losses, test_losses, current_loss, checkpoint_dir="checkpoints"):
    """Saves a training checkpoint including model weights, optimizer state, and epoch."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)

    # If model is wrapped in DataParallel, save the underlying module's state_dict
    model_state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    current_loss = test_losses[-1]
    print(f"Saving checkpoint to {checkpoint_path} (End of Epoch {epoch+1}, Loss: {current_loss:.4f})")
    checkpoint = {
        'epoch': epoch + 1,  # Save as the epoch number *completed*
        'model_state_dict': model_state_dict_to_save,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss_checkpointed': current_loss,
        'train_losses': train_losses,
        'test_losses': test_losses,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved successfully.")
    
def load_checkpoint(model, optimizer, scaler, device, checkpoint_dir="checkpoints"):
    """Loads model checkpoint if it exists.
    Returns: start_epoch (0-indexed), best_loss_checkpointed
    """
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)
    start_epoch = 0 # 0-indexed epoch to start training from
    best_loss_checkpointed = float('inf')
    checkpoint = None  # Initialize checkpoint to None

    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            saved_model_state_dict = checkpoint['model_state_dict']
            
            # Handle DataParallel differences
            current_is_parallel = isinstance(model, nn.DataParallel)
            
            if current_is_parallel:
                # If current model is DataParallel, saved keys might or might not have 'module.'
                # If saved keys don't have 'module.', add it.
                if not all(key.startswith('module.') for key in saved_model_state_dict.keys()):
                    print("Saved model was not DataParallel, current is. Adding 'module.' prefix.")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in saved_model_state_dict.items():
                        name = 'module.' + k
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(saved_model_state_dict) # Both parallel, direct load
            else:
                # If current model is NOT DataParallel, saved keys might have 'module.'
                # If saved keys have 'module.', strip it.
                if all(key.startswith('module.') for key in saved_model_state_dict.keys()):
                    print("Saved model was DataParallel, current is not. Stripping 'module.' prefix.")
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in saved_model_state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    model.load_state_dict(new_state_dict)
                else:
                     model.load_state_dict(saved_model_state_dict) # Both not parallel, direct load


            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] # This is the epoch number training should *start* from (0-indexed)
                                              # e.g., if saved after epoch 0, checkpoint['epoch'] is 1, so start_epoch is 1.
                                              # If saved after epoch N (0-indexed), checkpoint['epoch'] is N+1. Loop will be range(N+1, num_epochs)
            #start_epoch = 0 # Hard coded for the moment to avoid plotting errors
            best_loss_checkpointed = checkpoint.get('best_loss_checkpointed', float('inf'))

            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch +1} (0-indexed: {start_epoch}).")
            print(f"Best loss recorded in loaded checkpoint: {best_loss_checkpointed:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_loss_checkpointed = float('inf')
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    
    train_losses = []
    test_losses = []
    if checkpoint is not None:
        train_losses = checkpoint['train_losses'] if 'train_losses' in checkpoint else []
        test_losses = checkpoint['test_losses'] if 'test_losses' in checkpoint else []

    return start_epoch, best_loss_checkpointed, train_losses, test_losses


def plot_residual_vs_sigma(mu, sigma, y_true, param_names, job_id):
    """Generates plots to analyze the quality of uncertainty predictions."""
    residual = mu - y_true
    coverage_ratios = []

    for i, name in enumerate(param_names):
        plt.figure(figsize=(6, 5))
        plt.scatter(sigma[:, i], residual[:, i], alpha=0.3, s=10)
        plt.xlabel('Predicted σ')
        plt.ylabel('Residual (μ - y)')
        plt.title(f'Residual vs σ: {name}')
        plt.grid(True)
        plt.tight_layout()
        if folder_name is None:
            plt.savefig(f"ResNetPlots/{job_id}/Residual_vs_Sigma_{name}.png")
        else:
            plt.savefig(f"{folder_name}/ResNetPlots/{job_id}/Residual_vs_Sigma_{name}.png")
        plt.close()

        within_1sigma = ((mu[:, i] > y_true[:, i] - sigma[:, i]) & (mu[:, i] < y_true[:, i] + sigma[:, i])).float().mean().item()
        coverage_ratios.append((name, within_1sigma))

    print("\nCoverage within ±1σ:")
    for name, ratio in coverage_ratios:
        print(f"  {name}: {ratio * 100:.2f}%")

def evaluate_calibration(mu, sigma, y_true, param_names, job_id, epoch):
    """
    Evaluates how well-calibrated the model's uncertainty predictions are
    by comparing the expected vs. actual fraction of true values that fall
    within N-sigma intervals.
    """
    """
    Evaluates calibration, finds an optimal scaling factor `c` for each parameter's
    uncertainty, and plots both the original and calibrated curves.
    """
    print("Evaluating and fitting calibration curves...")
    # Theoretical fraction of data that should be within z sigma levels
    levels = np.array([1, 2, 3])
    theoretical_coverage = np.array([stats.norm.cdf(z) - stats.norm.cdf(-z) for z in levels])

    calibration_factors = {}
    
    plt.figure(figsize=(10, 8))
    
    # --- Objective Function for Optimization ---
    def calibration_loss(c, sigma_param, y_true_param):
        # Calculate empirical coverage with the calibrated sigma
        abs_z_calibrated = torch.abs((y_true_param - mu_param) / (c * sigma_param))
        empirical_coverage = np.array([(abs_z_calibrated < z).float().mean().item() for z in levels])
        # Return the sum of squared errors
        return np.sum((empirical_coverage - theoretical_coverage)**2)

    # --- Find Optimal 'c' for Each Parameter ---
    for i, param in enumerate(param_names):
        mu_param = mu[:, i]
        sigma_param = sigma[:, i]
        y_true_param = y_true[:, i]

        # Use scipy's bounded scalar optimizer to find the best `c`
        result = opt.minimize_scalar(
            calibration_loss,
            args=(sigma_param, y_true_param),
            bounds=(0.01, 1.0), # Search for c in a reasonable range
            method='bounded'
        )
        c_optimal = result.x
        calibration_factors[param] = c_optimal

        # --- Plotting ---
        # Calculate original empirical coverage
        abs_z_original = torch.abs((y_true_param - mu_param) / sigma_param)
        original_empirical = [(abs_z_original < z).float().mean().item() for z in levels]

        # Calculate calibrated empirical coverage
        abs_z_calibrated = torch.abs((y_true_param - mu_param) / (c_optimal * sigma_param))
        calibrated_empirical = [(abs_z_calibrated < z).float().mean().item() for z in levels]

        # Plot the lines
        p = plt.plot(levels, original_empirical, marker='o', ls=':', label=f'{param} (Original)')[0]
        plt.plot(levels, calibrated_empirical, marker='x', ls='-', color=p.get_color(), label=f'{param} (Calibrated, c={c_optimal:.2f})')

    plt.plot(levels, theoretical_coverage, 'k--', label='Expected (Gaussian)')
    plt.xlabel("Sigma Level")
    plt.ylabel("Fraction within Interval")
    plt.title(f"Epoch {epoch+1} - Calibration Curve (Original vs. Calibrated)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
    
    # Save the plot
    if folder_name is None:
        cal_dir = f"ResNetPlots/{job_id}/Calibration/"
    else:
        cal_dir = f"{folder_name}/ResNetPlots/{job_id}/Calibration/"
    os.makedirs(cal_dir, exist_ok=True)
    plt.savefig(os.path.join(cal_dir, f"E{epoch+1}_Calibration_Curve.png"))
    plt.close()
    
    print("Calibration factors found:", {k: round(v, 3) for k, v in calibration_factors.items()})
    return calibration_factors
    
# Plotting the first 10 average spectra from the train loader

def plot_diagnostic_spectra(dataset, num_samples, job_id, freq_axis, output_dir, noise_cfg: dict):
    """
    Loads samples, applies the same on-GPU noise augmentation used in training,
    and saves plots of the noised spectra for visual inspection.
    """
    print(f"Generating {num_samples} noised diagnostic plots...")

    # --- Step 1: Collect and noise samples ---
    collected_samples = []  # List of tuples: (noised_cube, noise_rms_added)
    
    # First, determine the device we will be working on.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Now, create the generator and explicitly place it on that same device.
    gen = torch.Generator(device=device)

    for idx in range(min(num_samples, len(dataset))):
        # Get the original, clean data from the dataset
        x_clean, _ = dataset[idx]
        
        if x_clean.ndim == 4 and x_clean.shape[0] == 1:
            x_clean = x_clean.squeeze(0)

        # NEW: The noise function expects a batch. Add a temporary batch dimension.
        x_batch = x_clean.unsqueeze(0)

        # NEW: Apply the same noise function used in the training loop
        # We assume the data is on CPU and move it to GPU if available for the noise function
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_noised_batch = _apply_noise_and_mask_on_device(
            x_batch.to(device),
            gen=gen,
            **noise_cfg
        ).cpu() # Move the result back to CPU for plotting

        # NEW: Get the noised tensor and remove the temporary batch dimension
        x_noised = x_noised_batch.squeeze(0)

        # NEW: Calculate the RMS of the noise that was actually added
        added_noise = x_noised - x_clean
        noise_rms = torch.std(added_noise).item()

        collected_samples.append((x_noised, noise_rms))

    if not collected_samples:
        print("Could not retrieve any samples for diagnostic plot.")
        return

    # --- Step 2: Loop through each noised sample and create a plot ---
    plot_dir = f"{output_dir}/Diagnostic_Plots/{job_id}/"
    os.makedirs(plot_dir, exist_ok=True)

    print("Fetching noise data worked, producing spectra...")

    for i, (spectrum_cube, noise_rms) in enumerate(collected_samples):
        height, width = spectrum_cube.shape[1], spectrum_cube.shape[2]
        spectrum_1d = spectrum_cube[:, height // 2, width // 2].cpu().numpy()

        # --- SNR CALCULATION (now using the calculated noise RMS) ---
        snr_text = "SNR: N/A"
        if noise_rms > 0 and freq_axis is not None: # MODIFIED: Check if noise_rms is positive
            C_KMS = 299792.458
            FREQ_K3_GHZ = 220.7090170
            VEL_WINDOW_KMS = 10
            freq_window_ghz = FREQ_K3_GHZ * (VEL_WINDOW_KMS / C_KMS)
            k3_freq_min, k3_freq_max = FREQ_K3_GHZ - freq_window_ghz, FREQ_K3_GHZ + freq_window_ghz
            k3_mask = (freq_axis >= k3_freq_min) & (freq_axis <= k3_freq_max)
            if np.any(k3_mask):
                # Calculate signal peak on the NOISY spectrum
                signal_peak = spectrum_cube[k3_mask, :, :].max().item()
                snr = signal_peak / noise_rms
                # RESTORED: This text is now meaningful
                snr_text = f"SNR (Peak/RMS): {snr:.2f}\nRMS added: {noise_rms:.4f}"
            else:
                snr_text = "SNR: k=3 line not in range"

        # --- Plotting ---
        plt.figure(figsize=(12, 6))
        x_axis = freq_axis if freq_axis is not None else np.arange(len(spectrum_1d))
        xlabel = "Frequency (GHz)" if freq_axis is not None else "Channel Number"

        plt.plot(x_axis, spectrum_1d, label="Noised Spectrum") # MODIFIED: Label clarifies this is noised

        if freq_axis is not None:
            FREQ_REST_13CO_GHZ = 220.4039
            VEL_WIDTH_KMS = 40
            freq_width_ghz = FREQ_REST_13CO_GHZ * (VEL_WIDTH_KMS / C_KMS)
            freq_min, freq_max = FREQ_REST_13CO_GHZ - freq_width_ghz, FREQ_REST_13CO_GHZ + freq_width_ghz
            plt.axvspan(freq_min, freq_max, color='red', alpha=0.2, label='Masked ¹³CO Region')
        
        plt.legend() # MODIFIED: Added legend to show labels

        # RESTORED: The SNR text box is now useful again
        plt.text(0.02, 0.95, snr_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.title(f"Diagnostic Spectrum with Training Noise - Sample {i+1}") # MODIFIED: Title is more descriptive
        plt.xlabel(xlabel)
        plt.ylabel("Central Pixel Intensity")
        plt.grid(True)
        plt.tight_layout()

        plot_filename = os.path.join(plot_dir, f"diagnostic_spectrum_sample_{i+1}.png")
        plt.savefig(plot_filename)
        plt.close()

    print(f"{len(collected_samples)} diagnostic plots saved to {plot_dir}")


def train(model, train_loader, optimizer, criterion, device, scaler, use_mdn, enable_profiling=False, profile_steps=10):
    model.train()
    running_loss = 0.0

    if enable_profiling:
        prof = profile(
            schedule=schedule(wait=2, warmup=2, active=profile_steps),
            on_trace_ready=tensorboard_trace_handler('./logdir'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        )
        prof.start()
    else:
        prof = None

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if not torch.isfinite(data).all():
            # Flatten input and find which samples are bad
            nan_mask = ~torch.isfinite(data.view(data.shape[0], -1).sum(dim=1))
            bad_indices = nan_mask.nonzero(as_tuple=True)[0]
            
            print(f"\n NaNs in input tensor at batch {batch_idx}, sample indices: {bad_indices.tolist()}")
            print("Corresponding target parameter values for bad inputs:")

            for idx in bad_indices.tolist():
                print(f"Sample {idx}: {target[idx].cpu().numpy()}")

            # Optional: Save for inspection
            torch.save({"data": data.cpu(), "target": target.cpu()}, f"nan_input_batch{batch_idx}.pt")
            
            raise ValueError("NaN input encountered — likely due to bad parameter combination")

        if torch.isnan(data).any():
            print("NaN in input data")
            raise ValueError("NaN in input")

        if data.shape[2] == 0:
            print("D = 0 in input")
            raise ValueError("Empty spectral dimension")

        if not torch.isfinite(target).all():
            print("Corrupted target:", target)
            raise ValueError("Target contains NaNs")

        # Debug input stats
        #print("Input stats: min =", data.min(), ", max =", data.max())

        # Flag for noisy labels during training
        use_noisy_labels = False
        # Label noise injection if flag is True
        if use_noisy_labels:
            #per_target_std = torch.tensor([0.1, 0.1, 0.1, 0.4], device=target.device)
            per_target_std = 0.1 # Example: constant std for all targets
            if per_target_std.shape[0] != target.shape[1] and per_target_std.shape != ():
                raise ValueError(f"Shape mismatch! per_target_std: {per_target_std.shape}, Labels: {target.shape}")
            noise = torch.randn_like(target) * per_target_std
        else:
            noise = torch.zeros_like(target)
        labels_noisy = target + noise

        optimizer.zero_grad()
        with autocast(device_type=device.type):
            output = model(data)
            # Check for empty or invalid outputs BEFORE loss
            if use_mdn is False:
                if output.numel() == 0:
                    raise RuntimeError(f"Empty output tensor from model! Shape: {output.shape}")
                if target.numel() == 0:
                    raise RuntimeError(f"Empty target tensor! Shape: {target.shape}")
                if output.shape != target.shape:
                    raise RuntimeError(f"Output-target shape mismatch: {output.shape} vs {target.shape}")
                if not torch.isfinite(output).all():
                    raise ValueError(f"Non-finite values in output: {output}")
                if not torch.isfinite(target).all():
                    raise ValueError(f"Non-finite values in target: {target}")

            if use_mdn:
                # Reconstruct the dictionary for easy, name-based access
                output_dict = {}
                for i, param_name in enumerate(args.model_params):
                    # Each parameter corresponds to two tensors in the flat tuple
                    mu = output[i * 2]
                    sigma = output[i * 2 + 1]
                    output_dict[param_name] = {'mu': mu, 'sigma': sigma} # Can use a dict or a tuple here
                
                # Now, pass the perfectly structured dictionary to your loss function
                loss = gaussian_nll_loss_dict(output_dict, target, std_weights=train_loader.dataset.dataset.dataset.scaler_stds)
            else:
                # The non-MDN case
                output = output # In this case, it's just a single tensor
                loss = criterion(output, target)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if prof is not None:
            prof.step()
            if batch_idx >= (2 + 2 + profile_steps):  # wait + warmup + active
                break  # Early exit after profiling steps

    if prof is not None:
        prof.stop()

    return running_loss / len(train_loader)


def test(model, test_loader, criterion, device, epoch, use_mdn, original_dataset, calibration_factors, folder_name=None):
    model.eval()
    test_loss = 0.0 # Total loss across all batches
    all_preds, all_targets, all_sigmas = [], [], []
    loss_per_param = {param: [] for param in args.model_params}

    print(f"scaler_stds: {test_loader.dataset.dataset.dataset.scaler_stds}")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast(device_type=device.type):
                # This is the raw output from the model (a tuple if mdn, a tensor otherwise)
                raw_output = model(data)

                if use_mdn:
                    # --- Step 1: Reconstruct the dictionary and calculate TOTAL loss ---
                    output_dict = {}
                    mu_list, sigma_list = [], []
                    for i, param_name in enumerate(args.model_params):
                        mu = raw_output[i * 2]
                        sigma = raw_output[i * 2 + 1]
                        output_dict[param_name] = {'mu': mu, 'sigma': sigma}
                        mu_list.append(mu)
                        sigma_list.append(sigma)
                    
                    # Create unified tensors for predictions and their uncertainties
                    predictions = torch.cat(mu_list, dim=1)
                    sigmas = torch.cat(sigma_list, dim=1)
                    
                    # Calculate the single, total loss value for this batch
                    loss = gaussian_nll_loss_dict(output_dict, target, std_weights=test_loader.dataset.dataset.dataset.scaler_stds)
                else:
                    # Standard case: output is already the predictions tensor
                    predictions = raw_output
                    sigmas = None # No sigmas in this case
                    loss = criterion(predictions, target)

            # --- Step 2: Accumulate total loss and collect batch predictions ---
            test_loss += loss.item()

            all_preds.append(predictions.cpu())
            all_targets.append(target.cpu())
            if use_mdn:
                all_sigmas.append(sigmas.cpu())

            # --- Step 3: Calculate PER-PARAMETER loss using the correct function ---
            for i, param in enumerate(args.model_params):
                if use_mdn:
                    # Use the correct, underlying PyTorch function for a single parameter
                    param_loss = F.gaussian_nll_loss(predictions[:, i], target[:, i], sigmas[:, i] ** 2, full=True)
                else:
                    # Standard case
                    param_loss = criterion(predictions[:, i], target[:, i])
                loss_per_param[param].append(param_loss.item())

    # --- Step 4: Aggregate results and plot ---
    avg_test_loss = test_loss / len(test_loader)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_sigmas = torch.cat(all_sigmas).numpy() if use_mdn else None

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print("Loss per parameter:")
    for param, losses in loss_per_param.items():
        avg_loss = np.mean(losses)
        print(f"  {param}: {avg_loss:.4f}")


    # Convert back to original scale for analysis
    if use_mdn:
        outputs_original, sigmas_original = original_dataset.inverse_transform_labels_with_uncertainty(all_preds, all_sigmas)
    else:
        outputs_original = original_dataset.inverse_transform_labels(all_preds)
    targets_original = original_dataset.inverse_transform_labels(all_targets)

    # Compute MAE
    #mae = mean_absolute_error(all_targets, all_preds)
    mae = torch.mean(torch.abs(outputs_original - targets_original), dim=0).numpy()  # shape: (n_outputs,)
    # Print mean absolute error for each parameter with parameter name in scientific notation
    print("Mean Absolute Error (MAE) for each parameter in original scale:")
    for i, param_name in enumerate(args.model_params):
        print(f"  {param_name}: {mae[i]:.4e}")

    print("Generating predicted vs. true plots...")
    if folder_name is None:
        PNG_DIR = f"./Epoch_Plots/{args.job_id}/"
        os.makedirs(PNG_DIR, exist_ok=True)
    else:
        PNG_DIR = f"{folder_name}/Epoch_Plots/{args.job_id}/"
        os.makedirs(PNG_DIR, exist_ok=True)

    # Convert tensors to numpy arrays
    all_outputs_original_np = outputs_original.numpy()
    if use_mdn:
        all_sigmas_original_np = sigmas_original.numpy()
    all_labels_original_np = targets_original.numpy()

    for i, current_param_name in enumerate(args.model_params):
        
        y_true = all_labels_original_np[:, i]
        
        y_pred = all_outputs_original_np[:, i]
        if use_mdn:
            y_err = all_sigmas_original_np[:, i]
            # Get the calibation factor for this parameter, default to 1.0 if not calibrated
            cal_factor = calibration_factors.get(current_param_name, 1.0)
            y_err *= cal_factor  # Apply calibration factor to the uncertainty

        plt.figure(figsize=(7, 7))  # Square aspect ratio

        # Log-log plot for selected parameters
        if current_param_name in args.log_scale_params:
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"Epoch {epoch+1} - {current_param_name} (Log-Log Scale)")

            # Filter valid data
            mask = (y_true > 0) & (y_pred > 0)

            if np.any(mask):
                if use_mdn:
                    plt.errorbar(y_true[mask], y_pred[mask], yerr=y_err[mask], fmt='o', alpha=0.3, markersize=4, capsize=2)
                    minval = min(y_true[mask].min(), y_pred[mask].min())
                    maxval = max(y_true[mask].max(), y_pred[mask].max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
                else:
                    plt.scatter(y_true[mask], y_pred[mask], alpha=0.3, s=10)
                    minval = min(y_true[mask].min(), y_pred[mask].min())
                    maxval = max(y_true[mask].max(), y_pred[mask].max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
                plt.xlim(minval, maxval)
                plt.ylim(minval, maxval)
            else:
                print(f"Warning: No positive data for log-log plot of {current_param_name}")
                plt.text(0.5, 0.5, 'No valid data', transform=plt.gca().transAxes,
                        ha='center', va='center')
                plt.xlim(1e-9, 1)
                plt.ylim(1e-9, 1)

        else:
            # Linear plot
            plt.title(f"Epoch {epoch+1} - {current_param_name} (Linear Scale)")
            if use_mdn:
                    plt.errorbar(y_true, y_pred, yerr=y_err, fmt='o', alpha=0.3, markersize=4, capsize=2)
                    minval = min(y_true.min(), y_pred.min())
                    maxval = max(y_true.max(), y_pred.max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
            else:
                    plt.scatter(y_true, y_pred, alpha=0.3, s=10)
                    minval = min(y_true.min(), y_pred.min())
                    maxval = max(y_true.max(), y_pred.max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')

        plt.xlabel(f"True {current_param_name}")
        plt.ylabel(f"Predicted {current_param_name}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.join(PNG_DIR, f"E{epoch+1}_Test_{current_param_name}.png")
        print("Saving plot:", plot_filename)
        plt.savefig(plot_filename, dpi=150)
        plt.close()


        # --- DERIVED RADIUS PLOT ---
        # Check if we have the necessary parameters to derive the radius
        if use_mdn and "plummer_shape" in args.model_params and "p" in args.model_params and "rr" in args.model_params and current_param_name == "plummer_shape":
            print("Generating derived radius plot...")
            try:
                # Get the indices of the required parameters
                ps_idx = args.model_params.index("plummer_shape")
                p_idx = args.model_params.index("p")
                rr_idx = args.model_params.index("rr")

                # --- 1. Get predictions and uncertainties from the original model outputs ---
                # These are PyTorch tensors, which is what we want for calculations
                y_pred = outputs_original[:, ps_idx]
                sigma_y = sigmas_original[:, ps_idx]
                
                p_pred = outputs_original[:, p_idx]
                sigma_p = sigmas_original[:, p_idx]

                # Get the true radius for the x-axis
                rr_true = targets_original[:, rr_idx]

                # --- 2. Calculate the derived radius and its uncertainty ---
                
                # Avoid division by zero
                # We use torch.where to handle p_pred being close to zero safely
                log10_rr_derived = torch.where(torch.abs(p_pred) > 1e-9, y_pred / p_pred, torch.zeros_like(p_pred))
                
                # Propagate uncertainty for the division z = y / p
                # (sigma_z / z)^2 = (sigma_y / y)^2 + (sigma_p / p)^2
                # We need to handle cases where y or p might be zero
                relative_err_y_sq = torch.where(torch.abs(y_pred) > 1e-9, (sigma_y / y_pred)**2, torch.zeros_like(y_pred))
                relative_err_p_sq = torch.where(torch.abs(p_pred) > 1e-9, (sigma_p / p_pred)**2, torch.zeros_like(p_pred))
                
                sigma_log10_rr = torch.abs(log10_rr_derived) * torch.sqrt(relative_err_y_sq + relative_err_p_sq)
                
                # Final derived radius
                rr_derived = torch.pow(10.0, log10_rr_derived)
                
                # Propagate uncertainty for the exponentiation rr = 10^z
                # sigma_rr = |rr * ln(10)| * sigma_z
                sigma_rr_derived = torch.abs(rr_derived * np.log(10)) * sigma_log10_rr

                # --- 3. Create the plot ---
                plt.figure(figsize=(7, 7))
                #plt.xscale('log')
                #plt.yscale('log')
                plt.title(f"Epoch {epoch+1} - Derived radius (from plummer_shape & prho)")

                # Convert to numpy for plotting
                rr_true_np = rr_true.numpy()
                rr_derived_np = rr_derived.numpy()
                sigma_rr_derived_np = sigma_rr_derived.numpy()

                # Filter for valid data points to plot
                mask = (rr_true_np > 0) & (rr_derived_np > 0)
                if np.any(mask):
                    plt.errorbar(
                        rr_true_np[mask], 
                        rr_derived_np[mask], 
                        yerr=sigma_rr_derived_np[mask], 
                        fmt='o', alpha=0.3, markersize=4, capsize=2, label='Derived rr'
                    )
                    
                    # Plot the y=x line
                    minval = min(rr_true_np[mask].min(), rr_derived_np[mask].min())
                    maxval = max(rr_true_np[mask].max(), rr_derived_np[mask].max())
                    plt.plot([minval, maxval], [minval, maxval], 'r--', label='y = x')
                    plt.xlim(minval, maxval)
                    plt.ylim(minval, maxval)

                plt.xlabel("True radius")
                plt.ylabel("Derived radius")
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.legend()
                plt.tight_layout()

                # Save the plot
                plot_filename = os.path.join(PNG_DIR, f"E{epoch+1}_Test_Derived_radius.png")
                plt.savefig(plot_filename, dpi=150)
                plt.close()
                
                print("Derived radius plot saved successfully.")
            
            except Exception as e:
                print(f"Could not generate derived radius plot. Error: {e}")

    return test_loss / len(test_loader), all_preds, all_targets, loss_per_param, all_sigmas


def save_training_plots(train_losses, test_losses, loss_per_param_history, labels, job_id, epoch, max_epochs, update_interval=10, folder_name=None):
    if (epoch + 1) % update_interval == 0 or (epoch + 1) == max_epochs:
        # Training and Validation loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if folder_name is None:
            plt.savefig(f"ResNetPlots/{job_id}/Train_vs_Validation_E{epoch+1}.png")
        else:
            plt.savefig(f"{folder_name}/ResNetPlots/{job_id}/Train_vs_Validation_E{epoch+1}.png")
        plt.close()
        print(f"Saved Train vs Validation loss plot for epoch {epoch+1}")

        # Per-parameter loss
        loss_arr = np.array(loss_per_param_history)  # shape: (epochs, n_params)
        plt.figure(figsize=(10, 6))
        for i, param in enumerate(labels):
            plt.plot(loss_arr[:, i], label=param)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss per Parameter")
        plt.title("Per-Parameter Validation Loss Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if folder_name is None:
            plt.savefig(f"ResNetPlots/{job_id}/val_loss_per_param_E{epoch+1}.png")
        else:
            plt.savefig(f"{folder_name}/ResNetPlots/{job_id}/val_loss_per_param_E{epoch+1}.png")

        plt.close()
        print(f"Saved per-parameter loss plot for epoch {epoch+1}")

# for early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

if __name__ == "__main__":
    # This block is executed when the script is run directly.
    # It handles argument parsing, data loading, model creation, and the main training loop.
    
    import argparse
    parser = argparse.ArgumentParser()
    # --- Defines all command-line arguments for the training script ---
    # e.g., data directories, batch size, model depth, covariance_type, etc.
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--original-file-list", type=str, default=None)
    parser.add_argument("--scaling-params-path", type=str, default=None)
    parser.add_argument("--wavelength-stride", type=int, default=1)
    #parser.add_argument("--load-preprocessed", type=bool, default=False)
    parser.add_argument('--load-preprocessed', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-local-nvme", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--job_id", type=str, default=None)
    parser.add_argument("--load_id", type=str, default=None)
    parser.add_argument("--model-depth", type=int, default=10)
    parser.add_argument("--model_params", type=str, nargs='+', default=['Dens', 'Lum', 'radius', 'prho'], help="Model parameters to predict")
    parser.add_argument("--log-scale-params", type=str, nargs='+', default=['Dens','Lum'], help="Log scale parameters for the model")
    parser.add_argument("--use_attention_heads", type=bool, default=False, help="Use attention heads in the model")
    parser.add_argument("--use_mdn", type=bool, default=True, help="Use MDN output heads for uncertainty estimation")
    parser.add_argument("--data-subset-fraction", type=float, default=1.0, help="Fraction of the data to use for this run (e.g., 0.1 for 10%).")
    parser.add_argument("--folder_name", type=str, default=None, help="Folder name for saving plots and checkpoints. If None, uses job_id.")
    parser.add_argument("--add-noise-level", type=float, default=0.0, help="Absolute RMS of noise to add to spectra (e.g., 0.01 K).")
    parser.add_argument("--snr-threshold", type=float, default=5.0, help="Minimum SNR required after adding noise.")
    parser.add_argument("--mask-13co", type=lambda x: x.lower() == 'true', default=True, help="Mask the 13CO contaminated region.")
    parser.add_argument("--use-cauchy-noise", type=lambda x: x.lower() == 'true', default=True, help="Enable variable noise augmentation using a Cauchy distribution.")
    parser.add_argument("--cauchy-mu", type=float, default=0.003, help="Location parameter for Cauchy noise RMS in Jy/beam. Default corresponds to 0.3K.")
    parser.add_argument("--cauchy-sigma", type=float, default=0.0032, help="Scale parameter for Cauchy noise RMS in Jy/beam. Default corresponds to 0.32K.")
    parser.add_argument("--cauchy-threshold", type=float, default=0.07, help="Maximum allowed noise RMS in Jy/beam. Default corresponds to 7K.")
    parser.add_argument("--covariance_type", type=str, default="full", help="Type of covariance for MDN: diagonal or full")
    parser.add_argument("--num-mixtures", type=int, default=1, help="Number of mixtures for MDN")
    args = parser.parse_args()

    # --- Data Loading ---
    print(f"Saving plots and checkpoints in folder: {args.folder_name}")
    print(f"Creating Dataloaders with the following model parameters:")
    print(f"Model parameters: {args.model_params}")
    print(f"DEBUG: load_preprocessed = {args.load_preprocessed}")

    train_loader, test_loader, dataset = create_dataloaders(
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
        log_scale_params = args.log_scale_params,
        data_subset_fraction=args.data_subset_fraction,
        use_cauchy_noise=args.use_cauchy_noise,
        cauchy_mu=args.cauchy_mu,
        cauchy_sigma=args.cauchy_sigma,
        cauchy_threshold=args.cauchy_threshold,
        add_noise_level=args.add_noise_level,
        snr_threshold=args.snr_threshold,
        mask_13co=args.mask_13co,
    )
    
    # Debugging: Check dataset type of trainloader
    print(f"DEBUG: Dataset class = {type(train_loader.dataset.dataset)}")

    # Output train/test dataset information
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Generate diagnostic plots for a few samples
    print("Retrieving frequency axis for plotting...")
    freq_axis = dataset.get_frequency_axis()

    # Call the diagnostic function, passing the axis as an argument
    #plot_diagnostic_spectra(train_loader, num_samples=10, job_id=args.job_id, freq_axis=freq_axis)

    # Create folder for plots
    os.makedirs(f"plots", exist_ok=True)

    # Specify checkpointing directory
    checkpoint_save_dir = f"checkpoints/{args.job_id}/"
    checkpoint_load_dir = f"checkpoints/{args.load_id}"

    #global TARGET_PARAMETERS
    TARGET_PARAMETERS = args.model_params

    max_epochs = args.num_epochs

    # Train with AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(device)}")

    # Debugging: Check input shape
    #sample_image, _ = next(iter(train_loader))
    #print(f"DEBUG: input shape = {sample_image.shape}") 
    #num_outputs = len(next(iter(train_loader))[1][0])
    num_outputs = len(TARGET_PARAMETERS)

    # Print model information
    print(f"Model parameters: {TARGET_PARAMETERS}")
    print(f"Log scale parameters: {args.log_scale_params}")
    print(f"Model depth: {args.model_depth}")
    model_depth = args.model_depth
    #model = generate_model(model_depth=model_depth, n_outputs=num_outputs)
    # Set config_name depending on depth
    if model_depth == 10: 
        # Get user input on number of layers
        print("Enter number of layers (3 or 4) for ResNet10: ")
        input_layers = input().strip()
        print(f"Chosen ResNet10 layers: {input_layers}")
        if input_layers == '3':
            config_name = "resnet10_3layers"
        elif input_layers == '4':
            config_name = "resnet10_2d_equivalent"
    elif model_depth == 18:
        config_name = "resnet18_2d_equivalent"
    elif model_depth == 34:
        config_name = "resnet34_2d"
    elif model_depth == 50:
        config_name = "resnet50_2d"
    else:
        raise ValueError(f"Unsupported model depth: {model_depth}. Supported depths are 10 or 18.")
    
    use_attention_heads = False
    use_mdn = args.use_mdn  # Set to True if you want to use MDN heads, False otherwise
    #use_attention_heads = args.use_attention_heads  # Set to True if you want to use attention heads, False otherwise
    print(f'Using parameter heads with an attention mechanism, use_attention_heads={use_attention_heads}') if use_attention_heads else print(f'Using simple feature heads, use_attention_heads={use_attention_heads}')
    model = generate_2d_model(config_name=config_name, use_batchnorm=False, TARGET_PARAMETERS=TARGET_PARAMETERS, n_outputs=num_outputs, use_attention_heads=use_attention_heads, attention_latent_dim=128, use_mdn=use_mdn, covariance_type=args.covariance_type, num_mixtures=args.num_mixtures)

    # Trying to use multi GPUs 
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Print model structure
    print(f"Config name: {config_name}")
    print(model)
    

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scaler = GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)

    # Print criterion, optimizer, scaler and scheduler information
    if use_mdn:
        print("Criterion: Gaussian NLL Loss with MDN heads")
    else:
        print(f"Criterion: {criterion}")
    print(f"Optimizer: {optimizer}")
    print(f"Scaler: {scaler}")
    print(f"Scheduler: {scheduler}")

    # Print batch size and number of workers
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")

    train_losses = []
    test_losses = []
    loss_per_param_history = []
    num_epochs = args.num_epochs
    checkpoint_interval = 5

    labels = args.model_params#, "NCH3CN", "incl", "phi"]
    # Global dictionary to store the latest calibration factors
    CALIBRATION_FACTORS = {}    
    #early stopper
    #early_stopper = EarlyStopping(patience=50, min_delta=1e-8)

    # Create directory for plots
    if args.folder_name is not None:
        os.makedirs(f"{args.folder_name}/ResNetPlots/{args.job_id}", exist_ok=True)
    else:
        os.makedirs(f"ResNetPlots/{args.job_id}", exist_ok=True)

    # Load checkpoint if exists
    # `best_loss_at_last_checkpoint_save` stores the loss value OF THE CHECKPOINT CURRENTLY ON DISK
    start_epoch, best_loss_at_last_checkpoint_save, train_losses, test_losses = load_checkpoint(model, optimizer, scaler, device, checkpoint_load_dir)

    max_epochs = start_epoch + args.num_epochs
    # If loading a checkpoint, set the starting epoch for training
    for epoch in range(start_epoch, max_epochs):
        print(f'\nEpoch {epoch+1}/{max_epochs}')
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device, scaler, enable_profiling=False, use_mdn = use_mdn)
        test_loss, preds, targets, loss_per_param, sigmas = test(model, test_loader, criterion, device, epoch, use_mdn=use_mdn, original_dataset=dataset, calibration_factors=CALIBRATION_FACTORS, folder_name=args.folder_name)
        # Scheduler step based on validation loss in case of ReduceLROnPlateau
        scheduler.step(test_loss) if isinstance(scheduler, ReduceLROnPlateau) else  scheduler.step()
        
        #early_stopper.step(test_loss)
        end_time = time.time()

        print(f"Epoch time: {end_time - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {test_loss:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.2e}")

        # Denormalize predictions and targets
        #label_min = np.load("Parameters/label_min.npy")
        #label_max = np.load("Parameters/label_max.npy")
        #label_min = np.load("/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data/labels_100/label_min.npy")
        #label_max = np.load("/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data/labels_100/label_max.npy")
        
        #denorm_preds = preds * (label_max - label_min) + label_min
        #denorm_targets = targets * (label_max - label_min) + label_min
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        loss_per_param_history.append([np.mean(losses) for losses in loss_per_param.values()])

        # Save live plots every N epochs
        save_training_plots(
            train_losses=np.array(train_losses),
            test_losses=np.array(test_losses),
            loss_per_param_history=loss_per_param_history,
            labels=labels,
            job_id=args.job_id,
            epoch=epoch,
            max_epochs = max_epochs,
            update_interval=5,
            folder_name=args.folder_name
        )

        # Checkpointing logic
        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == max_epochs:
            # --- CALIBRATION LOGIC ---
            if use_mdn:
                # We need the latest predictions and targets to calculate calibration
                # These are numpy arrays from the last test() call
                latest_preds_scaled = preds
                latest_sigmas_scaled = sigmas
                latest_targets_scaled = targets

                # Convert to original scale tensors
                targets_orig = dataset.inverse_transform_labels(latest_targets_scaled)
                preds_orig, sigmas_orig = dataset.inverse_transform_labels_with_uncertainty(latest_preds_scaled, latest_sigmas_scaled)
                
                # Calculate new calibration factors
                new_factors = evaluate_calibration(preds_orig, sigmas_orig, targets_orig, args.model_params, args.job_id, epoch)
                
                # Update the global dictionary with the latest factors
                CALIBRATION_FACTORS.update(new_factors)
            # --- CALIBRATION LOGIC ---

            # Your existing checkpointing logic
            if test_loss < best_loss_at_last_checkpoint_save:
                print(f"New best validation loss: {test_loss:.4f} (previous: {best_loss_at_last_checkpoint_save:.4f})")
                save_checkpoint(epoch, model, optimizer, scaler, train_losses, test_losses, test_loss, checkpoint_save_dir)
                print(f"Checkpoint saved at epoch {epoch+1}")
                best_loss_at_last_checkpoint_save = test_loss
            else:
                print(f"No improvement in validation loss: {test_loss:.4f} (best: {best_loss_at_last_checkpoint_save:.4f})")
                print(f"Checkpoint not saved at epoch {epoch+1}, continuing training...")
    

        # Save final results to CSV and run final evaluation at the end of training
        if epoch + 1 == max_epochs:
            print("\n--- End of Training: Final Evaluation ---")
            
            # Use the results from the last test() call: preds, targets, sigmas
            # These are numpy arrays in the scaled space.
            final_preds_scaled = preds
            final_targets_scaled = targets
            final_sigmas_scaled = sigmas # This will be None if use_mdn is False

            # Inverse transform to original physical scale
            # The inverse transform methods correctly return PyTorch tensors
            targets_original = dataset.inverse_transform_labels(final_targets_scaled)

            if use_mdn:
                outputs_original, sigmas_original = dataset.inverse_transform_labels_with_uncertainty(final_preds_scaled, final_sigmas_scaled)
                
                # --- START: NEW EVALUATION CALLS ---
                print("Generating final evaluation plots...")
                
                # Call the new evaluation functions with the original-scale Tensors
                plot_residual_vs_sigma(outputs_original, sigmas_original, targets_original, args.model_params, args.job_id, folder_name=args.folder_name)
                evaluate_calibration(outputs_original, sigmas_original, targets_original, args.model_params, args.job_id, epoch, folder_name=args.folder_name)

                print("Final evaluation plots saved.")
                # --- END: NEW EVALUATION CALLS ---

                # Create DataFrame for CSV
                preds_np = outputs_original.numpy()
                sigmas_np = sigmas_original.numpy()
                targets_np = targets_original.numpy()

                df = pd.DataFrame()
                for i, param_name in enumerate(args.model_params):
                    df[f"true_{param_name}"] = targets_np[:, i]
                    df[f"pred_{param_name}"] = preds_np[:, i]
                    df[f"sigma_{param_name}"] = sigmas_np[:, i]
            
            else: # Handle the case where MDN is not used
                outputs_original = dataset.inverse_transform_labels(final_preds_scaled)
                preds_np = outputs_original.numpy()
                targets_np = targets_original.numpy()
                df = pd.DataFrame()
                for i, param_name in enumerate(args.model_params):
                    df[f"true_{param_name}"] = targets_np[:, i]
                    df[f"pred_{param_name}"] = preds_np[:, i]


            # Save DataFrame to CSV
            results_dir = f"./Final_Results/{args.job_id}/"
            os.makedirs(results_dir, exist_ok=True)
            df.to_csv(os.path.join(results_dir, "final_predictions.csv"), index=False)
            print(f"Saved final predictions to {os.path.join(results_dir, 'final_predictions.csv')}")
    # Save the final model checkpoint
    save_checkpoint(max_epochs-1, model, optimizer, scaler, train_losses, test_losses, test_loss, checkpoint_save_dir)
    
    