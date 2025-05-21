import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torchvision.models as models
#from sklearn.metrics import mean_absolute_error
#from functools import partial
from dataloader import create_dataloaders
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Override the default print function to flush the output immediately
import functools
print = functools.partial(print, flush=True)



def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def spectral_conv1d(in_planes, out_planes, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Conv3d(in_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)

def spatial_conv2d(in_planes, out_planes, kernel_size=3, stride=1):
    #print(f"[spatial_conv2d] in_planes: {in_planes}, out_planes: {out_planes}, kernel_size: {kernel_size}, stride: {stride}")
    # Force SAME padding for downsampling
    if stride == 2:
        padding = 1 
    else:
        padding = kernel_size // 2
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, kernel_size, kernel_size),
        stride=(1, stride, stride),
        padding=(0, padding, padding),
        bias=False
    )

# Function to determine the number of groups for group normalization
def get_num_groups(num_channels):
    if num_channels == 0: return 1 # Avoid division by zero if planes is 0
    for g in [32, 16, 8, 4, 2]:
        if num_channels % g == 0:
            return g
    return 1



class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        error = (y_pred - y_true) ** 2
        weighted_error = error * self.weights.to(y_pred.device)
        return weighted_error.mean()
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = spatial_conv2d(in_planes, planes, kernel_size=3, stride=stride)
        #self.bn1 = nn.BatchNorm3d(planes)
        self.bn1 = nn.GroupNorm(get_num_groups(planes), planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spatial_conv2d(planes, planes, kernel_size=3, stride=1)
        #self.bn2 = nn.BatchNorm3d(planes)
        self.bn2 = nn.GroupNorm(get_num_groups(planes), planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #print(f"[BasicBlock] Input shape: {x.shape}")
        residual = x

        out = self.conv1(x)
        #print(f"[BasicBlock] After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #print(f"[BasicBlock] After conv2: {out.shape}")
        out = self.bn2(out)

        if self.downsample is not None:
            #print("[BasicBlock] Applying downsample")
            residual = self.downsample(x)
           # print(f"[BasicBlock] Downsampled residual shape: {residual.shape}")

        #print(f"[BasicBlock] Before addition: out={out.shape}, residual={residual.shape}")
        out += residual
        out = self.relu(out)
        #print(f"[BasicBlock] Output shape: {out.shape}")
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes, stride=(1, 1, 1))
        #self.bn1 = nn.BatchNorm3d(planes)
        self.bn1 = nn.GroupNorm(get_num_groups(planes), planes)
        self.conv2 = spatial_conv2d(planes, planes, kernel_size=3, stride=stride)
        #self.bn2 = nn.BatchNorm3d(planes)
        self.bn2 = nn.GroupNorm(get_num_groups(planes), planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, stride=(1, 1, 1))
        #self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.bn3 = nn.GroupNorm(get_num_groups(planes * self.expansion), planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #print(f"[Bottleneck] Input: {x.shape}")
        residual = x

        out = self.conv1(x)
        #print(f"[Bottleneck] After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #print(f"[Bottleneck] After conv2: {out.shape}")
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #print(f"[Bottleneck] After conv3: {out.shape}")
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            #print(f"[Bottleneck] Downsampled residual: {residual.shape}")

        #print(f"[Bottleneck] Before addition: out={out.shape}, residual={residual.shape}")
        out += residual
        out = self.relu(out)
        #print(f"[Bottleneck] Output: {out.shape}")
        return out

# 2D Adaptation of BasicBlock and Bottleneck:

class BasicBlock2D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, use_batchnorm=True): # Added BN toggle
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

# (Optional) For deeper models, a 2D Bottleneck
class Bottleneck2D(nn.Module):
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
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SpectralSpatialResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_outputs=3,
                 dropout_prob=0.4):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.spectral_conv = nn.Sequential(
            spectral_conv1d(1, 16, kernel_size=11, stride=4),
            #nn.BatchNorm3d(16),
            nn.GroupNorm(get_num_groups(16), 16),
            nn.ReLU(inplace=True),
            spectral_conv1d(16, 32, kernel_size=11, stride=4),
            #nn.BatchNorm3d(32),
            nn.GroupNorm(get_num_groups(32), 32),
            nn.ReLU(inplace=True),
            spectral_conv1d(32, block_inplanes[0], kernel_size=5, stride=2),
            #nn.BatchNorm3d(block_inplanes[0]),
            nn.GroupNorm(get_num_groups(block_inplanes[0]), block_inplanes[0]),
            nn.ReLU(inplace=True),
        )

        self.in_planes = block_inplanes[0]
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(block_inplanes[3] * block.expansion, 128),  # Dynamically infers input features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )
        # Output heads for each parameter
        self.output_heads = nn.ModuleDict({
            param: nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            ) for param in args.model_params
        })
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        out_channels = planes * block.expansion
        if stride != 1 or self.in_planes != out_channels:
            #if shortcut_type == 'A':
            #    downsample = partial(self._downsample_basic_block, planes=out_channels, stride=(1, stride, stride))
            #else:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                #nn.BatchNorm3d(out_channels))
                nn.GroupNorm(get_num_groups(out_channels), out_channels))
            #print(f"  --> Downsample Conv3d: in_channels={self.in_planes}, out_channels={out_channels}, stride=(1, {stride}, {stride})")
        #print(f"[make_layer] Creating block with in_planes={self.in_planes}, planes={planes}, stride={stride}")
        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).to(x.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def forward(self, x):
        #print(f"[Model] Input: {x.shape}")
        x = self.spectral_conv(x)
        #print(f"[Model] After spectral_conv: {x.shape}")

        x = self.layer1(x)
        #print(f"[Model] After layer1: {x.shape}")
        x = self.layer2(x)
        #print(f"[Model] After layer2: {x.shape}")
        x = self.layer3(x)
        #print(f"[Model] After layer3: {x.shape}")
        x = self.layer4(x)
        #print(f"[Model] After layer4: {x.shape}")

        x = self.avgpool(x)
        #print(f"[Model] After avgpool: {x.shape}")
        x = x.view(x.size(0), -1)
        #print(f"[Model] After flatten: {x.shape}")
        x = self.shared_fc(x)
        #print(f"[Model] Final output: {x.shape}")
        outputs = [self.output_heads[param](x) for param in args.model_params]
        return torch.cat(outputs, dim=1)


# 2D ResNet for Spectral Data
class Spectral2DResNet(nn.Module):
    def __init__(self,
                 block_2d, # BasicBlock2D or Bottleneck2D
                 spatial_layers_config, # e.g., [(64,1), (128,2), (256,2)] for (planes, stride)
                 block_inplanes_spectral=64, # Channels after last spectral conv
                 num_wavelengths_in=2000, # Initial spectral dim for calculation
                 initial_proj_channels=64, # Channels after 1x1 projection
                 n_outputs=4, # Assuming 4 params
                 dropout_prob=0.2, # Match old implementation
                 use_batchnorm=True, # To switch between BatchNorm and GroupNorm
                 fc_hidden_dim=512, # Hidden dim for shared FC
                 target_params_list=None): # For named output heads
        super().__init__()

        # --- Spectral Convolution Part ---
        # Calculate output spectral dimension after each conv to determine `reshaped_channels`
        # Layer 1
        spectral_kernel_size1, spectral_stride1 = 11, 4
        spec_padding1 = spectral_kernel_size1 // 2
        d_out1 = (num_wavelengths_in - spectral_kernel_size1 + 2 * spec_padding1) // spectral_stride1 + 1
        self.spectral_conv_s1 = nn.Sequential(
            spectral_conv1d(1, 16, kernel_size=spectral_kernel_size1, stride=spectral_stride1), # uses Conv3d
            nn.BatchNorm3d(16) if use_batchnorm else nn.GroupNorm(get_num_groups(16), 16),
            nn.ReLU(inplace=True)
        )
        # Layer 2
        spectral_kernel_size2, spectral_stride2 = 11, 4
        spec_padding2 = spectral_kernel_size2 // 2
        d_out2 = (d_out1 - spectral_kernel_size2 + 2 * spec_padding2) // spectral_stride2 + 1
        self.spectral_conv_s2 = nn.Sequential(
            spectral_conv1d(16, 32, kernel_size=spectral_kernel_size2, stride=spectral_stride2),
            nn.BatchNorm3d(32) if use_batchnorm else nn.GroupNorm(get_num_groups(32), 32),
            nn.ReLU(inplace=True)
        )
        # Layer 3
        spectral_kernel_size3, spectral_stride3 = 5, 2
        spec_padding3 = spectral_kernel_size3 // 2
        d_out3 = (d_out2 - spectral_kernel_size3 + 2 * spec_padding3) // spectral_stride3 + 1
        self.spectral_conv_s3 = nn.Sequential(
            spectral_conv1d(32, block_inplanes_spectral, kernel_size=spectral_kernel_size3, stride=spectral_stride3),
            nn.BatchNorm3d(block_inplanes_spectral) if use_batchnorm else nn.GroupNorm(get_num_groups(block_inplanes_spectral), block_inplanes_spectral),
            nn.ReLU(inplace=True)
        )
        self.final_spectral_dim = d_out3 # e.g., 63 if num_wavelengths_in=2000
        reshaped_channels = block_inplanes_spectral * self.final_spectral_dim # e.g., 64 * 63 = 4032

        # --- Entry Projection to 2D  ---
        self.entry_projection_2d = nn.Sequential(
            nn.Conv2d(reshaped_channels, initial_proj_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(initial_proj_channels) if use_batchnorm else nn.GroupNorm(get_num_groups(initial_proj_channels), initial_proj_channels),
            nn.ReLU(inplace=True)
        )
        current_channels = initial_proj_channels # This will be self.in_planes for the first 2D block

        # --- Spatial 2D Residual Layers  ---
        self.spatial_layers = nn.ModuleList()
        self.in_planes = current_channels # Track input planes for each block
        
        # spatial_layers_config = [(planes_out, stride, num_blocks_in_stage), ...]
        
        
        for i, (planes_out, stride, num_blocks) in enumerate(spatial_layers_config):
            self.spatial_layers.append(
                self._make_spatial_layer_2d(block_2d, planes_out, num_blocks, stride, use_batchnorm)
            )
        
        # --- Final Head ---
        self.avgpool_2d = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output channels from last spatial layer
        final_spatial_channels = spatial_layers_config[-1][0] * block_2d.expansion

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_spatial_channels, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)
        )

        if target_params_list is None: # Fallback if not provided
            target_params_list = [f"param_{i}" for i in range(n_outputs)]
        
        # Using separate heads
        self.output_heads = nn.ModuleDict()
        for param_name in target_params_list:
            if fc_hidden_dim == 512 and n_outputs == 4 and simple_feature_head:
                 self.output_heads[param_name] = nn.Linear(fc_hidden_dim, 1) # Simpler head
            else:
                 self.output_heads[param_name] = nn.Sequential(
                     nn.Linear(fc_hidden_dim, fc_hidden_dim // 2 if fc_hidden_dim // 2 >= 32 else 32 ), # Intermediate layer
                     nn.ReLU(),
                     nn.Linear(fc_hidden_dim // 2 if fc_hidden_dim // 2 >= 32 else 32, 1)
                 )


        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_spatial_layer_2d(self, block_2d, planes, num_blocks, stride, use_batchnorm):
        downsample = None
        if stride != 1 or self.in_planes != planes * block_2d.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block_2d.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block_2d.expansion) if use_batchnorm else nn.GroupNorm(get_num_groups(planes * block_2d.expansion), planes * block_2d.expansion),
            )

        layers = []
        layers.append(block_2d(self.in_planes, planes, stride, downsample, use_batchnorm=use_batchnorm))
        self.in_planes = planes * block_2d.expansion
        for _ in range(1, num_blocks):
            layers.append(block_2d(self.in_planes, planes, use_batchnorm=use_batchnorm))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is assumed to be: (N, 1, D_spec, H, W)
        # e.g., (batch_size_gpu, 1, 2000, 100, 100)

        # REMOVED
        # x = x.unsqueeze(1)

        # Ensure input x has the correct number of dimensions and channel size for the first conv
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), but got {x.ndim}D with shape {x.shape}")
        if x.shape[1] != 1: # self.spectral_conv_s1 expects in_channels=1
            raise ValueError(f"Expected input channel C=1, but got C={x.shape[1]} for shape {x.shape}")
        

        # Spectral Convolutions
        x = self.spectral_conv_s1(x)
        x = self.spectral_conv_s2(x)
        x = self.spectral_conv_s3(x) # Shape: (N, C_spec_out, D_reduced, H, W)

        # Reshape for 2D Convs
        N, C_spec, D_red, H, W = x.shape
        x = x.view(N, C_spec * D_red, H, W) # Combine spectral and channel dimensions

        # Project combined channels down
        x = self.entry_projection_2d(x) # Shape: (N, initial_proj_channels, H, W)

        # Spatial Feature Extraction (2D)
        for layer in self.spatial_layers:
            x = layer(x)

        # Global Pooling & FC Layers
        x = self.avgpool_2d(x)
        shared_features = self.shared_fc(x)

        outputs = []
        for param_name in self.output_heads: # Iterate in defined order
            outputs.append(self.output_heads[param_name](shared_features))
        
        return torch.cat(outputs, dim=1)





def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = SpectralSpatialResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = SpectralSpatialResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = SpectralSpatialResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = SpectralSpatialResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = SpectralSpatialResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = SpectralSpatialResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = SpectralSpatialResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model


S1_SpectralCNN_SpectralConv_like_head = True

def generate_2d_model(config_name="s1_spec_conv_like", **kwargs):
    """
    Generates a Spectral2DResNet model.
    """
    # Common settings
    default_kwargs = {
        'num_wavelengths_in': 2000,
        'block_inplanes_spectral': 64, # Channels after spectral convs
        'initial_proj_channels': 64,   # Channels after 1x1 projection into 2D
        'n_outputs': len(TARGET_PARAMETERS),
        'dropout_prob': 0.2,
        'use_batchnorm': True,         # Toggle between BatchNorm and GroupNorm
        'fc_hidden_dim': 512,
        'target_params_list': TARGET_PARAMETERS
    }
    

    if config_name == "resnet10_3layers":
        # Mimics SpectralCNN_SpectralConv: 3 spatial blocks (64->64, 64->128, 128->256)
        # Each "stage" here has 1 block.
        spatial_config = [
            (64, 1, 1),   # planes_out, stride, num_blocks_in_stage
            (128, 2, 1),
            (256, 2, 1)
        ]
        block_type = BasicBlock2D
        # Set head style if needed for specific model
        global simple_feature_head
        simple_feature_head = True


    elif config_name == "resnet10_2d_equivalent":
        # A ResNet10-like structure, but 2D spatial, 4 stages
        spatial_config = [
            (64, 1, 1),
            (128, 2, 1),
            (256, 2, 1),
            (512, 2, 1) # ResNet 4th stage
        ]
        block_type = BasicBlock2D
        simple_feature_head = True 

    elif config_name == "resnet18_2d_equivalent":
         # ResNet18 usually is [2,2,2,2] for 4 stages.
         spatial_config = [
             (64, 1, 2),
             (128, 2, 2),
             (256, 2, 2),
             (512, 2, 2)
         ]
         block_type = BasicBlock2D
         simple_feature_head = True

    else:
        raise ValueError(f"Unknown config_name: {config_name}")

    # Override defaults with any explicitly passed kwargs
    final_kwargs = {**default_kwargs, **kwargs}

    model = Spectral2DResNet(block_2d=block_type,
                             spatial_layers_config=spatial_config,
                             **final_kwargs)
    return model


# --- Checkpointing and Loading Functions ---
CHECKPOINT_FILENAME = f"model_checkpoint.pth"

def save_checkpoint(epoch, model, optimizer, scaler, current_loss, checkpoint_dir="checkpoints"):
    """Saves model checkpoint."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILENAME)

    # If model is wrapped in DataParallel, save the underlying module's state_dict
    model_state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    print(f"Saving checkpoint to {checkpoint_path} (End of Epoch {epoch+1}, Loss: {current_loss:.4f})")
    checkpoint = {
        'epoch': epoch + 1,  # Save as the epoch number *completed*
        'model_state_dict': model_state_dict_to_save,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss_checkpointed': current_loss
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
            start_epoch = 0 # Hard coded for the moment to avoid plotting errors
            best_loss_checkpointed = checkpoint.get('best_loss_checkpointed', float('inf'))

            print(f"Checkpoint loaded. Resuming training from epoch {start_epoch +1} (0-indexed: {start_epoch}).")
            print(f"Best loss recorded in loaded checkpoint: {best_loss_checkpointed:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            best_loss_checkpointed = float('inf')
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

    return start_epoch, best_loss_checkpointed





# Optimized Training Loop
def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Inject noise into the labels
        per_target_std = torch.tensor([0.5, 0.5, 0.2, 0.3], device=target.device)
        # Check if per_target_std matches labels shape
        if per_target_std.shape[0] != target.shape[1]:
            raise ValueError(f"Shape mismatch! per_target_std: {per_target_std.shape}, Labels: {target.shape}")
        noise = torch.randn_like(target) * per_target_std
        labels_noisy = target + noise

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            output = model(data)
            loss = criterion(output, labels_noisy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    
    #torch.save(model.state_dict(), "model_weights.pth")


    return running_loss / len(train_loader)


def test(model, test_loader, criterion, device, epoch):
    model.eval()
    test_loss = 0.0
    # Store all predictions and targets
    all_preds = []
    all_targets = []

    # Track loss per parameter
    loss_per_param = {param: [] for param in args.model_params}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with autocast(device_type='cuda'):
                output = model(data)
                loss = criterion(output, target)

                # Compute loss for each parameter
                for i, param in enumerate(args.model_params):
                    param_loss = criterion(output[:, i], target[:, i])
                    loss_per_param[param].append(param_loss.item())

                test_loss += loss.item()
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print("Loss per parameter:")
    for param, losses in loss_per_param.items():
        avg_loss = np.mean(losses)
        print(f"  {param}: {avg_loss:.4f}")

    # Convert back to original scale for analysis
    outputs_original = train_loader.dataset.dataset.inverse_transform_labels(all_preds)
    targets_original = train_loader.dataset.dataset.inverse_transform_labels(all_targets)

    # Compute MAE
    #mae = mean_absolute_error(all_targets, all_preds)
    mae = torch.mean(torch.abs(outputs_original - targets_original), dim=0).numpy()  # shape: (n_outputs,)
    print(f"Mean Absolute Error (MAE) original scale: {mae}")

    print("Generating predicted vs. true plots...")

    PNG_DIR = f"./Epoch_Plots/{args.job_id}/"
    os.makedirs(PNG_DIR, exist_ok=True)

    # Convert tensors to numpy arrays
    all_outputs_original_np = outputs_original.numpy()
    all_labels_original_np = targets_original.numpy()

    for i, current_param_name in enumerate(args.model_params):
        true_vals = all_labels_original_np[:, i]
        pred_vals = all_outputs_original_np[:, i]

        plt.figure(figsize=(7, 7))  # Square aspect ratio

        # Log-log plot for selected parameters
        if current_param_name in args.log_scale_params:
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f"Epoch {epoch+1} - {current_param_name} (Log-Log Scale)")

            # Filter valid data
            mask = (true_vals > 0) & (pred_vals > 0)

            if np.any(mask):
                plt.scatter(true_vals[mask], pred_vals[mask], alpha=0.3, s=10)
                min_val = min(true_vals[mask].min(), pred_vals[mask].min())
                max_val = max(true_vals[mask].max(), pred_vals[mask].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
            else:
                print(f"Warning: No positive data for log-log plot of {current_param_name}")
                plt.text(0.5, 0.5, 'No valid data', transform=plt.gca().transAxes,
                        ha='center', va='center')
                plt.xlim(1e-9, 1)
                plt.ylim(1e-9, 1)

        else:
            # Linear plot
            plt.title(f"Epoch {epoch+1} - {current_param_name} (Linear Scale)")
            plt.scatter(true_vals, pred_vals, alpha=0.3, s=10)
            min_val = min(true_vals.min(), pred_vals.min())
            max_val = max(true_vals.max(), pred_vals.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

        plt.xlabel(f"True {current_param_name}")
        plt.ylabel(f"Predicted {current_param_name}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save
        plot_filename = os.path.join(PNG_DIR, f"E{epoch+1}_Test_{current_param_name}.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close()

    

    

    return test_loss / len(test_loader), all_preds, all_targets, loss_per_param


def save_training_plots(train_losses, test_losses, loss_per_param_history, labels, job_id, epoch, max_epochs, update_interval=10):
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
        plt.savefig(f"ResNetPlots/{job_id}/Train_vs_Validation_E{epoch+1}.png")
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
        plt.savefig(f"ResNetPlots/{job_id}/val_loss_per_param_E{epoch+1}.png")
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
    import argparse

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    
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
    )
    # Create folder for plots
    os.makedirs(f"plots", exist_ok=True)

    # Specify checkpointing directory
    checkpoint_save_dir = f"checkpoints/{args.job_id}/"
    checkpoint_load_dir = f"checkpoints/{args.load_id}"

    global TARGET_PARAMETERS
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
    model = generate_2d_model(config_name="resnet10_3layers",use_batchnorm=False, target_params_list=TARGET_PARAMETERS, n_outputs=num_outputs)
    

    # Trying to use multi GPUs 
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    model = model.to(device)

    #from torchviz import make_dot
    #model_graph = make_dot(model(sample_image.to(device)[:1]), params=dict(model.named_parameters()))
    #model_graph.render("plots/model_graph", format="png")
    #print("Saved model graph to plots/model_graph.png")

    #from torchsummary import summary
    # Show model summary
    #summary(model, input_size=(1, 2000, 100, 100), device=str(device))

    #weights_path = f"model_weights.pth"
    #if os.path.exists(weights_path):
    #    model.load_state_dict(torch.load(weights_path))
    #    print("Model weights loaded.")
    #else:
    #    print("Model weights file not found")

    #criterion = nn.L1Loss()
    #criterion = WeightedMSELoss(weights=[1.0, 1.0, 5.0, 5.0])  # Adjust weights as needed
    #criterion = nn.MSELoss(reduction='mean')
    criterion = nn.HuberLoss(delta=0.1, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scaler = GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
    #scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # Example: Decrease LR every 10 epochs


    # Print criterion, optimizer, scaler and scheduler information
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
    
    #early stopper
    #early_stopper = EarlyStopping(patience=50, min_delta=1e-8)

    # Create directory for plots
    os.makedirs(f"ResNetPlots/{args.job_id}", exist_ok=True)

    # Load checkpoint if exists
    # `best_loss_at_last_checkpoint_save` stores the loss value OF THE CHECKPOINT CURRENTLY ON DISK
    start_epoch, best_loss_at_last_checkpoint_save = load_checkpoint(model, optimizer, scaler, device, checkpoint_load_dir)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        test_loss, preds, targets, loss_per_param = test(model, test_loader, criterion, device, epoch)
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
            update_interval=5
        )

        # Checkpointing logic
        # Condition 1: Is it a checkpointing epoch?
        if (epoch + 1) % checkpoint_interval == 0:
            # Condition 2: Is current loss better than the loss of the checkpoint on disk?
            if test_loss < best_loss_at_last_checkpoint_save:
                print(f"Current test loss ({test_loss:.4f}) is better than "
                      f"loss of saved checkpoint ({best_loss_at_last_checkpoint_save:.4f}).")
                save_checkpoint(epoch, model, optimizer, scaler, test_loss, checkpoint_save_dir)
                best_loss_at_last_checkpoint_save = test_loss # Update with the new best loss saved
            else:
                print(f"Current test loss ({test_loss:.4f}) is not better than "
                      f"loss of saved checkpoint ({best_loss_at_last_checkpoint_save:.4f}). "
                      f"Checkpoint not updated for epoch {epoch+1}.")
        else:
            if epoch < num_epochs -1 : # Avoid printing for the last epoch if it's not a checkpoint interval
                 print(f"Epoch {epoch+1} is not a designated checkpointing interval (every {checkpoint_interval} epochs).")
                
        #if early_stopper.early_stop:
        #    print(f"Early stopping triggered at epoch {epoch}")
        #    break
    
    # Save the model
    torch.save(model.module.state_dict(), f"Weights/ResNet3D_{model_depth}_weights.pth")
    
    