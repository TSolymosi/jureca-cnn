# lightning_resnet_script.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler # For LightningModule if not using PL's AMP

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import functools

# Override the default print function to flush the output immediately
print = functools.partial(print, flush=True)

# --- Model Helper Functions and Blocks ---
def get_inplanes():
    return [64, 128, 256, 512]

def spectral_conv1d(in_planes, out_planes, kernel_size=3, stride=1):
    padding = kernel_size // 2
    return nn.Conv3d(in_planes, out_planes, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)

def get_num_groups(num_channels):
    if num_channels == 0: return 1
    for g in [32, 16, 8, 4, 2]:
        if num_channels % g == 0:
            return g
    return 1

class BasicBlock2D(nn.Module):
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
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class Bottleneck2D(nn.Module): # Potentially used by generate_2d_model
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
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class Spectral2DResNet(nn.Module):
    # Used by generate_2d_model
    def __init__(self,
                 block_2d, spatial_layers_config, block_inplanes_spectral=64,
                 num_wavelengths_in=2000, initial_proj_channels=64, n_outputs=4,
                 dropout_prob=0.2, use_batchnorm=True, fc_hidden_dim=512,
                 target_params_list=None, simple_feature_head=True): # Added simple_feature_head
        super().__init__()
        self.use_batchnorm = use_batchnorm # Store for norm layers

        # --- Spectral Convolution Part ---
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
        reshaped_channels = block_inplanes_spectral * self.final_spectral_dim

        self.entry_projection_2d = nn.Sequential(
            nn.Conv2d(reshaped_channels, initial_proj_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(initial_proj_channels) if use_batchnorm else nn.GroupNorm(get_num_groups(initial_proj_channels), initial_proj_channels),
            nn.ReLU(inplace=True))
        
        current_channels = initial_proj_channels
        self.spatial_layers = nn.ModuleList()
        self.in_planes = current_channels
        for i, (planes_out, stride, num_blocks) in enumerate(spatial_layers_config):
            self.spatial_layers.append(
                self._make_spatial_layer_2d(block_2d, planes_out, num_blocks, stride, use_batchnorm))
        
        self.avgpool_2d = nn.AdaptiveAvgPool2d((1, 1))
        final_spatial_channels = spatial_layers_config[-1][0] * block_2d.expansion
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_spatial_channels, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob))

        if target_params_list is None: target_params_list = [f"param_{i}" for i in range(n_outputs)]
        self.output_heads = nn.ModuleDict()
        for param_name in target_params_list:
            if simple_feature_head: # Control head complexity
                 self.output_heads[param_name] = nn.Linear(fc_hidden_dim, 1)
            else:
                 self.output_heads[param_name] = nn.Sequential(
                     nn.Linear(fc_hidden_dim, fc_hidden_dim // 2 if fc_hidden_dim // 2 >= 32 else 32),
                     nn.ReLU(),
                     nn.Linear(fc_hidden_dim // 2 if fc_hidden_dim // 2 >= 32 else 32, 1))
        self._initialize_weights()

    def _make_spatial_layer_2d(self, block_2d, planes, num_blocks, stride, use_batchnorm):
        downsample = None
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
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x): # x is (N, 1, D_spec, H, W)
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(f"Expected 5D input (N, 1, D, H, W), got {x.shape}")
        x = self.spectral_conv_s1(x)
        x = self.spectral_conv_s2(x)
        x = self.spectral_conv_s3(x)
        N, C_spec, D_red, H, W = x.shape
        x = x.view(N, C_spec * D_red, H, W)
        x = self.entry_projection_2d(x)
        for layer in self.spatial_layers: x = layer(x)
        x = self.avgpool_2d(x)
        shared_features = self.shared_fc(x)
        outputs = [self.output_heads[param_name](shared_features) for param_name in self.output_heads]
        return torch.cat(outputs, dim=1)

# Global variable for generate_2d_model, will be set by args
_TARGET_PARAMETERS_FOR_MODEL_GEN = None
_SIMPLE_FEATURE_HEAD_FOR_MODEL_GEN = True


def generate_2d_model(config_name="resnet10_3layers", **kwargs):
    global _TARGET_PARAMETERS_FOR_MODEL_GEN, _SIMPLE_FEATURE_HEAD_FOR_MODEL_GEN
    if _TARGET_PARAMETERS_FOR_MODEL_GEN is None:
        raise ValueError("_TARGET_PARAMETERS_FOR_MODEL_GEN not set before calling generate_2d_model")

    default_kwargs = {
        'num_wavelengths_in': 2000, 'block_inplanes_spectral': 64,
        'initial_proj_channels': 64, 'n_outputs': len(_TARGET_PARAMETERS_FOR_MODEL_GEN),
        'dropout_prob': 0.2, 'use_batchnorm': True, 'fc_hidden_dim': 512,
        'target_params_list': _TARGET_PARAMETERS_FOR_MODEL_GEN,
        'simple_feature_head': _SIMPLE_FEATURE_HEAD_FOR_MODEL_GEN # Use global for head style
    }
    if config_name == "resnet10_3layers":
        spatial_config = [(64, 1, 1), (128, 2, 1), (256, 2, 1)]
        block_type = BasicBlock2D
    elif config_name == "resnet10_2d_equivalent":
        spatial_config = [(64, 1, 1), (128, 2, 1), (256, 2, 1), (512, 2, 1)]
        block_type = BasicBlock2D
    elif config_name == "resnet18_2d_equivalent":
         spatial_config = [(64, 1, 2), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
         block_type = BasicBlock2D
    else:
        raise ValueError(f"Unknown config_name: {config_name}")
    final_kwargs = {**default_kwargs, **kwargs}
    return Spectral2DResNet(block_2d=block_type, spatial_layers_config=spatial_config, **final_kwargs)


# --- PyTorch Lightning DataModule ---
class SpectralDataModule(pl.LightningDataModule):
    def __init__(self, args_config):
        super().__init__()
        self.args = args_config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.inverse_transform_func = None

    def setup(self, stage: str = None):
        # create_dataloaders is imported from dataloader.py
        from dataloader_lightning import create_dataloaders # Assuming this is where it is
        train_loader_tmp, val_loader_tmp, test_loader_tmp, dataset_ref = create_dataloaders(
            fits_dir=self.args.data_dir,
            original_file_list_path=self.args.original_file_list,
            scaling_params_path=self.args.scaling_params_path,
            wavelength_stride=self.args.wavelength_stride,
            load_preprocessed=self.args.load_preprocessed,
            preprocessed_dir=self.args.preprocessed_dir,
            use_local_nvme=self.args.use_local_nvme,
            batch_size=self.args.batch_size, # Dataloader will handle batching
            num_workers=self.args.num_workers,
            model_params=self.args.model_params,
            log_scale_params=self.args.log_scale_params,
            return_datasets=True # Modify create_dataloaders to return datasets
        )
        self.train_dataset = train_loader_tmp.dataset
        self.val_dataset = val_loader_tmp.dataset
        self.test_dataset = test_loader_tmp.dataset

        if hasattr(dataset_ref, 'inverse_transform_labels'):
            self.inverse_transform_func = dataset_ref.inverse_transform_labels
        else:
            print("Warning: inverse_transform_labels method not found on main dataset reference.")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.num_workers, pin_memory=True, persistent_workers=self.args.num_workers > 0)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, pin_memory=True, persistent_workers=self.args.num_workers > 0)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False,
                          num_workers=self.args.num_workers, pin_memory=True, persistent_workers=self.args.num_workers > 0)

# --- PyTorch Lightning Module ---
class SpectralLightningModule(pl.LightningModule):
    def __init__(self, model_arch_name: str, model_init_kwargs: dict,
                 learning_rate: float, weight_decay: float, scheduler_patience: int, scheduler_factor: float, min_lr: float,
                 criterion_name: str, criterion_huber_delta: float, criterion_weights: list = None,
                 per_target_std_noise: list = [0.5, 0.5, 0.2, 0.3],
                 custom_checkpoint_interval: int = 5,
                 args_config=None): # Pass the full args for plotting etc.
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args to self.hparams

        self.args_config = args_config # For plotting and other custom logic
        
        # Model Instantiation
        global _TARGET_PARAMETERS_FOR_MODEL_GEN, _SIMPLE_FEATURE_HEAD_FOR_MODEL_GEN
        _TARGET_PARAMETERS_FOR_MODEL_GEN = self.args_config.model_params
        _SIMPLE_FEATURE_HEAD_FOR_MODEL_GEN = model_init_kwargs.get('simple_feature_head', True) # Get from kwargs

        self.model = generate_2d_model(config_name=model_arch_name, **model_init_kwargs)

        # Criterion
        if criterion_name == 'Huber':
            self.criterion = nn.HuberLoss(delta=criterion_huber_delta, reduction='mean')
        elif criterion_name == 'MSE':
            self.criterion = nn.MSELoss(reduction='mean')
        elif criterion_name == 'WeightedMSE':
            if criterion_weights is None: raise ValueError("Weights needed for WeightedMSELoss")
            self.criterion = WeightedMSELoss(weights=criterion_weights) # Assuming WeightedMSELoss is defined
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
        
        self.per_target_std_noise = torch.tensor(per_target_std_noise)

        # For custom checkpointing (based on your previous script)
        self.best_loss_for_custom_checkpoint = float('inf')
        
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        current_device = target.device # Get device from target tensor

        # Inject noise
        noise_std = self.per_target_std_noise.to(current_device)
        if noise_std.shape[0] != target.shape[1]:
            raise ValueError("Shape mismatch for noise std and target labels.")
        noise = torch.randn_like(target) * noise_std
        labels_noisy = target + noise

        output = self(data)
        loss = self.criterion(output, labels_noisy)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        loss_per_param_batch = {}
        for i, param_name in enumerate(self.args_config.model_params):
            param_loss = self.criterion(output[:, i:i+1], target[:, i:i+1])
            loss_per_param_batch[param_name] = param_loss.item()
        
        self.validation_step_outputs.append({
            'val_loss_batch': loss.item(),
            'preds_batch': output.cpu(), 'targets_batch': target.cpu(),
            'loss_per_param_batch': loss_per_param_batch
        })
        return loss # Return overall loss for this batch

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs or not self.trainer.sanity_checking:
            self.validation_step_outputs.clear()
            return

        all_preds = torch.cat([x['preds_batch'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets_batch'] for x in self.validation_step_outputs])
        
        avg_val_loss_manual = np.mean([x['val_loss_batch'] for x in self.validation_step_outputs])
        # self.log('val_loss_epoch_manual', avg_val_loss_manual, logger=True, sync_dist=True) # PL logs val_loss already
        print(f"\nEpoch {self.current_epoch + 1} Validation Summary:")
        print(f"  Avg Validation Loss (from val_step): {self.trainer.logged_metrics.get('val_loss_epoch', avg_val_loss_manual):.4f}")


        loss_per_param_agg = {param: [] for param in self.args_config.model_params}
        for out_dict in self.validation_step_outputs:
            for p_name, p_loss in out_dict['loss_per_param_batch'].items():
                loss_per_param_agg[p_name].append(p_loss)
        
        print("  Avg Loss per parameter (Validation):")
        for p_name, losses_list in loss_per_param_agg.items():
            mean_p_loss = np.mean(losses_list) if losses_list else 0.0
            self.log(f'val_loss_{p_name}', mean_p_loss, logger=True, sync_dist=True)
            print(f"    {p_name}: {mean_p_loss:.4f}")

        dm = self.trainer.datamodule
        if dm and dm.inverse_transform_func:
            outputs_original = dm.inverse_transform_func(all_preds)
            targets_original = dm.inverse_transform_func(all_targets)
        else:
            outputs_original, targets_original = all_preds, all_targets
        if not isinstance(outputs_original, torch.Tensor): outputs_original = torch.from_numpy(outputs_original)
        if not isinstance(targets_original, torch.Tensor): targets_original = torch.from_numpy(targets_original)

        mae_per_output = torch.mean(torch.abs(outputs_original - targets_original), dim=0)
        print(f"  Mean Absolute Error (MAE) per output (original scale - Validation):")
        for i, param_name in enumerate(self.args_config.model_params):
            self.log(f'val_mae_{param_name}', mae_per_output[i].item(), logger=True, sync_dist=True)
            print(f"    {param_name}: {mae_per_output[i].item():.4f}")

        # Plotting (only on rank 0)
        if self.trainer.is_global_zero:
            self.plot_predictions_vs_true(
                outputs_original.numpy(), targets_original.numpy(),
                epoch=self.current_epoch, plot_prefix="Val", current_args=self.args_config
            )
            # Your save_training_plots can also be called here
            # It needs access to histories; PL loggers are better for this long-term
            # For now, let's assume you have a way to collect these if needed outside PL's logging

        # Your custom checkpointing logic
        current_val_loss = self.trainer.logged_metrics.get('val_loss_epoch', avg_val_loss_manual)
        if (self.current_epoch + 1) % self.hparams.custom_checkpoint_interval == 0:
            if current_val_loss < self.best_loss_for_custom_checkpoint:
                self.best_loss_for_custom_checkpoint = current_val_loss
                # Lightning automatically handles saving on rank 0 in DDP for trainer.save_checkpoint
                # We'll save to a specific path as per your old script
                custom_ckpt_dir = os.path.join(self.trainer.default_root_dir, f"checkpoints_custom/{self.args_config.job_id}")
                os.makedirs(custom_ckpt_dir, exist_ok=True)
                custom_ckpt_path = os.path.join(custom_ckpt_dir, "model_checkpoint.pth")
                
                # To save model, optimizer, scaler:
                # self.trainer.save_checkpoint(custom_ckpt_path) # This saves full PL state
                # To save only model state_dict like your old script:
                model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) or isinstance(self.model, pl.overrides. बाहर.parallel.LightningDistributedModule) else self.model.state_dict()
                # Optimizer and scaler state would also be needed for full resume-ability of your custom checkpoint
                # For now, just saving model state as per your example.
                torch.save({'epoch': self.current_epoch + 1,
                            'model_state_dict': model_state,
                            'best_loss_checkpointed': current_val_loss}, custom_ckpt_path)
                print(f"Custom checkpoint (model state) saved to {custom_ckpt_path} for epoch {self.current_epoch + 1}, val_loss {current_val_loss:.4f}")
            else:
                print(f"Custom checkpoint not saved (epoch {self.current_epoch + 1}): val_loss {current_val_loss:.4f} not better than {self.best_loss_for_custom_checkpoint:.4f}")
        
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        # Similar to validation_step, but for the test set
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log('test_loss_batch', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

        loss_per_param_batch = {}
        for i, param_name in enumerate(self.args_config.model_params):
            param_loss = self.criterion(output[:, i:i+1], target[:, i:i+1])
            loss_per_param_batch[param_name] = param_loss.item()

        self.test_step_outputs.append({
            'test_loss_batch': loss.item(),
            'preds_batch': output.cpu(), 'targets_batch': target.cpu(),
            'loss_per_param_batch': loss_per_param_batch
        })
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs: return

        all_preds = torch.cat([x['preds_batch'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['targets_batch'] for x in self.test_step_outputs])
        
        avg_test_loss_manual = np.mean([x['test_loss_batch'] for x in self.test_step_outputs])
        self.log('test_loss_epoch', avg_test_loss_manual, logger=True, sync_dist=True)
        print(f"\nTest Run Summary:")
        print(f"  Avg Test Loss: {avg_test_loss_manual:.4f}")

        loss_per_param_agg = {param: [] for param in self.args_config.model_params}
        # ... (aggregate and log per-param loss similar to validation) ...

        dm = self.trainer.datamodule
        if dm and dm.inverse_transform_func:
            outputs_original = dm.inverse_transform_func(all_preds)
            targets_original = dm.inverse_transform_func(all_targets)
        else:
            outputs_original, targets_original = all_preds, all_targets
        if not isinstance(outputs_original, torch.Tensor): outputs_original = torch.from_numpy(outputs_original)
        if not isinstance(targets_original, torch.Tensor): targets_original = torch.from_numpy(targets_original)

        mae_per_output = torch.mean(torch.abs(outputs_original - targets_original), dim=0)
        # ... (log MAE similar to validation) ...

        if self.trainer.is_global_zero:
            self.plot_predictions_vs_true(
                outputs_original.numpy(), targets_original.numpy(),
                epoch=self.current_epoch, plot_prefix="Test", current_args=self.args_config
            )
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.scheduler_factor,
                                      patience=self.hparams.scheduler_patience, min_lr=self.hparams.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def plot_predictions_vs_true(self, preds_np, targets_np, epoch, plot_prefix, current_args):
        # Your plotting logic from original 'test' function, slightly adapted
        print(f"Generating {plot_prefix} predicted vs. true plots for epoch {epoch+1}...")
        # Ensure job_id is part of current_args (passed as self.args_config)
        PNG_DIR = f"./Epoch_Plots/{current_args.job_id}/{plot_prefix}/" # Adjusted path
        os.makedirs(PNG_DIR, exist_ok=True)

        for i, current_param_name in enumerate(current_args.model_params):
            true_vals = targets_np[:, i]
            pred_vals = preds_np[:, i]
            plt.figure(figsize=(7, 7))
            if current_param_name in current_args.log_scale_params:
                plt.xscale('log'); plt.yscale('log')
                plt.title(f"{plot_prefix} Epoch {epoch+1} - {current_param_name} (Log-Log Scale)")
                mask = (true_vals > 1e-9) & (pred_vals > 1e-9)
                if np.any(mask):
                    plt.scatter(true_vals[mask], pred_vals[mask], alpha=0.3, s=10)
                    valid_true = true_vals[mask]
                    valid_pred = pred_vals[mask]
                    if len(valid_true) > 0 and len(valid_pred) > 0: # Check if arrays are non-empty
                        min_val = np.min([valid_true.min(), valid_pred.min()])
                        max_val = np.max([valid_true.max(), valid_pred.max()])
                        if min_val < max_val : # Ensure min_val is less than max_val for plotting line
                             plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
                else: # No valid data for log plot
                    plt.text(0.5, 0.5, 'No valid positive data', transform=plt.gca().transAxes, ha='center', va='center')
                    # Set some default log limits if no data
                    plt.xlim(1e-9, 1); plt.ylim(1e-9, 1)
            else: # Linear scale
                plt.title(f"{plot_prefix} Epoch {epoch+1} - {current_param_name} (Linear Scale)")
                plt.scatter(true_vals, pred_vals, alpha=0.3, s=10)
                if len(true_vals) > 0 and len(pred_vals) > 0:
                    min_val = np.min([true_vals.min(), pred_vals.min()])
                    max_val = np.max([true_vals.max(), pred_vals.max()])
                    if min_val < max_val:
                         plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

            plt.xlabel(f"True {current_param_name}"); plt.ylabel(f"Predicted {current_param_name}")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend(); plt.tight_layout()
            plot_filename = os.path.join(PNG_DIR, f"E{epoch+1}_{plot_prefix}_{current_param_name}.png")
            plt.savefig(plot_filename, dpi=150); plt.close()

# WeightedMSELoss needs to be defined if used
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
    def forward(self, y_pred, y_true):
        error = (y_pred - y_true) ** 2
        weighted_error = error * self.weights.to(y_pred.device)
        return weighted_error.mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--original-file-list", type=str, default=None)
    parser.add_argument("--scaling-params-path", type=str, default=None)
    parser.add_argument("--wavelength-stride", type=int, default=1)
    parser.add_argument('--load-preprocessed', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32) # Increased default for potential multi-GPU
    parser.add_argument("--use-local-nvme", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--job_id", type=str, default="pl_job")
    # parser.add_argument("--load_id", type=str, default=None) # For custom checkpoint loading path
    parser.add_argument("--model-arch-name", type=str, default="resnet10_3layers", choices=["resnet10_3layers", "resnet10_2d_equivalent", "resnet18_2d_equivalent"])
    parser.add_argument("--use-batchnorm", type=lambda x: x.lower() == 'true', default=False) # Default to GroupNorm
    parser.add_argument("--simple-feature-head", type=lambda x: x.lower() == 'true', default=True)

    parser.add_argument("--model_params", type=str, nargs='+', default=['Dens', 'Lum', 'radius', 'prho'])
    parser.add_argument("--log-scale-params", type=str, nargs='+', default=['Dens','Lum'])
    
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.2)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--criterion_name", type=str, default="Huber", choices=["Huber", "MSE", "WeightedMSE"])
    parser.add_argument("--criterion_huber_delta", type=float, default=0.1)
    parser.add_argument("--criterion_weights", type=float, nargs='+', default=None) # e.g., --criterion_weights 1.0 1.0 5.0 5.0
    parser.add_argument("--per_target_std_noise_train", type=float, nargs='+', default=[0.5, 0.5, 0.2, 0.3])
    parser.add_argument("--custom_checkpoint_interval", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    parser.add_argument("--pl_checkpoint_resume_path", type=str, default=None, help="Path to PyTorch Lightning checkpoint to resume from.")


    args = parser.parse_args()

    # Set globals for model generation (this is a bit of a hack, better to pass via kwargs)
    _TARGET_PARAMETERS_FOR_MODEL_GEN = args.model_params
    _SIMPLE_FEATURE_HEAD_FOR_MODEL_GEN = args.simple_feature_head


    pl.seed_everything(42, workers=True)

    # 1. DataModule
    data_module = SpectralDataModule(args_config=args)

    # 2. LightningModule
    model_init_kwargs_for_pl = {
        'use_batchnorm': args.use_batchnorm,
        'dropout_prob': 0.2, # Assuming a fixed dropout for now, or add to args
        'fc_hidden_dim': 512, # Assuming fixed, or add to args
        'simple_feature_head': args.simple_feature_head
        # Add other kwargs your generate_2d_model might expect from its **kwargs
    }
    lightning_model = SpectralLightningModule(
        model_arch_name=args.model_arch_name,
        model_init_kwargs=model_init_kwargs_for_pl,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        criterion_name=args.criterion_name,
        criterion_huber_delta=args.criterion_huber_delta,
        criterion_weights=args.criterion_weights,
        per_target_std_noise=args.per_target_std_noise_train,
        custom_checkpoint_interval=args.custom_checkpoint_interval,
        args_config=args # Pass full args for other uses like plotting
    )

    # --- Load your custom checkpoint state if it exists (for model weights only) ---
    # This is for your specific model_checkpoint.pth. PL's resume is different.
    # custom_load_dir = f"checkpoints/{args.load_id}" if args.load_id else None # From your original script
    # if custom_load_dir:
    #     custom_checkpoint_path = os.path.join(custom_load_dir, "model_checkpoint.pth")
    #     if os.path.exists(custom_checkpoint_path):
    #         print(f"Loading model state from custom checkpoint: {custom_checkpoint_path}")
    #         ckpt_data = torch.load(custom_checkpoint_path, map_location='cpu') # Load to CPU first
    #         lightning_model.model.load_state_dict(ckpt_data['model_state_dict'])
    #         lightning_model.best_loss_for_custom_checkpoint = ckpt_data.get('best_loss_checkpointed', float('inf'))
    #         # Note: Optimizer and scaler state from this custom checkpoint are NOT loaded here.
    #         # PL's `ckpt_path` in Trainer is preferred for full resume.

    # 3. Callbacks
    # Saves based on 'val_loss', top 1, min mode
    checkpoint_best_val_loss = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('./lightning_checkpoints/', args.job_id),
        filename='best-val-loss-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    # Saves the last checkpoint, useful for resuming
    checkpoint_last = ModelCheckpoint(
        dirpath=os.path.join('./lightning_checkpoints/', args.job_id),
        filename='last-epoch-{epoch:02d}',
        save_last=True # If PL < 1.5, use every_n_epochs=1 and save_top_k=-1 with a different filename pattern
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, verbose=True, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch') # Log LR per epoch
    
    callbacks_list = [checkpoint_best_val_loss, checkpoint_last, early_stop, lr_monitor]

    # 4. Logger
    logger = TensorBoardLogger("tb_logs", name=args.job_id, version=time.strftime("%Y%m%d-%H%M%S"))

    # 5. Trainer
    num_gpus = torch.cuda.device_count()
    trainer_strategy = 'auto'
    if num_gpus > 1:
        trainer_strategy = 'ddp_find_unused_parameters_true' if True else 'ddp' # Useful if some model outputs not used in loss
        print(f"Using {num_gpus} GPUs with DDP strategy: {trainer_strategy}")
    elif num_gpus == 1:
        print("Using 1 GPU.")
    else:
        print("No GPUs found, using CPU.")

    trainer = pl.Trainer(
        accelerator='auto',
        devices='auto', # Automatically detect GPUs or use CPU
        strategy=trainer_strategy,
        max_epochs=args.num_epochs,
        logger=logger,
        callbacks=callbacks_list,
        precision='16-mixed' if num_gpus > 0 else 32, # Mixed precision on GPU
        # deterministic=True, # Can slow down
        log_every_n_steps=50, # How often to log within an epoch
    )

    # 6. Training
    ckpt_to_resume = args.pl_checkpoint_resume_path
    if ckpt_to_resume and os.path.exists(ckpt_to_resume):
        print(f"Resuming training from PyTorch Lightning checkpoint: {ckpt_to_resume}")
        trainer.fit(lightning_model, datamodule=data_module, ckpt_path=ckpt_to_resume)
    else:
        if ckpt_to_resume: print(f"PL Checkpoint {ckpt_to_resume} not found. Starting fresh.")
        else: print("No PL resume checkpoint specified. Starting fresh training.")
        trainer.fit(lightning_model, datamodule=data_module)

    # 7. Testing (optional, uses the best checkpoint loaded by fit/test)
    print("\nRunning Test Phase...")
    # trainer.test(model=lightning_model, datamodule=data_module) # Uses best_model_path from checkpoint callback
    # Or test with a specific checkpoint:
    best_model_path = checkpoint_best_val_loss.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        print(f"Testing with best model from: {best_model_path}")
        trainer.test(ckpt_path=best_model_path, datamodule=data_module, model=lightning_model) # PL will load the model from ckpt_path
    else:
        print("No best model checkpoint found from training, testing with current model state.")
        trainer.test(model=lightning_model, datamodule=data_module)

    print(f"Training and testing finished. Logs and checkpoints in ./tb_logs and ./lightning_checkpoints")