import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torchvision.models as models
from sklearn.metrics import mean_absolute_error
from functools import partial
from dataloader import create_dataloaders
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import matplotlib.pyplot as plt
import numpy as np


def get_inplanes():
    # Standard number of output channels for ResNet blocks
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3D convolution with 3x3x3 kernel
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    # 3D convolution with 1x1x1 kernel for dimensionality changes
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def spectral_conv1d(in_planes, out_planes, kernel_size=3, stride=1):
    # 1D convolution along spectral axis (depth axis)
    padding = kernel_size // 2
    return nn.Conv3d(in_planes, out_planes, kernel_size=(kernel_size, 1, 1),
                     stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)

def spatial_conv2d(in_planes, out_planes, kernel_size=3, stride=1):
    # 2D spatial convolution using a 3D conv with size (1, kernel, kernel)
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


class WeightedMSELoss(nn.Module):
    # Weighted mean squared error loss
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, y_pred, y_true):
        error = (y_pred - y_true) ** 2
        weighted_error = error * self.weights.to(y_pred.device)
        return weighted_error.mean()
    

class BasicBlock(nn.Module):
    # Basic residual block used in ResNet18/34
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = spatial_conv2d(in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = spatial_conv2d(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
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

class Bottleneck(nn.Module):
    # Bottleneck residual block used in ResNet50+
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes, stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = spatial_conv2d(planes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, stride=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SpectralSpatialResNet(nn.Module):
    # Full ResNet-based model with spectral + spatial separation
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_outputs=3):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        # Spectral feature extractor (reduces spectral axis)
        self.spectral_conv = nn.Sequential(
            spectral_conv1d(1, 8, kernel_size=5, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            spectral_conv1d(8, 16, kernel_size=5, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            spectral_conv1d(16, block_inplanes[0], kernel_size=5, stride=2),
            nn.BatchNorm3d(block_inplanes[0]),
            nn.ReLU(inplace=True),
        )

        self.in_planes = block_inplanes[0]

        # Spatial residual blocks
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        # Final regression head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(block_inplanes[3] * block.expansion, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, n_outputs)
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        # Create a sequential block of residual layers
        downsample = None
        out_channels = planes * block.expansion
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, out_channels, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_channels))

        layers = [block(in_planes=self.in_planes, planes=planes, stride=stride, downsample=downsample)]
        self.in_planes = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.spectral_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def generate_model(model_depth, **kwargs):
    # Generate a model with given depth (ResNet10-200)
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



# Optimized training loop using AMP
# Trains the model over one epoch and records timing metrics for profiling

def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    batch_times = {
        'data': [],
        'forward': [],
        'loss': [],
        'backward': [],
        'step': [],
        'total': []
    }

    for batch_idx, (data, target) in enumerate(train_loader):
        t0 = time.time()

        # Transfer data to GPU
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        t1 = time.time()

        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(device_type='cuda'):
            output = model(data)
        t2 = time.time()

        # Compute loss
        with autocast(device_type='cuda'):
            loss = criterion(output, target)
        t3 = time.time()

        # Backpropagation
        scaler.scale(loss).backward()
        t4 = time.time()

        # Optimizer step and scaler update
        scaler.step(optimizer)
        scaler.update()
        t5 = time.time()

        running_loss += loss.item()

        # Collect timing
        batch_times['data'].append(t1 - t0)
        batch_times['forward'].append(t2 - t1)
        batch_times['loss'].append(t3 - t2)
        batch_times['backward'].append(t4 - t3)
        batch_times['step'].append(t5 - t4)
        batch_times['total'].append(t5 - t0)

    # Print average timings
    print("\n--- Average batch times (in seconds) ---")
    for k, v in batch_times.items():
        print(f"{k:>10}: {np.mean(v):.4f}")

    return running_loss / len(train_loader)


# Evaluation loop: tests model on validation set and computes per-parameter loss and MAE

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    param_losses = {param: [] for param in args.model_params}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            with autocast(device_type='cuda'):
                output = model(data)
                loss = criterion(output, target)

                # Per-parameter loss computation
                for i, param in enumerate(args.model_params):
                    param_loss = criterion(output[:, i], target[:, i])
                    param_losses[param].append(param_loss.item())

                test_loss += loss.item()

            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    # Concatenate predictions and ground truths
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print("Loss per parameter:")
    for param, losses in param_losses.items():
        avg_loss = np.mean(losses)
        print(f"  {param}: {avg_loss:.4f}")

    # De-standardize predictions/targets to physical units
    outputs_original = train_loader.dataset.inverse_transform_labels(all_preds)
    targets_original = train_loader.dataset.inverse_transform_labels(all_targets)

    # Compute MAE
    mae = torch.mean(torch.abs(outputs_original - targets_original), dim=0).numpy()
    print(f"Mean Absolute Error (MAE) original scale: {mae}")

    return test_loss / len(test_loader), all_preds, all_targets, mae, param_losses


# Simple early stopping helper to stop training when validation loss plateaus
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--job-id", type=str, default=None)
    parser.add_argument("--model-depth", type=int, default=18)
    parser.add_argument("--model-params", type=str, nargs='+', default=["Dens", "Lum", "radius", "prho"])
    parser.add_argument("--log-scale-params", type=str, nargs='+', default=["Dens", "Lum"])
    parser.add_argument("--normalization-method", type=str, default="zscore")
    args = parser.parse_args()
    
    print("Log-scale params: ", args.log_scale_params, type(args.log_scale_params))

    train_loader, test_loader, dataset = create_dataloaders(
        fits_dir=args.data_dir,
        file_list_path=args.original_file_list,
        scaling_params_path=args.scaling_params_path,
        wavelength_stride=args.wavelength_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_params=args.model_params,
        log_scale_params=args.log_scale_params,
        normalization_method=args.normalization_method,
    )
    # Create folder for plots
    os.makedirs(f"ResNetPlots", exist_ok=True)
    # Create specific folder
    os.makedirs(f"ResNetPlots/100_files", exist_ok=True)
    # Create job ID folder
    if args.job_id is not None:
        os.makedirs(f"ResNetPlots/100_files/{args.job_id}", exist_ok=True)

    # Train with AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(device)}")

    start_time = time.time()
    sample_image, _ = next(iter(train_loader))
    end_time = time.time()
    print(f"Time to load sample image: {end_time - start_time:.2f}s")
    print(f"DEBUG: input shape = {sample_image.shape}") 
    num_outputs = len(next(iter(train_loader))[1][0])

    model_depth = args.model_depth
    model = generate_model(model_depth=model_depth, n_outputs=num_outputs)

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

    criterion = nn.MSELoss()
    #criterion = WeightedMSELoss(weights=[1.0, 1.0, 5.0, 5.0])  # Adjust weights as needed
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)

    train_losses = []
    test_losses = []
    mae_values = []
    loss_per_param = {param: [] for param in args.model_params}
    num_epochs = args.num_epochs

    labels = ["Dens", "Lum", "radius", "prho"]#, "NCH3CN", "incl", "phi"]
    
    #early stopper
    #early_stopper = EarlyStopping(patience=50, min_delta=1e-8)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        test_loss, preds, targets, mae, epoch_losses = test(model, test_loader, criterion, device)
        scheduler.step(test_loss)
        #early_stopper.step(test_loss)
        end_time = time.time()

        print(f"Epoch time: {end_time - start_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {test_loss:.4f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: LR = {current_lr:.2e}")

        # Denormalize predictions and targets
        #label_min = np.load("Parameters/label_min.npy")
        #label_max = np.load("Parameters/label_max.npy")
        #label_min = np.load("/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data/labels_100/label_min.npy")
        #label_max = np.load("/p/scratch/pasta/CNN/17.03.25/Processed_Data/processed_data/labels_100/label_max.npy")
        
        #denorm_preds = preds * (label_max - label_min) + label_min
        #denorm_targets = targets * (label_max - label_min) + label_min
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        mae_values.append(mae.tolist())

        for param in args.model_params:
            avg_loss = np.mean(epoch_losses[param])
            if epoch == 0:
                loss_per_param[param] = [avg_loss]
            else:
                loss_per_param[param].append(avg_loss)

        #Print loss per parameter
        print(f"Loss per parameter: {loss_per_param}")

        if (epoch + 1) % 10 == 0:
            os.makedirs(f"ResNetPlots/100_files/{args.job_id}/Epochs", exist_ok=True)
            outputs_original = train_loader.dataset.inverse_transform_labels(preds)
            targets_original = train_loader.dataset.inverse_transform_labels(targets)

            log_params = set(train_loader.dataset.log_scale_params)
            for i, param in enumerate(labels):
                plt.figure(figsize=(6, 6))
                x = targets_original[:, i]
                y = outputs_original[:, i]

                # Compute shared axis range
                xy_min = min(x.min(), y.min())
                xy_max = max(x.max(), y.max())

                # Plot points
                plt.scatter(x, y, alpha=0.5)

                # Plot y = x reference line
                line_vals = np.linspace(xy_min, xy_max, 100)
                plt.plot(line_vals, line_vals, 'k--', label='Perfect prediction')

                # Set same limits for both axes
                plt.xlim([xy_min, xy_max])
                plt.ylim([xy_min, xy_max])

                plt.xlabel(f"True {param}")
                plt.ylabel(f"Predicted {param}")
                plt.title(f"Epoch {epoch+1} - {param} (Denorm)")

                if param in log_params:
                    plt.xscale("log")
                    plt.yscale("log")

                plt.grid(True, which="both", ls="--")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"ResNetPlots/100_files/{args.job_id}/Epochs/epoch_{epoch+1}_{param}_100.png")
                plt.close()




        #if early_stopper.early_stop:
        #    print(f"Early stopping triggered at epoch {epoch}")
        #    break
    
    # Save the model
    #torch.save(model.module.state_dict(), f"Weights/ResNet3D_{model_depth}_weights_100.pth")
    
    # ----------------------------
    # Loss and mae Plot
    # ----------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ResNetPlots/100_files/{args.job_id}/Train_vs_Validation_100.png")
    print("Saved: Train_vs_Validation_100.png")

    #mae_values = np.array(mae_values)  # shape: (epochs, n_outputs)
    #n_outputs = mae_values.shape[1]

    #plt.figure(figsize=(10, 6))
    #for i in range(n_outputs):
    #    plt.plot(mae_values[:, i], label=f'{labels[i]}')
    #plt.xlabel("Epoch")
    #plt.ylabel("MAE")
    #plt.title("Per-Parameter MAE Over Training")
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.savefig("ResNetPlots/100_files/per_parameter_mae_100.png")
    #print("Saved: per_parameter_mae_100.png")

    # plot loss per parameter
    for param in args.model_params:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_per_param[param], label=f'{param} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{param} Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"ResNetPlots/100_files/{args.job_id}/{param}_loss_100.png")
        print(f"Saved: {param}_loss_100.png")