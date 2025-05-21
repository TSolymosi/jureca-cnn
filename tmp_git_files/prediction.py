import torch
from ResNet3D import generate_model 
import torch.nn as nn
import numpy as np

import os
import random
import matplotlib.pyplot as plt
from astropy.io import fits
from dataloader import normalize, FitsDataset
import re

def config():
    # ====== CONFIGURATION ======
    model_depth = 34  # Adjust to match your training config
    num_outputs = 4   # Dens, Lum, radius, prho
    weights_path = "/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/Weights/ResNet3D_34_2000_weights.pth"  # Update path if needed

    # ====== LOAD MODEL ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = generate_model(model_depth=model_depth, n_outputs=num_outputs)

    model = nn.DataParallel(model)  # match how it was saved
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {weights_path}")

def extract_label(filename):
    pattern = r'([A-Za-z0-9_]+)=([-\d.eE+]+)'
    matches = re.findall(pattern, filename)
    label_dict = {k.lstrip('_'): float(v) for k, v in matches}
    expected_keys = ["Dens", "Lum", "radius", "prho"]#, "NCH3CN", "incl", "phi"]
    return [label_dict.get(key, 0.0) for key in expected_keys]



@torch.no_grad()
def predict_on_random_fits(model, dataset, label_min, label_max, device, n_samples=5):
    model.eval()
    sample_indices = random.sample(range(len(dataset)), n_samples)

    for idx in sample_indices:
        fits_path = dataset.fits_files[idx]
        filename = os.path.basename(fits_path)
        label = extract_label(filename)
        # Load and normalize input
        data, _ = dataset[idx]
        data = data.unsqueeze(0).to(device)  # Add batch dimension

        # Run model
        output = model(data).squeeze(0).cpu().numpy()
        denorm_output = output * (label_max - label_min + 1e-8) + label_min

        # Load raw FITS for visualization
        with fits.open(fits_path) as hdul:
            raw_data = hdul[0].data
            if raw_data is None:
                print(f"Skipped file with no data: {filename}")
                continue

        # Plot (choose a central slice)
        middle = raw_data.shape[0] // 2
        plt.figure(figsize=(6, 5))
        plt.imshow(raw_data[middle], cmap='inferno', origin='lower')
        plt.style.use('dark_background')
        plt.colorbar()

        # Plot styling
        plt.axis('off')

        # Add filename and predictions using figtext
        plt.figtext(0.5, 0.92, f"File: {label}", ha="center", fontsize=6, wrap=True, color='limegreen')

        labels = ["Dens", "Lum", "radius", "prho"]
        pred_lines = [f"{k}: {v:.3e}" for k, v in zip(labels, denorm_output)]
        pred_text = "Predicted:\n" + "\n".join(pred_lines)
        print(f"Predictions for {filename}: {pred_text}")
        plt.figtext(0.5, 0.02, pred_text, ha="center", fontsize=9, color='limegreen')

        plt.tight_layout()
        plt.savefig(f"/p/scratch/pasta/CNN/17.03.25/Test/Testing/prediction_plot_{idx}.png", dpi=150)
        plt.close()



if __name__ == "__main__":
    model_depth = 34  # Adjust to match your training config
    num_outputs = 4   # Dens, Lum, radius, prho
    weights_path = "/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/Weights/ResNet3D_34_2000_weights.pth"
    label_min_path = "/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/Parameters/label_min.npy"
    label_max_path = "/p/scratch/pasta/CNN/17.03.25/SpectralSpatial3DCNN/Parameters/label_max.npy"
    wavelength_stride = 1
    data_dir = "/p/scratch/pasta/CNN/17.03.25/Data"
    use_nvme = False
    num_samples = 5

    # ====== LOAD MODEL ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = generate_model(model_depth=model_depth, n_outputs=num_outputs)

    model = nn.DataParallel(model)  # match how it was saved
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {weights_path}")

    # Load label scaling
    label_min = np.load(label_min_path)
    label_max = np.load(label_max_path)

    # Load dataset (raw FITS files)
    dataset = FitsDataset(
        fits_dir=data_dir,
        wavelength_stride=wavelength_stride,
        use_local_nvme=use_nvme,
        load_preprocessed=False
    )
    dataset.label_min = label_min
    dataset.label_max = label_max

    # Run predictions
    predict_on_random_fits(model, dataset, label_min, label_max, device, n_samples=num_samples)
