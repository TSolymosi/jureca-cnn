# src/callbacks/diagnostic_plot.py
import os
import torch
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
from src.models.augmentation_utils import _apply_noise_and_mask_on_device
import matplotlib.pyplot as plt
import numpy as np

#from ResNet3D import plot_diagnostic_spectra
import inspect

class DiagnosticSpectraCallback(Callback):
    def __init__(self, num_samples=10, output_dir="./diagnostic_plots", use_cauchy_noise=True, cauchy_mu=0.003, cauchy_sigma=0.0032, cauchy_threshold=0.07, add_gauss_sigma=0.0, mask_frac=0.0, mask_mode="sample"):
        
        # --- DEBUGGING ---
        #print("\n\n--- PYTHON IS RUNNING THIS EXACT __init__ METHOD ---\n")
        #print(f"--- SIGNATURE AS SEEN BY PYTHON: {inspect.getfullargspec(self.__init__)}\n\n")
        
        super().__init__()

        

        #print("\n\n--- SUCCESS: Loading the NEW version of DiagnosticSpectraCallback! ---\n\n")
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.use_cauchy_noise = use_cauchy_noise
        self.noise_cfg = {
        "use_cauchy_gauss": use_cauchy_noise,
        "cauchy_mu": cauchy_mu,
        "cauchy_sigma": cauchy_sigma,
        "cauchy_threshold": cauchy_threshold,
        "add_gauss_sigma": add_gauss_sigma,
        "mask_frac": mask_frac,
        "mask_mode": mask_mode,
        }


    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        if self.use_cauchy_noise:
            train_loader = trainer.datamodule.train_dataloader()
            base_ds = train_loader.dataset
            while hasattr(base_ds, "dataset"):
                base_ds = base_ds.dataset
            freq_axis = getattr(base_ds, "get_frequency_axis", lambda: None)()
            self.plot_diagnostic_spectra(base_ds, self.num_samples, trainer.logger.name, freq_axis, self.output_dir, noise_cfg=self.noise_cfg)
        else:
            print("Cauchy noise is not used")

    def plot_diagnostic_spectra(self, dataset, num_samples, job_id, freq_axis, output_dir, noise_cfg: dict):
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
