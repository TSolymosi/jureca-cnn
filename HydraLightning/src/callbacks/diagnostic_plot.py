# src/callbacks/diagnostic_plot.py
import os
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
from ResNet3D import plot_diagnostic_spectra
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
            plot_diagnostic_spectra(base_ds, self.num_samples, trainer.logger.name, freq_axis, self.output_dir, noise_cfg=self.noise_cfg)
        else:
            print("Cauchy noise is not used")
