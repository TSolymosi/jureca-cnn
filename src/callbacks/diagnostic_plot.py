# src/callbacks/diagnostics.py
import os
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
from ResNet3D import plot_diagnostic_spectra 

class DiagnosticSpectraCallback(Callback):
    def __init__(self, num_samples=10, output_dir="./diagnostic_plots", use_cauchy_noise=True):
        super().__init__()
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.use_cauchy_noise = use_cauchy_noise


    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        if self.use_cauchy_noise:
            train_loader = trainer.datamodule.train_dataloader()
            base_ds = train_loader.dataset
            while hasattr(base_ds, "dataset"):
                base_ds = base_ds.dataset
            freq_axis = getattr(base_ds, "get_frequency_axis", lambda: None)()
            plot_diagnostic_spectra(base_ds, self.num_samples, trainer.logger.name, freq_axis, self.output_dir)
        else:
            print("Cauchy noise is not used")
