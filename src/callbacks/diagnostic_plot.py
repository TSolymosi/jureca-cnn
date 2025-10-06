# src/callbacks/diagnostics.py
import os
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
from ResNet3D import plot_diagnostic_spectra

class DiagnosticSpectraCallback(Callback):
    def __init__(
        self,
        num_samples: int = 10,
        output_dir: str = "./diagnostic_plots",
        use_cauchy_noise: bool = True,
        # --- pass-through noise config (matches model-side aug knobs) ---
        cauchy_mu: float = 0.003,
        cauchy_sigma: float = 0.0032,
        cauchy_threshold: float = 0.07,
        add_gauss_sigma: float = 0.0,
        mask_frac: float = 0.0,
        mask_mode: str = "sample",
    ):
        super().__init__()
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.use_cauchy_noise = use_cauchy_noise

        # Matches the keys expected by plot_diagnostic_spectra(...) in Setup 1
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
        # Let the dataset craft the spectra; optionally add noise as configured
        train_loader = trainer.datamodule.train_dataloader()
        base_ds = train_loader.dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset

        freq_axis = getattr(base_ds, "get_frequency_axis", lambda: None)()

        # Hand the full noise config through so diagnostics match your training aug policy
        plot_diagnostic_spectra(
            base_ds,
            self.num_samples,
            trainer.logger.name if trainer.logger else "no_logger",
            freq_axis,
            self.output_dir,
            noise_cfg=self.noise_cfg,
        )

    @rank_zero_only
    def on_test_start(self, trainer, pl_module):
        test_loader = trainer.datamodule.test_dataloader()
        base_ds = test_loader.dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset

        freq_axis = getattr(base_ds, "get_frequency_axis", lambda: None)()
        plot_diagnostic_spectra(
            base_ds,
            self.num_samples,
            getattr(trainer, "logger", None).name if trainer.logger else "no_logger",
            freq_axis,
            self.output_dir,
            noise_cfg=self.noise_cfg,
        )