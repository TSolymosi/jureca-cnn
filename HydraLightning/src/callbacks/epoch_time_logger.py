import time
from lightning.pytorch.callbacks import Callback
from lightning_utilities.core.rank_zero import rank_zero_only

class EpochTimeLogger(Callback):
    def __init__(self):
        self._t0 = None
        self._epoch_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    # do NOT read trainer.callback_metrics here
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        self._epoch_time = time.perf_counter() - self._t0
        print(f"[Epoch {trainer.current_epoch+1}] time={self._epoch_time:.2f}s", flush=True)

    # if you want to print a metric, do it here instead (after val syncs)
    # @rank_zero_only
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     # read a value you logged yourself into the module (or skip metrics entirely)
    #     # e.g., pl_module.last_val_loss set in your LightningModule.on_validation_epoch_end
    #     val_loss = getattr(pl_module, "last_val/loss", None)
    #     if val_loss is not None:
    #         print(f"[Epoch {trainer.current_epoch+1}] val/loss={float(val_loss):.5f}")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called after the validation epoch ends and all metrics have been synchronized.
        This is the correct place to read final, aggregated metric values.
        """
        # --- THIS IS THE MODIFIED LOGIC ---
        # Instead of reading from a custom attribute on pl_module, we read from
        # trainer.callback_metrics. This is the official, synchronized dictionary
        # containing all logged metrics for the current step.
        metrics = trainer.callback_metrics
        
        # Safely get the 'val/loss' metric. The .get() method returns None if the key
        # doesn't exist, preventing a crash.
        val_loss = metrics.get("val/loss")

        if val_loss is not None:
            # Print the final validation loss for the epoch.
            print(f"[Metrics] Epoch {trainer.current_epoch}: val/loss = {val_loss:.5f}", flush=True)
        else:
            # This can happen if validation is skipped or if the metric has a different name.
            print(f"[Metrics] Epoch {trainer.current_epoch}: val/loss not found in callback_metrics.", flush=True)