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
        print(f"[Epoch {trainer.current_epoch}] time={self._epoch_time:.2f}s")

    # if you want to print a metric, do it here instead (after val syncs)
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        # read a value you logged yourself into the module (or skip metrics entirely)
        # e.g., pl_module.last_val_loss set in your LightningModule.on_validation_epoch_end
        val_loss = getattr(pl_module, "last_val_loss", None)
        if val_loss is not None:
            print(f"[Epoch {trainer.current_epoch}] val_loss={float(val_loss):.5f}")
