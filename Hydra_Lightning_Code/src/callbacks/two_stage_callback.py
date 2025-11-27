# In src/callbacks/two_stage_callback.py

from lightning.pytorch.callbacks import Callback
import torch

class TwoStageMDNTraining(Callback):
    """
    A robust Lightning Callback for two-stage MDN training that correctly handles
    fresh starts and resuming from checkpoints.
    """
    def __init__(self, switch_epoch: int, use_two_stage: bool = True):
        super().__init__()
        self.switch_epoch = switch_epoch
        self.use_two_stage = use_two_stage
        self._stage_2_activated = False

    def setup(self, trainer, pl_module, stage: str):
        """The setup hook is a good place to set the initial state."""
        if stage == "fit":
            # Default to Stage 1 unless told otherwise
            pl_module.is_stage_1 = self.use_two_stage

    def on_train_start(self, trainer, pl_module):
        """
        Called after checkpoint loading but before the first training epoch.
        This is the perfect place to set the state for a resumed run.
        """
        if not self.use_two_stage:
            # Use the LightningModule's built-in rank-zero print
            pl_module.print("Two-stage training is disabled. Ensuring all params are trainable.")
            pl_module.is_stage_1 = False
            self._stage_2_activated = True
            for param in pl_module.model.cov_head.parameters():
                param.requires_grad = True
            return

        # --- THIS IS THE KEY LOGIC FOR RESUMING ---
        if trainer.current_epoch >= self.switch_epoch:
            # We are resuming a run that is already in Stage 2.
            pl_module.print(f"\n{'='*80}")
            pl_module.print(f"RESUMING in STAGE 2 (current_epoch={trainer.current_epoch} >= switch_epoch={self.switch_epoch}).")
            pl_module.print(f"Ensuring covariance head is unfrozen.")
            pl_module.print(f"{'='*80}\n", flush=True)
            
            pl_module.is_stage_1 = False
            self._stage_2_activated = True
            for param in pl_module.model.cov_head.parameters():
                param.requires_grad = True
        else:
            # We are starting a fresh run or resuming in Stage 1.
            pl_module.print(f"\n{'='*80}")
            pl_module.print(f"STARTING in STAGE 1 (current_epoch={trainer.current_epoch} < switch_epoch={self.switch_epoch}).")
            pl_module.print(f"Freezing covariance head.")
            pl_module.print(f"{'='*80}\n", flush=True)
            
            pl_module.is_stage_1 = True
            self._stage_2_activated = False
            for param in pl_module.model.cov_head.parameters():
                param.requires_grad = False

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the beginning of each training epoch.
        This handles the transition during a continuous run.
        """
        if not self.use_two_stage:
            return

        # Check if it's time to switch from Stage 1 to Stage 2
        if not self._stage_2_activated and trainer.current_epoch == self.switch_epoch:
            pl_module.print(f"\n{'='*80}")
            pl_module.print(f"TRANSITIONING to STAGE 2 at epoch {self.switch_epoch}.")
            pl_module.print("Unfreezing covariance head and reconfiguring optimizer.")
            pl_module.print(f"{'='*80}\n", flush=True)

            # 1. Unfreeze the covariance head
            for param in pl_module.model.cov_head.parameters():
                param.requires_grad = True
            
            # 2. Update state flags
            pl_module.is_stage_1 = False
            self._stage_2_activated = True

            # 3. Re-initialize the optimizer with a potentially new learning rate
            new_lr = pl_module.hparams.optim_cfg.lr / 5.0
            optimizer = torch.optim.AdamW(
                pl_module.parameters(),
                lr=new_lr,
                weight_decay=pl_module.hparams.optim_cfg.weight_decay,
            )
            trainer.optimizers = [optimizer]