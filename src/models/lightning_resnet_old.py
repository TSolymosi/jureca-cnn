import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from ResNet3D import generate_2d_model, gaussian_nll_loss_dict  # Your original code

import torch.distributed as dist
from typing import Optional, Dict

def _gather_cat_cpu(t) -> Optional[torch.Tensor]:
    """
    DDP-safe gather for CPU tensors of variable batch counts using all_gather_object.
    Returns concatenated tensor on rank 0; None on other ranks.
    """
    #print(f"[Rank {dist.get_rank()}] -> Entered _gather_cat_cpu")
    if t is None:
        obj = None
    elif isinstance(t, list):
        if len(t) == 0:
            obj = None
        else:
            obj = torch.cat(t, dim=0)  # Concatenate along batch dimension
    elif torch.is_tensor(t):
        obj = t
    else:
        obj = None

    if dist.is_available() and dist.is_initialized():
        objs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(objs, obj)
        if dist.get_rank() == 0:
            tensors = [o for o in objs if isinstance(o, torch.Tensor) and o.numel() > 0]
            if len(tensors) == 0:
                return None
            return torch.cat(tensors, dim=0)
        else:
            return None
    else:
        return obj

class LitResNetMDN(LightningModule):
    def __init__(self, model_cfg, optim_cfg, training_cfg, std_weights=None):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.config_name = self.get_config_name(model_cfg.model_depth)
        self.model = generate_2d_model(
            config_name=self.config_name,
            use_batchnorm=model_cfg.use_batchnorm,
            TARGET_PARAMETERS=model_cfg.target_params,
            n_outputs=len(model_cfg.target_params),
            use_attention_heads=model_cfg.use_attention_heads,
            attention_latent_dim=model_cfg.attention_latent_dim,
            use_mdn=model_cfg.use_mdn,
        )
        self.model_params = model_cfg.target_params
        self.use_mdn = model_cfg.use_mdn
        self.std_weights = std_weights
        self.criterion = nn.MSELoss(reduction='mean') if not model_cfg.use_mdn else None

        # IMPORTANT: set True if your head already applies softplus to sigma.
        # If your head outputs raw logits, set this False and the loss will apply softplus once.
        self.sigma_head_outputs_positive = getattr(model_cfg, "sigma_head_outputs_positive", True)
        self.sigma_floor = getattr(model_cfg, "sigma_floor", 1e-4)  # can try 5e-3 if unstable


    def on_validation_epoch_start(self):
        #print(f"[Rank {self.global_rank}] -> Entered validation_epoch_start")
        self._val_dataset = self.trainer.datamodule.get_dataset_reference()
        
        self._val_preds = []
        self._val_targets = []
        self._val_sigmas = [] if getattr(self, "use_mdn", False) else None

    def on_fit_start(self):
        # If not provided at construction, fetch from datamodule (after setup)
        if self.std_weights is None and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "dataset_ref", None) is not None:
                sw = dm.dataset_ref.scaler_stds
                # keep a CPU tensor / numpy, whatever your loss expects
                try:
                    sw = sw.detach().cpu()
                except Exception:
                    pass
                self.std_weights = sw
                if self.trainer.is_global_zero:
                    self.print(f"[on_fit_start] std_weights set from datamodule: {self.std_weights}")

    def get_config_name(self, model_depth):
        mapping = {
            10: lambda: "resnet10_2d_equivalent",
            18: lambda: "resnet18_2d_equivalent",
            34: lambda: "resnet34_2d",
            50: lambda: "resnet50_2d"
        }
        if model_depth not in mapping:
            raise ValueError(f"Unsupported model depth: {model_depth}. Supported depths are 10, 18, 34, 50.")
        return mapping[model_depth]()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        output = self(x)
        loss, per_param = self._compute_loss(output, y, return_per_param=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in per_param.items():
            self.log(f"train/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        output = self(x)

        if getattr(self, "use_mdn", False):
            mu_list, sigma_list = [], []
            for i, _ in enumerate(self.model_params):
                mu    = output[i * 2]        # expect [B] or [B,1]
                sigma = output[i * 2 + 1]    # expect [B] or [B,1]

                # Make shapes robust for concatenation
                mu    = mu.view(mu.shape[0], 1)
                sigma = sigma.view(sigma.shape[0], 1)

                # For plotting we want the same sigma notion as the loss uses
                if not self.sigma_head_outputs_positive:
                    sigma = F.softplus(sigma)
                #sigma = sigma.clamp_min(self.sigma_floor)

                mu_list.append(mu)
                sigma_list.append(sigma)

            preds  = torch.cat(mu_list, dim=1)     # [B, P]
            sigmas = torch.cat(sigma_list, dim=1)  # [B, P]
        else:
            preds = output
            sigmas = None

        # move to CPU immediately (avoid GPU growth)
        self._val_preds.append(preds.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        if sigmas is not None:
            self._val_sigmas.append(sigmas.detach().cpu())

        loss, per_param = self._compute_loss(output, y, return_per_param=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in per_param.items():
            self.log(f"val/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": loss}


    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        output = self(x)
        loss, per_param = self._compute_loss(output, y, return_per_param=True)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in per_param.items():
            self.log(f"test/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
        return {"test_loss": loss}


    def on_validation_epoch_end(self):
        #print(f"[Rank {self.global_rank}] -> Entered on_validation_epoch_end")

        preds_cpu = self._val_preds
        targets_cpu = self._val_targets
        sigmas_cpu = self._val_sigmas if self.use_mdn else None

        #print(f"[Rank {self.global_rank}] -> Starting gather")

        preds_all = _gather_cat_cpu(preds_cpu)
        #print(f"[Rank {self.global_rank}] -> Finished gathering preds")
        
        targets_all = _gather_cat_cpu(targets_cpu)
        #print(f"[Rank {self.global_rank}] -> Finished gathering targets")

        sigmas_all = _gather_cat_cpu(sigmas_cpu) if sigmas_cpu is not None else None
        #print(f"[Rank {self.global_rank}] -> Finished gathering sigmas")


        # expose to callbacks
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            self._val_cache: Dict[str, Optional[torch.Tensor]] = {
                "preds":   preds_all,
                "targets": targets_all,
                "sigmas":  sigmas_all,
            }
        else:
            self._val_cache = {}

        # free per-rank buffers
        self._val_preds = []
        self._val_targets = []
        self._val_sigmas = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optim_cfg.lr,
            weight_decay=self.hparams.optim_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def _compute_loss(self, output, target, return_per_param=False):
        if not self.use_mdn:
            return (self.criterion(output, target), {}) if return_per_param else self.criterion(output, target)

        losses = []
        per_param = {}

        for i, name in enumerate(self.model_params):
            mu    = output[i * 2].squeeze(-1)        # [B]
            sigma = output[i * 2 + 1].squeeze(-1)    # [B]

            if not self.sigma_head_outputs_positive:
                sigma = F.softplus(sigma)
            #sigma = sigma.clamp_min(self.sigma_floor)
            var = sigma * sigma

            y = target[:, i]
            li = F.gaussian_nll_loss(mu, y, var, full=True, reduction="none")  # [B]

            # Optional inverse-variance weighting in label space (recommended)
            if self.std_weights is not None:
                li = li / (float(self.std_weights[i]) ** 2)

            per_param[name] = li.mean()
            losses.append(li)

        loss = torch.stack(losses, dim=1).mean()  # mean over params and batch
        return (loss, per_param) if return_per_param else loss


