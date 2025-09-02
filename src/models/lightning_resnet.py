import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import torch.distributed as dist
from typing import Optional, Dict, List, Tuple

from ResNet3D import generate_2d_model  # keep your import
# NOTE: your original gaussian_nll_loss_dict is not strictly required,
# since we reproduce its math per-parameter below with std weighting.

def _gather_cat_cpu(t) -> Optional[torch.Tensor]:
    if t is None:
        obj = None
    elif isinstance(t, list):
        if len(t) == 0:
            obj = None
        else:
            obj = torch.cat(t, dim=0)
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
    """
    Ports the data checks + MDN/non-MDN loss behavior from your original train()/test()
    into Lightning. DDP safe, AMP through Lightning. Plotting intentionally omitted.
    """
    def __init__(self, model_cfg, optim_cfg, training_cfg,
                 std_weights=None,
                 dump_bad_batch: bool = True,
                 dump_dir: str = "./",
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # ----------------------- Model -----------------------
        self.config_name = self._get_config_name(model_cfg.model_depth)
        self.model = generate_2d_model(
            config_name=self.config_name,
            use_batchnorm=model_cfg.use_batchnorm,
            TARGET_PARAMETERS=model_cfg.target_params,
            n_outputs=len(model_cfg.target_params),
            use_attention_heads=model_cfg.use_attention_heads,
            attention_latent_dim=model_cfg.attention_latent_dim,
            use_mdn=model_cfg.use_mdn,
        )
        self.model_params: List[str] = model_cfg.target_params
        self.use_mdn: bool = model_cfg.use_mdn

        # --------------------- Loss setup --------------------
        self.criterion = nn.MSELoss(reduction='mean') if not self.use_mdn else None
        # If head outputs already-positive sigmas, keep True. Otherwise we softplus.
        self.sigma_head_outputs_positive = getattr(model_cfg, "sigma_head_outputs_positive", True)
        self.sigma_floor = getattr(model_cfg, "sigma_floor", 1e-4)

        # Optional inverse-variance weighting (original used dataset scaler stds)
        self.std_weights = std_weights  # can be tensor/np/list
        self.dump_bad_batch = dump_bad_batch
        self.dump_dir = dump_dir

        # cache for validation epoch end (rank0 only)
        self._val_dataset = None
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_sigmas: Optional[List[torch.Tensor]] = [] if self.use_mdn else None
        self._val_cache: Dict[str, Optional[torch.Tensor]] = {}

    # ------------------- Lightning lifecycle -------------------

    def on_fit_start(self):
        # Mirror your on_fit_start, pull stds from the datamodule when available
        if self.std_weights is None and self.trainer is not None:
            dm = self.trainer.datamodule
            if dm is not None and getattr(dm, "dataset_ref", None) is not None:
                sw = dm.dataset_ref.scaler_stds
                try:
                    sw = sw.detach().cpu()
                except Exception:
                    pass
                self.std_weights = sw
                if self.trainer.is_global_zero:
                    self.print(f"[on_fit_start] std_weights set from datamodule: {self.std_weights}")

    def on_validation_epoch_start(self):
        # keep a handle to original dataset for inverse transforms if needed later
        self._val_dataset = getattr(self.trainer.datamodule, "dataset_ref", None)
        self._val_preds = []
        self._val_targets = []
        self._val_sigmas = [] if self.use_mdn else None

    # ------------------------- Helpers -------------------------

    def _get_config_name(self, model_depth: int) -> str:
        mapping = {
            10: "resnet10_2d_equivalent",
            18: "resnet18_2d_equivalent",
            34: "resnet34_2d",
            50: "resnet50_2d"
        }
        if model_depth not in mapping:
            raise ValueError(f"Unsupported model depth: {model_depth}. Supported depths: 10,18,34,50.")
        return mapping[model_depth]

    def forward(self, x: torch.Tensor):
        return self.model(x)

    @staticmethod
    def _finite_mask_per_sample(x: torch.Tensor) -> torch.Tensor:
        # like your: ~torch.isfinite(data.view(B,-1).sum(dim=1))
        return ~torch.isfinite(x.view(x.shape[0], -1).sum(dim=1))

    def _sanity_checks(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int):
        """Replicates your original defensive checks."""
        # non-finite inputs
        if not torch.isfinite(x).all():
            nan_mask = self._finite_mask_per_sample(x)
            bad_indices = nan_mask.nonzero(as_tuple=True)[0]
            if self.trainer.is_global_zero:
                self.print(f"\nNaNs/Inf in input at batch {batch_idx}, sample indices: {bad_indices.tolist()}")
                self.print("Corresponding target values for bad inputs:")
                for idx in bad_indices.tolist():
                    self.print(f"  Sample {idx}: {y[idx].detach().cpu().numpy()}")
                if self.dump_bad_batch:
                    os.makedirs(self.dump_dir, exist_ok=True)
                    torch.save({"data": x.detach().cpu(), "target": y.detach().cpu()},
                               os.path.join(self.dump_dir, f"nan_input_batch{batch_idx}.pt"))
            raise ValueError("NaN/Inf input encountered — likely due to bad parameter combination")

        if torch.isnan(x).any():
            raise ValueError("NaN in input")

        # Spectral dim empty (keeps your exact check on dim=2)
        if x.shape[2] == 0:
            raise ValueError("Empty spectral dimension (D=0) in input")

        if not torch.isfinite(y).all():
            raise ValueError("Target contains NaNs/Inf")

    def _split_mdn(self, output) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Split raw model output into lists of mu and sigma tensors (B,1 each)."""
        mu_list, sigma_list = [], []
        for i, _ in enumerate(self.model_params):
            mu    = output[i * 2]
            sigma = output[i * 2 + 1]
            # make shapes uniform (B,1)
            mu    = mu.view(mu.shape[0], 1)
            sigma = sigma.view(sigma.shape[0], 1)
            if not self.sigma_head_outputs_positive:
                sigma = F.softplus(sigma)
            # optional stability floor (keep off by default like in your code)
            # sigma = sigma.clamp_min(self.sigma_floor)
            mu_list.append(mu)
            sigma_list.append(sigma)
        return mu_list, sigma_list

    def _compute_loss(self, output, target, return_per_param: bool = False):
        """
        - Non-MDN: MSE with strict shape/finite checks
        - MDN: per-parameter gaussian_nll (variance form), optional std_weights scaling
        """
        if not self.use_mdn:
            # shape/finite checks (like your original before loss)
            if output.numel() == 0:
                raise RuntimeError(f"Empty output tensor! Shape: {output.shape}")
            if target.numel() == 0:
                raise RuntimeError(f"Empty target tensor! Shape: {target.shape}")
            if output.shape != target.shape:
                raise RuntimeError(f"Output-target shape mismatch: {output.shape} vs {target.shape}")
            if (not torch.isfinite(output).all()) or (not torch.isfinite(target).all()):
                raise ValueError("Non-finite values in output/target")
            loss = self.criterion(output, target)
            return (loss, {}) if return_per_param else loss

        # MDN path
        per_param = {}
        losses = []
        reg_term = 0.0
        sigma_reg = False
        sigma_reg_weights = {
            "D":0.05,"L":0.05,"rr":0.05,"ro":0.05,"p":0.05,"Tlow":0.05,"plummer_shape":0.05,"NCH3CN":0.05
        }

        for i, name in enumerate(self.model_params):
            mu    = output[i*2].squeeze(-1)      # [B]
            sigma = output[i*2+1].squeeze(-1)    # [B]
            if not self.sigma_head_outputs_positive:
                sigma = F.softplus(sigma)
            # keep your stability floor consistent with the head’s +1e-6
            sigma = torch.clamp(sigma, min=1e-6)

            var = sigma * sigma
            y_i = target[:, i]

            li = F.gaussian_nll_loss(mu, y_i, var, full=True, reduction="none")  # [B]
            if self.std_weights is not None:
                sw_i = float(self.std_weights[i])
                li = li / (sw_i ** 2)

            per_param[name] = li.mean()
            losses.append(li)

            if sigma_reg:
                w = sigma_reg_weights.get(name, 0.0)
                if w > 0:
                    reg_term += w * torch.mean(torch.log(sigma + 1e-6))

        loss = torch.stack(losses, dim=1).mean() + reg_term
        return (loss, per_param) if return_per_param else loss

    # --------------------- training / val / test ---------------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # replicate your guard checks
        self._sanity_checks(x, y, batch_idx)

        output = self(x)

        # Non-MDN: enforce shape parity like your script
        if not self.use_mdn and (output.shape != y.shape):
            raise RuntimeError(f"Output-target shape mismatch: {output.shape} vs {y.shape}")

        loss, per_param = self._compute_loss(output, y, return_per_param=True)
        
        if self.global_step == 0 and self.trainer.is_global_zero:
            self.print(f"[DEBUG] first-batch train_loss: {loss.item():.6f}")


        # logging mirrors your keys
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in per_param.items():
            self.log(f"train/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # keep checks in val too, to surface issues early
        self._sanity_checks(x, y, batch_idx)

        output = self(x)

        # cache preds/targets/sigmas on CPU (for later inverse transforms/plots)
        if self.use_mdn:
            mu_list, sigma_list = self._split_mdn(output)
            preds  = torch.cat(mu_list, dim=1)     # [B, P]
            sigmas = torch.cat(sigma_list, dim=1)  # [B, P]
        else:
            preds = output
            sigmas = None

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

        # same checks as test() path
        self._sanity_checks(x, y, batch_idx)

        output = self(x)
        loss, per_param = self._compute_loss(output, y, return_per_param=True)

        # For parity with your original "loss per param" reporting:
        for k, v in per_param.items():
            self.log(f"test/{k}_nll", v, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"test_loss": loss}

    def on_validation_epoch_end(self):
        preds_all = _gather_cat_cpu(self._val_preds)
        targets_all = _gather_cat_cpu(self._val_targets)
        sigmas_all = _gather_cat_cpu(self._val_sigmas) if self.use_mdn else None

        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            self._val_cache = {
                "preds": preds_all,
                "targets": targets_all,
                "sigmas": sigmas_all,
            }
        else:
            self._val_cache = {}

        # free buffers
        self._val_preds = []
        self._val_targets = []
        self._val_sigmas = None

    # ------------------- Optim / Scheduler -------------------

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
