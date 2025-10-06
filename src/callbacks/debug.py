# debug.py
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import math
import torch.distributed as dist

def _grad_l2_norm(module) -> float:
    total_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach().float()
            total_sq += (g.norm(2).item()) ** 2
    return (total_sq ** 0.5) if total_sq > 0 else 0.0

def _print_optimizer_snapshot(trainer, pl_module, tag: str):
    if not getattr(trainer, "optimizers", None):
        print(f"[OPT-DEBUG][{tag}] no optimizers attached", flush=True)
        return

    opt = trainer.optimizers[0]
    try:
        pg = opt.param_groups[0]
    except Exception:
        print(f"[OPT-DEBUG][{tag}] optimizer has no param_groups?", flush=True)
        return

    lr           = pg.get("lr", None)
    weight_decay = pg.get("weight_decay", None)
    betas        = pg.get("betas", None)
    eps          = pg.get("eps", None)
    amsgrad      = pg.get("amsgrad", False)

    print(f"[OPT-DEBUG][{tag}] alg={opt.__class__.__name__} "
          f"lr={lr} weight_decay={weight_decay} betas={betas} eps={eps} amsgrad={amsgrad}", flush=True)

    # Current grad norm on model
    gn = _grad_l2_norm(pl_module)
    print(f"[OPT-DEBUG][{tag}] grad_l2_norm={gn:.6f}", flush=True)

    # Sample a parameter's AdamW state (step, ||exp_avg||, ||exp_avg_sq||)
    sample_p = None
    for p in pl_module.parameters():
        if p.requires_grad and p.grad is not None:
            sample_p = p
            break
    if sample_p is not None:
        st   = opt.state.get(sample_p, {})
        step = st.get("step", None)
        ea   = st.get("exp_avg", None)
        easq = st.get("exp_avg_sq", None)

        def _n(t):
            try:
                return float(t.norm().item())
            except Exception:
                return None

        print(f"[OPT-DEBUG][{tag}] state.step={step} | |exp_avg||={_n(ea)} | |exp_avg_sq||={_n(easq)}", flush=True)
    else:
        print(f"[OPT-DEBUG][{tag}] no grad-bearing parameter found this batch", flush=True)

    # AMP GradScaler (if enabled)
    scaler = getattr(getattr(trainer.strategy, "precision_plugin", None), "scaler", None)
    if scaler is not None:
        scale = None
        try:
            scale = scaler.get_scale()
        except Exception:
            scale = getattr(scaler, "_scale", None)
        print(f"[OPT-DEBUG][{tag}] amp_scale={scale}", flush=True)

    # LR scheduler metadata (no step yet, but useful to see config)
    cfgs = getattr(trainer, "lr_scheduler_configs", []) or []
    if cfgs:
        cfg = cfgs[0]
        sch = cfg.scheduler
        base_lr = getattr(sch, "base_lrs", [lr])[0] if hasattr(sch, "base_lrs") else lr
        print(f"[OPT-DEBUG][{tag}] scheduler={sch.__class__.__name__} base_lr={base_lr} monitor={getattr(cfg, 'monitor', None)}", flush=True)


# ---------- DDP-safe dataset summary helpers ----------
def _rank_world(trainer):
    # Prefer Lightning's attributes; fall back to torch.distributed if needed
    world = getattr(trainer, "world_size", None)
    rank  = getattr(trainer, "global_rank", None)
    if world is None:
        try:
            world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        except Exception:
            world = 1
    if rank is None:
        try:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        except Exception:
            rank = 0
    return rank, world

def _len_or_none(obj):
    try:
        return len(obj) if obj is not None else None
    except Exception:
        return None

def _get_ds(dm, *names):
    for n in names:
        ds = getattr(dm, n, None)
        if ds is not None:
            return ds
    return None
# ------------------------------------------------------


class DataloaderDebugCallback(Callback):

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        # model config summary
        cfg = getattr(pl_module, "hparams", None)
        if cfg is not None:
            print(f"[DEBUG] >>> model hparams: {cfg}", flush=True)

        # Log a single, consolidated summary on rank 0 without constructing loaders again.
        dm = getattr(trainer, "datamodule", None)
        rank, world = _rank_world(trainer)

        # Common attribute names for datasets after DataModule.setup('fit')
        train_ds = _get_ds(dm, "train_dataset", "train_ds", "train_set")
        val_ds   = _get_ds(dm, "val_dataset", "val_ds", "val_set")

        n_train = _len_or_none(train_ds)
        n_val   = _len_or_none(val_ds)
        batch_size = getattr(dm, "batch_size", None)

        # Estimate expected per-rank batches for DistributedSampler(drop_last=False).
        # (Lightning's default sampler pads last rank if needed.)
        def expected_batches(n):
            if n is None or batch_size is None or world is None:
                return None
            per_rank_samples = math.ceil(n / world)
            return math.ceil(per_rank_samples / batch_size)

        train_batches = expected_batches(n_train)
        val_batches   = expected_batches(n_val)

        print(
            "[DEBUG] >>> fit is starting, dataloaders should be constructed now",
            flush=True,
        )
        print(
            "[DEBUG][GLOBAL] dataset sizes: "
            f"train={n_train}, val={n_val} | "
            f"batch_size={batch_size}, world_size={world} | "
            f"expected_batches_per_rank(train={train_batches}, val={val_batches})",
            flush=True,
        )
        

    # @rank_zero_only
    # def on_train_dataloader(self, trainer, pl_module):
    #     print("[DEBUG] >>> train_dataloader() has been called", flush=True)

    # @rank_zero_only
    # def on_validation_dataloader(self, trainer, pl_module):
    #     print("[DEBUG] >>> val_dataloader() has been called", flush=True)

    # @rank_zero_only
    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     if batch_idx == 0:
    #         print("[DEBUG] >>> first training batch fetched!", flush=True)
    #         try:
    #             x, y, rms = batch
    #             print(f"[DEBUG] x={getattr(x, 'shape', None)}, y={getattr(y, 'shape', None)}, rms={rms}", flush=True)
    #         except Exception:
    #             print("[DEBUG] (batch structure not (x,y,rms))", flush=True)

    #         # Snapshot BEFORE optimizer step (after forward/backward hasn't run yet)
    #         _print_optimizer_snapshot(trainer, pl_module, tag="batch0_pre-step")

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            print("[DEBUG] >>> first training batch finished forward/backward/step (if acc_grad=1)", flush=True)
            # Snapshot AFTER optimizer step
            _print_optimizer_snapshot(trainer, pl_module, tag="batch0_post-step")
