import json
from pathlib import Path
from types import SimpleNamespace


def load_config(path):
    # The distributed config intentionally uses the JSON subset of YAML.
    text = "\n".join(line for line in Path(path).read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))
    cfg = json.loads(text)
    if cfg["data"].get("mat_cache", "none") not in ("none", "ram"):
        raise ValueError("mat_cache must be none or ram")
    tr = cfg["train"]
    if not isinstance(tr.get("save_stage1_checkpoints", True), bool):
        raise ValueError("save_stage1_checkpoints must be a boolean")
    for k in ("stage1_epochs", "stage2_epochs", "batch_size_per_gpu", "checkpoint_every"):
        if tr[k] < 1:
            raise ValueError(k + " must be positive")
    for stage in (1, 2):
        batch_size_for_stage(cfg, stage)
    if cfg["diffusion"]["num_timesteps"] < 2:
        raise ValueError("Training samples indices 1..T-1; T must be >= 2")
    if tr["validation"]["monitor"] not in ("psnr", "ssim", "validation_loss"):
        raise ValueError("Unsupported validation monitor")
    expected = "min" if tr["validation"]["monitor"] == "validation_loss" else "max"
    if tr["validation"]["mode"] != expected:
        raise ValueError("Monitor direction is inconsistent")
    if tr["validation"]["weights"] not in ("raw", "ema"):
        raise ValueError("Select raw or ema validation weights")
    if tr["validation"]["weights"] == "ema" and not tr["ema"]["enabled"]:
        raise ValueError("EMA validation requested with EMA disabled")
    if tr["validation"]["every"] < 1 or tr["early_stopping"]["patience"] < 1:
        raise ValueError("Validation interval and patience must be positive")
    if tr.get("stagewise_early_stopping", False):
        es = tr["early_stopping"]
        warmup = es.get("stage1_warmup_epochs")
        if es.get("warmup_reference", "history") not in ("history", "end"):
            raise ValueError("warmup_reference must be history or end")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < tr["stage1_epochs"]:
            raise ValueError("stage1_warmup_epochs must be an integer below the Stage 1 cap")
        if not es["enabled"] or es["monitor"] != tr["validation"]["monitor"]:
            raise ValueError("Stagewise stopping must be enabled and use the selection monitor")
        if tr["validation"]["every"] != 1:
            raise ValueError("Stagewise stopping requires validation every epoch")
        if not tr["reset_optimizer_stage2"] or not tr["reset_scheduler_stage2"]:
            raise ValueError("Stagewise stopping requires phase-boundary optimizer/scheduler resets")
    elif tr["early_stopping"]["monitor"] != "validation_loss":
        raise ValueError("Early stopping monitors validation_loss only")
    return cfg


def batch_size_for_stage(cfg, stage):
    if stage not in (1, 2):
        raise ValueError("Stage must be 1 or 2")
    tr = cfg["train"]
    value = tr.get("batch_size_per_gpu_stage" + str(stage), tr["batch_size_per_gpu"])
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("Stage batch size must be a positive integer")
    return value


def model_args(cfg):
    return SimpleNamespace(**cfg["model"], **cfg["diffusion"])


def stage_for_epoch(epoch, cfg):
    a, b = cfg["train"]["stage1_epochs"], cfg["train"]["stage2_epochs"]
    if not 1 <= epoch <= a+b:
        raise ValueError("Epoch outside the training budget")
    return 1 if epoch <= a else 2


def remaining_epochs(completed, cfg):
    total = cfg["train"]["stage1_epochs"] + cfg["train"]["stage2_epochs"]
    if not 0 <= completed <= total:
        raise ValueError("Invalid completed epoch")
    return range(completed+1, total+1)
