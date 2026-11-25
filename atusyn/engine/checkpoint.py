import copy
import inspect
import os
import pickle
import random
import warnings
import numpy as np
import torch


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 pickle paths for ordinary arrays/scalars used in RNG state.
        aliases = {
            ("numpy._core.multiarray", "_reconstruct"): "numpy.core.multiarray",
            ("numpy._core.multiarray", "scalar"): "numpy.core.multiarray",
            ("numpy._core.numeric", "_frombuffer"): "numpy.core.numeric",
        }
        return super().find_class(aliases.get((module, name), module), name)


class _NumpyCompatPickle:
    Unpickler = _NumpyCompatUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _NumpyCompatUnpickler(file, **kwargs).load()


def load_file(path, device="cpu"):
    # Only load checkpoints obtained from trusted sources.
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    try:
        return torch.load(path, **kwargs)
    except ModuleNotFoundError as error:
        if error.name not in ("numpy._core", "numpy._core.multiarray", "numpy._core.numeric"):
            raise
        warnings.warn(
            "Checkpoint references numpy._core but runtime NumPy is %s; "
            "retrying with limited array/scalar pickle-path compatibility. "
            "No checkpoint keys or values are replaced. File: %s" % (np.__version__, path),
            RuntimeWarning, stacklevel=2,
        )
        return torch.load(path, pickle_module=_NumpyCompatPickle, **kwargs)


def load_generator(model, checkpoint, weights="raw"):
    if weights == "ema":
        if not isinstance(checkpoint, dict) or checkpoint.get("ema") is None:
            raise ValueError("No explicit EMA weights in this checkpoint")
        state = checkpoint["ema"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif "gen_diffusive_2_dict" in checkpoint:
        state = checkpoint["gen_diffusive_2_dict"]
    else:
        state = checkpoint
    # DDP prefix only; do not rename layers or silently ignore unmatched keys.
    if state and all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def atomic_save(payload, path):
    temporary = str(path) + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if device.type != "cuda":
            raise ValueError("CUDA training state cannot be resumed on CPU exactly")
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def snapshot_cpu(value):
    """Freeze a training snapshot in CPU RAM without retaining live tensor aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: snapshot_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [snapshot_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(snapshot_cpu(v) for v in value)
    return copy.deepcopy(value)


def load_stage1_best(model, path, cfg, data_identity, selection, ema=None, memory_checkpoint=None):
    """Load the selected Stage 1 weights without rewinding the run epoch/RNG."""
    if selection["epoch"] is None or selection["best"] is None:
        raise ValueError("Stage 1 has no finite validation winner; cannot start Stage 2")
    if path is None and memory_checkpoint is None:
        raise ValueError("No in-memory Stage 1 winner is available")
    best = load_file(path) if memory_checkpoint is None else memory_checkpoint
    if best.get("stage") != 1 or best.get("completed_epoch") != selection["epoch"]:
        raise ValueError("best_stage1.pt does not match the recorded Stage 1 winner")
    if best.get("config") != cfg or best.get("data_identity") != data_identity:
        raise ValueError("Stage 1 best checkpoint config or data identity differs")
    if best.get("selection") != selection:
        raise ValueError("Stage 1 best checkpoint selection record differs")
    weights = cfg["train"]["validation"]["weights"]
    if best.get("validation_weights") != weights:
        raise ValueError("Stage 1 selected weight type differs from validation")
    load_generator(model, best, weights=weights)
    if ema is not None:
        # Start the new phase's EMA from the weights actually used to initialize it.
        ema.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
    parent = dict(checkpoint=os.path.basename(path) if memory_checkpoint is None else None, epoch=selection["epoch"],
                  monitor=cfg["train"]["validation"]["monitor"],
                  value=selection["best"], weights=weights)
    if memory_checkpoint is not None:
        parent["storage"] = "memory"
    return best, parent


class EMA:
    """Optional new EMA, independent of Adam; disabled in the default protocol."""
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between 0 and 1")
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1-self.decay)
            else:
                self.shadow[key].copy_(value)
