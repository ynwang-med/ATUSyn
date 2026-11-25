"""New automation, not evidence of historical automatic early stopping."""
import math


class Selection:
    def __init__(self, mode="max"):
        self.mode, self.best, self.epoch = mode, None, None

    def update(self, value, epoch):
        if not math.isfinite(value):
            return False
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best, self.epoch = value, epoch
        return improved

    def state_dict(self):
        return dict(mode=self.mode, best=self.best, epoch=self.epoch)

    def load_state_dict(self, state):
        self.__dict__.update(state)


class EarlyStopping:
    def __init__(self, enabled=False, patience=10):
        self.enabled, self.patience = enabled, patience
        self.best, self.best_epoch, self.bad_epochs = None, None, 0

    def update(self, loss, epoch):
        if math.isfinite(loss) and (self.best is None or loss < self.best):
            self.best, self.best_epoch, self.bad_epochs = loss, epoch, 0
        else:
            self.bad_epochs = epoch - (self.best_epoch or 0)
        return self.enabled and self.bad_epochs >= self.patience

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(state)


class StageEarlyStopping:
    """Phase-local patience with an explicit warmup reference policy."""
    def __init__(self, patience=10, warmup_epochs=0, mode="max", warmup_reference="history"):
        self.patience, self.warmup_epochs, self.mode = patience, warmup_epochs, mode
        if warmup_reference not in ("history", "end"):
            raise ValueError("warmup_reference must be history or end")
        self.warmup_reference = warmup_reference
        self.best, self.best_epoch, self.bad_epochs = None, None, 0

    def update(self, value, epoch):
        if self.warmup_reference == "end" and self.warmup_epochs > 0 and epoch == self.warmup_epochs:
            # Reset stopping only; the independent checkpoint selector keeps its global winner.
            self.best = value if math.isfinite(value) else None
            self.best_epoch, self.bad_epochs = epoch, 0
            return False
        improved = math.isfinite(value) and (self.best is None or
            (value > self.best if self.mode == "max" else value < self.best))
        if improved:
            self.best, self.best_epoch = value, epoch
        self.bad_epochs = max(0, epoch - max(self.warmup_epochs, self.best_epoch or 0))
        return epoch > self.warmup_epochs and self.bad_epochs >= self.patience

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__.update(state)


def early_stopping_for(cfg, stage):
    tr = cfg["train"]
    es = tr["early_stopping"]
    if tr.get("stagewise_early_stopping", False):
        return StageEarlyStopping(es["patience"], es["stage1_warmup_epochs"] if stage == 1 else 0,
                                  tr["validation"]["mode"], es.get("warmup_reference", "history"))
    return EarlyStopping(**{k: es[k] for k in ("enabled", "patience")})
