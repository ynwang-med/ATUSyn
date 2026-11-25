"""Explicit CPU-only protocol migration; never changes source checkpoints."""
import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atusyn.config import load_config
from atusyn.engine.checkpoint import atomic_save, load_file
from atusyn.engine.selection import Selection, early_stopping_for


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_equal(a, b, path="checkpoint"):
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor):
        equal = isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        equal = isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("State changed: " + path)
        for key in a:
            assert_equal(a[key], b[key], path + "." + str(key))
        return
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("State changed: " + path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_equal(x, y, path + "." + str(i))
        return
    else:
        equal = a == b
    if not equal:
        raise ValueError("State changed: " + path)


def replay(rows, cfg):
    selection = Selection(cfg["train"]["validation"]["mode"])
    early = early_stopping_for(cfg, 1)
    snapshots, first_stop = {}, None
    monitor = cfg["train"]["validation"]["monitor"]
    for row in rows:
        value = row[monitor]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Non-finite/missing validation metric in history")
        selection.update(value, row["epoch"])
        if early.update(value, row["epoch"]) and first_stop is None:
            first_stop = row["epoch"]
        snapshots[row["epoch"]] = (copy.deepcopy(selection.state_dict()), copy.deepcopy(early.state_dict()))
    return snapshots, first_stop


def migrate(source_run, epoch, cfg, output, check_only=False):
    source, output = Path(source_run).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be a separate run directory, not inside/above source")
    if output.exists():
        raise FileExistsError("Output already exists; nothing overwritten: " + str(output))
    source_path = source/("epoch_%04d.pt" % epoch)
    best_path, history_path = source/"best_stage1.pt", source/"history.jsonl"
    hashes = {p.name: file_hash(p) for p in (source_path, best_path, history_path)}
    checkpoint, best = load_file(source_path), load_file(best_path)
    old_cfg = checkpoint["config"]
    tr = cfg["train"]
    if not old_cfg["train"].get("stagewise_early_stopping") or not tr.get("stagewise_early_stopping"):
        raise ValueError("Only stagewise early-stopping runs can be migrated")
    if old_cfg["train"]["early_stopping"].get("warmup_reference", "history") != "history":
        raise ValueError("Source already uses a different/reset reference")
    expected = copy.deepcopy(old_cfg)
    expected["train"]["early_stopping"]["warmup_reference"] = "end"
    if expected != cfg:
        raise ValueError("Only warmup_reference may change; keep all other config fields identical")
    if tr["validation"]["monitor"] != "ssim" or tr["validation"]["weights"] != "ema" or not tr["ema"]["enabled"]:
        raise ValueError("This migration requires the EMA/SSIM protocol")
    warmup = tr["early_stopping"]["stage1_warmup_epochs"]
    if not warmup <= epoch < tr["stage1_epochs"]:
        raise ValueError("Resume epoch must be at/after warmup and below the Stage 1 cap")
    for obj in (checkpoint, best):
        if obj.get("stage") != 1 or obj.get("stage2_parent") or obj.get("stage2_fork"):
            raise ValueError("Only actual Stage 1 checkpoints are accepted; never Stage 2 last.pt")
        if obj.get("protocol_migration"):
            raise ValueError("Source has already been migrated")
        if obj.get("config") != old_cfg or obj.get("validation_weights") != "ema":
            raise ValueError("Source checkpoint config/selected weights do not match")
        for field in ("model", "ema", "optimizer", "scheduler", "rng", "selection", "early_stopping", "data_identity"):
            if obj.get(field) is None:
                raise ValueError("Incomplete training state: " + field)
        if len(obj["rng"]) != obj["world_size"]:
            raise ValueError("Incomplete per-rank RNG states")
    if checkpoint["completed_epoch"] != epoch:
        raise ValueError("Filename and completed epoch disagree")
    if best["world_size"] != checkpoint["world_size"]:
        raise ValueError("Best/source world sizes disagree")
    identity = {key: file_hash(cfg["data"][key]) for key in ("manifest", "split")}
    if checkpoint["data_identity"] != identity or best["data_identity"] != identity:
        raise ValueError("Split/manifest hashes differ; refusing migration")
    rows = []
    for line in history_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["epoch"] <= epoch:
            if row["stage"] != 1:
                raise ValueError("History contains Stage 2 within the requested Stage 1 prefix")
            rows.append(row)
    if [r["epoch"] for r in rows] != list(range(1, epoch+1)):
        raise ValueError("History must contain exactly epochs 1..resume_epoch in order, no gaps/duplicates")
    old_states, old_stop = replay(rows, old_cfg)
    new_states, new_stop = replay(rows, cfg)
    if old_stop is not None and old_stop < epoch:
        raise ValueError("Source history continued past its recorded stopping rule")
    if new_stop is not None:
        raise ValueError("Corrected rule would already have stopped; refusing forced continuation")
    if old_states[epoch][0] != checkpoint["selection"]:
        raise ValueError("History does not reproduce source selection")
    old_early = dict(checkpoint["early_stopping"])
    old_early.setdefault("warmup_reference", "history")
    if old_states[epoch][1] != old_early:
        raise ValueError("History does not reproduce source early-stopping state")
    winner = checkpoint["selection"]["epoch"]
    if best["completed_epoch"] != winner or best["selection"] != checkpoint["selection"]:
        raise ValueError("best_stage1.pt does not match the winner at the requested resume epoch")
    for obj, index in ((checkpoint, epoch), (best, winner)):
        if obj["validation"]["ssim"] != rows[index-1]["ssim"]:
            raise ValueError("Checkpoint and history SSIM disagree")
    provenance = dict(kind="warmup_reference_history_to_end", source_run=str(source),
                      source_checkpoint=source_path.name, source_hashes=hashes,
                      resumed_epoch=epoch, next_epoch=epoch+1, selected_epoch=winner,
                      warmup_epochs=warmup)
    def corrected(obj):
        result = dict(obj)
        result.update(config=copy.deepcopy(cfg), early_stopping=new_states[obj["completed_epoch"]][1],
                      stage1_completed_epoch=None, stage1_stop_reason=None, stopped_early=False,
                      protocol_migration=provenance)
        return result
    report = dict(provenance, original_config=old_cfg,
                  original_early_stopping=checkpoint["early_stopping"],
                  corrected_early_stopping=new_states[epoch][1], selection=checkpoint["selection"],
                  cleared_stage1_boundary=checkpoint.get("stage1_completed_epoch"),
                  original_stage1_stop_reason=checkpoint.get("stage1_stop_reason"))
    print(json.dumps({k: report[k] for k in ("resumed_epoch", "next_epoch", "selected_epoch", "corrected_early_stopping")}, indent=2), flush=True)
    if check_only:
        print("[CHECK OK] No output files created.", flush=True)
        return report
    output.mkdir(parents=True, exist_ok=False)
    # Publish last.pt only after round-trip verification of both checkpoint files.
    pending = output/"resume_pending.pt"
    changed = {"config", "early_stopping", "stage1_completed_epoch", "stage1_stop_reason",
               "stopped_early", "protocol_migration"}
    for obj, destination in ((checkpoint, pending), (best, output/"best_stage1.pt")):
        updated = corrected(obj)
        atomic_save(updated, destination)
        saved = load_file(destination)
        assert_equal(updated, saved)
        for key in obj:
            if key not in changed:
                assert_equal(obj[key], saved[key], key)
        del saved, updated
    for path in (source_path, best_path, history_path):
        if file_hash(path) != hashes[path.name]:
            raise ValueError("Source changed during migration; last.pt was not published. Stop the source run first.")
    (output/"history.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    (output/"config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    report["state_preservation_verified"] = True
    (output/"protocol_migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pending.rename(output/"last.pt")
    print("[DONE] Verified model/EMA/optimizer/scheduler/RNG and source hashes. Resume: " + str(output/"last.pt"), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", required=True)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    migrate(a.source_run, a.epoch, load_config(a.config), a.output, a.check_only)


if __name__ == "__main__":
    main()
