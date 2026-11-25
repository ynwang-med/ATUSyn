"""Index flat train/val/test MAT files in their original filename-list order."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atusyn.config import load_config
from atusyn.data.index import case_id, read_split, safe_path


def build_manifest(cfg):
    import h5py
    data = cfg["data"]
    if data["backend"] != "mat":
        raise ValueError("This command requires data.backend=mat")
    root = Path(data["root"])
    split = read_split(data["split"])
    rows, seen = [], set()
    for role in ("train", "val", "test"):
        names = (root / ("filenames_" + role + ".txt")).read_text(
            encoding="utf-8-sig").splitlines()
        if not names or any(not n or n != n.strip() for n in names):
            raise ValueError("Empty or padded filename lines in " + role)
        files = {field: "data_" + role + "_" + modality + ".mat"
                 for field, modality in (("source", "MRI"), ("target", "US"), ("label", "label"))}
        for filename in files.values():
            with h5py.File(root / filename, "r") as f:
                a = f[data["mat_variable"]]
                if data["mat_layout"] == "legacy" and a.ndim == 4:
                    n, shape = a.shape[3], (a.shape[2], a.shape[0], a.shape[1])
                elif data["mat_layout"] == "legacy" and a.ndim == 3:
                    n, shape = a.shape[0], (1, a.shape[2], a.shape[1])
                elif data["mat_layout"] == "NCHW" and a.ndim == 4:
                    n, shape = a.shape[0], a.shape[1:]
                else:
                    raise ValueError("Unsupported MAT layout: " + filename)
                size = cfg["model"]["image_size"]
                if n != len(names) or tuple(shape) != (1, size, size):
                    raise ValueError("MAT shape/count differs from filename list: " + filename)
        for index, name in enumerate(names):
            safe_path(root, name)
            case = case_id(name)
            if split.get(case) != role:
                raise ValueError("Patient partition mismatch: " + name + " in " + role)
            if name in seen:
                raise ValueError("Duplicate filename across partitions: " + name)
            seen.add(name)
            rows.append(dict(filename=name, case_id=case, **files, index=index))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/remind.yaml")
    a = p.parse_args()
    cfg = load_config(a.config)
    if not cfg["data"]["manifest"]:
        raise ValueError("Set data.manifest in the config")
    out = Path(cfg["data"]["manifest"])
    if out.exists():
        raise FileExistsError("Refusing to overwrite manifest: " + str(out))
    rows = build_manifest(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    split = read_split(cfg["data"]["split"])
    counts = Counter(split[r["case_id"]] for r in rows)
    for role in ("train", "val", "test"):
        cases = {r["case_id"] for r in rows if split[r["case_id"]] == role}
        print(role, counts[role], "slices,", len(cases), "patients")
    print("Absent split patients:", sorted(set(split) - {r["case_id"] for r in rows}))
    print("[DONE]", out)


if __name__ == "__main__":
    main()

