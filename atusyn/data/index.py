"""Patient-level partitioning; no random split and no test-to-validation fallback."""
import csv
import re
from pathlib import Path

ROLES = {"training": "train", "validation": "val", "inference": "test",
         "train": "train", "val": "val", "test": "test"}


def case_id(name):
    match = re.search(r"(Case\d+)", str(name))
    if not match:
        raise ValueError("Cannot infer public ReMIND ID: " + str(name))
    return match.group(1)


def read_split(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if rows and rows[0][:2] == ["case_id", "split"]:
        rows = rows[1:]
    result = {}
    for row in rows:
        if not row or not any(row):
            continue
        if len(row) != 2 or row[1].strip() not in ROLES:
            raise ValueError("Expected case_id,split: " + repr(row))
        key = case_id(row[0])
        if key in result:
            raise ValueError("Duplicate/conflicting patient: " + key)
        result[key] = ROLES[row[1].strip()]
    if not result:
        raise ValueError("Empty split")
    return result


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Path escapes data root: " + relative)
    return path


def read_manifest(root, manifest, split_path):
    split = read_split(split_path)
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    seen, present, source_indices = set(), set(), set()
    for row in rows:
        name = row["filename"]
        if not name or name in seen:
            raise ValueError("Duplicate/empty sample filename: " + name)
        safe_path(root, name)
        seen.add(name)
        case = row["case_id"]
        if case_id(name) != case or case not in split:
            raise ValueError("Unknown/mismatched patient: " + name)
        present.add(case)
        for field in ("source", "target", "label"):
            if not safe_path(root, row[field]).is_file():
                raise FileNotFoundError(row[field])
        identity = (str(safe_path(root, row["source"])), row.get("index", ""))
        if identity in source_indices:
            raise ValueError("The same source sample appears more than once")
        source_indices.add(identity)
        row["split"] = split[case]
    if not rows:
        raise ValueError("Empty manifest")
    missing = sorted(set(split)-present)
    return rows, missing


def image_manifest(root, split_path):
    split = read_split(split_path)
    result, seen = [], set()
    # Historical folder 'test' contains validation AND held-out test patients.
    for folder in ("train", "test"):
        parent = Path(root)/folder
        sources = sorted((parent/"mri").glob("*.png"))
        for source in sources:
            case = case_id(source.name)
            if case not in split:
                raise ValueError("Patient absent from split.csv: " + case)
            if source.name in seen:
                raise ValueError("Duplicate sample across folders: " + source.name)
            seen.add(source.name)
            paths = [source, parent/"us"/source.name, parent/"label2"/source.name]
            for p in paths:
                if not p.is_file():
                    raise FileNotFoundError(p)
            result.append(dict(filename=source.name, case_id=case,
                               source=str(paths[0].relative_to(root)),
                               target=str(paths[1].relative_to(root)),
                               label=str(paths[2].relative_to(root)), index=""))
    if not result:
        raise ValueError("No image triplets found")
    return result
