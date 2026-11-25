"""Split existing ReMIND evaluation MAT files without changing pixel values."""
import argparse
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atusyn.data.index import case_id, read_split

DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "splits/ReMIND/split.csv"


def inspect_inputs(input_dir, split_csv, variable):
    import h5py
    root = Path(input_dir).resolve()
    names = (root / "filenames_test.txt").read_text(encoding="utf-8-sig").splitlines()
    if not names or any(not n or n != n.strip() for n in names):
        raise ValueError("Filename list is empty or contains blank/padded lines")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filenames")
    split = read_split(split_csv)
    indices = {"val": [], "test": []}
    for i, name in enumerate(names):
        case = case_id(name)
        role = split.get(case)
        if role not in indices:
            raise ValueError("Evaluation input contains training/unknown patient: " + case)
        indices[role].append(i)
    if any(not rows for rows in indices.values()):
        raise ValueError("Both validation and test samples must be present")
    info = {}
    common_shape = None
    for modality in ("MRI", "US", "label"):
        path = root / ("data_test_" + modality + ".mat")
        with h5py.File(path, "r") as f:
            if set(f.keys()) != {variable} or not isinstance(f[variable], h5py.Dataset):
                raise ValueError("Expected a single numeric dataset: " + str(path))
            a = f[variable]
            if a.ndim not in (3, 4) or a.dtype.kind not in "buifc":
                raise ValueError("Expected numeric 3D/4D legacy MAT array: " + str(path))
            axis = 0 if a.ndim == 3 else 3
            if a.shape[axis] != len(names):
                raise ValueError("MAT sample count differs from filenames: " + str(path))
            if common_shape is not None and a.shape != common_shape:
                raise ValueError("MRI/US/label array shapes differ")
            common_shape = a.shape
            info[modality] = dict(shape=list(a.shape), dtype=str(a.dtype), sample_axis=axis)
    summary = {
        role: dict(n_samples=len(rows),
                   case_ids=sorted({case_id(names[i]) for i in rows}))
        for role, rows in indices.items()
    }
    return root, names, indices, info, summary


def split_files(input_dir, output_dir, split_csv=DEFAULT_SPLIT,
                variable="image_data", block_size=None, dry_run=False):
    import h5py
    import numpy as np

    if block_size is not None and block_size < 1:
        raise ValueError("block_size must be positive")
    root, names, indices, info, summary = inspect_inputs(input_dir, split_csv, variable)
    output = Path(output_dir).resolve()
    if output == root or output in root.parents:
        raise ValueError("Output must not be the input directory or its ancestor")
    if output.exists():
        raise FileExistsError("Choose a new output directory: " + str(output))
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary

    # Partial outputs are retained after interruption; never overwrite them on retry.
    output.mkdir(parents=True, exist_ok=False)
    for role in indices:
        (output / role).mkdir()
    for modality in ("MRI", "US", "label"):
        source_path = root / ("data_test_" + modality + ".mat")
        paths = {}
        with ExitStack() as stack:
            source = stack.enter_context(h5py.File(source_path, "r"))
            a = source[variable]
            axis = info[modality]["sample_axis"]
            header_size = source.userblock_size
            datasets = {}
            for role, rows in indices.items():
                path = output / role / ("data_" + role + "_" + modality + ".mat")
                paths[role] = path
                kwargs = {"userblock_size": header_size} if header_size else {}
                dest = stack.enter_context(h5py.File(path, "x", **kwargs))
                for key, value in source.attrs.items():
                    dest.attrs[key] = value
                shape = list(a.shape)
                shape[axis] = len(rows)
                chunks = list(shape)
                chunks[axis] = 1
                datasets[role] = dest.create_dataset(
                    variable, shape=shape, dtype=a.dtype, chunks=tuple(chunks),
                    compression="gzip", compression_opts=4)
                for key, value in a.attrs.items():
                    datasets[role].attrs[key] = value

            sample_bytes = a.dtype.itemsize * int(np.prod(a.shape) // len(names))
            max_samples = max(1, (64 * 1024 * 1024) // sample_bytes)
            step = block_size or min(a.chunks[axis] if a.chunks else 64, max_samples)
            selected = {role: np.asarray(rows) for role, rows in indices.items()}
            written = {role: 0 for role in indices}
            for start in range(0, len(names), step):
                end = min(start + step, len(names))
                selection = [slice(None)] * a.ndim
                selection[axis] = slice(start, end)
                block = a[tuple(selection)]
                for role, rows in selected.items():
                    local = rows[(rows >= start) & (rows < end)] - start
                    if not len(local):
                        continue
                    selection[axis] = slice(written[role], written[role] + len(local))
                    datasets[role][tuple(selection)] = np.take(block, local, axis=axis)
                    written[role] += len(local)
            if any(written[role] != len(rows) for role, rows in indices.items()):
                raise RuntimeError("Incomplete MAT export")
        # Preserve the MATLAB user block when the input has one.
        if header_size:
            with source_path.open("rb") as f:
                header = f.read(header_size)
            for path in paths.values():
                with path.open("r+b") as f:
                    f.write(header)
        print("[DONE] " + modality, flush=True)

    for role, rows in indices.items():
        with (output / role / ("filenames_" + role + ".txt")).open(
                "x", encoding="utf-8", newline="\n") as f:
            f.writelines(names[i] + "\n" for i in rows)
    with (output / "split_index.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "new_index", "original_index", "case_id", "filename"])
        for role, rows in indices.items():
            for new_index, original_index in enumerate(rows):
                name = names[original_index]
                writer.writerow([role, new_index, original_index, case_id(name), name])
    with (output / "split_summary.json").open("x", encoding="utf-8") as f:
        json.dump(dict(status="complete", variable=variable, arrays=info, splits=summary,
                       normalization="none", index_base=0), f, indent=2)
    print("[DONE] " + str(output), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Mixed data_test_*.mat and filenames_test.txt")
    p.add_argument("--output-dir", required=True, help="New directory containing val/ and test/")
    p.add_argument("--split-csv", default=str(DEFAULT_SPLIT))
    p.add_argument("--mat-variable", default="image_data")
    p.add_argument("--block-size", type=int, help="Samples per read; default follows input chunks with a 64 MiB cap")
    p.add_argument("--dry-run", action="store_true", help="Check filenames/MAT metadata only; write nothing")
    args = p.parse_args()
    split_files(args.input_dir, args.output_dir, args.split_csv,
                args.mat_variable, args.block_size, args.dry_run)


if __name__ == "__main__":
    main()

