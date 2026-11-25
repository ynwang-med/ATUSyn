import numpy as np
import torch
from time import perf_counter
from PIL import Image
from torch.utils.data import Dataset
from .index import read_manifest, safe_path


class ReMINDDataset(Dataset):
    def __init__(self, cfg, split):
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        self.cfg, self.split = cfg, split
        self.data = cfg["data"]
        records, self.absent_cases = read_manifest(
            self.data["root"], self.data["manifest"], self.data["split"])
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError("No independent " + split + " samples")
        self.case_ids = sorted({r["case_id"] for r in self.records})
        self._mat_cache = {}
        mode = self.data.get("mat_cache", "none")
        if mode not in ("none", "ram"):
            raise ValueError("mat_cache must be none or ram")
        if self.data["backend"] == "mat" and mode == "ram":
            self._cache_mat_files()

    def _cache_mat_files(self):
        import h5py
        paths = sorted({safe_path(self.data["root"], row[field])
                        for row in self.records for field in ("source", "target", "label")})
        for path in paths:
            start = perf_counter()
            print("[MAT CACHE][{}] Loading {}".format(self.split, path.name), flush=True)
            with h5py.File(path, "r") as f:
                value = f[self.data["mat_variable"]][()]
            layout = self.data["mat_layout"]
            if layout == "legacy" and value.ndim == 3:
                value = value.transpose(0, 2, 1)[:, None]
            elif layout == "legacy" and value.ndim == 4:
                value = value.transpose(3, 2, 0, 1)
            elif layout != "NCHW" or value.ndim != 4:
                raise ValueError("Unsupported MAT shape/layout: " + str(path))
            size = self.cfg["model"]["image_size"]
            if value.shape[1:] != (1, size, size):
                raise ValueError("Preprocess upstream; unexpected shape: " + str(value.shape))
            # Keep the stored dtype, but make each sample contiguous for RAM access.
            cached = np.ascontiguousarray(value)
            cached.setflags(write=False)
            self._mat_cache[path] = cached
            del value
            print("[MAT CACHE][{}] Ready {}, dtype={}, {:.3f} GiB, {:.1f}s".format(
                self.split, path.name, cached.dtype, cached.nbytes/1024**3,
                perf_counter()-start), flush=True)
        total = sum(a.nbytes for a in self._mat_cache.values())/1024**3
        print("[MAT CACHE][{}] Total {:.3f} GiB; samples now read from RAM".format(
            self.split, total), flush=True)

    def __len__(self):
        return len(self.records)

    def _read(self, row, field):
        path = safe_path(self.data["root"], row[field])
        if self.data["backend"] == "images":
            with Image.open(path) as im:
                if im.mode not in ("L", "I", "I;16", "F"):
                    raise ValueError("Expected grayscale image, got " + im.mode)
                value = np.array(im, dtype=np.float32)[None]
        elif self.data["backend"] == "mat":
            i = int(row["index"])
            if i < 0:
                raise ValueError("MAT index must be nonnegative")
            if path in self._mat_cache:
                # Own the float buffer so returned tensors cannot mutate the cache.
                value = np.array(self._mat_cache[path][i], dtype=np.float32, copy=True)
            else:
                import h5py
                with h5py.File(path, "r") as f:
                    a = f[self.data["mat_variable"]]
                    if self.data["mat_layout"] == "NCHW":
                        value = np.asarray(a[i], dtype=np.float32)
                    elif self.data["mat_layout"] == "legacy" and a.ndim == 3:
                        value = np.asarray(a[i], dtype=np.float32).T[None]
                    elif self.data["mat_layout"] == "legacy" and a.ndim == 4:
                        value = np.asarray(a[:, :, :, i], dtype=np.float32).transpose(2, 0, 1)
                    else:
                        raise ValueError("Unsupported MAT shape/layout")
        else:
            raise ValueError("backend must be images or mat")
        if value.shape != (1, self.cfg["model"]["image_size"], self.cfg["model"]["image_size"]):
            raise ValueError("Preprocess upstream; unexpected shape: " + str(value.shape))
        if field == "label":
            mode = self.data["label_encoding"]
            if mode == "raw_ids":
                if not np.allclose(value, np.rint(value)):
                    raise ValueError("Non-integer label IDs")
                value = value/self.cfg["text"]["max_value"]*2-1
            elif mode != "minus_one_one":
                raise ValueError("Invalid label_encoding")
        else:
            mode = self.data["image_range"]
            if mode == "uint8":
                value = value/127.5-1
            elif mode == "zero_one":
                value = value*2-1
            elif mode != "minus_one_one":
                raise ValueError("Invalid image_range")
        if not np.isfinite(value).all() or value.min() < -1.00001 or value.max() > 1.00001:
            raise ValueError("Invalid range for " + str(path))
        return torch.from_numpy(np.ascontiguousarray(value))

    def __getitem__(self, i):
        row = self.records[i]
        result = {k: self._read(row, k) for k in ("source", "target", "label")}
        result.update(filename=row["filename"], case_id=row["case_id"])
        return result
