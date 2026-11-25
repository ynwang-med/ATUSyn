import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
IMAGE_SIZE = 224
BATCH_SIZE = 64

class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _Logger:
    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print("[WARN]", msg)


class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size=224):
        self.root = Path(root)
        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.files[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.files[idx]}")
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img)


def build_usfm_encoder(repo_path, ckpt_path, device):
    sys.path.insert(0, repo_path)
    from usdsgen.modules.backbone.vision_transformer import build_vit

    model_cfg = AttrDict(
        pretrained=None,
        img_size=IMAGE_SIZE,
        in_chans=3,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_values=0.1,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
        use_mean_pooling=False,
    )

    model = build_vit(model_cfg, _Logger())
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    cleaned = {}
    for key, value in ckpt.items():
        new_key = key
        for prefix in ["module.", "model.", "backbone."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if set(missing) - {"head.weight", "head.bias"} or set(unexpected) - {"mask_token"}:
        raise RuntimeError("USFM backbone mismatch: missing={}, unexpected={}".format(missing, unexpected))
    print(f"[INFO] Loaded USFM weights: {ckpt_path}")
    print(f"[INFO] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[INFO] first missing:", missing[:5])
    if len(unexpected) > 0:
        print("[INFO] first unexpected:", unexpected[:5])

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(folder, model, device):
    ds = ImageFolderDataset(folder, image_size=IMAGE_SIZE)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in {folder}")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feat = model.forward_features(batch)
        if feat.ndim != 2:
            raise ValueError('Expected published pooled BxD USFM features')
        feats.append(feat.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


def compute_fid(real_feats, fake_feats, eps=1e-6):
    mu1 = np.mean(real_feats, axis=0)
    mu2 = np.mean(fake_feats, axis=0)
    sigma1 = np.cov(real_feats, rowvar=False)
    sigma2 = np.cov(fake_feats, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def polynomial_kernel(x, y, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * x.dot(y.T) + coef0) ** degree


def mmd2_unbiased(x, y):
    n = x.shape[0]
    m = y.shape[0]
    k_xx = polynomial_kernel(x, x)
    k_yy = polynomial_kernel(y, y)
    k_xy = polynomial_kernel(x, y)
    xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    return float(xx + yy - 2.0 * k_xy.mean())


def compute_kid(real_feats, fake_feats, subset_size=1000, n_subsets=100, seed=1234):
    n = real_feats.shape[0]
    m = fake_feats.shape[0]
    subset_size = min(subset_size, n, m)
    if subset_size < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_subsets):
        ridx = rng.choice(n, size=subset_size, replace=False)
        fidx = rng.choice(m, size=subset_size, replace=False)
        values.append(mmd2_unbiased(real_feats[ridx], fake_feats[fidx]))
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))
