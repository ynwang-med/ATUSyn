import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_uint8(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return (np.clip((x+1)/2, 0, 1)*255+0.5).astype(np.uint8)


def image_metrics(pred_u8, real_u8):
    if pred_u8.shape != real_u8.shape or pred_u8.ndim != 2:
        raise ValueError("Expected matched grayscale image shapes")
    a, b = pred_u8.astype(np.float32)/255, real_u8.astype(np.float32)/255
    return {"psnr": float(peak_signal_noise_ratio(b, a, data_range=1.0)),
            "ssim": float(structural_similarity(b, a, data_range=1.0))}


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    out = {"n_slices": len(rows)}
    for key in ("psnr", "ssim"):
        a = np.array([r[key] for r in rows], dtype=np.float64)
        out[key] = float(a.mean())
        out[key+"_std"] = float(a.std(ddof=0))
    return out
