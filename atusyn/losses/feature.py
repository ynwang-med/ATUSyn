"""Feature objective copied in mathematical form from the fixed-prompt reference."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize


class FeatureLoss(nn.Module):
    def __init__(self, clip, medsam):
        super().__init__()
        self.clip = clip
        self.medsam = medsam
        for model in (clip, medsam):
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)

    def forward(self, generated, real, labels):
        anchor = labels.repeat(1, 3, 1, 1)
        with torch.no_grad():
            a = self.clip.encode(anchor, modality="image")
            b = self.clip.encode(real.repeat(1, 3, 1, 1), modality="image")
            am = self.medsam.image_encoder(resize(anchor, (1024, 1024)))
            bm = self.medsam.image_encoder(resize(real, (1024, 1024)).repeat(1, 3, 1, 1))
        c = self.clip.encode(generated.repeat(1, 3, 1, 1), modality="image")
        cm = self.medsam.image_encoder(resize(generated, (1024, 1024)).repeat(1, 3, 1, 1))
        ds, dc, dsm, dcm = b-a, c-a, bm-am, cm-am
        # Preserve batch-level cosine and the original unregularized denominator.
        def direction(x, y):
            x, y = x.flatten(), y.flatten()
            return 1 - torch.dot(x, y) / (torch.linalg.norm(x)*torch.linalg.norm(y))
        return {"clip_mse": F.mse_loss(dc, ds), "clip_dir": direction(dc, ds),
                "medsam_l1": F.l1_loss(dcm, dsm), "medsam_dir": direction(dcm, dsm)}
