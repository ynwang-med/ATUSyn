import torch
from .sampling import sample
from ..models.prompts import build_prompt_brain_dynamic
from ..models.diffusion import q_sample_pairs, extract
from ..metrics.images import to_uint8, image_metrics, summarize


@torch.no_grad()
def validate(model, clip, loader, coeff, posterior, args, cfg, device):
    if loader.dataset.split != "val":
        raise ValueError("Checkpoint selection requires the validation partition")
    model.eval()
    rows, loss_sum, count = [], 0.0, 0
    for batch in loader:
        source, target, label = (batch[k].to(device) for k in ("source", "target", "label"))
        prompts = build_prompt_brain_dynamic(label, max_value=cfg["text"]["max_value"],
                                             min_pixels=cfg["text"]["min_pixels"])
        text = clip(prompts)
        pred = sample(model, posterior, source, label, text, args)
        for i, name in enumerate(batch["filename"]):
            values = image_metrics(to_uint8(pred[i,0]), to_uint8(target[i,0]))
            rows.append(dict(filename=name, case_id=batch["case_id"][i], **values))
        # Measuring loss must not change the next image's sampling RNG stream.
        devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            t = torch.randint(1, args.num_timesteps, (source.size(0),), device=device)
            xt, _, eps = q_sample_pairs(coeff, target, t)
            latent = torch.randn(source.size(0), args.nz, device=device)
            x0 = model(torch.cat((xt, source, label),1), t, latent, text_emb=text).clamp(-1,1)
            eps_pred = (xt-extract(coeff.a_s_cum,t,xt.shape)*x0[:,[0]]) / extract(coeff.sigmas_cum,t,xt.shape).clamp_min(1e-4)
            loss_sum += float((eps_pred-eps).square().flatten(1).mean(1).sum())
        count += source.size(0)
    result = summarize(rows)
    result["validation_loss"] = loss_sum/count
    model.train()
    return result, rows
