import torch
from ..models.diffusion import sample_posterior


@torch.no_grad()
def sample(model, coefficients, source, labels, text, args):
    x = torch.randn_like(labels)
    latent = torch.randn(x.size(0), args.nz, device=x.device)
    for i in reversed(range(args.num_timesteps)):
        t = torch.full((x.size(0),), i, dtype=torch.int64, device=x.device)
        x0 = model(torch.cat((x, source, labels), dim=1), t, latent, text_emb=text)
        x = sample_posterior(coefficients, x0[:, [0], :, :], x, t).detach()
    return x
