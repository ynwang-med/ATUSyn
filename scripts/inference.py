import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/remind.release.yaml")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--weights",choices=("raw","ema"),default="raw")
    p.add_argument("--split",choices=("test","val"),default="test")
    p.add_argument("--seed",type=int,default=1024)
    a=p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from atusyn.config import load_config,model_args
    from atusyn.data.dataset import ReMINDDataset
    from atusyn.models.backbones.ncsnpp_generator_adagn_dual_text import NCSNpp
    from atusyn.models.encoders import FrozenOpenCLIPEmbedder
    from atusyn.models.prompts import build_prompt_brain_dynamic
    from atusyn.models.diffusion import Posterior_Coefficients
    from atusyn.engine.checkpoint import load_file,load_generator
    from atusyn.engine.sampling import sample
    from atusyn.metrics.images import to_uint8
    from atusyn.data.index import safe_path
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    cfg=load_config(a.config);args=model_args(cfg);device=torch.device(a.device)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if device.type=="cuda":
        torch.cuda.set_device(device);torch.cuda.manual_seed(a.seed)
    data=ReMINDDataset(cfg,a.split)
    model=NCSNpp(args).to(device)
    ckpt=load_file(a.checkpoint)
    if "config" in ckpt:
        for k in ("model","diffusion"):
            if cfg[k]!=ckpt["config"][k]:raise ValueError("Checkpoint configuration mismatch: "+k)
        for key in ("max_value", "min_pixels"):
            if cfg["text"][key]!=ckpt["config"]["text"][key]:
                raise ValueError("Text/label conditioning mismatch: "+key)
    load_generator(model,ckpt,a.weights);model.eval()
    clip=FrozenOpenCLIPEmbedder(cfg["text"]["clip_checkpoint"],device=str(device)).to(device).eval()
    posterior=Posterior_Coefficients(args,device)
    # Model/encoder initialization must not alter the sampling seed.
    torch.manual_seed(a.seed)
    with torch.no_grad():
        for batch in DataLoader(data,batch_size=1,shuffle=False):
            if Path(batch["filename"][0]).suffix.lower() != ".png":
                raise ValueError("Output filenames must end in .png to preserve lossless quantization")
            source,label=batch["source"].to(device),batch["label"].to(device)
            text=clip(build_prompt_brain_dynamic(label,max_value=cfg["text"]["max_value"],min_pixels=cfg["text"]["min_pixels"]))
            fake=sample(model,posterior,source,label,text,args)
            for category,value in (("fake",fake[0,0]),("real",batch["target"][0,0])):
                path=safe_path(out/category,batch["filename"][0]);path.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(to_uint8(value)).save(path)
    (out/"inference.json").write_text(json.dumps(dict(checkpoint=str(a.checkpoint),weights=a.weights,
        seed=a.seed,split=a.split,steps=args.num_timesteps,n_images=len(data)),indent=2),encoding="utf-8")


if __name__=="__main__":main()
