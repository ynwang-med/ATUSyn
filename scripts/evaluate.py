import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--real",required=True);p.add_argument("--fake",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--usfm-repo");p.add_argument("--usfm-checkpoint")
    p.add_argument("--kid-subset-size",type=int,default=1000)
    p.add_argument("--kid-subsets",type=int,default=100)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--interslice",action="store_true")
    p.add_argument("--directions",help="Optional JSON mapping public Case ID to axial/coronal/sagittal")
    p.add_argument("--allow-index-gaps",action="store_true")
    a=p.parse_args()
    import numpy as np
    from PIL import Image
    from atusyn.metrics.images import image_metrics,summarize
    from atusyn.metrics.interslice import interslice
    def files(root):
        return {p.relative_to(root).as_posix():p for p in Path(root).rglob("*")
                if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")}
    def load(path):
        with Image.open(path) as im:
            if im.mode!="L":raise ValueError("Prepared grayscale uint8 images required")
            return np.array(im)
    real,fake=files(a.real),files(a.fake)
    if not real or real.keys()!=fake.keys():
        raise ValueError("Real/fake relative filenames must match exactly")
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    rows=[dict(filename=k,**image_metrics(load(fake[k]),load(real[k]))) for k in sorted(real)]
    result=summarize(rows)
    out.mkdir(parents=True,exist_ok=True)
    def write(name,rows):
        with (out/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write("per_slice.csv",rows)
    if bool(a.usfm_repo)!=bool(a.usfm_checkpoint):
        raise ValueError("Supply both USFM repo and checkpoint")
    if a.usfm_repo:
        from atusyn.metrics.usfm import build_usfm_encoder,extract_features,compute_fid,compute_kid
        if a.kid_subset_size<2 or a.kid_subsets<2 or len(real)<2:
            raise ValueError("FID/KID requires >=2 images and >=2 KID subsets")
        model=build_usfm_encoder(a.usfm_repo,a.usfm_checkpoint,a.device)
        r=extract_features(a.real,model,a.device);f=extract_features(a.fake,model,a.device)
        result["usfm_fid"]=compute_fid(r,f)
        result["usfm_kid"],result["usfm_kid_std"]=compute_kid(
            r,f,a.kid_subset_size,a.kid_subsets,a.seed)
    if a.interslice:
        directions=json.loads(Path(a.directions).read_text()) if a.directions else None
        vol,summary=interslice({k:(real[k],fake[k]) for k in real},load,
                              not a.allow_index_gaps,directions)
        write("per_volume.csv",vol);result.update(summary)
    result["aggregation"]="PSNR/SSIM: slice mean, population std; interslice: explicitly named macro means"
    (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"protocol.json").write_text(json.dumps(vars(a),indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
