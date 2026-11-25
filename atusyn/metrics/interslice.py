"""Legacy global-SSIM definition, not windowed skimage SSIM."""
import re
from collections import defaultdict
import numpy as np


def global_ssim(a, b):
    a,b=a.astype(np.float64),b.astype(np.float64)
    ma,mb=a.mean(),b.mean()
    va,vb=a.var(),b.var()
    cov=((a-ma)*(b-mb)).mean()
    return float(((2*ma*mb+0.01**2)*(2*cov+0.03**2))/
                 ((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2)))


def interslice(pairs, loader, consecutive=True, directions=None):
    groups=defaultdict(dict)
    for name,(real,fake) in pairs.items():
        match=re.match(r"(.+?)[_-](\d+)\.[^.]+$",name)
        if not match:
            raise ValueError("No final slice index: "+name)
        case=re.search(r"Case\d+",name).group()
        if directions is not None:
            if case not in directions:
                raise ValueError("Missing direction for "+case)
            if "_"+directions[case]+"_" not in name:
                continue
        volume,index=match[1],int(match[2])
        if index in groups[volume]:
            raise ValueError("Duplicate volume/slice index")
        groups[volume][index]=(real,fake,case)
    output=[]
    for volume,slices in sorted(groups.items()):
        indices=sorted(slices)
        rr,ff,rm,fm=[],[],[],[]
        for a,b in zip(indices[:-1],indices[1:]):
            if consecutive and b!=a+1:
                continue
            ra,fa,_=slices[a];rb,fb,_=slices[b]
            ra,fa,rb,fb=[loader(p).astype(np.float64)/255 for p in (ra,fa,rb,fb)]
            rr.append(global_ssim(ra,rb));ff.append(global_ssim(fa,fb))
            rm.append(float(np.abs(ra-rb).mean()));fm.append(float(np.abs(fa-fb).mean()))
        if rr:
            output.append(dict(volume=volume,case_id=slices[indices[0]][2],n_pairs=len(rr),
                               ssim_gap=abs(float(np.mean(rr)-np.mean(ff))),
                               mae_gap=abs(float(np.mean(rm)-np.mean(fm)))))
    if not output:
        raise ValueError("No adjacent slice pairs")
    bycase=defaultdict(list)
    for row in output:bycase[row["case_id"]].append(row)
    summary={"n_volumes":len(output),"n_pairs":sum(r["n_pairs"] for r in output)}
    for metric in ("ssim_gap","mae_gap"):
        summary["volume_macro_"+metric]=float(np.mean([r[metric] for r in output]))
        summary["patient_macro_"+metric]=float(np.mean([
            np.mean([r[metric] for r in rows]) for rows in bycase.values()]))
    return output,summary
