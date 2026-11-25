"""Build a filename manifest only. Never reads pixels or changes source images."""
import argparse
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atusyn.data.index import image_manifest,read_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--split",default="splits/ReMIND/split.csv")
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    rows=image_manifest(Path(a.root),a.split)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    indexed,missing=read_manifest(a.root,out,a.split)
    for role in ("train","val","test"):
        group=[r for r in indexed if r["split"]==role]
        print(role,len(group),"slices",len({r["case_id"] for r in group}),"patients")
    print("Absent cases:",missing)


if __name__=="__main__":main()
