import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atusyn.config import load_config


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/remind.release.yaml")
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    start=p.add_mutually_exclusive_group()
    start.add_argument("--resume")
    start.add_argument("--stage1-run", help="New Stage 2 fork from a stopped Stage 1 run's best checkpoint")
    a=p.parse_args()
    from atusyn.engine.train import train
    train(load_config(a.config),a.output,a.device,a.resume,stage1_run=a.stage1_run)


if __name__=="__main__":main()
