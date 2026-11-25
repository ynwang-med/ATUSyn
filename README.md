# ATUSyn

**Anatomy- and Text-Guided Ultrasound Synthesis**

Implementation for *Beyond Pixel Alignment: Anatomy- and Text-Guided Diffusion for CT/MRI-to-Ultrasound Synthesis*.


## Method

The NCSNpp generator receives noisy US, source CT/MRI and an anatomical label map as three channels. Label-derived prompts are encoded by frozen OpenCLIP. Both the text-time conditioning path and spatial bottleneck cross-attention are retained. Training incorporates frozen CLIP/MedSAM feature losses according to the configured schedule. The package covers ReMIND; private clinical datasets and separate ablation trainers are not included.

## Environment

Run commands from the repository root. The config files use JSON syntax with optional full-line comments, not general YAML. Paths are relative to the working directory.

`requirements.txt` is a candidate Python 3.10 legacy environment, not a lockfile exported from the successful server. Do not replace a working training environment just to match it. A tested environment specification is still needed before final release.

For a new isolated environment:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Choose matching CUDA wheels using the [PyTorch version instructions](https://pytorch.org/get-started/previous-versions/). The inherited CUDA operator requires a compatible toolkit, compiler and Ninja; compilation occurs on first CUDA use. The repository contains the required `segment_anything` source. See the third-party section below for resource sources and license notices.

## Resources and Data

Download the original ReMIND dataset from the [official TCIA ReMIND collection page](https://www.cancerimagingarchive.net/collection/remind/). Follow the access/download instructions, usage terms and citation guidance provided there. This repository does not redistribute the medical images.

The downloaded original data are not the prepared MAT inputs below. Use `splits/ReMIND/split.csv` for patient assignments and follow the input-format requirements below. The complete raw-to-prepared preprocessing pipeline is not yet packaged.

Obtain pretrained resources separately through their official channels:
- OpenCLIP ViT-H-14 / laion2b_s32b_b79k: `checkpoints/open_clip_pytorch_model.bin`.
- MedSAM ViT-B: `checkpoints/medsam_vit_b.pth`.
- Optional USFM evaluation: repository at `third_party/USFM`, weights at `checkpoints/USFM_latest.pth`.


Place prepared MAT files under the following ignored data directory:
```text
data/remind_mat/
  data_train_MRI.mat
  data_train_US.mat
  data_train_label.mat
  filenames_train.txt
  data_val_MRI.mat
  data_val_US.mat
  data_val_label.mat
  filenames_val.txt
  data_test_MRI.mat
  data_test_US.mat
  data_test_label.mat
  filenames_test.txt
```

## Training

Use `configs/remind.release.yaml`. The generator is initialized from scratch; CLIP and MedSAM use frozen pretrained weights. Training uses Adam (lr=1e-4), EMA (decay=0.999), and validation SSIM for model selection. The configuration specifies the loss schedule, batch sizes, learning-rate schedule and stopping settings.

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train.py \
  --config configs/remind.release.yaml --device cuda:0 \
  --output runs/remind_release > logs/remind_release.log 2>&1 &
```

Resume from an existing saved checkpoint:
```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --config configs/remind.release.yaml --device cuda:0 \
  --output runs/remind_release --resume runs/remind_release/last.pt
```

## Inference and Evaluation

Use the SAME weight type as validation selection. The CLI defaults to raw, so explicitly pass `--weights ema` for this experiment:
```bash
python scripts/inference.py --config configs/remind.release.yaml \
  --checkpoint runs/remind_release/best_stage2.pt --weights ema \
  --device cuda:0 --split test --seed 1024 --output runs/remind_release/test
```

Evaluate saved grayscale images:
```bash
python scripts/evaluate.py \
  --real runs/remind_release/test/real --fake runs/remind_release/test/fake \
  --output runs/remind_release/metrics
```

Add USFM-FID / USFM-KID:
```bash
python scripts/evaluate.py \
  --real runs/remind_release/test/real --fake runs/remind_release/test/fake \
  --output runs/remind_release/metrics_usfm --device cuda:0 \
  --usfm-repo third_party/USFM --usfm-checkpoint checkpoints/USFM_latest.pth \
  --kid-subset-size 1000 --kid-subsets 100
```