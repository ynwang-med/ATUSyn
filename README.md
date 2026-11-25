# ATUSyn

**Anatomy- and Text-Guided Ultrasound Synthesis**

Implementation for *Beyond Pixel Alignment: Anatomy- and Text-Guided Diffusion for CT/MRI-to-Ultrasound Synthesis*.

## Release Status

The author reports that the single-GPU EMA/SSIM experiment runs end to end. The public configuration below preserves that experiment's training settings and replaces machine-specific resource paths with relative paths. Final checkpoint identities, dependency versions and agreement with the final manuscript still require release verification. This is not a claim that all historical experiments used this protocol.

No medical images, private local configurations or trained weights are included in the code package. Licensing of the authors' contributions is still pending; the root LICENSE is a review notice, not a blanket permissive license.

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

Do not substitute different encoders without recording a changed experiment. Exact checkpoint hashes remain to be published.

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

The current prepared MAT cohort reported by the author has 25,112 training slices / 55 patients, 1,344 validation slices / 2 patients, and 5,376 test slices / 8 patients. Patient roles are defined by `splits/ReMIND/split.csv`; filename-list order must exactly match MAT sample order. The split contains public dataset case IDs, not names or hospital identifiers.

Inputs are prepared 256 x 256 grayscale arrays. The release config uses HDF5 MAT `image_data`, legacy axes, uint8 image normalization to [-1,1], and label IDs normalized with maximum 63. There is no new resizing in the loader. Raw-to-prepared preprocessing is not fully packaged.

For legacy 3D arrays, the sample axis is 0 and each selected slice's spatial axes are transposed. For legacy 4D arrays, the sample axis is 3 and the remaining axes are reordered as (2,0,1), producing C,H,W. Each source/US/label slice must become 1 x 256 x 256. Source/US values are normalized as x/127.5-1; raw integer label IDs as 2*label/63-1. Filename-list line order must match the MAT sample axis, not a newly sorted image list. The loader does not generate labels or infer their mapping from raw SynthSeg outputs.

Build the manifest once:
```bash
python scripts/prepare_mat_manifest.py --config configs/remind.release.yaml
```

This validates shapes, counts and patient partitions and writes `data/remind_mat_manifest.csv`, not medical images. Existing manifests are not overwritten. Do not change the manifest during training or resume. RAM caching loads stored arrays once; each process has its own cache.

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

Do not overwrite prior predictions. Real/fake relative filenames must match. PSNR/SSIM use saved uint8 grayscale images, data_range=1 after conversion to [0,1], slice means and population standard deviations. USFM scores are not Inception FID/KID. KID subset standard deviation is not a patient-level confidence interval.

Optional `--interslice` reports volume and patient macro means separately. `--directions configs/directions.local.json` supplies case-specific views. Consecutive indices are required unless `--allow-index-gaps` is explicitly selected. The implementation in `atusyn/metrics/interslice.py` uses global image SSIM, not windowed skimage SSIM. For each volume/view, gaps compare the mean real and mean generated adjacent-slice statistics; they are not means of pairwise absolute gaps.

## Third-Party Resources and Licenses

| Resource | Official source |
| --- | --- |
| DDGAN / NCSNpp-derived code | [NVIDIA DDGAN](https://github.com/NVlabs/denoising-diffusion-gan) |
| Score-model layers | [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) |
| Text/image encoder | [OpenCLIP](https://github.com/mlfoundations/open_clip) |
| Feature image encoder | [MedSAM](https://github.com/bowang-lab/MedSAM) |
| Evaluation encoder | [USFM](https://github.com/openmedlab/USFM) |
| Upstream anatomy preparation | [SynthSeg](https://github.com/BBillot/SynthSeg) |

Retained license texts are in [LICENSES](LICENSES/) and [operator LICENSE_MIT](atusyn/models/op/LICENSE_MIT); original copyright headers are preserved. Inherited NVIDIA restrictions must be considered before choosing a license for the authors' additions. The root LICENSE remains a review notice, not a blanket license grant. Check code, pretrained-weight and dataset terms separately; source-code permission alone does not authorize redistribution of weights or images.

