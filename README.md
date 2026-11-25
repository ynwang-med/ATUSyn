# ATUSyn

**Anatomy- and Text-Guided Ultrasound Synthesis**

This repository contains the implementation of **“Beyond Pixel Alignment: Anatomy- and Text-Guided Diffusion for CT/MRI-to-Ultrasound Synthesis.”**

## Method

ATUSyn uses an NCSNpp-based diffusion generator. The network receives a noisy ultrasound image, the source CT/MRI image, and an anatomical label map as three input channels. Anatomical-label-derived text prompts are encoded using a frozen OpenCLIP model. The implementation retains text/time conditioning and spatial cross-attention at the bottleneck. Frozen CLIP and MedSAM models provide feature-based supervision during training according to the configured loss schedule.

## Repository and reproducibility notes

- Run commands from the **repository root**; paths in the commands below are relative to that directory.
- Despite the `.yaml` extension, the release configuration uses **JSON-style syntax with optional full-line comments**, not general YAML syntax.
- The repository contains the training, inference, and evaluation entry points for the ReMIND release. It **does not yet package a complete conversion pipeline from the original TCIA ReMIND data to the prepared MAT inputs** used by the scripts.
- `requirements.txt` is a **candidate legacy Python 3.10 environment**, not an environment lockfile exported from the validated training server. Exact environment reproduction has not yet been confirmed.

## Environment setup

Create and activate a Conda environment with Python 3.10:

```bash
conda create -n atusyn python=3.10 -y
conda activate atusyn
```

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, make sure your PyTorch build is compatible with the installed NVIDIA driver and CUDA environment (see [PyTorch's installation guidance](https://pytorch.org/get-started/previous-versions/)). The inherited CUDA operator requires a compatible CUDA toolkit, C++ compiler, and Ninja, and may compile on first CUDA use. The required `segment_anything` source is included in the repository.

**Note:** `requirements.txt` is a candidate legacy Python 3.10 dependency list rather than a lockfile exported from the validated training server. The commands above are a starting point, **not a fully validated reproduction environment**. If dependency resolution or CUDA compilation fails, align PyTorch and CUDA versions with the target machine; a tested environment specification is still needed.

## Data and pretrained resources

### ReMIND dataset and splits

Download the original ReMIND dataset from the [official TCIA ReMIND collection](https://www.cancerimagingarchive.net/collection/remind/). Follow the collection's access, use, and citation requirements. **No medical images are redistributed in this repository.**

The intended patient-level train/validation/test split is documented at [`splits/ReMIND/split.csv`](splits/ReMIND/split.csv).  The original downloaded files cannot be passed directly to the training scripts: they must first be transformed into the prepared MAT structure shown below. 

Place the prepared files under:

```text
data/remind_mat/
├── data_train_MRI.mat
├── data_train_US.mat
├── data_train_label.mat
├── filenames_train.txt
├── data_val_MRI.mat
├── data_val_US.mat
├── data_val_label.mat
├── filenames_val.txt
├── data_test_MRI.mat
├── data_test_US.mat
├── data_test_label.mat
└── filenames_test.txt
```

### Pretrained ATUSyn weights — brain

The released **brain MRI-to-ultrasound** checkpoint is available here:

| Dataset | Model | Download |
| --- | --- | --- |
| ReMIND (brain) | ATUSyn | [Google Drive — brain checkpoint](https://drive.google.com/file/d/13hc4vU_fLR_KkpcUTjf1xpFOcrxrSTR_/view?usp=drive_link) |

Download the file and place it in `checkpoints/`. 

### Additional model resources

Obtain the following third-party model weights separately from their respective official sources, and place them at the paths expected by the configuration:

| Resource | Expected local path | Purpose |
| --- | --- | --- |
| OpenCLIP ViT-H-14 (`laion2b_s32b_b79k`) | `checkpoints/open_clip_pytorch_model.bin` | Text conditioning / CLIP-based supervision |
| MedSAM ViT-B | `checkpoints/medsam_vit_b.pth` | MedSAM-based supervision |
| USFM (optional) | `checkpoints/USFM_latest.pth` | USFM-based evaluation |

For optional USFM evaluation, the corresponding code is expected at `third_party/USFM`. These third-party weights are **not included** in the brain checkpoint download link above.

## Training

The released training configuration is `configs/remind.release.yaml`. The generator is initialized from scratch, while OpenCLIP and MedSAM use frozen pretrained weights. The supplied configuration specifies the staged loss schedule, batch sizes, learning-rate schedule, and stopping settings. The existing training documentation uses Adam with an initial learning rate of `1e-4`.

Start training (Linux):

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train.py \
  --config configs/remind.release.yaml --device cuda:0 \
  --output runs/remind_release > logs/remind_release.log 2>&1 &
```

Resume from a saved training checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --config configs/remind.release.yaml --device cuda:0 \
  --output runs/remind_release --resume runs/remind_release/last.pt
```

## Inference

The original release instructions select **EMA** weights for inference. The CLI defaults to raw model weights, so `--weights ema` is specified explicitly. To run inference with the downloaded brain checkpoint, use:

```bash
python scripts/inference.py --config configs/remind.release.yaml \
  --checkpoint checkpoints/ATUSyn_brain.pth --weights ema \
  --device cuda:0 --split test --seed 1024 \
  --output runs/remind_release/test
```

**Checkpoint compatibility check:** This command assumes the downloadable checkpoint uses the same saved structure as the release training checkpoints. If it is instead a standalone state dictionary, the loading code or packaging instructions will need to be adjusted. 

## Evaluation

Evaluate the saved grayscale images:

```bash
python scripts/evaluate.py \
  --real runs/remind_release/test/real \
  --fake runs/remind_release/test/fake \
  --output runs/remind_release/metrics
```

Optional USFM-FID and USFM-KID evaluation (requires separately acquired USFM code and weights):

```bash
python scripts/evaluate.py \
  --real runs/remind_release/test/real \
  --fake runs/remind_release/test/fake \
  --output runs/remind_release/metrics_usfm --device cuda:0 \
  --usfm-repo third_party/USFM \
  --usfm-checkpoint checkpoints/USFM_latest.pth \
  --kid-subset-size 1000 --kid-subsets 100
```

## Acknowledgements and third-party code

This implementation was developed by adapting and extending the open-source [SynDiff](https://github.com/icon-lab/SynDiff) project. We thank its authors for making the implementation available. SynDiff is distributed under the **MIT License**; retain its original copyright and permission notice for any SynDiff code included in this repository. 

If you use this implementation, please cite the ATUSyn paper and acknowledge the original SynDiff work where appropriate.

## Citation

**ATUSyn:** *Beyond Pixel Alignment: Anatomy- and Text-Guided Diffusion for CT/MRI-to-Ultrasound Synthesis*. A complete BibTeX entry can be added once the final publication metadata are available.

**SynDiff:** Özbey et al., *Unsupervised Medical Image Translation With Adversarial Diffusion Models*, IEEE Transactions on Medical Imaging, 2023. [DOI](https://doi.org/10.1109/TMI.2023.3290149).

## License

See the repository's [`LICENSE`](LICENSE) file for the applicable license terms. Original third-party copyright and license notices must be preserved where required. The source-code license does **not** grant rights to redistribute the ReMIND images or other third-party model weights.
