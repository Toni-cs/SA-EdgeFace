# SA-EdgeFace: When More Is Less

<p align="center">
  <em>A Capacity-Aware Study of Enhancement Techniques for Sub-Million-Parameter Face Recognition</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/params-0.93M-orange.svg" alt="Params">
  <img src="https://img.shields.io/badge/LFW-94.68%25-brightgreen.svg" alt="LFW">
</p>

## Overview

SA-EdgeFace is a lightweight face recognition model designed for **surveillance-edge deployment**. Built on ShuffleNetV2 0.5× (0.87M parameters), it achieves **94.68% LFW accuracy** with **11.1ms CPU latency**.

This repository accompanies our IEEE Access paper studying a critical question: **do enhancement techniques (attention, distillation, augmentation) transfer to the sub-million-parameter regime?** Through systematic ablation on CASIA-WebFace → LFW, we find that **three of four enhancement categories hurt** below 1M parameters, and the sole distillation strategy that works is governed by a **temporal threshold** rather than a weight threshold.

### Key Findings

| Technique | Effect (<1M params) | Recommendation |
|-----------|---------------------|----------------|
| Channel attention (ECA) | −0.29% (hurts) | Avoid |
| **Spatial attention** | **+AUC, ROC tail improvement** | **Use** |
| Multi-branch attention | −0.43% (capacity contention) | Avoid |
| KD from scratch | Diverges (NaN) | Avoid |
| **Curriculum KD** (delayed) | **+0.76% (sole accuracy gain)** | **Use** |
| Degradation augmentation | −0.95% (capacity threshold) | Avoid (<2M) |

## Results

### Ablation on LFW (CASIA-WebFace, 30 epochs, 10-fold protocol)

| Config | Description | Acc (%) | AUC | EER (%) | Params (M) |
|--------|-------------|---------|-----|---------|------------|
| C0 | baseline | 94.32 | 0.9860 | 5.85 | 0.866 |
| C1 | +channel attention | 94.03 | 0.9853 | 6.22 | 0.866 |
| C2 | +spatial attention | 94.35 | 0.9880 | 5.78 | 0.874 |
| C3 | +full SAB | 93.92 | 0.9851 | 6.12 | 0.930 |
| **C4** | **+curriculum distill** | **94.68** | **0.9882** | **5.47** | **0.930** |
| C5 | +augmentation | 93.73 | 0.9829 | 6.43 | 0.930 |

### Comparison with Baselines

| Method | Acc (%) | AUC | Params (M) |
|--------|---------|-----|------------|
| EdgeFace 1.0× | 96.55 | 0.9921 | 1.78 |
| InsightFace-s (teacher) | 99.22 | 0.9965 | 5.5 |
| **SA-EdgeFace (ours)** | **94.68** | **0.9882** | **0.93** |

### Efficiency (CPU, 112×112 input)

| Model | Params (M) | FLOPs (G) | Latency (ms) |
|-------|-----------|-----------|--------------|
| C0 baseline (0.5×) | 0.866 | 0.0228 | 8.3 |
| **C4 SA-EdgeFace** | **0.930** | **0.0259** | **11.1** |
| EdgeFace 1.0× | 1.778 | 0.0799 | 9.8 |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# For CUDA 12.8 (RTX 50 series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Data Preparation

1. **CASIA-WebFace** (training): Download and extract to `datasets/raw/casia-webface/{id}/{img}.jpg`
2. **LFW** (evaluation): Download `lfw.tgz` and `pairs.txt` from [LFW official](http://vis-www.cs.umass.edu/lfw/)

```bash
python scripts/prepare_lfw.py --output_dir datasets    # download + verify MD5
python scripts/align_lfw.py                             # align to 112×112
```

### Training

```bash
# Train all 6 ablation configs (CASIA-WebFace, 30 epochs)
python scripts/train_ablation.py \
  --data datasets/raw/casia-webface \
  --save_dir weights/ablation_casia \
  --epochs 30 --batch_size 256 --gpu 0

# Train EdgeFace 1.0× baseline
python scripts/train_edgeface_1x.py --data datasets/raw/casia-webface --gpu 0
```

### Evaluation

```bash
# LFW 10-fold cross-validation (all configs)
python scripts/eval_lfw_full.py

# Single config
python scripts/eval_lfw_single.py --config cfg4

# Efficiency benchmark
python scripts/eval_efficiency.py
```

## Architecture

### Backbone: ShuffleNetV2 0.5×

- 0.866M parameters, 0.0228 GFLOPs
- SAB modules inserted at end of stages 3 & 4
- 512-d embedding, ArcFace loss (s=30, m=0.3)

### SAB (Spatial Attention Block)

Three independently toggleable branches with residual gating:

| Branch | Function | Design |
|--------|----------|--------|
| Channel (ECA) | Channel recalibration | 1D conv, BN + residual |
| Spatial | Spatial attention | DW 5×5 + PW 1×1, BN + residual |
| Detail | Multi-scale local enhancement | DW 3×3 + 5×5, α=0.01 fixed |

### Curriculum Distillation

- **Teacher**: InsightFace buffalo_s (w600k_mbf, 512-d, 5.5M params)
- **Strategy**: Pure ArcFace for epochs 1–19, distillation activates at epoch 20 (lr decay phase)
- **Loss**: `L = L_arc + 0.1 · L_feat` (cosine feature distillation)
- Teacher features pre-computed and cached (no online teacher inference)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4) |
| Learning rate | 0.1 → 0.01 → 0.001 (step decay at 2/3, 5/6 of epochs) |
| Warmup | 2 epochs |
| Batch size | 256 |
| Epochs | 30 |
| ArcFace | s=30, m=0.3 |
| Gradient clipping | 10.0 |
| Distillation start | Epoch 20 (curriculum) |
| Augmentation | severity="light", p=0.3 (C5 only) |

## Project Structure

```
SA-EdgeFace/
├── qad_face/                    # Core library
│   ├── models/
│   │   ├── edgeface.py          # EdgeFace model + 6 config presets
│   │   ├── attention.py         # SAB (ECA + Spatial + Detail)
│   │   └── shufflenetv2.py      # ShuffleNetV2 backbone
│   ├── train/
│   │   ├── losses.py            # ArcFace + FeatureDistill loss
│   │   └── distill.py           # DistillTrainer (curriculum support)
│   ├── evaluation/
│   │   └── lfw_eval.py          # LFW 10-fold CV evaluation
│   └── utils.py                 # Chinese-path-safe I/O utilities
├── scripts/                     # Training & evaluation scripts
│   ├── train_ablation.py        # 6-config ablation training
│   ├── train_edgeface_1x.py     # EdgeFace 1.0× baseline
│   ├── eval_lfw_full.py         # Batch LFW evaluation
│   ├── eval_multiseed_lfw.py    # Multi-seed eval + paired t-test
│   ├── align_lfw.py             # LFW face alignment
│   └── parse_train_logs.py      # Training log parser
├── paper/                       # LaTeX paper (IEEE Access)
│   ├── main.tex
│   ├── methods.tex
│   └── figs/                    # Figures (ROC, training curves, etc.)
├── results/                     # Experiment results + training curves
├── weights/                     # Model checkpoints (gitignored)
├── datasets/                    # Datasets (gitignored)
├── requirements.txt
└── LICENSE
```

## Datasets

| Dataset | Role | Scale | Source |
|---------|------|-------|--------|
| CASIA-WebFace | Training | 490K imgs, 10.5K ids | [CASIA](http://vis-www.cs.umass.edu/lfw/) |
| LFW | Evaluation | 13K imgs, 5.7K ids | [LFW](http://vis-www.cs.umass.edu/lfw/) |

Both datasets are aligned to 112×112 using InsightFace buffalo_s detector + ArcFace 5-point template.

## Citation

If you find this work useful, please cite:

```bibtex
@article{saedgeface2026,
  title={When More Is Less: A Capacity-Aware Study of Enhancement Techniques for Sub-Million-Parameter Face Recognition},
  author={Toni-cs},
  journal={IEEE Access},
  year={2026},
  note={Under review}
}
```

## License

This project is released under the [MIT License](LICENSE). The training data and pretrained models follow their respective licenses.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for the buffalo_s teacher model and alignment tools
- [ShuffleNetV2](https://github.com/pytorch/vision) for the backbone architecture
- [CASIA-WebFace](http://vis-www.cs.umass.edu/lfw/) and [LFW](http://vis-www.cs.umass.edu/lfw/) datasets
