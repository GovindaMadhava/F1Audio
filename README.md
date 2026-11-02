# F1AUDIO: A Formula-1 Engine Audio Dataset & Baselines

[![Dataset](https://img.shields.io/badge/Zenodo-Dataset-blue.svg)](https://doi.org/10.5281/zenodo.21186137)
[![Paper](https://img.shields.io/badge/DCASE%202026-Paper-green.svg)](https://dcase.community/workshop2026/proceedings)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22954971-blue.svg)](https://doi.org/10.5281/zenodo.22954971)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the dataset extraction pipeline and baseline benchmark models for the **F1Audio** dataset, introduced at the 11th Workshop on Detection and Classification of Acoustic Scenes and Events (**DCASE 2026**).

The dataset consists of high-dimensional PaSST audio embeddings (1,295-d, 50 ms hop) extracted from Formula 1 race recordings, synchronized and aligned with car telemetry (RPM, Speed, Throttle, Gear, etc.) across 10 Grand Prix events from the 2024 season.

- **Zenodo Dataset:** [https://doi.org/10.5281/zenodo.21186137](https://doi.org/10.5281/zenodo.21186137)
- **Paper DOI:** [https://doi.org/10.5281/zenodo.22954971](https://doi.org/10.5281/zenodo.22954971)
- **Demo Video:** [YouTube Demo](https://youtu.be/ZsDxqnzAOLk)

---

## Repository Structure

```
├── data_pipeline/         # Dataset extraction & alignment pipeline
│   ├── passt_dataset.py   # PaSST feature extraction & FastF1 telemetry alignment
│   └── sanity_check.py    # Verification script for feature-telemetry consistency
├── stl/                   # Single-Task Learning (MLP) for continuous & discrete telemetry
├── linear_probe/          # Linear/Ridge regression baselines
├── stl_team/              # 10-way Team Classification (MLP)
├── linear_probe_team/     # Team Classification (Logistic Regression)
├── requirements.txt       # Dependencies
├── LICENSE                # MIT License
└── README.md
```

---

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GovindaMadhava/F1Audio.git
   cd F1Audio
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Pipeline (Feature Extraction & Alignment)

If working directly from source audio and raw FastF1 telemetry:

1. **Extract PaSST Embeddings & Align Telemetry**:
   Configure `INPUT_DIR` and `OUTPUT_DIR` in `data_pipeline/passt_dataset.py`, then run:
   ```bash
   python data_pipeline/passt_dataset.py
   ```
   This generates `.h5` files containing 1295-d timestamp embeddings and a unified `.parquet` manifest.

2. **Run Sanity Checks**:
   To verify that features and manifest timestamps align with zero NaN/Inf:
   ```bash
   python data_pipeline/sanity_check.py <path_to_features.h5> <path_to_manifest.parquet>
   ```

---

## Running Baselines

Each baseline directory contains a `config.py` file where you **must** configure your data paths.

> **Note**: Open `config.py` in the baseline folder you want to run and configure the `# ── USER: EDIT` section to point to your dataset directory and choose your context window (`CONTEXT_WINDOW_S`).

### 1. Single-Task Learning (MLP Telemetry Regression & Gear Classification)

```bash
cd stl

# Train models for individual targets
python train.py --target rpm
python train.py --target speed
python train.py --target throttle
python train.py --target gear

# Evaluate trained models on held-out test circuits
python test.py --target rpm
python test.py --target speed
python test.py --target throttle
python test.py --target gear
```

### 2. Linear Probe Baselines

```bash
cd linear_probe

# Train Ridge regression / Logistic regression
python train.py --target rpm

# Evaluate on test circuits
python test.py --target rpm
```

### 3. Team Classification

```bash
# MLP Classifier
cd stl_team
python train.py
python test.py

# Linear Probe (Logistic Regression)
cd linear_probe_team
python train.py
python test.py
```

---

## Output

Training logs, model checkpoints, scalers, and evaluation metrics/plots are saved to `./output/` by default (configurable in `config.py`).

---

## Citation

If you use this code or the F1Audio dataset in your research, please cite our paper:

```bibtex
@inproceedings{madhava2026,
  author    = {Madhava, Govinda and Lerch, Alexander},
  title     = {F1AUDIO: A Formula-1 Engine Audio Dataset},
  booktitle = {Proceedings of the 11th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2026)},
  address   = {Boston, USA},
  month     = {October},
  year      = {2026},
  pages     = {146--150},
  doi       = {10.5281/zenodo.22954971}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for more details.
