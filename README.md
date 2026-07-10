# DCASE F1 Audio Dataset & Baselines

This repository contains the dataset extraction pipeline and baseline models for the DCASE F1 Audio Dataset. The dataset consists of high-dimensional PaSST audio embeddings extracted from Formula 1 race recordings, aligned with telemetry data (RPM, Speed, Throttle, Gear, etc.).

## Repository Structure

```
├── baselines/                 # Baseline model implementations
│   ├── stl/                   # Single-Task Learning (MLP) for RPM, Speed, Throttle, Gear
│   ├── linear_probe/          # Linear/Ridge regression baselines
│   ├── stl_team/              # 10-way Team Classification (MLP)
│   └── linear_probe_team/     # Team Classification (Logistic Regression)
│
├── requirements.txt           # Minimal dependencies
└── README.md
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running Baselines

Each baseline directory contains a `config.py` file where you **must** configure your data paths.

> **Note**: Open the `config.py` in the baseline you want to run and look for the `# ── USER: EDIT` sections to point to your `data/features` directory and to set the `CONTEXT_WINDOW_S` (how many seconds of audio context to use).

### Example: STL MLP Baseline

```bash
cd baselines/stl

# Train models for different targets
python train.py --target rpm
python train.py --target speed
python train.py --target gear
python train.py --target throttle

# Evaluate trained models
python test.py --target rpm
python test.py --target gear
```

### Example: Team Classification (Linear Probe)

```bash
cd baselines/linear_probe_team

# Train the logistic regression model
python train.py

# Evaluate and plot confusion matrix
python test.py
```

## Output

Training logs, model checkpoints, scalers, and evaluation plots (such as confusion matrices and regression analysis) are saved to the `./output/` directory by default (configurable in `config.py`).

## License

MIT License. See `LICENSE` for more information.
