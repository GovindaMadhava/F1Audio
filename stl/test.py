# =============================================================================
# test.py  –  STL inference & evaluation
# Reads all configuration from config.py
#
# Usage:
#   python test.py --target rpm
#   python test.py --target speed
#   python test.py --target throttle
#   python test.py --target gear
# =============================================================================

import os
import argparse
import warnings
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import h5py
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, r2_score,
                             mean_squared_error, confusion_matrix)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    OUTPUT_DIR, get_paths,
    TEST_FILES,
    FEATURE_DIM, HOP_SIZE_S, CONTEXT_WINDOW_S,
    HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE,
    TEST_BATCH_SIZE, RANDOM_STATE,
)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_STATE)


# =============================================================================
# MODELS  (must exactly match train.py)
# =============================================================================

class STL_Regression(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, dropout_rate):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.head = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        return self.head(self.trunk(x))


class STL_Classification(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, dropout_rate, num_classes):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.head = nn.Linear(hidden_dim_2, num_classes)

    def forward(self, x):
        return self.head(self.trunk(x))


# =============================================================================
# DATASET
# =============================================================================

class MultiH5Dataset(Dataset):
    def __init__(self, df, context_padding, hop_size_s):
        self.df              = df.reset_index(drop=True)
        self.context_padding = context_padding
        self.hop_size_s      = hop_size_s
        self.handles         = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if row['h5_source_path'] not in self.handles:
            self.handles[row['h5_source_path']] = h5py.File(row['h5_source_path'], 'r')
        feat_matrix = self.handles[row['h5_source_path']][row['audio_key']]
        f_idx = int(round(row['timestamp_s'] / self.hop_size_s))
        start = f_idx - self.context_padding
        end   = f_idx + self.context_padding + 1
        window = np.zeros(((self.context_padding * 2) + 1, FEATURE_DIM), dtype=np.float32)
        v_s = max(0, start);  v_e = min(feat_matrix.shape[0], end)
        w_s = max(0, -start); w_e = w_s + (v_e - v_s)
        if w_s < w_e:
            window[w_s:w_e] = feat_matrix[v_s:v_e]
        labels = row[['target_rpm', 'target_speed', 'target_gear', 'target_throttle']].values.astype(np.float32)
        labels[3] = np.clip(labels[3], 0, 100)
        return torch.from_numpy(window.flatten()), torch.from_numpy(labels)


class TestScaledDataset(Dataset):
    def __init__(self, base_ds, in_scaler):
        self.base_ds   = base_ds
        self.in_scaler = in_scaler

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        f, l = self.base_ds[idx]
        f_s = self.in_scaler.transform(f.reshape(1, -1)).flatten()
        return torch.from_numpy(f_s).float(), l


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_regression_results(y_true, y_pred, task_name, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'STL {task_name.upper()} -- Regression Analysis', fontsize=16)

    r2        = r2_score(y_true, y_pred)
    residuals = y_true - y_pred

    axes[0, 0].scatter(y_true, y_pred, alpha=0.3, s=1)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values'); axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title(f'{task_name}: Predictions vs True')
    axes[0, 0].text(0.05, 0.95, f'R2 = {r2:.4f}', transform=axes[0, 0].transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predictions'); axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{task_name}: Residuals')
    axes[0, 1].text(0.05, 0.95, f'Residuals std: {np.std(residuals):.2f}',
                    transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top')

    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Prediction Error'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{task_name}: Error Distribution')

    sample_size = min(1000, len(y_true))
    axes[1, 1].plot(y_true[:sample_size], label='True',      alpha=0.7, linewidth=1)
    axes[1, 1].plot(y_pred[:sample_size], label='Predicted', alpha=0.7, linewidth=1)
    axes[1, 1].set_xlabel('Sample Index'); axes[1, 1].set_ylabel(task_name)
    axes[1, 1].set_title(f'{task_name}: Time Series Sample (first {sample_size} points)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_stl_{task_name}_regression.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, task_name, output_dir, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or 'auto', yticklabels=class_names or 'auto')
    plt.title(f'STL {task_name.upper()} -- Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_stl_{task_name}_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names or 'auto', yticklabels=class_names or 'auto')
    plt.title(f'STL {task_name.upper()} -- Normalised Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_stl_{task_name}_confusion_matrix_norm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return cm


def plot_gear_analysis(y_true, y_pred, gear_scores, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('STL GEAR -- Classification Analysis', fontsize=16)

    unique_gears = sorted(set(y_true) | set(y_pred))
    gear_labels  = [str(int(g)) for g in unique_gears]

    axes[0, 0].hist(y_true, bins=len(unique_gears), alpha=0.5, label='True',      align='left', rwidth=0.8)
    axes[0, 0].hist(y_pred, bins=len(unique_gears), alpha=0.5, label='Predicted', align='left', rwidth=0.8)
    axes[0, 0].set_xlabel('Gear'); axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Gear Distribution'); axes[0, 0].set_xticks(unique_gears)
    axes[0, 0].set_xticklabels(gear_labels); axes[0, 0].legend()

    confidence_scores = np.max(gear_scores, axis=1)
    axes[0, 1].hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0.5, color='r', linestyle='--', label='0.5 threshold')
    axes[0, 1].set_xlabel('Confidence (Max Softmax)'); axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Confidence Distribution'); axes[0, 1].legend()

    gear_acc = {}; gear_counts = {}
    for gear in unique_gears:
        mask = y_true == gear
        if np.sum(mask) > 0:
            gear_acc[gear]    = np.mean(y_pred[mask] == gear)
            gear_counts[gear] = np.sum(mask)
    x_pos = np.arange(len(gear_acc))
    axes[1, 0].bar(x_pos, list(gear_acc.values()), alpha=0.7)
    axes[1, 0].set_xlabel('Gear'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Gear Accuracy')
    axes[1, 0].set_xticks(x_pos); axes[1, 0].set_xticklabels([str(int(g)) for g in gear_acc.keys()])
    axes[1, 0].set_ylim([0, 1])
    for i, (gear, acc) in enumerate(gear_acc.items()):
        axes[1, 0].text(i, acc + 0.02, f'n={gear_counts[gear]}', ha='center', fontsize=9)

    axes[1, 1].axis('off')
    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    metrics_text  = f"Accuracy: {acc:.4f}\n"
    metrics_text += f"F1 Macro: {f1_macro:.4f}\n"
    metrics_text += f"F1 Weighted: {f1_weighted:.4f}\n\n"
    metrics_text += f"Total samples: {len(y_true)}\n"
    metrics_text += f"Unique gears: {len(unique_gears)}"
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].set_title('Performance Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_stl_gear_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def _load_test_data():
    dfs = []
    for entry in TEST_FILES:
        temp_df = pd.read_parquet(entry['manifest'])
        temp_df['h5_source_path'] = entry['features']
        dfs.append(temp_df)
    return pd.concat(dfs, ignore_index=True)


def test_regression(target_name):
    """Run inference and evaluation for a single regression target."""
    paths = get_paths(target_name)
    print(f"\n{'='*60}")
    print(f"STL TEST: {target_name.upper()}")
    print(f"{'='*60}")

    df = _load_test_data()
    print(f"Loaded {len(df)} test samples")

    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * ((padding * 2) + 1)
    print(f"Context window: {CONTEXT_WINDOW_S}s | Padding: {padding} | Input dim: {input_dim}")

    in_scaler  = joblib.load(paths['input_scaler'])
    out_scaler = joblib.load(paths['target_scaler'])

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    base_ds     = MultiH5Dataset(df, padding, HOP_SIZE_S)
    test_loader = DataLoader(TestScaledDataset(base_ds, in_scaler),
                             batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn, generator=g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = STL_Regression(input_dim, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE).to(device)
    ckpt  = torch.load(paths['best_model'], map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Label column indices in MultiH5Dataset: rpm=0, speed=1, gear=2, throttle=3
    col = {'rpm': 0, 'speed': 1, 'throttle': 3}[target_name]

    all_pred, all_true = [], []
    print("Running inference...")
    with torch.no_grad():
        for f, l in tqdm(test_loader):
            pred = model(f.to(device)).squeeze()
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(l[:, col].cpu().numpy())

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    pred_orig = out_scaler.inverse_transform(all_pred.reshape(-1, 1)).flatten()
    true_orig = all_true   # already in original units

    if target_name == 'throttle':
        pred_orig = np.clip(pred_orig, 0, 100)
        true_orig = np.clip(true_orig, 0, 100)
    if target_name == 'speed':
        pred_orig = np.clip(pred_orig, 0, None)

    rmse = np.sqrt(mean_squared_error(true_orig, pred_orig))
    r2   = r2_score(true_orig, pred_orig)

    print(f"\n--- {target_name.upper()} RESULTS ---")
    print(f"RMSE={rmse:.2f}, R2={r2:.4f}")
    print(f"  True range: [{true_orig.min():.2f}, {true_orig.max():.2f}]")
    print(f"  Pred range: [{pred_orig.min():.2f}, {pred_orig.max():.2f}]")

    plot_regression_results(true_orig, pred_orig, target_name, OUTPUT_DIR)
    print(f"  Regression plot saved")

    results = {'true': true_orig, 'pred': pred_orig}
    joblib.dump(results, paths['test_results'])
    print(f"Results saved: {paths['test_results']}")


def test_gear():
    """Run inference and evaluation for gear classification."""
    target_name = 'gear'
    paths = get_paths(target_name)
    print(f"\n{'='*60}")
    print(f"STL TEST: {target_name.upper()}")
    print(f"{'='*60}")

    df = _load_test_data()
    print(f"Loaded {len(df)} test samples")

    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * ((padding * 2) + 1)
    print(f"Context window: {CONTEXT_WINDOW_S}s | Padding: {padding} | Input dim: {input_dim}")

    in_scaler = joblib.load(paths['input_scaler'])

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    base_ds     = MultiH5Dataset(df, padding, HOP_SIZE_S)
    test_loader = DataLoader(TestScaledDataset(base_ds, in_scaler),
                             batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn, generator=g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt             = torch.load(paths['best_model'], map_location=device)
    num_gear_classes = ckpt['num_gear_classes']
    model = STL_Classification(input_dim, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE, num_gear_classes).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    all_pred, all_true, all_scores = [], [], []
    print("Running inference...")
    with torch.no_grad():
        for f, l in tqdm(test_loader):
            logits = model(f.to(device))
            scores = torch.softmax(logits, dim=1)
            all_pred.extend(torch.argmax(scores, dim=1).cpu().numpy())
            all_true.extend(l[:, 2].cpu().numpy())   # gear column = 2
            all_scores.extend(scores.cpu().numpy())

    all_pred   = np.array(all_pred)
    all_true   = np.array(all_true)
    all_scores = np.array(all_scores)

    acc         = accuracy_score(all_true, all_pred)
    f1_macro    = f1_score(all_true, all_pred, average='macro')
    f1_weighted = f1_score(all_true, all_pred, average='weighted')

    print(f"\n--- GEAR RESULTS ---")
    print(f"Accuracy={acc:.4f}, F1(macro)={f1_macro:.4f}, F1(weighted)={f1_weighted:.4f}")

    unique_gears = sorted(set(all_true) | set(all_pred))
    gear_labels  = [str(int(g)) for g in unique_gears]
    plot_confusion_matrix(all_true, all_pred, target_name, OUTPUT_DIR, gear_labels)
    plot_gear_analysis(all_true, all_pred, all_scores, OUTPUT_DIR)
    print(f"  Gear analysis plots saved")

    results = {'true': all_true, 'pred': all_pred, 'scores': all_scores}
    joblib.dump(results, paths['test_results'])
    print(f"Results saved: {paths['test_results']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='STL test script')
    parser.add_argument('--target', required=True,
                        choices=['rpm', 'speed', 'throttle', 'gear'],
                        help='Target to test')
    args = parser.parse_args()

    if args.target == 'gear':
        test_gear()
    else:
        test_regression(args.target)

    print(f"\nAll plots and results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
