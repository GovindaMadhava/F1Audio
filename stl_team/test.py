# =============================================================================
# test.py  –  STL Team Classification inference & evaluation
#
# Loads the best STL_Classification model and LabelEncoder from disk,
# streams features from held-out TEST_FILES, and computes:
#   - Overall accuracy, F1-macro, F1-weighted
#   - Per-team classification report
#   - Confusion matrix heatmaps (raw + normalised)
#   - Team analysis 4-panel plot
#
# Usage:
#   python test.py
# =============================================================================

import os
import sys
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
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    OUTPUT_DIR, get_paths, MODEL_VERSION_TAG,
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
# MODEL
# =============================================================================

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
# DATASETS
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
        team_str = row['team']
        return torch.from_numpy(window.flatten()), team_str


class TestScaledDataset(Dataset):
    def __init__(self, base_ds, in_scaler):
        self.base_ds   = base_ds
        self.in_scaler = in_scaler

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        f, team_str = self.base_ds[idx]
        f_s = self.in_scaler.transform(f.reshape(-1, 1295)).flatten()
        return torch.from_numpy(f_s).float(), team_str


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'STL Team Classifier -- Confusion Matrix\n({MODEL_VERSION_TAG})', fontsize=14)
    plt.xlabel('Predicted Team'); plt.ylabel('True Team')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_stl_team_confusion.png'), dpi=150, bbox_inches='tight')
    plt.close()

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'STL Team Classifier -- Normalised Confusion Matrix\n({MODEL_VERSION_TAG})', fontsize=14)
    plt.xlabel('Predicted Team'); plt.ylabel('True Team')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_stl_team_confusion_norm.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_team_analysis(y_true_idx, y_pred_idx, scores, class_names, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'STL Team Classifier -- Analysis ({MODEL_VERSION_TAG})', fontsize=16)

    n_classes = len(class_names)

    true_counts = np.bincount(y_true_idx, minlength=n_classes)
    pred_counts = np.bincount(y_pred_idx, minlength=n_classes)
    x = np.arange(n_classes); width = 0.4
    axes[0, 0].bar(x - width/2, true_counts, width, alpha=0.7, label='True')
    axes[0, 0].bar(x + width/2, pred_counts, width, alpha=0.7, label='Predicted')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].set_xlabel('Team'); axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Team Sample Distribution'); axes[0, 0].legend()

    confidence = np.max(scores, axis=1)
    axes[0, 1].hist(confidence, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=1.0/n_classes, color='r', linestyle='--', label=f'Chance ({1.0/n_classes:.2f})')
    axes[0, 1].set_xlabel('Confidence (Max Softmax)'); axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Confidence Distribution'); axes[0, 1].legend()

    per_team_acc, per_team_count = [], []
    for i in range(n_classes):
        mask = y_true_idx == i
        if np.sum(mask) > 0:
            per_team_acc.append(np.mean(y_pred_idx[mask] == i))
            per_team_count.append(int(np.sum(mask)))
        else:
            per_team_acc.append(0.0); per_team_count.append(0)

    axes[1, 0].bar(np.arange(n_classes), per_team_acc, alpha=0.7)
    axes[1, 0].set_xticks(np.arange(n_classes)); axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].set_xlabel('Team'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Team Accuracy'); axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].axhline(y=1.0/n_classes, color='r', linestyle='--', alpha=0.5, label=f'Chance ({1.0/n_classes:.2f})')
    axes[1, 0].legend(fontsize=8)
    for i, (acc, cnt) in enumerate(zip(per_team_acc, per_team_count)):
        axes[1, 0].text(i, acc + 0.02, f'n={cnt}', ha='center', fontsize=7, rotation=45)

    axes[1, 1].axis('off')
    acc         = accuracy_score(y_true_idx, y_pred_idx)
    f1_macro    = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_idx, y_pred_idx, average='weighted', zero_division=0)
    chance      = 1.0 / n_classes
    metrics_text  = f"Model: {MODEL_VERSION_TAG}\n\n"
    metrics_text += f"Overall Accuracy : {acc:.4f}\nF1 Macro         : {f1_macro:.4f}\n"
    metrics_text += f"F1 Weighted      : {f1_weighted:.4f}\n\n"
    metrics_text += f"Chance baseline  : {chance:.4f}\nAcc above chance : {acc - chance:+.4f}\n\n"
    metrics_text += f"Total samples    : {len(y_true_idx)}\nNum classes      : {n_classes}"
    axes[1, 1].text(0.05, 0.95, metrics_text, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Performance Summary')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_stl_team_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# TEST
# =============================================================================

def test_team():
    target_name = 'team'
    paths = get_paths(target_name)
    print(f"\n{'='*60}")
    print(f"STL TEST: {target_name.upper()}")
    print(f"{'='*60}")

    dfs = []
    for entry in TEST_FILES:
        temp_df = pd.read_parquet(entry['manifest'])
        temp_df['h5_source_path'] = entry['features']
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} test samples")

    print("\nTeam distribution in test set:")
    for team, count in df['team'].value_counts().items():
        print(f"  {team:<25} {count:>8} rows")

    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * ((padding * 2) + 1)
    print(f"\nContext window: {CONTEXT_WINDOW_S}s | Padding: {padding} | Input dim: {input_dim}")

    in_scaler = joblib.load(paths['input_scaler'])
    le        = joblib.load(paths['label_encoder'])
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    ckpt             = torch.load(paths['best_model'], map_location=device)
    num_team_classes = ckpt['num_team_classes']
    model = STL_Classification(input_dim, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE, num_team_classes).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator(); g.manual_seed(RANDOM_STATE)
    NUM_WORKERS = 0 if sys.platform == 'darwin' else 4

    base_ds     = MultiH5Dataset(df, padding, HOP_SIZE_S)
    test_loader = DataLoader(TestScaledDataset(base_ds, in_scaler),
                             batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'),
                             worker_init_fn=worker_init_fn, generator=g)

    all_pred, all_true_str, all_scores = [], [], []
    print("Running inference...")
    with torch.no_grad():
        for f, team_strs in tqdm(test_loader):
            logits = model(f.to(device))
            scores = torch.softmax(logits, dim=1)
            all_pred.extend(torch.argmax(scores, dim=1).cpu().numpy())
            all_true_str.extend(team_strs)
            all_scores.extend(scores.cpu().numpy())

    all_pred     = np.array(all_pred)
    all_true_str = np.array(all_true_str)
    all_scores   = np.array(all_scores)

    unseen = set(all_true_str) - set(le.classes_)
    if unseen:
        print(f"WARNING: test set contains teams not seen in training: {unseen}")
        mask         = np.isin(all_true_str, le.classes_)
        all_pred     = all_pred[mask]
        all_true_str = all_true_str[mask]
        all_scores   = all_scores[mask]

    all_true = le.transform(all_true_str)

    print("\n" + "="*60)
    print("FINAL TEST METRICS")
    print("="*60)

    acc         = accuracy_score(all_true, all_pred)
    f1_macro    = f1_score(all_true, all_pred, average='macro',    zero_division=0)
    f1_weighted = f1_score(all_true, all_pred, average='weighted', zero_division=0)
    chance      = 1.0 / len(le.classes_)

    print(f"\nOverall Accuracy : {acc:.4f}  (chance = {chance:.4f}, delta = {acc-chance:+.4f})")
    print(f"F1 Macro         : {f1_macro:.4f}")
    print(f"F1 Weighted      : {f1_weighted:.4f}")

    print("\nPer-team classification report:")
    print(classification_report(all_true, all_pred, target_names=le.classes_, zero_division=0))

    results = {
        'true': all_true, 'pred': all_pred, 'scores': all_scores,
        'class_names': le.classes_, 'true_strings': all_true_str,
    }
    joblib.dump(results, paths['test_results'])
    print(f"\nResults saved to: {paths['test_results']}")

    print("\nGenerating plots...")
    plot_confusion_matrix(all_true, all_pred, le.classes_, OUTPUT_DIR)
    plot_team_analysis(all_true, all_pred, all_scores, le.classes_, OUTPUT_DIR)
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    test_team()
