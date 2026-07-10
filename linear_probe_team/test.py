# =============================================================================
# test.py  –  Linear Probe Team Classifier inference & evaluation
#
# Loads the trained LogisticRegression model and LabelEncoder from disk,
# streams features from held-out TEST_FILES, and computes metrics + plots.
#
# Usage:
#   python test.py
# =============================================================================

import os
import warnings
import random

import numpy as np
import pandas as pd
import h5py
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from tqdm import tqdm

from config import (
    MODEL_VERSION_TAG, OUTPUT_DIR,
    INPUT_SCALER_PATH, LABEL_ENCODER_PATH, MODEL_TEAM_PATH,
    TEST_RESULTS_PATH,
    TEST_FILES,
    FEATURE_DIM, HOP_SIZE_S, CONTEXT_WINDOW_S,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPERS
# =============================================================================

def extract_window(feat_matrix, timestamp_s, context_padding, hop_size_s):
    f_idx    = int(round(timestamp_s / hop_size_s))
    start    = f_idx - context_padding
    end      = f_idx + context_padding + 1
    n_frames = context_padding * 2 + 1

    window = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
    v_s = max(0, start)
    v_e = min(feat_matrix.shape[0], end)
    w_s = max(0, -start)
    w_e = w_s + (v_e - v_s)
    if w_s < w_e:
        window[w_s:w_e] = feat_matrix[v_s:v_e]
    return window.flatten()


def load_features_and_labels(df, padding, hop_size_s, desc="Loading"):
    X_list, y_list = [], []
    handles = {}

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            path = row['h5_source_path']
            if path not in handles:
                handles[path] = h5py.File(path, 'r')

            feat_matrix = handles[path][row['audio_key']]
            x = extract_window(feat_matrix, row['timestamp_s'], padding, hop_size_s)
            X_list.append(x)
            y_list.append(row['team'])
    finally:
        for h in handles.values():
            h.close()

    return np.array(X_list, dtype=np.float32), np.array(y_list)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'LPT Team Classifier -- Confusion Matrix\n({MODEL_VERSION_TAG})', fontsize=14)
    plt.xlabel('Predicted Team'); plt.ylabel('True Team')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_lpt_team_confusion.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_team_analysis(y_true_idx, y_pred_idx, proba, class_names, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'LPT Team Classifier -- Analysis ({MODEL_VERSION_TAG})', fontsize=16)

    n_classes = len(class_names)

    true_counts = np.bincount(y_true_idx, minlength=n_classes)
    pred_counts = np.bincount(y_pred_idx, minlength=n_classes)
    x = np.arange(n_classes); width = 0.4
    axes[0, 0].bar(x - width/2, true_counts, width, alpha=0.7, label='True')
    axes[0, 0].bar(x + width/2, pred_counts, width, alpha=0.7, label='Predicted')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].set_xlabel('Team'); axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Team Sample Distribution'); axes[0, 0].legend()

    confidence = np.max(proba, axis=1)
    axes[0, 1].hist(confidence, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=1.0/n_classes, color='r', linestyle='--', label=f'Chance ({1.0/n_classes:.2f})')
    axes[0, 1].set_xlabel('Confidence (Max Probability)'); axes[0, 1].set_ylabel('Frequency')
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
    plt.savefig(os.path.join(output_dir, 'test_lpt_team_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def test():
    print(f"Testing linear probe team classifier: {MODEL_VERSION_TAG}")

    print("Loading test data manifests...")
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
    input_dim = FEATURE_DIM * (padding * 2 + 1)
    print(f"\nContext window: {CONTEXT_WINDOW_S}s | Padding: {padding} frames | Input dim: {input_dim}")

    print("Loading input scaler and label encoder...")
    in_scaler = joblib.load(INPUT_SCALER_PATH)
    le        = joblib.load(LABEL_ENCODER_PATH)
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    X_raw, y_team_str = load_features_and_labels(df, padding, HOP_SIZE_S, desc="Test features")
    print(f"Test feature matrix shape: {X_raw.shape}")

    unseen = set(y_team_str) - set(le.classes_)
    if unseen:
        print(f"WARNING: test set contains teams not seen in training: {unseen}")
        mask       = np.isin(y_team_str, le.classes_)
        X_raw      = X_raw[mask]
        y_team_str = y_team_str[mask]

    print("Scaling features...")
    X = in_scaler.transform(X_raw).astype(np.float32)
    del X_raw

    y_true = le.transform(y_team_str)

    print("Loading trained model...")
    model = joblib.load(MODEL_TEAM_PATH)

    print("Running inference...")
    y_pred = model.predict(X)
    proba  = model.predict_proba(X)

    print("\n" + "="*60)
    print("FINAL TEST METRICS")
    print("="*60)

    acc         = accuracy_score(y_true, y_pred)
    f1_macro    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    chance      = 1.0 / len(le.classes_)

    print(f"\nOverall Accuracy : {acc:.4f}  (chance = {chance:.4f}, delta = {acc-chance:+.4f})")
    print(f"F1 Macro         : {f1_macro:.4f}")
    print(f"F1 Weighted      : {f1_weighted:.4f}")

    print("\nPer-team classification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    results = {
        'true': y_true, 'pred': y_pred, 'proba': proba,
        'class_names': le.classes_, 'true_strings': y_team_str,
    }
    joblib.dump(results, TEST_RESULTS_PATH)
    print(f"\nResults saved to: {TEST_RESULTS_PATH}")

    print("\nGenerating plots...")
    plot_confusion_matrix(y_true, y_pred, le.classes_, OUTPUT_DIR)
    plot_team_analysis(y_true, y_pred, proba, le.classes_, OUTPUT_DIR)
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    test()
