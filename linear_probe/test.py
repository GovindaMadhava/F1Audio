# =============================================================================
# test.py  –  Linear Probe baseline inference & evaluation
#
# Targets: RPM (Ridge), Speed (Ridge), Throttle (Ridge), Gear (LogisticReg)
# Produces metrics and plots for comparison with the STL MLP baseline.
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
    accuracy_score, f1_score, r2_score,
    mean_squared_error, confusion_matrix,
)
from tqdm import tqdm

from config import (
    MODEL_VERSION_TAG, OUTPUT_DIR,
    INPUT_SCALER_PATH,
    MODEL_RPM_PATH, MODEL_SPEED_PATH, MODEL_THROTTLE_PATH, MODEL_GEAR_PATH,
    TEST_RESULTS_PATH,
    TEST_FILES,
    FEATURE_DIM, HOP_SIZE_S, CONTEXT_WINDOW_S,
    TEST_BATCH_SIZE, RANDOM_STATE,
)

warnings.filterwarnings("ignore", category=UserWarning)
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPERS
# =============================================================================

def extract_window(feat_matrix, timestamp_s, context_padding, hop_size_s):
    f_idx = int(round(timestamp_s / hop_size_s))
    start = f_idx - context_padding
    end   = f_idx + context_padding + 1
    n_frames = context_padding * 2 + 1

    window = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
    v_s = max(0, start)
    v_e = min(feat_matrix.shape[0], end)
    w_s = max(0, -start)
    w_e = w_s + (v_e - v_s)
    if w_s < w_e:
        window[w_s:w_e] = feat_matrix[v_s:v_e]
    return window.flatten()


def load_test_features(df, padding, hop_size_s, batch_size):
    n_frames  = padding * 2 + 1
    handles   = {}
    X_list, y_list = [], []

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Test features"):
            path = row['h5_source_path']
            if path not in handles:
                handles[path] = h5py.File(path, 'r')
            x = extract_window(handles[path][row['audio_key']],
                               row['timestamp_s'], padding, hop_size_s)
            y = np.array([
                row['target_rpm'],
                row['target_speed'],
                row['target_gear'],
                np.clip(row['target_throttle'], 0, 100),
            ], dtype=np.float32)
            X_list.append(x)
            y_list.append(y)
    finally:
        for h in handles.values():
            h.close()

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_regression_results(y_true, y_pred, task_name, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'LP {task_name.upper()} -- Regression Analysis', fontsize=16)

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

    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Prediction Error'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{task_name}: Error Distribution')

    sample_size = min(1000, len(y_true))
    axes[1, 1].plot(y_true[:sample_size], label='True',      alpha=0.7, linewidth=1)
    axes[1, 1].plot(y_pred[:sample_size], label='Predicted', alpha=0.7, linewidth=1)
    axes[1, 1].set_xlabel('Sample Index'); axes[1, 1].set_ylabel(task_name)
    axes[1, 1].set_title(f'{task_name}: Time Series (first {sample_size})')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_lp_{task_name}_regression.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, task_name, output_dir, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or 'auto',
                yticklabels=class_names or 'auto')
    plt.title(f'LP {task_name.upper()} -- Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'test_lp_{task_name}_confusion.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_gear_analysis(y_true, y_pred, gear_proba, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('LP GEAR -- Classification Analysis', fontsize=16)

    unique_gears = sorted(set(y_true) | set(y_pred))
    gear_labels  = [str(int(g)) for g in unique_gears]

    axes[0, 0].hist(y_true, bins=len(unique_gears), alpha=0.5, label='True',      align='left', rwidth=0.8)
    axes[0, 0].hist(y_pred, bins=len(unique_gears), alpha=0.5, label='Predicted', align='left', rwidth=0.8)
    axes[0, 0].set_xlabel('Gear'); axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Gear Distribution'); axes[0, 0].set_xticks(unique_gears)
    axes[0, 0].set_xticklabels(gear_labels); axes[0, 0].legend()

    confidence_scores = np.max(gear_proba, axis=1)
    axes[0, 1].hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0.5, color='r', linestyle='--', label='0.5 threshold')
    axes[0, 1].set_xlabel('Confidence (Max Probability)'); axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Confidence Distribution'); axes[0, 1].legend()

    gear_acc = {}; gear_counts = {}
    for gear in unique_gears:
        mask = y_true == gear
        if np.sum(mask) > 0:
            gear_acc[gear]    = np.mean(y_pred[mask] == gear)
            gear_counts[gear] = int(np.sum(mask))
    x_pos = np.arange(len(gear_acc))
    axes[1, 0].bar(x_pos, list(gear_acc.values()), alpha=0.7)
    axes[1, 0].set_xlabel('Gear'); axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Gear Accuracy')
    axes[1, 0].set_xticks(x_pos); axes[1, 0].set_xticklabels([str(int(g)) for g in gear_acc.keys()])
    axes[1, 0].set_ylim([0, 1])
    for i, (gear, acc) in enumerate(gear_acc.items()):
        axes[1, 0].text(i, acc + 0.02, f'n={gear_counts[gear]}', ha='center', fontsize=9)

    axes[1, 1].axis('off')
    acc          = accuracy_score(y_true, y_pred)
    f1_macro     = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    f1_weighted  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics_text  = f"Accuracy: {acc:.4f}\n"
    metrics_text += f"F1 Macro: {f1_macro:.4f}\n"
    metrics_text += f"F1 Weighted: {f1_weighted:.4f}\n\n"
    metrics_text += f"Total samples: {len(y_true)}\n"
    metrics_text += f"Unique gears: {len(unique_gears)}"
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].set_title('Performance Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_lp_gear_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def test():
    print(f"Testing linear probe: {MODEL_VERSION_TAG}")

    # --- Load test manifests ---
    print("Loading test data manifests...")
    dfs = []
    for entry in TEST_FILES:
        temp_df = pd.read_parquet(entry['manifest'])
        temp_df['h5_source_path'] = entry['features']
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} test samples")

    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * (padding * 2 + 1)
    print(f"Context window: {CONTEXT_WINDOW_S}s | Padding: {padding} frames | Input dim: {input_dim}")

    # --- Load scaler + test features ---
    print("Loading input scaler...")
    in_scaler = joblib.load(INPUT_SCALER_PATH)

    X_raw, labels = load_test_features(df, padding, HOP_SIZE_S, TEST_BATCH_SIZE)
    print(f"Test feature matrix shape: {X_raw.shape}")

    print("Scaling features...")
    X = in_scaler.transform(X_raw).astype(np.float32)
    del X_raw

    y_rpm      = labels[:, 0]
    y_speed    = labels[:, 1]
    y_gear     = labels[:, 2].astype(int)
    y_throttle = labels[:, 3]

    # --- Load models ---
    print("Loading trained models...")
    model_rpm      = joblib.load(MODEL_RPM_PATH)
    model_speed    = joblib.load(MODEL_SPEED_PATH)
    model_throttle = joblib.load(MODEL_THROTTLE_PATH)
    model_gear     = joblib.load(MODEL_GEAR_PATH)

    # --- Inference ---
    print("Running inference...")
    pred_rpm      = model_rpm.predict(X)
    pred_speed    = model_speed.predict(X)
    pred_throttle = np.clip(model_throttle.predict(X), 0, 100)
    pred_gear     = model_gear.predict(X)
    gear_proba    = model_gear.predict_proba(X)

    # --- Metrics ---
    print("\n" + "="*60)
    print("FINAL TEST METRICS")
    print("="*60)

    print("\n--- REGRESSION TASKS ---")
    for name, y_true, y_pred in [
        ('RPM',      y_rpm,      pred_rpm),
        ('Speed',    y_speed,    pred_speed),
        ('Throttle', y_throttle, pred_throttle),
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")
        print(f"  True range: [{y_true.min():.2f}, {y_true.max():.2f}]")
        print(f"  Pred range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    print("\n--- CLASSIFICATION TASKS ---")
    gear_acc        = accuracy_score(y_gear, pred_gear)
    gear_f1_macro   = f1_score(y_gear, pred_gear, average='macro',    zero_division=0)
    gear_f1_weighted = f1_score(y_gear, pred_gear, average='weighted', zero_division=0)
    print(f"GEAR: Accuracy={gear_acc:.4f}, F1(macro)={gear_f1_macro:.4f}, "
          f"F1(weighted)={gear_f1_weighted:.4f}")

    # --- Save results ---
    results = {
        'true': {'rpm': y_rpm, 'speed': y_speed, 'throttle': y_throttle, 'gear': y_gear},
        'pred': {'rpm': pred_rpm, 'speed': pred_speed, 'throttle': pred_throttle, 'gear': pred_gear},
        'scores': {'gear': gear_proba},
    }
    joblib.dump(results, TEST_RESULTS_PATH)
    print(f"\nResults saved to: {TEST_RESULTS_PATH}")

    # --- Plots ---
    print("\nGenerating plots...")
    for task, y_true, y_pred in [
        ('rpm',      y_rpm,      pred_rpm),
        ('speed',    y_speed,    pred_speed),
        ('throttle', y_throttle, pred_throttle),
    ]:
        plot_regression_results(y_true, y_pred, task, OUTPUT_DIR)
        print(f"  {task} regression plot saved")

    unique_gears = sorted(set(y_gear) | set(pred_gear))
    gear_labels  = [str(int(g)) for g in unique_gears]
    plot_confusion_matrix(y_gear, pred_gear, 'gear', OUTPUT_DIR, gear_labels)
    plot_gear_analysis(y_gear, pred_gear, gear_proba, OUTPUT_DIR)
    print(f"  gear analysis plots saved")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    test()
