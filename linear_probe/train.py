# =============================================================================
# train.py  –  Linear Probe baseline training script
#
# Models trained (one per task, all independent):
#   RPM      - sklearn Ridge regression
#   Speed    - sklearn Ridge regression
#   Throttle - sklearn Ridge regression (throttle-weighted, same as MLP)
#   Gear     - sklearn LogisticRegression (multinomial, solver=saga)
#
# Data consistency guarantee:
#   The manifest loading, 80/20 train/val split, context-window extraction,
#   throttle clipping, and input StandardScaler fitting all use exactly the
#   same logic and RANDOM_STATE as the STL MLP baseline.
#
# Reproducibility:
#   All randomness is controlled via RANDOM_STATE.
#
# Usage:
#   python train.py
# =============================================================================

import os
import random
import warnings
import pickle

import numpy as np
import pandas as pd
import h5py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score, mean_squared_error
from tqdm import tqdm

from config import (
    MODEL_VERSION_TAG, OUTPUT_DIR,
    INPUT_SCALER_PATH,
    MODEL_RPM_PATH, MODEL_SPEED_PATH, MODEL_THROTTLE_PATH, MODEL_GEAR_PATH,
    LOG_FILENAME, COMPLETE_FLAG_PATH, RESUME_CHECKPOINT_PATH,
    TRAIN_FILES,
    FEATURE_DIM, HOP_SIZE_S, CONTEXT_WINDOW_S,
    VAL_SPLIT, RANDOM_STATE,
    SCALER_SAMPLE_SIZE,
    RIDGE_ALPHA, LR_C, LR_MAX_ITER, LR_SOLVER, LR_PENALTY,
)

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPERS
# =============================================================================

def extract_window(feat_matrix, timestamp_s, context_padding, hop_size_s):
    """Return a flattened context window for a single row."""
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
    """
    Stream ALL rows of df from H5 files into two numpy arrays:
      X : (N, input_dim)  - raw (unscaled) flattened context windows
      y : (N, 4)          - [rpm, speed, gear, throttle]  (original scale)
    """
    X_list, y_list = [], []
    handles = {}

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            path = row['h5_source_path']
            if path not in handles:
                handles[path] = h5py.File(path, 'r')

            feat_matrix = handles[path][row['audio_key']]
            x = extract_window(feat_matrix, row['timestamp_s'], padding, hop_size_s)
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


def compute_throttle_weights(throttle_values, n_bins=20):
    bins      = np.linspace(0, 100, n_bins + 1)
    counts, _ = np.histogram(throttle_values, bins=bins)
    counts    = np.maximum(counts, 1)
    bin_idx   = np.clip(np.digitize(throttle_values, bins=bins) - 1, 0, len(counts) - 1)
    weights   = 1.0 / counts[bin_idx]
    weights   = weights / np.mean(weights)
    return weights


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * (padding * 2 + 1)
    print(f"Context window: {CONTEXT_WINDOW_S}s | Padding: {padding} frames | Input dim: {input_dim}")

    # --- Load manifests and split ---
    print("\nLoading manifests...")
    all_dfs = []
    for entry in TRAIN_FILES:
        temp_df = pd.read_parquet(entry['manifest'])
        temp_df['h5_source_path'] = entry['features']
        all_dfs.append(temp_df)
    full_df = pd.concat(all_dfs, ignore_index=True)

    train_df, val_df = train_test_split(full_df, test_size=VAL_SPLIT, random_state=RANDOM_STATE)
    print(f"Total: {len(full_df)} | Train: {len(train_df)} | Val: {len(val_df)}")

    # --- Load or resume feature matrices ---
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"\nResume checkpoint found. Loading from: {RESUME_CHECKPOINT_PATH}")
        with open(RESUME_CHECKPOINT_PATH, 'rb') as f:
            ckpt = pickle.load(f)
        X_train   = ckpt['X_train']
        y_train   = ckpt['y_train']
        X_val     = ckpt['X_val']
        y_val     = ckpt['y_val']
        in_scaler = joblib.load(INPUT_SCALER_PATH)
        print(f"Loaded {X_train.shape[0]} train rows and {X_val.shape[0]} val rows.")
    else:
        print("\nNo resume checkpoint found. Loading features from H5 files...")

        X_train_raw, y_train = load_features_and_labels(
            train_df, padding, HOP_SIZE_S, desc="Train features")
        X_val_raw, y_val = load_features_and_labels(
            val_df, padding, HOP_SIZE_S, desc="Val features")

        if os.path.exists(INPUT_SCALER_PATH):
            print("Loading existing input scaler...")
            in_scaler = joblib.load(INPUT_SCALER_PATH)
        else:
            print(f"Fitting input scaler on {SCALER_SAMPLE_SIZE} samples...")
            in_scaler  = StandardScaler()
            sample_idx = np.random.choice(
                len(X_train_raw), min(SCALER_SAMPLE_SIZE, len(X_train_raw)), replace=False)
            in_scaler.fit(X_train_raw[sample_idx])
            joblib.dump(in_scaler, INPUT_SCALER_PATH)
            print(f"Input scaler saved to: {INPUT_SCALER_PATH}")

        print("Scaling features...")
        X_train = in_scaler.transform(X_train_raw).astype(np.float32)
        X_val   = in_scaler.transform(X_val_raw).astype(np.float32)
        del X_train_raw, X_val_raw

        print(f"Saving resume checkpoint to: {RESUME_CHECKPOINT_PATH}")
        with open(RESUME_CHECKPOINT_PATH, 'wb') as f:
            pickle.dump({
                'X_train': X_train, 'y_train': y_train,
                'X_val':   X_val,   'y_val':   y_val,
            }, f)

    # --- Throttle sample weights ---
    throttle_weights = compute_throttle_weights(y_train[:, 3])

    # --- Train one model per task ---
    # y column order: 0=rpm, 1=speed, 2=gear, 3=throttle
    log_row   = {}
    task_info = [
        ('rpm',      0, 'regression'),
        ('speed',    1, 'regression'),
        ('gear',     2, 'classification'),
        ('throttle', 3, 'regression'),
    ]
    model_paths = {
        'rpm':      MODEL_RPM_PATH,
        'speed':    MODEL_SPEED_PATH,
        'throttle': MODEL_THROTTLE_PATH,
        'gear':     MODEL_GEAR_PATH,
    }

    print("\n" + "="*60)
    print(f"LINEAR PROBE TRAINING  -  {MODEL_VERSION_TAG}")
    print("="*60)

    for task, col, kind in task_info:
        y_tr = y_train[:, col]
        y_vl = y_val[:,   col]

        if kind == 'regression':
            sample_weight = throttle_weights if task == 'throttle' else None
            print(f"\n[{task.upper()}] Fitting Ridge (alpha={RIDGE_ALPHA}) ...")
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, random_state=RANDOM_STATE)
            model.fit(X_train, y_tr, sample_weight=sample_weight)

            p_tr = model.predict(X_train)
            p_vl = model.predict(X_val)
            if task == 'throttle':
                p_tr = np.clip(p_tr, 0, 100)
                p_vl = np.clip(p_vl, 0, 100)

            train_rmse = np.sqrt(mean_squared_error(y_tr, p_tr))
            val_rmse   = np.sqrt(mean_squared_error(y_vl, p_vl))
            train_r2   = r2_score(y_tr, p_tr)
            val_r2     = r2_score(y_vl, p_vl)

            print(f"  Train  RMSE={train_rmse:.4f}  R2={train_r2:.4f}")
            print(f"  Val    RMSE={val_rmse:.4f}  R2={val_r2:.4f}")

            log_row[f'train_rmse_{task}'] = train_rmse
            log_row[f'train_r2_{task}']   = train_r2
            log_row[f'val_rmse_{task}']   = val_rmse
            log_row[f'val_r2_{task}']     = val_r2

        else:  # gear classification
            print(f"\n[{task.upper()}] Fitting LogisticRegression "
                  f"(C={LR_C}, solver={LR_SOLVER}, max_iter={LR_MAX_ITER}) ...")
            model = LogisticRegression(
                C=LR_C, penalty=LR_PENALTY, solver=LR_SOLVER,
                multi_class='multinomial', max_iter=LR_MAX_ITER,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
            )
            model.fit(X_train, y_tr.astype(int))

            p_tr = model.predict(X_train)
            p_vl = model.predict(X_val)

            train_acc         = accuracy_score(y_tr.astype(int), p_tr)
            val_acc           = accuracy_score(y_vl.astype(int), p_vl)
            train_f1_macro    = f1_score(y_tr.astype(int), p_tr, average='macro',    zero_division=0)
            val_f1_macro      = f1_score(y_vl.astype(int), p_vl, average='macro',    zero_division=0)
            train_f1_weighted = f1_score(y_tr.astype(int), p_tr, average='weighted', zero_division=0)
            val_f1_weighted   = f1_score(y_vl.astype(int), p_vl, average='weighted', zero_division=0)

            print(f"  Train  Acc={train_acc:.4f}  F1(macro)={train_f1_macro:.4f}  F1(weighted)={train_f1_weighted:.4f}")
            print(f"  Val    Acc={val_acc:.4f}  F1(macro)={val_f1_macro:.4f}  F1(weighted)={val_f1_weighted:.4f}")

            log_row[f'train_acc_{task}']          = train_acc
            log_row[f'train_f1_macro_{task}']     = train_f1_macro
            log_row[f'train_f1_weighted_{task}']  = train_f1_weighted
            log_row[f'val_acc_{task}']            = val_acc
            log_row[f'val_f1_macro_{task}']       = val_f1_macro
            log_row[f'val_f1_weighted_{task}']    = val_f1_weighted

        joblib.dump(model, model_paths[task])
        print(f"  Model saved to: {model_paths[task]}")

    # --- Log + completion ---
    log_row['model_version'] = MODEL_VERSION_TAG
    log_df = pd.DataFrame([log_row])
    log_df.to_csv(LOG_FILENAME, index=False)
    print(f"\nTraining log saved to: {LOG_FILENAME}")

    with open(COMPLETE_FLAG_PATH, 'w') as f:
        f.write(f"Linear probe training complete. Version: {MODEL_VERSION_TAG}\n")
    print(f"Completion flag written: {COMPLETE_FLAG_PATH}")

    if os.path.exists(RESUME_CHECKPOINT_PATH):
        os.remove(RESUME_CHECKPOINT_PATH)
        print("Resume checkpoint deleted.")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
