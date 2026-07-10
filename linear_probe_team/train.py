# =============================================================================
# train.py  –  Linear Probe Team Classifier training script
#
# Task: 10-way team classification using LogisticRegression (multinomial, saga)
# Class imbalance: class_weight='balanced' handles unequal team representation.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from config import (
    MODEL_VERSION_TAG, OUTPUT_DIR,
    INPUT_SCALER_PATH, LABEL_ENCODER_PATH, MODEL_TEAM_PATH,
    LOG_FILENAME, COMPLETE_FLAG_PATH, RESUME_CHECKPOINT_PATH,
    TRAIN_FILES,
    FEATURE_DIM, HOP_SIZE_S, CONTEXT_WINDOW_S,
    VAL_SPLIT, RANDOM_STATE,
    SCALER_SAMPLE_SIZE,
    LR_C, LR_MAX_ITER, LR_SOLVER, LR_PENALTY, LR_CLASS_WEIGHT,
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

    team_counts = train_df['team'].value_counts()
    print("\nTeam distribution in training set:")
    for team, count in team_counts.items():
        print(f"  {team:<25} {count:>8} rows")

    # --- Load or resume ---
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"\nResume checkpoint found. Loading from: {RESUME_CHECKPOINT_PATH}")
        with open(RESUME_CHECKPOINT_PATH, 'rb') as f:
            ckpt = pickle.load(f)
        X_train    = ckpt['X_train']
        y_train    = ckpt['y_train']
        X_val      = ckpt['X_val']
        y_val      = ckpt['y_val']
        in_scaler  = joblib.load(INPUT_SCALER_PATH)
        le         = joblib.load(LABEL_ENCODER_PATH)
        print(f"Loaded {X_train.shape[0]} train rows and {X_val.shape[0]} val rows.")
        print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    else:
        print("\nNo resume checkpoint found. Loading features from H5 files...")

        X_train_raw, y_train_str = load_features_and_labels(
            train_df, padding, HOP_SIZE_S, desc="Train features")
        X_val_raw, y_val_str = load_features_and_labels(
            val_df, padding, HOP_SIZE_S, desc="Val features")

        le = LabelEncoder()
        le.fit(y_train_str)
        joblib.dump(le, LABEL_ENCODER_PATH)
        print(f"\nLabelEncoder saved to: {LABEL_ENCODER_PATH}")
        print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

        y_train = le.transform(y_train_str)
        y_val   = le.transform(y_val_str)

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

    # --- Train ---
    print("\n" + "="*60)
    print(f"LINEAR PROBE TEAM CLASSIFICATION  -  {MODEL_VERSION_TAG}")
    print("="*60)
    print(f"\n[TEAM] Fitting LogisticRegression "
          f"(C={LR_C}, solver={LR_SOLVER}, max_iter={LR_MAX_ITER}, "
          f"class_weight={LR_CLASS_WEIGHT}) ...")

    model = LogisticRegression(
        C=LR_C, penalty=LR_PENALTY, solver=LR_SOLVER,
        multi_class='multinomial', max_iter=LR_MAX_ITER,
        class_weight=LR_CLASS_WEIGHT,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    p_tr = model.predict(X_train)
    p_vl = model.predict(X_val)

    train_acc         = accuracy_score(y_train, p_tr)
    val_acc           = accuracy_score(y_val,   p_vl)
    train_f1_macro    = f1_score(y_train, p_tr, average='macro',    zero_division=0)
    val_f1_macro      = f1_score(y_val,   p_vl, average='macro',    zero_division=0)
    train_f1_weighted = f1_score(y_train, p_tr, average='weighted', zero_division=0)
    val_f1_weighted   = f1_score(y_val,   p_vl, average='weighted', zero_division=0)

    print(f"\n  Train  Acc={train_acc:.4f}  F1(macro)={train_f1_macro:.4f}  F1(weighted)={train_f1_weighted:.4f}")
    print(f"  Val    Acc={val_acc:.4f}  F1(macro)={val_f1_macro:.4f}  F1(weighted)={val_f1_weighted:.4f}")

    print("\nValidation per-team classification report:")
    print(classification_report(y_val, p_vl, target_names=le.classes_, zero_division=0))

    # --- Save ---
    joblib.dump(model, MODEL_TEAM_PATH)
    print(f"Model saved to: {MODEL_TEAM_PATH}")

    log_row = {
        'model_version': MODEL_VERSION_TAG, 'n_classes': len(le.classes_),
        'train_acc': train_acc, 'train_f1_macro': train_f1_macro,
        'train_f1_weighted': train_f1_weighted, 'val_acc': val_acc,
        'val_f1_macro': val_f1_macro, 'val_f1_weighted': val_f1_weighted,
    }
    pd.DataFrame([log_row]).to_csv(LOG_FILENAME, index=False)
    print(f"Training log saved to: {LOG_FILENAME}")

    with open(COMPLETE_FLAG_PATH, 'w') as f:
        f.write(f"Linear probe team training complete. Version: {MODEL_VERSION_TAG}\n")
    print(f"Completion flag written: {COMPLETE_FLAG_PATH}")

    if os.path.exists(RESUME_CHECKPOINT_PATH):
        os.remove(RESUME_CHECKPOINT_PATH)
        print("Resume checkpoint deleted.")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
