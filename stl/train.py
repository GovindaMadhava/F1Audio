# =============================================================================
# train.py  –  Unified STL training script
#
# Trains a single-task model for one of four targets:
#   RPM       (regression, MSELoss)
#   Speed     (regression, MSELoss)
#   Throttle  (regression, weighted MSELoss with histogram sample weights)
#   Gear      (classification, CrossEntropyLoss)
#
# Architecture: input -> 256 -> ReLU -> Dropout -> 128 -> ReLU -> Dropout -> head
#               (identical trunk across all targets; head is Linear(1) for
#                regression, Linear(num_classes) for classification)
#
# Usage:
#   python train.py --target rpm
#   python train.py --target speed
#   python train.py --target throttle
#   python train.py --target gear
#
# All configuration is read from config.py in this directory.
# =============================================================================

import os
import sys
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
import joblib
import h5py
from tqdm import tqdm
import csv

from config import (
    OUTPUT_DIR, get_paths,
    TRAIN_FILES,
    FEATURE_DIM, HOP_SIZE_S, CONTEXT_WINDOW_S,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, EARLY_STOPPING_PATIENCE,
    VAL_SPLIT, RANDOM_STATE,
    HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE,
    THROTTLE_BINS,
)


# =============================================================================
# REPRODUCIBILITY: seed everything
# =============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # if multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(RANDOM_STATE)


# =============================================================================
# MODELS
# =============================================================================

class STL_Regression(nn.Module):
    """Trunk identical to MTL shared layers; single regression head."""
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
    """Trunk identical to MTL shared layers; single classification head."""
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
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True, path='best_model.pt',
                 is_classification=False):
        self.patience        = patience
        self.delta           = delta
        self.verbose         = verbose
        self.path            = path
        self.is_classification = is_classification
        self.counter         = 0
        self.best_loss       = None
        self.early_stop      = False
        self.best_train_loss = None
        self.best_val_loss   = None

    def __call__(self, val_loss, train_loss, model, num_classes=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(model, num_classes)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(model, num_classes)
            self.counter = 0

    def save_checkpoint(self, model, num_classes=None):
        save_dict = {'model': model.state_dict()}
        if self.is_classification and num_classes is not None:
            save_dict['num_gear_classes'] = num_classes
        torch.save(save_dict, self.path)

    def get_state(self):
        return {
            'counter':         self.counter,
            'best_loss':       self.best_loss,
            'early_stop':      self.early_stop,
            'best_train_loss': self.best_train_loss,
            'best_val_loss':   self.best_val_loss,
        }

    def set_state(self, state):
        self.counter         = state['counter']
        self.best_loss       = state['best_loss']
        self.early_stop      = state['early_stop']
        self.best_train_loss = state['best_train_loss']
        self.best_val_loss   = state['best_val_loss']


# =============================================================================
# DATASETS
# =============================================================================

class RegressionH5Dataset(Dataset):
    """Dataset for regression targets (rpm, speed, throttle)."""
    def __init__(self, df, context_padding, hop_size_s, target_col, return_weight=False):
        self.df              = df.reset_index(drop=True)
        self.context_padding = context_padding
        self.hop_size_s      = hop_size_s
        self.target_col      = target_col
        self.return_weight   = return_weight
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

        label = np.float32(row[self.target_col])
        if self.target_col == 'target_throttle':
            label = np.float32(np.clip(label, 0, 100))

        feat_tensor = torch.from_numpy(window.flatten())
        label_tensor = torch.tensor(label)

        if self.return_weight:
            weight = torch.tensor(row['throttle_weight'], dtype=torch.float32)
            return feat_tensor, label_tensor, weight
        return feat_tensor, label_tensor


class ClassificationH5Dataset(Dataset):
    """Dataset for classification targets (gear)."""
    def __init__(self, df, context_padding, hop_size_s, target_col):
        self.df              = df.reset_index(drop=True)
        self.context_padding = context_padding
        self.hop_size_s      = hop_size_s
        self.target_col      = target_col
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
        label = int(row[self.target_col])
        return torch.from_numpy(window.flatten()), torch.tensor(label, dtype=torch.long)


class ScaledRegressionDataset(Dataset):
    """Wraps a regression dataset with input + target scaling."""
    def __init__(self, base_ds, in_scaler, out_scaler):
        self.base_ds    = base_ds
        self.in_scaler  = in_scaler
        self.out_scaler = out_scaler
        self.return_weight = base_ds.return_weight

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        if self.return_weight:
            f, l, w = self.base_ds[idx]
            f_s = self.in_scaler.transform(f.reshape(1, -1)).flatten()
            l_s = float(self.out_scaler.transform([[l.item()]])[0, 0])
            return torch.from_numpy(f_s).float(), torch.tensor(l_s, dtype=torch.float32), w
        else:
            f, l = self.base_ds[idx]
            f_s = self.in_scaler.transform(f.reshape(1, -1)).flatten()
            l_s = float(self.out_scaler.transform([[l.item()]])[0, 0])
            return torch.from_numpy(f_s).float(), torch.tensor(l_s, dtype=torch.float32)


class ScaledClassificationDataset(Dataset):
    """Wraps a classification dataset with input scaling only."""
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
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='STL training script')
    parser.add_argument('--target', required=True,
                        choices=['rpm', 'speed', 'throttle', 'gear'],
                        help='Target to train')
    args = parser.parse_args()

    target_name = args.target
    is_classification = (target_name == 'gear')
    is_throttle = (target_name == 'throttle')
    target_col = f'target_{target_name}'

    paths = get_paths(target_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== STL Training: {target_name.upper()} ===")

    # --- Load data ---
    print("Loading training data...")
    all_dfs = []
    for entry in TRAIN_FILES:
        temp_df = pd.read_parquet(entry['manifest'])
        temp_df['h5_source_path'] = entry['features']
        all_dfs.append(temp_df)
    full_df = pd.concat(all_dfs, ignore_index=True)
    train_df, val_df = train_test_split(full_df, test_size=VAL_SPLIT, random_state=RANDOM_STATE)
    print(f"Total: {len(full_df)} | Train: {len(train_df)} | Val: {len(val_df)}")

    # --- Compute context window ---
    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * ((padding * 2) + 1)
    print(f"Context window: {CONTEXT_WINDOW_S}s | Padding: {padding} frames | Input dim: {input_dim}")

    # --- Throttle histogram sample weights (throttle target only) ---
    if is_throttle:
        print("Computing throttle sample weights...")
        throttle_bins = np.linspace(0, 100, THROTTLE_BINS + 1)
        counts, _     = np.histogram(train_df['target_throttle'], bins=throttle_bins)
        counts        = np.maximum(counts, 1)
        bin_indices   = np.clip(np.digitize(train_df['target_throttle'], bins=throttle_bins) - 1,
                                0, len(counts) - 1)
        weights       = 1.0 / counts[bin_indices]
        weights       = weights / np.mean(weights)
        train_df = train_df.copy(); val_df = val_df.copy()
        train_df['throttle_weight'] = weights
        val_df['throttle_weight']   = 1.0

    # --- Build raw datasets ---
    if is_classification:
        num_gear_classes = int(full_df[target_col].max() + 1)
        print(f"Gear classes: {num_gear_classes}")
        train_raw_ds = ClassificationH5Dataset(train_df, padding, HOP_SIZE_S, target_col)
        val_raw_ds   = ClassificationH5Dataset(val_df,   padding, HOP_SIZE_S, target_col)
    else:
        train_raw_ds = RegressionH5Dataset(train_df, padding, HOP_SIZE_S, target_col,
                                           return_weight=is_throttle)
        val_raw_ds   = RegressionH5Dataset(val_df,   padding, HOP_SIZE_S, target_col,
                                           return_weight=is_throttle)

    # --- Scalers ---
    INPUT_SCALER_PATH  = paths['input_scaler']
    TARGET_SCALER_PATH = paths.get('target_scaler')

    if is_classification:
        # Classification: input scaler only
        if not os.path.exists(INPUT_SCALER_PATH):
            print("Fitting input scaler on training data...")
            in_scaler = StandardScaler()
            sample_idx = np.random.choice(len(train_raw_ds), min(50000, len(train_raw_ds)), replace=False)
            for i in tqdm(sample_idx, desc="Fitting scaler"):
                f, _ = train_raw_ds[i]
                in_scaler.partial_fit(f.reshape(1, -1))
            joblib.dump(in_scaler, INPUT_SCALER_PATH)
            print("Scaler saved.")
        else:
            print("Loading existing scaler...")
            in_scaler = joblib.load(INPUT_SCALER_PATH)
        out_scaler = None
    else:
        # Regression: input + target scalers
        if not os.path.exists(INPUT_SCALER_PATH) or not os.path.exists(TARGET_SCALER_PATH):
            print("Fitting scalers on training data...")
            in_scaler, out_scaler = StandardScaler(), StandardScaler()
            target_samples = []
            sample_idx = np.random.choice(len(train_raw_ds), min(50000, len(train_raw_ds)), replace=False)
            for i in tqdm(sample_idx, desc="Fitting scalers"):
                if is_throttle:
                    f, l, _ = train_raw_ds[i]
                else:
                    f, l = train_raw_ds[i]
                in_scaler.partial_fit(f.reshape(1, -1))
                target_samples.append([l.item()])
            out_scaler.fit(np.array(target_samples))
            joblib.dump(in_scaler,  INPUT_SCALER_PATH)
            joblib.dump(out_scaler, TARGET_SCALER_PATH)
            print("Scalers saved.")
        else:
            print("Loading existing scalers...")
            in_scaler  = joblib.load(INPUT_SCALER_PATH)
            out_scaler = joblib.load(TARGET_SCALER_PATH)

    # --- DataLoaders with reproducible worker initialisation ---
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    if is_classification:
        train_ds = ScaledClassificationDataset(train_raw_ds, in_scaler)
        val_ds   = ScaledClassificationDataset(val_raw_ds,   in_scaler)
    else:
        train_ds = ScaledRegressionDataset(train_raw_ds, in_scaler, out_scaler)
        val_ds   = ScaledRegressionDataset(val_raw_ds,   in_scaler, out_scaler)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8,
        worker_init_fn=worker_init_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8,
        worker_init_fn=worker_init_fn, generator=g,
    )

    # --- Model, optimiser ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if is_classification:
        model = STL_Classification(input_dim, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE,
                                   num_gear_classes).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = STL_Regression(input_dim, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE).to(device)
        criterion = nn.MSELoss()

    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    amp_scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE, path=paths['best_model'],
        is_classification=is_classification,
    )
    log_history = []
    start_epoch = 0

    # --- Resume ---
    RESUME_CHECKPOINT_PATH = paths['resume_checkpoint']
    if os.path.exists(RESUME_CHECKPOINT_PATH):
        print(f"\nResume checkpoint found: {RESUME_CHECKPOINT_PATH}")
        ckpt = torch.load(RESUME_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        early_stopping.set_state(ckpt['early_stopping'])
        log_history = ckpt['log_history']
        start_epoch = ckpt['epoch']
        print(f"Resuming from epoch {start_epoch + 1} / {EPOCHS}")
        print(f"EarlyStopping: counter={early_stopping.counter}, best_val_loss={early_stopping.best_loss:.4f}")
    else:
        print("\nNo resume checkpoint found. Starting from scratch.")

    # --- Training loop ---
    for epoch in range(start_epoch, EPOCHS):

        # ---- Train ----
        model.train()
        train_loss = 0.0

        if is_classification:
            for f, l in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                f, l = f.to(device), l.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(f), l)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                train_loss += loss.item()
        elif is_throttle:
            for f, l, w in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                f, l, w = f.to(device), l.to(device), w.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    pred = model(f).squeeze()
                    se   = (pred - l) ** 2
                    loss = (w * se).sum() / w.sum()
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                train_loss += loss.item()
        else:
            for f, l in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                f, l = f.to(device), l.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(f).squeeze(), l)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
                train_loss += loss.item()

        # ---- Validate ----
        model.eval()
        val_loss = 0.0

        if is_classification:
            v_true, v_pred = [], []
            with torch.no_grad():
                for f, l in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    f, l = f.to(device), l.to(device)
                    with torch.amp.autocast('cuda'):
                        logits = model(f)
                        val_loss += criterion(logits, l).item()
                    v_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    v_true.extend(l.cpu().numpy())
        else:
            v_true, v_pred = [], []
            with torch.no_grad():
                if is_throttle:
                    for f, l, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                        f, l = f.to(device), l.to(device)
                        with torch.amp.autocast('cuda'):
                            pred = model(f).squeeze()
                            val_loss += F.mse_loss(pred, l).item()
                        inv_pred = out_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()
                        inv_true = out_scaler.inverse_transform(l.cpu().numpy().reshape(-1, 1)).flatten()
                        v_pred.extend(np.clip(inv_pred, 0, 100))
                        v_true.extend(np.clip(inv_true, 0, 100))
                else:
                    for f, l in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                        f, l = f.to(device), l.to(device)
                        with torch.amp.autocast('cuda'):
                            pred = model(f).squeeze()
                            val_loss += criterion(pred, l).item()
                        inv_pred = out_scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()
                        inv_true = out_scaler.inverse_transform(l.cpu().numpy().reshape(-1, 1)).flatten()
                        v_pred.extend(inv_pred); v_true.extend(inv_true)

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        # ---- Logging ----
        if is_classification:
            val_acc = accuracy_score(v_true, v_pred)
            val_f1  = f1_score(v_true, v_pred, average='macro', zero_division=0)
            epoch_log = {
                'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                'val_acc': val_acc, 'val_f1_macro': val_f1,
            }
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{EPOCHS}  [{target_name.upper()}]")
            print(f"{'='*60}")
            print(f"TRAIN LOSS     | {train_loss:.4f}")
            print(f"VAL LOSS       | {val_loss:.4f}")
            print(f"VAL ACC        | {val_acc:.4f}")
            print(f"VAL F1 (macro) | {val_f1:.4f}")
        else:
            val_r2 = r2_score(v_true, v_pred)
            epoch_log = {
                'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
                f'val_R2_{target_name}': val_r2,
            }
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{EPOCHS}  [{target_name.upper()}]")
            print(f"{'='*60}")
            print(f"TRAIN LOSS | {train_loss:.4f}")
            print(f"VAL LOSS   | {val_loss:.4f}")
            print(f"VAL R2     | {val_r2:.4f}")

        log_history.append(epoch_log)

        early_stopping(val_loss, train_loss, model,
                       num_classes=num_gear_classes if is_classification else None)

        LOG_FILENAME = paths['log_filename']
        write_header = not os.path.exists(LOG_FILENAME)
        with open(LOG_FILENAME, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=epoch_log.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(epoch_log)

        save_dict = {
            'epoch':          epoch + 1,
            'model':          model.state_dict(),
            'optimizer':      optimizer.state_dict(),
            'amp_scaler':     amp_scaler.state_dict(),
            'early_stopping': early_stopping.get_state(),
            'log_history':    log_history,
        }
        if is_classification:
            save_dict['num_gear_classes'] = num_gear_classes
        torch.save(save_dict, RESUME_CHECKPOINT_PATH)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- Finish ---
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE  [{target_name.upper()}]")
    print(f"{'='*60}")
    ckpt = torch.load(paths['best_model'])
    model.load_state_dict(ckpt['model'])
    print(f"Best Model | Train Loss: {early_stopping.best_train_loss:.4f} | Val Loss: {early_stopping.best_val_loss:.4f}")

    COMPLETE_FLAG_PATH = paths['complete_flag']
    with open(COMPLETE_FLAG_PATH, 'w') as fh:
        fh.write(f"Training complete. Best val loss: {early_stopping.best_val_loss:.4f}\n")
    print(f"Completion flag written: {COMPLETE_FLAG_PATH}")

    if os.path.exists(RESUME_CHECKPOINT_PATH):
        os.remove(RESUME_CHECKPOINT_PATH)
        print("Resume checkpoint deleted.")


if __name__ == '__main__':
    main()
