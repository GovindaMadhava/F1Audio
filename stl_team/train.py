# =============================================================================
# train.py  –  STL Team Classification training script
#
# Task: 10-way team classification
# Architecture: input → 256 → ReLU → Dropout → 128 → ReLU → Dropout → Linear(num_classes)
#
# Class imbalance:
#   CrossEntropyLoss(weight=class_weights) where class_weights follow sklearn's
#   'balanced' strategy: w_c = n_samples / (n_classes * count_c).
#
# Usage:
#   python train.py
# =============================================================================

import os
import sys
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
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

TARGET_NAME = 'team'
PATHS = get_paths(TARGET_NAME)

INPUT_SCALER_PATH      = PATHS['input_scaler']
LABEL_ENCODER_PATH     = PATHS['label_encoder']
BEST_MODEL_PATH        = PATHS['best_model']
LOG_FILENAME           = PATHS['log_filename']
RESUME_CHECKPOINT_PATH = PATHS['resume_checkpoint']
COMPLETE_FLAG_PATH     = PATHS['complete_flag']


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
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True, path='best_model.pt'):
        self.patience        = patience
        self.delta           = delta
        self.verbose         = verbose
        self.path            = path
        self.counter         = 0
        self.best_loss       = None
        self.early_stop      = False
        self.best_train_loss = None
        self.best_val_loss   = None

    def __call__(self, val_loss, train_loss, model, num_classes):
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

    def save_checkpoint(self, model, num_classes):
        torch.save({'model': model.state_dict(), 'num_team_classes': num_classes}, self.path)

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
        label = int(row['target_team'])
        return torch.from_numpy(window.flatten()), torch.tensor(label, dtype=torch.long)


class ScaledDataset(Dataset):
    def __init__(self, base_ds, in_scaler):
        self.base_ds   = base_ds
        self.in_scaler = in_scaler

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        f, l = self.base_ds[idx]
        f_s = self.in_scaler.transform(f.reshape(-1, 1295)).flatten()
        return torch.from_numpy(f_s).float(), l


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== STL Training: {TARGET_NAME.upper()} ===")

    print("Loading training data...")
    all_dfs = []
    for entry in TRAIN_FILES:
        temp_df = pd.read_parquet(entry['manifest'])
        temp_df['h5_source_path'] = entry['features']
        all_dfs.append(temp_df)
    full_df = pd.concat(all_dfs, ignore_index=True)
    train_df, val_df = train_test_split(full_df, test_size=VAL_SPLIT, random_state=RANDOM_STATE)
    print(f"Total: {len(full_df)} | Train: {len(train_df)} | Val: {len(val_df)}")

    # --- LabelEncoder ---
    le = LabelEncoder()
    le.fit(train_df['team'])
    joblib.dump(le, LABEL_ENCODER_PATH)
    num_team_classes = len(le.classes_)
    print(f"LabelEncoder saved: {LABEL_ENCODER_PATH}")
    print(f"Team classes ({num_team_classes}): {list(le.classes_)}")

    train_df = train_df.copy()
    val_df   = val_df.copy()
    train_df['target_team'] = le.transform(train_df['team'])
    val_df['target_team']   = le.transform(val_df['team'])

    team_counts = train_df['target_team'].value_counts().sort_index()
    print("\nTeam distribution in training set:")
    for cls_idx, count in team_counts.items():
        print(f"  {le.classes_[cls_idx]:<25} {count:>8} rows")

    # --- Balanced class weights ---
    n_samples = len(train_df)
    class_counts = train_df['target_team'].value_counts().sort_index().values.astype(np.float64)
    class_weights = n_samples / (num_team_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"\nClass weights (balanced): {dict(zip(le.classes_, class_weights.round(4)))}")

    NUM_WORKERS = 0 if sys.platform == 'darwin' else 8

    padding   = int((CONTEXT_WINDOW_S - 1) / 2 / HOP_SIZE_S)
    input_dim = FEATURE_DIM * ((padding * 2) + 1)
    print(f"Context window: {CONTEXT_WINDOW_S}s | Padding: {padding} frames | Input dim: {input_dim}")

    train_raw_ds = MultiH5Dataset(train_df, padding, HOP_SIZE_S)
    val_raw_ds   = MultiH5Dataset(val_df,   padding, HOP_SIZE_S)

    # --- Input scaler ---
    if not os.path.exists(INPUT_SCALER_PATH):
        print("Fitting input scaler (random 50k sample)...")
        in_scaler  = StandardScaler()
        scaler_target = 50000

        all_keys = []
        for entry in TRAIN_FILES:
            with h5py.File(entry['features'], 'r') as h5f:
                for key in h5f.keys():
                    all_keys.append((entry['features'], key, h5f[key].shape[0]))

        total_rows = sum(n for _, _, n in all_keys)
        n_sample = min(scaler_target, total_rows)
        print(f"  {len(all_keys)} driver keys, {total_rows} total rows, sampling {n_sample}")

        rng = np.random.RandomState(RANDOM_STATE)
        chosen = rng.choice(total_rows, size=n_sample, replace=False)
        chosen.sort()

        cum = np.cumsum([n for _, _, n in all_keys])
        key_starts = np.concatenate([[0], cum[:-1]])

        collected = 0
        for ki, (h5_path, key, n_rows) in enumerate(all_keys):
            mask = (chosen >= key_starts[ki]) & (chosen < cum[ki])
            if not mask.any():
                continue
            local_idx = chosen[mask] - key_starts[ki]
            with h5py.File(h5_path, 'r') as h5f:
                ds = h5f[key]
                rows = ds[sorted(local_idx)]
                in_scaler.partial_fit(rows)
                collected += len(rows)
        joblib.dump(in_scaler, INPUT_SCALER_PATH)
        print(f"Scaler saved ({collected} samples across {len(all_keys)} keys).")
    else:
        print("Loading existing scaler...")
        in_scaler = joblib.load(INPUT_SCALER_PATH)

    # --- DataLoaders ---
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(RANDOM_STATE)

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        ScaledDataset(train_raw_ds, in_scaler),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=use_cuda, num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn, generator=g,
    )
    val_loader = DataLoader(
        ScaledDataset(val_raw_ds, in_scaler),
        batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=use_cuda, num_workers=NUM_WORKERS,
        worker_init_fn=worker_init_fn, generator=g,
    )

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    use_amp = (device.type == 'cuda')

    model      = STL_Classification(input_dim, HIDDEN_DIM_1, HIDDEN_DIM_2, DROPOUT_RATE, num_team_classes).to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    amp_scaler = torch.amp.GradScaler('cuda') if use_amp else None
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, path=BEST_MODEL_PATH)
    log_history    = []
    start_epoch    = 0

    # --- Resume ---
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

        model.train()
        train_loss = 0.0
        for f, l in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            f, l = f.to(device), l.to(device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(f), l)
                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss = criterion(model(f), l)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        v_true, v_pred = [], []
        with torch.no_grad():
            for f, l in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                f, l = f.to(device), l.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(f)
                        val_loss += criterion(logits, l).item()
                else:
                    logits = model(f)
                    val_loss += criterion(logits, l).item()
                v_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                v_true.extend(l.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        val_acc     = accuracy_score(v_true, v_pred)
        val_f1      = f1_score(v_true, v_pred, average='macro', zero_division=0)

        epoch_log = {
            'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_acc': val_acc, 'val_f1_macro': val_f1,
        }
        log_history.append(epoch_log)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}  [{TARGET_NAME.upper()}]")
        print(f"{'='*60}")
        print(f"TRAIN LOSS     | {train_loss:.4f}")
        print(f"VAL LOSS       | {val_loss:.4f}")
        print(f"VAL ACC        | {val_acc:.4f}")
        print(f"VAL F1 (macro) | {val_f1:.4f}")

        early_stopping(val_loss, train_loss, model, num_team_classes)

        write_header = not os.path.exists(LOG_FILENAME)
        with open(LOG_FILENAME, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=epoch_log.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(epoch_log)

        torch.save({
            'epoch':            epoch + 1,
            'model':            model.state_dict(),
            'optimizer':        optimizer.state_dict(),
            'amp_scaler':       amp_scaler.state_dict() if use_amp else {},
            'early_stopping':   early_stopping.get_state(),
            'log_history':      log_history,
            'num_team_classes': num_team_classes,
        }, RESUME_CHECKPOINT_PATH)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE  [{TARGET_NAME.upper()}]")
    print(f"{'='*60}")
    ckpt = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(ckpt['model'])
    print(f"Best Model | Train Loss: {early_stopping.best_train_loss:.4f} | Val Loss: {early_stopping.best_val_loss:.4f}")

    with open(COMPLETE_FLAG_PATH, 'w') as fh:
        fh.write(f"Training complete. Best val loss: {early_stopping.best_val_loss:.4f}\n")
    print(f"Completion flag written: {COMPLETE_FLAG_PATH}")

    if os.path.exists(RESUME_CHECKPOINT_PATH):
        os.remove(RESUME_CHECKPOINT_PATH)
        print("Resume checkpoint deleted.")


if __name__ == '__main__':
    main()
