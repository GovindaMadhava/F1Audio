import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import h5py
import json
import warnings
from tqdm import tqdm
from hear21passt.base import load_model, get_timestamp_embeddings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===== CONFIGURATION =====
INPUT_DIR  = '/Volumes/GM_TB/_Formula1_dataset_MIG/RedBullRingSpielberg^Austria'
OUTPUT_DIR = '/Volumes/GM_TB/_passt_timestamps'

TARGET_SR       = 32000
TARGET_RMS      = 0.1

# PaSST internal constants (do NOT change these)
PASST_WINDOW_S  = 10.0          # PaSST's positional encoding is trained for 10-second clips
PASST_HOP_S     = 0.05          # get_timestamp_embeddings returns one embedding every 50 ms
PASST_HOP_SAMP  = int(PASST_HOP_S * TARGET_SR)   # = 1600 samples per embedding step
PASST_WIN_SAMP  = int(PASST_WINDOW_S * TARGET_SR) # = 320000 samples per chunk fed to model

BATCH_SIZE      = 8  # number of 10-second chunks processed per GPU call — tune to VRAM
# =========================


def format_duration(seconds):
    s = int(seconds)
    hrs, s = divmod(s, 3600)
    mins, s = divmod(s, 60)
    return f"{hrs}h {mins}m {s}s"


def load_and_preprocess_audio(wav_path):
    """
    Load a WAV file, downmix to mono, resample to 32 kHz,
    apply global RMS normalisation, and hard-clip to [-1, 1].

    Returns: waveform tensor of shape (1, N_samples)
    """
    waveform, sr_native = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr_native != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr_native, TARGET_SR)
        waveform = resampler(waveform)

    # Global RMS normalisation — applied to the whole file so that relative
    # within-race dynamics (pit lane vs full throttle) are preserved.
    curr_rms = torch.sqrt(torch.mean(waveform ** 2))
    if curr_rms > 0:
        gain = TARGET_RMS / (curr_rms + 1e-8)
        waveform = waveform * gain

    waveform = torch.clamp(waveform, -1.0, 1.0)
    return waveform  # shape: (1, N)


def extract_timestamp_embeddings_sliding(waveform, model, device):
    """
    Extract PaSST timestamp embeddings from an arbitrarily long waveform
    by sliding a 10-second window with 50 ms steps.

    Why we cannot pass the full 90-minute waveform directly
    -------------------------------------------------------
    get_timestamp_embeddings internally feeds the audio to the PaSST
    transformer using a fixed-length time positional encoding trained on
    10-second clips.  Feeding a 5,400-second waveform would require
    5,400 / 10 = 540× more positional slots than the model has — the
    positional encoding would wrap or be repeated, producing poor
    embeddings for anything beyond the first 10 seconds.

    How the sliding window works
    ----------------------------
    We cut the waveform into PASST_WIN_SAMP (= 10 s × 32000 = 320,000)
    sample chunks, feed them in batches to get_timestamp_embeddings, and
    concatenate the resulting embedding arrays.

    Hop size is 10 s (= PASST_WIN_SAMP samples), meaning windows are
    NON-overlapping at the chunk level.  Within each 10-second chunk,
    get_timestamp_embeddings already returns one 1295-d embedding every
    50 ms (PASST_HOP_S), giving 200 embeddings per chunk.

    Why non-overlapping chunks are fine here
    ----------------------------------------
    Each 10-s chunk is internally processed by PaSST using its own
    50 ms sliding receptive-field windows (with 160 ms context each),
    so the dense 50 ms grid is fully preserved inside every chunk.
    The only place where context is slightly missing is at the very
    boundary between two consecutive chunks (the last ~80 ms of chunk N
    and first ~80 ms of chunk N+1 do not share context).  Over a 90-
    minute race this affects < 0.3 % of all embeddings and is negligible
    for the downstream regression tasks we are targeting.

    If you need seamless boundary context, use overlapping chunks
    (e.g. 10 s window, 9.95 s hop) and discard the first / last
    embeddings of each chunk except the first/last chunk — but this
    increases computation ~200×.

    Returns
    -------
    all_embeddings : np.ndarray, shape (T_total, 1295)
    all_timestamps : np.ndarray, shape (T_total,)  — in seconds
    """
    n_samples = waveform.shape[1]

    # --- Build list of (start, end) sample indices for each 10-s chunk ---
    chunk_starts = list(range(0, n_samples, PASST_WIN_SAMP))

    all_embeddings = []
    all_timestamps = []

    # Offset (in seconds) of the start of the current chunk within the full file.
    # We use this to convert the chunk-local timestamps returned by
    # get_timestamp_embeddings (which always start near 0 ms) into
    # global file timestamps.
    time_offset_s = 0.0

    # Process chunks in mini-batches to saturate the GPU without OOM.
    for batch_start in range(0, len(chunk_starts), BATCH_SIZE):
        batch_chunk_starts = chunk_starts[batch_start : batch_start + BATCH_SIZE]

        # ---- Assemble a batch of 10-s tensors, zero-padding the last chunk ----
        padded_batch = []
        actual_lengths = []  # number of REAL samples in each chunk (before padding)

        for cs in batch_chunk_starts:
            chunk = waveform[:, cs : cs + PASST_WIN_SAMP]  # (1, <=320000)
            actual_lengths.append(chunk.shape[1])

            if chunk.shape[1] < PASST_WIN_SAMP:
                # Last chunk: zero-pad on the right to 10 s
                pad_len = PASST_WIN_SAMP - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            padded_batch.append(chunk)

        # batch shape: (B, 1, 320000)  →  get_timestamp_embeddings expects (B, 320000)
        batch_tensor = torch.cat(padded_batch, dim=0).to(device)  # (B, 320000)

        with torch.no_grad():
            # emb: (B, n_ts_per_chunk, 1295)
            # ts:  (B, n_ts_per_chunk)  — in milliseconds, local to each chunk
            emb, ts = get_timestamp_embeddings(batch_tensor, model)

        emb = emb.cpu().numpy()  # (B, n_ts, 1295)
        ts  = ts.cpu().numpy()   # (B, n_ts)  — milliseconds

        # ---- Flatten batch dimension, drop padded embeddings, fix timestamps ----
        for i, (cs, actual_len) in enumerate(zip(batch_chunk_starts, actual_lengths)):
            chunk_emb = emb[i]  # (n_ts, 1295)
            chunk_ts  = ts[i]   # (n_ts,) in ms

            # How many embeddings correspond to real audio?
            # Each embedding covers up to PASST_HOP_S seconds of audio.
            # The last valid embedding center must be <= actual_audio_duration.
            actual_duration_s = actual_len / TARGET_SR
            valid_mask = (chunk_ts / 1000.0) <= actual_duration_s
            chunk_emb = chunk_emb[valid_mask]
            chunk_ts  = chunk_ts[valid_mask]

            # Convert chunk-local ms timestamps → global file timestamps in seconds
            global_ts_s = (chunk_ts / 1000.0) + time_offset_s

            all_embeddings.append(chunk_emb)
            all_timestamps.append(global_ts_s)

            time_offset_s += actual_duration_s

    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (T_total, 1295)
    all_timestamps = np.concatenate(all_timestamps, axis=0)  # (T_total,)

    return all_embeddings, all_timestamps


def align_telemetry(csv_path, timestamps_s):
    """
    Interpolate telemetry CSV columns onto the embedding timestamps.

    The CSV is indexed by 'RaceTimeSeconds'. We reindex to the dense
    50 ms embedding grid using linear interpolation for continuous
    channels and nearest-neighbour for discrete ones (gear, brake).

    Returns a DataFrame with one row per embedding timestamp.
    """
    tele_df = pd.read_csv(csv_path)
    df_numeric = tele_df.set_index('RaceTimeSeconds').copy()
    df_numeric = df_numeric.loc[~df_numeric.index.duplicated(keep='first')]

    # Union can produce float duplicates when a telemetry timestamp coincidentally
    # lands exactly on a 50 ms boundary. Deduplicate before interpolating.
    full_idx = df_numeric.index.union(timestamps_s)
    full_idx = full_idx[~full_idx.duplicated()]

    df_resampled = df_numeric.reindex(full_idx).infer_objects(copy=False)
    df_resampled = df_resampled.interpolate(method='linear').reindex(timestamps_s)

    for col in ['nGear', 'Brake']:
        if col in df_numeric.columns:
            interp = df_numeric[col].reindex(full_idx).infer_objects(copy=False)
            interp = interp.interpolate(method='nearest').reindex(timestamps_s)
            df_resampled[col] = interp.bfill().ffill().fillna(0)

            if col == 'nGear':
                df_resampled[col] = df_resampled[col].round().astype(int)
            else:
                df_resampled[col] = (df_resampled[col] > 0.5).astype(bool)

    df_resampled = df_resampled.bfill().ffill().fillna(0)
    return df_resampled


def process_unified_pipeline(input_dir, output_dir):
    input_dir = input_dir.rstrip(os.sep)
    dir_name  = os.path.basename(input_dir)
    os.makedirs(output_dir, exist_ok=True)

    h5_path      = os.path.join(output_dir, f"{dir_name}_features.h5")
    parquet_path = os.path.join(output_dir, f"{dir_name}_manifest.parquet")
    json_path    = os.path.join(output_dir, f"{dir_name}_summary.json")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"🚀  Starting extraction: {dir_name}  |  device: {device}")
    print(f"📐  Embedding resolution: {PASST_HOP_S * 1000:.0f} ms per embedding "
          f"({1 / PASST_HOP_S:.0f} embeddings/sec)")

    model = load_model().to(device)
    model.eval()

    wav_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith('.wav') and not f.startswith('._')
    ])

    manifest_accumulator    = []
    total_audio_duration_s  = 0.0

    with h5py.File(h5_path, "w") as h5f:
        for wav_filename in tqdm(wav_files, desc="Files"):
            basename = os.path.splitext(wav_filename)[0]
            csv_path = os.path.join(input_dir, f"{basename}.csv")
            wav_path = os.path.join(input_dir, wav_filename)

            print(f"\n📂  {wav_filename}")
            if not os.path.exists(csv_path):
                print(f"⚠️   CSV not found — skipping {wav_filename}")
                continue

            try:
                # ---- 1. Load + preprocess ----
                waveform = load_and_preprocess_audio(wav_path)
                curr_duration_s = waveform.shape[1] / TARGET_SR
                total_audio_duration_s += curr_duration_s
                print(f"    Audio loaded: {format_duration(curr_duration_s)}  "
                      f"({waveform.shape[1]:,} samples)")

                # ---- 2. Extract dense timestamp embeddings ----
                print(f"    Extracting embeddings via 10-s sliding window …")
                features, timestamps_s = extract_timestamp_embeddings_sliding(
                    waveform, model, device
                )
                # features:     (T, 1295)
                # timestamps_s: (T,)  — global seconds within the audio file

                print(f"    Embeddings: {features.shape}  "
                      f"(1 per {PASST_HOP_S * 1000:.0f} ms)")

                # ---- 3. Store embeddings + timestamps in HDF5 ----
                grp = h5f.create_group(wav_filename)
                grp.create_dataset("embeddings",  data=features,      compression="gzip")
                grp.create_dataset("timestamps_s", data=timestamps_s, compression="gzip")

                # ---- 4. Align telemetry to the embedding grid ----
                df_resampled = align_telemetry(csv_path, timestamps_s)
                df_resampled.index.name = 'timestamp_s'
                df_resampled = df_resampled.reset_index()

                df_resampled = df_resampled.rename(columns={
                    'RPM': 'target_rpm', 'Speed': 'target_speed',
                    'nGear': 'target_gear', 'Throttle': 'target_throttle',
                    'Brake': 'target_brake'
                })

                df_resampled['audio_key'] = wav_filename
                parts = basename.split('_')
                df_resampled['team']   = parts[0] if len(parts) > 0 else 'unknown'
                df_resampled['driver'] = parts[1] if len(parts) > 1 else 'unknown'
                df_resampled['track']  = parts[2] if len(parts) > 2 else 'unknown'

                final_cols = [
                    'audio_key', 'timestamp_s', 'team', 'driver', 'track',
                    'target_rpm', 'target_speed', 'target_gear',
                    'target_throttle', 'target_brake'
                ]
                manifest_accumulator.append(df_resampled[final_cols])
                print(f"✅  Done: {wav_filename}")

            except Exception as e:
                print(f"\n❌  Error on {basename}: {e}")
                import traceback; traceback.print_exc()

    # ---- 5. Save manifest + summary ----
    if manifest_accumulator:
        final_df = pd.concat(manifest_accumulator, ignore_index=True)
        final_df.to_parquet(parquet_path, index=False)

        def split_summary(df, col):
            return {
                item: {
                    "total_samples": int(cnt),
                    "duration_formatted": format_duration(cnt * PASST_HOP_S)
                }
                for item, cnt in df.groupby(col).size().items()
            }

        summary = {
            "dataset_name":            dir_name,
            "total_audio_duration":    format_duration(total_audio_duration_s),
            "total_embeddings":        len(final_df),
            "embedding_dim":           1295,
            "parameters": {
                "embedding_hop_seconds":   PASST_HOP_S,
                "passt_window_seconds":    PASST_WINDOW_S,
                "sample_rate":             TARGET_SR,
                "target_rms":             TARGET_RMS,
            },
            "note": (
                "Each embedding is a 1295-d PaSST timestamp feature extracted "
                f"every {PASST_HOP_S * 1000:.0f} ms. Users may average any "
                "contiguous window of embeddings to obtain coarser-resolution "
                "features (e.g. 20 embeddings = 1 second)."
            ),
            "team_split":   split_summary(final_df, 'team'),
            "driver_split": split_summary(final_df, 'driver'),
            "track_split":  split_summary(final_df, 'track'),
        }

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"\n✅  All done. Output in: {output_dir}")
        print(f"   Total embeddings : {len(final_df):,}")
        print(f"   Total audio time : {format_duration(total_audio_duration_s)}")
    else:
        print("⚠️  No data processed.")


if __name__ == '__main__':
    process_unified_pipeline(INPUT_DIR, OUTPUT_DIR)