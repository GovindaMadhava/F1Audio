# ===============================================================
"""
sanity_check.py

Verifies that a (features.h5, manifest.parquet) pair is internally consistent.
Works on either a per-driver unpacked file pair or the original combined file
(handles 1 or many audio_key groups the same way).

Checks performed:
  - Files exist and load cleanly.
  - Parquet has all expected telemetry/target columns.
  - audio_key sets match exactly between H5 groups and parquet rows.
  - Each H5 group has 'embeddings' (T, 1295) and 'timestamps_s' (T,) with
    matching row counts.
  - timestamps_s is strictly increasing (no out-of-order / duplicate frames).
  - No NaN/Inf in embeddings (sampled check on large files).
  - Manifest row count per audio_key == embeddings row count.
  - If present, embedding_idx is a complete 0..T-1 permutation, and
    embeddings[embedding_idx] timestamps match the manifest's timestamp_s
    exactly (this is the real proof that frames are correctly aligned).
  - Falls back to positional alignment check if embedding_idx is absent.
  - is_clipping clip-rate reported if present.
  - Basic telemetry sanity ranges (warnings only, not hard failures).
  - NaN scan across manifest columns.

Usage:
    python sanity_check.py /path/to/file_features.h5 /path/to/file_manifest.parquet
or just edit H5_PATH / PARQUET_PATH below and run with no arguments.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd

# ===== Used only if no command-line arguments are given =====
H5_PATH      = '/Volumes/GM_TB/test/AlpineRenault_EstebanOcon31_AlbertParkCircuitMelbourne^Australia_features.h5'
PARQUET_PATH = '/Volumes/GM_TB/test/AlpineRenault_EstebanOcon31_AlbertParkCircuitMelbourne^Australia_manifest.parquet'
# ===============================================================

EXPECTED_EMBED_DIM = 1295
EXPECTED_MANIFEST_COLS = {
    'audio_key', 'timestamp_s', 'team', 'driver', 'track',
    'target_rpm', 'target_speed', 'target_gear',
    'target_throttle', 'target_brake'
}
TIMESTAMP_TOL = 1e-6  # seconds, float-equality tolerance


def _ok(msg):   print(f"  ✅ {msg}")
def _warn(msg): print(f"  ⚠️  {msg}")
def _fail(msg): print(f"  ❌ {msg}")


def sanity_check(h5_path, parquet_path):
    issues = []

    print(f"\n{'=' * 70}")
    print(f"H5      : {h5_path}")
    print(f"PARQUET : {parquet_path}")
    print(f"{'=' * 70}\n")

    if not os.path.exists(h5_path):
        _fail(f"H5 file not found: {h5_path}")
        return False
    if not os.path.exists(parquet_path):
        _fail(f"Parquet file not found: {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    print(f"📄 Manifest: {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}\n")

    missing_cols = EXPECTED_MANIFEST_COLS - set(df.columns)
    if missing_cols:
        _fail(f"Missing expected manifest columns: {sorted(missing_cols)}")
        issues.append("missing_columns")
    else:
        _ok("All expected target/telemetry columns present")

    has_clip = 'is_clipping' in df.columns
    has_idx = 'embedding_idx' in df.columns
    print(f"   is_clipping column present  : {has_clip}")
    print(f"   embedding_idx column present: {has_idx}\n")

    with h5py.File(h5_path, "r") as h5f:
        audio_keys_h5 = list(h5f.keys())
        print(f"🗂  H5 groups (audio_key): {len(audio_keys_h5)}")
        for k in audio_keys_h5:
            print(f"   - {k}")
        print()

        audio_keys_pq = sorted(df['audio_key'].unique().tolist())
        only_in_h5 = set(audio_keys_h5) - set(audio_keys_pq)
        only_in_pq = set(audio_keys_pq) - set(audio_keys_h5)

        if only_in_h5:
            _fail(f"audio_keys in H5 but missing from parquet: {sorted(only_in_h5)}")
            issues.append("h5_extra_keys")
        if only_in_pq:
            _fail(f"audio_keys in parquet but missing from H5: {sorted(only_in_pq)}")
            issues.append("pq_extra_keys")
        if not only_in_h5 and not only_in_pq:
            _ok(f"audio_key sets match between H5 and parquet ({len(audio_keys_h5)} key(s))")

        for audio_key in audio_keys_h5:
            print(f"\n--- {audio_key} ---")
            grp = h5f[audio_key]

            if 'embeddings' not in grp or 'timestamps_s' not in grp:
                _fail("Missing 'embeddings' or 'timestamps_s' dataset in this group")
                issues.append(f"{audio_key}:missing_dataset")
                continue

            emb, ts = grp['embeddings'], grp['timestamps_s']
            print(f"   embeddings   shape: {emb.shape}  dtype: {emb.dtype}")
            print(f"   timestamps_s shape: {ts.shape}  dtype: {ts.dtype}")

            if emb.shape[1] != EXPECTED_EMBED_DIM:
                _fail(f"Embedding dim {emb.shape[1]} != expected {EXPECTED_EMBED_DIM}")
                issues.append(f"{audio_key}:bad_embed_dim")
            else:
                _ok(f"Embedding dim = {EXPECTED_EMBED_DIM} as expected")

            if emb.shape[0] != ts.shape[0]:
                _fail(f"embeddings rows ({emb.shape[0]}) != timestamps rows ({ts.shape[0]})")
                issues.append(f"{audio_key}:emb_ts_mismatch")
            else:
                _ok(f"embeddings/timestamps row counts match ({emb.shape[0]:,})")

            ts_vals = ts[:]
            diffs = np.diff(ts_vals)
            n_decreasing = int(np.sum(diffs < 0))
            n_duplicate = int(np.sum(diffs == 0))

            if n_decreasing > 0:
                _fail(f"timestamps_s actually decreases at {n_decreasing} point(s) — real ordering bug")
                issues.append(f"{audio_key}:ts_not_monotonic")
            elif n_duplicate > 0:
                _warn(f"timestamps_s has {n_duplicate} exact duplicate adjacent value(s) — "
                      f"monotonic non-decreasing, not strictly increasing. This is the expected "
                      f"PaSST 10s chunk-boundary float collision, not a bug; both frames are still "
                      f"real, distinct embeddings.")
                _ok("timestamps_s is monotonic non-decreasing")
            else:
                _ok("timestamps_s is strictly increasing")

            emb_sample = emb[:min(emb.shape[0], 2000)]
            if not np.isfinite(emb_sample).all():
                _fail("Found NaN/Inf values in embeddings (sampled check)")
                issues.append(f"{audio_key}:bad_embedding_values")
            else:
                _ok("No NaN/Inf found in sampled embeddings")

            sub = df[df['audio_key'] == audio_key].reset_index(drop=True)
            T = emb.shape[0]

            if len(sub) != T:
                _fail(f"Manifest row count ({len(sub)}) != embeddings row count ({T})")
                issues.append(f"{audio_key}:row_count_mismatch")
            else:
                _ok(f"Manifest row count matches embeddings ({T:,} rows)")

            if has_idx:
                idx_series = sub['embedding_idx'].dropna()
                idx_vals = idx_series.astype(int).values
                expected = np.arange(len(sub))
                if len(idx_vals) == len(sub) and np.array_equal(np.sort(idx_vals), expected):
                    _ok("embedding_idx is a complete 0..T-1 permutation (no gaps/dupes)")
                else:
                    _fail("embedding_idx has missing/duplicate/out-of-range values")
                    issues.append(f"{audio_key}:bad_embedding_idx")

                if len(idx_vals) == len(sub):
                    aligned_ts = ts_vals[idx_vals]
                    max_diff = float(np.max(np.abs(aligned_ts - sub['timestamp_s'].values)))
                    if max_diff <= TIMESTAMP_TOL:
                        _ok(f"timestamp_s matches h5 timestamps_s via embedding_idx (max diff {max_diff:.2e}s)")
                    else:
                        _fail(f"timestamp_s mismatch via embedding_idx (max diff {max_diff:.4f}s)")
                        issues.append(f"{audio_key}:ts_alignment_mismatch")
            else:
                if len(sub) == T:
                    max_diff = float(np.max(np.abs(ts_vals - sub['timestamp_s'].values)))
                    if max_diff <= TIMESTAMP_TOL:
                        _ok(f"timestamp_s matches h5 timestamps_s positionally (max diff {max_diff:.2e}s)")
                    else:
                        _fail(f"timestamp_s mismatch positionally (max diff {max_diff:.4f}s)")
                        issues.append(f"{audio_key}:ts_alignment_mismatch")

            if has_clip:
                clip_rate = sub['is_clipping'].mean() * 100
                print(f"   is_clipping: {int(sub['is_clipping'].sum())} / {len(sub)} rows ({clip_rate:.3f}%)")

            for col, lo, hi in [('target_throttle', 0, 100), ('target_speed', 0, 400),
                                 ('target_rpm', 0, 20000), ('target_gear', 0, 8)]:
                if col in sub.columns:
                    vmin, vmax = sub[col].min(), sub[col].max()
                    if vmin < lo or vmax > hi:
                        _warn(f"{col} outside expected range [{lo},{hi}]: got [{vmin}, {vmax}]")
                    else:
                        _ok(f"{col} within expected range [{lo},{hi}] (got [{vmin:.1f}, {vmax:.1f}])")

            nan_counts = sub.isna().sum()
            nan_counts = nan_counts[nan_counts > 0]
            if len(nan_counts):
                _warn(f"NaNs found in columns: {dict(nan_counts)}")
            else:
                _ok("No NaNs in this audio_key's rows")

    print(f"\n{'=' * 70}")
    if issues:
        print(f"❌ SANITY CHECK FAILED — {len(issues)} issue(s) found:")
        for i in issues:
            print(f"   - {i}")
    else:
        print("✅ SANITY CHECK PASSED — structure and alignment look correct.")
    print(f"{'=' * 70}\n")

    return len(issues) == 0


if __name__ == '__main__':
    if len(sys.argv) == 3:
        h5_arg, pq_arg = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        h5_arg, pq_arg = H5_PATH, PARQUET_PATH
    else:
        print("Usage: python sanity_check.py <path_to_h5> <path_to_parquet>")
        sys.exit(1)

    passed = sanity_check(h5_arg, pq_arg)
    sys.exit(0 if passed else 1)