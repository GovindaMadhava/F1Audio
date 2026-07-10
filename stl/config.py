# =============================================================================
# config.py  –  Shared configuration for all 4 STL models
# Targets: RPM (reg), Speed (reg), Throttle (reg), Gear (classification)
#
# All train/test scripts in this directory import from here.
# Architecture, data, and hyperparameters are identical across all targets
# so that STL vs MTL comparisons are fair.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1. VERSION TAG & OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
MODEL_VERSION_TAG = 'STL_4s'

# ── USER: Set this to the directory where model checkpoints, scalers,
#          and training logs will be saved.
OUTPUT_DIR = './output/stl'

# -----------------------------------------------------------------------------
# 2. PER-TARGET DERIVED PATHS
#    Call get_paths('rpm') / get_paths('speed') / etc. to get all file paths
#    for a given target. Each target gets its own scaler, model, log, etc.
# -----------------------------------------------------------------------------
def get_paths(target_name):
    tag = f'{MODEL_VERSION_TAG}_{target_name}'
    return {
        'input_scaler':      os.path.join(OUTPUT_DIR, f'input_scaler_{tag}.joblib'),
        'target_scaler':     os.path.join(OUTPUT_DIR, f'target_scaler_{tag}.joblib'),
        'best_model':        os.path.join(OUTPUT_DIR, f'best_model_{tag}.pt'),
        'log_filename':      os.path.join(OUTPUT_DIR, f'training_log_{tag}.csv'),
        'test_results':      os.path.join(OUTPUT_DIR, f'test_results_{tag}.pkl'),
        'resume_checkpoint': os.path.join(OUTPUT_DIR, f'resume_checkpoint_{tag}.pt'),
        'complete_flag':     os.path.join(OUTPUT_DIR, f'training_complete_{tag}.flag'),
    }

# -----------------------------------------------------------------------------
# 3. TRAINING + VALIDATION FILES  (80/20 split done in train script)
#
# ── USER: Edit these paths to point to your extracted feature files.
#          Each entry needs a manifest .parquet and a features .h5 file,
#          as produced by dataset/extract_features.py.
# -----------------------------------------------------------------------------
TRAIN_FILES = [
    {'manifest': "data/features/RedBullRingSpielberg^Austria_manifest.parquet",
     'features': "data/features/RedBullRingSpielberg^Austria_features.h5"},

    {'manifest': "data/features/AlbertParkCircuitMelbourne^Australia_manifest.parquet",
     'features': "data/features/AlbertParkCircuitMelbourne^Australia_features.h5"},

    {'manifest': "data/features/AutodromoNazionaleMonza^Italy_manifest.parquet",
     'features': "data/features/AutodromoNazionaleMonza^Italy_features.h5"},

    {'manifest': "data/features/BakuCityCircuit^Azerbaijan_manifest.parquet",
     'features': "data/features/BakuCityCircuit^Azerbaijan_features.h5"},

    {'manifest': "data/features/HungaroringBudapest^Hungary_manifest.parquet",
     'features': "data/features/HungaroringBudapest^Hungary_features.h5"},

    {'manifest': "data/features/JeddahCornicheCircuit^SaudiArabia_manifest.parquet",
     'features': "data/features/JeddahCornicheCircuit^SaudiArabia_features.h5"},

    {'manifest': "data/features/CircuitdeBarcelonaCatalunya^Spain_manifest.parquet",
     'features': "data/features/CircuitdeBarcelonaCatalunya^Spain_features.h5"},

    {'manifest': "data/features/CircuitdeSpaFrancorchamps^Belgium_manifest.parquet",
     'features': "data/features/CircuitdeSpaFrancorchamps^Belgium_features.h5"},
]

# -----------------------------------------------------------------------------
# 4. TEST FILES  (held-out circuits)
#
# ── USER: Edit these paths to point to your held-out test feature files.
# -----------------------------------------------------------------------------
TEST_FILES = [
    {'manifest': "data/features/MiamiIntlAutodrome^USA_manifest.parquet",
     'features': "data/features/MiamiIntlAutodrome^USA_features.h5"},

    {'manifest': "data/features/BahrainInternationalCircuitSakhir^Bahrain_manifest.parquet",
     'features': "data/features/BahrainInternationalCircuitSakhir^Bahrain_features.h5"},
]

# -----------------------------------------------------------------------------
# 5. FEATURE / AUDIO PARAMETERS  (must match feature extraction settings)
# -----------------------------------------------------------------------------
FEATURE_DIM      = 1295
HOP_SIZE_S       = 0.5

# ── USER: CHANGE THIS to run context-window ablations.
#    Controls how many surrounding frames are concatenated with the current
#    frame before being fed to the model.
#    Examples:  1.0s -> padding=0, input_dim=1295   (current frame only)
#               4.0s -> padding=3, input_dim=9065   (7 frames)
#               9.0s -> padding=8, input_dim=22015  (17 frames)
CONTEXT_WINDOW_S = 4.0

# -----------------------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
EPOCHS                  = 50
BATCH_SIZE              = 256
LEARNING_RATE           = 0.001
EARLY_STOPPING_PATIENCE = 5
VAL_SPLIT               = 0.2
RANDOM_STATE            = 42

# Model architecture  (same dimensions as MTL shared trunk)
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT_RATE = 0.2

# Throttle histogram weighting (used only for throttle target)
THROTTLE_BINS = 20

# -----------------------------------------------------------------------------
# 7. TEST / INFERENCE PARAMETERS
# -----------------------------------------------------------------------------
TEST_BATCH_SIZE = 128
