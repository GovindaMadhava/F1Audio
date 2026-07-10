# =============================================================================
# config.py  –  Configuration for the STL Team Classification baseline
#
# Task:   10-way team classification
# Model:  MLP with same architecture as STL gear but num_classes = num_teams
# Label:  'team' column from the manifest parquet (string → integer via
#         LabelEncoder fit on training data, saved for test reuse).
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1. VERSION TAG & OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
MODEL_VERSION_TAG = 'STL_TEAM_1s'

# ── USER: Set this to the directory where model files and logs will be saved.
OUTPUT_DIR = './output/stl_team'

# -----------------------------------------------------------------------------
# 2. PER-TARGET DERIVED PATHS
# -----------------------------------------------------------------------------
def get_paths(target_name):
    tag = f'{MODEL_VERSION_TAG}_{target_name}'
    return {
        'input_scaler':      os.path.join(OUTPUT_DIR, f'input_scaler_{tag}.joblib'),
        'label_encoder':     os.path.join(OUTPUT_DIR, f'label_encoder_{tag}.joblib'),
        'best_model':        os.path.join(OUTPUT_DIR, f'best_model_{tag}.pt'),
        'log_filename':      os.path.join(OUTPUT_DIR, f'training_log_{tag}.csv'),
        'test_results':      os.path.join(OUTPUT_DIR, f'test_results_{tag}.pkl'),
        'resume_checkpoint': os.path.join(OUTPUT_DIR, f'resume_checkpoint_{tag}.pt'),
        'complete_flag':     os.path.join(OUTPUT_DIR, f'training_complete_{tag}.flag'),
    }

# -----------------------------------------------------------------------------
# 3. TRAINING + VALIDATION FILES
#
# ── USER: Edit these paths to point to your extracted feature files.
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
# 5. FEATURE / AUDIO PARAMETERS
# -----------------------------------------------------------------------------
FEATURE_DIM      = 1295
HOP_SIZE_S       = 0.5

# ── USER: CHANGE THIS to run context-window ablations.
CONTEXT_WINDOW_S = 1.0

# -----------------------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
EPOCHS                  = 50
BATCH_SIZE              = 256
LEARNING_RATE           = 0.001
EARLY_STOPPING_PATIENCE = 5
VAL_SPLIT               = 0.2
RANDOM_STATE            = 42

HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT_RATE = 0.2

# -----------------------------------------------------------------------------
# 7. TEST / INFERENCE PARAMETERS
# -----------------------------------------------------------------------------
TEST_BATCH_SIZE = 128
