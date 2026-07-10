# =============================================================================
# config.py  –  Configuration for the Linear Probe Team Classifier
#
# Task:   10-way team classification using LogisticRegression
# Label:  'team' column from manifest parquet (string → integer via LabelEncoder)
# =============================================================================

import os

# -----------------------------------------------------------------------------
# 1. VERSION TAG & OUTPUT DIRECTORY
# -----------------------------------------------------------------------------
MODEL_VERSION_TAG = 'LPT_1s'

# ── USER: Set this to the directory where model files and logs will be saved.
OUTPUT_DIR = './output/linear_probe_team'

# -----------------------------------------------------------------------------
# 2. DERIVED OUTPUT PATHS
# -----------------------------------------------------------------------------
INPUT_SCALER_PATH  = os.path.join(OUTPUT_DIR, f'input_scaler_{MODEL_VERSION_TAG}.joblib')
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, f'label_encoder_{MODEL_VERSION_TAG}.joblib')
MODEL_TEAM_PATH    = os.path.join(OUTPUT_DIR, f'model_team_{MODEL_VERSION_TAG}.joblib')

LOG_FILENAME       = os.path.join(OUTPUT_DIR, f'training_log_{MODEL_VERSION_TAG}.csv')
TEST_RESULTS_PATH  = os.path.join(OUTPUT_DIR, f'test_results_{MODEL_VERSION_TAG}.pkl')

COMPLETE_FLAG_PATH     = os.path.join(OUTPUT_DIR, f'training_complete_{MODEL_VERSION_TAG}.flag')
RESUME_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f'resume_checkpoint_{MODEL_VERSION_TAG}.pkl')

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
FEATURE_DIM = 1295
HOP_SIZE_S  = 0.5

# ── USER: CHANGE THIS to run context-window ablations.
CONTEXT_WINDOW_S = 1.0

# -----------------------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS
# -----------------------------------------------------------------------------
VAL_SPLIT    = 0.2
RANDOM_STATE = 42

SCALER_SAMPLE_SIZE = 50_000

# LogisticRegression hyperparameters
LR_C            = 1.0
LR_MAX_ITER     = 1000
LR_SOLVER       = 'saga'
LR_PENALTY      = 'l2'
LR_CLASS_WEIGHT = 'balanced'

# -----------------------------------------------------------------------------
# 7. TEST / INFERENCE PARAMETERS
# -----------------------------------------------------------------------------
TEST_BATCH_SIZE = 128
