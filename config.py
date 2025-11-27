"""
Configuration file for the LANL Earthquake Prediction project.
Centralizes all key parameters, file paths, and constants.
"""

import os

# ============================================================================
# File Paths
# ============================================================================
DATA_DIR = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_FOLDER = 'test'
SCALER_PATH = 'scaler.pkl'
TRAIN_LOADER_PATH = 'train_loader.pth'
VAL_LOADER_PATH = 'val_loader.pth'

# Model checkpoint paths
BEST_MODEL_PATH = 'best_model.pth'
MLPER_MODEL_PATH = 'mlper_regression_model.pth'
MLPER_INSPIRED_MODEL_PATH = 'mlper_inspired_model.pth'

# Output paths
RESULTS_DIR = '.'
MODEL_COMPARISON_RESULTS = os.path.join(RESULTS_DIR, 'model_comparison_results.csv')
BEST_EPOCHS_SUMMARY = os.path.join(RESULTS_DIR, 'best_epochs_summary.csv')
TRAINING_HISTORIES = os.path.join(RESULTS_DIR, 'training_histories.json')
SUBMISSION_PATH = 'submission.csv'

# ============================================================================
# Data Parameters
# ============================================================================
MAIN_SEGMENT_SIZE = 150_000
SUB_SEGMENT_SIZE = 15_000
TEST_SIZE = 0.2  # Fraction of data to use for validation
RANDOM_SEED = 42

# Maximum rows to load from training data (set to None for full dataset)
# For development, you can limit this to speed up processing
MAX_TRAIN_ROWS = None  # Set to e.g., 60_000_000 for testing

# ============================================================================
# Model Training Parameters
# ============================================================================
# Default training parameters
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.001
BATCH_SIZE = 32
IMAGE_BATCH_SIZE = 8

# Long training experiment parameters
LONG_TRAINING_EPOCHS = 150
IMAGE_MODEL_EPOCHS = 100

# Best epochs from previous experiments (for final submission)
# These should be updated based on best_epochs_summary.csv results
BEST_EPOCHS = {
    'LSTM': 134,
    'Hybrid CNN-LSTM': 198,
    'MLPER-Inspired (Image)': 164,
    'Hybrid w/ Attention': 184,
    '1D CNN': 5,
}

# ============================================================================
# Model Architecture Parameters
# ============================================================================
# These will be determined automatically from feature engineering
# but can be overridden if needed
NUM_FEATURES = None  # Will be set automatically

# ============================================================================
# Image Model Parameters
# ============================================================================
# Spectrogram generation parameters (in image_transformer.py)
# These are kept here for reference but may be hardcoded in image_transformer.py
SPECTROGRAM_NFFT = 2048
SPECTROGRAM_HOP_LENGTH = 512

# ============================================================================
# Optimizer Configurations for Experiment Sweep
# ============================================================================
OPTIMIZER_CONFIGS = [
    {'name': 'Adam_OneCycleLR', 'optimizer': 'Adam', 'scheduler': 'OneCycleLR', 'lr': 0.001, 'weight_decay': 1e-5},
    {'name': 'Adam_StaticLR', 'optimizer': 'Adam', 'scheduler': None, 'lr': 0.001, 'weight_decay': 0.0},
    {'name': 'AdamW_OneCycleLR', 'optimizer': 'AdamW', 'scheduler': 'OneCycleLR', 'lr': 0.001, 'weight_decay': 0.01},
    {'name': 'SGD_Momentum_OneCycleLR', 'optimizer': 'SGD', 'scheduler': 'OneCycleLR', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
    {'name': 'Adam_CyclicLR', 'optimizer': 'Adam', 'scheduler': 'CyclicLR', 'lr': 1e-4, 'max_lr': 1e-2, 'weight_decay': 1e-5}
]

# ============================================================================
# Utility Functions
# ============================================================================
def get_data_path(path_key):
    """Get a data path by key."""
    paths = {
        'train': TRAIN_DATA_PATH,
        'test': TEST_FOLDER,
        'scaler': SCALER_PATH,
        'val_loader': VAL_LOADER_PATH,
    }
    return paths.get(path_key, path_key)

def ensure_dir(path):
    """Ensure a directory exists."""
    dir_path = os.path.dirname(path) if os.path.isfile(path) else path
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

