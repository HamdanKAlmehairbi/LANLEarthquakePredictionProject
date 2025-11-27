"""
Centralized data preparation script.
This script loads the raw training data, performs feature engineering, scaling,
and train/validation split, then saves the processed data loaders and scaler.

This avoids redundant data loading in other scripts (main.py, double_descent_experiment.py, etc.)
"""

import pickle
import torch
import config
import utils
import data_loader
import data_pipeline


def prepare_and_save_dataset():
    """
    Main function to prepare the dataset and save processed components.
    
    This function:
    1. Loads the raw training data
    2. Performs feature engineering and scaling
    3. Creates train/validation split
    4. Saves train_loader, val_loader, and scaler for reuse
    
    Returns:
        train_loader, val_loader, num_features, feature_names, scaler
    """
    print("=" * 70)
    print("Dataset Preparation - Centralized Pipeline")
    print("=" * 70)
    
    # Set random seed for reproducibility
    utils.set_seed(config.RANDOM_SEED)
    
    # Load raw data
    print(f"\n--- Loading training data from {config.TRAIN_DATA_PATH} ---")
    df = data_loader.load_data(
        config.TRAIN_DATA_PATH,
        nrows=config.MAX_TRAIN_ROWS if config.MAX_TRAIN_ROWS else None
    )
    
    # Prepare sequence data (tabular/sequence models)
    print("\n--- Preparing sequence data (feature engineering, scaling, split) ---")
    train_loader, val_loader, num_features, feature_names, scaler = data_pipeline.prepare_data(
        df,
        main_segment_size=config.MAIN_SEGMENT_SIZE,
        sub_segment_size=config.SUB_SEGMENT_SIZE,
        test_size=config.TEST_SIZE
    )
    
    # Save scaler for test evaluation and submission
    print(f"\n--- Saving scaler to {config.SCALER_PATH} ---")
    config.ensure_dir(config.SCALER_PATH)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {config.SCALER_PATH}")
    
    # Save train and validation DataLoaders for reuse
    print(f"\n--- Saving train DataLoader to {config.TRAIN_LOADER_PATH} ---")
    torch.save(train_loader, config.TRAIN_LOADER_PATH)
    print(f"✓ Train DataLoader saved to {config.TRAIN_LOADER_PATH}")
    
    print(f"\n--- Saving validation DataLoader to {config.VAL_LOADER_PATH} ---")
    torch.save(val_loader, config.VAL_LOADER_PATH)
    print(f"✓ Validation DataLoader saved to {config.VAL_LOADER_PATH}")
    
    # Prepare spectrogram data (image models)
    print("\n--- Preparing spectrogram data for image models ---")
    try:
        img_train_loader, img_val_loader = data_pipeline.prepare_spectrogram_data(
            df,
            segment_size=config.MAIN_SEGMENT_SIZE,
            test_size=config.TEST_SIZE
        )
        print("✓ Spectrogram data prepared successfully")
        
        # Save image data loaders
        torch.save(img_train_loader, 'img_train_loader.pth')
        torch.save(img_val_loader, 'img_val_loader.pth')
        print("✓ Image data loaders saved")
        
        has_image_data = True
    except Exception as e:
        print(f"⚠ Warning: Could not prepare image data: {e}")
        print("  This may be due to missing librosa or other dependencies.")
        print("  Image models will be skipped.")
        has_image_data = False
        img_train_loader, img_val_loader = None, None
    
    print("\n" + "=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Number of features: {num_features}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    if has_image_data:
        print(f"  Image training samples: {len(img_train_loader.dataset)}")
        print(f"  Image validation samples: {len(img_val_loader.dataset)}")
    print(f"\nSaved files:")
    print(f"  - {config.SCALER_PATH}")
    print(f"  - {config.TRAIN_LOADER_PATH}")
    print(f"  - {config.VAL_LOADER_PATH}")
    if has_image_data:
        print(f"  - img_train_loader.pth")
        print(f"  - img_val_loader.pth")
    
    return train_loader, val_loader, num_features, feature_names, scaler, img_train_loader, img_val_loader


def load_prepared_dataset():
    """
    Load previously prepared dataset components.
    
    This function loads the saved train_loader, val_loader, and scaler.
    Note: This requires that prepare_and_save_dataset() has been run first.
    
    Returns:
        train_loader, val_loader, num_features, feature_names, scaler
    """
    import os
    
    if not os.path.exists(config.SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found at {config.SCALER_PATH}. "
            "Please run prepare_dataset.py first to prepare the data."
        )
    
    if not os.path.exists(config.TRAIN_LOADER_PATH):
        raise FileNotFoundError(
            f"Train loader not found at {config.TRAIN_LOADER_PATH}. "
            "Please run prepare_dataset.py first to prepare the data."
        )
    
    # Load scaler
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load data loaders (weights_only=False needed for DataLoader objects)
    train_loader = torch.load(config.TRAIN_LOADER_PATH, weights_only=False)
    val_loader = torch.load(config.VAL_LOADER_PATH, weights_only=False) if os.path.exists(config.VAL_LOADER_PATH) else None
    
    # Get num_features from the first batch
    sample_batch = next(iter(train_loader))
    num_features = sample_batch[0].shape[-1]
    
    # Feature names - we'll need to load a small sample to get these
    # For now, return None and let the calling code handle it
    feature_names = None
    
    print(f"✓ Loaded scaler from {config.SCALER_PATH}")
    print(f"✓ Loaded train loader from {config.TRAIN_LOADER_PATH}")
    if val_loader:
        print(f"✓ Loaded validation loader from {config.VAL_LOADER_PATH}")
    print(f"✓ Number of features: {num_features}")
    
    return train_loader, val_loader, num_features, feature_names, scaler


if __name__ == "__main__":
    # Run the preparation pipeline
    prepare_and_save_dataset()

