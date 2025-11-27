"""
Final Kaggle Submission Script with K-Fold Cross-Validation

This script implements a robust 5-Fold Cross-Validation strategy to reduce overfitting
and improve generalization to the test set. Instead of training on the full dataset
with a fixed epoch count, it:

1. Splits the training data into 5 folds
2. Trains 5 separate models, each on 4 folds (using 1 fold for validation concept)
3. Generates predictions from all 5 models
4. Averages the predictions to create a final, more robust submission

This approach significantly reduces overfitting compared to training on the full dataset
with epochs derived from a smaller training set.

This script automatically:
- Reads training_histories.json to identify the best model-optimizer combination
- Performs 5-Fold Cross-Validation training
- Averages predictions across all folds
- Generates final submission.csv

Usage:
    python predict_submission.py

Prerequisites:
    python prepare_dataset.py  # Must be run first (optional, for pre-prepared loaders)
    python experiment.py       # Must be run to generate training_histories.json
"""

import pandas as pd
import torch
import pickle
import os
import shutil
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import config
import utils
import data_loader
import data_pipeline
import models
import trainer
from image_models import MLPERRegressionModel


# Model name mapping to model classes
MODEL_MAP = {
    "1D CNN": models.CNNModel,
    "LSTM": models.LSTMModel,
    "Hybrid CNN-LSTM": models.HybridModel,
    "Hybrid w/ Attention": models.HybridAttention,
    "MLPER-Inspired (Image)": MLPERRegressionModel,
}


def load_best_models_info():
    """
    Load best model information by parsing training_histories.json.
    
    Returns:
        best_model_info: DataFrame row with best model
    """
    if not os.path.exists(config.TRAINING_HISTORIES):
        raise FileNotFoundError(
            f"{config.TRAINING_HISTORIES} not found. "
            "Please run 'python experiment.py' first."
        )
    
    import json
    
    with open(config.TRAINING_HISTORIES, 'r') as f:
        histories = json.load(f)
    
    results_list = []
    for model_name, optimizers in histories.items():
        for optimizer_name, data in optimizers.items():
            results_list.append({
                'Model': model_name,
                'Optimizer': optimizer_name,
                'Best_Val_MAE': data.get('best_val_mae'),
                'Best_Epoch': data.get('best_epoch')
            })
    
    summary_df = pd.DataFrame(results_list).sort_values('Best_Val_MAE').reset_index(drop=True)
    
    best_model_info = summary_df.iloc[0]
    
    print("=" * 70)
    print("Model Selection from Experiment Results")
    print("=" * 70)
    print("\nBest performers (sorted by validation MAE):")
    print(summary_df.head(10).to_string(index=False))
    print(f"\n✓ Selected best performer: {best_model_info['Model']} with {best_model_info['Optimizer']}")
    
    return best_model_info


def train_on_full_dataset(model_class, model_name, optimizer_config, num_features, epochs, is_image_model=False):
    """
    Train a model on the entire training dataset (no validation split).
    
    Args:
        model_class: Model class to instantiate
        model_name: Name of the model (for logging)
        optimizer_config: Dictionary containing optimizer configuration (from config.OPTIMIZER_CONFIGS)
        num_features: Number of input features (ignored for image models)
        epochs: Number of epochs to train
        is_image_model: Whether this is an image-based model
    
    Returns:
        Trained model, scaler, feature_names
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name} on FULL training dataset")
    print(f"{'='*70}")
    
    # Load full training data
    print(f"\n--- Loading full training data from {config.TRAIN_DATA_PATH} ---")
    df_full = data_loader.load_data(
        config.TRAIN_DATA_PATH,
        nrows=config.MAX_TRAIN_ROWS if config.MAX_TRAIN_ROWS else None
    )
    
    if is_image_model:
        # Prepare spectrogram data without validation split
        print("\n--- Preparing spectrogram data (full dataset) ---")
        img_train_loader, _ = data_pipeline.prepare_spectrogram_data(
            df_full, segment_size=config.MAIN_SEGMENT_SIZE, test_size=0.0
        )
        
        # Create a dummy validation loader for training loop compatibility
        from torch.utils.data import DataLoader, TensorDataset
        dummy_val = TensorDataset(
            torch.zeros(10, 1, 128, 128),
            torch.zeros(10, 1, 128, 128),
            torch.zeros(10, 1)
        )
        val_loader = DataLoader(dummy_val, batch_size=8, shuffle=False)
        
        model = model_class()
        print(f"Parameters: {utils.count_parameters(model):,}")
        
        history = trainer.train_image_model(
            model, img_train_loader, val_loader, epochs=epochs
        )
        
        # For image models, we don't need a scaler
        scaler = None
        feature_names = None
        
    else:
        # Prepare sequence data without validation split
        print("\n--- Preparing sequence data (full dataset, no validation split) ---")
        train_loader, _, num_features, feature_names, scaler = data_pipeline.prepare_data(
            df_full,
            main_segment_size=config.MAIN_SEGMENT_SIZE,
            sub_segment_size=config.SUB_SEGMENT_SIZE,
            test_size=0.0  # No validation split - use all data for training
        )
        
        model = model_class(input_features=num_features)
        print(f"Parameters: {utils.count_parameters(model):,}")
        
        # Dynamic optimizer and scheduler factory based on optimizer_config
        device = utils.get_device()
        model.to(device)
        criterion = torch.nn.L1Loss()
        
        # Optimizer factory
        if optimizer_config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif optimizer_config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['optimizer']}")
        
        # Scheduler factory
        scheduler = None
        if optimizer_config['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer_config.get('max_lr', optimizer_config['lr'] * 3),
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif optimizer_config['scheduler'] == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=optimizer_config['lr'],
                max_lr=optimizer_config.get('max_lr', optimizer_config['lr'] * 10),
                step_size_up=len(train_loader) * 5,  # 5 epochs up, 5 epochs down
                mode='triangular2'
            )
        elif optimizer_config['scheduler'] is None:
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {optimizer_config['scheduler']}")
        
        print(f"Using optimizer: {optimizer_config['optimizer']} with scheduler: {optimizer_config['scheduler']}")
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if scheduler:  # Only step if scheduler exists
                    scheduler.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Train MAE: {train_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save scaler for test prediction
        with open(config.SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved scaler to {config.SCALER_PATH}")
    
    print(f"\n✓ {model_name} training complete!")
    return model, scaler, feature_names


def predict_test_set(model, model_name, scaler, feature_names, is_image_model=False):
    """
    Generate predictions on the test set.
    
    Args:
        model: Trained model
        model_name: Name of the model
        scaler: Fitted scaler from training (None for image models)
        feature_names: Feature names from training (None for image models)
        is_image_model: Whether this is an image-based model
    
    Returns:
        Dictionary mapping segment_id to predicted time_to_failure
    """
    print(f"\n{'='*70}")
    print(f"Generating predictions for {model_name}")
    print(f"{'='*70}")
    
    if not os.path.exists(config.TEST_FOLDER):
        raise FileNotFoundError(
            f"Test folder not found at {config.TEST_FOLDER}. "
            "Please ensure the test data is available."
        )
    
    if is_image_model:
        test_loader, segment_ids = data_pipeline.prepare_test_spectrogram_data(
            test_folder=config.TEST_FOLDER,
            segment_size=config.MAIN_SEGMENT_SIZE
        )
    else:
        test_loader, segment_ids = data_pipeline.prepare_test_data(
            test_folder=config.TEST_FOLDER,
            main_segment_size=config.MAIN_SEGMENT_SIZE,
            sub_segment_size=config.SUB_SEGMENT_SIZE,
            scaler=scaler,
            feature_names=feature_names
        )
    
    # Generate predictions
    device = utils.get_device()
    model.to(device)
    model.eval()
    
    predictions = {}
    
    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            if is_image_model:
                mag, phase = batch
                mag, phase = mag.to(device), phase.to(device)
                outputs = model(mag, phase)
            else:
                inputs = batch[0]
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            batch_preds = outputs.cpu().numpy().flatten()
            
            for pred in batch_preds:
                if batch_idx < len(segment_ids):
                    segment_id = segment_ids[batch_idx]
                    predictions[segment_id] = float(pred)
                    batch_idx += 1
    
    print(f"✓ Generated {len(predictions)} predictions")
    return predictions


def is_image_model(model_name):
    """Check if a model name corresponds to an image-based model."""
    return "Image" in model_name or "MLPER" in model_name


def main():
    """
    Main submission workflow using K-Fold Cross-Validation.
    Trains 5 models on different folds and averages their predictions.
    """
    print("=" * 70)
    print("Kaggle Submission Pipeline with 5-Fold Cross-Validation")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Read training_histories.json to identify best model")
    print("  2. Split training data into 5 folds")
    print("  3. Train 5 models (one per fold)")
    print("  4. Average predictions across all folds")
    print("  5. Save submission.csv")
    print("=" * 70)
    
    # Set random seed
    utils.set_seed(config.RANDOM_SEED)
    
    # --- 1. Load Data and Identify Best Performer ---
    best_model_info = load_best_models_info()
    best_model_name = best_model_info['Model']
    best_optimizer_name = best_model_info['Optimizer']
    best_epochs = int(best_model_info['Best_Epoch'])
    
    ModelClass = MODEL_MAP.get(best_model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown model name: {best_model_name}")
    
    is_image = is_image_model(best_model_name)
    if is_image:
        raise NotImplementedError("K-Fold CV for image models is not yet implemented. Please use sequence models.")
    
    # Find optimizer configuration
    from config import OPTIMIZER_CONFIGS
    opt_config = next((item for item in OPTIMIZER_CONFIGS if item['name'] == best_optimizer_name), None)
    if opt_config is None:
        raise ValueError(f"Optimizer config '{best_optimizer_name}' not found!")
    
    print(f"\n--- Using Best Model: {best_model_name} with {best_optimizer_name} for {best_epochs} epochs ---")
    
    # --- 2. Load Full Dataset for CV Splitting ---
    print("\n--- Loading full training dataset ---")
    df_full = data_loader.load_data(
        config.TRAIN_DATA_PATH,
        nrows=config.MAX_TRAIN_ROWS if config.MAX_TRAIN_ROWS else None
    )
    print(f"Loaded {len(df_full):,} rows")
    
    # --- 3. Convert DataFrame to Features (before splitting to prevent data leakage) ---
    print("\n--- Converting DataFrame to feature sequences ---")
    X_full, y_full, scaler_template, feature_names = data_pipeline.convert_df_to_features(
        df_full,
        main_segment_size=config.MAIN_SEGMENT_SIZE,
        sub_segment_size=config.SUB_SEGMENT_SIZE
    )
    num_features = X_full.shape[2]
    print(f"Feature sequences shape: {X_full.shape}")
    print(f"Number of features: {num_features}")
    
    # --- 4. Prepare Test Data Once (we'll use fold-specific scalers later) ---
    print("\n--- Preparing test data structure ---")
    if not os.path.exists(config.TEST_FOLDER):
        raise FileNotFoundError(f"Test folder not found at {config.TEST_FOLDER}")
    
    import glob
    test_files = sorted(glob.glob(os.path.join(config.TEST_FOLDER, '*.csv')))
    segment_ids = [os.path.basename(f).replace('.csv', '') for f in test_files]
    print(f"Found {len(segment_ids)} test segments")
    
    # --- 5. K-Fold Cross-Validation Loop ---
    kf = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    all_fold_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full)):
        print(f"\n{'='*70}")
        print(f"Fold {fold + 1}/5")
        print(f"{'='*70}")
        print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Split data for this fold
        X_train_fold = X_full[train_idx]
        y_train_fold = y_full[train_idx]
        X_val_fold = X_full[val_idx]  # Not used for training, but kept for reference
        
        # Fit scaler ONLY on this fold's training data
        print("Fitting scaler on fold training data...")
        num_samples, seq_len, num_feat = X_train_fold.shape
        X_train_flat = X_train_fold.reshape(-1, num_feat)
        scaler = StandardScaler()
        X_train_scaled_flat = scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled_flat.reshape(num_samples, seq_len, num_feat)
        
        # Create DataLoader for this fold
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        
        # Initialize model and optimizer for this fold
        model = ModelClass(input_features=num_features)
        device = utils.get_device()
        model.to(device)
        
        # Create optimizer
        if opt_config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif opt_config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
        
        # Create scheduler if needed
        scheduler = None
        if opt_config['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_config.get('max_lr', opt_config['lr'] * 3),
                steps_per_epoch=len(train_loader),
                epochs=best_epochs,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif opt_config['scheduler'] == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=opt_config['lr'],
                max_lr=opt_config.get('max_lr', opt_config['lr'] * 10),
                step_size_up=len(train_loader) * 5,
                mode='triangular2'
            )
        
        # Train the model for this fold
        print(f"\nTraining model for fold {fold + 1}...")
        trainer.train_for_submission(model, train_loader, optimizer, scheduler, epochs=best_epochs)
        
        # Generate predictions on test set using this fold's scaler
        print(f"\nGenerating test predictions for fold {fold + 1}...")
        fold_predictions = predict_test_set(
            model=model,
            model_name=f"Fold {fold+1}",
            scaler=scaler,
            feature_names=feature_names,
            is_image_model=False
        )
        all_fold_predictions.append(fold_predictions)
        print(f"✓ Fold {fold + 1} complete")
    
    # --- 6. Ensemble Fold Predictions (Average) ---
    print("\n" + "=" * 70)
    print("Averaging predictions across all folds")
    print("=" * 70)
    
    final_predictions = {}
    for seg_id in segment_ids:
        fold_preds = [preds.get(seg_id, 0.0) for preds in all_fold_predictions]
        final_predictions[seg_id] = np.mean(fold_preds)
    
    print(f"✓ Averaged predictions from {len(all_fold_predictions)} folds")
    
    # --- 7. Create Submission File ---
    print(f"\n--- Creating submission file: {config.SUBMISSION_PATH} ---")
    submission_df = pd.DataFrame([
        {'seg_id': seg_id, 'time_to_failure': pred}
        for seg_id, pred in sorted(final_predictions.items())
    ])
    
    # Remove submission.csv if it exists as a directory (Windows issue)
    if os.path.exists(config.SUBMISSION_PATH) and os.path.isdir(config.SUBMISSION_PATH):
        print(f"⚠ Warning: {config.SUBMISSION_PATH} exists as a directory. Removing it...")
        shutil.rmtree(config.SUBMISSION_PATH)
    
    # Ensure parent directory exists
    submission_dir = os.path.dirname(config.SUBMISSION_PATH)
    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)
    
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"✓ Submission file saved to {config.SUBMISSION_PATH}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Submission Summary")
    print("=" * 70)
    print(f"Model: {best_model_name} with {best_optimizer_name}")
    print(f"Training strategy: 5-Fold Cross-Validation")
    print(f"Epochs per fold: {best_epochs}")
    print(f"Number of test predictions: {len(final_predictions)}")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {submission_df['time_to_failure'].mean():.4f}")
    print(f"  Std:  {submission_df['time_to_failure'].std():.4f}")
    print(f"  Min:  {submission_df['time_to_failure'].min():.4f}")
    print(f"  Max:  {submission_df['time_to_failure'].max():.4f}")
    print(f"\nSubmission file: {config.SUBMISSION_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
