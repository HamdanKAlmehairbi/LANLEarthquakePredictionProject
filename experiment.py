"""
Long-Term Training Experiment - Full Training Pipeline
Trains all models for 200 epochs, tracking best validation MAE and generating loss curves.

This script replaces both main.py and double_descent_experiment.py with a unified,
streamlined workflow that:
1. Loads pre-prepared data from prepare_dataset.py
2. Trains all models (sequence-based and image-based) for 200 epochs
3. Tracks best epochs and validation MAE for each model
4. Generates best_epochs_summary.csv and training_histories.json

Usage:
    python experiment.py

Prerequisites:
    python prepare_dataset.py  # Must be run first to prepare data
"""

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

import config
import utils
import models
import trainer
from image_models import MLPERRegressionModel
from prepare_dataset import load_prepared_dataset


def train_lstm_static(model, model_name, train_loader, val_loader, epochs=200, learning_rate=0.001):
    """
    Specialized training function for LSTM with a static learning rate,
    replicating the original successful training environment.
    """
    device = utils.get_device()
    model.to(device)
    criterion = torch.nn.L1Loss()
    # Use a plain Adam optimizer with a fixed learning rate and NO weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}
    best_val_mae = float('inf')
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} with STATIC learning rate: {learning_rate}")
    print(f"{'='*70}")
    
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
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)
        
        # Track the best epoch
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            best_epoch = epoch + 1
        print(f'Epoch {epoch+1:3d}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Best Val MAE: {best_val_mae:.4f} (Epoch {best_epoch})')
    print(f"\n✓ {model_name} training complete!")
    print(f"  Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    
    return history, best_epoch, best_val_mae


def train_model_full(model, model_name, train_loader, val_loader, optimizer, scheduler, epochs=200, 
                     is_image_model=False):
    """
    Train model for full epochs with per-epoch printing and track best validation MAE.
    
    Args:
        model: PyTorch model to train
        model_name: Name of the model (for logging)
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: PyTorch optimizer instance
        scheduler: PyTorch scheduler instance (can be None)
        epochs: Number of epochs to train
        is_image_model: Whether this is an image model (affects batch unpacking)
    
    Returns:
        history: Dictionary with train_loss, val_loss, epoch lists
        best_epoch: Epoch number with lowest validation MAE
        best_val_mae: Lowest validation MAE value
    """
    device = utils.get_device()
    model.to(device)
    criterion = torch.nn.L1Loss()
    
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}
    best_val_mae = float('inf')
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Training {model_name} for {epochs} epochs")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            if is_image_model:
                mag, phase, labels = batch
                mag, phase, labels = mag.to(device), phase.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(mag, phase)
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Step scheduler after each batch if it's a per-batch scheduler (e.g., OneCycleLR)
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if is_image_model:
                    mag, phase, labels = batch
                    mag, phase, labels = mag.to(device), phase.to(device), labels.to(device)
                    outputs = model(mag, phase)
                else:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(val_loader)
        
        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track best epoch
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            best_epoch = epoch + 1
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)
        
        # Print every epoch
        print(f'Epoch {epoch+1:3d}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Best Val MAE: {best_val_mae:.4f} (Epoch {best_epoch}) | LR: {current_lr:.6f}')
    
    print(f"\n✓ {model_name} training complete!")
    print(f"  Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    
    return history, best_epoch, best_val_mae


def main():
    """
    Main experiment workflow:
    1. Load pre-prepared data from prepare_dataset.py
    2. Train all models with all optimizer configurations
    3. Track best epochs and save results
    """
    print("=" * 70)
    print("Optimizer Comparison Experiment - All Models with Multiple Optimizers")
    print("=" * 70)
    
    # Setup
    utils.set_seed(config.RANDOM_SEED)
    
    # Import optimizer configurations
    from config import OPTIMIZER_CONFIGS
    
    # Load pre-prepared data
    print("\n--- Loading pre-prepared dataset ---")
    try:
        train_loader, val_loader, num_features, _, scaler = load_prepared_dataset()
        print(f"✓ Loaded pre-prepared sequence data")
        print(f"  Number of features: {num_features}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run 'python prepare_dataset.py' first to prepare the data.")
        return
    
    # Load image data loaders if available
    print("\n--- Loading pre-prepared image data ---")
    has_image_data = False
    img_train_loader, img_val_loader = None, None
    
    if os.path.exists('img_train_loader.pth') and os.path.exists('img_val_loader.pth'):
        try:
            img_train_loader = torch.load('img_train_loader.pth', weights_only=False)
            img_val_loader = torch.load('img_val_loader.pth', weights_only=False)
            has_image_data = True
            print(f"✓ Loaded pre-prepared image data")
            print(f"  Image training samples: {len(img_train_loader.dataset)}")
            print(f"  Image validation samples: {len(img_val_loader.dataset)}")
        except Exception as e:
            print(f"⚠ Warning: Could not load image data: {e}")
            print("  Image models will be skipped.")
    else:
        print("⚠ Image data loaders not found. Image models will be skipped.")
        print("  (This is OK if you haven't prepared image data)")
    
    # Define all models to train
    sequence_models = {
        "1D CNN": models.CNNModel,
        "LSTM": models.LSTMModel,
        "Hybrid CNN-LSTM": models.HybridModel,
        "Hybrid w/ Attention": models.HybridAttention,
    }
    
    # Store results - nested structure: {model_name: {optimizer_name: history}}
    all_histories = {model_name: {} for model_name in sequence_models.keys()}
    if has_image_data:
        all_histories["MLPER-Inspired (Image)"] = {}
    
    # Train all sequence models with all optimizer configurations
    print("\n" + "=" * 70)
    print("Training Sequence-Based Models with Multiple Optimizers")
    print("=" * 70)
    
    for model_name, ModelClass in sequence_models.items():
        print(f"\n{'#'*70}")
        print(f"# Model: {model_name}")
        print(f"{'#'*70}")
        
        # Create a sample model to get parameter count (for display only)
        sample_model = ModelClass(input_features=num_features)
        print(f"Parameters: {utils.count_parameters(sample_model):,}")
        
        # Loop through optimizer configurations
        for opt_config in OPTIMIZER_CONFIGS:
            print(f"\n{'-'*70}")
            print(f"Optimizer Config: {opt_config['name']}")
            print(f"{'-'*70}")
            
            # Create a fresh model instance for each optimizer config
            model = ModelClass(input_features=num_features)
            
            # Optimizer factory
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
            
            # Scheduler factory
            scheduler = None
            if opt_config['scheduler'] == 'OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt_config.get('max_lr', opt_config['lr'] * 3),
                    steps_per_epoch=len(train_loader),
                    epochs=config.LONG_TRAINING_EPOCHS,
                    pct_start=0.3,
                    anneal_strategy='cos'
                )
            elif opt_config['scheduler'] == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=opt_config['lr'],
                    max_lr=opt_config.get('max_lr', opt_config['lr'] * 10),
                    step_size_up=len(train_loader) * 5,  # 5 epochs up, 5 epochs down
                    mode='triangular2'
                )
            elif opt_config['scheduler'] is None:
                scheduler = None
            else:
                raise ValueError(f"Unknown scheduler: {opt_config['scheduler']}")
            
            # Train the model
            history, best_epoch, best_val_mae = train_model_full(
                model=model,
                model_name=f"{model_name} - {opt_config['name']}",
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=config.LONG_TRAINING_EPOCHS,
                is_image_model=False
            )
            
            # Store results
            all_histories[model_name][opt_config['name']] = history
    
    # Train image model if available
    if has_image_data:
        print("\n" + "=" * 70)
        print("Training Image-Based Model with Multiple Optimizers")
        print("=" * 70)
        
        print(f"\n{'#'*70}")
        print(f"# Model: MLPER-Inspired (Image)")
        print(f"{'#'*70}")
        
        img_model = MLPERRegressionModel()
        print(f"Parameters: {utils.count_parameters(img_model):,}")
        
        # Loop through optimizer configurations
        for opt_config in OPTIMIZER_CONFIGS:
            print(f"\n{'-'*70}")
            print(f"Optimizer Config: {opt_config['name']}")
            print(f"{'-'*70}")
            
            # Create a fresh model instance for each optimizer config
            img_model = MLPERRegressionModel()
            
            # Optimizer factory
            if opt_config['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(
                    img_model.parameters(), 
                    lr=opt_config['lr'], 
                    weight_decay=opt_config.get('weight_decay', 0.0)
                )
            elif opt_config['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW(
                    img_model.parameters(), 
                    lr=opt_config['lr'], 
                    weight_decay=opt_config.get('weight_decay', 0.0)
                )
            elif opt_config['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(
                    img_model.parameters(), 
                    lr=opt_config['lr'], 
                    momentum=opt_config.get('momentum', 0.9),
                    weight_decay=opt_config.get('weight_decay', 0.0)
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
            
            # Scheduler factory
            scheduler = None
            if opt_config['scheduler'] == 'OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=opt_config.get('max_lr', opt_config['lr'] * 3),
                    steps_per_epoch=len(img_train_loader),
                    epochs=config.IMAGE_MODEL_EPOCHS,
                    pct_start=0.3,
                    anneal_strategy='cos'
                )
            elif opt_config['scheduler'] == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=opt_config['lr'],
                    max_lr=opt_config.get('max_lr', opt_config['lr'] * 10),
                    step_size_up=len(img_train_loader) * 5,  # 5 epochs up, 5 epochs down
                    mode='triangular2'
                )
            elif opt_config['scheduler'] is None:
                scheduler = None
            else:
                raise ValueError(f"Unknown scheduler: {opt_config['scheduler']}")
            
            # Train the model
            history, best_epoch, best_val_mae = train_model_full(
                model=img_model,
                model_name=f"MLPER-Inspired (Image) - {opt_config['name']}",
                train_loader=img_train_loader,
                val_loader=img_val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=config.IMAGE_MODEL_EPOCHS,
                is_image_model=True
            )
            
            # Store results
            all_histories["MLPER-Inspired (Image)"][opt_config['name']] = history
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    # Save full histories as JSON (nested structure)
    histories_dict = {}
    for model_name, optimizer_results in all_histories.items():
        histories_dict[model_name] = {}
        for optimizer_name, history in optimizer_results.items():
            # Find best epoch and best val MAE for this optimizer
            best_val_mae = min(history['val_loss'])
            best_epoch = history['epoch'][history['val_loss'].index(best_val_mae)]
            
            histories_dict[model_name][optimizer_name] = {
                'epoch': history['epoch'],
                'train_loss': [float(x) for x in history['train_loss']],
                'val_loss': [float(x) for x in history['val_loss']],
                'best_epoch': int(best_epoch),
                'best_val_mae': float(best_val_mae)
            }
    
    config.ensure_dir(config.TRAINING_HISTORIES)
    with open(config.TRAINING_HISTORIES, "w") as f:
        json.dump(histories_dict, f, indent=2)
    print(f"\n✓ Training histories saved to {config.TRAINING_HISTORIES}")
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review {config.TRAINING_HISTORIES} to see all optimizer results")
    print(f"  2. Run analysis_notebook.ipynb to visualize optimizer comparisons")


if __name__ == "__main__":
    main()

