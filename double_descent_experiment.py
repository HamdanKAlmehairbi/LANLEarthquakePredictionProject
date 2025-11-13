"""
Double Descent Experiment - Full Training
Trains all models for 200 epochs, tracking best validation MAE and generating loss curves.
"""
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

import utils
import data_loader
import data_pipeline
import models
import trainer
from image_models import MLPERRegressionModel


def train_model_full(model, model_name, train_loader, val_loader, epochs=200, 
                     learning_rate=0.001, is_image_model=False):
    """
    Train model for full epochs with per-epoch printing and track best validation MAE.
    
    Returns:
        history: Dictionary with train_loss, val_loss, epoch lists
        best_epoch: Epoch number with lowest validation MAE
        best_val_mae: Lowest validation MAE value
    """
    device = utils.get_device()
    model.to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
        
        # Track best epoch
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            best_epoch = epoch + 1
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)
        
        # Print every epoch
        print(f'Epoch {epoch+1:3d}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | Best Val MAE: {best_val_mae:.4f} (Epoch {best_epoch})')
    
    print(f"\n✓ {model_name} training complete!")
    print(f"  Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    
    return history, best_epoch, best_val_mae


def main():
    """
    Train all models for 200 epochs, track best epochs, and generate plots.
    """
    print("=" * 70)
    print("Full Training Experiment - All Models for 200 Epochs")
    print("=" * 70)
    
    # Setup
    utils.set_seed(42)
    DATA_PATH = 'data/train.csv'
    
    # Load and prepare data
    print("\n--- Loading and preparing data ---")
    df = data_loader.load_data(DATA_PATH)
    
    # Prepare sequence data
    train_loader, val_loader, num_features, _, scaler = data_pipeline.prepare_data(df)
    print(f"Number of features: {num_features}")
    
    # Save scaler for test evaluation
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved to scaler.pkl for test evaluation")
    
    # Prepare image data
    print("\n--- Preparing spectrogram data for image model ---")
    try:
        img_train_loader, img_val_loader = data_pipeline.prepare_spectrogram_data(df)
        has_image_data = True
        print("✓ Image data prepared successfully")
    except Exception as e:
        print(f"Warning: Could not prepare image data: {e}")
        print("Skipping image model experiment.")
        has_image_data = False
    
    # Define all models to train
    sequence_models = {
        "1D CNN": models.CNNModel,
        "LSTM": models.LSTMModel,
        "Hybrid CNN-LSTM": models.HybridModel,
        "Hybrid w/ Attention": models.HybridAttention,
    }
    
    # Store results
    all_histories = {}
    best_epochs = {}
    best_val_maes = {}
    
    # Train all sequence models
    for model_name, ModelClass in sequence_models.items():
        model = ModelClass(input_features=num_features)
        print(f"\nModel: {model_name}")
        print(f"Parameters: {utils.count_parameters(model):,}")
        
        history, best_epoch, best_val_mae = train_model_full(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=200,
            is_image_model=False
        )
        
        all_histories[model_name] = history
        best_epochs[model_name] = best_epoch
        best_val_maes[model_name] = best_val_mae
    
    # Train image model if available
    if has_image_data:
        img_model = MLPERRegressionModel()
        print(f"\nModel: MLPER-Inspired (Image)")
        print(f"Parameters: {utils.count_parameters(img_model):,}")
        
        history, best_epoch, best_val_mae = train_model_full(
            model=img_model,
            model_name="MLPER-Inspired (Image)",
            train_loader=img_train_loader,
            val_loader=img_val_loader,
            epochs=200,
            is_image_model=True
        )
        
        all_histories["MLPER-Inspired (Image)"] = history
        best_epochs["MLPER-Inspired (Image)"] = best_epoch
        best_val_maes["MLPER-Inspired (Image)"] = best_val_mae
    
    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    # Save best epochs summary
    best_epochs_df = pd.DataFrame({
        'Model': list(best_epochs.keys()),
        'Best_Epoch': list(best_epochs.values()),
        'Best_Val_MAE': list(best_val_maes.values())
    })
    best_epochs_df = best_epochs_df.sort_values('Best_Val_MAE')
    best_epochs_df.to_csv("best_epochs_summary.csv", index=False)
    print("\nBest Epochs Summary:")
    print(best_epochs_df.to_string(index=False))
    
    # Save full histories as JSON
    histories_dict = {}
    for model_name, history in all_histories.items():
        histories_dict[model_name] = {
            'epoch': history['epoch'],
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'best_epoch': int(best_epochs[model_name]),
            'best_val_mae': float(best_val_maes[model_name])
        }
    
    with open("training_histories.json", "w") as f:
        json.dump(histories_dict, f, indent=2)
    print("\n✓ Training histories saved to training_histories.json")
    
    # Generate plots
    print("\n--- Generating plots ---")
    num_models = len(all_histories)
    cols = 2
    rows = (num_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, history) in enumerate(all_histories.items()):
        ax = axes[idx]
        epochs = history['epoch']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        best_epoch = best_epochs[model_name]
        best_val_mae = best_val_maes[model_name]
        
        ax.plot(epochs, train_loss, label='Train MAE', linewidth=2, alpha=0.8)
        ax.plot(epochs, val_loss, label='Val MAE', linewidth=2, alpha=0.8)
        ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=2, 
                  label=f'Best Epoch: {best_epoch}')
        ax.scatter([best_epoch], [best_val_mae], color='red', s=100, zorder=5,
                  label=f'Best Val MAE: {best_val_mae:.4f}')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title(f'{model_name}\nBest: Epoch {best_epoch}, Val MAE: {best_val_mae:.4f}', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_curves_all_models.png', dpi=300, bbox_inches='tight')
    print("✓ Plots saved to training_curves_all_models.png")
    plt.close()
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for model_name, history in all_histories.items():
        ax.plot(history['epoch'], history['val_loss'], label=model_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MAE', fontsize=12)
    ax.set_title('Validation Loss Comparison - All Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('validation_loss_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved to validation_loss_comparison.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print("\nSummary:")
    print(best_epochs_df.to_string(index=False))


if __name__ == "__main__":
    main()
