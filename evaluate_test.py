"""
Evaluate trained models on test data from /test folder.
This script loads trained models and generates predictions on the actual test set.
"""
import pandas as pd
import torch
import numpy as np
import pickle
import os

import utils
import data_loader
import data_pipeline
import models
from image_models import MLPERRegressionModel


def evaluate_on_test(model, model_name, test_loader, segment_ids, is_image_model=False):
    """
    Evaluate a model on test data and return predictions.
    
    Returns:
        predictions: Dictionary mapping segment_id to predicted time_to_failure
    """
    device = utils.get_device()
    model.to(device)
    model.eval()
    
    predictions = {}
    batch_idx = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if is_image_model:
                mag, phase = batch
                mag, phase = mag.to(device), phase.to(device)
                outputs = model(mag, phase)
            else:
                inputs = batch[0]  # TensorDataset returns tuple
                inputs = inputs.to(device)
                outputs = model(inputs)
            
            # Get predictions for this batch
            batch_preds = outputs.cpu().numpy().flatten()
            
            # Map predictions to segment IDs
            for pred in batch_preds:
                if batch_idx < len(segment_ids):
                    segment_id = segment_ids[batch_idx]
                    predictions[segment_id] = float(pred)
                    batch_idx += 1
    
    return predictions


def main():
    """
    Load all trained models and evaluate on test data.
    """
    print("=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)
    
    # Load scaler (should be saved from training)
    print("\n--- Loading scaler ---")
    if os.path.exists('scaler.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Loaded scaler from scaler.pkl")
        
        # Load training data just to get feature names and num_features
        df_train = data_loader.load_data('data/train.csv', nrows=150_000)  # Just need one segment
        _, _, num_features, feature_names, _ = data_pipeline.prepare_data(
            df_train, test_size=0.0  # No split needed, just for feature extraction
        )
    else:
        print("Warning: scaler.pkl not found. Creating new scaler from training data...")
        df_train = data_loader.load_data('data/train.csv')
        train_loader, val_loader, num_features, feature_names, scaler = data_pipeline.prepare_data(
            df_train, test_size=0.2
        )
        # Save scaler for future use
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✓ Scaler saved to scaler.pkl")
    
    # Prepare test data
    print("\n--- Preparing test data ---")
    test_loader, segment_ids = data_pipeline.prepare_test_data(
        test_folder='test',
        scaler=scaler,
        feature_names=feature_names
    )
    
    # Prepare test spectrogram data
    try:
        test_img_loader, test_img_segment_ids = data_pipeline.prepare_test_spectrogram_data(
            test_folder='test'
        )
        has_image_test = True
    except Exception as e:
        print(f"Warning: Could not prepare image test data: {e}")
        has_image_test = False
    
    # Define models to evaluate
    sequence_models = {
        "1D CNN": models.CNNModel,
        "LSTM": models.LSTMModel,
        "Hybrid CNN-LSTM": models.HybridModel,
        "Hybrid w/ Attention": models.HybridAttention,
    }
    
    all_predictions = {}
    
    # Evaluate sequence models
    print("\n--- Evaluating sequence models on test data ---")
    for model_name, ModelClass in sequence_models.items():
        print(f"\nEvaluating {model_name}...")
        
        # Load model (you'll need to have trained models saved)
        # For now, we'll check if model files exist
        model_file = f"{model_name.lower().replace(' ', '_').replace('/', '_')}_best.pth"
        
        try:
            model = ModelClass(input_features=num_features)
            # Try to load saved state dict if it exists
            if os.path.exists(model_file):
                model.load_state_dict(torch.load(model_file, map_location='cpu'))
                print(f"  Loaded weights from {model_file}")
            else:
                print(f"  Warning: {model_file} not found. Using untrained model.")
            
            predictions = evaluate_on_test(
                model=model,
                model_name=model_name,
                test_loader=test_loader,
                segment_ids=segment_ids,
                is_image_model=False
            )
            
            all_predictions[model_name] = predictions
            print(f"  ✓ Generated {len(predictions)} predictions")
            
        except Exception as e:
            print(f"  ✗ Error evaluating {model_name}: {e}")
    
    # Evaluate image model
    if has_image_test:
        print("\n--- Evaluating image model on test data ---")
        try:
            img_model = MLPERRegressionModel()
            if os.path.exists('mlper_regression_model.pth'):
                img_model.load_state_dict(torch.load('mlper_regression_model.pth', map_location='cpu'))
                print("  Loaded weights from mlper_regression_model.pth")
            
            predictions = evaluate_on_test(
                model=img_model,
                model_name="MLPER-Inspired (Image)",
                test_loader=test_img_loader,
                segment_ids=test_img_segment_ids,
                is_image_model=True
            )
            
            all_predictions["MLPER-Inspired (Image)"] = predictions
            print(f"  ✓ Generated {len(predictions)} predictions")
            
        except Exception as e:
            print(f"  ✗ Error evaluating image model: {e}")
    
    # Save predictions
    print("\n--- Saving predictions ---")
    for model_name, predictions in all_predictions.items():
        # Create submission format: segment_id, time_to_failure
        submission_df = pd.DataFrame([
            {'seg_id': seg_id, 'time_to_failure': pred}
            for seg_id, pred in sorted(predictions.items())
        ])
        
        output_file = f"test_predictions_{model_name.lower().replace(' ', '_').replace('/', '_')}.csv"
        submission_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved {model_name} predictions to {output_file}")
    
    # Create combined predictions file
    if all_predictions:
        combined_df = pd.DataFrame({
            'seg_id': sorted(segment_ids)
        })
        
        for model_name, predictions in all_predictions.items():
            col_name = model_name.lower().replace(' ', '_').replace('/', '_')
            combined_df[col_name] = combined_df['seg_id'].map(predictions)
        
        combined_df.to_csv('test_predictions_all_models.csv', index=False)
        print(f"\n✓ Combined predictions saved to test_predictions_all_models.csv")
        print(f"\nSummary:")
        print(f"  Total test segments: {len(segment_ids)}")
        print(f"  Models evaluated: {len(all_predictions)}")
    
    print("\n" + "=" * 70)
    print("Test evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

