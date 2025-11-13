import pandas as pd
import torch

# Import project modules
import utils
import data_loader
import data_pipeline
import models
import trainer
from image_models import MLPERRegressionModel


def main():
    """
    Main script to run the complete training and evaluation pipeline.
    It loads data, prepares it, trains all models, and saves the results.
    """
    # --- 1. Setup ---
    utils.set_seed(42)
    DATA_PATH = 'data/train.csv'  # Adjust this path as needed
    
    # --- 2. Data Loading and Preparation ---
    df = data_loader.load_data(DATA_PATH)
    train_loader, val_loader, num_features, _ = data_pipeline.prepare_data(df)

    # Save validation DataLoader for use in the analysis notebook
    torch.save(val_loader, "val_loader.pth")
    print("Saved validation DataLoader to val_loader.pth")

    # --- 3. Model Training (Tabular/Sequence Models) ---
    model_definitions = {
        "1D CNN": models.CNNModel,
        "LSTM": models.LSTMModel,
        "Hybrid CNN-LSTM": models.HybridModel,
        "Hybrid w/ Attention": models.HybridAttention,
    }
    
    results = []
    
    for name, ModelClass in model_definitions.items():
        print(f"\n--- Training {name} ---")
        model = ModelClass(input_features=num_features)
        print(f"Parameters: {utils.count_parameters(model):,}")
        
        history = trainer.train_model(model, train_loader, val_loader, epochs=30)
        mae, rmse, r2, inf_time = trainer.evaluate_model(model, val_loader)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R-squared': r2,
            'Parameter Count': utils.count_parameters(model),
            'Inference Time (ms/sample)': inf_time,
        })
        
        # Save the best model for later analysis in the notebook
        if name == "Hybrid w/ Attention":
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Hybrid w/ Attention model state to best_model.pth")

    # --- 4. Image-Based Model Training (Epoch Selection) ---
    print("\n--- Training MLPER-Inspired (Image-Based) Model ---")
    print("Testing epochs: 3, 5, 7, 9 to find optimal training duration...")
    try:
        img_train_loader, img_val_loader = data_pipeline.prepare_spectrogram_data(df)
        
        # Test different epoch counts
        epoch_candidates = [3, 5, 7, 9]
        best_epochs = None
        best_val_loss = float('inf')
        best_model_state = None
        epoch_results = []
        
        for epochs in epoch_candidates:
            print(f"\n--- Testing {epochs} epochs ---")
            img_model = MLPERRegressionModel()
            print(f"Parameters: {utils.count_parameters(img_model):,}")
            
            img_history = trainer.train_image_model(img_model, img_train_loader, img_val_loader, epochs=epochs)
            
            # Get final validation loss
            final_val_loss = img_history['val_loss'][-1]
            epoch_results.append({
                'Epochs': epochs,
                'Train MAE': img_history['train_loss'][-1],
                'Val MAE': final_val_loss,
            })
            
            print(f"Final validation MAE: {final_val_loss:.4f}")
            
            # Track best model
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_epochs = epochs
                best_model_state = img_model.state_dict().copy()
        
        # Print epoch comparison
        print("\n--- Epoch Comparison Results ---")
        epoch_df = pd.DataFrame(epoch_results)
        print(epoch_df.to_string(index=False))
        
        # Use the best model (already trained)
        print(f"\n--- Using model trained for {best_epochs} epochs (best validation performance) ---")
        final_model = MLPERRegressionModel()
        final_model.load_state_dict(best_model_state)
        
        # Evaluate final model
        img_mae, img_rmse, img_r2, img_inf_time = trainer.evaluate_image_model(final_model, img_val_loader)
        
        results.append({
            'Model': f'MLPER-Inspired (Image, {best_epochs} epochs)',
            'MAE': img_mae,
            'RMSE': img_rmse,
            'R-squared': img_r2,
            'Parameter Count': utils.count_parameters(final_model),
            'Inference Time (ms/sample)': img_inf_time,
        })
        
        torch.save(final_model.state_dict(), "mlper_regression_model.pth")
        print(f"Saved MLPER regression model state (trained for {best_epochs} epochs) to mlper_regression_model.pth")
        
    except Exception as e:
        print(f"Warning: Could not train image model. Error: {e}")
        print("This may be due to missing librosa or other dependencies.")
        import traceback
        traceback.print_exc()

    # --- 5. Save Results ---
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\n--- Experiment Complete ---")
    print("Results saved to model_comparison_results.csv")
    print(results_df)


if __name__ == "__main__":
    main()


