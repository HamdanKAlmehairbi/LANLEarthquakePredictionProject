import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from features import create_features
from image_transformer import generate_spectrograms


def prepare_data(df, main_segment_size=150_000, sub_segment_size=15_000, test_size=0.2):
    """
    Orchestrates the full data preparation pipeline:
    1. Segments data into sequences of sub-segments.
    2. Engineers features for each sub-segment.
    3. Scales features using StandardScaler.
    4. Performs a sequential (time-series aware) train/validation split.
    5. Creates PyTorch DataLoaders.
    """
    print("Starting data preparation: Engineering features into sequences...")
    num_sub_segments = main_segment_size // sub_segment_size
    num_main_segments = len(df) // main_segment_size
    X_sequences, y_sequences, feature_names = [], [], None
    for i in tqdm(range(num_main_segments)):
        main_segment_features = []
        for j in range(num_sub_segments):
            start = i * main_segment_size + j * sub_segment_size
            end = start + sub_segment_size
            sub_segment = df['acoustic_data'].iloc[start:end]
            features = create_features(sub_segment, sub_segment_size)
            if feature_names is None:
                feature_names = features.index.tolist()
            main_segment_features.append(features)
        
        X_sequences.append(pd.concat(main_segment_features, axis=1).T.values)
        y_sequences.append(df['time_to_failure'].iloc[(i+1)*main_segment_size - 1])
    X_seq_featured = np.array(X_sequences)
    y_seq_featured = np.array(y_sequences)
    
    # Scale the features
    num_samples, seq_len, num_features = X_seq_featured.shape
    X_seq_reshaped = X_seq_featured.reshape(num_samples * seq_len, num_features)
    scaler = StandardScaler()
    X_seq_scaled = scaler.fit_transform(X_seq_reshaped)
    X_final = X_seq_scaled.reshape(num_samples, seq_len, num_features)
    print(f'\nFinal data shape for temporal models: {X_final.shape}')

    # Correct Time-Series Split
    split_index = int(num_samples * (1 - test_size))
    X_train_s, X_val_s = X_final[:split_index], X_final[split_index:]
    y_train_s, y_val_s = y_seq_featured[:split_index], y_seq_featured[split_index:]
    print(f'Train shape: {X_train_s.shape}, Validation shape: {X_val_s.shape}')

    # Create PyTorch DataLoaders
    X_train_tensor = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_s, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_s, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, num_features, feature_names, scaler

def prepare_spectrogram_data(df, segment_size=150_000, test_size=0.2):
    """
    Prepares data for image-based models by converting time-series segments to spectrograms.
    Returns DataLoaders with (mag_spec, phase_spec, time_to_failure) tuples.
    """
    print("Starting spectrogram data preparation...")
    num_segments = len(df) // segment_size
    
    mag_specs, phase_specs, targets = [], [], []
    
    for i in tqdm(range(num_segments)):
        segment = df['acoustic_data'].values[i * segment_size:(i + 1) * segment_size]
        target = df['time_to_failure'].iloc[(i + 1) * segment_size - 1]
        
        # Generate spectrograms
        mag_spec, phase_spec = generate_spectrograms(segment)
        
        # Normalize to [0, 1] range for model input
        def normalize(img):
            img_min, img_max = img.min(), img.max()
            return (img - img_min) / (img_max - img_min + 1e-6)
        
        mag_specs.append(normalize(mag_spec))
        phase_specs.append(normalize(phase_spec))
        targets.append(target)
    
    # Convert to tensors
    mag_tensor = torch.tensor(np.array(mag_specs), dtype=torch.float32)
    phase_tensor = torch.tensor(np.array(phase_specs), dtype=torch.float32)
    target_tensor = torch.tensor(np.array(targets), dtype=torch.float32).view(-1, 1)
    
    # Time-series split
    split_index = int(len(targets) * (1 - test_size))
    mag_train, mag_val = mag_tensor[:split_index], mag_tensor[split_index:]
    phase_train, phase_val = phase_tensor[:split_index], phase_tensor[split_index:]
    y_train, y_val = target_tensor[:split_index], target_tensor[split_index:]
    
    # Create datasets (using tuple format for mag, phase, target)
    train_dataset = TensorDataset(mag_train, phase_train, y_train)
    val_dataset = TensorDataset(mag_val, phase_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f'Spectrogram data: Train={len(train_dataset)}, Val={len(val_dataset)}')
    return train_loader, val_loader


def prepare_test_data(test_folder='test', main_segment_size=150_000, sub_segment_size=15_000, scaler=None, feature_names=None):
    """
    Prepares test data from /test folder for final evaluation.
    Each CSV file in test folder contains one segment of 150,000 samples.
    
    Args:
        test_folder: Path to test folder containing CSV files
        main_segment_size: Size of each main segment (150,000)
        sub_segment_size: Size of each sub-segment (15,000)
        scaler: Fitted StandardScaler from training data (must be provided)
        feature_names: Feature names from training data (must be provided)
    
    Returns:
        test_loader: DataLoader for test data
        segment_ids: List of segment IDs (filenames) for each sample
    """
    import os
    import glob
    
    if scaler is None or feature_names is None:
        raise ValueError("scaler and feature_names must be provided from training data")
    
    print(f"Loading test data from {test_folder}...")
    test_files = sorted(glob.glob(os.path.join(test_folder, '*.csv')))
    print(f"Found {len(test_files)} test segments")
    
    X_sequences = []
    segment_ids = []
    num_sub_segments = main_segment_size // sub_segment_size
    
    for test_file in tqdm(test_files):
        # Load test segment
        df_test = pd.read_csv(test_file, dtype={'acoustic_data': np.int16})
        segment_id = os.path.basename(test_file).replace('.csv', '')
        
        # Check if segment has correct size
        if len(df_test) != main_segment_size:
            print(f"Warning: {segment_id} has {len(df_test)} samples, expected {main_segment_size}")
            continue
        
        # Extract features same way as training
        main_segment_features = []
        for j in range(num_sub_segments):
            start = j * sub_segment_size
            end = start + sub_segment_size
            sub_segment = df_test['acoustic_data'].iloc[start:end]
            features = create_features(sub_segment, sub_segment_size)
            main_segment_features.append(features)
        
        X_sequences.append(pd.concat(main_segment_features, axis=1).T.values)
        segment_ids.append(segment_id)
    
    X_test = np.array(X_sequences)
    
    # Scale using the same scaler from training
    num_samples, seq_len, num_features = X_test.shape
    X_test_reshaped = X_test.reshape(num_samples * seq_len, num_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_final = X_test_scaled.reshape(num_samples, seq_len, num_features)
    
    print(f'Test data shape: {X_test_final.shape}')
    
    # Create DataLoader (no labels for test data)
    X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return test_loader, segment_ids


def prepare_test_spectrogram_data(test_folder='test', segment_size=150_000):
    """
    Prepares test spectrogram data from /test folder for image models.
    
    Args:
        test_folder: Path to test folder containing CSV files
        segment_size: Size of each segment (150,000)
    
    Returns:
        test_loader: DataLoader for test spectrogram data
        segment_ids: List of segment IDs (filenames) for each sample
    """
    import os
    import glob
    
    print(f"Loading test spectrogram data from {test_folder}...")
    test_files = sorted(glob.glob(os.path.join(test_folder, '*.csv')))
    print(f"Found {len(test_files)} test segments")
    
    mag_specs, phase_specs = [], []
    segment_ids = []
    
    for test_file in tqdm(test_files):
        df_test = pd.read_csv(test_file, dtype={'acoustic_data': np.int16})
        segment_id = os.path.basename(test_file).replace('.csv', '')
        
        if len(df_test) != segment_size:
            print(f"Warning: {segment_id} has {len(df_test)} samples, expected {segment_size}")
            continue
        
        segment = df_test['acoustic_data'].values
        
        # Generate spectrograms
        mag_spec, phase_spec = generate_spectrograms(segment)
        
        # Normalize to [0, 1] range
        def normalize(img):
            img_min, img_max = img.min(), img.max()
            return (img - img_min) / (img_max - img_min + 1e-6)
        
        mag_specs.append(normalize(mag_spec))
        phase_specs.append(normalize(phase_spec))
        segment_ids.append(segment_id)
    
    # Convert to tensors
    mag_tensor = torch.tensor(np.array(mag_specs), dtype=torch.float32)
    phase_tensor = torch.tensor(np.array(phase_specs), dtype=torch.float32)
    
    # Create dataset
    test_dataset = TensorDataset(mag_tensor, phase_tensor)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f'Test spectrogram data: {len(test_dataset)} segments')
    return test_loader, segment_ids



