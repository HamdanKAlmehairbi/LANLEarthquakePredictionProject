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
    
    # Correct Time-Series Split (BEFORE scaling to prevent data leakage)
    num_samples, seq_len, num_features = X_seq_featured.shape
    
    if test_size == 0.0:
        # Use all data for training (no validation split)
        X_train_raw = X_seq_featured
        y_train_s = y_seq_featured
        X_val_raw = np.array([]).reshape(0, seq_len, num_features)
        y_val_s = np.array([])
    else:
        split_index = int(num_samples * (1 - test_size))
        X_train_raw, X_val_raw = X_seq_featured[:split_index], X_seq_featured[split_index:]
        y_train_s, y_val_s = y_seq_featured[:split_index], y_seq_featured[split_index:]
    
    # Scale the features: fit scaler ONLY on training data, then transform both
    X_train_reshaped = X_train_raw.reshape(-1, num_features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_s = X_train_scaled.reshape(X_train_raw.shape[0], seq_len, num_features)
    
    # Transform validation data using the scaler fitted on training data (if validation set exists)
    if len(X_val_raw) > 0:
        X_val_reshaped = X_val_raw.reshape(-1, num_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_s = X_val_scaled.reshape(X_val_raw.shape[0], seq_len, num_features)
    else:
        X_val_s = np.array([]).reshape(0, seq_len, num_features)
    
    print(f'\nFinal data shape for temporal models: Train={X_train_s.shape}, Val={X_val_s.shape if len(X_val_s) > 0 else (0, seq_len, num_features)}')

    # Create PyTorch DataLoaders
    X_train_tensor = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_s, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    if len(X_val_s) > 0:
        X_val_tensor = torch.tensor(X_val_s, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_s, dtype=torch.float32).view(-1, 1)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    else:
        # Create empty validation loader
        val_dataset = TensorDataset(torch.tensor([], dtype=torch.float32).reshape(0, seq_len, num_features),
                                   torch.tensor([], dtype=torch.float32).reshape(0, 1))
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
    if test_size == 0.0:
        # Use all data for training (no validation split)
        mag_train, mag_val = mag_tensor, torch.tensor([], dtype=torch.float32).reshape(0, *mag_tensor.shape[1:])
        phase_train, phase_val = phase_tensor, torch.tensor([], dtype=torch.float32).reshape(0, *phase_tensor.shape[1:])
        y_train, y_val = target_tensor, torch.tensor([], dtype=torch.float32).reshape(0, 1)
    else:
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


def convert_df_to_features(df, main_segment_size=150_000, sub_segment_size=15_000):
    """
    Converts a raw dataframe to numpy arrays of features and labels.
    This is used for K-Fold Cross-Validation where we need to split the data
    before scaling to prevent data leakage.
    
    Args:
        df: Raw dataframe with 'acoustic_data' and 'time_to_failure' columns
        main_segment_size: Size of each main segment (150,000)
        sub_segment_size: Size of each sub-segment (15,000)
    
    Returns:
        X_sequences: numpy array of shape (num_samples, seq_len, num_features)
        y_sequences: numpy array of shape (num_samples,)
        scaler_template: A StandardScaler instance (not fitted yet)
        feature_names: List of feature names
    """
    print("Converting DataFrame to feature sequences...")
    num_sub_segments = main_segment_size // sub_segment_size
    num_main_segments = len(df) // main_segment_size
    
    X_sequences = []
    y_sequences = []
    feature_names = None
    
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
        
        # Stack sub-segment features into a sequence
        X_sequences.append(pd.concat(main_segment_features, axis=1).T.values)
        y_sequences.append(df['time_to_failure'].iloc[(i+1)*main_segment_size - 1])
    
    X_array = np.array(X_sequences)
    y_array = np.array(y_sequences)
    
    # Return an unfitted scaler template that will be fitted per fold
    return X_array, y_array, StandardScaler(), feature_names



