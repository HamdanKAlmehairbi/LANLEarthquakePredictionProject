import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import get_device
import torch.nn as nn


def train_model(model, train_loader, val_loader, epochs=30, learning_rate=0.001):
    """Trains a PyTorch model and returns the training history."""
    device = get_device()
    model.to(device)
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    history = {'train_loss': [], 'val_loss': []}
    
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
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    return history


def evaluate_model(model, loader):
    """Evaluates a model on a given data loader and returns performance metrics."""
    device = get_device()
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.numpy())
    end_time = time.time()
    
    inference_time = (end_time - start_time) / len(loader.dataset) * 1000  # ms per sample
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    return mae, rmse, r2, inference_time


def train_image_model(model, train_loader, val_loader, epochs=30, learning_rate=0.001):
    """Trains an image-based model that takes (mag, phase) inputs."""
    device = get_device()
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for mag, phase, labels in train_loader:
            mag, phase = mag.to(device), phase.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(mag, phase)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for mag, phase, labels in val_loader:
                mag, phase = mag.to(device), phase.to(device)
                labels = labels.to(device)
                outputs = model(mag, phase)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print every epoch for image models (they train for fewer epochs)
        if epochs <= 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}, LR: {current_lr:.6f}')
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1}/{epochs}, Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}, LR: {current_lr:.6f}')
    
    return history


def evaluate_image_model(model, loader):
    """Evaluates an image-based model on a data loader."""
    device = get_device()
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    
    start_time = time.time()
    with torch.no_grad():
        for mag, phase, labels in loader:
            mag, phase = mag.to(device), phase.to(device)
            outputs = model(mag, phase)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.numpy())
    end_time = time.time()
    
    inference_time = (end_time - start_time) / len(loader.dataset) * 1000
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    return mae, rmse, r2, inference_time


def plot_loss(history, title):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training MAE')
    plt.plot(history['val_loss'], label='Validation MAE')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()


def train_for_submission(model, train_loader, optimizer, scheduler=None, epochs=30):
    """
    A simplified training loop for submission without validation.
    Used in K-Fold Cross-Validation where we train on fold-specific data.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        scheduler: Optional learning rate scheduler
        epochs: Number of epochs to train
    """
    device = get_device()
    model.to(device)
    criterion = nn.L1Loss()
    
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
            if scheduler is not None:
                scheduler.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            print(f'Epoch {epoch+1}/{epochs}, Train MAE: {train_loss:.4f}, LR: {current_lr:.6f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Train MAE: {train_loss:.4f}')


