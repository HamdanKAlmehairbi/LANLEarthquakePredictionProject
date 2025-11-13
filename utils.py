import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_inline import backend_inline


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """Returns the available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')
        print(f'CUDA Version: {torch.version.cuda}')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')
        # Check if NVIDIA GPU exists but PyTorch can't use it
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                print('⚠️  WARNING: NVIDIA GPU detected but PyTorch cannot use it!')
                print('   This usually means PyTorch was installed without CUDA support.')
                print('   To fix this, install PyTorch with CUDA support:')
                print('   Visit: https://pytorch.org/get-started/locally/')
                print('   Or run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
        except:
            pass  # nvidia-smi not available or other error
    return device


def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_plotting_style():
    """Sets a consistent and professional style for matplotlib plots."""
    backend_inline.set_matplotlib_formats('retina')
    sns.set_theme(style='whitegrid', context='talk', font_scale=1.1)
    sns.set_palette('deep')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 120,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    print("Plotting style configured.")


