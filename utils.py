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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
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


