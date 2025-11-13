import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(path_to_csv, nrows=60_000_000):
    """
    Loads a fraction of the LANL earthquake training data.
    If the file is not found, it creates a dummy dataset for demonstration.
    """
    try:
        df = pd.read_csv(
            path_to_csv,
            nrows=nrows,
            dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}
        )
        print(f"Loaded {nrows:,} rows from {path_to_csv}")
    except FileNotFoundError:
        print(f"Dataset not found at {path_to_csv}. Creating dummy data.")
        df = pd.DataFrame({
            'acoustic_data': np.random.randint(-100, 100, size=nrows),
            'time_to_failure': np.linspace(10, 0, nrows)
        })
    return df


def plot_eda_segment(df):
    """Plots a sample segment of acoustic data vs. time to failure."""
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    color = 'tab:blue'
    ax1.set_title('Acoustic Data and Time to Failure for a Sample Segment')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Acoustic Data', color=color)
    ax1.plot(df['acoustic_data'].values[:150000], color=color, alpha=0.7, label='Acoustic Data')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time to Failure', color=color)
    ax2.plot(df['time_to_failure'].values[:150000], color=color, label='Time to Failure')
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    plt.show()


