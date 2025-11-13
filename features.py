import pandas as pd
import numpy as np


def create_features(segment, segment_size=150_000):
    """Calculates a set of statistical and spectral features for a given data segment."""
    features = {}
    x = pd.Series(segment)
    
    # Time-domain statistical features
    features['mean'] = x.mean()
    features['std'] = x.std()
    features['max'] = x.max()
    features['min'] = x.min()
    features['abs_mean'] = np.abs(x).mean()
    
    # Quantiles
    features['q99'] = np.quantile(x, 0.99)
    features['q95'] = np.quantile(x, 0.95)
    features['q05'] = np.quantile(x, 0.05)
    features['q01'] = np.quantile(x, 0.01)
    
    # Frequency-domain features (FFT)
    fft_coeffs = np.fft.fft(x)
    fft_real = np.real(fft_coeffs[:5000])
    fft_imag = np.imag(fft_coeffs[:5000])
    
    features['fft_real_mean'] = fft_real.mean()
    features['fft_real_std'] = fft_real.std()
    features['fft_imag_mean'] = fft_imag.mean()
    features['fft_imag_std'] = fft_imag.std()

    # Rolling features over chunks
    num_chunks = 10
    chunk_size = segment_size // num_chunks
    chunk_stds = [x.iloc[i*chunk_size:(i+1)*chunk_size].std() for i in range(num_chunks)]
    features['mean_of_chunk_stds'] = np.mean(chunk_stds)
    features['std_of_chunk_stds'] = np.std(chunk_stds)
    
    return pd.Series(features)


