import numpy as np
from scipy.signal import spectrogram  # kept for potential future use
import matplotlib.pyplot as plt
import librosa


def generate_spectrograms(signal: np.ndarray, sr: int = 100):
    """
    Generates magnitude (in dB) and phase spectrograms from a time-series signal.
    Parameters are simplified but inspired by Damikoukas & Lagaros (2023).

    Returns:
        mag_db: 2D numpy array (frequency_bins x time_frames)
        phase:  2D numpy array (frequency_bins x time_frames), radians
    """
    # STFT parameters (simplified)
    n_fft = 400            # DFT points
    hop_length = n_fft - 6 # ~6-sample overlap
    win_length = 8         # Hann window length

    # Compute complex STFT
    stft_result = librosa.stft(
        y=signal.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        dtype=np.complex64,
    )

    magnitude = np.abs(stft_result)
    phase = np.angle(stft_result)

    # Convert magnitude to decibels for better numerical stability as image input
    mag_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    return mag_db, phase


def plot_spectrograms(mag_spec: np.ndarray, phase_spec: np.ndarray, title_prefix: str = ""):
    """Visualize a pair of spectrograms (magnitude in dB, phase in radians)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    img1 = axes[0].imshow(mag_spec, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'{title_prefix} Magnitude Spectrogram (dB)')
    axes[0].set_ylabel('Frequency Bins')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')

    img2 = axes[1].imshow(phase_spec, aspect='auto', origin='lower', cmap='twilight')
    axes[1].set_title(f'{title_prefix} Phase Spectrogram (radians)')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Frequency Bins')
    fig.colorbar(img2, ax=axes[1])

    plt.tight_layout()
    plt.show()


