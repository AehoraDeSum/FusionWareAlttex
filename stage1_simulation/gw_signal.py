"""Gravitational wave signal generation."""

import numpy as np
from typing import Tuple, Optional


def generate_gravitational_wave(
    time_points: np.ndarray,
    frequency: float,
    amplitude: float,
    phase: float = 0.0,
    waveform_type: str = "sine",
) -> np.ndarray:
    """
    Generate gravitational wave strain signal.
    
    Parameters:
    -----------
    time_points : np.ndarray
        Time array in seconds
    frequency : float
        GW frequency in Hz
    amplitude : float
        Strain amplitude (dimensionless, typically ~1e-21)
    phase : float
        Initial phase in radians
    waveform_type : str
        Type of waveform: "sine", "chirp", or "burst"
        
    Returns:
    --------
    np.ndarray
        Gravitational wave strain as function of time
    """
    omega = 2 * np.pi * frequency
    
    if waveform_type == "sine":
        strain = amplitude * np.sin(omega * time_points + phase)
    elif waveform_type == "chirp":
        # Chirp signal: frequency increases with time
        chirp_rate = frequency / time_points[-1]
        instantaneous_freq = frequency + chirp_rate * time_points
        phase_accumulated = 2 * np.pi * np.cumsum(instantaneous_freq) * np.diff(time_points)[0]
        strain = amplitude * np.sin(phase_accumulated + phase)
    elif waveform_type == "burst":
        # Gaussian burst
        t0 = time_points[len(time_points) // 2]
        sigma = 1.0 / frequency
        envelope = np.exp(-0.5 * ((time_points - t0) / sigma) ** 2)
        strain = amplitude * envelope * np.sin(omega * time_points + phase)
    else:
        raise ValueError(f"Unknown waveform type: {waveform_type}")
    
    return strain


def generate_noise(
    time_points: np.ndarray,
    noise_level: float = 1e-22,
    noise_type: str = "white",
) -> np.ndarray:
    """
    Generate noise signal.
    
    Parameters:
    -----------
    time_points : np.ndarray
        Time array
    noise_level : float
        Noise amplitude (strain units)
    noise_type : str
        Type of noise: "white" or "colored"
        
    Returns:
    --------
    np.ndarray
        Noise signal
    """
    if noise_type == "white":
        noise = np.random.normal(0, noise_level, size=len(time_points))
    elif noise_type == "colored":
        # 1/f noise (pink noise)
        fft_size = len(time_points)
        freqs = np.fft.fftfreq(fft_size, d=time_points[1] - time_points[0])
        # Avoid division by zero
        freqs[0] = freqs[1]
        power_spectrum = 1.0 / np.abs(freqs)
        power_spectrum = power_spectrum / np.max(power_spectrum)
        noise_fft = np.random.normal(0, 1, fft_size) * np.sqrt(power_spectrum)
        noise = np.real(np.fft.ifft(noise_fft))
        noise = noise * noise_level / np.std(noise)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noise


def create_synthetic_signal(
    time_points: np.ndarray,
    has_signal: bool = True,
    frequency_range: Tuple[float, float] = (10, 1000),
    amplitude_range: Tuple[float, float] = (1e-21, 1e-19),
    noise_level: float = 1e-22,
    waveform_type: str = "sine",
) -> Tuple[np.ndarray, bool]:
    """
    Create synthetic signal with or without gravitational wave.
    
    Parameters:
    -----------
    time_points : np.ndarray
        Time array
    has_signal : bool
        Whether to include a gravitational wave signal
    frequency_range : Tuple[float, float]
        Range of possible GW frequencies (Hz)
    amplitude_range : Tuple[float, float]
        Range of possible strain amplitudes
    noise_level : float
        Background noise level
    waveform_type : str
        Type of waveform
        
    Returns:
    --------
    Tuple[np.ndarray, bool]
        Strain signal and label (True if signal present)
    """
    # Generate noise
    noise = generate_noise(time_points, noise_level)
    
    if has_signal:
        # Randomly sample frequency and amplitude
        frequency = np.random.uniform(frequency_range[0], frequency_range[1])
        amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Generate GW signal
        gw_signal = generate_gravitational_wave(
            time_points, frequency, amplitude, phase, waveform_type
        )
        
        total_signal = gw_signal + noise
        label = True
    else:
        total_signal = noise
        label = False
    
    return total_signal, label
