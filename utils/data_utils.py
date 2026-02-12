"""Data utilities for handling datasets."""
import numpy as np
import h5py
from pathlib import Path


def save_dataset(data, labels, filepath):
    """Save dataset to HDF5 file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip')
        f.create_dataset('labels', data=labels, compression='gzip')


def load_dataset(filepath):
    """Load dataset from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        data = f['data'][:]
        labels = f['labels'][:]
    return data, labels


def normalize_data(data):
    """Normalize data to zero mean and unit variance."""
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    return (data - mean) / std
