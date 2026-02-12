"""Parallel version of dataset generation for faster processing."""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
from multiprocessing import Pool, cpu_count
from functools import partial

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1_simulation.rubidium_cell import RubidiumVaporCell
from stage1_simulation.gw_signal import create_synthetic_signal
from utils.config_loader import load_config
from utils.data_utils import save_dataset


def generate_single_sample(args):
    """Generate a single sample (for parallel processing)."""
    (i, time_points, gw_config, cell, noise_level) = args
    
    # Randomly decide if this sample has a signal
    has_signal = np.random.rand() > 0.5
    
    # Generate gravitational wave strain (or noise only)
    frequency_range = gw_config.get("frequency_range", [10, 1000])
    amplitude_range = gw_config.get("amplitude_range", [1e-21, 1e-19])
    
    # Ensure numeric types
    frequency_range = [float(f) for f in frequency_range]
    amplitude_range = [float(a) for a in amplitude_range]
    noise_level = float(noise_level)
    
    strain_signal, label = create_synthetic_signal(
        time_points,
        has_signal=has_signal,
        frequency_range=tuple(frequency_range),
        amplitude_range=tuple(amplitude_range),
        noise_level=noise_level,
    )
    
    # Generate quantum response
    response = cell.generate_response(
        time_points,
        strain_signal,
        noise_level=noise_level,
        laser_detuning=np.random.uniform(-1e6, 1e6),  # Random detuning
        laser_rabi_frequency=np.random.uniform(0.5e6, 2e6),  # Random Rabi frequency
    )
    
    return response, int(label)


def generate_dataset_parallel(
    num_samples: int,
    time_steps: int = 1000,
    dt: float = 0.001,
    output_path: str = "data/synthetic_dataset.h5",
    config: dict = None,
    num_workers: int = None,
):
    """
    Generate synthetic dataset using parallel processing.
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    time_steps : int
        Number of time steps per sample
    dt : float
        Time step in seconds
    output_path : str
        Path to save dataset
    config : dict
        Configuration dictionary
    num_workers : int
        Number of parallel workers (None = use all available CPUs)
    """
    if config is None:
        config = load_config()
    
    if num_workers is None:
        num_workers = cpu_count()
    
    sim_config = config.get("simulation", {})
    gw_config = sim_config.get("gravitational_wave", {})
    
    # Initialize rubidium cell (will be recreated in each worker)
    rb_params = sim_config.get("rubidium_params", {})
    cell_params = sim_config.get("cell_params", {})
    
    # Time array (ensure numeric types)
    time_steps = int(time_steps)
    dt = float(dt)
    time_points = np.linspace(0, time_steps * dt, time_steps)
    
    noise_level = float(gw_config.get("noise_level", 1e-22))
    
    print(f"Generating {num_samples} samples using {num_workers} parallel workers...")
    print(f"Time steps per sample: {time_steps}")
    print(f"Time step size: {dt} seconds")
    
    # Prepare arguments for parallel processing
    # Note: We pass cell parameters instead of cell object (QuTiP objects aren't easily pickled)
    cell_params_dict = {
        'temperature': float(cell_params.get("temperature", 300)),
        'pressure': float(cell_params.get("pressure", 1e-3)),
        'cell_length': float(cell_params.get("length", 0.1)),
        'atomic_mass': int(rb_params.get("atomic_mass", 87)),
        'decay_rate': float(rb_params.get("decay_rate", 1e6)),
    }
    
    # Create worker function that initializes cell
    def worker_with_cell(args):
        i, time_points, gw_config, cell_params_dict, noise_level = args
        cell = RubidiumVaporCell(**cell_params_dict)
        return generate_single_sample((i, time_points, gw_config, cell, noise_level))
    
    # Prepare arguments
    args_list = [
        (i, time_points, gw_config, cell_params_dict, noise_level)
        for i in range(num_samples)
    ]
    
    # Process in parallel
    data = []
    labels = []
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(worker_with_cell, args_list),
            total=num_samples,
            desc="Generating samples"
        ))
    
    # Unpack results
    for response, label in results:
        data.append(response)
        labels.append(label)
    
    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Normalize data
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean) / data_std
    
    print(f"\nDataset statistics:")
    print(f"  Signal samples: {np.sum(labels)}")
    print(f"  Noise samples: {np.sum(1 - labels)}")
    print(f"  Data shape: {data.shape}")
    print(f"  Data mean: {np.mean(data):.6f}, std: {np.std(data):.6f}")
    
    # Save dataset
    save_dataset(data, labels, output_path)
    print(f"\nDataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic quantum dataset (parallel)")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=1000,
        help="Number of time steps per sample",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Time step in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_dataset.h5",
        help="Output file path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: use all CPUs)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config()
    
    # Generate dataset
    generate_dataset_parallel(
        num_samples=args.num_samples,
        time_steps=args.time_steps,
        dt=args.dt,
        output_path=args.output,
        config=config,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
