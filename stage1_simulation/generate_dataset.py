"""Generate synthetic dataset using quantum simulation."""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1_simulation.rubidium_cell import RubidiumVaporCell
from stage1_simulation.gw_signal import create_synthetic_signal
from utils.config_loader import load_config
from utils.data_utils import save_dataset


def generate_dataset(
    num_samples: int,
    time_steps: int = 1000,
    dt: float = 0.001,
    output_path: str = "data/synthetic_dataset.h5",
    config: dict = None,
):
    """
    Generate synthetic dataset of quantum responses to gravitational waves.
    
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
    """
    if config is None:
        config = load_config()
    
    sim_config = config.get("simulation", {})
    gw_config = sim_config.get("gravitational_wave", {})
    
    # Initialize rubidium cell
    rb_params = sim_config.get("rubidium_params", {})
    cell_params = sim_config.get("cell_params", {})
    
    # Ensure numeric types
    cell = RubidiumVaporCell(
        temperature=float(cell_params.get("temperature", 300)),
        pressure=float(cell_params.get("pressure", 1e-3)),
        cell_length=float(cell_params.get("length", 0.1)),
        atomic_mass=int(rb_params.get("atomic_mass", 87)),
        decay_rate=float(rb_params.get("decay_rate", 1e6)),
    )
    
    # Time array (ensure numeric types)
    time_steps = int(time_steps)
    dt = float(dt)
    time_points = np.linspace(0, time_steps * dt, time_steps)
    
    # Storage arrays
    data = []
    labels = []
    
    print(f"Generating {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        # Randomly decide if this sample has a signal
        has_signal = np.random.rand() > 0.5
        
        # Generate gravitational wave strain (or noise only)
        frequency_range = gw_config.get("frequency_range", [10, 1000])
        amplitude_range = gw_config.get("amplitude_range", [1e-21, 1e-19])
        noise_level = gw_config.get("noise_level", 1e-22)
        
        # Ensure numeric types (YAML may parse scientific notation as strings)
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
        
        # Store data (response signal)
        data.append(response)
        labels.append(int(label))
    
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
    parser = argparse.ArgumentParser(description="Generate synthetic quantum dataset")
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
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config()
    
    # Generate dataset
    generate_dataset(
        num_samples=args.num_samples,
        time_steps=args.time_steps,
        dt=args.dt,
        output_path=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()
