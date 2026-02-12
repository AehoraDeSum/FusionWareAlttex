"""Rubidium atomic vapor cell quantum simulation using QuTiP."""

import numpy as np
import qutip as qt
from scipy.constants import h, hbar, c, k as k_B
from typing import Tuple, Optional


class RubidiumVaporCell:
    """
    Quantum simulation of rubidium atomic vapor cell response to gravitational waves.
    
    The cell contains rubidium atoms that interact with laser light. Gravitational
    waves cause strain that modulates the atomic energy levels and transition frequencies.
    """
    
    def __init__(
        self,
        temperature: float = 300.0,  # Kelvin
        pressure: float = 1e-3,  # Torr
        cell_length: float = 0.1,  # meters
        atomic_mass: int = 87,
        decay_rate: float = 1e6,  # Hz
    ):
        """
        Initialize rubidium vapor cell parameters.
        
        Parameters:
        -----------
        temperature : float
            Cell temperature in Kelvin
        pressure : float
            Gas pressure in Torr
        cell_length : float
            Length of the vapor cell in meters
        atomic_mass : int
            Atomic mass number (85 or 87)
        decay_rate : float
            Spontaneous decay rate in Hz
        """
        self.temperature = temperature
        self.pressure = pressure
        self.cell_length = cell_length
        self.atomic_mass = atomic_mass
        self.decay_rate = decay_rate
        
        # Rubidium D2 line parameters (780 nm)
        self.wavelength = 780e-9  # meters
        self.frequency = c / self.wavelength  # Hz
        self.omega0 = 2 * np.pi * self.frequency  # rad/s
        
        # Energy levels (simplified two-level system)
        self.ground_energy = -1.59e-18  # Joules
        self.excited_energy = -1.58e-18  # Joules
        self.transition_energy = self.excited_energy - self.ground_energy
        
        # Quantum operators for two-level atom
        self.sigma_minus = qt.destroy(2)  # |g><e|
        self.sigma_plus = qt.create(2)    # |e><g|
        self.sigma_z = qt.sigmaz()        # |e><e| - |g><g|
        
        # Initial state: ground state
        self.initial_state = qt.basis(2, 0)  # |g>
        
    def hamiltonian(
        self,
        laser_detuning: float = 0.0,
        strain: float = 0.0,
        laser_rabi_frequency: float = 1e6,  # Hz
    ) -> qt.Qobj:
        """
        Construct Hamiltonian for the atom-laser system with gravitational wave strain.
        
        Parameters:
        -----------
        laser_detuning : float
            Laser detuning from resonance in Hz
        strain : float
            Gravitational wave strain (dimensionless)
        laser_rabi_frequency : float
            Rabi frequency of the laser in Hz
            
        Returns:
        --------
        qt.Qobj
            Hamiltonian operator
        """
        # Strain modifies the transition frequency
        # Strain sensitivity: ~1e-21 strain causes ~1 Hz frequency shift
        strain_shift = strain * 1e21  # Convert to frequency shift (Hz)
        total_detuning = laser_detuning + strain_shift
        
        # Atomic Hamiltonian: energy splitting
        H_atom = 0.5 * hbar * 2 * np.pi * total_detuning * self.sigma_z
        
        # Laser-atom interaction: Rabi coupling
        H_laser = 0.5 * hbar * 2 * np.pi * laser_rabi_frequency * (
            self.sigma_plus + self.sigma_minus
        )
        
        return H_atom + H_laser
    
    def collapse_operators(self) -> list:
        """
        Collapse operators for Lindblad master equation (spontaneous emission).
        
        Returns:
        --------
        list
            List of collapse operators
        """
        # Spontaneous decay from excited to ground state
        gamma = self.decay_rate
        return [np.sqrt(gamma) * self.sigma_minus]
    
    def evolve(
        self,
        time_points: np.ndarray,
        strain_signal: np.ndarray,
        laser_detuning: float = 0.0,
        laser_rabi_frequency: float = 1e6,
    ) -> Tuple[qt.Result, np.ndarray]:
        """
        Evolve the quantum system under gravitational wave strain.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time points for evolution
        strain_signal : np.ndarray
            Gravitational wave strain as function of time
        laser_detuning : float
            Laser detuning from resonance in Hz
        laser_rabi_frequency : float
            Rabi frequency of the laser in Hz
            
        Returns:
        --------
        Tuple[qt.Result, np.ndarray]
            Evolution result and measured signal (excited state population)
        """
        # Time-dependent Hamiltonian function
        def hamiltonian_func(t, args):
            # Interpolate strain at time t
            strain = np.interp(t, time_points, strain_signal)
            H = self.hamiltonian(
                laser_detuning=laser_detuning,
                strain=strain,
                laser_rabi_frequency=laser_rabi_frequency,
            )
            return H
        
        # Solve master equation
        c_ops = self.collapse_operators()
        result = qt.mesolve(
            hamiltonian_func,
            self.initial_state,
            time_points,
            c_ops,
            args={},
        )
        
        # Measure excited state population as signal
        excited_population = qt.expect(self.sigma_plus * self.sigma_minus, result.states)
        
        return result, excited_population
    
    def generate_response(
        self,
        time_points: np.ndarray,
        strain_signal: np.ndarray,
        noise_level: float = 1e-22,
        **kwargs
    ) -> np.ndarray:
        """
        Generate quantum response signal with noise.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Time points
        strain_signal : np.ndarray
            Gravitational wave strain
        noise_level : float
            Measurement noise level (strain units)
        **kwargs
            Additional parameters for evolve()
            
        Returns:
        --------
        np.ndarray
            Measured signal with noise
        """
        _, signal = self.evolve(time_points, strain_signal, **kwargs)
        
        # Add measurement noise
        noise = np.random.normal(0, noise_level * np.max(signal), size=signal.shape)
        noisy_signal = signal + noise
        
        return noisy_signal
