"""Interface for prototype detector hardware."""

import serial
import time
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class DetectorInterface:
    """
    Interface to prototype detector hardware.
    
    The detector consists of:
    - Rubidium vapor cell in quartz glass tube
    - Precision laser system
    - Magnetic shielding
    - Piezoelectric actuators for calibration
    """
    
    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        timeout: float = 1.0,
    ):
        """
        Initialize detector interface.
        
        Parameters:
        -----------
        serial_port : str
            Serial port path
        baud_rate : int
            Serial communication baud rate
        timeout : float
            Serial timeout in seconds
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.connection: Optional[serial.Serial] = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to detector hardware."""
        try:
            self.connection = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=self.timeout,
            )
            time.sleep(2)  # Wait for connection to stabilize
            self.is_connected = True
            print(f"Connected to detector at {self.serial_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to detector: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from detector."""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.is_connected = False
            print("Disconnected from detector")
    
    def send_command(self, command: str) -> Optional[str]:
        """Send command to detector and read response."""
        if not self.is_connected:
            raise RuntimeError("Not connected to detector")
        
        try:
            self.connection.write(f"{command}\n".encode())
            time.sleep(0.1)
            response = self.connection.readline().decode().strip()
            return response
        except Exception as e:
            print(f"Error sending command: {e}")
            return None
    
    def read_signal(self, duration: float = 1.0, sampling_rate: float = 1000.0) -> np.ndarray:
        """
        Read signal from detector.
        
        Parameters:
        -----------
        duration : float
            Duration to read in seconds
        sampling_rate : float
            Sampling rate in Hz
            
        Returns:
        --------
        np.ndarray
            Signal data
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to detector")
        
        num_samples = int(duration * sampling_rate)
        signal = []
        
        # Request data acquisition
        self.send_command(f"ACQUIRE {num_samples} {sampling_rate}")
        
        # Read data samples
        for _ in range(num_samples):
            response = self.send_command("READ")
            if response:
                try:
                    value = float(response)
                    signal.append(value)
                except ValueError:
                    signal.append(0.0)
            time.sleep(1.0 / sampling_rate)
        
        return np.array(signal)
    
    def set_laser_power(self, power: float) -> bool:
        """
        Set laser power.
        
        Parameters:
        -----------
        power : float
            Laser power in mW
            
        Returns:
        --------
        bool
            Success status
        """
        response = self.send_command(f"LASER_POWER {power}")
        return response == "OK"
    
    def set_laser_frequency(self, frequency: float) -> bool:
        """
        Set laser frequency.
        
        Parameters:
        -----------
        frequency : float
            Laser frequency in Hz
            
        Returns:
        --------
        bool
            Success status
        """
        response = self.send_command(f"LASER_FREQ {frequency}")
        return response == "OK"
    
    def set_magnetic_field(self, field_strength: float) -> bool:
        """
        Set magnetic field strength.
        
        Parameters:
        -----------
        field_strength : float
            Magnetic field in Gauss
            
        Returns:
        --------
        bool
            Success status
        """
        response = self.send_command(f"MAGNETIC_FIELD {field_strength}")
        return response == "OK"
    
    def get_status(self) -> dict:
        """Get detector status."""
        status = {}
        
        response = self.send_command("STATUS")
        if response:
            # Parse status response (format: "key1:value1,key2:value2,...")
            for pair in response.split(","):
                if ":" in pair:
                    key, value = pair.split(":")
                    try:
                        status[key] = float(value)
                    except ValueError:
                        status[key] = value
        
        return status


class PiezoelectricActuator:
    """Interface for piezoelectric actuators used for calibration."""
    
    def __init__(self, detector: DetectorInterface, channel: int = 0):
        """
        Initialize piezoelectric actuator.
        
        Parameters:
        -----------
        detector : DetectorInterface
            Detector interface object
        channel : int
            Actuator channel number
        """
        self.detector = detector
        self.channel = channel
        self.strain_sensitivity = 1e-9  # Strain per volt
    
    def set_voltage(self, voltage: float) -> bool:
        """
        Set actuator voltage.
        
        Parameters:
        -----------
        voltage : float
            Voltage in volts (typically 0-100V)
            
        Returns:
        --------
        bool
            Success status
        """
        response = self.detector.send_command(f"PIEZO {self.channel} {voltage}")
        return response == "OK"
    
    def apply_strain(self, strain: float) -> bool:
        """
        Apply strain using actuator.
        
        Parameters:
        -----------
        strain : float
            Strain to apply (dimensionless)
            
        Returns:
        --------
        bool
            Success status
        """
        voltage = strain / self.strain_sensitivity
        return self.set_voltage(voltage)
    
    def get_voltage(self) -> Optional[float]:
        """Get current actuator voltage."""
        response = self.detector.send_command(f"PIEZO_READ {self.channel}")
        if response:
            try:
                return float(response)
            except ValueError:
                return None
        return None


class DetectorCalibrator:
    """Calibration system for prototype detector."""
    
    def __init__(
        self,
        detector: DetectorInterface,
        strain_sensitivity: float = 1e-9,
    ):
        """
        Initialize calibrator.
        
        Parameters:
        -----------
        detector : DetectorInterface
            Detector interface
        strain_sensitivity : float
            Strain sensitivity in strain per volt
        """
        self.detector = detector
        self.piezo = PiezoelectricActuator(detector, channel=0)
        self.piezo.strain_sensitivity = strain_sensitivity
        self.calibration_data = []
    
    def calibrate(
        self,
        voltage_range: Tuple[float, float] = (0, 100),
        num_steps: int = 100,
        sampling_duration: float = 1.0,
    ) -> dict:
        """
        Perform calibration by applying known strains and measuring response.
        
        Parameters:
        -----------
        voltage_range : Tuple[float, float]
            Voltage range for calibration
        num_steps : int
            Number of calibration steps
        sampling_duration : float
            Duration to sample at each step
            
        Returns:
        --------
        dict
            Calibration results
        """
        print("Starting calibration...")
        
        voltages = np.linspace(voltage_range[0], voltage_range[1], num_steps)
        responses = []
        
        for i, voltage in enumerate(voltages):
            print(f"Calibration step {i+1}/{num_steps}: {voltage:.2f}V", end="\r")
            
            # Apply voltage
            self.piezo.set_voltage(voltage)
            time.sleep(0.1)  # Wait for actuator to settle
            
            # Read response
            signal = self.detector.read_signal(duration=sampling_duration)
            mean_response = np.mean(signal)
            responses.append(mean_response)
            
            # Store calibration point
            strain = voltage * self.piezo.strain_sensitivity
            self.calibration_data.append({
                'voltage': voltage,
                'strain': strain,
                'response': mean_response,
            })
        
        print("\nCalibration complete!")
        
        # Fit calibration curve (linear response expected)
        voltages_array = np.array(voltages)
        responses_array = np.array(responses)
        
        # Linear fit: response = a * voltage + b
        coeffs = np.polyfit(voltages_array, responses_array, 1)
        sensitivity = coeffs[0]  # Response per volt
        
        calibration_result = {
            'voltages': voltages.tolist(),
            'responses': responses.tolist(),
            'sensitivity': float(sensitivity),
            'offset': float(coeffs[1]),
            'strain_sensitivity': self.piezo.strain_sensitivity,
        }
        
        return calibration_result
    
    def save_calibration(self, filepath: str):
        """Save calibration data to file."""
        import json
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.calibration_data, f, indent=2)
        
        print(f"Calibration data saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration data from file."""
        import json
        
        with open(filepath, 'r') as f:
            self.calibration_data = json.load(f)
        
        print(f"Calibration data loaded from {filepath}")
