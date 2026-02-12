"""Calibration script for prototype detector."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage3_detector.detector_interface import DetectorInterface, DetectorCalibrator
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Calibrate prototype detector")
    parser.add_argument(
        "--device",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial device path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/calibration.json",
        help="Output file for calibration data",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of calibration steps",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    detector_config = config.get("detector", {})
    piezo_config = detector_config.get("piezoelectric", {})
    
    # Initialize detector
    detector = DetectorInterface(
        serial_port=args.device,
        baud_rate=detector_config.get("baud_rate", 115200),
    )
    
    # Connect
    if not detector.connect():
        print("Failed to connect to detector. Using simulation mode...")
        print("(In real deployment, ensure hardware is connected)")
        return
    
    try:
        # Initialize calibrator
        calibrator = DetectorCalibrator(
            detector,
            strain_sensitivity=piezo_config.get("strain_sensitivity", 1e-9),
        )
        
        # Perform calibration
        voltage_range = tuple(piezo_config.get("voltage_range", [0, 100]))
        calibration_result = calibrator.calibrate(
            voltage_range=voltage_range,
            num_steps=args.steps,
        )
        
        # Save calibration
        calibrator.save_calibration(args.output)
        
        print(f"\nCalibration Results:")
        print(f"  Sensitivity: {calibration_result['sensitivity']:.6e} response/volt")
        print(f"  Strain sensitivity: {calibration_result['strain_sensitivity']:.2e} strain/volt")
        print(f"  Offset: {calibration_result['offset']:.6e}")
        
    finally:
        detector.disconnect()


if __name__ == "__main__":
    main()
