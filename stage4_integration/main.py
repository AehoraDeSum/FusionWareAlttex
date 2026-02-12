"""Main script for real-time gravitational wave detection system."""

import argparse
import signal
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage4_integration.realtime_analyzer import RealTimeAnalyzer
from stage4_integration.visualizer import RealTimeVisualizer
from stage3_detector.detector_interface import DetectorInterface
from utils.config_loader import load_config


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down...")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time gravitational wave detection system"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Serial device path (None for simulation mode)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )
    parser.add_argument(
        "--no-ligo",
        action="store_true",
        help="Disable LIGO/Virgo cross-referencing",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override config
    if args.no_ligo:
        config.setdefault("realtime", {}).setdefault("ligo_api", {})["enabled"] = False
    
    # Initialize detector
    detector = None
    if args.device:
        detector = DetectorInterface(
            serial_port=args.device,
            baud_rate=config.get("detector", {}).get("baud_rate", 115200),
        )
        if not detector.connect():
            print("Warning: Failed to connect to detector. Using simulation mode.")
            detector = None
    else:
        print("Running in simulation mode (no hardware connection)")
    
    # Initialize analyzer
    print("Initializing real-time analyzer...")
    analyzer = RealTimeAnalyzer(
        model_path=args.model,
        detector=detector,
        config=config,
    )
    
    # Initialize visualizer
    visualizer = None
    if not args.no_viz:
        realtime_config = config.get("realtime", {})
        viz_config = realtime_config.get("visualization", {})
        
        visualizer = RealTimeVisualizer(
            analyzer=analyzer,
            update_rate=viz_config.get("update_rate", 10),
            save_plots=viz_config.get("save_plots", True),
            output_dir=viz_config.get("output_dir", "data/plots"),
        )
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start analyzer
        analyzer.start()
        
        # Start visualizer
        if visualizer:
            visualizer.start()
        
        print("\n" + "="*60)
        print("Gravitational Wave Detection System - RUNNING")
        print("="*60)
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Keep running and process matplotlib events
        import time
        try:
            while True:
                time.sleep(0.1)
                # Process matplotlib events to keep window responsive
                if visualizer and visualizer.fig:
                    visualizer.fig.canvas.flush_events()
        except KeyboardInterrupt:
            pass
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop analyzer
        analyzer.stop()
        
        # Stop visualizer
        if visualizer:
            visualizer.stop()
        
        # Disconnect detector
        if detector:
            detector.disconnect()
        
        print("System stopped.")


if __name__ == "__main__":
    main()
