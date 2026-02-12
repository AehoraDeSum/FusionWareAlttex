"""Real-time analyzer for gravitational wave detection."""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from collections import deque
import threading
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2_ai_model.model import create_model
from stage3_detector.detector_interface import DetectorInterface
from stage4_integration.ligo_client import LIGOClient
from utils.config_loader import load_config


class RealTimeAnalyzer:
    """
    Real-time analyzer that processes detector data using AI model
    and cross-references with LIGO/Virgo data.
    """
    
    def __init__(
        self,
        model_path: str,
        detector: Optional[DetectorInterface] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize real-time analyzer.
        
        Parameters:
        -----------
        model_path : str
            Path to trained model checkpoint
        detector : Optional[DetectorInterface]
            Detector interface (None for simulation mode)
        config : Optional[Dict]
            Configuration dictionary
        """
        if config is None:
            from utils.config_loader import load_config
            config = load_config()
        
        self.config = config
        self.detector = detector
        self.realtime_config = config.get("realtime", {})
        
        # Load model
        print("Loading AI model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        model = create_model(config)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model.to(device)
        print(f"Model loaded on {device}")
        
        # LIGO client
        ligo_config = self.realtime_config.get("ligo_api", {})
        if ligo_config.get("enabled", True):
            self.ligo_client = LIGOClient(
                base_url=ligo_config.get("base_url", "https://gracedb.ligo.org/api/")
            )
        else:
            self.ligo_client = None
        
        # Data buffers
        self.sampling_rate = self.realtime_config.get("sampling_rate", 1000)
        self.window_size = self.realtime_config.get("window_size", 1024)
        self.data_buffer = deque(maxlen=self.window_size)
        
        # Detection history
        self.detections = []
        
        # Threading
        self.running = False
        self.analysis_thread = None
        self.ligo_thread = None
    
    def preprocess_signal(self, signal: np.ndarray) -> torch.Tensor:
        """
        Preprocess signal for model input.
        
        Parameters:
        -----------
        signal : np.ndarray
            Raw signal data
            
        Returns:
        --------
        torch.Tensor
            Preprocessed tensor
        """
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Pad or truncate to window size
        if len(signal) < self.window_size:
            signal = np.pad(signal, (0, self.window_size - len(signal)))
        elif len(signal) > self.window_size:
            signal = signal[:self.window_size]
        
        # Convert to tensor: [1, 1, window_size]
        tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def analyze_signal(self, signal: np.ndarray) -> Dict:
        """
        Analyze signal using AI model.
        
        Parameters:
        -----------
        signal : np.ndarray
            Signal data
            
        Returns:
        --------
        Dict
            Analysis results with prediction and confidence
        """
        # Preprocess
        input_tensor = self.preprocess_signal(signal)
        
        # Model prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        result = {
            'has_signal': predicted_class == 1,
            'confidence': confidence,
            'probabilities': {
                'noise': probabilities[0, 0].item(),
                'signal': probabilities[0, 1].item(),
            },
            'timestamp': datetime.utcnow(),
        }
        
        return result
    
    def process_detection(self, analysis_result: Dict):
        """
        Process a detection, including LIGO cross-referencing.
        
        Parameters:
        -----------
        analysis_result : Dict
            Analysis result from model
        """
        if analysis_result['has_signal'] and analysis_result['confidence'] > 0.7:
            detection_time = analysis_result['timestamp']
            
            # Cross-reference with LIGO/Virgo
            ligo_match = None
            if self.ligo_client:
                ligo_match = self.ligo_client.cross_reference(
                    detection_time,
                    time_tolerance=timedelta(seconds=10),
                )
            
            detection = {
                'timestamp': detection_time,
                'confidence': analysis_result['confidence'],
                'probabilities': analysis_result['probabilities'],
                'ligo_match': ligo_match,
            }
            
            self.detections.append(detection)
            
            # Print detection
            print(f"\n{'='*60}")
            print(f"GRAVITATIONAL WAVE DETECTION")
            print(f"Time: {detection_time}")
            print(f"Confidence: {analysis_result['confidence']:.2%}")
            print(f"Signal probability: {analysis_result['probabilities']['signal']:.2%}")
            
            if ligo_match:
                print(f"✓ MATCHED with LIGO/Virgo event: {ligo_match.get('graceid')}")
                print(f"  False alarm rate: {ligo_match.get('far', 'N/A')}")
            else:
                print("⚠ No matching LIGO/Virgo event found")
            print(f"{'='*60}\n")
    
    def run_analysis_loop(self):
        """Main analysis loop running in separate thread."""
        update_interval = 1.0 / self.realtime_config.get("update_interval", 1.0)
        
        while self.running:
            try:
                # Read signal from detector
                if self.detector and self.detector.is_connected:
                    signal = self.detector.read_signal(
                        duration=update_interval,
                        sampling_rate=self.sampling_rate,
                    )
                else:
                    # Simulation mode: generate random signal
                    signal = np.random.normal(0, 1, int(self.sampling_rate * update_interval))
                
                # Add to buffer
                self.data_buffer.extend(signal)
                
                # Analyze when buffer is full
                if len(self.data_buffer) >= self.window_size:
                    signal_array = np.array(list(self.data_buffer))
                    analysis_result = self.analyze_signal(signal_array)
                    self.process_detection(analysis_result)
                
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"Error in analysis loop: {e}")
                time.sleep(1.0)
    
    def run_ligo_monitor(self):
        """Monitor LIGO/Virgo events in separate thread."""
        check_interval = self.realtime_config.get("ligo_api", {}).get("check_interval", 60)
        
        while self.running:
            try:
                if self.ligo_client:
                    new_events = self.ligo_client.check_new_events(
                        check_interval=check_interval,
                    )
                    
                    if new_events:
                        print(f"\nNew LIGO/Virgo events detected: {len(new_events)}")
                        for event in new_events:
                            print(f"  - {event.get('graceid')}: FAR={event.get('far', 'N/A')}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Error in LIGO monitor: {e}")
                time.sleep(check_interval)
    
    def start(self):
        """Start real-time analysis."""
        if self.running:
            return
        
        self.running = True
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(target=self.run_analysis_loop, daemon=True)
        self.analysis_thread.start()
        
        # Start LIGO monitor thread
        if self.ligo_client:
            self.ligo_thread = threading.Thread(target=self.run_ligo_monitor, daemon=True)
            self.ligo_thread.start()
        
        print("Real-time analysis started")
    
    def stop(self):
        """Stop real-time analysis."""
        self.running = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        
        if self.ligo_thread:
            self.ligo_thread.join(timeout=5)
        
        print("Real-time analysis stopped")
    
    def get_detections(self) -> List[Dict]:
        """Get list of all detections."""
        return self.detections.copy()
