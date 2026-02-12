"""Real-time visualization for gravitational wave detection."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import threading
import time
import sys
import platform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class RealTimeVisualizer:
    """Real-time visualization of detector data and detections."""
    
    def __init__(
        self,
        analyzer,
        update_rate: float = 10.0,
        save_plots: bool = True,
        output_dir: str = "data/plots",
    ):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        analyzer : RealTimeAnalyzer
            Real-time analyzer instance
        update_rate : float
            Plot update rate in Hz
        save_plots : bool
            Whether to save plots to disk
        output_dir : str
            Directory to save plots
        """
        self.analyzer = analyzer
        self.update_rate = update_rate
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data buffers for plotting
        self.time_buffer = deque(maxlen=1000)
        self.signal_buffer = deque(maxlen=1000)
        self.confidence_buffer = deque(maxlen=1000)
        
        # Figure setup
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('Gravitational Wave Detector - Real-time Monitor', fontsize=14)
        
        # Subplot 1: Signal
        self.ax_signal = self.axes[0]
        self.line_signal, = self.ax_signal.plot([], [], 'b-', linewidth=1)
        self.ax_signal.set_ylabel('Signal Amplitude')
        self.ax_signal.set_title('Detector Signal')
        self.ax_signal.grid(True)
        
        # Subplot 2: Confidence
        self.ax_confidence = self.axes[1]
        self.line_confidence, = self.ax_confidence.plot([], [], 'g-', linewidth=2, label='Confidence')
        self.ax_confidence.axhline(y=0.7, color='r', linestyle='--', label='Threshold')
        self.ax_confidence.set_ylabel('Detection Confidence')
        self.ax_confidence.set_title('AI Model Confidence')
        self.ax_confidence.set_ylim(0, 1)
        self.ax_confidence.legend()
        self.ax_confidence.grid(True)
        
        # Subplot 3: Detection log
        self.ax_log = self.axes[2]
        self.ax_log.axis('off')
        self.text_log = self.ax_log.text(0.05, 0.95, '', transform=self.ax_log.transAxes,
                                         fontsize=10, verticalalignment='top',
                                         family='monospace')
        
        self.running = False
        self.animation = None
    
    def update_plot(self, frame):
        """Update plot with latest data."""
        # Get current signal from analyzer buffer
        if len(self.analyzer.data_buffer) > 0:
            signal = np.array(list(self.analyzer.data_buffer))
            time_points = np.arange(len(signal)) / self.analyzer.sampling_rate
            
            # Update signal plot
            self.line_signal.set_data(time_points, signal)
            self.ax_signal.set_xlim(0, max(time_points) if len(time_points) > 0 else 1)
            if len(signal) > 0:
                y_min, y_max = np.min(signal), np.max(signal)
                margin = (y_max - y_min) * 0.1
                self.ax_signal.set_ylim(y_min - margin, y_max + margin)
        
        # Update confidence from recent detections
        detections = self.analyzer.get_detections()
        if detections:
            recent = detections[-10:]  # Last 10 detections
            times = [(d['timestamp'] - datetime.utcnow()).total_seconds() for d in recent]
            confidences = [d['confidence'] for d in recent]
            
            if len(times) > 0:
                self.line_confidence.set_data(times, confidences)
                if len(times) > 1:
                    self.ax_confidence.set_xlim(min(times), max(times))
        
        # Update detection log
        log_text = "Recent Detections:\n" + "="*50 + "\n"
        for det in detections[-5:]:  # Last 5 detections
            time_str = det['timestamp'].strftime("%H:%M:%S")
            conf = det['confidence']
            ligo = "✓ LIGO" if det.get('ligo_match') else "⚠ Local"
            log_text += f"{time_str} | {conf:.2%} | {ligo}\n"
        
        self.text_log.set_text(log_text)
        
        return self.line_signal, self.line_confidence, self.text_log
    
    def start(self):
        """Start visualization."""
        if self.running:
            return
        
        self.running = True
        
        # Enable interactive mode
        plt.ion()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=1000.0 / self.update_rate,
            blit=False,
            cache_frame_data=False,  # Disable caching to avoid warnings
        )
        
        # Show window - this should make it visible
        self.fig.show()
        plt.show(block=False)
        
        # Force initial draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        print("Visualization started - Window should be visible")
        print(f"Backend: {plt.get_backend()}")
        print("If window is not visible, check your Dock or try Cmd+Tab to find it")
    
    def stop(self):
        """Stop visualization."""
        self.running = False
        
        if self.animation:
            self.animation.event_source.stop()
        
        if self.save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = self.output_dir / f"detection_{timestamp}.png"
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        
        plt.close(self.fig)
    
    def save_detection_plot(self, detection: dict, signal: np.ndarray):
        """Save a detailed plot for a specific detection."""
        if not self.save_plots:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Signal plot
        time_points = np.arange(len(signal)) / self.analyzer.sampling_rate
        axes[0].plot(time_points, signal, 'b-', linewidth=1)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Signal Amplitude')
        axes[0].set_title(f"Detection at {detection['timestamp']}")
        axes[0].grid(True)
        
        # Probability plot
        probs = detection['probabilities']
        axes[1].bar(['Noise', 'Signal'], [probs['noise'], probs['signal']], 
                   color=['red', 'green'], alpha=0.7)
        axes[1].set_ylabel('Probability')
        axes[1].set_title(f"Confidence: {detection['confidence']:.2%}")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, axis='y')
        
        if detection.get('ligo_match'):
            fig.suptitle(f"✓ MATCHED with LIGO event: {detection['ligo_match'].get('graceid')}",
                        fontsize=12, color='green')
        
        plt.tight_layout()
        
        timestamp = detection['timestamp'].strftime("%Y%m%d_%H%M%S")
        plot_path = self.output_dir / f"detection_{timestamp}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
