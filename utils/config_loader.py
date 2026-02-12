"""Configuration loader utility."""
import yaml
import os
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
