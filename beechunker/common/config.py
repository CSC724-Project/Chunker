import os
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

with open (os.path.join(os.path.dirname(__file__), 'default_config.json'), 'r') as f:
    DEFAULT_CONFIG = json.load(f)

class Config:
    """
    Configuration class for BeeChunker.
    """
    def __init__(self, config_path=None):
        """
        Initialize the configuration class.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path or Path(__file__).parent / 'default_config.json'
        self.config = DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """Load config from the config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update the default config with loaded config
                    for section, values in loaded_config.items():
                        if section in self.config:
                            self.config[section].update(values)
                        else:
                            self.config[section] = values
                logging.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logging.error(f"Error loading configuration: {e}")
    
    def save(self):
        """Save the current configuration to the config file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            logging.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def get(self, section, key=None):
        """Get configuration value."""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def ensure_paths_exist(self):
        """Ensure all configured paths exist."""
        paths = [
            self.get("monitor", "db_path"),
            os.path.dirname(self.get("monitor", "log_path")),
            os.path.dirname(self.get("optimizer", "log_path")),
            self.get("ml", "models_dir"),
            os.path.dirname(self.get("ml", "log_path"))
        ]
        
        for path in paths:
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)

# Create a global config instance
config = Config()