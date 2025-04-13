"""Logging setup for BeeChunker components.
This module sets up logging for different components of the BeeChunker application."""
import os
import logging
from .config import config

def setup_logging(component):
    """Set up logging for a component."""
    log_path = config.get(component, "log_path")
    
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    else:
        # Configure logging to console only
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    logger = logging.getLogger(f"chunkflow.{component}")
    logger.info(f"Initialized logging for {component}")
    return logger