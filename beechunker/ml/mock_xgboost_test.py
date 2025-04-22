import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import logging
from beechunker.ml.xgboost_model import BeeChunkerXGBoost
from beechunker.common.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(filepath):
    """Load training data from the test_results28k_filtered.csv file."""
    logger.info(f"Loading training data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Apply column mapping to match the expected column names in XGBoost model
    # column_mapping = {
    #     'file_size_KB': 'file_size',
    #     'chunk_size_KB': 'chunk_size',
    #     'read_ops': 'read_count',
    #     'write_ops': 'write_count',
    #     'avg_read_KB': 'avg_read_size',
    #     'avg_write_KB': 'avg_write_size',
    #     'max_read_KB': 'max_read_size',
    #     'max_write_KB': 'max_write_size',
    #     'throughput_KBps': 'throughput_mbps'
    # }
    
    # Rename columns according to mapping
    # df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Convert KB values to bytes for consistency with the XGBoost model expectations
    # if 'file_size' in df.columns:
    #     df['file_size'] = df['file_size'] * 1024  # Convert KB to bytes
    # if 'chunk_size' in df.columns:
    #     df['chunk_size'] = df['chunk_size'] * 1024  # Convert KB to bytes
    
    # # Filter out rows with errors or missing values
    # df = df.dropna(subset=['file_size', 'chunk_size', 'read_count', 'write_count', 'throughput_mbps'])
    # if 'error_message' in df.columns:
    #     df = df[df['error_message'].isna() | (df['error_message'] == '')]
    
    # Add timestamp if not present (using current time as placeholder)
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now()
    
    return df

def load_prediction_data(filepath):
    """Load prediction data from the combined_logs.csv file."""
    logger.info(f"Loading prediction data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Filter out rows with missing values
    # df = df.dropna(subset=['file_size', 'chunk_size', 'read_count', 'write_count', 'throughput_mbps'])
    
    # Add timestamp if not present
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now()
    
    # Take a sample of the data for testing purposes
    if len(df) > 20:
        df = df.sample(20, random_state=42)
    
    return df

def test_xgboost_pipeline():
    """Test the complete XGBoost pipeline using real data files."""
    logger.info("Starting XGBoost pipeline test")
    
    # Initialize model
    model = BeeChunkerXGBoost()
    
    # Ensure models directory exists in config
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Set required config values
    config.set("ml", "models_dir", models_dir)
    config.set("ml", "min_training_samples", 50)  # Set minimum training samples
    
    # Load real training data
    train_data_path = os.path.join("data", "logs", "test_results28k_filtered.csv")
    train_df = load_training_data(train_data_path)
    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Train model directly with DataFrame
    logger.info("Training model...")
    success = model.train(train_df)
    if not success:
        logger.error("Failed to train model")
        return False
    
    # Load real prediction data
    predict_data_path = os.path.join("data", "logs", "test.csv")
    test_df = load_prediction_data(predict_data_path)
    logger.info(f"Loaded {len(test_df)} test samples")
    
    # Make predictions one by one for each test sample
    logger.info("Making predictions...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        # Create a DataFrame with just this row
        sample_df = pd.DataFrame([row])
        
        # Get the optimal chunk size for this sample
        optimal_chunk_size = model.predict(sample_df)
        
        if optimal_chunk_size is None:
            logger.error(f"Failed to make prediction for sample {idx}")
            continue
            
        # Get current chunk size for comparison
        current_chunk_size = row['chunk_size_KB']
        
        # Store prediction details
        predictions.append({
            'file_path': row['file_path'],
            'file_size_KB': row['file_size_KB'],
            'current_chunk_size': current_chunk_size,
            'optimal_chunk_size': optimal_chunk_size,
            'needs_optimization': current_chunk_size != optimal_chunk_size
        })
    
    if not predictions:
        logger.error("Failed to make any valid predictions")
        return False
    
    # Print results
    logger.info("\nPrediction Results:")
    for result in predictions:
        logger.info(f"\nFile: {result['file_path']}")
        logger.info(f"File size: {result['file_size_KB']}KB")
        logger.info(f"Current chunk size: {result['current_chunk_size']}KB")
        logger.info(f"Optimal chunk size: {result['optimal_chunk_size']}KB")
        logger.info(f"Needs optimization: {result['needs_optimization']}")
    
    return True

if __name__ == "__main__":
    test_xgboost_pipeline() 