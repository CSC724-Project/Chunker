import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from beechunker.ml.xgboost_model import BeeChunkerXGBoost
from beechunker.common.config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_training_data(num_samples=1000):
    """Create mock training data similar to BeeGFS logs."""
    np.random.seed(42)
    
    # Generate random data
    data = {
        'file_path': [f"/data/file_{i}.dat" for i in range(num_samples)],
        'file_size': np.random.randint(1024*1024, 1024*1024*1024, num_samples),  # 1MB to 1GB
        'chunk_size': np.random.choice([128*1024, 256*1024, 512*1024, 1024*1024], num_samples),  # 128KB to 1MB
        'read_count': np.random.randint(10, 1000, num_samples),
        'write_count': np.random.randint(5, 500, num_samples),
        'avg_read_size': np.random.randint(4096, 1024*1024, num_samples),
        'avg_write_size': np.random.randint(4096, 1024*1024, num_samples),
        'max_read_size': np.random.randint(8192, 2*1024*1024, num_samples),
        'max_write_size': np.random.randint(8192, 2*1024*1024, num_samples),
        'throughput_mbps': np.random.uniform(50, 500, num_samples)
    }
    
    # Add timestamps
    base_time = datetime.now() - timedelta(days=7)
    data['timestamp'] = [base_time + timedelta(minutes=i) for i in range(num_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure max sizes are larger than avg sizes
    df['max_read_size'] = df[['max_read_size', 'avg_read_size']].max(axis=1)
    df['max_write_size'] = df[['max_write_size', 'avg_write_size']].max(axis=1)
    
    return df

def create_mock_prediction_data(num_files=5):
    """Create mock data for prediction testing."""
    np.random.seed(43)
    
    # Generate random data for a few files
    data = {
        'file_path': [f"/data/test_file_{i}.dat" for i in range(num_files)],
        'file_size': np.random.randint(1024*1024, 1024*1024*1024, num_files),
        'chunk_size': np.random.choice([128*1024, 256*1024, 512*1024, 1024*1024], num_files),
        'read_count': np.random.randint(10, 1000, num_files),
        'write_count': np.random.randint(5, 500, num_files),
        'avg_read_size': np.random.randint(4096, 1024*1024, num_files),
        'avg_write_size': np.random.randint(4096, 1024*1024, num_files),
        'max_read_size': np.random.randint(8192, 2*1024*1024, num_files),
        'max_write_size': np.random.randint(8192, 2*1024*1024, num_files),
        'throughput_mbps': np.random.uniform(50, 500, num_files)
    }
    
    # Add timestamps (same format as training data)
    base_time = datetime.now()
    data['timestamp'] = [base_time + timedelta(minutes=i) for i in range(num_files)]
    
    df = pd.DataFrame(data)
    df['max_read_size'] = df[['max_read_size', 'avg_read_size']].max(axis=1)
    df['max_write_size'] = df[['max_write_size', 'avg_write_size']].max(axis=1)
    
    return df

def test_xgboost_pipeline():
    """Test the complete XGBoost pipeline."""
    logger.info("Starting XGBoost pipeline test")
    
    # Initialize model
    model = BeeChunkerXGBoost()
    
    # Create mock training data
    train_df = create_mock_training_data()
    logger.info(f"Created {len(train_df)} training samples")
    
    # Ensure models directory exists in config
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Set required config values
    config.set("ml", "models_dir", models_dir)
    
    # Save mock data as CSV (simulating log file)
    log_path = os.path.join(models_dir, "mock_training_data.csv")
    train_df.to_csv(log_path, index=False)
    
    # Update config to use mock data
    config.set("ml", "log_path", log_path)
    config.set("ml", "min_training_samples", 50)  # Set minimum training samples
    
    # Train model
    logger.info("Training model...")
    success = model.train()
    if not success:
        logger.error("Failed to train model")
        return False
    
    # Create test data
    test_df = create_mock_prediction_data()
    logger.info(f"Created {len(test_df)} test samples")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(test_df)
    
    if predictions is None:
        logger.error("Failed to make predictions")
        return False
    
    # Print results
    logger.info("\nPrediction Results:")
    for _, row in predictions.iterrows():
        logger.info(f"\nFile: {row['file_path']}")
        logger.info(f"Current chunk size: {row['current_chunk_size']}KB")
        logger.info(f"Predicted optimal chunk size: {row['predicted_chunk_size']}KB")
        logger.info(f"Current probability: {row['current_probability']:.3f}")
        logger.info(f"Predicted probability: {row['predicted_probability']:.3f}")
        logger.info(f"Needs optimization: {row['needs_optimization']}")
        if row['needs_optimization']:
            logger.info(f"Expected improvement: {row['optimization_gain']:.3f}")
    
    return True

if __name__ == "__main__":
    test_xgboost_pipeline() 