import os
import sys
import logging
import pandas as pd
from beechunker.ml.xgboost_model import BeeChunkerXGBoost

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("__main__")

def main():
    """Test the XGBoost model with test.csv data"""
    try:
        # Create model instance
        model = BeeChunkerXGBoost()
        
        # Load test data
        log_path = 'data/logs/test.csv'
        if not os.path.exists(log_path):
            logger.error(f"Error: Test CSV file not found at {log_path}")
            return 1
        
        # Load CSV file
        df = pd.read_csv(log_path)
        logger.info(f"Loaded test data with {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Print first row for verification
        logger.info("\nFirst data row:")
        logger.info(df.iloc[0].to_string())
        
        # Train model (this will use k-fold cross-validation)
        logger.info("\nTraining model...")
        if not model.train(log_path):
            logger.error("Failed to train model")
            return 1
        
        # Make predictions
        logger.info("\nMaking predictions...")
        results = model.predict(df)
        
        # Print results
        if results is not None:
            logger.info("\nPrediction results:")
            logger.info(f"Total files: {len(results)}")
            logger.info(f"Files needing optimization: {results['needs_optimization'].sum()}")
            
            # Print optimization details for files that need it
            needs_opt = results[results['needs_optimization'] == True]
            if len(needs_opt) > 0:
                logger.info("\nFiles needing optimization:")
                for _, row in needs_opt.iterrows():
                    logger.info(f"File: {row['file_path']}")
                    logger.info(f"  Current chunk size: {row['current_chunk_size']}KB (probability: {row['current_probability']:.3f})")
                    logger.info(f"  Optimal chunk size: {row['predicted_chunk_size']}KB (probability: {row['predicted_probability']:.3f})")
                    logger.info(f"  Optimization gain: {row['optimization_gain']:.3f}")
                    logger.info("")
            
            logger.info("Test completed successfully")
            return 0
        else:
            logger.error("Prediction returned None")
            return 1
    
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 