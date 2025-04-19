import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_dir):
    """
    Load the saved XGBoost model and preprocessor.
    
    Args:
        model_dir (str): Directory containing the saved model files
    
    Returns:
        tuple: (xgboost model, preprocessor, model info)
    """
    model_path = Path(model_dir)
    
    # Load XGBoost model
    model = xgb.Booster()
    model.load_model(str(model_path / "model.json"))
    
    # Load preprocessor
    preprocessor = joblib.load(model_path / "preprocessor.joblib")
    
    # Load model info
    with open(model_path / "model_info.json", "r") as f:
        model_info = json.load(f)
    
    return model, preprocessor, model_info

def predict(model, preprocessor, data_path, output_path=None):
    """
    Make predictions on new data using the saved model.
    
    Args:
        model: Loaded XGBoost model
        preprocessor: Loaded preprocessor
        data_path (str): Path to the input CSV file
        output_path (str, optional): Path to save predictions. If None, will use input path with '_predictions' suffix
    
    Returns:
        pandas.DataFrame: Original data with predictions added
    """
    # Load and preprocess data
    df = pd.read_csv(data_path)
    X, _ = preprocessor._extract_features(df)
    
    # Make predictions
    dtest = xgb.DMatrix(X)
    pred_proba = model.predict(dtest)
    predictions = (pred_proba > 0.5).astype(int)
    
    # Add predictions to dataframe
    df['predicted_OT'] = predictions
    df['predicted_OT_probability'] = pred_proba
    
    # Save predictions if output path is specified
    if output_path is None:
        output_path = str(Path(data_path).with_suffix('')) + '_predictions.csv'
    
    df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to: {output_path}")
    
    # Print prediction statistics
    logging.info("\nPrediction Statistics:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Predicted optimal (OT=1): {sum(predictions)} ({sum(predictions)/len(predictions)*100:.2f}%)")
    logging.info(f"Predicted suboptimal (OT=0): {len(predictions)-sum(predictions)} ({(1-sum(predictions)/len(predictions))*100:.2f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Make predictions using saved XGBoost model')
    parser.add_argument('model_dir', help='Directory containing the saved model files')
    parser.add_argument('data_path', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to save predictions (optional)')
    
    args = parser.parse_args()
    
    # Load model and preprocessor
    logging.info(f"Loading model from {args.model_dir}")
    model, preprocessor, model_info = load_model(args.model_dir)
    
    # Log model info
    logging.info("\nModel Information:")
    logging.info(f"Training timestamp: {model_info['timestamp']}")
    logging.info(f"Best fold metrics:")
    for metric, values in model_info['metrics'].items():
        logging.info(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")
    
    # Make predictions
    logging.info(f"\nMaking predictions on {args.data_path}")
    predictions = predict(model, preprocessor, args.data_path, args.output)
    
    return predictions

if __name__ == "__main__":
    main() 