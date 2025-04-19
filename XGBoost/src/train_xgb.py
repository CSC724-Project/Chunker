import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import KFold
from pathlib import Path
import json
import os
import joblib

from src.data.preprocessor import DataPreprocessor
from src.models.xgb_predictor import XGBChunkPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_xgb_model(data_path, n_splits=5, params=None, save_dir="models"):
    """
    Train the XGBoost optimal throughput predictor using k-fold cross validation.
    
    Args:
        data_path (str): Path to the CSV data file
        n_splits (int): Number of folds for cross-validation
        params (dict): XGBoost parameters
        save_dir (str): Directory to save the model and preprocessor
    """
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor()
    X, y = preprocessor.load_data(data_path)
    
    # Initialize K-fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize lists to store metrics
    logloss_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    
    # Default XGBoost parameters
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error', 'auc'],
            'eta': 0.01,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'seed': 42
        }
    
    # Initialize trainer with preprocessor
    trainer = XGBChunkPredictor(preprocessor=preprocessor)
    
    # Perform k-fold training
    best_fold_score = float('inf')
    best_fold = None
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logging.info(f"\nTraining Fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        logloss, accuracy, metrics = trainer.train_fold(
            X_train, y_train,
            X_val, y_val,
            params=params
        )
        
        # Store metrics
        logloss_scores.append(logloss)
        accuracy_scores.append(accuracy)
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        auc_scores.append(metrics['auc_roc'])
        
        # Keep track of best model
        if logloss < best_fold_score:
            best_fold_score = logloss
            best_fold = fold
            best_trainer = trainer
        
        logging.info(f"Fold {fold} - Logloss: {logloss:.4f}, Accuracy: {accuracy:.4f}")
        logging.info(f"         Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        logging.info(f"         F1: {metrics['f1']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Calculate and log average metrics
    avg_logloss = np.mean(logloss_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(auc_scores)
    
    std_logloss = np.std(logloss_scores)
    std_accuracy = np.std(accuracy_scores)
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_f1 = np.std(f1_scores)
    std_auc = np.std(auc_scores)
    
    logging.info("\nOverall Results:")
    logging.info(f"Average Logloss: {avg_logloss:.4f} (±{std_logloss:.4f})")
    logging.info(f"Average Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    logging.info(f"Average Precision: {avg_precision:.4f} (±{std_precision:.4f})")
    logging.info(f"Average Recall: {avg_recall:.4f} (±{std_recall:.4f})")
    logging.info(f"Average F1 Score: {avg_f1:.4f} (±{std_f1:.4f})")
    logging.info(f"Average AUC-ROC: {avg_auc:.4f} (±{std_auc:.4f})")
    
    # Save results
    results = {
        "metrics": {
            "logloss": {
                "mean": float(avg_logloss),
                "std": float(std_logloss),
                "folds": [float(score) for score in logloss_scores]
            },
            "accuracy": {
                "mean": float(avg_accuracy),
                "std": float(std_accuracy),
                "folds": [float(score) for score in accuracy_scores]
            },
            "precision": {
                "mean": float(avg_precision),
                "std": float(std_precision),
                "folds": [float(score) for score in precision_scores]
            },
            "recall": {
                "mean": float(avg_recall),
                "std": float(std_recall),
                "folds": [float(score) for score in recall_scores]
            },
            "f1": {
                "mean": float(avg_f1),
                "std": float(std_f1),
                "folds": [float(score) for score in f1_scores]
            },
            "auc_roc": {
                "mean": float(avg_auc),
                "std": float(std_auc),
                "folds": [float(score) for score in auc_scores]
            }
        },
        "parameters": {
            "n_splits": n_splits,
            "model_params": params,
            "features": preprocessor.get_feature_names(),
            "best_fold": best_fold
        }
    }
    
    # Create save directories
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save model, preprocessor and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_path / f"xgb_model_{timestamp}"
    model_path.mkdir(exist_ok=True)
    
    # Save the best model and preprocessor
    best_trainer.best_model.save_model(str(model_path / "model.json"))
    joblib.dump(preprocessor, model_path / "preprocessor.joblib")
    
    # Save model info
    with open(model_path / "model_info.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "metrics": results["metrics"],
            "parameters": results["parameters"]
        }, f, indent=4)
    
    # Save results to results directory
    with open(results_dir / "xgboost_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"\nModel saved to: {model_path}")
    return best_trainer

if __name__ == "__main__":
    # Get the script's directory
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / "src" / "data" / "train_OT65.csv"
    train_xgb_model(data_path) 