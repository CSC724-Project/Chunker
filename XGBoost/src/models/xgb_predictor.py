import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class XGBChunkPredictor:
    def __init__(self, preprocessor=None):
        self.model = None
        self.preprocessor = preprocessor
        self.best_model = None
        self.feature_importance = None
        
        # Create plots directory
        self.plots_dir = Path("results/plots/xgboost")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def train_fold(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model on a single fold."""
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
                'seed': 42,
                'scale_pos_weight': 1  # Will be updated based on class distribution
            }
        elif 'learning_rate' in params:
            # Convert learning_rate to eta if present
            params['eta'] = params.pop('learning_rate')
        
        # Calculate class weights if needed
        if self.preprocessor is not None:
            class_stats = self.preprocessor.get_target_stats()['class_balance']
            if 0 in class_stats and 1 in class_stats:
                params['scale_pos_weight'] = class_stats[0] / class_stats[1]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model with early stopping
        num_boost_round = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            evals=[(dtrain, 'train'), (dval, 'val')],
            verbose_eval=100  # Print evaluation every 100 rounds
        )
        
        # Store feature importance
        self.feature_importance = model.get_score(importance_type='gain')
        
        # Plot feature importance
        self._plot_feature_importance()
        
        # Calculate metrics
        metrics = self._calculate_metrics(model, X_val, y_val)
        
        # Plot predictions
        self._plot_predictions(model, X_val, y_val)
        
        # Store best model
        self.best_model = model
        
        return metrics['logloss'], metrics['accuracy'], metrics
    
    def _calculate_metrics(self, model, X_val, y_val):
        """Calculate classification metrics."""
        # Get predictions
        dval = xgb.DMatrix(X_val)
        val_pred_proba = model.predict(dval)
        val_pred = (val_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, val_pred)
        precision = precision_score(y_val, val_pred)
        recall = recall_score(y_val, val_pred)
        f1 = f1_score(y_val, val_pred)
        auc_roc = roc_auc_score(y_val, val_pred_proba)
        conf_matrix = confusion_matrix(y_val, val_pred)
        
        # Calculate logloss
        epsilon = 1e-15
        val_pred_proba = np.clip(val_pred_proba, epsilon, 1 - epsilon)
        logloss = -np.mean(y_val * np.log(val_pred_proba) + (1 - y_val) * np.log(1 - val_pred_proba))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'logloss': logloss,
            'confusion_matrix': conf_matrix
        }
        
        # Log metrics
        logging.info("\nValidation Metrics:")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"AUC-ROC: {auc_roc:.4f}")
        logging.info(f"Log Loss: {logloss:.4f}")
        logging.info("\nConfusion Matrix:")
        logging.info(f"{conf_matrix}")
        
        return metrics
    
    def _plot_predictions(self, model, X_val, y_val):
        """Plot ROC curve and confusion matrix."""
        dval = xgb.DMatrix(X_val)
        pred_proba = model.predict(dval)
        pred = (pred_proba > 0.5).astype(int)
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_val, pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig(self.plots_dir / 'roc_curve.png')
        plt.close()
        
        # Plot confusion matrix
        conf_matrix = confusion_matrix(y_val, pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(self.plots_dir / 'confusion_matrix.png')
        plt.close()
        
        # Plot probability distribution
        plt.figure(figsize=(10, 6))
        for i in range(2):
            mask = y_val == i
            plt.hist(pred_proba[mask], bins=50, alpha=0.5, label=f'Class {i}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution by Class')
        plt.legend()
        plt.savefig(self.plots_dir / 'probability_distribution.png')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot feature importance scores."""
        if self.feature_importance is None:
            return
        
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['Importance'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance (gain)')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png')
        plt.close()
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.best_model is None:
            raise ValueError("Model hasn't been trained yet")
        
        dtest = xgb.DMatrix(X)
        return self.best_model.predict(dtest) 