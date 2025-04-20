import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from beechunker.common.config import config
from beechunker.common.beechunker_logging import setup_logging

logger = setup_logging("xgboost_model")

class BeeChunkerXGBoost:
    """XGBoost model for chunk size optimization."""
    
    def __init__(self):
        """Initialize the XGBoost model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.models_dir = config.get("ml", "models_dir")
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _calculate_io_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate I/O efficiency metric for each row."""
        return (df['throughput_mbps'] * 1024 * 1024) / (df['file_size'] * (df['read_count'] + df['write_count']))
    
    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training or prediction."""
        # Calculate I/O efficiency
        df['io_efficiency'] = self._calculate_io_efficiency(df)
        
        # Select base features
        feature_cols = [
            'file_size', 'chunk_size', 'read_count', 'write_count',
            'avg_read_size', 'avg_write_size', 'max_read_size', 'max_write_size',
            'throughput_mbps', 'io_efficiency'
        ]
        
        # Add time-based features if available
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            feature_cols.extend(['hour', 'day_of_week'])
        
        X = df[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if self.scaler is None else self.scaler.transform(X)
        
        return X_scaled, feature_cols
    
    def _label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label data based on I/O efficiency and throughput using 65th percentile threshold.
        This is specific to BeeGFS logs where we need to determine optimal chunk sizes.
        """
        try:
            # Calculate I/O efficiency
            df['io_efficiency'] = self._calculate_io_efficiency(df)
            
            # Get 65th percentile thresholds (based on empirical analysis)
            io_threshold = df['io_efficiency'].quantile(0.65)
            throughput_threshold = df['throughput_mbps'].quantile(0.65)
            
            # Label as optimal (1) if both metrics are above their thresholds
            df['is_optimal'] = ((df['io_efficiency'] >= io_threshold) & 
                              (df['throughput_mbps'] >= throughput_threshold)).astype(int)
            
            # Log labeling statistics
            optimal_count = df['is_optimal'].sum()
            total_count = len(df)
            logger.info(f"Data labeling complete: {optimal_count}/{total_count} "
                      f"({optimal_count/total_count*100:.2f}%) samples labeled as optimal")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data labeling: {e}")
            raise
    
    def train(self) -> bool:
        """Train the XGBoost model on the log data using k-fold cross validation."""
        try:
            # Load training data from logs
            log_path = config.get("ml", "log_path")
            if not os.path.exists(log_path):
                logger.error(f"Training data not found at {log_path}")
                return False
            
            # Load and clean data
            df = pd.read_csv(log_path)
            if len(df) < config.get("ml", "min_training_samples", 100):
                logger.warning(f"Not enough samples for training. Need at least "
                            f"{config.get('ml', 'min_training_samples', 100)} samples.")
                return False
            
            # Clean data
            df = df.dropna(subset=['file_size', 'chunk_size', 'read_count', 'write_count', 
                                 'throughput_mbps'])
            
            # Label the data based on I/O efficiency and throughput
            try:
                df = self._label_data(df)
            except Exception as e:
                logger.error(f"Failed to label data: {e}")
                return False
            
            # Prepare features
            X, feature_names = self._prepare_features(df)
            y = df['is_optimal']
            
            # Initialize K-fold
            n_splits = 5
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Initialize metrics storage
            metrics = {
                'logloss': [], 'accuracy': [], 'precision': [],
                'recall': [], 'f1': [], 'auc': []
            }
            
            # XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'error', 'auc'],
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1
            }
            
            # Perform k-fold training
            best_model = None
            best_score = float('inf')
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
                
                # Train model
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=100,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
                
                # Evaluate model
                val_preds = model.predict(dval)
                fold_metrics = self._calculate_metrics(y_val, val_preds)
                
                # Store metrics
                for metric, value in fold_metrics.items():
                    metrics[metric].append(value)
                
                # Update best model
                if fold_metrics['logloss'] < best_score:
                    best_score = fold_metrics['logloss']
                    best_model = model
                
                logger.info(f"Fold {fold} - Logloss: {fold_metrics['logloss']:.4f}, "
                          f"AUC: {fold_metrics['auc']:.4f}")
            
            # Save best model and components
            self.model = best_model
            self.feature_names = feature_names
            
            model_path = os.path.join(self.models_dir, "xgboost_model.json")
            scaler_path = os.path.join(self.models_dir, "xgboost_scaler.joblib")
            
            self.model.save_model(model_path)
            dump(self.scaler, scaler_path)
            
            # Save model info
            model_info = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'metrics': {
                    name: {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    } for name, values in metrics.items()
                },
                'parameters': params,
                'features': feature_names,
                'labeling_thresholds': {
                    'io_efficiency': float(io_threshold),
                    'throughput': float(throughput_threshold)
                }
            }
            
            with open(os.path.join(self.models_dir, "xgboost_model_info.json"), "w") as f:
                import json
                json.dump(model_info, f, indent=4)
            
            # Set last training time
            self.set_last_training_time()
            
            logger.info("XGBoost model trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict optimal chunk size probabilities for input data."""
        try:
            if self.model is None:
                if not self.load():
                    raise RuntimeError("Model not loaded")
            
            # Prepare features
            X, _ = self._prepare_features(df)
            
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            
            # Get probabilities
            probabilities = self.model.predict(dtest)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'file_path': df['file_path'],
                'chunk_size': df['chunk_size'],
                'probability': probabilities,
                'is_optimal': (probabilities >= 0.5).astype(int)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def load(self) -> bool:
        """Load the trained model and components."""
        try:
            model_path = os.path.join(self.models_dir, "xgboost_model.json")
            scaler_path = os.path.join(self.models_dir, "xgboost_scaler.joblib")
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path]):
                logger.error("Model files not found")
                return False
            
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.scaler = load(scaler_path)
            
            # Load feature names from model info
            try:
                with open(os.path.join(self.models_dir, "xgboost_model_info.json"), "r") as f:
                    model_info = json.load(f)
                    self.feature_names = model_info['features']
            except:
                logger.warning("Could not load feature names from model info")
            
            logger.info("XGBoost model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _calculate_metrics(self, y_true, y_pred) -> dict:
        """Calculate various metrics for model evaluation."""
        from sklearn.metrics import log_loss, accuracy_score, precision_score
        from sklearn.metrics import recall_score, f1_score, roc_auc_score
        
        return {
            'logloss': log_loss(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred.round()),
            'precision': precision_score(y_true, y_pred.round()),
            'recall': recall_score(y_true, y_pred.round()),
            'f1': f1_score(y_true, y_pred.round()),
            'auc': roc_auc_score(y_true, y_pred)
        }
    
    def set_last_training_time(self):
        """Set the last training time."""
        with open(os.path.join(self.models_dir, "xgboost_last_training_time.txt"), "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def get_last_training_time(self):
        """Get the last training time."""
        try:
            with open(os.path.join(self.models_dir, "xgboost_last_training_time.txt"), "r") as f:
                last_training_time = f.read().strip()
            return datetime.strptime(last_training_time, "%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            logger.warning("Last training time file not found. Returning None.")
            return None
        except Exception as e:
            logger.error(f"Error reading last training time: {e}")
            return None
    
    def get_new_data_count(self, last_training_time):
        """Get the number of new data points since the last training time."""
        try:
            df = pd.read_csv(config.get("ml", "log_path"))
            new_data_count = len(df[df['timestamp'] > last_training_time])
            return new_data_count
        except Exception as e:
            logger.error(f"Error getting new data count: {e}")
            return 0 