import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from beechunker.common.config import config
from beechunker.ml.xgboost_feature_engine import XGBoostFeatureEngine
from beechunker.common.beechunker_logging import setup_logging
from datetime import datetime

logger = setup_logging("xgboost_model")

class BeeChunkerXGBoost:
    """XGBoost model for chunk size optimization."""
    
    def __init__(self):
        """Initialize the XGBoost model."""
        self.model = None
        self.feature_engine = None
        self.models_dir = config.get("ml", "models_dir")
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _calculate_io_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """Calculate I/O efficiency metric for each row."""
        return (df['throughput_mbps'] * 1024 * 1024) / (df['file_size'] * (df['read_count'] + df['write_count']))
    
    def _label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label data based on I/O efficiency and throughput using 65th percentile threshold."""
        # Calculate I/O efficiency
        df['io_efficiency'] = self._calculate_io_efficiency(df)
        
        # Get 65th percentile thresholds
        io_threshold = df['io_efficiency'].quantile(0.65)
        throughput_threshold = df['throughput_mbps'].quantile(0.65)
        
        # Label as optimal (1) if both metrics are above their thresholds
        df['is_optimal'] = ((df['io_efficiency'] >= io_threshold) & 
                          (df['throughput_mbps'] >= throughput_threshold)).astype(int)
        
        return df
    
    def train(self) -> bool:
        """Train the XGBoost model on the log data."""
        try:
            # Load training data from logs
            log_path = config.get("ml", "log_path")
            if not os.path.exists(log_path):
                logger.error(f"Training data not found at {log_path}")
                return False
            
            df = pd.read_csv(log_path)
            
            # Clean data
            df = df.dropna()
            
            # Label the data based on I/O efficiency and throughput
            df = self._label_data(df)
            
            # Prepare features
            feature_cols = [
                'file_size', 'chunk_size', 'read_count', 'write_count',
                'avg_read_size', 'avg_write_size', 'max_read_size', 'max_write_size',
                'throughput_mbps', 'io_efficiency'
            ]
            
            X = df[feature_cols]
            y = df['is_optimal']
            
            # Train XGBoost model
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1
            }
            
            dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
            self.model = xgb.train(params, dtrain, num_boost_round=100)
            
            # Save model
            model_path = os.path.join(self.models_dir, "xgboost_model.joblib")
            dump(self.model, model_path)
            
            logger.info("XGBoost model trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict optimal chunk size probabilities for input data.
        
        Args:
            df: DataFrame with features including current chunk size
            
        Returns:
            DataFrame with predictions including:
                - probability: Probability of chunk size being optimal
                - is_optimal: Binary prediction (1 if probability >= 0.5)
        """
        try:
            if self.model is None:
                if not self.load():
                    raise RuntimeError("Model not loaded")
            
            # Prepare features
            feature_cols = [
                'file_size', 'chunk_size', 'read_count', 'write_count',
                'avg_read_size', 'avg_write_size', 'max_read_size', 'max_write_size',
                'throughput_mbps'
            ]
            
            # Calculate I/O efficiency
            df['io_efficiency'] = self._calculate_io_efficiency(df)
            feature_cols.append('io_efficiency')
            
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(df[feature_cols], feature_names=feature_cols)
            
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
        """Load the trained model from disk."""
        try:
            model_path = os.path.join(self.models_dir, "xgboost_model.joblib")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model = load(model_path)
            logger.info("XGBoost model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def set_last_training_time(self):
        with open(os.path.join(self.models_dir, "xgboost_last_training_time.txt"), "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def get_last_training_time(self):
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

    def _calculate_io_efficiency(self, df):
        """Calculate I/O efficiency score based on file size and I/O operations."""
        df['io_efficiency'] = (df['throughput_mbps'] * 1024 * 1024) / (df['file_size'] * (df['read_count'] + df['write_count']))
        return df

    def _label_data(self, df):
        """Label data based on I/O efficiency and throughput."""
        # Calculate I/O efficiency
        df = self._calculate_io_efficiency(df)
        
        # Get 65th percentile of I/O efficiency
        efficiency_threshold = df['io_efficiency'].quantile(0.65)
        
        # Label data points
        df['is_optimal'] = ((df['io_efficiency'] >= efficiency_threshold) & 
                          (df['throughput_mbps'] >= df['throughput_mbps'].quantile(0.65))).astype(int)
        
        return df

    def train(self) -> bool:
        """Train the XGBoost model using log data."""
        logger.info("Starting XGBoost training")
        
        try:
            # Load data from logs
            if not os.path.exists(self.log_path):
                logger.error(f"Log file not found: {self.log_path}")
                return False
                
            df = pd.read_csv(self.log_path)
            if len(df) < config.get("ml", "min_training_samples"):
                logger.warning(f"Not enough samples for training. Need at least {config.get('ml', 'min_training_samples')} samples.")
                return False
            
            # Clean and preprocess data
            df_clean = self.feature_engine.clean(self.log_path)
            if df_clean is None:
                logger.error("Failed to clean and preprocess data")
                return False
            
            # Label the data
            df_labeled = self._label_data(df_clean)
            
            # Prepare features for training
            X, feature_names = self.feature_engine.prepare_features(df_labeled, training=True)
            if X is None or feature_names is None:
                logger.error("Failed to prepare features")
                return False
                
            y = df_labeled['is_optimal']
            
            # Train XGBoost model
            dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1
            }
            
            self.model = xgb.train(params, dtrain, num_boost_round=100)
            
            # Save model and components
            dump(self.model, os.path.join(self.models_dir, "xgboost_model.joblib"))
            dump(self.feature_engine, os.path.join(self.models_dir, "xgboost_feature_engine.joblib"))
            
            # Set last training time
            self.set_last_training_time()
            
            logger.info("XGBoost model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during XGBoost training: {e}")
            return False

    def load(self) -> bool:
        """Load the trained XGBoost model and components."""
        try:
            model_path = os.path.join(self.models_dir, "xgboost_model.joblib")
            feature_engine_path = os.path.join(self.models_dir, "xgboost_feature_engine.joblib")
            
            if not all(os.path.exists(p) for p in [model_path, feature_engine_path]):
                logger.warning("XGBoost model files not found")
                return False
            
            self.model = load(model_path)
            self.feature_engine = load(feature_engine_path)
            
            logger.info("Successfully loaded XGBoost model")
            return True
            
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")
            return False

    def predict(self, df) -> pd.DataFrame:
        """Predict optimal chunk sizes for the given dataframe."""
        if self.model is None:
            if not self.load():
                logger.error("XGBoost model not loaded. Cannot predict.")
                return None
        
        try:
            # Clean and preprocess the input data
            df_clean = self.feature_engine._calculate_derived_features(df)
            
            # Prepare features for prediction
            X, _ = self.feature_engine.prepare_features(df_clean)
            if X is None:
                logger.error("Failed to prepare features for prediction")
                return None
            
            # Create DMatrix for prediction
            dmatrix = xgb.DMatrix(X)
            
            # Get probability of current chunk size being optimal
            is_optimal_prob = self.model.predict(dmatrix)
            
            # Initialize results list
            results = []
            
            # For each file, find the optimal chunk size
            for i, row in df_clean.iterrows():
                current_chunk_kb = row['chunk_size'] // 1024
                
                if is_optimal_prob[i] >= 0.5:
                    # Current chunk size is predicted to be optimal
                    optimal_chunk_size = current_chunk_kb
                else:
                    # Try different chunk sizes and find the best one
                    test_chunks = pd.DataFrame([row] * len(self.chunk_size_options))
                    for j, chunk_size in enumerate(self.chunk_size_options):
                        test_chunks.iloc[j, test_chunks.columns.get_loc('chunk_size')] = chunk_size * 1024
                    
                    # Calculate derived features for test chunks
                    test_chunks = self.feature_engine._calculate_derived_features(test_chunks)
                    
                    # Prepare features for test chunks
                    X_test, _ = self.feature_engine.prepare_features(test_chunks)
                    if X_test is None:
                        logger.error("Failed to prepare features for chunk size options")
                        continue
                        
                    dtest = xgb.DMatrix(X_test)
                    
                    # Get probabilities for each chunk size
                    chunk_probs = self.model.predict(dtest)
                    
                    # Select the chunk size with highest probability of being optimal
                    optimal_idx = np.argmax(chunk_probs)
                    optimal_chunk_size = self.chunk_size_options[optimal_idx]
                
                results.append({
                    'file_path': row['file_path'],
                    'file_size': row['file_size'],
                    'current_chunk_size': current_chunk_kb,
                    'predicted_chunk_size': optimal_chunk_size,
                    'current_to_predicted_ratio': current_chunk_kb / optimal_chunk_size if optimal_chunk_size > 0 else float('inf')
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error making XGBoost predictions: {e}")
            return None 