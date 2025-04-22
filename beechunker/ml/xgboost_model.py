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
    
    def _prepare_features(self, df: pd.DataFrame, training: bool = False) -> tuple:
        """
        Prepare features for training or prediction.
        Args:
            df: DataFrame with raw features
            training: Whether this is for training (True) or prediction (False)
        Returns:
            tuple: (scaled_features, feature_names)
        """
        try:
            # Calculate I/O efficiency
            # df['io_efficiency'] = self._calculate_io_efficiency(df)
            
            # Base features that are always required
            base_features = [
                'file_size_KB', 'chunk_size_KB', 'read_count', 'write_count',
                'avg_read_size_KB', 'avg_write_size_KB', 'max_read_size_KB', 'max_write_size_KB',
                'throughput_mbps'#, 'io_efficiency'
            ]
            
            # Add time-based features if we're training or if they were used in training
            if training:
                self.use_time_features = 'timestamp' in df.columns
                if self.use_time_features:
                    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                    base_features.extend(['hour', 'day_of_week'])
            else:
                # For prediction, check if we used time features during training
                if hasattr(self, 'use_time_features') and self.use_time_features:
                    if 'timestamp' not in df.columns:
                        # If time features were used in training but not available for prediction,
                        # add default values
                        df['hour'] = 12  # Middle of the day
                        df['day_of_week'] = 3  # Middle of the week
                    else:
                        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                    base_features.extend(['hour', 'day_of_week'])
            
            # Extract features
            X = df[base_features].copy()
            
            # Handle any missing values with median
            X = X.fillna(X.median())
            
            # Scale features
            if training:
                # Fit and transform for training data
                X_scaled = self.scaler.fit_transform(X)
            else:
                # Only transform for prediction data
                if not hasattr(self.scaler, 'mean_'):
                    raise RuntimeError("Scaler is not fitted. Need to train the model first.")
                X_scaled = self.scaler.transform(X)
            
            return X_scaled, base_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            logger.error(f"Available columns: {df.columns.tolist()}")
            raise
    
    def _label_data(self, df: pd.DataFrame) -> tuple:
        """
        Label data based on I/O efficiency and throughput using 65th percentile threshold.
        This is specific to BeeGFS logs where we need to determine optimal chunk sizes.
        
        Returns:
            tuple: (labeled_dataframe, thresholds_dict)
        """
        try:
            # Calculate I/O efficiency
            # df['io_efficiency'] = self._calculate_io_efficiency(df)
            
            # Get 65th percentile thresholds (based on empirical analysis)
            # io_threshold = df['io_efficiency'].quantile(0.65)
            throughput_threshold = df['throughput_mbps'].quantile(0.65)
            
            # Label as optimal (1) if both metrics are above their thresholds
            df['is_optimal'] = (#(df['io_efficiency'] >= io_threshold) & 
                              (df['throughput_mbps'] >= throughput_threshold)).astype(int)
            
            # Log labeling statistics
            optimal_count = df['is_optimal'].sum()
            total_count = len(df)
            logger.info(f"Data labeling complete: {optimal_count}/{total_count} "
                      f"({optimal_count/total_count*100:.2f}%) samples labeled as optimal")
            
            # Return both the labeled dataframe and the thresholds
            thresholds = {
                # 'io_efficiency': float(io_threshold),
                'throughput': float(throughput_threshold)
            }
            
            return df, thresholds
            
        except Exception as e:
            logger.error(f"Error in data labeling: {e}")
            raise
    
    def train(self, input_data) -> bool:
        """Train the XGBoost model on the log data using k-fold cross validation.
        
        Args:
            input_data: Either a file path (str) or a DataFrame with features and chunk sizes.
        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            logger.info("Starting XGBoost training")
            
            # If input_data is a DataFrame, check the sample count
            if isinstance(input_data, pd.DataFrame):
                df = input_data
                if len(df) < config.get("ml", "min_training_samples"):
                    logger.warning("Not enough samples for training. Need at least %d samples.", 
                                config.get("ml", "min_training_samples"))
                    return False
                
                # Save the DataFrame to a temporary file for processing
                temp_csv_path = os.path.join(self.models_dir, "temp_training_data.csv")
                df.to_csv(temp_csv_path, index=False)
                logger.info(f"Saved input DataFrame to temporary file: {temp_csv_path}")
                
                # Use the file path for further processing
                log_path = temp_csv_path
            else:
                # Assume input_data is a file path
                log_path = input_data
                
                # Check if the file exists
                if not os.path.exists(log_path):
                    logger.error(f"Input file not found: {log_path}")
                    return False
                
                # Check the sample count from the file
                try:
                    df_check = pd.read_csv(log_path)
                    if len(df_check) < config.get("ml", "min_training_samples"):
                        logger.warning("Not enough samples for training. Need at least %d samples.", 
                                    config.get("ml", "min_training_samples"))
                        return False
                except Exception as e:
                    logger.error(f"Error checking input file: {e}")
                    return False
            
            # Load and clean data
            df = pd.read_csv(log_path)
            
            # Clean data
            df = df.dropna(subset=['file_size', 'chunk_size', 'read_count', 'write_count', 
                                 'throughput_mbps'])
            
            # Label the data based on I/O efficiency and throughput
            try:
                df, thresholds = self._label_data(df)
            except Exception as e:
                logger.error(f"Failed to label data: {e}")
                return False
            
            # Prepare features
            X, feature_names = self._prepare_features(df, training=True)
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
                'labeling_thresholds': thresholds
            }
            
            with open(os.path.join(self.models_dir, "xgboost_model_info.json"), "w") as f:
                import json
                json.dump(model_info, f, indent=4)
            
            # Set last training time
            self.set_last_training_time()
            
            # Clean up temporary file if it was created
            if isinstance(input_data, pd.DataFrame) and os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                logger.info(f"Removed temporary file: {temp_csv_path}")
            
            logger.info("XGBoost model trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            # Clean up temporary file if there was an error
            if isinstance(input_data, pd.DataFrame):
                temp_csv_path = os.path.join(self.models_dir, "temp_training_data.csv")
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
                    logger.info(f"Removed temporary file: {temp_csv_path}")
            return False
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict optimal chunk sizes for the given dataframe.
        For each file, it will:
        1. Test the current chunk size
        2. If not optimal, test a range of chunk sizes to find the best one
        3. Return the optimal chunk size and related metrics
        """
        try:
            if self.model is None:
                if not self.load():
                    raise RuntimeError("Model not loaded")
            
            # Define chunk size options (in KB)
            chunk_size_options = [128, 256, 512, 1024, 2048, 4096, 8192]
            results = []
            
            # Process each file
            for idx, row in df.iterrows():
                current_chunk_kb = row['chunk_size'] // 1024
                file_features = row.copy()
                
                # First, evaluate current chunk size
                X_current, _ = self._prepare_features(pd.DataFrame([file_features]), training=False)
                dtest_current = xgb.DMatrix(X_current, feature_names=self.feature_names)
                current_prob = self.model.predict(dtest_current)[0]
                
                if current_prob >= 0.5:
                    # Current chunk size is predicted to be optimal
                    optimal_chunk_size = current_chunk_kb
                    optimal_prob = current_prob
                else:
                    # Test different chunk sizes
                    test_chunks = []
                    for chunk_size in chunk_size_options:
                        test_row = file_features.copy()
                        test_row['chunk_size'] = chunk_size * 1024  # Convert KB to bytes
                        test_chunks.append(test_row)
                    
                    # Prepare features for all test chunks
                    test_df = pd.DataFrame(test_chunks)
                    X_test, _ = self._prepare_features(test_df, training=False)
                    dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
                    
                    # Get probabilities for each chunk size
                    chunk_probs = self.model.predict(dtest)
                    
                    # Find the chunk size with highest probability
                    best_idx = np.argmax(chunk_probs)
                    optimal_chunk_size = chunk_size_options[best_idx]
                    optimal_prob = chunk_probs[best_idx]
                
                # Store results
                results.append({
                    'file_path': row['file_path'],
                    'file_size': row['file_size'],
                    'current_chunk_size': current_chunk_kb,
                    'current_probability': current_prob,
                    'predicted_chunk_size': optimal_chunk_size,
                    'predicted_probability': optimal_prob,
                    'is_current_optimal': current_prob >= 0.5,
                    'needs_optimization': current_chunk_kb != optimal_chunk_size,
                    'optimization_gain': optimal_prob - current_prob if current_chunk_kb != optimal_chunk_size else 0
                })
                
                logger.debug(f"File: {row['file_path']}, "
                           f"Current: {current_chunk_kb}KB (prob: {current_prob:.3f}), "
                           f"Optimal: {optimal_chunk_size}KB (prob: {optimal_prob:.3f})")
            
            results_df = pd.DataFrame(results)
            
            # Log summary statistics
            needs_opt = results_df['needs_optimization'].sum()
            total = len(results_df)
            avg_gain = results_df[results_df['needs_optimization']]['optimization_gain'].mean()
            
            logger.info(f"Prediction complete: {needs_opt}/{total} files need optimization")
            if needs_opt > 0:
                logger.info(f"Average predicted optimization gain: {avg_gain:.3f}")
            
            return results_df
            
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