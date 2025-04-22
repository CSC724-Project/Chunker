import os
import json
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
        return (df['throughput_KBps']) / (df['file_size_KB'] * (df['read_ops'] + df['write_ops']))
    
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
            # Base features that are always required
            base_features = [
                'file_size_KB', 'chunk_size_KB', 'access_count', 
                'avg_read_KB', 'avg_write_KB', 'max_read_KB', 'max_write_KB',
                'read_ops', 'write_ops', 'throughput_KBps'
            ]
            
            # Check if all required features are present
            missing_features = [col for col in base_features if col not in df.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required features: {missing_features}")
            
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
        Label data based on throughput with sophisticated optimal throughput calculation.
        Uses a combination-based approach similar to the OptimalThroughputProcessor.
        
        Returns:
            tuple: (labeled_dataframe, thresholds_dict)
        """
        try:
            # Check if required columns exist
            required_columns = ['file_size_KB', 'chunk_size_KB', 'throughput_KBps', 'read_ops', 'write_ops']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Required columns missing: {missing_columns}. Available: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Ensure access_count exists
            if 'access_count' not in df.columns:
                df['access_count'] = df['read_ops'] + df['write_ops']
            
            # Create access_count_label based on access_count ranges
            df['access_count_label'] = df['access_count'].apply(
                lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
            )
            
            # Create combination for grouping
            df['combination'] = (
                df['file_size_KB'].astype(str) + ' | ' + 
                df['access_count_label'].astype(str)
            )
            
            logger.info(f"Created {df['combination'].nunique()} unique file size and access pattern combinations")
            
            # Get thresholds for each combination
            # Use minimum of 3 samples per combination or fall back to global threshold
            global_threshold = df['throughput_KBps'].quantile(0.65)
            logger.info(f"Global throughput threshold: {global_threshold:.2f} KBps (65th percentile)")
            
            # Initialize threshold column
            df['threshold'] = 0.0
            
            # Calculate thresholds per combination
            for combo, group in df.groupby('combination'):
                if len(group) >= 3:
                    # Use combination-specific threshold if enough samples
                    combo_threshold = group['throughput_KBps'].quantile(0.65)
                    df.loc[df['combination'] == combo, 'threshold'] = combo_threshold
                    logger.info(f"Combination '{combo}': {len(group)} samples, threshold = {combo_threshold:.2f} KBps")
                else:
                    # Use global threshold for small groups
                    df.loc[df['combination'] == combo, 'threshold'] = global_threshold
                    logger.info(f"Combination '{combo}': only {len(group)} samples, using global threshold")
            
            # Set optimal flag based on threshold
            df['is_optimal'] = (df['throughput_KBps'] >= df['threshold']).astype(int)
            
            # Additional chunk size rule: for larger files (>1MB), larger chunks tend to be better
            large_file_mask = df['file_size_KB'] > 1024
            file_groups = df[large_file_mask].groupby('file_path')
            
            for file_path, file_df in file_groups:
                if len(file_df) > 1:  # Only if we have multiple chunk sizes for the file
                    # Among optimal chunks, favor the larger ones for large files
                    optimal_chunks = file_df[file_df['is_optimal'] == 1]
                    if len(optimal_chunks) > 1:
                        # For large files, only keep the larger half of optimal chunk sizes
                        median_chunk = optimal_chunks['chunk_size_KB'].median()
                        small_optimal_chunks = optimal_chunks[optimal_chunks['chunk_size_KB'] < median_chunk]
                        
                        # Mark smaller chunks as non-optimal
                        if len(small_optimal_chunks) > 0:
                            df.loc[small_optimal_chunks.index, 'is_optimal'] = 0
                            logger.info(f"Large file {file_path}: marked {len(small_optimal_chunks)} smaller chunks as non-optimal")
            
            # For small files (<1MB), favor smaller chunks
            small_file_mask = df['file_size_KB'] <= 1024
            file_groups = df[small_file_mask].groupby('file_path')
            
            for file_path, file_df in file_groups:
                if len(file_df) > 1:  # Only if we have multiple chunk sizes for the file
                    # Among optimal chunks, favor the smaller ones for small files
                    optimal_chunks = file_df[file_df['is_optimal'] == 1]
                    if len(optimal_chunks) > 1:
                        # For small files, only keep the smaller half of optimal chunk sizes
                        median_chunk = optimal_chunks['chunk_size_KB'].median()
                        large_optimal_chunks = optimal_chunks[optimal_chunks['chunk_size_KB'] > median_chunk]
                        
                        # Mark larger chunks as non-optimal
                        if len(large_optimal_chunks) > 0:
                            df.loc[large_optimal_chunks.index, 'is_optimal'] = 0
                            logger.info(f"Small file {file_path}: marked {len(large_optimal_chunks)} larger chunks as non-optimal")
            
            # Log labeling statistics
            optimal_count = df['is_optimal'].sum()
            total_count = len(df)
            logger.info(f"Data labeling complete: {optimal_count}/{total_count} "
                      f"({optimal_count/total_count*100:.2f}%) samples labeled as optimal")
            
            # Ensure we have both positive and negative examples
            if optimal_count == 0:
                logger.warning("No optimal examples found, setting top 25% by throughput as optimal")
                top_indices = df.nlargest(int(len(df) * 0.25), 'throughput_KBps').index
                df.loc[top_indices, 'is_optimal'] = 1
            elif optimal_count == total_count:
                logger.warning("All examples labeled as optimal, setting bottom 25% by throughput as non-optimal")
                bottom_indices = df.nsmallest(int(len(df) * 0.25), 'throughput_KBps').index
                df.loc[bottom_indices, 'is_optimal'] = 0
            
            # Return labeled dataframe and thresholds
            thresholds = {
                'global_throughput': float(global_threshold),
                'combinations': df['combination'].nunique(),
                'optimal_ratio': float(optimal_count / total_count)
            }
            
            # Clean up temporary columns
            df.drop(columns=['threshold', 'combination', 'access_count_label'], inplace=True, errors='ignore')
            
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
            logger.info(f"Loaded data with {len(df)} rows and columns: {df.columns.tolist()}")
            
            # Check for required columns
            required_columns = ['file_path', 'file_size_KB', 'chunk_size_KB', 'throughput_KBps']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return False
            
            # Clean data - remove rows with missing values in essential columns
            original_len = len(df)
            df = df.dropna(subset=['file_size_KB', 'chunk_size_KB', 'throughput_KBps'])
            if len(df) < original_len:
                logger.info(f"Removed {original_len - len(df)} rows with missing values in essential columns")
            
            # Label the data based on I/O efficiency and throughput
            try:
                df, thresholds = self._label_data(df)
            except Exception as e:
                logger.error(f"Failed to label data: {e}")
                return False
            
            # Prepare features
            try:
                X, feature_names = self._prepare_features(df, training=True)
                y = df['is_optimal']
            except Exception as e:
                logger.error(f"Failed to prepare features: {e}")
                return False
            
            # Initialize K-fold
            n_splits = min(5, len(df))  # Ensure we don't try more splits than samples
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
            
            # Print detailed fold header
            logger.info("\n" + "="*50)
            logger.info(f"Beginning {n_splits}-fold cross-validation")
            logger.info("="*50)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Print fold details
                logger.info(f"\nFold {fold}/{n_splits}:")
                logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
                logger.info(f"Positive samples in training: {sum(y_train)} ({sum(y_train)/len(y_train)*100:.1f}%)")
                logger.info(f"Positive samples in validation: {sum(y_val)} ({sum(y_val)/len(y_val)*100:.1f}%)")
                
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
                
                # Print fold metrics
                logger.info("Fold metrics:")
                for metric_name, value in fold_metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")
                
                # Store metrics
                for metric, value in fold_metrics.items():
                    metrics[metric].append(value)
                
                # Update best model
                if fold_metrics['logloss'] < best_score:
                    best_score = fold_metrics['logloss']
                    best_model = model
                    logger.info(f"  New best model (logloss: {best_score:.4f})")
            
            # Print summary of all folds
            logger.info("\n" + "="*50)
            logger.info("Cross-validation summary:")
            for metric, values in metrics.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                logger.info(f"{metric} = {mean_value:.4f} ± {std_value:.4f}")
            logger.info("="*50 + "\n")
            
            # Save best model and components
            self.model = best_model
            self.feature_names = feature_names
            
            model_path = os.path.join(self.models_dir, "xgboost_model.json")
            scaler_path = os.path.join(self.models_dir, "xgboost_scaler.joblib")
            feature_path = os.path.join(self.models_dir, "xgboost_model_info.json")
            
            # Save the model, scaler, and feature information
            self.model.save_model(model_path)
            dump(self.scaler, scaler_path)
            
            # Save feature names and additional model information
            model_info = {
                'features': feature_names if isinstance(feature_names, list) else feature_names.tolist(),
                'thresholds': thresholds,
                'trained_at': datetime.now().isoformat(),
                'metrics': {k: float(np.mean(v)) for k, v in metrics.items()}
            }
            
            with open(feature_path, 'w') as f:
                json.dump(model_info, f, indent=2)
                
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            logger.info(f"Model info saved to {feature_path}")
            
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
    
    def predict(self, df: pd.DataFrame) -> int:
        """
        Predict optimal chunk size for the given dataframe.
        Returns:
            int: The optimal chunk size in KB
        """
        try:
            if self.model is None:
                if not self.load():
                    raise RuntimeError("Model not loaded")
            
            # Verify required columns
            required_columns = ['file_path', 'file_size_KB', 'chunk_size_KB', 'throughput_KBps']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for prediction: {missing_columns}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return None
            
            # Take the first row in the dataframe 
            row = df.iloc[0]
            
            current_chunk_kb = row['chunk_size_KB']
            file_features = row.copy()
            file_size_kb = row['file_size_KB']
            
            logger.info(f"Predicting optimal chunk size for file: {row['file_path']}")
            logger.info(f"File size: {file_size_kb}KB, Current chunk size: {current_chunk_kb}KB")
            
            # Use a heuristic approach based on file size to recommend chunk sizes
            # This provides a better initial estimate before applying the model
            
            # For very small files (< 256KB), small chunks (64-128KB) are typically best
            if file_size_kb < 256:
                possible_chunks = [64, 128, min(file_size_kb, 256)]
                logger.info(f"Small file (<256KB): Testing chunk sizes {possible_chunks}")
            
            # For small files (256KB-1MB), medium-small chunks (128-512KB) are typically best
            elif file_size_kb < 1024: 
                possible_chunks = [64, 128, 256, min(file_size_kb, 512)]
                logger.info(f"Medium-small file (256KB-1MB): Testing chunk sizes {possible_chunks}")
            
            # For medium files (1-4MB), medium chunks (256-1024KB) are typically best
            elif file_size_kb < 4096:
                possible_chunks = [128, 256, 512, min(file_size_kb, 1024)]
                logger.info(f"Medium file (1-4MB): Testing chunk sizes {possible_chunks}")
            
            # For large files (4-16MB), medium-large chunks (512-2048KB) are typically best
            elif file_size_kb < 16384:
                possible_chunks = [256, 512, 1024, min(file_size_kb, 2048)]
                logger.info(f"Large file (4-16MB): Testing chunk sizes {possible_chunks}")
            
            # For very large files (>16MB), large chunks (1024-8192KB) are typically best
            else:
                possible_chunks = [512, 1024, 2048, 4096, min(file_size_kb, 8192)]
                logger.info(f"Very large file (>16MB): Testing chunk sizes {possible_chunks}")
            
            # Add current chunk size to the list if it's not already there
            if current_chunk_kb not in possible_chunks:
                possible_chunks.append(int(current_chunk_kb))
                possible_chunks.sort()
            
            # Make sure all chunks are <= file size
            valid_chunk_sizes = [c for c in possible_chunks if c <= file_size_kb]
            
            if not valid_chunk_sizes:
                logger.info(f"No valid chunk sizes for file size {file_size_kb}KB, defaulting to 64KB")
                return 64  # Default to smallest chunk size
            
            logger.info(f"Testing chunk sizes: {valid_chunk_sizes}")
            
            # Test different valid chunk sizes
            test_chunks = []
            for chunk_size in valid_chunk_sizes:
                test_row = file_features.copy()
                test_row['chunk_size_KB'] = chunk_size
                test_chunks.append(test_row)
            
            # Prepare features for all test chunks
            test_df = pd.DataFrame(test_chunks)
            X_test, _ = self._prepare_features(test_df, training=False)
            dtest = xgb.DMatrix(X_test, feature_names=self.feature_names)
            
            # Get probabilities for each chunk size
            chunk_probs = self.model.predict(dtest)
            
            # Log probabilities for each chunk size
            for i, (size, prob) in enumerate(zip(valid_chunk_sizes, chunk_probs)):
                logger.info(f"Chunk size {size}KB: probability {prob:.4f}")
            
            # Select best chunk size
            best_idx = np.argmax(chunk_probs)
            best_chunk_size = valid_chunk_sizes[best_idx]
            best_prob = chunk_probs[best_idx]
            
            # Check if current chunk size is in the valid options
            if current_chunk_kb in valid_chunk_sizes:
                current_idx = valid_chunk_sizes.index(current_chunk_kb)
                current_prob = chunk_probs[current_idx]
                logger.info(f"Current chunk size {current_chunk_kb}KB: probability {current_prob:.4f}")
                
                # Make more aggressive recommendations:
                # 1. Only keep current if it's the absolute best
                # 2. For files > 1MB, prefer larger chunks
                # 3. For files < 1MB, prefer smaller chunks
                
                # If current is best, keep it
                if current_idx == best_idx:
                    logger.info(f"Current chunk size {current_chunk_kb}KB is optimal")
                    return int(current_chunk_kb)
                
                # For large files, bias toward larger chunks
                if file_size_kb > 1024:
                    # If current chunk is too small and not optimal, recommend larger
                    if current_chunk_kb < best_chunk_size:
                        logger.info(f"File is large ({file_size_kb}KB), recommending larger chunk: {best_chunk_size}KB")
                        return int(best_chunk_size)
                # For small files, bias toward smaller chunks    
                else:
                    # If current chunk is too large and not optimal, recommend smaller
                    if current_chunk_kb > best_chunk_size:
                        logger.info(f"File is small ({file_size_kb}KB), recommending smaller chunk: {best_chunk_size}KB")
                        return int(best_chunk_size)
                
                # If difference in probabilities is significant (>0.1), recommend change
                if best_prob > current_prob + 0.1:
                    logger.info(f"Significant performance improvement expected with {best_chunk_size}KB")
                    return int(best_chunk_size)
                
                # Otherwise, keep current chunk size to minimize changes
                logger.info(f"Keeping current chunk size {current_chunk_kb}KB (performance difference not significant)")
                return int(current_chunk_kb)
            
            # Current chunk size not in valid options, return best chunk size
            logger.info(f"Selected optimal chunk size: {best_chunk_size}KB with probability {best_prob:.4f}")
            return int(best_chunk_size)
                
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