import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from beechunker.common.beechunker_logging import setup_logging

logger = setup_logging("xgboost_feature_engine")

class XGBoostFeatureEngine:
    """Feature engineering class for XGBoost model."""
    
    def __init__(self):
        """Initialize the feature engineering class."""
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def clean(self, input_path: str) -> pd.DataFrame:
        """Clean and preprocess the input data."""
        try:
            # Read data
            df = pd.read_csv(input_path)
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Convert timestamps to datetime if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate derived features
            df = self._calculate_derived_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return None
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features from raw data."""
        try:
            # Calculate I/O related features
            df['io_per_chunk'] = (df['read_count'] + df['write_count']) / (df['file_size'] / df['chunk_size'])
            df['bytes_per_io'] = df['file_size'] / (df['read_count'] + df['write_count'])
            df['throughput_per_chunk'] = df['throughput_mbps'] / (df['file_size'] / df['chunk_size'])
            
            # Calculate chunk size ratios
            df['chunk_to_file_ratio'] = df['chunk_size'] / df['file_size']
            
            # Calculate efficiency metrics
            df['io_efficiency'] = (df['throughput_mbps'] * 1024 * 1024) / (df['file_size'] * (df['read_count'] + df['write_count']))
            df['chunk_efficiency'] = df['throughput_mbps'] / df['chunk_size']
            
            # Add time-based features if timestamp is present
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
            return df
            
        except Exception as e:
            logger.error(f"Error calculating derived features: {e}")
            return df
    
    def prepare_features(self, df: pd.DataFrame, training: bool = False) -> tuple:
        """Prepare features for model training or prediction."""
        try:
            # Select base features
            base_features = [
                'file_size',
                'read_count',
                'write_count',
                'io_per_chunk',
                'bytes_per_io',
                'throughput_per_chunk',
                'chunk_to_file_ratio',
                'io_efficiency',
                'chunk_efficiency'
            ]
            
            # Add time-based features if available
            if 'hour' in df.columns and 'day_of_week' in df.columns:
                base_features.extend(['hour', 'day_of_week'])
            
            # Extract features
            X = df[base_features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            if training:
                X_scaled = self.scaler.fit_transform(X)
                self.feature_names = base_features
            else:
                X_scaled = self.scaler.transform(X)
            
            return X_scaled, base_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None, None 