import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_scaler = StandardScaler()
        self.feature_names = None
        self.feature_stats = {}
        self.plots_dir = Path("results/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the data."""
        df = pd.read_csv(csv_path)
        self._analyze_data(df)  # Analyze data distribution
        return self._extract_features(df)
    
    def _analyze_data(self, df: pd.DataFrame) -> None:
        """Analyze data distribution and log statistics."""
        logging.info("\nData Analysis:")
        
        # Analyze class distribution
        class_dist = df['OT'].value_counts()
        self.target_stats = {
            'class_distribution': class_dist.to_dict(),
            'class_balance': (class_dist / len(df)).to_dict()
        }
        
        # Log basic statistics
        logging.info("\nClass Distribution:")
        for label, count in class_dist.items():
            logging.info(f"Class {label}: {count} ({count/len(df)*100:.2f}%)")
        
        # Plot class distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x='OT')
        plt.title('Target Class Distribution')
        plt.savefig(self.plots_dir / 'class_distribution.png')
        plt.close()
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and engineer features from raw data."""
        # Log transform size-related features
        df['log_file_size'] = np.log1p(df['file_size_KB'])
        df['log_chunk_size'] = np.log1p(df['chunk_size_KB'])
        
        # Domain-specific feature engineering
        df['size_ratio'] = df['chunk_size_KB'] / df['file_size_KB']
        df['ops_per_chunk'] = df['access_count'] / (df['file_size_KB'] / df['chunk_size_KB'])
        
        # Normalize features by file size
        for col in ['avg_read_KB', 'avg_write_KB', 'max_read_KB', 'max_write_KB']:
            df[f'norm_{col}'] = df[col] / df['file_size_KB']
        
        # Operation patterns
        total_ops = df['read_ops'] + df['write_ops']
        df['read_ratio'] = df['read_ops'] / total_ops
        df['write_ratio'] = df['write_ops'] / total_ops
        df['ops_density'] = total_ops / df['file_size_KB']  # ops per KB
        
        # Performance metrics
        df['throughput_per_op'] = df['throughput_KBps'] / total_ops
        df['throughput_density'] = df['throughput_KBps'] / df['file_size_KB']
        df['log_throughput'] = np.log1p(df['throughput_KBps'])
        
        # Access count related features
        df['log_access_count'] = np.log1p(df['access_count'])
        df['access_density'] = df['access_count'] / df['file_size_KB']  # accesses per KB
        df['access_label_encoded'] = pd.Categorical(df['access_count_label']).codes
        
        # Combination feature (if it's categorical)
        df['combination_encoded'] = pd.Categorical(df['combination']).codes
        
        # Selected features
        features = [
            'log_file_size',
            'log_chunk_size',
            'size_ratio',
            'ops_per_chunk',
            'ops_density',
            'read_ratio',
            'write_ratio',
            'norm_avg_read_KB',
            'norm_avg_write_KB',
            'norm_max_read_KB',
            'norm_max_write_KB',
            'throughput_density',
            'throughput_per_op',
            'log_throughput',
            'log_access_count',
            'access_density',
            'access_label_encoded',
            'combination_encoded'
        ]
        
        self.feature_names = features
        
        # Analyze feature correlations
        correlation_matrix = df[features + ['OT']].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_correlations.png')
        plt.close()
        
        # Log correlations with target
        correlations = correlation_matrix['OT'].sort_values(ascending=False)
        logging.info("\nFeature correlations with OT:")
        for feat, corr in correlations.items():
            if feat != 'OT':
                logging.info(f"{feat}: {corr:.3f}")
        
        X = df[features].values
        y = df['OT'].values
        
        # Clean data
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        return X_scaled, y
    
    def get_feature_names(self):
        """Return list of feature names."""
        return self.feature_names
    
    def get_target_stats(self):
        """Return target statistics."""
        return self.target_stats
    
    def prepare_k_fold(self, X, y, n_splits=5):
        """Prepare k-fold cross validation splits."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return kf.split(X, y)
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using the fitted scaler."""
        X_train_scaled = self.feature_scaler.transform(X_train)
        if X_test is not None:
            X_test_scaled = self.feature_scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled 