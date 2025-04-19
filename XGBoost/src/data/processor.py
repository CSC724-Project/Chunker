import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any
import torch
from ..config import config

class FeatureProcessor:
    def __init__(self):
        """Initialize the feature processor"""
        self.scaler = StandardScaler()
        self.feature_names = config['metrics'].METRICS_TO_COLLECT + [
            'file_size',
            'is_compressed',
            'access_frequency',
            'current_chunk_size',
            'num_targets',
            'io_efficiency',
            'network_saturation',
            'storage_pressure'
        ]
        self.fitted = False

    def engineer_features(self, 
                         system_metrics: Dict[str, float], 
                         file_metrics: Dict[str, Any]) -> np.ndarray:
        """Create feature vector from raw metrics"""
        features = []
        
        # System metrics
        for metric in config['metrics'].METRICS_TO_COLLECT:
            features.append(system_metrics.get(metric, 0.0))
        
        # File metrics
        features.extend([
            file_metrics.get('file_size', 0),
            float(file_metrics.get('is_compressed', False)),
            file_metrics.get('access_frequency', 0.0),
            file_metrics.get('current_chunk_size', 0),
            file_metrics.get('num_targets', 0)
        ])
        
        # Derived features
        features.extend([
            self._calculate_io_efficiency(system_metrics),
            self._calculate_network_saturation(system_metrics),
            self._calculate_storage_pressure(system_metrics)
        ])
        
        return np.array(features, dtype=np.float32)

    def process_features(self, features: np.ndarray, training: bool = False) -> torch.Tensor:
        """Scale features and convert to PyTorch tensor"""
        if training and not self.fitted:
            self.scaler.fit(features)
            self.fitted = True
        
        if self.fitted:
            features = self.scaler.transform(features)
        
        return torch.FloatTensor(features)

    def _calculate_io_efficiency(self, metrics: Dict[str, float]) -> float:
        """Calculate I/O efficiency score"""
        throughput = metrics.get('io_throughput', 0.0)
        latency = metrics.get('network_latency', 0.0)
        return throughput / (latency + 1e-6)  # Avoid division by zero

    def _calculate_network_saturation(self, metrics: Dict[str, float]) -> float:
        """Calculate network saturation score"""
        bandwidth = metrics.get('network_bandwidth', 0.0)
        concurrent_ops = metrics.get('concurrent_ops', 0.0)
        return (bandwidth * concurrent_ops) / 100.0

    def _calculate_storage_pressure(self, metrics: Dict[str, float]) -> float:
        """Calculate storage pressure score"""
        storage_usage = metrics.get('storage_usage', 0.0)
        metadata_ops = metrics.get('metadata_ops_rate', 0.0)
        return (storage_usage + metadata_ops) / 2.0

    def get_optimal_chunk_size(self, features: np.ndarray) -> int:
        """Calculate optimal chunk size based on features (for training data)"""
        # This is a simplified heuristic - you might want to adjust based on your needs
        file_size = features[self.feature_names.index('file_size')]
        io_efficiency = features[self.feature_names.index('io_efficiency')]
        network_saturation = features[self.feature_names.index('network_saturation')]
        
        base_chunk_size = 1024  # 1MB base chunk size
        
        # Adjust based on file size
        if file_size < 1024 * 1024:  # < 1MB
            base_chunk_size = 64  # 64KB
        elif file_size > 1024 * 1024 * 1024:  # > 1GB
            base_chunk_size = 4096  # 4MB
            
        # Adjust based on I/O efficiency and network saturation
        multiplier = 1.0
        if io_efficiency > 0.8:
            multiplier *= 1.5
        if network_saturation < 0.3:
            multiplier *= 1.2
            
        chunk_size = int(base_chunk_size * multiplier)
        
        # Ensure within bounds
        return max(config['model'].MIN_CHUNK_SIZE,
                  min(config['model'].MAX_CHUNK_SIZE, chunk_size)) 