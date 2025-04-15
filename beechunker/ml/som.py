import os
import numpy as np
import pandas as pd
import joblib
from beechunker.common.beechunker_logging import setup_logging
from datetime import datetime
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from beechunker.common.config import config

logger = setup_logging("som")

class BeeChunkerSOM:
    """
    Self-Organizing Map (SOM) for prediction of chunk size
    """
    def __init__(self):
        """Init the SOM"""
        self.som = None
        self.scaler = None
        self.chunk_size_map = None
        self.feature_names = None
        self.models_dir = config.get("ml", "models_dir") 
        os.makedirs(self.models_dir, exist_ok=True)
    
    def set_last_training_time(self):
        with open(os.path.join(self.models_dir, "last_training_time.txt"), "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    def get_last_training_time(self):
        try:
            with open(os.path.join(self.models_dir, "last_training_time.txt"), "r") as f:
                last_training_time = f.read().strip()
            return datetime.strptime(last_training_time, "%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            logger.warning("Last training time file not found. Returning None.")
            return None
        except Exception as e:
            logger.error(f"Error reading last training time: {e}")
            return None
    
    def get_new_data_count(self, last_training_time):
        """
        Get the number of new data points since the last training time.
        Args:
            last_training_time (datetime): Last training time.
        Returns:
            int: Number of new data points.
        """
        try:
            # Load the data
            df = pd.read_csv(config.get("ml", "log_path"))
            # Filter the data based on the last training time
            new_data_count = len(df[df['timestamp'] > last_training_time])
            return new_data_count
        except Exception as e:
            logger.error(f"Error getting new data count: {e}")
            return 0
    
    
    def train(self, df) -> bool:
        """
        Train the SOM on the provided DataFrame.

        Args:
            df (pandas dataframe): DataFrame with features and chunk sizes.
        """
        logger.info("Starting SOM training")
        
        if len(df) < config.get("ml", "min_training_samples"):
            logger.warning("Not enough samples for training the SOM. Need at least %d samples.", config.get("ml", "min_training_samples"))
            return False
        
        # Extract features and chunk sizes
        X, y, self.feature_names = self._extract_features(df)
        
        # Scale the features
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Save scaler and feature names
        joblib.dump(self.scaler, os.path.join(self.models_dir, "scaler.joblib"))
        with open(os.path.join(self.models_dir, "feature_names.txt"), "w") as f:
            f.write("\n".join(self.feature_names))
        logger.info("Features and chunk sizes extracted and scaled")
        
        # Determine the SOM size based on data points
        # Rule of thumb : map units = 5 * sqrt(n_samples)
        map_size = int(np.ceil(5 * np.sqrt(len(X)))) # using X here instead of X_scaled because we need the original data for visualization
        # Make it a square
        map_size = min(max(map_size, 10), 30)
        logger.info("SOM map size: %d", map_size)
        
        # Init and train the SOM
        self.som = MiniSom(
            map_size, map_size,
            X_scaled.shape[1],
            sigma=1.0, # Initial neighborhood radius
            learning_rate=0.5, # Initial learning rate
            neighborhood_function='gaussian',
            random_seed=42
        )
        
        
        self.som.random_weights_init(X_scaled)
        
        self.som.train_batch(
            X_scaled,
            num_iteration=config.get("ml", "som_iterations"),
            verbose=True
        )
        
        # Create chunk size map
        self._create_chunk_size_map(X_scaled, y)
        logger.info("SOM training completed")
        
        # Save the trained SOM model
        joblib.dump(self.som, os.path.join(self.models_dir, "som_model.joblib"))
        np.save(os.path.join(self.models_dir, "chunk_size_map.npy"), self.chunk_size_map)
        
        # Create visualizations
        self._create_visualizations(X_scaled, y, df)
        
        # calculate the quantization error
        qe = self.som.quantization_error(X_scaled)
        logger.info("Quantization error: %f", qe)
        
        return True
    
    def _extract_features(self, df):
        """Extract and engineer features for training - supports both original and simplified data."""
        # Check if we're dealing with simplified data (categorical features)
        is_simplified = ('file_size_cat' in df.columns or 'chunk_size_cat' in df.columns)
        
        if is_simplified:
            # For simplified categorical data
            logger.info("Using simplified categorical features for training")
            
            # Get all numeric columns except chunk_size (our target)
            feature_columns = [col for col in df.columns 
                            if col != 'chunk_size' and col != 'actual_chunk_size']
            
            # If actual_chunk_size exists but chunk_size doesn't, use it as the target
            if 'chunk_size' not in df.columns and 'actual_chunk_size' in df.columns:
                df['chunk_size'] = df['actual_chunk_size']
                # Only try to remove if it's actually in the list
                if 'actual_chunk_size' in feature_columns:
                    feature_columns.remove('actual_chunk_size')
        else:
            # Original feature engineering for standard data format
            logger.info("Using standard feature engineering for training")
            
            # Feature engineering
            df['read_write_ratio'] = df['read_count'] / (df['write_count'] + 1)  # Avoid division by zero
            
            # Extract file extension
            if 'file_path' in df.columns:
                df['file_extension'] = df['file_path'].apply(lambda x: os.path.splitext(x)[1].lower())
                
                # One-hot encode common extensions
                common_extensions = ['.txt', '.csv', '.log', '.dat', '.bin', '.json', '.xml', '.db']
                for ext in common_extensions:
                    df[f'ext_{ext}'] = (df['file_extension'] == ext).astype(int)
                df['ext_other'] = (~df['file_extension'].isin(common_extensions)).astype(int)
                
                # Directory depth
                df['dir_depth'] = df['file_path'].apply(lambda x: len(x.split('/')))
            
            # Select features
            feature_columns = ['file_size', 'access_count', 'avg_read_size', 'avg_write_size', 
                            'max_read_size', 'max_write_size', 'read_count', 'write_count', 
                            'read_write_ratio', 'dir_depth'] + \
                            [f'ext_{ext}' for ext in common_extensions] + ['ext_other']
            
            # Remove features that don't exist in the dataset
            feature_columns = [col for col in feature_columns if col in df.columns]
    
        # Make sure all values are numeric
        for col in feature_columns:
            if df[col].dtype == 'object':
                logger.warning(f"Converting column {col} to numeric")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        X = df[feature_columns]
        y = df['chunk_size']
        
        return X, y, feature_columns
    
    def _create_chunk_size_map(self, X_scaled, y):
        """Create a map of chunk sizes by finding the most common chunk size for each node."""
        map_size = self.som.get_weights().shape[0]  # Get map dimensions
        self.chunk_size_map = np.zeros((map_size, map_size))
        
        # Track how many samples are assigned to each node
        activation_map = np.zeros((map_size, map_size))
        
        # Map each data point to its BMU (Best Matching Unit)
        for i, x in enumerate(X_scaled):
            # Find the winning node for this sample
            winner = self.som.winner(x)
            # Add the chunk size to that node
            self.chunk_size_map[winner] += y.iloc[i]
            # Increment the counter for that node
            activation_map[winner] += 1
        
        # Average the chunk sizes for each node with at least one mapped sample
        mask = activation_map > 0
        self.chunk_size_map[mask] = self.chunk_size_map[mask] / activation_map[mask]
        
        # Interpolate values for empty nodes
        if not np.all(mask):
            from scipy.ndimage import distance_transform_edt
            
            # Get coordinates of nodes with values
            coords = np.argwhere(mask)
            # Get values at those coordinates
            values = self.chunk_size_map[mask]
            
            # Calculate distances to filled nodes
            dist, indices = distance_transform_edt(
                ~mask, return_distances=True, return_indices=True
            )
            
            # Use indices to fill in empty nodes with nearest neighbor values
            i, j = indices
            self.chunk_size_map[~mask] = self.chunk_size_map[i[~mask], j[~mask]]
    
    def _create_visualizations(self, X_scaled, y, df):
        """Create visualizations of the SOM for analysis."""
        try:
            # Create directory for visualizations
            vis_dir = os.path.join(self.models_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Plot U-Matrix (distances between neurons)
            plt.figure(figsize=(10, 10))
            plt.pcolor(self.som.distance_map().T, cmap='bone_r')
            plt.colorbar()
            plt.title('U-Matrix: Distances Between Neurons')
            plt.savefig(os.path.join(vis_dir, "u_matrix.png"))
            plt.close()
            
            # Plot chunk size map
            plt.figure(figsize=(10, 10))
            plt.pcolor(self.chunk_size_map.T, cmap='viridis')
            plt.colorbar(label='Chunk Size (KB)')
            plt.title('Chunk Size Map')
            plt.savefig(os.path.join(vis_dir, "chunk_size_map.png"))
            plt.close()
            
            # Plot component planes for important features
            os.makedirs(os.path.join(vis_dir, "component_planes"), exist_ok=True)
            for i, feature in enumerate(self.feature_names):
                plt.figure(figsize=(8, 8))
                plt.pcolor(self.som.get_weights()[:, :, i].T, cmap='coolwarm')
                plt.colorbar(label=feature)
                plt.title(f'Component Plane: {feature}')
                plt.savefig(os.path.join(vis_dir, "component_planes", f"{feature}.png"))
                plt.close()
            
            # Plot hit map (number of samples mapped to each neuron)
            hit_map = np.zeros((self.som.get_weights().shape[0], self.som.get_weights().shape[1]))
            for x in X_scaled:
                winner = self.som.winner(x)
                hit_map[winner] += 1
                
            plt.figure(figsize=(10, 10))
            plt.pcolor(hit_map.T, cmap='Blues')
            plt.colorbar(label='Number of Samples')
            plt.title('Hit Map: Samples per Neuron')
            plt.savefig(os.path.join(vis_dir, "hit_map.png"))
            plt.close()
            
            logger.info(f"SOM visualizations saved to {vis_dir}")
            
        except Exception as e:
            logger.error(f"Error creating SOM visualization: {e}")
    
    def load(self):
        """Load trained SOM model."""
        try:
            model_path = os.path.join(self.models_dir, "som_model.joblib")
            scaler_path = os.path.join(self.models_dir, "scaler.joblib")
            map_path = os.path.join(self.models_dir, "chunk_size_map.npy")
            features_path = os.path.join(self.models_dir, "feature_names.txt")
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, map_path]):
                logger.warning("SOM model files not found")
                return False
            
            self.som = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.chunk_size_map = np.load(map_path)
            
            with open(features_path, 'r') as f:
                self.feature_names = f.read().strip().split('\n')
            
            logger.info("Successfully loaded SOM model")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SOM model: {e}")
            return False
        
    
    def predict(self, features):
        """Predict optimal chunk size for given features."""
        if self.som is None or self.chunk_size_map is None:
            if not self.load():
                logger.error("SOM model not loaded. Cannot predict.")
                return config.get("ml", "min_chunk_size")
        
        try:
            # Prep the features
            feature_vector = []
            for feature in self.feature_names:
                feature_vector.append(features.get(feature, 0))
            
            # Scale the features
            X_scaled = self.scaler.transform([feature_vector])
            
            # Find the best matching unit (BMU)
            bmu = self.som.winner(X_scaled[0])
            
            # Get chunk size from map
            chunk_size = self.chunk_size_map[bmu]
            
            logger.info(f"BMU: {bmu}, Chunk size: {chunk_size}")
            
            # Round to the nearest valid chunk size (multiples of 512KB)
            min_chunk = config.get("optimizer", "min_chunk_size")
            max_chunk = config.get("optimizer", "max_chunk_size")
            chunk_size_kb = round (chunk_size / min_chunk) * min_chunk
            
            # Enforce limits
            chunk_size_kb = round(chunk_size / (min_chunk * 10)) * min_chunk
            
            logger.info(f"BMU: {bmu}")
            logger.info(f"Predicted chunk size: {chunk_size_kb} KB")
            
            return int(chunk_size_kb)
        except Exception as e:
            logger.error(f"Error predicting chunk size: {e}")
            return config.get("ml", "min_chunk_size")
        
            