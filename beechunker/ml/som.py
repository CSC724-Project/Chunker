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
from beechunker.ml.feature_engineering import FeatureEngineering

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
        self.feature_engine = FeatureEngineering()
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
        
    def train(self, input_data) -> bool:
        """
        Train the SOM on the provided data.

        Args:
            input_data: Either a file path (str) or a DataFrame with features and chunk sizes.
        Returns:
            bool: True if training was successful, False otherwise.
        """
        logger.info("Starting SOM training")
        
        # If input_data is a DataFrame, check the sample count
        if isinstance(input_data, pd.DataFrame):
            df = input_data
            if len(df) < config.get("ml", "min_training_samples"):
                logger.warning("Not enough samples for training the SOM. Need at least %d samples.", 
                            config.get("ml", "min_training_samples"))
                return False
            
            # Save the DataFrame to a temporary file for processing
            temp_csv_path = os.path.join(self.models_dir, "temp_training_data.csv")
            df.to_csv(temp_csv_path, index=False)
            logger.info(f"Saved input DataFrame to temporary file: {temp_csv_path}")
            
            # Use the file path with the clean method
            input_path = temp_csv_path
        else:
            # Assume input_data is a file path
            input_path = input_data
            
            # Check if the file exists
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            # Check the sample count from the file
            try:
                df_check = pd.read_csv(input_path)
                if len(df_check) < config.get("ml", "min_training_samples"):
                    logger.warning("Not enough samples for training the SOM. Need at least %d samples.", 
                                config.get("ml", "min_training_samples"))
                    return False
            except Exception as e:
                logger.error(f"Error checking input file: {e}")
                return False
        
        try:
            # Clean and preprocess the data using the FeatureEngineering class
            df_clean = self.feature_engine.clean(input_path)
            
            # Extract features and chunk sizes
            X, y, self.feature_names = self._extract_features_from_preprocessed(df_clean)
            
            # Scale the features
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Save scaler, PCA model and feature names
            joblib.dump(self.scaler, os.path.join(self.models_dir, "som_scaler.joblib"))
            pca_model_path = 'pca_model.joblib'
            if os.path.exists(pca_model_path):
                # Copy PCA model from the source location to the models directory
                import shutil
                target_path = os.path.join(self.models_dir, "pca_model.joblib")
                shutil.copy2(pca_model_path, target_path)
                logger.info(f"Copied PCA model to {target_path}")
            
            with open(os.path.join(self.models_dir, "feature_names.txt"), "w") as f:
                f.write("\n".join(self.feature_names))
            logger.info("Features and chunk sizes extracted and scaled")
            
            # Determine the SOM size based on data points
            # Rule of thumb : map units = 5 * sqrt(n_samples)
            map_size = int(np.ceil(5 * np.sqrt(len(X))))
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
            joblib.dump(self.chunk_size_map, os.path.join(self.models_dir, "som_chunk_size_map.joblib"))
            
            # Create visualizations
            self._create_visualizations(X_scaled, y, df_clean)
            
            # Calculate the quantization error
            qe = self.som.quantization_error(X_scaled)
            logger.info("Quantization error: %f", qe)
            
            # Set the last training time
            self.set_last_training_time()
            
            # Clean up temporary file if it was created
            if isinstance(input_data, pd.DataFrame) and os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                logger.info(f"Removed temporary file: {temp_csv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during SOM training: {e}")
            # Clean up temporary file if there was an error
            if isinstance(input_data, pd.DataFrame):
                temp_csv_path = os.path.join(self.models_dir, "temp_training_data.csv")
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
                    logger.info(f"Removed temporary file: {temp_csv_path}")
            return False
    
    def _extract_features_from_preprocessed(self, df_clean):
        """
        Extract features from preprocessed data.
        
        Args:
            df_clean (pandas.DataFrame): Preprocessed dataframe from feature engineering
        
        Returns:
            tuple: (X, y, feature_names) - features, target, and feature names
        """
        # Get PCA transformed data if available
        try:
            # Try to load PCA model if it exists
            pca_model_path = os.path.join(self.models_dir, "pca_model.joblib")
            if not os.path.exists(pca_model_path):
                pca_model_path = 'pca_model.joblib'  # Fallback to current directory
            
            if os.path.exists(pca_model_path):
                # Create a temporary scaler just for PCA transformation
                temp_scaler = MinMaxScaler()
                temp_scaler.fit(df_clean.select_dtypes(include=['number']).values)
                
                # Get PCA features
                X_pca = self.feature_engine.prepare_features_for_pca(df_clean, temp_scaler, pca_model_path)
                feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
                
                logger.info(f"Using PCA features for SOM training: {len(feature_names)} components")
                return X_pca, df_clean['chunk_size_kb'], feature_names
            
        except Exception as e:
            logger.warning(f"Error using PCA features, falling back to standard features: {e}")
        
        # Fallback to standard features if PCA fails
        logger.info("Using standard features for SOM training")
        
        # Remove non-numeric columns and target variable
        numeric_cols = df_clean.select_dtypes(include=['number']).columns
        feature_cols = [col for col in numeric_cols 
                       if col != 'chunk_size' and col != 'chunk_size_kb']
        
        X = df_clean[feature_cols].values
        y = df_clean['chunk_size_kb']
        
        return X, y, feature_cols
    
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
        """Load trained SOM model and components."""
        try:
            model_path = os.path.join(self.models_dir, "som_model.joblib")
            scaler_path = os.path.join(self.models_dir, "som_scaler.joblib")
            map_path = os.path.join(self.models_dir, "som_chunk_size_map.joblib")
            features_path = os.path.join(self.models_dir, "feature_names.txt")
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, map_path]):
                logger.warning("SOM model files not found")
                return False
            
            self.som = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.chunk_size_map = joblib.load(map_path)
            
            with open(features_path, 'r') as f:
                self.feature_names = f.read().strip().split('\n')
            
            logger.info("Successfully loaded SOM model")
            return True
            
        except Exception as e:
            logger.error(f"Error loading SOM model: {e}")
            return False
        
        
    def predict(self, df):
        """
        Predict optimal chunk sizes for the given dataframe.
        
        Args:
            df (pandas.DataFrame): Dataframe with raw features
                
        Returns:
            pandas.DataFrame: Dataframe with predictions
        """
        if self.som is None or self.chunk_size_map is None or self.scaler is None:
            if not self.load():
                logger.error("SOM model not loaded. Cannot predict.")
                return None
        
        try:
            # Try to load PCA model if it exists
            pca_model_path = os.path.join(self.models_dir, "pca_model.joblib")
            pca_model = None
            
            if os.path.exists(pca_model_path):
                try:
                    pca_model = joblib.load(pca_model_path)
                    logger.info("Successfully loaded PCA model for prediction")
                except Exception as e:
                    logger.error(f"Error loading PCA model: {e}")
            
            # Prepare features for prediction
            X = self.feature_engine.prepare_features_for_pca(df, self.scaler, pca_model_path)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict for each row
            predictions = []
            for i, x in enumerate(X_scaled):
                # Find the best matching unit (BMU)
                bmu = self.som.winner(x)
                
                # Get the chunk size from the map
                predicted_chunk_size = self.chunk_size_map[bmu]
                
                # Convert to KB if it's in bytes
                if predicted_chunk_size > 10000:  # Likely in bytes
                    predicted_chunk_size = predicted_chunk_size / 1024
                
                # Round to nearest 128KB (typical chunk size increments in BeeGFS)
                predicted_chunk_size_kb = round(predicted_chunk_size / 128) * 128
                
                # Ensure it's in a reasonable range
                # Use the correct config access method based on your config structure
                min_chunk = config.get("optimizer").get("min_chunk_size", 128)
                max_chunk = config.get("optimizer").get("max_chunk_size", 8192)
                predicted_chunk_size_kb = max(min_chunk, min(predicted_chunk_size_kb, max_chunk))
                
                # Store prediction
                predictions.append({
                    'file_path': df.iloc[i]['file_path'],
                    'file_size': df.iloc[i]['file_size'],
                    'current_chunk_size': df.iloc[i]['chunk_size'] // 1024,  # Convert to KB
                    'predicted_chunk_size': predicted_chunk_size_kb,
                    'current_to_predicted_ratio': df.iloc[i]['chunk_size'] / (predicted_chunk_size_kb * 1024),
                    'bmu': bmu
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            logger.error("Feature mismatch detected. Printing debug information:")
            logger.error(f"Scaler expects {self.scaler.scale_.shape[0]} features")
            logger.error(f"Feature names: {self.feature_names}")
            return None
    
    def visualize_predictions(self, df_pred):
        """
        Visualize where the predictions fall on the SOM map.
        
        Args:
            df_pred (pandas.DataFrame): Dataframe with predictions from the predict method
        """
        try:
            # Create directory for visualizations
            vis_dir = os.path.join(self.models_dir, "prediction_visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            map_size = self.som.get_weights().shape[0]
            
            # Plot the chunk size map
            plt.figure(figsize=(12, 10))
            plt.pcolor(self.chunk_size_map.T, cmap='viridis')
            plt.colorbar(label='Chunk Size (KB)')
            
            # Add a counter for plots at each position
            bmu_counts = {}
            
            # Plot each prediction as a point
            for _, row in df_pred.iterrows():
                bmu = row['bmu']
                bmu_key = f"{bmu[0]},{bmu[1]}"
                
                # Count how many points are at this location
                if bmu_key not in bmu_counts:
                    bmu_counts[bmu_key] = 0
                bmu_counts[bmu_key] += 1
                
                # Add jitter based on count to avoid overlap
                count = bmu_counts[bmu_key]
                jitter_x = 0.1 * (count % 3 - 1)  # -0.1, 0, 0.1
                jitter_y = 0.1 * (count // 3 - 1) # Spread vertically for more points
                
                plt.plot(bmu[0] + 0.5 + jitter_x, bmu[1] + 0.5 + jitter_y, 'ro', 
                        markersize=8, markeredgecolor='black')
                
            plt.title('Predicted Chunk Sizes on SOM Map')
            plt.xlim(0, map_size)
            plt.ylim(0, map_size)
            plt.savefig(os.path.join(vis_dir, 'som_predictions.png'))
            plt.close()

            # Compare current vs predicted
            plt.figure(figsize=(12, 6))
            x = np.arange(len(df_pred))
            width = 0.35
            
            plt.bar(x - width/2, df_pred['current_chunk_size'], width, label='Current Chunk Size (KB)')
            plt.bar(x + width/2, df_pred['predicted_chunk_size'], width, label='Predicted Chunk Size (KB)')
            
            # Add file sizes as text above bars
            for i, row in df_pred.iterrows():
                file_size_mb = row['file_size'] / 1_000_000
                plt.text(i, max(row['current_chunk_size'], row['predicted_chunk_size']) + 200, 
                        f"{file_size_mb:.0f}MB", ha='center', va='bottom', rotation=90, fontsize=8)
            
            plt.xlabel('File')
            plt.ylabel('Chunk Size (KB)')
            plt.title('Current vs Predicted Chunk Sizes')
            plt.xticks(x, [os.path.basename(path) for path in df_pred['file_path']], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'chunk_size_comparison.png'))
            plt.close()
            
            # Plot the ratio of current to predicted chunk sizes
            plt.figure(figsize=(12, 6))
            colors = ['red' if ratio > 1 else 'green' for ratio in df_pred['current_to_predicted_ratio']]
            plt.bar(x, df_pred['current_to_predicted_ratio'], color=colors)
            plt.axhline(y=1, color='black', linestyle='--')
            plt.xlabel('File')
            plt.ylabel('Current / Predicted Ratio')
            plt.title('Ratio of Current to Predicted Chunk Sizes')
            plt.xticks(x, [os.path.basename(path) for path in df_pred['file_path']], rotation=45, ha='right')
            
            # Add annotations
            for i, ratio in enumerate(df_pred['current_to_predicted_ratio']):
                plt.text(i, ratio + 0.1, f"{ratio:.2f}x", ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'chunk_size_ratio.png'))
            plt.close()
            
            # Generate a distribution of predicted chunk sizes
            plt.figure(figsize=(10, 6))
            plt.hist(df_pred['predicted_chunk_size'], bins=10, alpha=0.7)
            plt.xlabel('Predicted Chunk Size (KB)')
            plt.ylabel('Number of Files')
            plt.title('Distribution of Predicted Chunk Sizes')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'predicted_size_distribution.png'))
            plt.close()
            
            logger.info(f"Prediction visualizations saved to {vis_dir}")
            
        except Exception as e:
            logger.error(f"Error creating prediction visualizations: {e}")