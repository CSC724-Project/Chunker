import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and components
def load_model(model_dir='/home/jgajbha/Chunker/som/'):
    """Load the SOM model and its components."""
    try:
        som = joblib.load(os.path.join(model_dir, 'som_model.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'som_scaler.joblib'))
        chunk_size_map = joblib.load(os.path.join(model_dir, 'som_chunk_size_map.joblib'))
        
        # Try to load PCA model if it exists
        pca_model_path = os.path.join(model_dir, 'pca_model.joblib')
        pca_model = None
        if os.path.exists(pca_model_path):
            try:
                pca_model = joblib.load(pca_model_path)
                print(f"Successfully loaded PCA model")
            except Exception as e:
                print(f"Error loading PCA model: {e}")
        else:
            print(f"PCA model not found at {pca_model_path}")
            
            # Try to find PCA model in alternative locations
            alternative_paths = [
                "/home/jgajbha/Chunker/data/preprocess/pca_model.joblib",
                "./pca_model.joblib",
                "../pca_model.joblib"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    try:
                        pca_model = joblib.load(alt_path)
                        print(f"Successfully loaded PCA model from {alt_path}")
                        break
                    except Exception as e:
                        print(f"Error loading PCA model from {alt_path}: {e}")
        
        # Load feature names if available, otherwise use default
        feature_names_path = os.path.join(model_dir, 'feature_names.txt')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = f.read().strip().split('\n')
        else:
            # Default PCA features (adjust this based on your model)
            feature_names = [f'PC{i+1}' for i in range(scaler.scale_.shape[0])]
        
        return som, scaler, chunk_size_map, feature_names, pca_model
    except FileNotFoundError as e:
        print(f"Error loading model components: {e}")
        print("Make sure the model files exist in the specified directory.")
        return None, None, None, None, None

# Feature engineering function - for PCA
def prepare_features_for_pca(df, scaler, pca_model=None):
    """Prepare raw features that would be used for PCA."""
    # Create a copy of the dataframe to avoid modifying the original
    df_features = df.copy()
    
    # Feature engineering - match exactly what was done in preprocessing
    # Extract file size in MB for better readability
    df_features['file_size_mb'] = df_features['file_size'] / (1024 * 1024)
    df_features['chunk_size_kb'] = df_features['chunk_size'] / 1024

    # Extract file extension from path
    df_features['file_extension'] = df_features['file_path'].apply(lambda x: os.path.splitext(x)[1].lower() if '.' in x else '')

    # Extract directory depth
    df_features['dir_depth'] = df_features['file_path'].apply(lambda x: len(x.split('/')))

    # Create meaningful ratios and derived features
    df_features['read_write_ratio'] = df_features['read_count'] / df_features['write_count'].replace(0, 1)  # Avoid division by zero
    df_features['avg_access_size'] = (df_features['avg_read_size'] * df_features['read_count'] + 
                                     df_features['avg_write_size'] * df_features['write_count']) / df_features['access_count'].replace(0, 1)
    df_features['max_access_size'] = df_features[['max_read_size', 'max_write_size']].max(axis=1)
    df_features['read_percentage'] = df_features['read_count'] / df_features['access_count'].replace(0, 1) * 100

    # Optional: Extract BeeGFS-specific patterns
    import re
    df_features['path_chunk_hint'] = df_features['file_path'].apply(
        lambda x: int(re.search(r'testdir_(\d+)K', x).group(1)) * 1024 if re.search(r'testdir_(\d+)K', x) else 0
    )
    
    # If we have the PCA model, we can extract its expected feature names
    expected_features = None
    if pca_model is not None and hasattr(pca_model, 'feature_names_in_'):
        expected_features = list(pca_model.feature_names_in_)
        print(f"PCA expects exactly these features: {expected_features}")
    
    # If we have expected feature names, use only those
    if expected_features:
        # Check if all expected features are available
        missing_features = [f for f in expected_features if f not in df_features.columns]
        
        # If we're missing any features, try to create them
        if missing_features:
            print(f"Missing expected features: {missing_features}")
            for feat in missing_features:
                df_features[feat] = 0  # Create placeholder feature with zeros
        
        # Extract only the expected features in the correct order
        feature_matrix = df_features[expected_features].values
        print(f"Using exactly the {len(expected_features)} features the PCA model expects")
    else:
        # Fallback method - select features based on preprocessing without one-hot encoding
        feature_cols = [
            'file_size', 'access_count', 'avg_read_size', 'avg_write_size',
            'max_read_size', 'max_write_size', 'read_count', 'write_count',
            'throughput_mbps', 'file_size_mb', 'dir_depth', 'read_write_ratio', 
            'avg_access_size', 'max_access_size', 'read_percentage', 'path_chunk_hint'
        ]
        
        # Check which features are available
        missing_features = [f for f in feature_cols if f not in df_features.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for feat in missing_features:
                df_features[feat] = 0  # Create placeholder feature with zeros
        
        feature_matrix = df_features[feature_cols].values
        print(f"Using {len(feature_cols)} features based on fallback approach")
    
    # Apply the PCA transformation
    if pca_model is not None:
        try:
            # Scale the data first (PCA expects scaled data)
            from sklearn.preprocessing import StandardScaler
            pre_scaler = StandardScaler()
            X_scaled = pre_scaler.fit_transform(feature_matrix)
            
            print(f"Input shape for PCA: {X_scaled.shape}")
            print(f"PCA expects {pca_model.n_components_} components from {pca_model.n_features_in_} features")
            
            # Apply PCA transformation
            if X_scaled.shape[1] == pca_model.n_features_in_:
                X_pca = pca_model.transform(X_scaled)
                print(f"Successfully applied PCA transformation: {feature_matrix.shape} -> {X_pca.shape}")
                return X_pca
            else:
                print(f"ERROR: PCA expects {pca_model.n_features_in_} features but we have {X_scaled.shape[1]}")
                # Falling back...
        except Exception as e:
            print(f"Error applying PCA transformation: {e}")
    
    # Fallback approach - use a subset of raw features
    print("Using placeholder approximation for PCA features")
    placeholder_pca = np.zeros((len(df), scaler.scale_.shape[0]))
    # Use the first n features, where n is the number of features expected by the scaler
    feature_data = feature_matrix[:, :scaler.scale_.shape[0]]
    for i in range(min(scaler.scale_.shape[0], feature_data.shape[1])):
        placeholder_pca[:, i] = feature_data[:, i]
        
    return placeholder_pca

# Predict function
def predict_chunk_sizes(df, som, scaler, chunk_size_map, feature_names, pca_model=None):
    """Predict optimal chunk sizes for the given data."""
    # For regular features (non-PCA approach)
    try:
        # Prepare features for the model
        X = prepare_features_for_pca(df, scaler, pca_model)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Predict for each row
        predictions = []
        for i, x in enumerate(X_scaled):
            # Find the best matching unit (BMU)
            bmu = som.winner(x)
            
            # Get the chunk size from the map
            predicted_chunk_size = chunk_size_map[bmu]
            
            # Convert to KB if it's in bytes
            if predicted_chunk_size > 10000:  # Likely in bytes
                predicted_chunk_size = predicted_chunk_size / 1024
            
            # Round to nearest 128KB (typical chunk size increments in BeeGFS)
            predicted_chunk_size_kb = round(predicted_chunk_size / 128) * 128
            
            # Ensure it's in a reasonable range
            predicted_chunk_size_kb = max(128, min(predicted_chunk_size_kb, 8192))
            
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
        print(f"Error in prediction: {e}")
        print("Feature mismatch detected. Printing debug information:")
        print(f"Scaler expects {scaler.scale_.shape[0]} features")
        print(f"Feature names: {feature_names}")
        raise e

# Visualization function
def visualize_predictions(df_pred, chunk_size_map, map_size=20):
    """Visualize where the predictions fall on the SOM map."""
    plt.figure(figsize=(12, 10))
    
    # Plot the chunk size map
    plt.pcolor(chunk_size_map.T, cmap='viridis')
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
    plt.savefig('som_predictions.png')
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
    plt.savefig('chunk_size_comparison.png')
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
    plt.savefig('chunk_size_ratio.png')
    plt.close()
    
    # Generate a distribution of predicted chunk sizes
    plt.figure(figsize=(10, 6))
    plt.hist(df_pred['predicted_chunk_size'], bins=10, alpha=0.7)
    plt.xlabel('Predicted Chunk Size (KB)')
    plt.ylabel('Number of Files')
    plt.title('Distribution of Predicted Chunk Sizes')
    plt.grid(alpha=0.3)
    plt.savefig('predicted_size_distribution.png')
    plt.close()

# Main function
def main():
    
    df = pd.read_csv("/home/jgajbha/Chunker/data/beegfs_test_results.csv")
    print(f"Loaded {len(df)} records.")
    
    # Load model components
    som, scaler, chunk_size_map, feature_names, pca_model = load_model()
    if som is None:
        return
    
    print(f"Model expects {scaler.scale_.shape[0]} features")
    
    # Check if we need to generate the PCA model
    if pca_model is None:
        print("No PCA model found. You might need to save your PCA model during preprocessing.")
        print("Attempting to create an approximation based on your training data...")
        
        try:
            # Create a simple PCA approximation - THIS IS JUST FOR DEMONSTRATION
            # In reality, you should save and load the actual PCA model used in training
            from sklearn.decomposition import PCA
            pca_model = PCA(n_components=scaler.scale_.shape[0])
            print("Created placeholder PCA model for demonstration purposes.")
            print("Note: This will not match your actual training PCA transformation!")
        except Exception as e:
            print(f"Failed to create PCA approximation: {e}")
    
    # Make predictions
    predictions = predict_chunk_sizes(df, som, scaler, chunk_size_map, feature_names, pca_model)
    
    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nPredictions:")
    print(predictions)
    
    # Save to CSV
    predictions.to_csv('chunk_size_predictions.csv', index=False)
    print("\nSaved predictions to 'chunk_size_predictions.csv'")
    
    # Calculate improvement potential
    current_sizes = df['chunk_size'].sum() / 1024  # KB
    predicted_sizes = predictions['predicted_chunk_size'].sum()
    
    print("\nStorage Impact Analysis:")
    print(f"Total current chunk size allocation: {current_sizes:.0f} KB")
    print(f"Total predicted chunk size allocation: {predicted_sizes:.0f} KB")
    
    if predicted_sizes < current_sizes:
        savings = (current_sizes - predicted_sizes) / current_sizes * 100
        print(f"Potential storage reduction: {savings:.2f}%")
    else:
        increase = (predicted_sizes - current_sizes) / current_sizes * 100
        print(f"Storage increase for better performance: {increase:.2f}%")
    
    # Analyze prediction diversity
    unique_predictions = predictions['predicted_chunk_size'].nunique()
    print(f"\nPrediction diversity: {unique_predictions} unique chunk sizes predicted")
    print("Chunk size distribution:")
    print(predictions['predicted_chunk_size'].value_counts().sort_index())
    
    # Visualize results
    visualize_predictions(predictions, chunk_size_map)
    print("Created visualizations: 'som_predictions.png', 'chunk_size_comparison.png', and 'chunk_size_ratio.png'")

if __name__ == "__main__":
    main()