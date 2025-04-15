import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from scipy.spatial.distance import cdist
import time


df = pd.read_csv("/home/jgajbha/Chunker/data/preprocess/som_data_pca.csv")

print("Data loaded. Shape:", df.shape)
print(df.head())

# Extract features and targets
X = df.iloc[:, :-1].values  # All PC columns
chunk_sizes = df.iloc[:, -1].values   # Store chunk size in a differently named variable to avoid confusion

# Feature scaling for SOM (minisom prefers 0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Features scaled to range [0,1]")

# Calculate optimal map size
# Rule of thumb: map_units ≈ 5 * sqrt(n_samples)
def calculate_map_size(n_samples):
    map_units = int(5 * np.sqrt(n_samples))
    # For small datasets, ensure at least a 2x2 map
    return max(2, map_units)

map_size = calculate_map_size(X.shape[0])
print(f"Calculated map size: {map_size}x{map_size}")

# For large datasets, we'll use a smaller map size for practical reasons
map_size = 20  # Override for better performance with large datasets
print(f"Using map size: {map_size}x{map_size}")

# Initialize SOM
som_dim = X_scaled.shape[1]  # Number of features (PCs)
som = MiniSom(map_size, map_size, som_dim, 
              sigma=map_size/4,         # Initial neighborhood radius (1/4 of map size)
              learning_rate=0.5,        # Initial learning rate
              neighborhood_function='gaussian',
              random_seed=42)

# Initialize weights
som.random_weights_init(X_scaled)
print("SOM initialized with random weights")

# Track training time and progress
start_time = time.time()

# Longer training for better results with large datasets
n_iterations = 20000
print(f"Starting SOM training with {n_iterations} iterations...")

# Train SOM
som.train(X_scaled, n_iterations, verbose=True)

# Calculate training time
training_time = time.time() - start_time
print(f"SOM training completed in {training_time:.2f} seconds")

# Evaluation metrics
# 1. Quantization error (average distance between each data vector and its BMU)
quantization_error = som.quantization_error(X_scaled)
print(f"Quantization error: {quantization_error:.6f}")

# 2. Topographic error (proportion of data vectors for which the first and second BMUs are not neighbors)
def topographic_error(som, data):
    error_count = 0
    for x in data:
        winner = som.winner(x)
        
        # Create a distance matrix excluding the winner itself
        distances = np.zeros((som._weights.shape[0], som._weights.shape[1]))
        for i in range(som._weights.shape[0]):
            for j in range(som._weights.shape[1]):
                if (i, j) == winner:
                    # Set the winner's distance to infinity
                    distances[i, j] = float('inf')
                else:
                    distances[i, j] = np.linalg.norm(som._weights[i, j] - x)
        
        # Find the second best matching unit
        second_bmu = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Check if the first and second BMUs are adjacent
        if abs(winner[0] - second_bmu[0]) > 1 or abs(winner[1] - second_bmu[1]) > 1:
            error_count += 1
    
    return error_count / len(data)

topo_error = topographic_error(som, X_scaled)
print(f"Topographic error: {topo_error:.6f}")

# 3. Generate mapped locations for clustering metrics
winners = np.array([som.winner(x) for x in X_scaled])
# Convert 2D coordinates to 1D cluster labels
labels = winners[:, 0] * map_size + winners[:, 1]

# Only calculate silhouette if we have enough distinct clusters and samples
n_clusters = len(np.unique(labels))
if n_clusters > 1 and n_clusters < len(X_scaled):
    silhouette = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {silhouette:.6f}")
else:
    print("Silhouette score: Not available (need at least 2 clusters with >1 sample)")

# Create chunk size map
chunk_size_map = np.zeros((map_size, map_size))
count_map = np.zeros((map_size, map_size))

for i, x in enumerate(X_scaled):
    winner = som.winner(x)
    chunk_size_map[winner] += chunk_sizes[i]
    count_map[winner] += 1

# Average the chunk sizes
mask = count_map > 0
chunk_size_map[mask] = chunk_size_map[mask] / count_map[mask]

print("\nChunk size map (KB):")
print(chunk_size_map)

# Create hit map (count of samples mapped to each neuron)
print("\nHit map (samples per neuron):")
print(count_map)

# Visualizations
# 1. U-Matrix visualization (distances between neurons)
plt.figure(figsize=(12, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar(label='Distance')
plt.title('U-Matrix: Distances Between Neurons')
plt.savefig('som_u_matrix.png')
plt.close()

# 2. Chunk size map visualization
plt.figure(figsize=(12, 10))
plt.pcolor(chunk_size_map.T, cmap='viridis')
plt.colorbar(label='Chunk Size (KB)')
plt.title('SOM Chunk Size Map')

# Add labels to cells - but only if not too many cells
if map_size <= 20:
    for i in range(map_size):
        for j in range(map_size):
            if count_map[i, j] > 0:
                plt.text(i + 0.5, j + 0.5, f'{chunk_size_map[i, j]:.0f}', 
                        ha='center', va='center', 
                        color='white' if chunk_size_map[i, j] > 3000 else 'black',
                        fontsize=8)

plt.savefig('som_chunk_size_map.png')
plt.close()

# 3. Hit map visualization
plt.figure(figsize=(12, 10))
plt.pcolor(count_map.T, cmap='Blues')
plt.colorbar(label='Sample Count')
plt.title('SOM Hit Map')

# Add count labels - but only if not too many cells
if map_size <= 20:
    for i in range(map_size):
        for j in range(map_size):
            if count_map[i, j] > 0:
                plt.text(i + 0.5, j + 0.5, f'{int(count_map[i, j])}', 
                        ha='center', va='center', 
                        color='white' if count_map[i, j] > np.mean(count_map[count_map>0]) else 'black',
                        fontsize=8)

plt.savefig('som_hit_map.png')
plt.close()

# 4. Component planes - show how each feature influences the map
fig = plt.figure(figsize=(18, 12))
for i in range(X.shape[1]):
    plt.subplot(2, 3, i+1)
    component_plane = np.zeros((map_size, map_size))
    for x in range(map_size):
        for y in range(map_size):
            component_plane[x, y] = som._weights[x, y, i]
    plt.pcolor(component_plane.T, cmap='coolwarm')
    plt.colorbar()
    plt.title(f'PC{i+1}')
plt.tight_layout()
plt.savefig('som_component_planes.png')
plt.close()

# 5. Visualization of data points on the map
fig, ax = plt.subplots(figsize=(14, 12))  # Create figure and axis objects explicitly
# Plot the SOM grid
ax.pcolor(np.zeros((map_size, map_size)), cmap='Blues', alpha=0.3, 
          edgecolors='gray', linewidths=0.5)

# Create a colormap for chunk sizes
# Use numpy's min and max functions explicitly on the array
chunk_min, chunk_max = np.min(chunk_sizes), np.max(chunk_sizes)
norm = plt.Normalize(chunk_min, chunk_max)
cmap = plt.colormaps.get_cmap('viridis')

# For large datasets, sample a subset of points to plot to avoid overcrowding
max_points_to_plot = 500
if len(X_scaled) > max_points_to_plot:
    indices = np.random.choice(len(X_scaled), max_points_to_plot, replace=False)
    plot_X = X_scaled[indices]
    plot_chunk_sizes = chunk_sizes[indices]
else:
    plot_X = X_scaled
    plot_chunk_sizes = chunk_sizes

# Plot each data point
for i, x in enumerate(plot_X):
    winner = som.winner(x)
    # Add jitter to avoid perfect overlap
    jitter = 0.2 * (np.random.rand(2) - 0.5)
    ax.plot(winner[0] + 0.5 + jitter[0], winner[1] + 0.5 + jitter[1], 'o', 
            markersize=8, markeredgecolor='black', 
            markerfacecolor=cmap(norm(plot_chunk_sizes[i])))

# Create scalar mappable for colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # You need to set an array for the colorbar to work properly
plt.colorbar(sm, ax=ax, label='Chunk Size (KB)')

ax.set_title('Data Points Mapped to SOM')
ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
plt.savefig('som_mapped_data.png')
plt.close()

# 6. Summary metrics table
print("\nSOM Training Summary:")
print("====================")
print(f"Map size: {map_size}x{map_size}")
print(f"Input features: {X.shape[1]} principal components")
print(f"Training samples: {X.shape[0]}")
print(f"Training iterations: {n_iterations}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Quantization error: {quantization_error:.6f}")
print(f"Topographic error: {topo_error:.6f}")

if n_clusters > 1 and n_clusters < len(X_scaled):
    print(f"Silhouette score: {silhouette:.6f}")

# Additional useful stats:
print("\nChunk size distribution statistics:")
active_neurons = np.sum(count_map > 0)
print(f"Active neurons: {active_neurons} out of {map_size*map_size} ({active_neurons/(map_size*map_size)*100:.1f}%)")

chunk_sizes_neurons = chunk_size_map[count_map > 0]
print(f"Chunk size range: {np.min(chunk_sizes_neurons):.0f} KB - {np.max(chunk_sizes_neurons):.0f} KB")
print(f"Average chunk size: {np.mean(chunk_sizes_neurons):.0f} KB")
print(f"Median chunk size: {np.median(chunk_sizes_neurons):.0f} KB")

# Count clusters by chunk size range
chunk_ranges = {
    '0-500 KB': 0,
    '500-1000 KB': 0,
    '1000-2000 KB': 0,
    '2000-4000 KB': 0,
    '4000+ KB': 0
}

for cs in chunk_sizes_neurons:
    if cs < 500:
        chunk_ranges['0-500 KB'] += 1
    elif cs < 1000:
        chunk_ranges['500-1000 KB'] += 1
    elif cs < 2000:
        chunk_ranges['1000-2000 KB'] += 1
    elif cs < 4000:
        chunk_ranges['2000-4000 KB'] += 1
    else:
        chunk_ranges['4000+ KB'] += 1

print("\nChunk size distribution by range:")
for range_name, count in chunk_ranges.items():
    print(f"{range_name}: {count} neurons ({count/active_neurons*100:.1f}%)")

# Only print detailed neuron info if we have a reasonable number of neurons
if map_size <= 10:
    print("\nChunk size distribution per neuron:")
    for i in range(map_size):
        for j in range(map_size):
            if count_map[i, j] > 0:
                print(f"Neuron ({i},{j}): {chunk_size_map[i, j]:.0f} KB, {int(count_map[i, j])} samples")

# Print prediction function that can be used later
print("\n# Prediction Function for Production Use:")
print("def predict_chunk_size(model, scaler, pca_features):")
print("    # Scale the features")
print("    pca_scaled = scaler.transform([pca_features])")
print("    # Find the best matching unit")
print("    bmu = model.winner(pca_scaled[0])")
print("    # Return the chunk size for that unit")
print("    return chunk_size_map[bmu]")

# Save the model and related components for later use
import joblib
joblib.dump(som, 'som_model.joblib')
joblib.dump(scaler, 'som_scaler.joblib')
joblib.dump(chunk_size_map, 'som_chunk_size_map.joblib')
print("\nModel saved to 'som_model.joblib'")

# 7. Plot the distribution of chunk sizes
plt.figure(figsize=(10, 6))
chunk_sizes_flat = chunk_size_map[count_map > 0].flatten()
plt.hist(chunk_sizes_flat, bins=20, alpha=0.7, color='teal')
plt.xlabel('Chunk Size (KB)')
plt.ylabel('Number of Neurons')
plt.title('Distribution of Chunk Sizes Across SOM Neurons')
plt.grid(alpha=0.3)
plt.savefig('som_chunk_size_distribution.png')
plt.close()

# 8. Plot the relation between quantization error and number of samples
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axis explicitly
# Reshape for scatter plot
x = count_map[count_map > 0].flatten()
y_errors = np.zeros_like(x)
for i in range(map_size):
    for j in range(map_size):
        if count_map[i, j] > 0:
            # Calculate average quantization error for this neuron
            samples_in_cell = []
            errors_in_cell = []
            for k, sample in enumerate(X_scaled):
                if som.winner(sample) == (i, j):
                    samples_in_cell.append(sample)
                    errors_in_cell.append(np.linalg.norm(sample - som._weights[i, j]))
            
            if samples_in_cell:
                idx = np.where((count_map > 0).flatten())[0][list(zip(*np.where(count_map > 0))).index((i, j))]
                y_errors[idx] = np.mean(errors_in_cell)

ax.scatter(x, y_errors, alpha=0.7, c='purple')
ax.set_xlabel('Number of Samples per Neuron')
ax.set_ylabel('Average Quantization Error')
ax.set_title('Sample Count vs. Quantization Error per Neuron')
ax.grid(alpha=0.3)
plt.savefig('som_error_vs_samples.png')
plt.close()