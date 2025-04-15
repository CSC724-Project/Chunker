import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import re

# 1. Load the dataset
df = pd.read_csv('/home/jgajbha/Chunker/data/beegfs_test_results.csv')

print(f"Original dataset shape: {df.shape}")
print("\nSample data:")
print(df.head())

# 2. Initial data exploration
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Handle missing values if any
if missing_values.sum() > 0:
    print("Handling missing values...")
    # For numerical columns, fill with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

# 3. Feature Engineering
print("\nPerforming feature engineering...")

# Extract file size in MB for better readability
df['file_size_mb'] = df['file_size'] / (1024 * 1024)
df['chunk_size_kb'] = df['chunk_size'] / 1024

# Extract file extension from path
df['file_extension'] = df['file_path'].apply(lambda x: os.path.splitext(x)[1].lower() if '.' in x else '')

# Extract directory depth
df['dir_depth'] = df['file_path'].apply(lambda x: len(x.split('/')))

# Create meaningful ratios and derived features
df['read_write_ratio'] = df['read_count'] / df['write_count'].replace(0, 1)  # Avoid division by zero
df['avg_access_size'] = (df['avg_read_size'] * df['read_count'] + df['avg_write_size'] * df['write_count']) / df['access_count'].replace(0, 1)
df['max_access_size'] = df[['max_read_size', 'max_write_size']].max(axis=1)
df['read_percentage'] = df['read_count'] / df['access_count'].replace(0, 1) * 100

# Optional: Extract BeeGFS-specific patterns
# Example: Extract initial chunk size from path if present
df['path_chunk_hint'] = df['file_path'].apply(
    lambda x: int(re.search(r'testdir_(\d+)K', x).group(1)) * 1024 if re.search(r'testdir_(\d+)K', x) else 0
)

# 4. Outlier Detection and Handling
print("\nDetecting and handling outliers...")

# Function to plot boxplots for visualizing outliers
def plot_boxplots(dataframe, columns, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    dataframe[columns].boxplot(vert=False)
    plt.title('Boxplots for Feature Distribution')
    plt.tight_layout()
    plt.savefig('feature_boxplots.png')
    plt.close()

# Select numeric columns for outlier detection (excluding the target variable chunk_size)
numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                if col != 'chunk_size' and col != 'chunk_size_kb']

# Plot boxplots for key features
plot_boxplots(df, numeric_cols[:min(10, len(numeric_cols))])  # Plot first 10 to avoid overcrowding

# Use Z-score method for outlier detection
z_scores = stats.zscore(df[numeric_cols])
abs_z_scores = np.abs(z_scores)
outlier_rows = (abs_z_scores > 3).any(axis=1)
print(f"Found {outlier_rows.sum()} potential outliers out of {len(df)} records ({outlier_rows.sum()/len(df)*100:.2f}%)")

# Store outliers separately for investigation
outliers = df[outlier_rows].copy()
outliers.to_csv('outliers.csv', index=False)

# Remove outliers for the main analysis
df_clean = df[~outlier_rows].copy()
print(f"Dataset shape after outlier removal: {df_clean.shape}")

# 5. Feature Correlation Analysis
print("\nAnalyzing feature correlations...")

# Calculate correlation matrix
correlation = df_clean.select_dtypes(include=['number']).corr()

# Plot correlation heatmap
plt.figure(figsize=(16, 14))
mask = np.triu(correlation)
sns.heatmap(correlation, annot=False, mask=mask, cmap='coolwarm', 
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# Print correlation with chunk_size (target variable)
target_correlation = correlation['chunk_size'].sort_values(ascending=False)
print("\nTop 10 features correlated with chunk_size:")
print(target_correlation.head(10))

# Find highly correlated features (potential redundancy)
correlation_threshold = 0.85
high_correlation = (correlation.abs() > correlation_threshold) & (correlation.abs() < 1.0)
correlated_features = []

for i in range(len(high_correlation.columns)):
    for j in range(i):
        if high_correlation.iloc[i, j]:
            colname_i = high_correlation.columns[i]
            colname_j = high_correlation.columns[j]
            correlated_features.append((colname_i, colname_j, correlation.iloc[i, j]))

if correlated_features:
    print("\nHighly correlated feature pairs (potential redundancy):")
    for feat1, feat2, corr in correlated_features:
        print(f"{feat1} and {feat2}: {corr:.3f}")

# 6. Feature Selection
print("\nPerforming feature selection...")

# Select feature columns to use, excluding file_path and target chunk_size/chunk_size_kb
feature_cols = [col for col in df_clean.columns 
                if col != 'file_path' and col != 'chunk_size' and col != 'chunk_size_kb' 
                and df_clean[col].dtype != 'object']

# Handle categorical features (one-hot encoding)
categorical_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object' and col != 'file_path']
if categorical_cols:
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
    feature_cols = [col for col in df_encoded.columns 
                    if col != 'file_path' and col != 'chunk_size' and col != 'chunk_size_kb']
else:
    df_encoded = df_clean.copy()

# Create feature matrix and target vector
X = df_encoded[feature_cols]
y = df_encoded['chunk_size_kb']  # Using KB is more intuitive than bytes

# Method 1: Feature selection using correlation with target
top_k = min(10, len(feature_cols))  # Select top 10 features or fewer if not enough features
correlation_selected_features = target_correlation.index[:top_k+1]  # +1 because first one is chunk_size itself
correlation_selected_features = [f for f in correlation_selected_features if f in feature_cols]

# Method 2: Feature selection using SelectKBest
selector = SelectKBest(f_regression, k=top_k)
selector.fit(X, y)
f_scores = pd.DataFrame({'Feature': X.columns, 'F_Score': selector.scores_})
f_scores = f_scores.sort_values('F_Score', ascending=False)
selected_features = f_scores['Feature'].head(top_k).tolist()

print("\nTop features by correlation:")
print(correlation_selected_features)
print("\nTop features by F-score:")
print(selected_features)

# 7. Feature Scaling
print("\nScaling features...")

# For regular scaling, use StandardScaler
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# For scaling with outlier resistance, use RobustScaler
robust_scaler = RobustScaler()
X_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(X), columns=X.columns)

# 8. Dimensionality Reduction with PCA
print("\nReducing dimensionality with PCA...")

# Determine number of components to keep 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Create component names
pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]

# Create a dataframe with PCA results
df_pca = pd.DataFrame(X_pca, columns=pca_cols)
df_pca['chunk_size_kb'] = y.values

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features: {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.7)
plt.plot(range(len(pca.explained_variance_ratio_)), 
         np.cumsum(pca.explained_variance_ratio_), 'r-')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.savefig('pca_variance.png')
plt.close()

# Visualize data in 2D PCA space, color by chunk size
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=30)
plt.colorbar(scatter, label='Chunk Size (KB)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Data with Chunk Size')
plt.savefig('pca_visualization.png')
plt.close()

# 9. Create different datasets for SOM training
print("\nCreating final datasets for SOM training...")

# Dataset 1: Full preprocessed dataset (with scaling)
df_full = pd.DataFrame(X_scaled)
df_full['chunk_size_kb'] = y.values
df_full.to_csv('som_data_full.csv', index=False)

# Dataset 2: Top features by correlation
X_correlation = X_scaled[[col for col in correlation_selected_features if col in X_scaled.columns]]
df_correlation = pd.DataFrame(X_correlation)
df_correlation['chunk_size_kb'] = y.values
df_correlation.to_csv('som_data_correlation.csv', index=False)

# Dataset 3: Top features by F-score
X_fscore = X_scaled[selected_features]
df_fscore = pd.DataFrame(X_fscore)
df_fscore['chunk_size_kb'] = y.values
df_fscore.to_csv('som_data_fscore.csv', index=False)

# Dataset 4: PCA transformed data
df_pca.to_csv('som_data_pca.csv', index=False)

print("\nProcessing complete. Created the following files:")
print("1. som_data_full.csv - All features with scaling")
print("2. som_data_correlation.csv - Top features by correlation")
print("3. som_data_fscore.csv - Top features by F-score")
print("4. som_data_pca.csv - PCA transformed data")
print("5. outliers.csv - Outlier records for investigation")
print("6. Various visualization PNG files")

# Print some summary statistics about the final dataset
print(f"\nFinal dataset size: {df_full.shape[0]} records, {df_full.shape[1]-1} features")
print(f"Number of unique chunk sizes in data: {df_clean['chunk_size_kb'].nunique()}")
print(f"Range of chunk sizes: {df_clean['chunk_size_kb'].min()} KB - {df_clean['chunk_size_kb'].max()} KB")
from joblib import dump
dump(pca, 'pca_model.joblib')