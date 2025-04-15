# Take a csv file with the following columns:
# file_path,file_size,chunk_size,access_count,avg_read_size,avg_write_size,max_read_size,max_write_size,read_count,write_count,throughput_mbps
# Apply PCA to it and save the transformed data to a new csv file
# Save the PCA model to a joblib file
import joblib
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


class FeatureEngineering:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.selected_features = None
        
    def plot_boxplots(self, dataframe, columns, figsize=(12, 8)):
        """Plot boxplots for the specified columns in the dataframe. (outlier detection)"""
        plt.figure(figsize=figsize)
        dataframe[columns].boxplot(vert=False)
        plt.title('Boxplots for Feature Distribution')
        plt.tight_layout()
        plt.savefig('feature_boxplots.png')
        plt.close()
    
    def clean(self, file_path):
            """Main function that orchestrates the entire data cleaning and preprocessing pipeline."""
            # 1. Load and validate data
            df = self._load_and_validate_data(file_path)
            
            # 2. Handle missing values
            df = self._handle_missing_values(df)
            
            # 3. Perform feature engineering
            df = self._perform_feature_engineering(df)
            
            # 4. Detect and remove outliers
            df_clean, outliers = self._detect_and_remove_outliers(df)
            
            # 5. Analyze feature correlations
            self._analyze_feature_correlations(df_clean)
            
            # 6. Perform feature selection
            feature_cols, X, y, correlation_selected_features, selected_features = self._perform_feature_selection(df_clean)
            
            # 7. Scale features
            X_scaled, X_robust_scaled = self._scale_features(X)
            
            # 8. Reduce dimensionality with PCA
            X_pca, pca, pca_cols = self._reduce_dimensionality(X_scaled)
            
            # 9. Create datasets for SOM training
            self._create_som_datasets(X_scaled, y, correlation_selected_features, selected_features, X_pca, pca_cols)
            
            # 10. Print summary and save models
            self._print_summary(df_clean, pca)
            
            return df_clean
    
    def _load_and_validate_data(self, file_path):
        """Load data from CSV and validate its structure."""
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            # Check if the file is a CSV
            if not file_path.endswith('.csv'):
                raise ValueError(f"The file {file_path} is not a CSV file.")
            # Read the CSV file
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading the file {file_path}: {e}")
        
        # Check if the required columns are present
        required_columns = ['file_path', 'file_size', 'chunk_size', 'access_count', 'avg_read_size', 
                          'avg_write_size', 'max_read_size', 'max_write_size', 'read_count', 
                          'write_count', 'throughput_mbps']
        
        # Debug print
        print(f"Required columns: {required_columns}")
        print(f"Data columns: {df.columns.tolist()}")
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing: {missing_columns}")
        # Check for missing values
        if df.isnull().values.any():
            raise ValueError("The data contains missing values.")
            
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            # Fill missing values with the mean of each column
            print(f"Missing values in each column:\n{missing_values}")
            print("Filling missing values with the median of each column.")
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median())
        return df
    
    def _perform_feature_engineering(self, df):
        """Create new features from existing ones."""
        print("Performing feature engineering...")
        # Extract file size in MB for better readability
        df['file_size_mb'] = df['file_size'] / (1024 * 1024)
        df['chunk_size_kb'] = df['chunk_size'] / 1024
        
        # Extract extension from file path
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
        
        return df
    
    def _detect_and_remove_outliers(self, df):
        """Detect and remove outliers using Z-score method."""
        print("\nDetecting and removing outliers...")
        
        # Select numeric columns for outlier detection (excluding the target variable chunk_size)
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                        if col != 'chunk_size' and col != 'chunk_size_kb']
        
        # Plot boxplots for numeric columns
        self.plot_boxplots(df, numeric_cols[:min(10, len(numeric_cols))]) # Plot first 10 numeric columns
        
        # Z-score method for outlier detection
        z_scores = stats.zscore(df[numeric_cols])
        abs_z_scores = np.abs(z_scores)
        outlier_rows = (abs_z_scores > 3).any(axis=1)
        print(f"Found {outlier_rows.sum()} potential outliers out of {len(df)} records ({outlier_rows.sum()/len(df)*100:.2f}%)")
        
        # Store the outlier rows for further analysis
        outliers = df[outlier_rows].copy()
        outliers.to_csv('outliers.csv', index=False)
        
        # Remove outliers from the original dataframe
        df_clean = df[~outlier_rows].copy()
        print(f"Removed {outlier_rows.sum()} outliers. Remaining records: {len(df_clean)}")
        
        return df_clean, outliers
    
    def _analyze_feature_correlations(self, df_clean):
        """Analyze correlations between features."""
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
                
        return target_correlation
    
    def _perform_feature_selection(self, df_clean):
        """Select the most relevant features for modeling."""
        print("\nPerforming feature selection...")
        
        # Select features excluding file_path and target chunk_size/chunk_size_kb
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
        y = df_encoded['chunk_size_kb']
        
        # Method 1: Feature selection using correlation with target
        correlation = df_clean.select_dtypes(include=['number']).corr()
        target_correlation = correlation['chunk_size'].sort_values(ascending=False)
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
        
        return feature_cols, X, y, correlation_selected_features, selected_features
    
    def _scale_features(self, X):
        """Scale the features using standardization."""
        print("\nScaling features...")

        # For regular scaling, use StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # For scaling with outlier resistance, use RobustScaler
        robust_scaler = RobustScaler()
        X_robust_scaled = pd.DataFrame(robust_scaler.fit_transform(X), columns=X.columns)
        
        return X_scaled, X_robust_scaled
    
    def _reduce_dimensionality(self, X_scaled):
        """Reduce dimensionality with PCA."""
        print("\nReducing dimensionality with PCA...")

        # Determine number of components to keep 95% of variance
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        # Create component names
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]

        print(f"Original number of features: {X_scaled.shape[1]}")
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
        self._visualize_pca_2d(X_pca)
        
        return X_pca, pca, pca_cols
    
    def _visualize_pca_2d(self, X_pca, y=None):
        """Visualize the data in 2D PCA space."""
        plt.figure(figsize=(10, 8))
        if y is not None:
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=30)
            plt.colorbar(scatter, label='Chunk Size (KB)')
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=30)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Visualization of Data')
        plt.savefig('pca_visualization.png')
        plt.close()
    
    def _create_som_datasets(self, X_scaled, y, correlation_selected_features, selected_features, X_pca, pca_cols):
        """Create different datasets for SOM training."""
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
        df_pca = pd.DataFrame(X_pca, columns=pca_cols)
        df_pca['chunk_size_kb'] = y.values
        df_pca.to_csv('som_data_pca.csv', index=False)

        print("\nProcessing complete. Created the following files:")
        print("1. som_data_full.csv - All features with scaling")
        print("2. som_data_correlation.csv - Top features by correlation")
        print("3. som_data_fscore.csv - Top features by F-score")
        print("4. som_data_pca.csv - PCA transformed data")
        print("5. outliers.csv - Outlier records for investigation")
        print("6. Various visualization PNG files")
    
    def _print_summary(self, df_clean, pca):
        """Print summary statistics and save models."""
        # Print some summary statistics about the final dataset
        print(f"\nFinal dataset size: {df_clean.shape[0]} records, {df_clean.shape[1]-1} features")
        print(f"Number of unique chunk sizes in data: {df_clean['chunk_size_kb'].nunique()}")
        print(f"Range of chunk sizes: {df_clean['chunk_size_kb'].min()} KB - {df_clean['chunk_size_kb'].max()} KB")
        
        # Save PCA model
        from joblib import dump
        dump(pca, 'pca_model.joblib')
        print("PCA model saved as pca_model.joblib")
    
    def plot_boxplots(self, df, columns):
        """Plot boxplots for the given columns."""
        n_cols = 2
        n_rows = (len(columns) + n_cols - 1) // n_cols
        plt.figure(figsize=(15, n_rows * 4))
        for i, col in enumerate(columns):
            plt.subplot(n_rows, n_cols, i + 1)
            df.boxplot(column=col)
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
        plt.savefig('boxplots.png')
        plt.close()
    
    def prepare_features_for_pca(self, df, scaler, pca_model_path=''):
        """
        Prepare raw features for PCA transformation.
        
        Args:
            df (pd.DataFrame): Input dataframe with raw features
            scaler (object): The scaler used for feature scaling
            pca_model_path (str): Path to the saved PCA model
        
        Returns:
            np.ndarray: PCA transformed features or fallback features
        """
        # Load PCA model if path is provided
        pca_model = None
        if pca_model_path and os.path.exists(pca_model_path):
            from joblib import load
            try:
                pca_model = load(pca_model_path)
                print(f"Loaded PCA model from {pca_model_path}")
            except Exception as e:
                print(f"Error loading PCA model: {e}")
        
        # Create engineered features using the same method as in the preprocessing
        df_features = self._create_engineered_features(df)
        
        # Get the expected features based on the PCA model
        expected_features, feature_matrix = self._extract_features_for_pca(df_features, pca_model)
        
        # Apply PCA transformation if model is available
        if pca_model is not None:
            try:
                # Apply scaling and PCA transformation
                return self._apply_pca_transformation(feature_matrix, pca_model)
            except Exception as e:
                print(f"Error applying PCA transformation: {e}")
        
        # Fallback to using scaled features directly
        return self._apply_fallback_approach(feature_matrix, scaler)

    def _create_engineered_features(self, df):
        """Create engineered features from raw data."""
        # Create a copy of the dataframe to avoid modifying the original
        df_features = df.copy()
        
        # Feature engineering - match exactly what was done in preprocessing
        # Extract file size in MB for better readability
        df_features['file_size_mb'] = df_features['file_size'] / (1024 * 1024)
        df_features['chunk_size_kb'] = df_features['chunk_size'] / 1024

        # Extract file extension from path
        df_features['file_extension'] = df_features['file_path'].apply(
            lambda x: os.path.splitext(x)[1].lower() if '.' in x else '')

        # Extract directory depth
        df_features['dir_depth'] = df_features['file_path'].apply(
            lambda x: len(x.split('/')))

        # Create meaningful ratios and derived features
        df_features['read_write_ratio'] = df_features['read_count'] / df_features['write_count'].replace(0, 1)
        df_features['avg_access_size'] = (df_features['avg_read_size'] * df_features['read_count'] + 
                                        df_features['avg_write_size'] * df_features['write_count']) / df_features['access_count'].replace(0, 1)
        df_features['max_access_size'] = df_features[['max_read_size', 'max_write_size']].max(axis=1)
        df_features['read_percentage'] = df_features['read_count'] / df_features['access_count'].replace(0, 1) * 100

        # Optional: Extract BeeGFS-specific patterns
        import re
        df_features['path_chunk_hint'] = df_features['file_path'].apply(
            lambda x: int(re.search(r'testdir_(\d+)K', x).group(1)) * 1024 if re.search(r'testdir_(\d+)K', x) else 0
        )
        
        return df_features

    def _extract_features_for_pca(self, df_features, pca_model):
        """Extract features needed for PCA transformation."""
        # If we have the PCA model, extract its expected feature names
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
            feature_cols = self._get_default_feature_columns()
            
            # Check which features are available
            missing_features = [f for f in feature_cols if f not in df_features.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                for feat in missing_features:
                    df_features[feat] = 0  # Create placeholder feature with zeros
            
            feature_matrix = df_features[feature_cols].values
            print(f"Using {len(feature_cols)} features based on fallback approach")
        
        return expected_features, feature_matrix

    def _get_default_feature_columns(self):
        """Return the default feature columns used for PCA."""
        return [
            'file_size', 'access_count', 'avg_read_size', 'avg_write_size',
            'max_read_size', 'max_write_size', 'read_count', 'write_count',
            'throughput_mbps', 'file_size_mb', 'dir_depth', 'read_write_ratio', 
            'avg_access_size', 'max_access_size', 'read_percentage', 'path_chunk_hint'
        ]

    def _apply_pca_transformation(self, feature_matrix, pca_model):
        """Apply PCA transformation to the feature matrix."""
        from sklearn.preprocessing import StandardScaler
        pre_scaler = StandardScaler()
        X_scaled = pre_scaler.fit_transform(feature_matrix)
        
        print(f"Input shape for PCA: {X_scaled.shape}")
        print(f"PCA expects {pca_model.n_components_} components from {pca_model.n_features_in_} features")
        
        # Apply PCA transformation if dimensions match
        if X_scaled.shape[1] == pca_model.n_features_in_:
            X_pca = pca_model.transform(X_scaled)
            print(f"Successfully applied PCA transformation: {feature_matrix.shape} -> {X_pca.shape}")
            return X_pca
        else:
            print(f"ERROR: PCA expects {pca_model.n_features_in_} features but we have {X_scaled.shape[1]}")
            raise ValueError("Feature dimensions do not match PCA model requirements")

    def _apply_fallback_approach(self, feature_matrix, scaler):
        """Apply fallback approach when PCA transformation fails."""
        print("Using placeholder approximation for PCA features")
        placeholder_pca = np.zeros((len(feature_matrix), scaler.scale_.shape[0]))
        
        # Use the first n features, where n is the number of features expected by the scaler
        feature_data = feature_matrix[:, :min(feature_matrix.shape[1], scaler.scale_.shape[0])]
        
        for i in range(min(scaler.scale_.shape[0], feature_data.shape[1])):
            placeholder_pca[:, i] = feature_data[:, i]
            
        return placeholder_pca
            