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