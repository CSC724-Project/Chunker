#!/usr/bin/env python3
"""
Data simplification script for BeeChunker SOM training data.
This script transforms raw access data into simplified categorical features.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def simplify_data(input_file, output_file=None):
    """
    Transform training data into simplified categorical features.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the simplified data (default: input_file + "_simplified.csv")
    
    Returns:
        DataFrame with simplified features
    """
    # Set default output file if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_simplified{input_path.suffix}")
    
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    original_columns = list(df.columns)
    print(f"Loaded {len(df)} rows with {len(original_columns)} columns")
    
    # Remove file_path column
    if 'file_path' in df.columns:
        df = df.drop(columns=['file_path'])
        print("Removed file_path column")
    
    # Create simplified features
    simplified_df = pd.DataFrame()
    
    # 1. File size categories (4 categories)
    file_size_bins = [0, 200000000, 750000000, 1500000000, float('inf')]
    file_size_labels = [1, 2, 3, 4]  # Small, Medium, Large, Very Large
    simplified_df['file_size_cat'] = pd.cut(df['file_size'], bins=file_size_bins, labels=file_size_labels)
    
    # 2. Chunk size categories (4 categories)
    chunk_size_bins = [0, 256000, 1024000, 3072000, float('inf')]
    chunk_size_labels = [1, 2, 3, 4]  # Small, Medium, Large, Very Large
    simplified_df['chunk_size_cat'] = pd.cut(df['chunk_size'], bins=chunk_size_bins, labels=chunk_size_labels)
    
    # 3. High/low for access count
    access_count_median = df['access_count'].median()
    simplified_df['access_count_high'] = (df['access_count'] > access_count_median).astype(int)
    
    # 4. High/low for other numerical variables
    numerical_columns = [
        'avg_read_size', 'avg_write_size', 'max_read_size', 'max_write_size',
        'read_count', 'write_count', 'throughput_mbps'
    ]
    
    for col in numerical_columns:
        if col in df.columns:
            median_value = df[col].median()
            simplified_df[f'{col}_high'] = (df[col] > median_value).astype(int)
    
    # 5. Calculate read/write dominance (read-dominant=1, write-dominant=0)
    if 'read_count' in df.columns and 'write_count' in df.columns:
        simplified_df['read_dominant'] = (df['read_count'] > df['write_count']).astype(int)
    
    # 6. Verify we have all numerical data
    non_numeric_columns = simplified_df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_columns) > 0:
        print(f"Converting non-numeric columns to numeric: {non_numeric_columns}")
        for col in non_numeric_columns:
            simplified_df[col] = pd.to_numeric(simplified_df[col], errors='coerce')
            # Fill any missing values with median
            if simplified_df[col].isnull().any():
                simplified_df[col] = simplified_df[col].fillna(simplified_df[col].median())
    
    # 7. Add the original chunk_size as the target variable
    simplified_df['actual_chunk_size'] = df['chunk_size']
    
    # Save the simplified data
    simplified_df.to_csv(output_file, index=False)
    print(f"Saved simplified data with {len(simplified_df)} rows and {len(simplified_df.columns)} columns to {output_file}")
    print(f"New columns: {list(simplified_df.columns)}")
    
    return simplified_df

def main():
    parser = argparse.ArgumentParser(description='Simplify BeeChunker training data for SOM')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output-file', '-o', help='Output CSV file path')
    
    args = parser.parse_args()
    
    simplify_data(args.input_file, args.output_file)

if __name__ == '__main__':
    main()