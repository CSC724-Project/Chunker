"""Command-line interface for SOM training."""
import os
import sys
import click
import pandas as pd
import logging

from beechunker.common.beechunker_logging import setup_logging
from beechunker.common.config import config
from beechunker.ml.som import BeeChunkerSOM

@click.group()
def cli():
    """BeeChunker SOM trainer command-line interface."""
    pass

@cli.command()
@click.option('--input-csv', '-i', help='Input CSV file with access pattern data')
def train(input_csv):
    """Train the SOM model."""
    logger = setup_logging("ml")
    
    som = BeeChunkerSOM()
    
    if input_csv:
        # Load data from CSV
        if not os.path.exists(input_csv):
            logger.error(f"Input file not found: {input_csv}")
            sys.exit(1)
        
        logger.info(f"Loading data from {input_csv}")
        df = pd.read_csv(input_csv)
    else:
        # Load data from database
        logger.info("Loading data from database")
        import sqlite3
        
        db_path = config.get("monitor", "db_path")
        if not os.path.exists(db_path):
            logger.error(f"Database not found: {db_path}")
            sys.exit(1)
        
        conn = sqlite3.connect(db_path)
        query = """
        SELECT 
            a.file_path,
            m.chunk_size,
            m.file_size,
            COUNT(a.access_time) as access_count,
            AVG(a.read_size) as avg_read_size,
            AVG(a.write_size) as avg_write_size,
            MAX(a.read_size) as max_read_size,
            MAX(a.write_size) as max_write_size,
            COUNT(CASE WHEN a.access_type = 'read' THEN 1 END) as read_count,
            COUNT(CASE WHEN a.access_type = 'write' THEN 1 END) as write_count
        FROM file_access a
        JOIN file_metadata m ON a.file_path = m.file_path
        GROUP BY a.file_path
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
    
    if len(df) < config.get("ml", "min_training_samples"):
        logger.warning(f"Not enough data for training. Need at least {config.get('ml', 'min_training_samples')} samples.")
        sys.exit(1)
    
    logger.info(f"Training with {len(df)} samples")
    success = som.train(df)
    
    if success:
        logger.info("Training completed successfully")
    else:
        logger.error("Training failed")
        sys.exit(1)

@cli.command()
@click.option('--file-size', '-s', type=int, required=True, help='File size in bytes')
@click.option('--read-size', '-r', type=int, default=4096, help='Average read size')
@click.option('--write-size', '-w', type=int, default=4096, help='Average write size')
@click.option('--read-count', '-rc', type=int, default=10, help='Estimated read operations')
@click.option('--write-count', '-wc', type=int, default=5, help='Estimated write operations')
@click.option('--extension', '-e', default='', help='File extension (e.g., .txt)')
def predict(file_size, read_size, write_size, read_count, write_count, extension):
    """Predict optimal chunk size for a file with given characteristics."""
    logger = setup_logging("ml")
    
    som = BeeChunkerSOM()
    if not som.load():
        logger.error("Failed to load SOM model")
        sys.exit(1)
    
    # Prepare features for prediction
    features = {
        'file_size': file_size,
        'avg_read_size': read_size,
        'avg_write_size': write_size,
        'max_read_size': read_size * 2,
        'max_write_size': write_size * 2,
        'read_count': read_count,
        'write_count': write_count,
        'access_count': read_count + write_count,
        'read_write_ratio': read_count / (write_count + 1),
        'dir_depth': 3  # Default value
    }
    
    # Add file extension features
    common_extensions = ['.txt', '.csv', '.log', '.dat', '.bin', '.json', '.xml', '.db']
    for ext in common_extensions:
        features[f'ext_{ext}'] = 1 if extension == ext else 0
    features['ext_other'] = 1 if extension and extension not in common_extensions else 0
    
    # Predict chunk size
    chunk_size = som.predict(features)
    
    logger.info(f"Predicted chunk size: {chunk_size}KB")
    click.echo(f"Predicted optimal chunk size: {chunk_size}KB")

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()