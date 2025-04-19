"""Command-line interface for SOM training."""
import os
import sys
import click
import pandas as pd
import logging

from beechunker.common.beechunker_logging import setup_logging
from beechunker.common.config import config
from beechunker.ml.som import BeeChunkerSOM
from beechunker.ml.xgboost import BeeChunkerXGBoost

@click.group()
def cli():
    """BeeChunker SOM trainer command-line interface."""
    pass

@cli.command()
@click.option('--model', type=click.Choice(['som', 'xgboost'], case_sensitive=False), default='som',
              help='Model to use for prediction')
@click.option('--file-size', '-s', type=int, required=True, help='File size in bytes')
@click.option('--read-size', '-r', type=int, default=4096, help='Average read size')
@click.option('--write-size', '-w', type=int, default=4096, help='Average write size')
@click.option('--read-count', '-rc', type=int, default=10, help='Estimated read operations')
@click.option('--write-count', '-wc', type=int, default=5, help='Estimated write operations')
@click.option('--extension', '-e', default='', help='File extension (e.g., .txt)')
def predict(model, file_size, read_size, write_size, read_count, write_count, extension):
    """Predict optimal chunk size for a file with given characteristics."""
    logger = setup_logging("ml")
    
    if model == 'som':
        model_instance = BeeChunkerSOM()
    else:
        model_instance = BeeChunkerXGBoost()
        
    if not model_instance.load():
        logger.error(f"Failed to load {model} model")
        sys.exit(1)
    
    # Create a dummy file path based on extension
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    file_path = f'/dummy/path/file{extension}'
    
    # Create a DataFrame for a single prediction
    df = pd.DataFrame([{
        'file_path': file_path,
        'file_size': file_size,
        'chunk_size': 1048576,  # Default 1MB chunk size
        'access_count': read_count + write_count,
        'avg_read_size': read_size,
        'avg_write_size': write_size,
        'max_read_size': read_size * 2,
        'max_write_size': write_size * 2,
        'read_count': read_count,
        'write_count': write_count,
        'throughput_mbps': 100.0  # Default throughput
    }])
    
    # Predict chunk size
    predictions = model_instance.predict(df)
    
    if predictions is not None and not predictions.empty:
        chunk_size = predictions.iloc[0]['predicted_chunk_size']
        logger.info(f"Predicted chunk size ({model}): {chunk_size}KB")
        click.echo(f"Predicted optimal chunk size: {chunk_size}KB")
    else:
        logger.error("Failed to predict chunk size")
        click.echo("Failed to predict chunk size. See log for details.")
        sys.exit(1)

@cli.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--model', type=click.Choice(['som', 'xgboost'], case_sensitive=False), default='som',
              help='Model to train')
def train(data_path, model):
    """Train a model using the provided data."""
    logger = setup_logging("ml")
    
    try:
        # Load training data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from {data_path}")
        
        # Initialize and train the model
        if model == 'som':
            model_instance = BeeChunkerSOM()
        else:
            model_instance = BeeChunkerXGBoost()
            
        if model_instance.train(df):
            logger.info(f"Successfully trained {model} model")
            click.echo(f"Successfully trained {model} model")
        else:
            logger.error(f"Failed to train {model} model")
            click.echo(f"Failed to train {model} model. See log for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error training model: {e}")
        click.echo(f"Error: {str(e)}")
        sys.exit(1)

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
            COUNT(CASE WHEN a.access_type = 'write' THEN 1 END) as write_count,
            SUM(CASE WHEN a.access_type = 'read' THEN a.bytes_transferred ELSE 0 END) / 
            NULLIF(SUM(CASE WHEN a.access_type = 'read' THEN a.elapsed_time ELSE 0 END), 0) as throughput_mbps
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
@click.option('--input-csv', '-i', required=True, help='Input CSV file with access pattern data')
@click.option('--output-csv', '-o', help='Output CSV file to save predictions')
def batch_predict(input_csv, output_csv):
    """Predict chunk sizes for a batch of files from a CSV."""
    logger = setup_logging("ml")
    
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        sys.exit(1)
    
    logger.info(f"Loading data from {input_csv}")
    df = pd.read_csv(input_csv)
    
    som = BeeChunkerSOM()
    if not som.load():
        logger.error("Failed to load SOM model")
        sys.exit(1)
    
    logger.info(f"Predicting chunk sizes for {len(df)} files")
    predictions = som.predict(df)
    
    if predictions is not None and not predictions.empty:
        if output_csv:
            predictions.to_csv(output_csv, index=False)
            logger.info(f"Saved predictions to {output_csv}")
            click.echo(f"Predictions saved to {output_csv}")
        
        # Generate visualizations
        logger.info("Generating prediction visualizations")
        som.visualize_predictions(predictions)
        click.echo("Prediction visualizations created")
        
        # Print summary
        current_sizes = df['chunk_size'].sum() / 1024  # KB
        predicted_sizes = predictions['predicted_chunk_size'].sum()
        
        click.echo("\nStorage Impact Analysis:")
        click.echo(f"Total current chunk size allocation: {current_sizes:.0f} KB")
        click.echo(f"Total predicted chunk size allocation: {predicted_sizes:.0f} KB")
        
        if predicted_sizes < current_sizes:
            savings = (current_sizes - predicted_sizes) / current_sizes * 100
            click.echo(f"Potential storage reduction: {savings:.2f}%")
        else:
            increase = (predicted_sizes - current_sizes) / current_sizes * 100
            click.echo(f"Storage increase for better performance: {increase:.2f}%")
    else:
        logger.error("Failed to predict chunk sizes")
        click.echo("Failed to predict chunk sizes. See log for details.")
        sys.exit(1)

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == '__main__':
    main()