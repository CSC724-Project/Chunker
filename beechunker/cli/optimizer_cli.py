import time
import click
import os
import sys
import sqlite3

from beechunker.common.beechunker_logging import setup_logging
from beechunker.optimizer.chunk_manager import ChunkSizeOptimizer
from beechunker.common.config import config

@click.group()
def cli():
    """BeeChunker optimizer command-line interface."""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--force', '-f', is_flag=True, help='Force optimization even if current size is optimal')
@click.option('--dry-run', '-d', is_flag=True, help='Do not actually change chunk sizes')
def optimize_file(file_path, force, dry_run):
    """Optimize chunk size for a single file."""
    logger = setup_logging("optimizer")
    
    try:
        optimizer = ChunkSizeOptimizer()
        
        if optimizer.optimize_file(file_path, force=force, dry_run=dry_run):
            click.echo(f"Successfully optimized {file_path}")
        else:
            click.echo(f"Failed to optimize {file_path}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--recursive', '-r', is_flag=True, help='Recursively process subdirectories')
@click.option('--force', '-f', is_flag=True, help='Force optimization even if current size is optimal')
@click.option('--dry-run', '-d', is_flag=True, help='Do not actually change chunk sizes')
@click.option('--file-type', '-t', multiple=True, help='File extensions to include (e.g., .txt)')
@click.option('--min-size', type=int, help='Minimum file size in bytes')
@click.option('--max-size', type=int, help='Maximum file size in bytes')
def optimize_dir(directory, recursive, force, dry_run, file_type, min_size, max_size):
    """Optimize chunk sizes for all files in a directory."""
    logger = setup_logging("optimizer")
    
    try:
        optimizer = ChunkSizeOptimizer()
        file_types = list(file_type) if file_type else None
        
        optimized, failed, skipped = optimizer.optimize_directory(
            directory, 
            recursive=recursive,
            force=force,
            dry_run=dry_run,
            file_types=file_types,
            min_size=min_size,
            max_size=max_size
        )
        
        click.echo(f"Optimization complete: {optimized} files optimized, {failed} failed, {skipped} skipped")
        
        if failed > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--min-size', type=int, help='Minimum file size in bytes')
@click.option('--max-size', type=int, help='Maximum file size in bytes')
@click.option('--min-access', type=int, help='Minimum number of accesses')
@click.option('--extension', '-e', multiple=True, help='File extensions to include')
@click.option('--limit', '-l', type=int, help='Maximum number of files to process')
@click.option('--dry-run', '-d', is_flag=True, help='Do not actually change chunk sizes')
def bulk_optimize(min_size, max_size, min_access, extension, limit, dry_run):
    """Bulk optimize files based on database query."""
    logger = setup_logging("optimizer")
    
    try:
        optimizer = ChunkSizeOptimizer()
        
        # Build query params
        query_params = {}
        if min_size:
            query_params['min_size'] = min_size
        if max_size:
            query_params['max_size'] = max_size
        if min_access:
            query_params['min_access'] = min_access
        if extension:
            query_params['extensions'] = extension
        
        # Run bulk optimization
        optimized, failed = optimizer.bulk_optimize(query_params, dry_run=dry_run, limit=limit)
        
        click.echo(f"Bulk optimization complete: {optimized} files optimized, {failed} failed")
        
        if failed > 0:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def analyze(file_path):
    """Analyze a file and show predicted optimal chunk size without changing it."""
    logger = setup_logging("optimizer")
    
    try:
        optimizer = ChunkSizeOptimizer()
        
        # Get current chunk size
        current_size = optimizer.get_current_chunk_size(file_path)
        if current_size is None:
            click.echo(f"Could not determine current chunk size for {file_path}")
            sys.exit(1)
        
        # Get file features
        features = optimizer.get_file_features(file_path)
        
        # Predict optimal chunk size
        optimal_size = optimizer.predict_chunk_size(file_path)
        
        # Print analysis
        click.echo(f"File: {file_path}")
        click.echo(f"Size: {os.path.getsize(file_path) // 1024} KB")
        click.echo(f"Current chunk size: {current_size} KB")
        click.echo(f"Optimal chunk size: {optimal_size} KB")
        
        if current_size == optimal_size:
            click.echo("Status: Already optimal")
        elif current_size < optimal_size:
            click.echo(f"Recommendation: Increase chunk size ({current_size}KB -> {optimal_size}KB)")
        else:
            click.echo(f"Recommendation: Decrease chunk size ({current_size}KB -> {optimal_size}KB)")
        
        # Print feature importance
        click.echo("\nKey features for prediction:")
        important_features = [
            ('file_size', 'File size', 'bytes'),
            ('avg_read_size', 'Average read size', 'bytes'),
            ('avg_write_size', 'Average write size', 'bytes'),
            ('read_count', 'Read operations', 'count'),
            ('write_count', 'Write operations', 'count'),
            ('read_write_ratio', 'Read/write ratio', '')
        ]
        
        for key, name, unit in important_features:
            value = features.get(key, 'N/A')
            if unit == 'bytes' and isinstance(value, (int, float)):
                if value >= 1048576:
                    value_str = f"{value/1048576:.2f} MB"
                elif value >= 1024:
                    value_str = f"{value/1024:.2f} KB"
                else:
                    value_str = f"{value} bytes"
            elif unit == 'count' and isinstance(value, (int, float)):
                value_str = f"{int(value)}"
            else:
                value_str = f"{value}"
                
            click.echo(f"  {name}: {value_str}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)

@cli.command()
def get_stats():
    """Show optimization statistics."""
    logger = setup_logging("optimizer")
    
    try:
        db_path = config.get("monitor", "db_path")
        
        if not os.path.exists(db_path):
            click.echo(f"Database not found: {db_path}")
            sys.exit(1)
        
        # Connect to database
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if optimization history table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='optimization_history'")
            if not cursor.fetchone():
                click.echo("No optimization history found")
                return
            
            # Get optimization counts
            cursor.execute("SELECT COUNT(*) as count FROM optimization_history")
            total_optimizations = cursor.fetchone()['count']
            
            # Get recent optimizations
            recent_cutoff = time.time() - (24 * 60 * 60)  # Last 24 hours
            cursor.execute("SELECT COUNT(*) as count FROM optimization_history WHERE optimization_time > ?", (recent_cutoff,))
            recent_optimizations = cursor.fetchone()['count']
            
            # Get chunk size statistics
            cursor.execute("""
                SELECT 
                    AVG(old_chunk_size) as avg_old,
                    AVG(new_chunk_size) as avg_new,
                    AVG(new_chunk_size - old_chunk_size) as avg_change
                FROM optimization_history
            """)
            stats = cursor.fetchone()
            
            # Get increase/decrease counts
            cursor.execute("SELECT COUNT(*) as count FROM optimization_history WHERE new_chunk_size > old_chunk_size")
            increases = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM optimization_history WHERE new_chunk_size < old_chunk_size")
            decreases = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM optimization_history WHERE new_chunk_size = old_chunk_size")
            unchanged = cursor.fetchone()['count']
            
            # Print statistics
            click.echo(f"Total optimizations: {total_optimizations}")
            click.echo(f"Recent optimizations (24h): {recent_optimizations}")
            click.echo(f"Average old chunk size: {stats['avg_old']:.2f} KB")
            click.echo(f"Average new chunk size: {stats['avg_new']:.2f} KB")
            click.echo(f"Average chunk size change: {stats['avg_change']:.2f} KB")
            click.echo(f"Chunk size increases: {increases}")
            click.echo(f"Chunk size decreases: {decreases}")
            click.echo(f"Chunk size unchanged: {unchanged}")
            
            # Get top optimized files
            cursor.execute("""
                SELECT 
                    file_path, 
                    old_chunk_size,
                    new_chunk_size,
                    (new_chunk_size - old_chunk_size) as change
                FROM optimization_history
                ORDER BY ABS(change) DESC
                LIMIT 5
            """)
            
            if cursor.fetchone():  # Rewind cursor
                cursor.execute("""
                    SELECT 
                        file_path, 
                        old_chunk_size,
                        new_chunk_size,
                        (new_chunk_size - old_chunk_size) as change
                    FROM optimization_history
                    ORDER BY ABS(change) DESC
                    LIMIT 5
                """)
                
                click.echo("\nTop optimizations by change magnitude:")
                for row in cursor.fetchall():
                    change_str = f"+{row['change']}" if row['change'] > 0 else f"{row['change']}"
                    click.echo(f"  {row['file_path']}: {row['old_chunk_size']} KB -> {row['new_chunk_size']} KB ({change_str} KB)")
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            click.echo(f"Error: {e}")
            sys.exit(1)
        finally:
            if 'conn' in locals():
                conn.close()
                
    except Exception as e:
        logger.error(f"Error: {e}")
        click.echo(f"Error: {e}")
        sys.exit(1)

def main():
    """Main entry point for the optimizer CLI."""
    cli()

if __name__ == '__main__':
    main()