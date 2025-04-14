"""Command line interface for monitoring Beechunker."""
import os
import sys
import time
import click
import sqlite3
from pathlib import Path
from watchdog.observers import Observer

from beechunker.common.beechunker_logging import setup_logging
from beechunker.common.config import config
from beechunker.monitor.db_manager import DBManager
from beechunker.optimizer.file_watcher import FileMonitorHandler

@click.group()
def cli():
    """BeeChunker monitoring command-line interface."""
    pass

@cli.command()
def run():
    """Run the monitor service continuously."""
    logger = setup_logging("monitor")
    logger.info("Starting BeeChunker monitor service")
    
    # Initialize database
    db_path = config.get("monitor", "db_path")
    logger.info(f"Using database at {db_path}")
    
    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize database manager
    db_manager = DBManager(db_path)
    db_manager.initialize_db()
    
    # Get mount points from config
    mount_points = config.get("beegfs", "mount_points")
    if not mount_points:
        logger.error("No BeeGFS mount points configured. Exiting.")
        sys.exit(1)
    
    logger.info(f"Monitoring mount points: {mount_points}")
    
    # Set up file system event handler
    event_handler = FileMonitorHandler(db_manager)
    
    # Set up observers for each mount point
    observers = []
    for mount_point in mount_points:
        if not os.path.exists(mount_point):
            logger.warning(f"Mount point does not exist: {mount_point}")
            continue
            
        observer = Observer()
        observer.schedule(event_handler, mount_point, recursive=True)
        observer.start()
        observers.append(observer)
        logger.info(f"Started monitoring {mount_point}")
    
    if not observers:
        logger.error("No valid mount points to monitor. Exiting.")
        sys.exit(1)
    
    try:
        # Keep the service running
        while True:
            time.sleep(60)  # Check every minute if observers are still running
            for observer in observers:
                if not observer.is_alive():
                    logger.error("Observer has died. Restarting service.")
                    for obs in observers:
                        if obs.is_alive():
                            obs.stop()
                    sys.exit(1)  # Let systemd restart the service
    
    except KeyboardInterrupt:
        logger.info("Monitor service stopped by user")
        for observer in observers:
            observer.stop()
    
    for observer in observers:
        observer.join()

@cli.command()
@click.option('--days', '-d', type=int, default=7, help='Number of days of data to keep')
def cleanup(days):
    """Clean up old monitoring data."""
    logger = setup_logging("monitor")
    
    # Initialize database
    db_path = config.get("monitor", "db_path")
    logger.info(f"Using database at {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Calculate cutoff time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Delete old records
        cursor.execute("DELETE FROM file_access WHERE access_time < ?", (cutoff_time,))
        deleted_count = cursor.rowcount
        
        # Commit changes
        conn.commit()
        
        logger.info(f"Deleted {deleted_count} records older than {days} days")
        
    except Exception as e:
        logger.error(f"Error cleaning up database: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

@cli.command()
def stats():
    """Show monitoring statistics."""
    logger = setup_logging("monitor")
    
    # Initialize database
    db_path = config.get("monitor", "db_path")
    logger.info(f"Using database at {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute("SELECT COUNT(*) as count FROM file_access")
        total_records = cursor.fetchone()['count']
        
        # Get total files
        cursor.execute("SELECT COUNT(DISTINCT file_path) as count FROM file_access")
        total_files = cursor.fetchone()['count']
        
        # Get read/write counts
        cursor.execute("SELECT access_type, COUNT(*) as count FROM file_access GROUP BY access_type")
        access_types = {row['access_type']: row['count'] for row in cursor.fetchall()}
        
        # Get recent activity
        recent_cutoff = time.time() - (24 * 60 * 60)  # Last 24 hours
        cursor.execute("SELECT COUNT(*) as count FROM file_access WHERE access_time > ?", (recent_cutoff,))
        recent_activity = cursor.fetchone()['count']
        
        # Print statistics
        click.echo(f"Total access events: {total_records}")
        click.echo(f"Total unique files: {total_files}")
        click.echo(f"Read events: {access_types.get('read', 0)}")
        click.echo(f"Write events: {access_types.get('write', 0)}")
        click.echo(f"Recent activity (24h): {recent_activity}")
        
        # Get top accessed files
        cursor.execute("""
            SELECT file_path, COUNT(*) as count 
            FROM file_access 
            GROUP BY file_path 
            ORDER BY count DESC 
            LIMIT 5
        """)
        click.echo("\nTop accessed files:")
        for row in cursor.fetchall():
            click.echo(f"  {row['file_path']}: {row['count']} accesses")
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

@cli.command()
def get_chunk_size():
    """Get the chunk size of a file."""
    logger = setup_logging("monitor")
    
    # Initialize database
    db_path = config.get("monitor", "db_path")
    logger.info(f"Using database at {db_path}")
    
    if not os.path.exists(db_path):
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get chunk size for a specific file
        file_path = input("Enter the file path: ")
        cursor.execute("SELECT chunk_size FROM file_metadata WHERE file_path = ?", (file_path,))
        row = cursor.fetchone()
        
        if row:
            click.echo(f"Chunk size for {file_path}: {row[0]} KB")
        else:
            click.echo(f"No chunk size information found for {file_path}")
        
    except Exception as e:
        logger.error(f"Error getting chunk size: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main entry point for the monitor CLI."""
    cli()

if __name__ == '__main__':
    main()