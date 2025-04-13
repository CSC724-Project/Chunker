"""Command line interface for monitoring Beechunkers."""
import os
import sys
import time
import click
import sqlite3
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from beechunker.common.beechunker_logging import setup_logging
from beechunker.common.config import config
from beechunker.monitor.access_tracker import FileAccessEvent
from beechunker.monitor.db_manager import DBManager

class FileMonitorHandler(FileSystemEventHandler):
    """Watchdog event handler for file access monitoring."""
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.logger = logging.getLogger("beechunker.monitor.handler")
    
    def on_modified(self, event):
        """Handle file modified events."""
        if event.is_directory:
            return
        
        self.logger.info(f"File modified: {event.src_path}")
        
        try:
            # Get the file metadata
            file_size = os.path.getsize(event.src_path)
            
            # Get chunk size 
            chunk_size = self._get_chunk_size(event.src_path)
            
            # Create access event
            access_event = FileAccessEvent(
                file_path=event.src_path,
                file_size=file_size,
                chunk_size=chunk_size,
                access_type="write",
                access_time=time.time(),
                read_size=0, # Unknown for basic watchdog
                write_size=4096, # Default estimate
            )
            
            # Store the access event in the database
            self.db_manager.add_access_event(access_event)
            
        except Exception as e:
            self.logger.error(f"Error processing file modified event: {e}")
        
    
    def on_opened(self, event):
        """Handle file opened events."""
        if event.is_directory:
            return # Ignore directories
        
        self.logger.debug(f"File opened: {event.src_path}")
        
        try:
            # Get the file metadata
            file_size = os.path.getsize(event.src_path)
            
            # Get chunk size (implement actual BeeGFS-specific logic here)
            chunk_size = self._get_chunk_size(event.src_path)
            
            # Create access event
            access_event = FileAccessEvent(
                file_path=event.src_path,
                file_size=file_size,
                chunk_size=chunk_size,
                access_type="read",  # Assuming read for open
                access_time=time.time(),
                read_size=4096,  # Default estimate
                write_size=0  # Unknown for basic watchdog
            )
            
            # Store in database
            self.db_manager.add_access_event(access_event)
            
        except Exception as e:
            self.logger.error(f"Error processing file opened event: {e}")
    
    def _get_chunk_size(self, file_path):
        """Get the chunk size of a file in KB."""
        try:
            # This is a simplified approach - in a real implementation, you would
            # use BeeGFS tools to get the actual chunk size
            import subprocess
            
            result = subprocess.run(
                ["beegfs-ctl", "--getentryinfo", file_path],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.splitlines():
                if "ChunkSize:" in line:
                    chunk_info = line.strip().split(":")[-1].strip()
                    
                    # Convert to KB
                    if chunk_info.endswith("K"):
                        return int(chunk_info[:-1])
                    elif chunk_info.endswith("M"):
                        return int(chunk_info[:-1]) * 1024
                    elif chunk_info.endswith("G"):
                        return int(chunk_info[:-1]) * 1024 * 1024
                    else:
                        # Assume bytes if no unit
                        return int(chunk_info) // 1024
            
            # Default if not found
            return 512  # 512KB default
            
        except Exception as e:
            self.logger.error(f"Error getting chunk size for {file_path}: {e}")
            return 512  # Default if error
        

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

def main():
    """Main entry point for the monitor CLI."""
    cli()

if __name__ == '__main__':
    main()