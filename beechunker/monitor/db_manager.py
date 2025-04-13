"""
Database manager for BeeChunker access monitoring.
"""
import os
import sqlite3
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

class DBManager:
    """
    Manages the SQLite database for file access patterns.
    """
    
    def __init__(self, db_path):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger("beechunker.monitor.db_manager")
        
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    def initialize_db(self):
        """Initialize the database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create file_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_size INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL
                )
            """)
            
            # Create file_access table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_access (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    access_time REAL NOT NULL,
                    read_size INTEGER NOT NULL,
                    write_size INTEGER NOT NULL,
                    FOREIGN KEY (file_path) REFERENCES file_metadata (file_path)
                )
            """)
            
            # Create index on file_path and access_time
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_access_path_time
                ON file_access (file_path, access_time)
            """)
            
            # Create index on access_time for cleanup
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_access_time
                ON file_access (access_time)
            """)
            
            # Create throughput_metrics table for performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS throughput_metrics (
                    file_path TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    bytes_transferred INTEGER NOT NULL,
                    operation_type TEXT NOT NULL,
                    throughput_mbps REAL NOT NULL,
                    PRIMARY KEY (file_path, start_time)
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def add_access_event(self, event):
        """
        Add a file access event to the database.
        
        Args:
            event: FileAccessEvent object
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert file metadata
            cursor.execute("""
                INSERT INTO file_metadata (file_path, file_size, chunk_size, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_size = ?,
                    chunk_size = ?,
                    last_seen = ?
            """, (
                event.file_path, event.file_size, event.chunk_size, event.access_time, event.access_time,
                event.file_size, event.chunk_size, event.access_time
            ))
            
            # Insert access event
            cursor.execute("""
                INSERT INTO file_access (file_path, access_type, access_time, read_size, write_size)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event.file_path, event.access_type, event.access_time, event.read_size, event.write_size
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error adding access event: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def record_throughput(self, file_path, operation_type, start_time, end_time, bytes_transferred):
        """
        Record throughput metrics for a file operation.
        
        Args:
            file_path (str): Path to the file
            operation_type (str): Type of operation ("read" or "write")
            start_time (float): Start time of the operation (timestamp)
            end_time (float): End time of the operation (timestamp)
            bytes_transferred (int): Number of bytes transferred
        """
        try:
            duration = end_time - start_time
            if duration <= 0:
                return
            
            throughput_mbps = (bytes_transferred / duration) / (1024 * 1024)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO throughput_metrics 
                (file_path, start_time, end_time, bytes_transferred, operation_type, throughput_mbps)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                file_path, start_time, end_time, bytes_transferred, operation_type, throughput_mbps
            ))
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error recording throughput: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_file_access_patterns(self, file_path=None, days=30):
        """
        Get file access patterns from the database.
        
        Args:
            file_path (str, optional): Path to specific file, or None for all files
            days (int): Number of days of history to include
            
        Returns:
            list: List of access pattern dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            if file_path:
                # Query for a specific file
                cursor.execute("""
                    SELECT 
                        m.file_path,
                        m.file_size,
                        m.chunk_size,
                        COUNT(a.id) as access_count,
                        AVG(a.read_size) as avg_read_size,
                        AVG(a.write_size) as avg_write_size,
                        MAX(a.read_size) as max_read_size,
                        MAX(a.write_size) as max_write_size,
                        COUNT(CASE WHEN a.access_type = 'read' THEN 1 END) as read_count,
                        COUNT(CASE WHEN a.access_type = 'write' THEN 1 END) as write_count,
                        AVG(t.throughput_mbps) as avg_throughput
                    FROM file_metadata m
                    LEFT JOIN file_access a ON m.file_path = a.file_path AND a.access_time > ?
                    LEFT JOIN throughput_metrics t ON m.file_path = t.file_path AND t.start_time > ?
                    WHERE m.file_path = ?
                    GROUP BY m.file_path
                """, (cutoff_time, cutoff_time, file_path))
            else:
                # Query for all files
                cursor.execute("""
                    SELECT 
                        m.file_path,
                        m.file_size,
                        m.chunk_size,
                        COUNT(a.id) as access_count,
                        AVG(a.read_size) as avg_read_size,
                        AVG(a.write_size) as avg_write_size,
                        MAX(a.read_size) as max_read_size,
                        MAX(a.write_size) as max_write_size,
                        COUNT(CASE WHEN a.access_type = 'read' THEN 1 END) as read_count,
                        COUNT(CASE WHEN a.access_type = 'write' THEN 1 END) as write_count,
                        AVG(t.throughput_mbps) as avg_throughput
                    FROM file_metadata m
                    LEFT JOIN file_access a ON m.file_path = a.file_path AND a.access_time > ?
                    LEFT JOIN throughput_metrics t ON m.file_path = t.file_path AND t.start_time > ?
                    GROUP BY m.file_path
                """, (cutoff_time, cutoff_time))
            
            results = []
            for row in cursor.fetchall():
                results.append(dict(row))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting access patterns: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def export_training_data(self, output_csv=None, days=30):
        """
        Export training data for the SOM model.
        
        Args:
            output_csv (str, optional): Path to output CSV file
            days (int): Number of days of history to include
            
        Returns:
            str or None: Path to the CSV file if successful, None otherwise
        """
        try:
            import pandas as pd
            
            # Get access patterns
            patterns = self.get_file_access_patterns(days=days)
            
            if not patterns:
                self.logger.warning("No access patterns found for training data export")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(patterns)
            
            # Fill missing values
            df = df.fillna(0)
            
            # Add derived features
            df['read_write_ratio'] = df['read_count'] / (df['write_count'] + 1)  # Avoid division by zero
            
            # Add file extension
            df['file_extension'] = df['file_path'].apply(lambda x: os.path.splitext(x)[1].lower())
            
            # Add directory depth
            df['dir_depth'] = df['file_path'].apply(lambda x: len(x.split(os.sep)))
            
            # Generate output path if not provided
            if not output_csv:
                output_dir = os.path.join(os.path.dirname(self.db_path), "exports")
                os.makedirs(output_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_csv = os.path.join(output_dir, f"training_data_{timestamp}.csv")
            
            # Save to CSV
            df.to_csv(output_csv, index=False)
            self.logger.info(f"Exported training data to {output_csv}")
            
            return output_csv
            
        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
            return None
            
    def cleanup_old_data(self, days=30):
        """
        Clean up old data from the database.
        
        Args:
            days (int): Number of days of data to keep
            
        Returns:
            int: Number of records deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            # Delete old access records
            cursor.execute("DELETE FROM file_access WHERE access_time < ?", (cutoff_time,))
            access_deleted = cursor.rowcount
            
            # Delete old throughput records
            cursor.execute("DELETE FROM throughput_metrics WHERE start_time < ?", (cutoff_time,))
            throughput_deleted = cursor.rowcount
            
            # Remove metadata for files with no recent access
            cursor.execute("""
                DELETE FROM file_metadata 
                WHERE file_path NOT IN (SELECT DISTINCT file_path FROM file_access)
            """)
            metadata_deleted = cursor.rowcount
            
            conn.commit()
            
            total_deleted = access_deleted + throughput_deleted + metadata_deleted
            self.logger.info(f"Cleaned up {total_deleted} old records")
            
            return total_deleted
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return 0
        finally:
            if 'conn' in locals():
                conn.close()