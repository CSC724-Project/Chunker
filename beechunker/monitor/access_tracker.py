"""
Tracks file access patterns for BeeGFS files.
"""
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional, List
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from beechunker.custom_types.events.file_access_event import FileAccessEvent

class BeeGFSEventHandler(FileSystemEventHandler):
    """Watchdog event handler for BeeGFS file access monitoring."""
    
    def __init__(self, db_manager):
        """Initialize the event handler with a database manager."""
        super().__init__()
        self.db_manager = db_manager
        self.logger = logging.getLogger("beechunker.monitor.event_handler")
    
    def on_modified(self, event):
        """Handle file modified events."""
        if event.is_directory:
            return
        
        self.logger.debug(f"File modified: {event.src_path}")
        self._process_event(event.src_path, "write")
    
    def on_created(self, event):
        """Handle file created events."""
        if event.is_directory:
            return
        
        self.logger.debug(f"File created: {event.src_path}")
        self._process_event(event.src_path, "write")
    
    def on_opened(self, event):
        """Handle file opened events."""
        if event.is_directory or not hasattr(event, 'src_path'):
            return
        
        self.logger.debug(f"File opened: {event.src_path}")
        self._process_event(event.src_path, "read")
    
    def on_accessed(self, event):
        """Handle file accessed events."""
        if event.is_directory:
            return
        
        self.logger.debug(f"File accessed: {event.src_path}")
        self._process_event(event.src_path, "read")
    
    def _process_event(self, file_path, access_type):
        """Process a file event and create an access event."""
        try:
            if not os.path.exists(file_path):
                return
            
            # Get file metadata
            file_size = os.path.getsize(file_path)
            
            # Get chunk size
            chunk_size = self._get_chunk_size(file_path)
            
            # Default read/write sizes
            read_size = 4096 if access_type == "read" else 0
            write_size = 4096 if access_type == "write" else 0
            
            # Create access event
            access_event = FileAccessEvent(
                file_path=file_path,
                file_size=file_size,
                chunk_size=chunk_size,
                access_type=access_type,
                access_time=time.time(),
                read_size=read_size,
                write_size=write_size
            )
            
            # Store in database
            self.db_manager.add_access_event(access_event)
            
        except Exception as e:
            self.logger.error(f"Error processing event for {file_path}: {e}")
    
    def _get_chunk_size(self, file_path):
        """Get the chunk size of a file in KB from BeeGFS."""
        try:
            # Use beegfs-ctl to get chunk size
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

class FileSystemMonitor:
    """
    Monitors file system operations in BeeGFS.
    Uses watchdog for monitoring file system events.
    """
    
    def __init__(self, mount_points, db_manager):
        """
        Initialize the file system monitor.
        
        Args:
            mount_points (list): List of BeeGFS mount points to monitor
            db_manager: Database manager for storing access events
        """
        self.mount_points = mount_points
        self.db_manager = db_manager
        self.logger = logging.getLogger("beechunker.monitor.fs_monitor")
        self.observers = []
        self.event_handler = None
        
        # Validate mount points
        self._validate_mount_points()
    
    def _validate_mount_points(self):
        """Validate that mount points exist and are BeeGFS mounts."""
        valid_mount_points = []
        
        for mount_point in self.mount_points:
            if not os.path.exists(mount_point):
                self.logger.warning(f"Mount point does not exist: {mount_point}")
                continue
            
            # Check if it's a BeeGFS mount (simplified)
            try:
                with open("/proc/mounts", "r") as f:
                    mounts = f.read()
                    if mount_point in mounts and "beegfs" in mounts:
                        valid_mount_points.append(mount_point)
                    else:
                        self.logger.warning(f"Not a BeeGFS mount: {mount_point}")
            except Exception as e:
                self.logger.error(f"Error validating mount point {mount_point}: {e}")
        
        self.mount_points = valid_mount_points
        if not valid_mount_points:
            self.logger.error("No valid BeeGFS mount points found")
    
    def start_monitoring(self):
        """
        Start monitoring file system events using watchdog.
        Sets up observers for each mount point.
        """
        self.logger.info(f"Starting monitoring on {len(self.mount_points)} mount points")
        
        if not self.mount_points:
            self.logger.error("No valid mount points to monitor")
            return False
        
        # Create event handler
        self.event_handler = BeeGFSEventHandler(self.db_manager)
        
        # Create observers for each mount point
        for mount_point in self.mount_points:
            try:
                observer = Observer()
                observer.schedule(self.event_handler, mount_point, recursive=True)
                observer.start()
                self.observers.append(observer)
                self.logger.info(f"Started monitoring {mount_point}")
            except Exception as e:
                self.logger.error(f"Error starting observer for {mount_point}: {e}")
        
        if not self.observers:
            self.logger.error("Failed to start any observers")
            return False
        
        self.logger.info(f"Successfully started {len(self.observers)} observers")
        return True
    
    def stop_monitoring(self):
        """Stop monitoring file system events by stopping all observers."""
        self.logger.info("Stopping file system monitoring")
        
        for observer in self.observers:
            try:
                observer.stop()
                self.logger.debug("Stopped observer")
            except Exception as e:
                self.logger.error(f"Error stopping observer: {e}")
        
        # Wait for all observers to complete
        for observer in self.observers:
            try:
                observer.join(timeout=5.0)
                if observer.is_alive():
                    self.logger.warning("Observer did not terminate cleanly")
            except Exception as e:
                self.logger.error(f"Error joining observer thread: {e}")
        
        # Clear the observers list
        self.observers = []
        self.logger.info("All file system monitoring stopped")
    
    def is_monitoring(self):
        """Check if monitoring is active."""
        return any(observer.is_alive() for observer in self.observers)
    
    def process_event(self, event_type, file_path, size=None):
        """
        Process a file system event manually.
        
        Args:
            event_type (str): Type of event ("read" or "write")
            file_path (str): Path to the file
            size (int, optional): Size of the read or write
        """
        # This can be used for manually processing events if needed
        if self.event_handler:
            self.event_handler._process_event(file_path, event_type)