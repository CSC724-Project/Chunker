"""
Tracks file access patterns for BeeGFS files.
"""
import os
import time
from dataclasses import dataclass
from typing import Literal, Optional
import logging

@dataclass
class FileAccessEvent:
    """Data class for file access events."""
    file_path: str
    file_size: int
    chunk_size: int
    access_type: Literal["read", "write"]
    access_time: float
    read_size: int = 0
    write_size: int = 0
    
    def __post_init__(self):
        """Validate event after initialization."""
        if self.access_type not in ["read", "write"]:
            raise ValueError(f"Invalid access type: {self.access_type}")
        
        if self.file_size < 0:
            raise ValueError(f"Invalid file size: {self.file_size}")
        
        if self.chunk_size < 0:
            raise ValueError(f"Invalid chunk size: {self.chunk_size}")

class FileSystemMonitor:
    """
    Monitors file system operations in BeeGFS.
    This is a more advanced implementation that could use Linux inotify/fanotify
    or BeeGFS-specific monitoring tools for more detailed access tracking.
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
        Start monitoring file system events.
        In a production environment, this would use inotify/fanotify or BeeGFS tools.
        """
        self.logger.info(f"Starting monitoring on {len(self.mount_points)} mount points")
        
        # Set up inotify or fanotify or BeeGFS-specific monitoring
        # This is a placeholder for actual monitoring logic
        
        
        self.logger.info("Using watchdog for file monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring file system events."""
        self.logger.info("Stopping file system monitoring")
        
        # Clean up any resources
        
    def process_event(self, event_type, file_path, size=None):
        """
        Process a file system event.
        
        Args:
            event_type (str): Type of event ("read" or "write")
            file_path (str): Path to the file
            size (int, optional): Size of the read or write
        """
        try:
            if not os.path.exists(file_path):
                return
            
            # Get file metadata
            file_size = os.path.getsize(file_path)
            
            # Get chunk size (simplified)
            chunk_size = 512  # Default
            try:
                import subprocess
                result = subprocess.run(
                    ["beegfs-ctl", "--getentryinfo", file_path],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.splitlines():
                    if "ChunkSize:" in line:
                        chunk_info = line.strip().split(":")[-1].strip()
                        if chunk_info.endswith("K"):
                            chunk_size = int(chunk_info[:-1])
                        elif chunk_info.endswith("M"):
                            chunk_size = int(chunk_info[:-1]) * 1024
                        elif chunk_info.endswith("G"):
                            chunk_size = int(chunk_info[:-1]) * 1024 * 1024
                        else:
                            chunk_size = int(chunk_info) // 1024
            except Exception as e:
                self.logger.debug(f"Error getting chunk size for {file_path}: {e}")
            
            # Create event
            if size is None:
                size = 4096  # Default
                
            read_size = size if event_type == "read" else 0
            write_size = size if event_type == "write" else 0
            
            event = FileAccessEvent(
                file_path=file_path,
                file_size=file_size,
                chunk_size=chunk_size,
                access_type=event_type,
                access_time=time.time(),
                read_size=read_size,
                write_size=write_size
            )
            
            # Store in database
            self.db_manager.add_access_event(event)
            
        except Exception as e:
            self.logger.error(f"Error processing event for {file_path}: {e}")