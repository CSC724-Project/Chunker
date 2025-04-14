import logging
import os
import time

from beechunker.monitor.access_tracker import FileAccessEvent
from watchdog.events import FileSystemEventHandler

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
        
