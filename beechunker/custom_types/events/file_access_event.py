from dataclasses import dataclass
from typing import Literal

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
