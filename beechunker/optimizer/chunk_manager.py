import os
import re
import sqlite3
import subprocess
import time
from typing import Optional
from beechunker.common.beechunker_logging import setup_logging
import logging
from beechunker.ml.som import BeeChunkerSOM
from beechunker.common.config import config
from beechunker.monitor.db_manager import DBManager

class ChunkSizeOptimizer:
    """Manages chunk size optimization for BeeGFS files."""
    def __init__(self):
        """Initialize the chunk size optimizer."""
        self.logger = logging.getLogger("beechunker.optimizer")
        self.som = BeeChunkerSOM()
        
        # Load the SOM model
        if not self.som.load():
            self.logger.error("Failed to load the SOM model. Make sure the model has been trained.")
            raise RuntimeError("Failed to load the SOM model.")
        
        self.db_path = config.get("monitor", "db_path")
        self.db_manager = DBManager(self.db_path)
        
        # Load min/max chunk sizes from config
        self.min_chunk_size = config.get("optimizer", "min_chunk_size")
        self.max_chunk_size = config.get("optimizer", "max_chunk_size")
        
        self.logger.info("ChunkSizeOptimizer initialized.")
        
    def get_file_features(self, file_path) -> dict: # kept the return type as dict for simplicity for now
        """Extract features from the file for chunk size optimization (prediction using the SOM model)."""
        try:
            # Get the file metadata
            file_size = os.path.getsize(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            dir_depth = len(file_path.split('/'))
            
            # Get access patterns from db if available
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
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
                    AVG(t.throughput_mbps) as throughput_mbps
                FROM file_metadata m
                LEFT JOIN file_access a ON m.file_path = a.file_path
                LEFT JOIN throughput_metrics t ON m.file_path = t.file_path
                WHERE m.file_path = ?
                GROUP BY m.file_path
            """, (file_path,))
            
            row = cursor.fetchone()
            conn.close()
            
            # Default values if no data is found
            if not row or row[0]==0:
                self.logger.warning(f"No access data found for {file_path}. Using default values.")
                # Default values based on the file size
                if file_size < 1024 * 1024: # 1 MB
                    avg_read_size = 4096
                    avg_write_size = 4096
                    max_read_size = 8192
                    max_write_size = 8192
                    throughput_mbps = 50.0  # Default small file throughput
                elif file_size < 1024 * 1024 * 100: # 100 MB
                    avg_read_size = 65536
                    avg_write_size = 32768
                    max_read_size = 131072
                    max_write_size = 65536
                    read_count = 20
                    write_count = 10
                    throughput_mbps = 100.0  # Default medium file throughput
                else:  # Large files (>=100MB)
                    avg_read_size = 1048576
                    avg_write_size = 524288
                    max_read_size = 4194304
                    max_write_size = 1048576
                    read_count = 50
                    write_count = 20
                    throughput_mbps = 150.0  # Default large file throughput
                
                access_count = read_count + write_count
            else:
                file_path = row[0]
                file_size = row[1]
                chunk_size = row[2]
                access_count = row[3] or 1
                avg_read_size = row[4] or 4096
                avg_write_size = row[5] or 4096
                max_read_size = row[6] or avg_read_size * 2
                max_write_size = row[7] or avg_write_size * 2
                read_count = row[8] or 0
                write_count = row[9] or 0
                throughput_mbps = row[10] or 100.0

            # Calculate read/write ratio
            read_write_ratio = read_count / max(1, write_count)

            # Prepare features for prediction
            features = {
                'file_size': file_size,
                'avg_read_size': avg_read_size,
                'avg_write_size': avg_write_size,
                'max_read_size': max_read_size,
                'max_write_size': max_write_size,
                'read_count': read_count,
                'write_count': write_count,
                'access_count': access_count,
                'read_write_ratio': read_write_ratio,
                'dir_depth': dir_depth,
                'throughput_mbps': throughput_mbps  # Add the throughput feature
            }
            
            # Add file extension as a feature
            common_extensions = ['.txt', '.csv', '.log', '.dat', '.bin', '.json', '.xml', '.db']
            for ext in common_extensions:
                features[f'ext_{ext}'] = 1 if file_extension == ext else 0
            features['ext_other'] = 1 if file_extension and file_extension not in common_extensions else 0
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features from {file_path}: {e}")
            return None
    
    def predict_chunk_size(self, file_path) -> int:
        """Predict the optimal chunk size for a file using the SOM model."""
        try:
            # Get file features
            features = self.get_file_features(file_path)
            if features is None:
                raise ValueError(f"Failed to extract features from {file_path}")
                
            # Convert dictionary to DataFrame for SOM prediction
            import pandas as pd
            df = pd.DataFrame([features])
            df['file_path'] = file_path
            df['chunk_size'] = self.get_current_chunk_size(file_path) * 1024  # Convert KB to bytes to match expected format
            
            # Predict chunk size using the SOM model
            predictions = self.som.predict(df)
            
            if predictions is None or len(predictions) == 0:
                raise ValueError("SOM prediction failed")
                
            # Extract the predicted chunk size from the result
            optimal_chunk_size = int(predictions.iloc[0]['predicted_chunk_size'])
            
            # Log the predicted chunk size
            self.logger.info(f"Predicted chunk size for {file_path}: {optimal_chunk_size}KB")
            self.logger.info(f"Features used for prediction: {features}")
            
            return optimal_chunk_size
                
        except Exception as e:
            self.logger.error(f"Error predicting chunk size for {file_path}: {e}")
            raise
    
    def get_current_chunk_size(self, file_path) -> int:
        """Get the current chunk size of a file in KB from BeeGFS."""
        try:
            result = subprocess.run(
                ["beegfs-ctl", "--getentryinfo", file_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            current_chunk_size = None
            chunk_size_pattern = re.compile(r'[Cc]hunk[Ss]ize:\s*(\d+)([KMGkmg]?)')
            
            for line in result.stdout.splitlines():
                match = chunk_size_pattern.search(line)
                if match: # updated to use regex for better matching
                    chunk_info = line.strip().split(":")[-1].strip()
            
            # Parse the chunk size
                    if chunk_info.endswith("K"):
                        current_chunk_size = int(chunk_info[:-1])
                    elif chunk_info.endswith("M"):
                        current_chunk_size = int(chunk_info[:-1]) * 1024
                    elif chunk_info.endswith("G"):
                        current_chunk_size = int(chunk_info[:-1]) * 1024 * 1024
                    else:
                        # Assume bytes if no unit
                        current_chunk_size = int(chunk_info) // 1024
                    
                    break
            
            if current_chunk_size is None:
                self.logger.warning(f"Couldn't determine current chunk size for {file_path}")
                return None
                
            return current_chunk_size
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting current chunk size for {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting current chunk size for {file_path}: {e}")
            return None
        
    def set_chunk_size(self, file_path, chunk_size_kb) -> bool:
        """Set the chunk size of a file in BeeGFS by creating a new copy with the desired chunk size."""
        import time
        import os
        import subprocess

        temp_file_path = None
        backup_path = None

        try:
            chunk_size_kb = max(self.min_chunk_size, min(chunk_size_kb, self.max_chunk_size))
            chunk_size_bytes = chunk_size_kb * 1024
            temp_file_path = f"{file_path}.tmp_{int(time.time())}"

            # Clean up any existing temporary file first
            if os.path.exists(temp_file_path):
                self.logger.warning(f"{temp_file_path} already exists. Removing it.")
                try:
                    os.remove(temp_file_path)
                except PermissionError:
                    # If we can't remove it with normal permissions, use sudo
                    rm_cmd = ["sudo", "rm", "-f", temp_file_path]
                    subprocess.run(rm_cmd, check=True, capture_output=True, text=True)
                    self.logger.info(f"Removed existing temp file using sudo")
            
            self.logger.info(f"Creating new file with chunk size {chunk_size_kb}KB ({chunk_size_bytes} bytes) for {file_path}")

            # Create file with correct chunk size
            create_cmd = [
                "sudo",
                "beegfs-ctl",
                "--createfile",
                f"--chunksize={chunk_size_bytes}",
                "--storagepoolid=default",
                temp_file_path
            ]
            
            self.logger.debug(f"Running command: {' '.join(create_cmd)}")
            
            result = subprocess.run(create_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"beegfs-ctl createfile failed: {result.stderr}")
                self.logger.error(f"Command output: {result.stdout}")
                return False
                
            # Fix permissions on the newly created file
            chown_cmd = ["sudo", "chown", f"{os.getuid()}:{os.getgid()}", temp_file_path]
            subprocess.run(chown_cmd, check=True, capture_output=True, text=True)
            
            # Copy the data
            buffer_size = 10 * 1024 * 1024  # 10MB
            with open(file_path, 'rb') as src_file, open(temp_file_path, 'wb') as dst_file:
                while True:
                    buffer = src_file.read(buffer_size)
                    if not buffer:
                        break
                    dst_file.write(buffer)

            if os.path.getsize(file_path) != os.path.getsize(temp_file_path):
                self.logger.error("File size mismatch after copy.")
                # Use sudo to remove if needed
                try:
                    os.remove(temp_file_path)
                except PermissionError:
                    subprocess.run(["sudo", "rm", "-f", temp_file_path], check=False)
                return False

            # Verify the chunk size
            try:
                new_chunk_size = self.get_current_chunk_size(temp_file_path)
                if new_chunk_size != chunk_size_kb:
                    self.logger.error(f"Failed to set chunk size: expected {chunk_size_kb}KB but got {new_chunk_size}KB")
                    try:
                        os.remove(temp_file_path)
                    except PermissionError:
                        subprocess.run(["sudo", "rm", "-f", temp_file_path], check=False)
                    return False
            except Exception as e:
                self.logger.warning(f"Could not verify chunk size: {e}")
                # Continue anyway since we copied the data successfully

            # Preserve original file attributes 
            original_stat = os.stat(file_path)
            try:
                os.chmod(temp_file_path, original_stat.st_mode)
            except PermissionError:
                chmod_cmd = ["sudo", "chmod", f"{original_stat.st_mode & 0o777:o}", temp_file_path]
                subprocess.run(chmod_cmd, check=False)

            # Swap files - we might need sudo for this too
            backup_path = f"{file_path}.bak_{int(time.time())}"
            try:
                os.rename(file_path, backup_path)
            except PermissionError:
                mv_cmd = ["sudo", "mv", file_path, backup_path]
                subprocess.run(mv_cmd, check=True)
                
            try:
                os.rename(temp_file_path, file_path)
            except PermissionError:
                mv_cmd = ["sudo", "mv", temp_file_path, file_path]
                subprocess.run(mv_cmd, check=True)
                
            try:
                os.remove(backup_path)
            except PermissionError:
                rm_cmd = ["sudo", "rm", "-f", backup_path]
                subprocess.run(rm_cmd, check=True)

            self.logger.info(f"Successfully set chunk size to {chunk_size_kb}KB for {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return False
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except PermissionError:
                    # If we can't remove it with normal permissions, use sudo
                    try:
                        subprocess.run(["sudo", "rm", "-f", temp_file_path], check=False)
                        self.logger.info(f"Removed temporary file using sudo")
                    except:
                        self.logger.warning(f"Could not remove temporary file: {temp_file_path}")
            
            if backup_path and os.path.exists(backup_path):
                # Only restore if original is missing
                if not os.path.exists(file_path):
                    try:
                        os.rename(backup_path, file_path)
                    except PermissionError:
                        # Try with sudo
                        try:
                            subprocess.run(["sudo", "mv", backup_path, file_path], check=True)
                            self.logger.info(f"Restored original file from backup using sudo")
                        except Exception as e:
                            self.logger.error(f"Failed to restore from backup: {e}")

    def optimize_file(self, file_path, force=False, dry_run=False, model_type: Optional[str] = None) -> bool:
        """Optimize the chunk size of a single file."""
        # Check if the file exists
        if not os.path.exists(file_path):
            self.logger.error(f"File {file_path} does not exist.")
            return False
        
        try:
            # Get the current chunk size
            current_chunk_size = self.get_current_chunk_size(file_path)
            if current_chunk_size is None:
                self.logger.error(f"Could not determine current chunk size for {file_path}.")
                return False
            
            # Predict the optimal chunk size
            optimal_chunk_size = self.predict_chunk_size(file_path)
            
            # Check if optimization is needed
            if current_chunk_size == optimal_chunk_size and not force:
                self.logger.info(f"Chunk size for {file_path} is already optimal ({current_chunk_size}KB). No changes made.")
                return True
            
            self.logger.info(f"Optimizing {file_path}: {current_chunk_size}KB -> {optimal_chunk_size}KB")
            
            # Apply the optimization if it is not a dry run
            if not dry_run:
                success = self.set_chunk_size(file_path, optimal_chunk_size)
                if success:
                    self.logger.info(f"Successfully optimized chunk size for {file_path} to {optimal_chunk_size}KB.")
                
                    # Record the change in the database
                    try:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute ("""
                            INSERT INTO optimization_history 
                            (file_path, old_chunk_size, new_chunk_size, optimization_time)
                            VALUES (?, ?, ?, ?)
                        """, (file_path, current_chunk_size, optimal_chunk_size, time.time()))
                        conn.commit()
                        conn.close()
                    except sqlite3.OperationalError as e:
                        # Table doesn't exist yet, create it
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS optimization_history (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                file_path TEXT NOT NULL,
                                old_chunk_size INTEGER NOT NULL,
                                new_chunk_size INTEGER NOT NULL,
                                optimization_time REAL NOT NULL
                            )
                        """)
                        cursor.execute("""
                            INSERT INTO optimization_history 
                            (file_path, old_chunk_size, new_chunk_size, optimization_time)
                            VALUES (?, ?, ?, ?)
                        """, (file_path, current_chunk_size, optimal_chunk_size, time.time()))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        self.logger.error(f"Error recording optimization: {e}")
                    
                    return True
                else:
                    self.logger.error(f"Failed to optimize chunk size for {file_path}.")
                    return False
            else:
                self.logger.info(f"Dry run: Optimization for {file_path} would change chunk size from {current_chunk_size}KB to {optimal_chunk_size}KB.")
                return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing chunk size for {file_path}: {e}")
            return False
    
    def optimize_directory(self, directory, recursive=False, force=False, dry_run=False, 
                        file_types=None, min_size=None, max_size=None):
        """Optimize chunk sizes for all files in a directory."""
        if not os.path.isdir(directory):
            self.logger.error(f"Directory not found: {directory}")
            return 0
        
        optimized_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process files in the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            
            # Skip if not a file
            if not os.path.isfile(item_path):
                if os.path.isdir(item_path) and recursive:
                    # Recursively process subdirectory
                    sub_optimized, sub_failed, sub_skipped = self.optimize_directory(
                        item_path, recursive=True, force=force, dry_run=dry_run,
                        file_types=file_types, min_size=min_size, max_size=max_size
                    )
                    optimized_count += sub_optimized
                    failed_count += sub_failed
                    skipped_count += sub_skipped
                continue
            
            # Check file extension if specified
            if file_types and not any(item.endswith(ext) for ext in file_types):
                self.logger.debug(f"Skipping {item_path} - file type not in target list")
                skipped_count += 1
                continue
            
            # Check file size if specified
            file_size = os.path.getsize(item_path)
            if min_size and file_size < min_size:
                self.logger.debug(f"Skipping {item_path} - file size below minimum ({file_size} < {min_size})")
                skipped_count += 1
                continue
            if max_size and file_size > max_size:
                self.logger.debug(f"Skipping {item_path} - file size above maximum ({file_size} > {max_size})")
                skipped_count += 1
                continue
            
            # Optimize the file
            if self.optimize_file(item_path, force=force, dry_run=dry_run):
                optimized_count += 1
            else:
                failed_count += 1
        
        self.logger.info(f"Directory {directory}: {optimized_count} optimized, {failed_count} failed, {skipped_count} skipped")
        return optimized_count, failed_count, skipped_count
    
    def bulk_optimize(self, query_params=None, dry_run=False, limit=None):
        """Optimize chunk sizes for files based on database query."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query
            base_query = """
                SELECT DISTINCT m.file_path
                FROM file_metadata m
                JOIN file_access a ON m.file_path = a.file_path
            """
            
            where_clauses = []
            params = []
            
            if query_params:
                if 'min_size' in query_params:
                    where_clauses.append("m.file_size >= ?")
                    params.append(query_params['min_size'])
                
                if 'max_size' in query_params:
                    where_clauses.append("m.file_size <= ?")
                    params.append(query_params['max_size'])
                
                if 'min_access' in query_params:
                    where_clauses.append("(SELECT COUNT(*) FROM file_access WHERE file_path = m.file_path) >= ?")
                    params.append(query_params['min_access'])
                
                if 'extensions' in query_params:
                    ext_placeholders = ",".join(["?" for _ in query_params['extensions']])
                    where_clauses.append(f"LOWER(SUBSTR(m.file_path, INSTR(m.file_path, '.'), LENGTH(m.file_path))) IN ({ext_placeholders})")
                    params.extend(query_params['extensions'])
            
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
            
            if limit:
                base_query += f" LIMIT {limit}"
            
            # Execute query
            cursor.execute(base_query, params)
            
            # Process results
            files = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            self.logger.info(f"Found {len(files)} files for bulk optimization")
            
            # Optimize each file
            optimized_count = 0
            failed_count = 0
            
            for file_path in files:
                if os.path.exists(file_path):
                    if self.optimize_file(file_path, dry_run=dry_run):
                        optimized_count += 1
                    else:
                        failed_count += 1
                else:
                    self.logger.warning(f"File no longer exists: {file_path}")
                    failed_count += 1
            
            self.logger.info(f"Bulk optimization complete: {optimized_count} optimized, {failed_count} failed")
            return optimized_count, failed_count
            
        except Exception as e:
            self.logger.error(f"Error during bulk optimization: {e}")
            return 0, 0