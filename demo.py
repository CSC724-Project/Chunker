#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import random
import statistics
from typing import List, Dict, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
import sqlite3

from beechunker.common.config import config
from beechunker.monitor.db_manager import DBManager
from beechunker.optimizer.file_watcher import FileMonitorHandler
from watchdog.observers import Observer
from beechunker.optimizer.chunk_manager import ChunkSizeOptimizer


class BeeChunkerDemo:
    def __init__(self, model_type: str = "rf"):
        """Initialize all the services
        
        Args:
            model_type (str): The model type to use for optimization ("rf", "som", or "xgb")
        """
        print(f"Initializing BeeChunker Demo with {model_type} model...")
        
        # Suppress all logs
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Disable beechunker logging
        for logger_name in logging.Logger.manager.loggerDict:
            if logger_name.startswith('beechunker'):
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        
        # Store model type
        self.model_type = model_type

        # Initialize database
        self.db_path = config.get("monitor", "db_path")
        self.db_manager = DBManager(self.db_path)
        self.db_manager.initialize_db()

        # Initialize monitoring
        self.file_monitor_handler = FileMonitorHandler(self.db_manager)
        self.mount_points = config.get("beegfs", "mount_points")
        
        if not self.mount_points:
            print("No BeeGFS mount points configured. Using current directory.")
            self.mount_points = [os.getcwd()]
        
        # Set up observers for each mount point
        self.observers = []
        for mount_point in self.mount_points:
            if not os.path.exists(mount_point):
                print(f"Mount point does not exist: {mount_point}")
                continue

            observer = Observer()
            observer.schedule(self.file_monitor_handler, mount_point, recursive=True)
            observer.start()
            self.observers.append(observer)
            print(f"Started monitoring {mount_point}")

        if not self.observers:
            print("No valid mount points to monitor. Exiting.")
            sys.exit(1)

        # Initialize the optimizer
        try:
            self.optimizer = ChunkSizeOptimizer()
            print(f"BeeChunker optimizer initialized with {model_type} model.")
        except Exception as e:
            print(f"Error initializing optimizer: {e}")
            sys.exit(1)
        
        # Create test directory
        self.test_dir = os.path.join(self.mount_points[0], "beechunker_demo")
        os.makedirs(self.test_dir, exist_ok=True)
        print(f"Test directory: {self.test_dir}")
        
        # Results storage
        self.results = []
        
    def simulate_access_patterns(self, file_path: str, file_size: int):
        """Simulate access patterns for a file using BeeGFS tools"""
        print(f"Simulating access patterns for {file_path}...")
        
        # Get the current chunk size
        current_chunk_size = self.optimizer.get_current_chunk_size(file_path)
        if current_chunk_size is None:
            current_chunk_size = 512  # Default
        
        # Determine appropriate access patterns for file size
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb <= 1:  # Small files
            read_size = 4096
            write_size = 4096
            read_count = 10
            write_count = 5
            throughput = 50.0
        elif file_size_mb <= 100:  # Medium files
            read_size = 65536
            write_size = 32768
            read_count = 20
            write_count = 10
            throughput = 150.0
        else:  # Large files
            read_size = 1048576
            write_size = 524288
            read_count = 50
            write_count = 20
            throughput = 300.0
        
        # Simulate the access patterns by directly inserting into the database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert file metadata
            cursor.execute("""
                INSERT OR REPLACE INTO file_metadata 
                (file_path, file_size, chunk_size, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, file_size, current_chunk_size * 1024, time.time(), time.time()))
            
            # Simulate read operations
            for _ in range(read_count):
                cursor.execute("""
                    INSERT INTO file_access 
                    (file_path, access_type, access_time, read_size, write_size)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, "read", time.time(), read_size, 0))
            
            # Simulate write operations
            for _ in range(write_count):
                cursor.execute("""
                    INSERT INTO file_access 
                    (file_path, access_type, access_time, read_size, write_size)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, "write", time.time(), 0, write_size))
            
            # Insert throughput metrics
            cursor.execute("""
                INSERT INTO throughput_metrics 
                (file_path, start_time, end_time, bytes_transferred, operation_type, throughput_mbps)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, time.time()-1, time.time(), file_size, "read", throughput))
            
            conn.commit()
            conn.close()
            
            # Trigger manual processing of access information
            try:
                # Use beegfs-ctl to get file info which might trigger monitoring
                result = subprocess.run(
                    ["beegfs-ctl", "--getentryinfo", file_path],
                    capture_output=True,
                    text=True
                )
                
                # Simulate some actual file access with dd
                # Read the first 1MB of the file
                subprocess.run(
                    f"dd if={file_path} of=/dev/null bs=1M count=1 iflag=direct",
                    shell=True,
                    capture_output=True
                )
                
                # Write to the file using dd
                subprocess.run(
                    f"dd if=/dev/zero of={file_path} bs=1M count=1 oflag=direct conv=notrunc",
                    shell=True,
                    capture_output=True
                )
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Could not trigger BeeGFS monitoring: {e}")
            
            print(f"Simulated {read_count} reads and {write_count} writes with throughput {throughput} MB/s")
            
        except Exception as e:
            print(f"Error simulating access patterns: {e}")

    def cleanup(self):
        """Clean up resources when done"""
        for observer in self.observers:
            observer.stop()
        
        for observer in self.observers:
            observer.join()
            
        print("Stopped all observers.")
    
    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the file system

        Args:
            file_path (str): Path to the file to be removed

        Returns:
            bool: True if the file was removed successfully, False otherwise
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            else:
                return False
        except PermissionError:
            # If we can't remove it with normal permissions, use sudo
            try:
                rm_cmd = ["sudo", "rm", "-f", file_path]
                subprocess.run(rm_cmd, check=True, capture_output=True, text=True)
                return True
            except subprocess.SubprocessError:
                return False
        except Exception:
            return False
    
    def create_file(self, file_path: str, file_size: int) -> bool:
        """Create a file of a specified size

        Args:
            file_path (str): Path to the file to be created
            file_size (int): Size of the file to be created in bytes

        Returns:
            bool: True if the file was created successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create the file with random data
            with open(file_path, "wb") as f:
                # Write in chunks to avoid memory issues with large files
                chunk_size = min(10 * 1024 * 1024, file_size)  # 10MB or file_size
                remaining = file_size
                
                while remaining > 0:
                    current_chunk = min(chunk_size, remaining)
                    f.write(os.urandom(current_chunk))
                    remaining -= current_chunk
                    
            actual_size = os.path.getsize(file_path)
            return True
        except Exception:
            return False
    
    def measure_file_performance(self, file_path: str, iterations: int = 3) -> Dict:
        """Measure file read and write performance

        Args:
            file_path (str): Path to the file
            iterations (int): Number of iterations for averaging

        Returns:
            Dict: Dictionary with read/write speeds and throughput
        """
        if not os.path.exists(file_path):
            return None
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)  # Convert to MB
        
        # Measure write performance
        write_times = []
        for i in range(iterations):
            temp_file = f"{file_path}.temp_{i}"
            start_time = time.time()
            with open(file_path, "rb") as src, open(temp_file, "wb") as dst:
                dst.write(src.read())
            end_time = time.time()
            write_times.append(end_time - start_time)
            os.remove(temp_file)
            
        # Measure read performance
        read_times = []
        for _ in range(iterations):
            start_time = time.time()
            with open(file_path, "rb") as f:
                data = f.read()
            end_time = time.time()
            read_times.append(end_time - start_time)
            
        # Calculate average speeds and throughput
        avg_write_time = statistics.mean(write_times)
        avg_read_time = statistics.mean(read_times)
        
        write_speed_mbps = file_size_mb / avg_write_time if avg_write_time > 0 else 0
        read_speed_mbps = file_size_mb / avg_read_time if avg_read_time > 0 else 0
        
        # Get current chunk size
        current_chunk_size = self.optimizer.get_current_chunk_size(file_path)
        
        return {
            "file_path": file_path,
            "file_size": file_size,
            "chunk_size": current_chunk_size,
            "avg_write_time": avg_write_time,
            "avg_read_time": avg_read_time,
            "write_speed_mbps": write_speed_mbps,
            "read_speed_mbps": read_speed_mbps,
            "throughput_mbps": (write_speed_mbps + read_speed_mbps) / 2  # Average of read and write
        }
    
    def optimize_file(self, file_path: str) -> bool:
        """Optimize chunk size for a file using specified model type

        Args:
            file_path (str): Path to the file

        Returns:
            bool: True if optimization was successful, False otherwise
        """
        print(f"Optimizing chunk size for {file_path} using {self.model_type} model...")
        return self.optimizer.optimize_file(file_path, force=True, dry_run=False, model_type=self.model_type)
    
    def compare_performance(self, before: Dict, after: Dict) -> Dict:
        """Compare performance before and after optimization

        Args:
            before (Dict): Performance metrics before optimization
            after (Dict): Performance metrics after optimization

        Returns:
            Dict: Performance comparison with percentages
        """
        if not before or not after:
            return None
            
        write_speed_diff = after["write_speed_mbps"] - before["write_speed_mbps"]
        read_speed_diff = after["read_speed_mbps"] - before["read_speed_mbps"]
        throughput_diff = after["throughput_mbps"] - before["throughput_mbps"]
        
        write_speed_pct = (write_speed_diff / before["write_speed_mbps"]) * 100 if before["write_speed_mbps"] > 0 else 0
        read_speed_pct = (read_speed_diff / before["read_speed_mbps"]) * 100 if before["read_speed_mbps"] > 0 else 0
        throughput_pct = (throughput_diff / before["throughput_mbps"]) * 100 if before["throughput_mbps"] > 0 else 0
        
        chunk_diff = after["chunk_size"] - before["chunk_size"]
        chunk_pct = (chunk_diff / before["chunk_size"]) * 100 if before["chunk_size"] > 0 else 0
        
        return {
            "file_path": before["file_path"],
            "file_size": before["file_size"],
            "before_chunk_size": before["chunk_size"],
            "after_chunk_size": after["chunk_size"],
            "chunk_diff": chunk_diff,
            "chunk_pct": chunk_pct,
            "before_write_speed_mbps": before["write_speed_mbps"],
            "after_write_speed_mbps": after["write_speed_mbps"],
            "write_speed_diff": write_speed_diff,
            "write_speed_pct": write_speed_pct,
            "before_read_speed_mbps": before["read_speed_mbps"],
            "after_read_speed_mbps": after["read_speed_mbps"],
            "read_speed_diff": read_speed_diff,
            "read_speed_pct": read_speed_pct,
            "before_throughput_mbps": before["throughput_mbps"],
            "after_throughput_mbps": after["throughput_mbps"],
            "throughput_diff": throughput_diff,
            "throughput_pct": throughput_pct
        }
    
    def run_file_test(self, file_size: int, file_type: str, size_category: str) -> Dict:
        """Test a single file size with optimization

        Args:
            file_size (int): Size of the file to test in bytes
            file_type (str): Type/extension of the file (e.g., "txt", "bin")
            size_category (str): Category of file size ("small", "medium", "large")

        Returns:
            Dict: Performance comparison result
        """
        print(f"\n===== Testing {size_category} file ({file_size} bytes) =====")
        
        # Create a test file
        file_path = os.path.join(self.test_dir, f"{size_category}_{file_type}.dat")
        
        # Remove the file if it already exists
        self.remove_file(file_path)
        
        # Create a new file
        if not self.create_file(file_path, file_size):
            print(f"Failed to create test file: {file_path}")
            return None
        
        # Simulate access patterns
        print(f"Simulating access patterns for {file_path}...")
        self.simulate_access_patterns(file_path, file_size)
        print("Access patterns simulated.")
        
        time.sleep(2)
        
        # Wait for the file to be fully written
        time.sleep(1)
        
        # Measure performance before optimization
        print("Measuring performance before optimization...")
        before_metrics = self.measure_file_performance(file_path)
        if not before_metrics:
            print("Failed to measure performance before optimization")
            self.remove_file(file_path)
            return None
            
        print(f"Before optimization:")
        print(f"  Chunk size: {before_metrics['chunk_size']} KB")
        print(f"  Write speed: {before_metrics['write_speed_mbps']:.2f} MB/s")
        print(f"  Read speed: {before_metrics['read_speed_mbps']:.2f} MB/s")
        print(f"  Throughput: {before_metrics['throughput_mbps']:.2f} MB/s")
        
        # Optimize the file
        if not self.optimize_file(file_path):
            print(f"Failed to optimize file: {file_path}")
            self.remove_file(file_path)
            return None
            
        # Wait for optimization to settle
        time.sleep(1)
        
        # Measure performance after optimization
        print("Measuring performance after optimization...")
        after_metrics = self.measure_file_performance(file_path)
        if not after_metrics:
            print("Failed to measure performance after optimization")
            self.remove_file(file_path)
            return None
            
        print(f"After optimization:")
        print(f"  Chunk size: {after_metrics['chunk_size']} KB")
        print(f"  Write speed: {after_metrics['write_speed_mbps']:.2f} MB/s")
        print(f"  Read speed: {after_metrics['read_speed_mbps']:.2f} MB/s")
        print(f"  Throughput: {after_metrics['throughput_mbps']:.2f} MB/s")
        
        # Compare results
        comparison = self.compare_performance(before_metrics, after_metrics)
        if comparison:
            print("\nPerformance comparison:")
            print(f"  Chunk size: {comparison['before_chunk_size']} KB -> {comparison['after_chunk_size']} KB ({comparison['chunk_pct']:.2f}%)")
            print(f"  Write speed: {comparison['write_speed_pct']:.2f}% change")
            print(f"  Read speed: {comparison['read_speed_pct']:.2f}% change")
            print(f"  Throughput: {comparison['throughput_pct']:.2f}% change")
        
        # Clean up test file
        self.remove_file(file_path)
        
        return comparison
    
    def run_demo(self):
        """Run the full demonstration with different file sizes"""
        print(f"\n====== BeeChunker Optimization Demo ({self.model_type.upper()} Model) ======")
        
        # Define file sizes for testing
        file_sizes = {
            "small": 1 * 1024 * 1024,        # 10 MB
            "medium": 500 * 1024 * 1024,      # 500 MB
            "large": 2 * 1024 * 1024 * 1024   # 2GB
        }
        
        results = []
        
        # Test each file size
        for size_category, file_size in file_sizes.items():
            comparison = self.run_file_test(file_size, "dat", size_category)
            if comparison:
                results.append(comparison)
        
        # Print summary of all results
        if results:
            print("\n====== Summary of Optimization Results ======")
            
            # Create a pandas DataFrame for better formatting
            df = pd.DataFrame(results)
            
            # Add a 'Result' column to show if performance improved or degraded
            df['Result'] = df.apply(
                lambda row: 'Improved' if row['throughput_pct'] > 0 else 'Degraded', axis=1
            )
            
            # Select relevant columns for display
            summary_df = df[[
                'file_size', 
                'before_chunk_size', 
                'after_chunk_size',
                'chunk_pct',
                'before_throughput_mbps', 
                'after_throughput_mbps',
                'throughput_pct',
                'Result'
            ]]
            
            # Format file sizes in MB
            summary_df['file_size'] = summary_df['file_size'].apply(lambda x: f"{x/(1024*1024):.1f} MB")
            
            # Rename columns for readability
            summary_df.columns = [
                'File Size', 
                'Before Chunk Size (KB)', 
                'After Chunk Size (KB)',
                'Chunk Size Change (%)',
                'Before Throughput (MB/s)', 
                'After Throughput (MB/s)',
                'Throughput Change (%)',
                'Overall Result'
            ]
            
            print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
            
            # Calculate average improvement
            avg_throughput_change = df['throughput_pct'].mean()
            print(f"\nAverage throughput change: {avg_throughput_change:.2f}%")
            
            improved_count = len(df[df['throughput_pct'] > 0])
            print(f"Files with improved performance: {improved_count}/{len(df)}")
        
        print("\n====== Demo Complete ======")


if __name__ == "__main__":
    # Parse command line arguments for model type
    import argparse
    
    parser = argparse.ArgumentParser(description='BeeChunker Demo Script')
    parser.add_argument('--model', type=str, choices=['rf', 'som', 'xgb'], default='rf',
                        help='Model type to use for optimization (default: rf)')
    
    args = parser.parse_args()
    
    try:
        demo = BeeChunkerDemo(model_type=args.model)
        demo.run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error during demo: {e}")
    finally:
        if 'demo' in locals():
            demo.cleanup()
        print("Demo cleanup complete.")