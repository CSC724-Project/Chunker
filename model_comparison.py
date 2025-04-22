#!/usr/bin/env python3

import os
import sys
import time
import subprocess
import random
import statistics
import argparse
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from beechunker.common.config import config
from beechunker.monitor.db_manager import DBManager
from beechunker.optimizer.file_watcher import FileMonitorHandler
from watchdog.observers import Observer
from beechunker.optimizer.chunk_manager import ChunkSizeOptimizer


class BeeChunkerModelComparison:
    def __init__(self):
        """Initialize the model comparison utility"""
        print("Initializing BeeChunker Model Comparison Tool...")
        
        # Suppress all logs
        logging.getLogger().setLevel(logging.CRITICAL)
        for logger_name in logging.Logger.manager.loggerDict:
            if logger_name.startswith('beechunker'):
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        
        # Initialize database
        self.db_path = config.get("monitor", "db_path")
        self.db_manager = DBManager(self.db_path)
        self.db_manager.initialize_db()

        # Initialize monitoring
        self.file_monitor_handler = FileMonitorHandler(self.db_manager)
        self.mount_points = config.get("beegfs", "mount_points")
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')  # Suppress stderr output
        
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

        # Initialize the optimizer for each model type
        try:
            # We'll create one optimizer and switch model types as needed
            self.optimizer = ChunkSizeOptimizer()
            print("BeeChunker optimizer initialized.")
        except Exception as e:
            print(f"Error initializing optimizer: {e}")
            sys.exit(1)
        
        # Create test directory
        self.test_dir = os.path.join(self.mount_points[0], "beechunker_comparison")
        os.makedirs(self.test_dir, exist_ok=True)
        print(f"Test directory: {self.test_dir}")
        
        # Results storage
        self.results = []
        
    def simulate_access_patterns(self, file_path: str, file_size: int, access_pattern: str = "mixed"):
        """Simulate different access patterns for a file

        Args:
            file_path (str): Path to the file
            file_size (int): Size of the file in bytes
            access_pattern (str): Type of access pattern to simulate:
                                 "read_heavy" - More reads than writes
                                 "write_heavy" - More writes than reads
                                 "mixed" - Balanced read/write operations
                                 "sequential" - Large sequential operations
                                 "random" - Small random operations
        """
        print(f"Simulating {access_pattern} access patterns for {file_path}...")
        
        # Get the current chunk size
        current_chunk_size = self.optimizer.get_current_chunk_size(file_path)
        if current_chunk_size is None:
            current_chunk_size = 512  # Default
        
        # Determine appropriate access patterns for file size and pattern type
        file_size_mb = file_size / (1024 * 1024)
        
        # Base parameters
        if file_size_mb <= 10:  # Small files
            base_read_size = 4096
            base_write_size = 4096
            base_read_count = 10
            base_write_count = 5
            base_throughput = 50.0
        elif file_size_mb <= 100:  # Medium files
            base_read_size = 65536
            base_write_size = 32768
            base_read_count = 20
            base_write_count = 10
            base_throughput = 150.0
        else:  # Large files
            base_read_size = 1048576
            base_write_size = 524288
            base_read_count = 50
            base_write_count = 20
            base_throughput = 300.0
            
        # Adjust based on access pattern
        if access_pattern == "read_heavy":
            read_count = base_read_count * 3
            write_count = base_write_count // 2
            read_size = base_read_size * 2
            write_size = base_write_size
        elif access_pattern == "write_heavy":
            read_count = base_read_count // 2
            write_count = base_write_count * 3
            read_size = base_read_size
            write_size = base_write_size * 2
        elif access_pattern == "sequential":
            read_count = base_read_count
            write_count = base_write_count
            read_size = base_read_size * 4
            write_size = base_write_size * 4
        elif access_pattern == "random":
            read_count = base_read_count * 2
            write_count = base_write_count * 2
            read_size = base_read_size // 2
            write_size = base_write_size // 2
        else:  # mixed (default)
            read_count = base_read_count
            write_count = base_write_count
            read_size = base_read_size
            write_size = base_write_size
        
        # Simulate the access patterns
        try:
            # Use beegfs-ctl to get file info which might trigger monitoring
            try:
                result = subprocess.run(
                    ["beegfs-ctl", "--getentryinfo", file_path],
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError:
                print("Warning: Could not get BeeGFS entry info")
                
            # Simulate actual file access with dd
            # Read operations
            for _ in range(min(read_count, 5)):  # Limit actual operations for speed
                offset = random.randint(0, max(1, file_size - read_size))
                try:
                    subprocess.run(
                        f"dd if={file_path} of=/dev/null bs={read_size} count=1 skip={offset // read_size} iflag=direct",
                        shell=True,
                        capture_output=True
                    )
                except subprocess.SubprocessError:
                    pass
                
            # Write operations
            for _ in range(min(write_count, 5)):  # Limit actual operations for speed
                offset = random.randint(0, max(1, file_size - write_size))
                try:
                    subprocess.run(
                        f"dd if=/dev/zero of={file_path} bs={write_size} count=1 seek={offset // write_size} oflag=direct conv=notrunc",
                        shell=True,
                        capture_output=True
                    )
                except subprocess.SubprocessError:
                    pass
                    
            print(f"Simulated {read_count} reads and {write_count} writes with {access_pattern} pattern")
            
        except Exception as e:
            print(f"Error simulating access patterns: {e}")

    def cleanup(self):
        """Clean up resources when done"""
        for observer in self.observers:
            observer.stop()
        
        for observer in self.observers:
            observer.join()
        
        sys.stderr = self.original_stderr
            
        # Clean up test directory
        try:
            for file in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.test_dir)
            print(f"Removed test directory: {self.test_dir}")
        except Exception:
            print(f"Warning: Could not remove test directory: {self.test_dir}")
        
        print("Stopped all observers and cleaned up resources.")
    
    def remove_file(self, file_path: str) -> bool:
        """Remove a file from the file system"""
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
    
    def create_file(self, file_path: str, file_size: int, pattern: str = "random") -> bool:
        """Create a file of a specified size with a specific data pattern

        Args:
            file_path (str): Path to the file to be created
            file_size (int): Size of the file to be created in bytes
            pattern (str): Data pattern - "random", "zeros", "sequential"

        Returns:
            bool: True if the file was created successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create the file with specified pattern
            with open(file_path, "wb") as f:
                # Write in chunks to avoid memory issues with large files
                chunk_size = min(10 * 1024 * 1024, file_size)  # 10MB or file_size
                remaining = file_size
                
                while remaining > 0:
                    current_chunk = min(chunk_size, remaining)
                    
                    if pattern == "zeros":
                        f.write(b'\0' * current_chunk)
                    elif pattern == "sequential":
                        # Create repeating sequence of bytes
                        sequence = bytes(range(256)) * (current_chunk // 256 + 1)
                        f.write(sequence[:current_chunk])
                    else:  # random (default)
                        f.write(os.urandom(current_chunk))
                        
                    remaining -= current_chunk
                    
            return True
        except Exception as e:
            print(f"Error creating file: {e}")
            return False
    
    def measure_file_performance(self, file_path: str, iterations: int = 3) -> Dict:
        """Measure file read and write performance"""
        if not os.path.exists(file_path):
            return None
            
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)  # Convert to MB
        
        # Determine test parameters based on file size
        if file_size > 1024 * 1024 * 1024:  # >1GB
            # For large files, use sequential read/write with large blocks
            read_block_size = 10 * 1024 * 1024  # 10MB
            read_count = min(10, max(1, int(file_size / read_block_size)))
            write_block_size = 10 * 1024 * 1024  # 10MB
        else:
            # For smaller files, use whole file operations
            read_block_size = file_size
            read_count = 1
            write_block_size = file_size
            
        # Measure write performance
        write_times = []
        for i in range(iterations):
            temp_file = f"{file_path}.temp_{i}"
            
            # Use dd for more accurate timing
            start_time = time.time()
            subprocess.run(
                f"dd if={file_path} of={temp_file} bs={write_block_size} count={read_count} conv=fsync",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            end_time = time.time()
            
            write_times.append(end_time - start_time)
            os.remove(temp_file)
            
        # Measure read performance
        read_times = []
        for i in range(iterations):
            # Use dd for more accurate timing
            start_time = time.time()
            subprocess.run(
                f"dd if={file_path} of=/dev/null bs={read_block_size} count={read_count}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
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
    
    def optimize_file(self, file_path: str, model_type: str) -> Dict:
        """Optimize chunk size for a file using specified model

        Args:
            file_path (str): Path to the file
            model_type (str): Model type to use for optimization ("rf" or "xgb")

        Returns:
            Dict: Dictionary with old and new chunk sizes
        """
        try:
            # Get current chunk size
            current_chunk_size = self.optimizer.get_current_chunk_size(file_path)
            
            # Predict optimal chunk size
            optimal_chunk_size = self.optimizer.predict_chunk_size(file_path, model_type=model_type)
            
            # Apply the optimization
            success = self.optimizer.set_chunk_size(file_path, optimal_chunk_size)
            
            if success:
                # Verify the new chunk size
                new_chunk_size = self.optimizer.get_current_chunk_size(file_path)
                return {
                    "file_path": file_path,
                    "old_chunk_size": current_chunk_size,
                    "new_chunk_size": new_chunk_size,
                    "success": True
                }
            else:
                return {
                    "file_path": file_path,
                    "old_chunk_size": current_chunk_size,
                    "new_chunk_size": None,
                    "success": False
                }
        except Exception as e:
            print(f"Error optimizing file: {e}")
            return {
                "file_path": file_path,
                "old_chunk_size": None,
                "new_chunk_size": None,
                "success": False
            }
    
    def test_file_scenario(self, file_size: int, data_pattern: str, access_pattern: str, model_type: str) -> Dict:
        """Test a specific file scenario with a particular model

        Args:
            file_size (int): File size in bytes
            data_pattern (str): Data pattern for file creation
            access_pattern (str): Access pattern to simulate
            model_type (str): Model type to test ("rf" or "xgb")

        Returns:
            Dict: Performance results
        """
        # Create unique filename based on parameters
        file_name = f"{model_type}_{data_pattern}_{access_pattern}_{file_size // (1024*1024)}MB.dat"
        file_path = os.path.join(self.test_dir, file_name)
        
        # Remove existing file
        self.remove_file(file_path)
        
        # Create test file
        if not self.create_file(file_path, file_size, data_pattern):
            print(f"Failed to create test file: {file_path}")
            return None
        
        # Simulate access patterns
        self.simulate_access_patterns(file_path, file_size, access_pattern)
        
        # Wait for operations to settle
        time.sleep(2)
        
        # Measure performance before optimization
        before_metrics = self.measure_file_performance(file_path)
        if not before_metrics:
            print(f"Failed to measure initial performance: {file_path}")
            self.remove_file(file_path)
            return None
        
        # Optimize with specified model
        optimization_result = self.optimize_file(file_path, model_type)
        if not optimization_result["success"]:
            print(f"Failed to optimize with {model_type}: {file_path}")
            self.remove_file(file_path)
            return None
        
        # Wait for optimization to settle
        time.sleep(2)
        
        # Measure performance after optimization
        after_metrics = self.measure_file_performance(file_path)
        if not after_metrics:
            print(f"Failed to measure post-optimization performance: {file_path}")
            self.remove_file(file_path)
            return None
        
        # Calculate performance changes
        throughput_change = after_metrics["throughput_mbps"] - before_metrics["throughput_mbps"]
        throughput_pct = (throughput_change / before_metrics["throughput_mbps"]) * 100 if before_metrics["throughput_mbps"] > 0 else 0
        
        read_change = after_metrics["read_speed_mbps"] - before_metrics["read_speed_mbps"]
        read_pct = (read_change / before_metrics["read_speed_mbps"]) * 100 if before_metrics["read_speed_mbps"] > 0 else 0
        
        write_change = after_metrics["write_speed_mbps"] - before_metrics["write_speed_mbps"]
        write_pct = (write_change / before_metrics["write_speed_mbps"]) * 100 if before_metrics["write_speed_mbps"] > 0 else 0
        
        # Clean up
        self.remove_file(file_path)
        
        # Return complete results dictionary
        return {
            "model_type": model_type,
            "file_size": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "data_pattern": data_pattern,
            "access_pattern": access_pattern,
            "old_chunk_size": optimization_result["old_chunk_size"],
            "new_chunk_size": optimization_result["new_chunk_size"],
            "before_read_speed": before_metrics["read_speed_mbps"],
            "after_read_speed": after_metrics["read_speed_mbps"],
            "read_speed_change": read_change,
            "read_speed_pct": read_pct,
            "before_write_speed": before_metrics["write_speed_mbps"],
            "after_write_speed": after_metrics["write_speed_mbps"],
            "write_speed_change": write_change,
            "write_speed_pct": write_pct,
            "before_throughput": before_metrics["throughput_mbps"],
            "after_throughput": after_metrics["throughput_mbps"],
            "throughput_change": throughput_change,
            "throughput_pct": throughput_pct
        }
    
    def generate_comparison_plots(self, results: List[Dict]):
        """Generate comparison plots from test results"""
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Create directory for plots
        plots_dir = os.path.join(os.getcwd(), "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Throughput Change by File Size and Model
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df, 
            x='file_size_mb', 
            y='throughput_pct', 
            hue='model_type',
            palette=['#3498db', '#e74c3c']
        )
        plt.title('Throughput Improvement by File Size and Model')
        plt.xlabel('File Size (MB)')
        plt.ylabel('Throughput Improvement (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'throughput_by_size.png'))
        
        # 2. Chunk Size Selection Comparison
        plt.figure(figsize=(10, 6))
        # Create a new dataframe with melted chunk sizes
        chunk_df = df[['model_type', 'file_size_mb', 'old_chunk_size', 'new_chunk_size']].copy()
        chunk_df = pd.melt(
            chunk_df, 
            id_vars=['model_type', 'file_size_mb'],
            value_vars=['old_chunk_size', 'new_chunk_size'],
            var_name='chunk_type', 
            value_name='chunk_size'
        )
        # Plot chunk sizes
        sns.barplot(
            data=chunk_df,
            x='file_size_mb',
            y='chunk_size',
            hue='chunk_type',
            palette=['#7f8c8d', '#2ecc71'],  # Gray for old, Green for new
            alpha=0.7
        )
        plt.title('Chunk Size Selection by File Size and Model Type')
        plt.xlabel('File Size (MB)')
        plt.ylabel('Chunk Size (KB)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'chunk_size_selection.png'))
        
        # 3. Performance by Access Pattern
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x='access_pattern',
            y='throughput_pct',
            hue='model_type',
            palette=['#3498db', '#e74c3c']
        )
        plt.title('Throughput Improvement by Access Pattern')
        plt.xlabel('Access Pattern')
        plt.ylabel('Throughput Improvement (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'throughput_by_pattern.png'))
        
        # 4. Read vs. Write Performance
        plt.figure(figsize=(12, 6))
        # Create separate dataframes for read and write
        read_df = df[['model_type', 'read_speed_pct']].copy()
        read_df['operation'] = 'Read'
        read_df.rename(columns={'read_speed_pct': 'improvement'}, inplace=True)
        
        write_df = df[['model_type', 'write_speed_pct']].copy()
        write_df['operation'] = 'Write'
        write_df.rename(columns={'write_speed_pct': 'improvement'}, inplace=True)
        
        # Combine
        perf_df = pd.concat([read_df, write_df])
        
        # Plot
        sns.boxplot(
            data=perf_df,
            x='operation',
            y='improvement',
            hue='model_type',
            palette=['#3498db', '#e74c3c']
        )
        plt.title('Read vs. Write Performance Improvement by Model')
        plt.xlabel('Operation Type')
        plt.ylabel('Performance Improvement (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'read_vs_write.png'))
        
        # 5. Model Comparison Summary
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x='model_type',
            y='throughput_pct',
            palette=['#3498db', '#e74c3c']
        )
        plt.title('Overall Performance Comparison by Model')
        plt.xlabel('Model Type')
        plt.ylabel('Throughput Improvement (%)')
        
        # Add mean line 
        means = df.groupby('model_type')['throughput_pct'].mean()
        for i, model in enumerate(means.index):
            plt.axhline(
                y=means[model], 
                xmin=i/len(means) - 0.2, 
                xmax=i/len(means) + 0.2,
                color='black', 
                linestyle='--'
            )
            plt.text(
                i, 
                means[model] + 2, 
                f'Mean: {means[model]:.1f}%', 
                ha='center'
            )
            
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
        
        print(f"Plots saved to: {plots_dir}")
    
    def run_comparison(self, num_trials: int = 3):
        """Run a comprehensive model comparison
        
        Args:
            num_trials (int): Number of trials to run for each scenario
        """
        print(f"\n=== BeeChunker Model Comparison (RF vs. XGBoost) ===")
        print(f"Running {num_trials} trials for each scenario...")
        
        # Define test scenarios
        file_sizes = [
            10 * 1024 * 1024,       # 10 MB
            100 * 1024 * 1024,      # 100 MB
            1024 * 1024 * 1024      # 1 GB
        ]
        
        data_patterns = ["random", "sequential"]
        access_patterns = ["read_heavy", "write_heavy", "mixed", "sequential", "random"]
        model_types = ["rf", "xgb"]
        
        # For a shorter test, uncomment these lines:
        # data_patterns = ["random"]
        # access_patterns = ["mixed", "read_heavy"]
        
        all_results = []
        
        # Run tests for each combination
        total_tests = len(file_sizes) * len(data_patterns) * len(access_patterns) * len(model_types) * num_trials
        test_count = 0
        
        for file_size in file_sizes:
            size_mb = file_size / (1024 * 1024)
            for data_pattern in data_patterns:
                for access_pattern in access_patterns:
                    for model_type in model_types:
                        for trial in range(num_trials):
                            test_count += 1
                            print(f"\nTest {test_count}/{total_tests}: "
                                  f"{size_mb:.0f}MB file, {data_pattern} data, "
                                  f"{access_pattern} access, {model_type} model (Trial {trial+1}/{num_trials})")
                            
                            result = self.test_file_scenario(
                                file_size=file_size,
                                data_pattern=data_pattern,
                                access_pattern=access_pattern,
                                model_type=model_type
                            )
                            
                            if result:
                                all_results.append(result)
                                
                                # Show interim result
                                print(f"  Before chunk size: {result['old_chunk_size']} KB")
                                print(f"  After chunk size:  {result['new_chunk_size']} KB")
                                print(f"  Throughput change: {result['throughput_pct']:.2f}%")
        
        # Generate summary statistics and visualizations
        if all_results:
            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Generate visualizations
            self.generate_comparison_plots(all_results)
            
            # Print summary statistics
            print("\n=== Model Comparison Summary ===")
            
            # Model performance comparison
            model_stats = results_df.groupby('model_type')['throughput_pct'].agg(['mean', 'std', 'min', 'max', 'count'])
            print("\nOverall Performance by Model:")
            print(tabulate(model_stats, headers=['Model', 'Mean Throughput Improvement (%)', 'Std Dev', 'Min', 'Max', 'Count'],
                         tablefmt='grid', floatfmt='.2f'))
            
            # Performance by file size
            size_stats = results_df.groupby(['model_type', 'file_size_mb'])['throughput_pct'].mean().reset_index()
            size_stats = size_stats.pivot(index='file_size_mb', columns='model_type', values='throughput_pct')
            print("\nMean Throughput Improvement (%) by File Size:")
            print(tabulate(size_stats, headers='keys', tablefmt='grid', floatfmt='.2f'))
            
            # Performance by access pattern
            pattern_stats = results_df.groupby(['model_type', 'access_pattern'])['throughput_pct'].mean().reset_index()
            pattern_stats = pattern_stats.pivot(index='access_pattern', columns='model_type', values='throughput_pct')
            print("\nMean Throughput Improvement (%) by Access Pattern:")
            print(tabulate(pattern_stats, headers='keys', tablefmt='grid', floatfmt='.2f'))
            
            # Read vs. Write performance
            read_write_stats = results_df.groupby('model_type')[['read_speed_pct', 'write_speed_pct']].mean()
            print("\nRead vs. Write Speed Improvement (%):")
            print(tabulate(read_write_stats, headers=['Model', 'Read Speed', 'Write Speed'],
                         tablefmt='grid', floatfmt='.2f'))
            
            # Chunk size selection
            chunk_stats = results_df.groupby(['model_type', 'file_size_mb'])[['old_chunk_size', 'new_chunk_size']].mean().reset_index()
            chunk_diff = results_df.groupby(['model_type', 'file_size_mb']).apply(
                lambda x: ((x['new_chunk_size'] - x['old_chunk_size']) / x['old_chunk_size'] * 100).mean()
            ).reset_index(name='chunk_change_pct')
            
            print("\nChunk Size Selection by File Size:")
            for model in model_types:
                model_chunks = chunk_stats[chunk_stats['model_type'] == model]
                model_diffs = chunk_diff[chunk_diff['model_type'] == model]
                print(f"\n{model.upper()} Model Chunk Sizes:")
                model_table = pd.merge(model_chunks, model_diffs, on=['model_type', 'file_size_mb'])
                print(tabulate(model_table[['file_size_mb', 'old_chunk_size', 'new_chunk_size', 'chunk_change_pct']],
                             headers=['File Size (MB)', 'Old Chunk (KB)', 'New Chunk (KB)', 'Change (%)'],
                             tablefmt='grid', floatfmt='.2f'))
            
            # Statistical significance test
            from scipy import stats
            rf_results = results_df[results_df['model_type'] == 'rf']['throughput_pct']
            xgb_results = results_df[results_df['model_type'] == 'xgb']['throughput_pct']
            
            t_stat, p_value = stats.ttest_ind(rf_results, xgb_results)
            print("\nStatistical Significance Test (t-test):")
            print(f"t-statistic: {t_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                if rf_results.mean() > xgb_results.mean():
                    print("Result: Random Forest performs significantly better than XGBoost (p < 0.05)")
                else:
                    print("Result: XGBoost performs significantly better than Random Forest (p < 0.05)")
            else:
                print("Result: No statistically significant difference between models (p >= 0.05)")
            
            # Save results
            results_csv = "model_comparison_results.csv"
            results_df.to_csv(results_csv, index=False)
            print(f"\nDetailed results saved to: {results_csv}")
            
            # Create a summary table that highlights the winner in each category
            print("\n=== Final Model Comparison ===")
            summary_table = []
            
            # Overall winner
            rf_overall = rf_results.mean()
            xgb_overall = xgb_results.mean()
            overall_winner = "RF" if rf_overall > xgb_overall else "XGBoost"
            summary_table.append(["Overall Performance", f"{rf_overall:.2f}%", f"{xgb_overall:.2f}%", overall_winner])
            
            # File size categories
            for size in sorted(results_df['file_size_mb'].unique()):
                rf_size = results_df[(results_df['model_type'] == 'rf') & (results_df['file_size_mb'] == size)]['throughput_pct'].mean()
                xgb_size = results_df[(results_df['model_type'] == 'xgb') & (results_df['file_size_mb'] == size)]['throughput_pct'].mean()
                size_winner = "RF" if rf_size > xgb_size else "XGBoost"
                summary_table.append([f"{size:.0f}MB Files", f"{rf_size:.2f}%", f"{xgb_size:.2f}%", size_winner])
            
            # Access patterns
            for pattern in sorted(results_df['access_pattern'].unique()):
                rf_pattern = results_df[(results_df['model_type'] == 'rf') & (results_df['access_pattern'] == pattern)]['throughput_pct'].mean()
                xgb_pattern = results_df[(results_df['model_type'] == 'xgb') & (results_df['access_pattern'] == pattern)]['throughput_pct'].mean()
                pattern_winner = "RF" if rf_pattern > xgb_pattern else "XGBoost"
                summary_table.append([f"{pattern.replace('_', ' ').title()} Access", f"{rf_pattern:.2f}%", f"{xgb_pattern:.2f}%", pattern_winner])
            
            # Read operations
            rf_read = results_df[results_df['model_type'] == 'rf']['read_speed_pct'].mean()
            xgb_read = results_df[results_df['model_type'] == 'xgb']['read_speed_pct'].mean()
            read_winner = "RF" if rf_read > xgb_read else "XGBoost"
            summary_table.append(["Read Operations", f"{rf_read:.2f}%", f"{xgb_read:.2f}%", read_winner])
            
            # Write operations
            rf_write = results_df[results_df['model_type'] == 'rf']['write_speed_pct'].mean()
            xgb_write = results_df[results_df['model_type'] == 'xgb']['write_speed_pct'].mean()
            write_winner = "RF" if rf_write > xgb_write else "XGBoost"
            summary_table.append(["Write Operations", f"{rf_write:.2f}%", f"{xgb_write:.2f}%", write_winner])
            
            # Print the final summary table
            print(tabulate(summary_table, headers=["Category", "RF Performance", "XGBoost Performance", "Winner"],
                         tablefmt='grid'))
            
            # Overall conclusion
            rf_wins = sum(1 for row in summary_table if row[3] == "RF")
            xgb_wins = sum(1 for row in summary_table if row[3] == "XGBoost")
            
            print("\n=== Final Verdict ===")
            if rf_wins > xgb_wins:
                print(f"Random Forest wins in {rf_wins} out of {len(summary_table)} categories!")
                print(f"Random Forest provides better overall chunk size optimization.")
            elif xgb_wins > rf_wins:
                print(f"XGBoost wins in {xgb_wins} out of {len(summary_table)} categories!")
                print(f"XGBoost provides better overall chunk size optimization.")
            else:
                print(f"It's a tie! Both models win in {rf_wins} categories each.")
                print(f"Both models perform similarly for chunk size optimization.")
        
        # Clean up all test files
        self.cleanup()
        
        print("\nModel comparison completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BeeChunker Model Comparison Tool')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials per scenario (default: 3)')
    
    args = parser.parse_args()
    
    try:
        comparison = BeeChunkerModelComparison()
        comparison.run_comparison(num_trials=args.trials)
    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"Error during comparison: {e}")
    finally:
        if 'comparison' in locals():
            comparison.cleanup()
        print("Comparison completed and resources cleaned up.")