# BeeChunker

**Intelligent Chunk Size Optimization for BeeGFS using Self-Organizing Maps**

BeeChunker is an intelligent system for optimizing chunk sizes in BeeGFS (BeeGFS Distributed Filesystem) storage systems by analyzing file access patterns and predicting optimal chunk sizes for files using machine learning.

## Overview

BeeGFS is a parallel file system that distributes file data across multiple storage servers. The "chunk size" determines how a file is divided and distributed, which can significantly impact performance. However, determining the optimal chunk size for a file is challenging as it depends on many factors:

- File size
- Access patterns (read vs. write)
- I/O sizes
- Workload characteristics
- File type

BeeChunker solves this problem by:
1. Continuously monitoring file access patterns
2. Training a Self-Organizing Map (SOM) model on the collected data
3. Predicting optimal chunk sizes based on file characteristics
4. Automatically applying these optimizations to both existing and new files

## Features

- **Continuous Monitoring**: Tracks file access patterns across BeeGFS mount points
- **ML-Based Optimization**: Uses Self-Organizing Maps to predict optimal chunk sizes
- **Automatic Rechunking**: Identifies and rechunks suboptimally chunked files
- **New File Optimization**: Sets optimal chunk sizes for newly created files
- **Visualization**: Provides insights into access patterns and chunk size distributions

## Architecture

BeeChunker consists of three main components:

### 1. Monitor Service

The monitor service tracks file access operations on BeeGFS mount points and stores the data in a SQLite database. It records:

- File paths and sizes
- Current chunk sizes
- Read/write operations
- Access counts and sizes
- Performance metrics

### 2. SOM Trainer

The Self-Organizing Map (SOM) trainer analyzes the collected data to find patterns between file characteristics and optimal chunk sizes. SOMs are unsupervised neural networks that create a topological mapping of high-dimensional data.

Key features:
- Feature engineering from raw access data
- Unsupervised learning of access patterns
- Creation of a chunk size map
- Visualization of model insights (U-Matrix, Component Planes, Cluster Analysis)

### 3. Optimizer

The optimizer component predicts and applies optimal chunk sizes for files:
- Extracts features from files
- Uses the trained SOM to predict optimal chunk sizes
- Applies changes using BeeGFS tools
- Tracks optimization history

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/beechunker.git
cd beechunker
```

2. Install the package:

```bash
pip install -e .
```

3. Configure BeeGFS mount points in `/opt/beechunker/data/config.json`:

```json
{
  "beegfs": {
    "mount_points": [
      "/mnt/beegfs",
      "/mnt/beegfs2"
    ]
  }
}
```

### Requirements

- Python 3.8+
- BeeGFS installation
- Access to BeeGFS command-line tools
- Root or sudo access (for some operations)

## Configuration

The default configuration is located at `beechunker/common/default_config.json`. You can modify this file or create a custom configuration at `/opt/beechunker/data/config.json`.

Default configuration:

```json
{
  "monitor": {
    "db_path": "/opt/beechunker/data/access_patterns.db",
    "log_path": "/opt/beechunker/data/logs/monitor.log",
    "polling_interval": 300
  },
  "optimizer": {
    "log_path": "/opt/beechunker/data/logs/optimizer.log",
    "min_chunk_size": 512,
    "max_chunk_size": 4096,
    "scan_interval": 3600,
    "chunk_diff_threshold": 0.3
  },
  "ml": {
    "models_dir": "/opt/beechunker/data/models",
    "log_path": "/opt/beechunker/data/logs/trainer.log",
    "training_interval": 86400,
    "min_training_samples": 100,
    "som_iterations": 5000
  },
  "beegfs": {
    "mount_points": []
  }
}
```

## Usage

BeeChunker provides several command-line interfaces to interact with its components:

### Setting Up Systemd Services

To run BeeChunker components as systemd services (recommended for production use), follow these steps:

1. Create a systemd service file for the monitor:

```bash
sudo nano /etc/systemd/system/beechunker-monitor.service
```

Add the following content:

```
[Unit]
Description=BeeChunker Monitor Service
After=network.target

[Service]
Type=simple
User=root
Group=root
ExecStart=/usr/local/bin/beechunker-monitor run
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

2. Create a systemd service file for the optimizer:

```bash
sudo nano /etc/systemd/system/beechunker-optimizer.service
```

Add the following content:

```
[Unit]
Description=BeeChunker Optimizer Service
After=network.target

[Service]
Type=simple
User=root
Group=root
ExecStart=/usr/local/bin/beechunker-optimizer run
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

3. Reload the systemd daemon to recognize the new service files:

```bash
sudo systemctl daemon-reload
```

4. Enable the services to start automatically at boot (optional):

```bash
sudo systemctl enable beechunker-optimizer.service
sudo systemctl enable beechunker-monitor.service
```

5. Start the services:

```bash
sudo systemctl start beechunker-monitor
sudo systemctl start beechunker-optimizer
```

Note: You may need to adjust the paths in the ExecStart lines if your beechunker executables are installed in a different location. You can find the actual location with `which beechunker-optimizer`.

### Service Management

After setting up the systemd services, you can manage them as follows:

Start all services:

```bash
sudo systemctl start beechunker-monitor
sudo systemctl start beechunker-optimizer
```

Stop all services:

```bash
sudo systemctl stop beechunker-monitor
sudo systemctl stop beechunker-optimizer
```

Purge all BeeChunker data (use with caution):

```bash
# Stop all services first
sudo systemctl stop beechunker-monitor
sudo systemctl stop beechunker-optimizer

# Remove database and logs
sudo rm -rf /opt/beechunker/data/access_patterns.db
sudo rm -rf /opt/beechunker/data/logs/*

# Remove trained models
sudo rm -rf /opt/beechunker/data/models/*

# Reset configuration (optional)
sudo cp beechunker/common/default_config.json /opt/beechunker/data/config.json
```

### Monitor Service

The monitoring service tracks file access patterns in your BeeGFS file system.

Start the monitoring service as a systemd service:

```bash
sudo systemctl start beechunker-monitor
```

Or run it manually:

```bash
beechunker-monitor run
```

View monitoring statistics:

```bash
beechunker-monitor stats
```

Clean up old monitoring data:

```bash
beechunker-monitor cleanup --days 30
```

### SOM Trainer

Train the SOM model with your collected access pattern data:

```bash
beechunker-train train --input-csv /path/to/training_data.csv
```

Or let it load data from the monitoring database:

```bash
beechunker-train train
```

Predict a chunk size for a specific file:

```bash
beechunker-train predict --file-size 1073741824 --read-count 100 --write-count 20
```

Set up automatic periodic training with cron:

```bash
# Edit crontab to run training daily at 2:00 AM
crontab -e
# Add the line:
0 2 * * * /usr/bin/beechunker-train train
```

### Optimizer

The optimizer service applies the trained model to optimize file chunk sizes.

Start the optimization service as a systemd service:

```bash
sudo systemctl start beechunker-optimizer
```

Or run it manually:

```bash
beechunker-optimizer run
```

Optimize a single file:

```bash
beechunker-optimizer optimize-file /path/to/file
```

Optimize all files in a directory:

```bash
beechunker-optimizer optimize-dir /path/to/directory --recursive
```

Analyze a file without changing it:

```bash
beechunker-optimizer analyze /path/to/file
```

Bulk optimize files based on database query:

```bash
beechunker-optimizer bulk-optimize --min-access 10 --min-size 104857600
```

Show optimization statistics:

```bash
beechunker-optimizer get-stats
```

### Making Predictions

Predict the optimal chunk size for a file with specific characteristics:

```bash
beechunker-train predict --file-size 104857600 --read-size 8192 --write-size 4096 --read-count 50 --write-count 10 --extension .csv
```

## Integration with BeeGFS

BeeChunker integrates with BeeGFS through the following mechanisms:

1. **File Access Monitoring**: Uses watchdog to track file operations
2. **Chunk Size Management**: Uses BeeGFS command-line tools to get and set chunk sizes
3. **Default Pattern Setting**: Sets default chunk patterns for directories based on predicted optimal sizes

## System Components

The system is organized into the following components:

```
beechunker/
├── cli/                    # Command-line interfaces
│   ├── monitor_cli.py      # Monitor service CLI
│   ├── optimizer_cli.py    # Optimizer service CLI
│   └── trainer_cli.py      # Trainer service CLI
├── common/                 # Common utilities
│   ├── beechunker_logging.py  # Logging setup
│   ├── config.py           # Configuration management
│   └── default_config.json # Default configuration
├── ml/                     # Machine learning components
│   ├── feature_engineering.py # Feature engineering
│   ├── som.py              # Self-Organizing Map implementation
│   └── visualization.py    # Visualization tools
├── monitor/                # Monitoring components
│   ├── access_tracker.py   # File access tracking
│   └── db_manager.py       # Database management
└── optimizer/              # Optimization components
    ├── chunk_manager.py    # Chunk size management
    └── file_watcher.py     # New file detection
```

## How It Works

1. **Data Collection**: The monitor service tracks file accesses and stores them in the database.

2. **Feature Engineering**: Features are extracted from the raw data, including:
   - File size
   - Average read/write sizes
   - Read/write ratios
   - File extensions
   - Directory depth

3. **SOM Training**: The SOM learns correlations between file features and optimal chunk sizes.

4. **Optimization**: Files are analyzed and their chunk sizes are optimized using BeeGFS commands.

## Troubleshooting

### Common Issues

1. **Permissions Issues**: 
   - Many operations require root access, especially when modifying chunk sizes
   - Ensure the user running BeeChunker has proper permissions on BeeGFS mount points

2. **Missing Database**: 
   - If you get "Database not found" errors, make sure the monitor service has run at least once
   - You can manually create the required directories: `sudo mkdir -p /opt/beechunker/data/logs`

3. **Model Training Failures**:
   - Ensure you have at least 100 file access records (default minimum)
   - Check logs at `/opt/beechunker/data/logs/trainer.log`

4. **Optimizer Not Finding BeeGFS Command**:
   - Ensure `beegfs-ctl` is in your PATH
   - Try running with sudo if you have permission issues

### Checking Service Status

Check service status and logs:

```bash
# Check service status
sudo systemctl status beechunker-monitor
sudo systemctl status beechunker-optimizer

# View service logs
journalctl -u beechunker-monitor.service
journalctl -u beechunker-optimizer.service

# View application logs
tail -f /opt/beechunker/data/logs/monitor.log
tail -f /opt/beechunker/data/logs/optimizer.log
tail -f /opt/beechunker/data/logs/trainer.log
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- BeeGFS team for the excellent parallel filesystem
- MiniSom library for the Self-Organizing Map implementation

