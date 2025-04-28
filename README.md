# BeeChunker

**Intelligent Chunk Size Optimization for BeeGFS using Machine Learning Models**

BeeChunker is an intelligent system for optimizing chunk sizes in BeeGFS (Fraunhofer Parallel File System) storage systems by analyzing file access patterns and predicting optimal chunk sizes using various machine learning models including ~~Self-Organizing Maps (SOM)~~, Random Forest (RF), and XGBoost.

## Overview

BeeGFS is a parallel file system that distributes file data across multiple storage servers using chunks. The "chunk size" determines how a file is divided and distributed, which significantly impacts I/O performance. However, determining the optimal chunk size for a file is challenging as it depends on many factors:

- File size
- Access patterns (read vs. write ratio)
- I/O operation sizes
- Workload characteristics
- File type and extension

BeeChunker solves this problem by:
1. Continuously monitoring file access patterns
2. Training machine learning models on the collected data
3. Predicting optimal chunk sizes based on file characteristics
4. Automatically applying these optimizations to existing and new files

## System Requirements

- Python 3.8 or higher
- **BeeGFS installation** (critical requirement)
- Access to BeeGFS command-line tools (`beegfs-ctl`)
- Root or sudo access (for some operations)
- Sufficient disk space for the monitoring database and log files

## Codebase Structure

The BeeChunker codebase is organized into several key components:

```
beechunker/
├── cli/                    # Command-line interfaces
│   ├── monitor_cli.py      # Monitor service CLI for tracking file access
│   ├── optimizer_cli.py    # Optimizer service CLI for applying chunk optimizations
│   └── trainer_cli.py      # Trainer service CLI for ML model training
├── common/                 # Common utilities
│   ├── beechunker_logging.py # Logging setup
│   ├── config.py           # Configuration management
│   ├── default_config.json # Default configuration
│   └── file_access_event.py # File access event class
├── ml/                     # Machine learning components
│   ├── feature_engineering.py # Feature engineering
│   ├── feature_extraction.py  # Extract features from raw data
│   ├── random_forest.py    # Random Forest model
│   ├── som.py              # Self-Organizing Map implementation ** NOT USED IN THE FINAL IMPLEMENTATION **
│   ├── visualization.py    # Visualization tools
│   └── xgboost_model.py    # XGBoost model implementation
├── monitor/                # Monitoring components
│   ├── access_tracker.py   # File access tracking
│   └── db_manager.py       # Database management
└── optimizer/              # Optimization components
    ├── chunk_manager.py    # Chunk size management
    └── file_watcher.py     # New file detection
```

## Architecture and Workflow

[image](architecture.png)

BeeChunker consists of three main services that work together:

### 1. Monitor Service

The monitor service (`monitor_cli.py`) continuously watches BeeGFS mount points to track file access operations:

- Uses the `watchdog` library to detect file operations (read/write)
- Captures file metadata including current chunk size using `beegfs-ctl --getentryinfo`
- Records access patterns, read/write operations, and performance metrics in a SQLite database
- Handles cleanup of old monitoring data to prevent database bloat

### 2. Trainer Service

The trainer service (`trainer_cli.py`) analyzes the collected data to train machine learning models:

- Supports multiple ML models:
  - **Random Forest (RF)**: Ensemble learning method using decision trees for accurate chunk size prediction
  - **XGBoost**: Gradient boosting implementation for high-performance predictions
  - ~~**Self-Organizing Map (SOM)**: Unsupervised neural network creating a topological mapping of file access patterns~~
- Processes raw access data through extensive feature engineering
- Creates visualizations of model insights (U-Matrix, Component Planes, Cluster Analysis)
- Saves trained models to disk for use by the optimizer

### 3. Optimizer Service

The optimizer service (`optimizer_cli.py`) applies ML predictions to optimize file chunk sizes:

- Extracts features from files matching what the models were trained on
- Uses the trained models to predict optimal chunk sizes
- Creates new files with the optimal chunk size and swaps them in place
- Tracks optimization history and performance improvements
- Can be run continuously (service mode) or on-demand (CLI mode)

## Data Flow

1. **Data Collection**: The monitor service records file access operations in the SQLite database
2. **Feature Engineering**: Raw data is processed to extract relevant features:
   - File size and current chunk size
   - Read/write operation counts and ratios
   - Average and maximum read/write sizes
   - File extension characteristics
   - Access patterns and throughput metrics
3. **Model Training**: The trainer service processes this data to train models that correlate file characteristics with optimal chunk sizes
4. **Prediction & Optimization**: The optimizer applies these models to predict and set optimal chunk sizes for files

## Installation

1. Extract the contents of the submitted zip file:

```bash
unzip beechunker.zip
cd BeeChunker
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

4. Create necessary directories:

```bash
sudo mkdir -p /opt/beechunker/data/logs
sudo mkdir -p /opt/beechunker/data/models
sudo chown -R $USER:$USER /opt/beechunker
```

5. Configure BeeGFS mount points in a custom configuration file: (not a requirement as the default config will be applied in case the user doesnt manually define a config file)

```bash
mkdir -p /opt/beechunker/data
cat > /opt/beechunker/data/config.json << EOL
{
  "monitor": {
    "db_path": "/opt/beechunker/data/access_patterns.db",
    "log_path": "/opt/beechunker/data/logs/monitor.log",
    "polling_interval": 300
  },
  "optimizer": {
    "log_path": "/opt/beechunker/data/logs/optimizer.log",
    "min_chunk_size": 64,
    "max_chunk_size": 4096
  },
  "ml": {
    "models_dir": "/opt/beechunker/data/models",
    "log_path": "/opt/beechunker/data/logs/trainer.log",
    "training_interval": 86400,
    "min_training_samples": 100,
    "som_iterations": 5000,
    "n_estimators": 100,
    "hgb_iter": 1000,
    "ot_quantile": 0.65
  },
  "beegfs": {
    "mount_points": [
      "/mnt/beegfs"
    ]
  }
}
EOL
```

## Running the Services

BeeChunker provides three main services that can be run independently or as systemd services.

### Setting Up Systemd Services

The project includes example service files in the `services/` directory. You'll need to adapt these to your environment.

#### 1. Monitor Service

1. Copy and modify the example service file:

```bash
sudo cp services/beechunker-monitor.service.example /etc/systemd/system/beechunker-monitor.service
sudo nano /etc/systemd/system/beechunker-monitor.service
```

2. Update the service file with your specific paths and username:

```
[Unit]
Description=BeeChunker Monitor Service
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME  # Replace with your username
WorkingDirectory=/path/to/BeeChunker  # Replace with your path
Environment="BEECHUNKER_CONFIG=/opt/beechunker/data/config.json"
ExecStart=/path/to/BeeChunker/venv/bin/python /path/to/BeeChunker/beechunker/cli/monitor_cli.py run
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable beechunker-monitor.service
sudo systemctl start beechunker-monitor.service
```

4. Check the service status:

```bash
sudo systemctl status beechunker-monitor.service
```

#### 2. Optimizer Service

1. Copy and modify the example service file:

```bash
sudo cp services/beechunker-optimizer.service.example /etc/systemd/system/beechunker-optimizer.service
sudo nano /etc/systemd/system/beechunker-optimizer.service
```

2. Update the service file with your specific paths and username:

```
[Unit]
Description=BeeChunker Optimizer Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME  # Replace with your username
WorkingDirectory=/path/to/BeeChunker  # Replace with your path
Environment="BEECHUNKER_CONFIG=/opt/beechunker/data/config.json"
ExecStart=/path/to/BeeChunker/venv/bin/python /path/to/BeeChunker/beechunker/cli/optimizer_cli.py run
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable beechunker-optimizer.service
sudo systemctl start beechunker-optimizer.service
```

4. Check the service status:

```bash
sudo systemctl status beechunker-optimizer.service
```

#### 3. Trainer Service

1. Copy and modify the example service file:

```bash
sudo cp services/beechunker-trainer.service.example /etc/systemd/system/beechunker-trainer.service
sudo nano /etc/systemd/system/beechunker-trainer.service
```

2. Update the service file with your specific paths and username:

```
[Unit]
Description=BeeChunker Trainer Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME  # Replace with your username
WorkingDirectory=/path/to/BeeChunker  # Replace with your path
Environment="BEECHUNKER_CONFIG=/opt/beechunker/data/config.json"
ExecStart=/path/to/BeeChunker/venv/bin/python /path/to/BeeChunker/beechunker/cli/trainer_cli.py train
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable beechunker-trainer.service
sudo systemctl start beechunker-trainer.service
```

4. Check the service status: (might show not running or stopped which is normal)

```bash
sudo systemctl status beechunker-trainer.service
```

### Creating Symbolic Links

To run BeeChunker commands easily from anywhere in your system, you can create symbolic links to the main CLI scripts. This allows you to use commands like `beechunker-monitor` instead of the full path.

```bash
# Create symbolic links in a directory that's in your PATH
sudo ln -s $(pwd)/beechunker/cli/monitor_cli.py /usr/local/bin/beechunker-monitor
sudo ln -s $(pwd)/beechunker/cli/optimizer_cli.py /usr/local/bin/beechunker-optimizer
sudo ln -s $(pwd)/beechunker/cli/trainer_cli.py /usr/local/bin/beechunker-trainer

# Make them executable
sudo chmod +x /usr/local/bin/beechunker-monitor
sudo chmod +x /usr/local/bin/beechunker-optimizer
sudo chmod +x /usr/local/bin/beechunker-trainer
```

After creating these symbolic links, you can run commands like:

```bash
beechunker-monitor run
beechunker-optimizer optimize-file /path/to/file
```

## Setting Up Cron Jobs

Although the trainer service currently has limitations with model compatibility, you can still set up a cron job to run the trainer periodically. This may be useful for future versions when these issues are resolved.

```bash
# Edit crontab file
crontab -e
```

Add one of the following lines to the crontab file:

```
# Run the trainer daily at 2:00 AM
0 2 * * * /path/to/BeeChunker/venv/bin/python /path/to/BeeChunker/beechunker/cli/trainer_cli.py train >> /opt/beechunker/data/logs/cron_trainer.log 2>&1

# Or, if you created symbolic links:
0 2 * * * /usr/local/bin/beechunker-trainer train >> /opt/beechunker/data/logs/cron_trainer.log 2>&1
```

Save the file to set up the cron job. Remember that due to the current limitations in the trainer service, this cron job might not work correctly until the preprocessing compatibility issues are resolved.

## Running Services from Command Line

If you prefer to run the services manually or for testing purposes, you can run them directly from the command line:

#### Monitor Service

```bash
# Start the monitoring service
python beechunker/cli/monitor_cli.py run

# Check monitoring statistics
python beechunker/cli/monitor_cli.py stats

# Clean up old data (keep last 30 days)
python beechunker/cli/monitor_cli.py cleanup --days 30
```

#### Trainer Service

```bash
# Train models using data from the database
python beechunker/cli/trainer_cli.py train

# Train using a specific CSV file
python beechunker/cli/trainer_cli.py train --input-csv /path/to/data.csv

# Make a prediction for a specific file
python beechunker/cli/trainer_cli.py predict --file-size 1073741824 --read-count 100 --write-count 20
```

#### Optimizer Service

```bash
# Run optimizer service continuously
python beechunker/cli/optimizer_cli.py run

# Optimize a single file
python beechunker/cli/optimizer_cli.py optimize-file /path/to/file

# Choose a specific model type (rf, som, or xgb)
python beechunker/cli/optimizer_cli.py optimize-file /path/to/file --model-type xgb

# Optimize all files in a directory
python beechunker/cli/optimizer_cli.py optimize-dir /path/to/directory --recursive

# Analyze without changing anything (dry run)
python beechunker/cli/optimizer_cli.py optimize-dir /path/to/directory --dry-run

# Analyze file to show predicted chunk size without changing
python beechunker/cli/optimizer_cli.py analyze /path/to/file

# Bulk optimize based on database query
python beechunker/cli/optimizer_cli.py bulk-optimize --min-access 10
```

## Model Comparison and Demo

BeeChunker includes two utility scripts to demonstrate and evaluate the system:

### Model Status and Limitations

**Important Notes on Model Availability:**

- **SOM Model (DEPRECATED)**: The Self-Organizing Map model was implemented as a proof of concept and is not intended for production use. It should be considered deprecated and unusable for actual optimization.

- **Trainer Service Limitations**: The trainer service currently has compatibility issues between models due to different preprocessing requirements. This has not been fully resolved yet, so automatic training of models may not work as expected. Manual model comparisons and testing are recommended instead.

- **Recommended Model**: The Random Forest (RF) model is currently the most stable and recommended model for production use. XGBoost is available for experimental comparisons.

### Demo Script

The `demo.py` script demonstrates the system by creating test files, simulating access patterns, and showing performance improvements from chunk size optimization:

```bash
# Run the demo using the Random Forest model (default)
python demo.py --model rf

# Run the demo using the XGBoost model
python demo.py --model xgb
```

### Model Comparison

The `model_comparison.py` script runs a comprehensive comparison between the Random Forest and XGBoost models:

```bash
# Run the model comparison with 3 trials per scenario (default)
python model_comparison.py

# Run with more trials for more robust results
python model_comparison.py --trials 5
```

This will generate detailed comparison plots in the `comparison_plots/` directory and print a summary of model performance.

## Monitoring and Troubleshooting

### Checking Service Status

```bash
# Check service status
sudo systemctl status beechunker-monitor
sudo systemctl status beechunker-optimizer
sudo systemctl status beechunker-trainer

# View service logs
journalctl -u beechunker-monitor.service
journalctl -u beechunker-optimizer.service
journalctl -u beechunker-trainer.service

# View application logs
tail -f /opt/beechunker/data/logs/monitor.log
tail -f /opt/beechunker/data/logs/optimizer.log
tail -f /opt/beechunker/data/logs/trainer.log
```

### Common Issues and Solutions

1. **Missing BeeGFS Tools**:
   - Error: "Command 'beegfs-ctl' not found"
   - Solution: Ensure BeeGFS is installed and tools are in PATH

2. **Permission Issues**:
   - Error: "Permission denied"
   - Solution: Run with sudo or adjust file permissions

3. **Database Issues**:
   - Error: "Database not found" or "no such table"
   - Solution: Ensure monitor service has run at least once to create database schema

4. **Model Training Failures**:
   - Error: "Not enough samples for training"
   - Solution: Gather more access data (at least 100 samples by default)

5. **Service Won't Start**:
   - Check logs: `journalctl -u beechunker-monitor.service -n 50`
   - Verify paths in service file

## Advanced Configuration

The BeeChunker configuration file supports many customization options:

```json
{
  "monitor": {
    "db_path": "/opt/beechunker/data/access_patterns.db",
    "log_path": "/opt/beechunker/data/logs/monitor.log",
    "polling_interval": 300  // Check interval in seconds
  },
  "optimizer": {
    "log_path": "/opt/beechunker/data/logs/optimizer.log",
    "min_chunk_size": 64,    // Minimum chunk size in KB
    "max_chunk_size": 4096   // Maximum chunk size in KB
  },
  "ml": {
    "models_dir": "/opt/beechunker/data/models",
    "log_path": "/opt/beechunker/data/logs/trainer.log",
    "training_interval": 86400,  // Training interval in seconds (daily)
    "min_training_samples": 100, // Minimum samples required for training
    "som_iterations": 5000,      // SOM training iterations
    "n_estimators": 100,         // RF tree count
    "hgb_iter": 1000,            // XGBoost iterations
    "ot_quantile": 0.65          // Optimal Throughput quantile threshold
  },
  "beegfs": {
    "mount_points": ["/mnt/beegfs"]  // BeeGFS mount points to monitor
  }
}
```

## Performance Considerations

- **Database Size**: The monitor database can grow large over time. Use the cleanup function regularly.
- **CPU Usage**: Model training can be CPU intensive. Consider running training during off-peak hours.
- **Storage Overhead**: Changing chunk sizes creates temporary files. Ensure sufficient free space.
- **Optimization Frequency**: Frequent chunk size changes can cause overhead. Use appropriate thresholds.

## Implementation Details

### Monitor Service

The monitor service tracks file operations using the watchdog library and BeeGFS command-line tools:

- Uses file system event handlers to detect read/write operations
- Records access events in a SQLite database
- Tracks file metadata including size, chunk size, and access patterns
- Maintains separate tables for file metadata, access events, and throughput metrics

### Machine Learning Models

BeeChunker implements different ML models for chunk size prediction:

1. **Random Forest (RF)** :
   - Ensemble of decision trees for robust classification
   - Stacks multiple RF models for higher accuracy
   - Includes feature importance analysis
   - Most stable and reliable model for production use

2. **XGBoost** :
   - Gradient boosting implementation for high accuracy
   - Fast prediction with low memory footprint
   - Handles complex feature interactions well
   - Currently in experimental stage

3. **Self-Organizing Map (SOM)** (DEPRECATED):
   - Was implemented as a proof of concept only
   - Not intended for production use
   - Included in the codebase for academic purposes only
   - **Should NOT be used for actual chunk size optimization**

### Optimizer Implementation

The optimizer uses a sophisticated approach to change chunk sizes:

1. Extracts features from the file to predict optimal chunk size
2. Uses BeeGFS tools to create a new file with the optimal chunk size
3. Copies data from the original file to the new file
4. Performs an atomic swap to replace the original file
5. Records the optimization in the database for tracking

### Models (pre-trained)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License - see the LICENSE file for details.

## Acknowledgments

- BeeGFS team for their excellent parallel filesystem
- MiniSom library for the Self-Organizing Map implementation
- The scikit-learn and XGBoost teams for their machine learning libraries

## Team Contributions (Listed either by files or entire directories) (all paths are relative to the root path of BeeChunker)

1. Jayesh Bhagyesh Gajbhar (jgajbha) - 
- beechunker/cli/
- beechunker/common/
- beechunker/custom_types
- beechunker/ml/feature_engineering.py beechunker/ml/som.py beechunker/ml/visualization.py (These files are not being used in the final implementation)
- beechunker/monitor/
- beechunker/optimizer/
- setup.py
- demo.py
- model_comparison.py
- services/
- models/chunk_size_map.npy models/som_model.joblib

2. Aryan Gupta (agupta72) -
- beechunker/ml/random_forest.py
- beechunker/ml/feature_extraction.py
- data/
- models/rf_base.joblib
- models/rf_model.joblib

3. Tanishq Virendrabhai Todkar (ttodkar)
- beechunker/ml/xgboost_feature_engine.py
- beechunker/ml/xgboost_model.py