# BeeChunker Model Comparison Script (model_comparison.py)

## Overview

This script rigorously tests and compares the performance of Random Forest (RF) and XGBoost (XGB) models for chunk size optimization in BeeGFS file systems. It provides detailed performance metrics, visualizations, and statistical analysis to determine which model delivers superior throughput improvements across various scenarios.

## Testing Parameters

### File Sizes
- Small: 10 MB
- Medium: 100 MB
- Large: 1 GB

### Data Patterns
- Random: Files created with random data
- Sequential: Files created with sequential byte patterns

### Access Patterns
- Read-heavy: Predominantly read operations (3x reads, 0.5x writes)
- Write-heavy: Predominantly write operations (0.5x reads, 3x writes)
- Mixed: Balanced read and write operations
- Sequential: Large, sequential read/write operations (4x larger IO sizes)
- Random: Small, random read/write operations (0.5x smaller IO sizes, 2x more operations)

## Methodology

1. **Test File Creation**: For each combination of file size and data pattern, a test file is created in a dedicated test directory.

2. **Access Pattern Simulation**: 
   - Realistic file access patterns are simulated using both actual file operations with `dd` commands and direct entries in the monitoring database.
   - Read and write operations are executed with appropriate sizes and frequencies to match the intended access pattern.

3. **Performance Measurement**:
   - Pre-optimization baseline performance is measured using real I/O operations with precise timing.
   - Performance metrics include read speed (MB/s), write speed (MB/s), and overall throughput.
   
4. **Model-based Optimization**:
   - Each model (RF and XGB) predicts an optimal chunk size for the file based on its access patterns.
   - The chunk size is applied to the file using BeeGFS tools.
   
5. **Post-optimization Performance**:
   - Performance measurements are repeated after optimization.
   - Improvement percentages are calculated for read, write, and overall throughput.

6. **Comprehensive Analysis**:
   - Multiple trials are run for each scenario to ensure statistical validity.
   - T-tests are performed to determine statistical significance of performance differences.
   - Detailed performance tables are generated for each testing dimension.
   - Visualizations are created showing performance across different variables.

## Output and Analysis

The script generates:

1. **Performance Plots**:
   - Throughput improvement by file size and model
   - Chunk size selection patterns
   - Performance by access pattern
   - Read vs. write performance comparison
   - Overall model comparison

2. **Summary Statistics**:
   - Overall performance by model
   - Performance breakdown by file size
   - Performance breakdown by access pattern
   - Read vs. write speed improvements
   - Chunk size selection patterns

3. **Final Verdict**:
   - Category-by-category comparison with winners
   - Statistical significance of performance differences
   - Overall recommendation based on comprehensive analysis

## Practical Applications

This detailed analysis helps BeeGFS administrators make evidence-based decisions about which model to deploy for optimal file system performance based on their specific workloads and usage patterns.

# Results

## Final Model Comparison

| Category            | RF Performance   | XGBoost Performance   | Winner   |
|---------------------|------------------|----------------------|----------|
| Overall Performance | 7.14%            | 15.83%               | XGBoost  |
| 10MB Files          | 1.75%            | 11.46%               | XGBoost  |
| 100MB Files         | 8.36%            | 19.27%               | XGBoost  |
| 1024MB Files        | 11.31%           | 16.75%               | XGBoost  |
| Mixed Access        | 8.87%            | 12.82%               | XGBoost  |
| Random Access       | 12.84%           | 16.35%               | XGBoost  |
| Read Heavy Access   | 5.85%            | 16.50%               | XGBoost  |
| Sequential Access   | 4.36%            | 18.79%               | XGBoost  |
| Write Heavy Access  | 3.78%            | 14.67%               | XGBoost  |
| Read Operations     | 8.59%            | 18.44%               | XGBoost  |
| Write Operations    | 3.60%            | 8.81%                | XGBoost  |

## Final Verdict
XGBoost wins in 11 out of 11 categories!
XGBoost provides better overall chunk size optimization.