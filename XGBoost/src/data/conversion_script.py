import pandas as pd

# Define the mapping of old column names to new column names
column_mapping = {
    'file_path': 'file_path',  # unchanged, but will be dropped
    'file_size': 'file_size_KB',  # already in KB
    'chunk_size': 'chunk_size_KB',  # already in KB
    'access_count': 'access_count',  # unchanged
    'access_count_label': 'access_count_label',  # will be dropped
    'avg_read_size': 'avg_read_KB',  # already in KB
    'avg_write_size': 'avg_write_KB',  # already in KB
    'max_read_size': 'max_read_KB',  # already in KB
    'max_write_size': 'max_write_KB',  # already in KB
    'read_count': 'read_ops',  # rename
    'write_count': 'write_ops',  # rename
    'throughput_mbps': 'throughput_KBps'  # convert to Kbps
}

# Load the CSV file
df = pd.read_csv('beegfs_test_results4.csv')

# Rename the columns
df.rename(columns=column_mapping, inplace=True)

# Drop the 'file_path' and 'access_count_label' columns
df.drop(columns=['file_path', 'access_count_label'], inplace=True)

# Convert throughput from Mbps to Kbps (1 Mbps = 1000 Kbps)
df['throughput_KBps'] = df['throughput_KBps'] * 1000

# Save the updated DataFrame to a new CSV file
df.to_csv('beegfs_test_results4_updated.csv', index=False)

print("Column names updated, 'file_path' and 'access_count_label' dropped, and throughput converted. New file saved as 'beegfs_test_results4_updated.csv'.")