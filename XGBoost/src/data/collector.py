import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import Dict, Any, List, Optional
import subprocess
import json
import os
from ..config import config

class MetricCollector:
    def __init__(self):
        """Initialize the metric collector with InfluxDB connection"""
        self.client = influxdb_client.InfluxDBClient(
            url=config['beegfs'].INFLUXDB_URL,
            token=config['beegfs'].INFLUXDB_TOKEN,
            org=config['beegfs'].INFLUXDB_ORG
        )
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics from BeeGFS"""
        query = f'''
        from(bucket: "{config['beegfs'].INFLUXDB_BUCKET}")
          |> range(start: -{config['metrics'].HISTORY_WINDOW})
          |> filter(fn: (r) => r["_measurement"] == "beegfs_metrics")
          |> last()
        '''
        
        result = self.query_api.query(query)
        metrics = {metric: 0.0 for metric in config['metrics'].METRICS_TO_COLLECT}
        
        for table in result:
            for record in table.records:
                metric_name = record.get_field()
                if metric_name in metrics:
                    metrics[metric_name] = float(record.get_value())
        
        return metrics

    def collect_file_metrics(self, file_path: str) -> Dict[str, Any]:
        """Collect metrics specific to a file using beegfs-ctl"""
        try:
            # Get file attributes using beegfs-ctl
            cmd = f"beegfs-ctl --getentryinfo {file_path}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            file_info = self._parse_beegfs_ctl_output(result.stdout)
            
            # Get standard file stats
            stats = os.stat(file_path)
            
            return {
                'file_size': stats.st_size,
                'file_type': os.path.splitext(file_path)[1],
                'last_access_time': stats.st_atime,
                'last_modified_time': stats.st_mtime,
                'stripe_pattern': file_info.get('stripe_pattern', 'unknown'),
                'current_chunk_size': file_info.get('chunk_size', 0),
                'num_targets': file_info.get('num_targets', 0),
                'is_compressed': self._check_if_compressed(file_path),
                'access_frequency': self._estimate_access_frequency(file_path)
            }
        except Exception as e:
            print(f"Error collecting file metrics: {e}")
            return {}

    def _parse_beegfs_ctl_output(self, output: str) -> Dict[str, Any]:
        """Parse the output of beegfs-ctl command"""
        info = {}
        try:
            for line in output.split('\n'):
                if 'Stripe pattern:' in line:
                    info['stripe_pattern'] = line.split(':')[1].strip()
                elif 'ChunkSize:' in line:
                    size_str = line.split(':')[1].strip()
                    info['chunk_size'] = int(size_str.replace('KB', ''))
                elif 'Number of storage targets:' in line:
                    info['num_targets'] = int(line.split(':')[1].strip())
        except Exception as e:
            print(f"Error parsing beegfs-ctl output: {e}")
        return info

    def _check_if_compressed(self, file_path: str) -> bool:
        """Check if file is compressed based on extension or content"""
        compressed_extensions = {'.gz', '.zip', '.bz2', '.xz', '.7z', '.rar'}
        return os.path.splitext(file_path)[1].lower() in compressed_extensions

    def _estimate_access_frequency(self, file_path: str) -> float:
        """Estimate file access frequency from historical data"""
        query = f'''
        from(bucket: "{config['beegfs'].INFLUXDB_BUCKET}")
          |> range(start: -{config['metrics'].HISTORY_WINDOW})
          |> filter(fn: (r) => r["_measurement"] == "file_access" and r["path"] == "{file_path}")
          |> count()
        '''
        
        result = self.query_api.query(query)
        count = 0
        for table in result:
            for record in table.records:
                count = record.get_value()
        
        # Normalize to accesses per hour
        hours = float(config['metrics'].HISTORY_WINDOW.replace('h', ''))
        return count / max(hours, 1) 