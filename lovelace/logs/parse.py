import os
import re
import csv
from collections import defaultdict

# Regex to find the timestamp in the filename (e.g., 20260318_213323)
FILE_TIME_PATTERN = re.compile(r'_(\d{8}_\d{6})')

# Regex to capture the specific metrics from the log line
METRIC_PATTERN = re.compile(
    r"Avg prompt throughput: (?P<prompt_tp>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<gen_tp>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, "
    r"Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<gpu_kv>[\d.]+)%, "
    r"Prefix cache hit rate: (?P<prefix_hit>[\d.]+)%, "
    r"MM cache hit rate: (?P<mm_hit>[\d.]+)%"
)

# Mapping internal keys to your specific descriptive headers
HEADER_MAPPING = {
    'prompt_tp': 'Avg prompt throughput (token/s)',
    'gen_tp': 'Avg generation throughput (token/s)',
    'running': 'Running (reqs)',
    'waiting': 'Waiting (reqs)',
    'gpu_kv': 'GPU KV cache usage (%)',
    'prefix_hit': 'Prefix cache hit rate (%)',
    'mm_hit': 'MM cache hit rate (%)'
}

def parse_logs(directory):
    aggregated_data = defaultdict(list)

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return aggregated_data

    for filename in os.listdir(directory):
        if filename.endswith(".log"):
            time_match = FILE_TIME_PATTERN.search(filename)
            if not time_match:
                continue
            
            group_timestamp = time_match.group(1)
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as f:
                for line in f:
                    metric_match = METRIC_PATTERN.search(line)
                    if metric_match:
                        # Extract raw data
                        raw_data = metric_match.groupdict()
                        
                        # Create entry with descriptive headers
                        formatted_entry = {
                            'Log Group Timestamp': group_timestamp,
                            'Source File': filename
                        }
                        
                        # Map the regex groups to your specific column names
                        for key, new_label in HEADER_MAPPING.items():
                            formatted_entry[new_label] = raw_data[key]
                            
                        aggregated_data[group_timestamp].append(formatted_entry)

    return aggregated_data

def save_to_csv(aggregated_data, output_file):
    if not aggregated_data:
        print("No matching log data found to save.")
        return

    # Define the order of columns in the CSV
    fieldnames = [
        'Log Group Timestamp', 
        'Source File',
        'Avg prompt throughput (token/s)',
        'Avg generation throughput (token/s)',
        'Running (reqs)',
        'Waiting (reqs)',
        'GPU KV cache usage (%)',
        'Prefix cache hit rate (%)',
        'MM cache hit rate (%)'
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by timestamp for chronological order
        for timestamp in sorted(aggregated_data.keys()):
            for entry in aggregated_data[timestamp]:
                writer.writerow(entry)

if __name__ == "__main__":
    # Change '.' to the path where your .log files are located
    target_folder = './' 
    output_filename = 'aggregated_metrics.csv'
    
    log_results = parse_logs(target_folder)
    save_to_csv(log_results, output_filename)
    
    if log_results:
        print(f"Done! Results written to {output_filename}")
