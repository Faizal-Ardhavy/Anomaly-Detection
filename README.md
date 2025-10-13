# Anomaly Detection with Multiple Datasets

This project uses BERT embeddings and KMeans clustering to detect anomalies in log files. It now supports processing multiple datasets for more comprehensive anomaly detection.

## Features

- **Transform raw log strings to semantic vectors** using BERT embeddings
- **Support for multiple log datasets** (Apache, Nginx, System logs, etc.)
- **KMeans clustering** for anomaly detection
- **Flexible dataset loading** - load from single file, multiple files, or glob patterns

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Embeddings from Log Files (bert.py)

The `bert.py` script processes log files and generates embeddings:

```python
# Edit the log_files list in bert.py to add your log files
log_files = [
    r"..\dataset\Apache.log",
    r"..\dataset\Nginx.log",
    r"..\dataset\System.log",
]

# Run the script
python bert.py
```

This will generate `.npy` files for each log file:
- `Apache_embeddings.npy`
- `Nginx_embeddings.npy`
- `System_embeddings.npy`

### 2. Anomaly Detection with KMeans (kmeans.py)

The `kmeans.py` script provides a comprehensive class for anomaly detection:

#### Basic Usage

```python
from kmeans import LogAnomalyDetector

# Initialize the detector
detector = LogAnomalyDetector()

# Method 1: Load all .npy files in the current directory
detector.load_embeddings('*.npy')

# Method 2: Load specific files
detector.load_embeddings(['apache_embeddings.npy', 'nginx_embeddings.npy'])

# Method 3: Load a single file
detector.load_embeddings('apache_embeddings.npy')

# Train KMeans clustering
detector.train_kmeans(n_clusters=5)

# Check if a log is anomalous
test_log = "User login failed due to wrong password"
is_anomaly, distance, cluster_id, threshold = detector.predict_anomaly(test_log)

print(f"Is Anomaly: {is_anomaly}")
print(f"Cluster ID: {cluster_id}")
print(f"Distance: {distance:.4f}")
```

#### Transform Raw Logs to Vectors

```python
from kmeans import LogAnomalyDetector

detector = LogAnomalyDetector()

# Transform a single log string
log_text = "ERROR: Connection timeout"
vector = detector.transform_log_to_vector(log_text)
print(f"Vector shape: {vector.shape}")  # (768,)

# Transform multiple logs
logs = [
    "INFO: Server started",
    "WARNING: High memory usage",
    "ERROR: Database connection failed"
]
vectors = detector.transform_logs_to_vectors(logs)
print(f"Vectors shape: {vectors.shape}")  # (3, 768)
```

## File Structure

```
├── bert.py                 # Generate embeddings from log files
├── kmeans.py              # KMeans clustering and anomaly detection
├── npyenncoder.py         # Utility to view .npy files
├── requirements.txt       # Project dependencies
└── .gitignore            # Git ignore rules
```

## Key Functions in kmeans.py

### LogAnomalyDetector Class

- `transform_log_to_vector(log_text)`: Transform a single log string to a 768-dimensional vector
- `transform_logs_to_vectors(log_texts)`: Transform multiple log strings to vectors
- `load_embeddings(embedding_files)`: Load embeddings from .npy files (supports glob patterns)
- `train_kmeans(n_clusters)`: Train KMeans clustering model
- `predict_anomaly(log_text, threshold_percentile)`: Predict if a log is anomalous

## Examples

### Example 1: Process Multiple Log Files

```python
# In bert.py, add multiple log files
log_files = [
    r"..\dataset\Apache.log",
    r"..\dataset\Nginx.log",
    r"..\dataset\System.log",
]
```

### Example 2: Detect Anomalies from Multiple Sources

```python
from kmeans import LogAnomalyDetector

# Initialize and load all datasets
detector = LogAnomalyDetector()
detector.load_embeddings('*.npy')  # Load all .npy files

# Train with 5 clusters
detector.train_kmeans(n_clusters=5)

# Test various logs
test_logs = [
    "INFO: Normal operation",
    "ERROR: Unusual system behavior detected",
    "WARNING: Unexpected network traffic pattern"
]

for log in test_logs:
    is_anomaly, distance, cluster_id, threshold = detector.predict_anomaly(log)
    print(f"\nLog: {log}")
    print(f"Anomaly: {is_anomaly}, Cluster: {cluster_id}, Distance: {distance:.4f}")
```

## Notes

- The `.npy` files are excluded from git via `.gitignore`
- Environment files (`.env`, `account.env`) are also excluded
- Make sure to set up your HuggingFace token in `account.env` if using bert.py
- The BERT model (`bert-base-uncased`) generates 768-dimensional embeddings

## Customization

You can customize the anomaly detection by:
- Adjusting `n_clusters` in `train_kmeans()`
- Changing `threshold_percentile` in `predict_anomaly()` (default: 95)
- Using different BERT models in `LogAnomalyDetector(model_name="...")`

## Troubleshooting

### ModuleNotFoundError: No module named 'transformers'
```bash
pip install transformers
```

### ModuleNotFoundError: No module named 'sklearn'
```bash
pip install scikit-learn
```

### File encoding issues with requirements.txt
The requirements.txt file should be UTF-8 encoded. If you have issues, try:
```bash
iconv -f UTF-16LE -t UTF-8 requirements.txt > requirements_utf8.txt
mv requirements_utf8.txt requirements.txt
```
