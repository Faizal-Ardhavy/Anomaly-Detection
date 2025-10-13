# Quick Start Guide - Multi-Dataset Anomaly Detection

## Overview

This project now supports processing and analyzing **multiple log datasets** simultaneously for improved anomaly detection using BERT embeddings and KMeans clustering.

## 🎯 Key Features

### 1. Transform Raw Logs to Semantic Vectors
```python
from kmeans import LogAnomalyDetector

detector = LogAnomalyDetector()

# Single log
vector = detector.transform_log_to_vector("ERROR: Connection failed")
# Returns: numpy array of shape (768,)

# Multiple logs
vectors = detector.transform_logs_to_vectors([
    "INFO: Server started",
    "ERROR: Database error"
])
# Returns: numpy array of shape (n_logs, 768)
```

### 2. Support Multiple Datasets

**Before (Single Dataset):**
```python
# Only Apache logs
np.load('apache_embeddings.npy')
```

**After (Multiple Datasets):**
```python
# Load multiple datasets at once
detector.load_embeddings([
    'apache_embeddings.npy',
    'nginx_embeddings.npy', 
    'system_embeddings.npy'
])

# Or use glob pattern
detector.load_embeddings('*.npy')  # Loads ALL .npy files
```

### 3. Comprehensive Anomaly Detection
```python
# Train on combined datasets
detector.train_kmeans(n_clusters=5)

# Check if a log is anomalous
is_anomaly, distance, cluster_id, threshold = detector.predict_anomaly(
    "Unusual system behavior detected"
)
```

## 📁 File Structure

```
Anomaly-Detection/
│
├── bert.py                     # Generate embeddings from log files
│   └── Now supports multiple log files in a list
│
├── kmeans.py                   # Main anomaly detection class
│   ├── LogAnomalyDetector class
│   ├── transform_log_to_vector()
│   ├── transform_logs_to_vectors()
│   ├── load_embeddings()       # Multi-dataset support
│   ├── train_kmeans()
│   └── predict_anomaly()
│
├── example_multi_dataset.py    # Complete usage examples
│   ├── Example 1: Single dataset
│   ├── Example 2: Multiple specific datasets
│   ├── Example 3: Glob pattern loading
│   ├── Example 4: Transform and detect
│   └── Example 5: Batch transform
│
├── README.md                   # Full documentation
├── npyenncoder.py             # Utility to view .npy files
└── requirements.txt           # Dependencies (includes scikit-learn)
```

## 🚀 Quick Start

### Step 1: Generate Embeddings

Edit `bert.py` to add your log files:

```python
log_files = [
    r"..\dataset\Apache.log",
    r"..\dataset\Nginx.log",      # NEW!
    r"..\dataset\System.log",     # NEW!
]
```

Run:
```bash
python bert.py
```

Output:
```
apache_embeddings.npy
nginx_embeddings.npy
system_embeddings.npy
```

### Step 2: Detect Anomalies

```python
from kmeans import LogAnomalyDetector

# Initialize
detector = LogAnomalyDetector()

# Load ALL datasets
detector.load_embeddings('*.npy')

# Train model
detector.train_kmeans(n_clusters=5)

# Test a log
is_anomaly, distance, cluster_id, threshold = detector.predict_anomaly(
    "Critical system failure"
)

print(f"Anomaly: {is_anomaly}")
```

## 🔧 Configuration Options

### bert.py Configuration

```python
# Add as many log files as needed
log_files = [
    r"..\dataset\Apache.log",
    r"..\dataset\Nginx.log",
    r"..\dataset\System.log",
    r"..\dataset\Application.log",
    # Add more...
]
```

### kmeans.py Configuration

```python
# Different ways to load embeddings

# Option 1: All files
detector.load_embeddings('*.npy')

# Option 2: Specific files
detector.load_embeddings([
    'apache_embeddings.npy',
    'nginx_embeddings.npy'
])

# Option 3: Pattern matching
detector.load_embeddings('*_embeddings.npy')

# Option 4: Single file
detector.load_embeddings('apache_embeddings.npy')
```

### KMeans Parameters

```python
# Adjust number of clusters
detector.train_kmeans(n_clusters=5)  # Default: 5

# Adjust anomaly threshold
detector.predict_anomaly(
    log_text, 
    threshold_percentile=95  # Default: 95 (top 5% are anomalies)
)
```

## 📊 Example Output

```
Loading embeddings from: apache_embeddings.npy
  Loaded 500 embeddings with dimension 768
Loading embeddings from: nginx_embeddings.npy
  Loaded 300 embeddings with dimension 768
Loading embeddings from: system_embeddings.npy
  Loaded 200 embeddings with dimension 768

Total embeddings loaded: 1000
Embedding dimension: 768

Training KMeans with 5 clusters...
KMeans training completed!

Cluster distribution:
  Cluster 0: 230 samples (23.00%)
  Cluster 1: 195 samples (19.50%)
  Cluster 2: 210 samples (21.00%)
  Cluster 3: 180 samples (18.00%)
  Cluster 4: 185 samples (18.50%)
```

## 🎓 Use Cases

### Use Case 1: Multi-Server Monitoring
```python
# Combine logs from different servers
detector.load_embeddings([
    'web_server_embeddings.npy',
    'app_server_embeddings.npy',
    'db_server_embeddings.npy'
])
```

### Use Case 2: Different Log Types
```python
# Combine different types of logs
detector.load_embeddings([
    'access_log_embeddings.npy',
    'error_log_embeddings.npy',
    'security_log_embeddings.npy'
])
```

### Use Case 3: Time-Series Analysis
```python
# Combine logs from different time periods
detector.load_embeddings([
    'logs_january_embeddings.npy',
    'logs_february_embeddings.npy',
    'logs_march_embeddings.npy'
])
```

## 💡 Tips

1. **Performance**: Process large log files in batches using `bert.py`
2. **Clustering**: Start with 5 clusters and adjust based on results
3. **Threshold**: Use 95th percentile for balanced anomaly detection
4. **Memory**: Large embeddings may require significant RAM
5. **Updates**: Retrain the model when adding new datasets

## 🔍 Troubleshooting

### Issue: "No embeddings were loaded"
**Solution**: Check that .npy files exist in the current directory

### Issue: "ValueError: text input must be of type `str`"
**Solution**: Ensure log strings are properly formatted as strings

### Issue: "ModuleNotFoundError: No module named 'sklearn'"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Next Steps

1. ✅ Generate embeddings from your log files using `bert.py`
2. ✅ Run `example_multi_dataset.py` to see all features in action
3. ✅ Customize the number of clusters based on your data
4. ✅ Adjust the anomaly threshold for your use case
5. ✅ Integrate with your monitoring system

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Code Examples**: See `example_multi_dataset.py`
- **Utility Tool**: Use `npyenncoder.py` to inspect .npy files

---

**Questions?** Check the README.md for detailed explanations and advanced usage patterns.
