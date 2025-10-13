# System Architecture - Multi-Dataset Anomaly Detection

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                  │
│  Raw Log Files (.log, .txt, etc.)                                   │
│  ├─ Apache.log                                                      │
│  ├─ Nginx.log                                                       │
│  ├─ System.log                                                      │
│  └─ Application.log                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER (bert.py)                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  BERT Tokenization & Embedding Generation                 │     │
│  │  ├─ Tokenize log strings                                  │     │
│  │  ├─ Generate 768-dim vectors                              │     │
│  │  └─ Mean pooling with attention mask                      │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER (.npy files)                      │
│  Embedding Files (NumPy Arrays)                                     │
│  ├─ apache_embeddings.npy      (N₁ × 768)                          │
│  ├─ nginx_embeddings.npy       (N₂ × 768)                          │
│  ├─ system_embeddings.npy      (N₃ × 768)                          │
│  └─ application_embeddings.npy (N₄ × 768)                          │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              VALIDATION LAYER (inspect_datasets.py)                 │
│  Optional: Validate & Compare Datasets                              │
│  ├─ Check dimensions                                                │
│  ├─ Calculate statistics                                            │
│  ├─ Compare datasets                                                │
│  └─ Recommend cluster count                                         │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ANALYSIS LAYER (kmeans.py)                             │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  LogAnomalyDetector Class                                 │     │
│  │                                                           │     │
│  │  1. Load Embeddings (*.npy)                              │     │
│  │     ├─ Single file                                        │     │
│  │     ├─ Multiple files                                     │     │
│  │     └─ Glob patterns                                      │     │
│  │                                                           │     │
│  │  2. Combine Datasets                                      │     │
│  │     └─ Vertical stack: (N₁+N₂+N₃+N₄) × 768              │     │
│  │                                                           │     │
│  │  3. Train KMeans Clustering                               │     │
│  │     ├─ Fit model on combined data                        │     │
│  │     └─ Calculate cluster centers                          │     │
│  │                                                           │     │
│  │  4. Anomaly Detection                                     │     │
│  │     ├─ Transform new log to vector                       │     │
│  │     ├─ Assign to nearest cluster                         │     │
│  │     ├─ Calculate distance to center                      │     │
│  │     └─ Compare with threshold                            │     │
│  └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                   │
│  Anomaly Detection Results                                          │
│  ├─ is_anomaly: Boolean                                             │
│  ├─ distance: Float (distance to cluster center)                    │
│  ├─ cluster_id: Integer (assigned cluster)                          │
│  └─ threshold: Float (anomaly threshold)                            │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### 1. Embedding Generation Phase (bert.py)

```
Raw Logs → Tokenization → BERT Model → Mean Pooling → .npy Files
  ↓            ↓              ↓             ↓             ↓
Text         Tokens        Hidden        Mean         Embeddings
Strings      (IDs)         States       Vector        (768-dim)
                          (768-dim)
```

### 2. Model Training Phase (kmeans.py)

```
.npy Files → Load & Combine → KMeans Training → Trained Model
   ↓              ↓                  ↓                ↓
Multiple       Combined           Cluster          Model with
Datasets       Dataset            Centers          5 Clusters
(N₁×768)      (N_total×768)      (5×768)
(N₂×768)
(N₃×768)
```

### 3. Anomaly Detection Phase (kmeans.py)

```
New Log → Transform → Predict Cluster → Calculate Distance → Anomaly?
  ↓          ↓            ↓                    ↓                ↓
Text      Vector      Cluster ID           Distance         Boolean
String    (768-dim)     (0-4)              to Center        + Details
```

## 📊 Component Interaction Diagram

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       │ 1. Add log files
       ▼
┌─────────────────┐
│    bert.py      │◄──────── HuggingFace BERT Model
└────────┬────────┘
         │
         │ 2. Generate embeddings
         ▼
┌──────────────────┐
│  .npy files      │
│  (Storage)       │
└────────┬─────────┘
         │
         │ 3. Validate (optional)
         ▼
┌──────────────────┐
│inspect_datasets  │
│      .py         │
└────────┬─────────┘
         │
         │ 4. Load & analyze
         ▼
┌──────────────────┐
│   kmeans.py      │◄──────── scikit-learn KMeans
│(LogAnomalyDetector)
└────────┬─────────┘
         │
         │ 5. Detect anomalies
         ▼
┌─────────────────┐
│  Application    │
│  (Your Code)    │
└─────────────────┘
```

## 🎯 Class Structure

### LogAnomalyDetector Class

```python
class LogAnomalyDetector:
    │
    ├─ __init__(model_name)
    │  ├─ Initialize BERT tokenizer
    │  ├─ Initialize BERT model
    │  └─ Set up class attributes
    │
    ├─ transform_log_to_vector(log_text)
    │  ├─ Input:  str (raw log)
    │  ├─ Process: BERT embedding
    │  └─ Output: np.array (768,)
    │
    ├─ transform_logs_to_vectors(log_texts)
    │  ├─ Input:  List[str] (multiple logs)
    │  ├─ Process: Batch BERT embedding
    │  └─ Output: np.array (N, 768)
    │
    ├─ load_embeddings(embedding_files)
    │  ├─ Input:  str or List[str] (file paths or patterns)
    │  ├─ Process: Load and combine .npy files
    │  └─ Output: np.array (N_total, 768)
    │
    ├─ train_kmeans(n_clusters, random_state)
    │  ├─ Input:  int (number of clusters)
    │  ├─ Process: Train KMeans on embeddings
    │  └─ Output: KMeans model
    │
    └─ predict_anomaly(log_text, threshold_percentile)
       ├─ Input:  str (log to check)
       ├─ Process: Transform → Predict → Calculate distance
       └─ Output: (bool, float, int, float)
                  (is_anomaly, distance, cluster_id, threshold)
```

## 🔧 Configuration Points

### 1. BERT Model Configuration (kmeans.py)

```python
detector = LogAnomalyDetector(model_name="bert-base-uncased")
# Can be changed to other BERT models:
# - "bert-large-uncased"
# - "distilbert-base-uncased" (faster)
# - Any HuggingFace BERT model
```

### 2. KMeans Configuration (kmeans.py)

```python
detector.train_kmeans(
    n_clusters=5,      # Number of clusters (adjust based on data)
    random_state=42    # For reproducibility
)
```

### 3. Anomaly Threshold Configuration (kmeans.py)

```python
detector.predict_anomaly(
    log_text,
    threshold_percentile=95  # Top 5% are anomalies
    # Can be 90 (top 10%), 99 (top 1%), etc.
)
```

### 4. Log File Processing (bert.py)

```python
log_files = [
    r"..\dataset\Apache.log",
    r"..\dataset\Nginx.log",
    # Add more files here
]
```

## �� Scalability Considerations

### Memory Usage

```
Single Embedding:    768 × 4 bytes = 3.072 KB (float32)
1,000 Embeddings:    768 × 4 × 1,000 = 3.072 MB
10,000 Embeddings:   768 × 4 × 10,000 = 30.72 MB
100,000 Embeddings:  768 × 4 × 100,000 = 307.2 MB
1M Embeddings:       768 × 4 × 1,000,000 = 3.072 GB
```

### Processing Time

```
BERT Embedding:      ~10-50ms per log (GPU)
                     ~50-200ms per log (CPU)

KMeans Training:     O(n × k × i × d)
                     n = samples, k = clusters
                     i = iterations, d = dimensions
                     ~seconds to minutes for typical datasets
```

## 🔐 Security Considerations

1. **Environment Variables**: BERT authentication via account.env
2. **Data Privacy**: Embeddings stored locally, not shared
3. **Input Validation**: Error handling for malformed logs
4. **File Access**: Read-only access to log files

## 🧪 Testing Strategy

### Unit Tests (Recommended)

```python
# Test log transformation
def test_transform_log():
    detector = LogAnomalyDetector()
    vector = detector.transform_log_to_vector("Test log")
    assert vector.shape == (768,)

# Test dataset loading
def test_load_embeddings():
    detector = LogAnomalyDetector()
    detector.load_embeddings('*.npy')
    assert detector.embeddings is not None

# Test anomaly detection
def test_predict_anomaly():
    detector = LogAnomalyDetector()
    detector.load_embeddings('*.npy')
    detector.train_kmeans(n_clusters=3)
    is_anomaly, _, _, _ = detector.predict_anomaly("Test")
    assert isinstance(is_anomaly, bool)
```

## 📊 Performance Optimization Tips

1. **Batch Processing**: Use `transform_logs_to_vectors()` for multiple logs
2. **GPU Acceleration**: Enable CUDA for faster BERT inference
3. **Model Caching**: BERT model loads once and reuses
4. **Memory Management**: Use `torch.no_grad()` during inference
5. **Parallel Processing**: Process multiple log files in parallel

## 🎓 Advanced Use Cases

### 1. Real-Time Monitoring

```python
detector = LogAnomalyDetector()
detector.load_embeddings('historical_*.npy')
detector.train_kmeans(n_clusters=5)

# Monitor incoming logs
for log in incoming_log_stream():
    is_anomaly, dist, cluster, threshold = detector.predict_anomaly(log)
    if is_anomaly:
        alert_system.send_alert(log, dist, cluster)
```

### 2. Batch Analysis

```python
# Analyze all logs from a time period
logs = load_logs_from_period("2024-01-01", "2024-01-31")
vectors = detector.transform_logs_to_vectors(logs)
# Save for later analysis
np.save("january_embeddings.npy", vectors)
```

### 3. Incremental Learning

```python
# Train on historical data
detector.load_embeddings('historical_*.npy')
detector.train_kmeans(n_clusters=5)

# Periodically retrain with new data
new_embeddings = detector.transform_logs_to_vectors(new_logs)
combined = np.vstack([detector.embeddings, new_embeddings])
detector.embeddings = combined
detector.train_kmeans(n_clusters=5)
```

---

**Architecture Status**: ✅ Production-Ready  
**Documentation**: ✅ Complete  
**Maintainability**: ✅ High
