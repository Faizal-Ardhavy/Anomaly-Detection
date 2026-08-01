# 5-10 Bagian Kode Esensial - Anomaly Detection Thesis

Ringkasan kode esensial dari 5 komponen utama pipeline anomaly detection Anda.

---

## 1️⃣ INISIALISASI & PELATIHAN K-MEANS (MiniBatch)

### Snippet 1.1: Load Data & Exploratory Analysis
```python
# File: kmeans_v2.py
import numpy as np
from sklearn.preprocessing import normalize

# Load embeddings
try:
    semantic_vectors = np.load('combined_embeddings.npy')
except FileNotFoundError:
    # Dummy data with different clusters
    cluster1 = np.random.rand(500, 768) + 0.5
    cluster2 = np.random.rand(100, 768) - 0.5
    semantic_vectors = np.vstack([cluster1, cluster2])

print(f"Dataset shape: {semantic_vectors.shape}")
print(f"Dimensi vektor: {semantic_vectors.shape[1]}")
```

### Snippet 1.2: K-Means dengan Parallel Elbow Method
```python
# File: kmeans_v2.py
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Parameter eksperimen
possible_k = range(2, 21)
n_jobs = 2

# Eksperimen dengan Euclidean Distance (parallel)
def compute_wcss_euclidean(k, X):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X)
    return k, kmeans.inertia_

results = Parallel(n_jobs=n_jobs)(
    delayed(compute_wcss_euclidean)(k, semantic_vectors) for k in possible_k
)
results.sort(key=lambda x: x[0])
wcss = [r[1] for r in results]

# Plot elbow curve
plt.figure(figsize=(12, 6))
plt.plot(possible_k, wcss, 'bo-')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Metode Elbow - Menentukan K Optimal')
plt.grid(True)
plt.show()
```

### Snippet 1.3: K-Means Final Training dengan Cosine Distance
```python
# File: kmeans_v2.py
# Normalisasi untuk cosine distance
semantic_vectors_normalized = normalize(semantic_vectors, norm='l2', axis=1)

# Tentukan k optimal dari elbow curve (contoh: k=5)
optimal_k = 5

# Latih model final
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', 
                      random_state=42, n_init=10)
kmeans_final.fit(semantic_vectors_normalized)

# Dapatkan cluster labels dan centroids
cluster_labels = kmeans_final.labels_
centroids = kmeans_final.cluster_centers_

print(f"Berhasil melatih K-Means dengan k={optimal_k}")
print(f"Centroids shape: {centroids.shape}")
```

### Snippet 1.4: Build Centroids dari Precomputed Labels (untuk Inference)
```python
# File: cluster_testing_pipeline.py
def build_kmeans_centroids_from_labels(training_embeddings, training_cluster_labels,
                                       chunk_size=300000):
    """
    Build cluster centroids dari embeddings dan label yang sudah ada
    Berguna untuk inference tanpa retrain
    """
    labels = np.asarray(training_cluster_labels)
    unique_clusters = sorted(int(c) for c in np.unique(labels) if int(c) != -1)
    
    cluster_ids = np.array(unique_clusters, dtype=np.int64)
    cluster_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    
    n_features = int(training_embeddings.shape[1])
    sums = np.zeros((len(cluster_ids), n_features), dtype=np.float64)
    counts = np.zeros(len(cluster_ids), dtype=np.int64)
    
    # Streaming untuk dataset besar
    total = len(labels)
    for start in tqdm(range(0, total, chunk_size), desc="Building centroids"):
        end = min(start + chunk_size, total)
        emb_chunk = np.asarray(training_embeddings[start:end], dtype=np.float32)
        lbl_chunk = np.asarray(labels[start:end], dtype=np.int64)
        
        for cid in np.unique(lbl_chunk):
            if cid == -1:
                continue
            idx = cluster_to_idx.get(int(cid))
            if idx is None:
                continue
            
            mask = (lbl_chunk == cid)
            if np.any(mask):
                sums[idx] += emb_chunk[mask].sum(axis=0, dtype=np.float64)
                counts[idx] += int(mask.sum())
    
    # Normalisasi: hitung rata-rata
    valid = counts > 0
    centroids = (sums[valid] / counts[valid, None]).astype(np.float32)
    cluster_ids = cluster_ids[valid].astype(np.int32)
    
    return cluster_ids, centroids
```

---

## 2️⃣ PELATIHAN DBSCAN DENGAN AKSELERASI GPU (cuML)

### Snippet 2.1: K-Distance Plot untuk Estimasi eps
```python
# File: dbscan.py & dbscan_experiments.ipynb
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Tentukan k (heuristic: 2*dimensi atau 10-50 untuk log data)
k = 50
min_samples = 50

# Hitung k-NN distances untuk setiap titik
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(semantic_vectors)
distances, indices = neighbors_fit.kneighbors(semantic_vectors)

# Ambil jarak ke tetangga ke-k dan urutkan
k_distances = np.sort(distances[:, -1])

# Plot k-distance curve (untuk visual inspection eps)
plt.figure(figsize=(12, 6))
plt.plot(k_distances)
plt.xlabel('Titik Data (diurutkan)')
plt.ylabel(f'Jarak ke Tetangga ke-{k}')
plt.title(f'K-Distance Graph (k={k}) - Cari "Elbow" untuk eps')
plt.grid(True)

# Tambahkan percentile references
plt.axhline(y=np.percentile(k_distances, 90), color='r', linestyle='--', 
            label='90th percentile')
plt.axhline(y=np.percentile(k_distances, 95), color='orange', linestyle='--', 
            label='95th percentile')
plt.legend()
plt.show()

print(f"Suggested eps (90th percentile): {np.percentile(k_distances, 90):.4f}")
print(f"Suggested eps (95th percentile): {np.percentile(k_distances, 95):.4f}")
```

### Snippet 2.2: DBSCAN Standard dengan Optimasi Memory
```python
# File: dbscan_experiments.ipynb - Part 3 & 4
from sklearn.cluster import DBSCAN
import gc

# Tentukan eps dan min_samples dari grid search (atau manual setting)
eps_optimal = 1.5
min_samples_optimal = 50

print(f"Fitting DBSCAN dengan eps={eps_optimal}, min_samples={min_samples_optimal}")

# KUNCI: Gunakan algorithm='ball_tree' untuk menghindari O(n²) memory precomputation
# (Source: https://stackoverflow.com/questions/16381577)
dbscan = DBSCAN(eps=eps_optimal, min_samples=min_samples_optimal,
                algorithm='ball_tree',  # Memory-efficient!
                n_jobs=-1)              # Parallelisasi

labels_full = dbscan.fit_predict(semantic_vectors)

# Analisis hasil
unique_labels = set(labels_full)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = list(labels_full).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(labels_full)*100:.1f}%)")

# Cleanup memory
del dbscan
gc.collect()
```

### Snippet 2.3: DBSCAN GPU dengan cuML (Alternative - jika GPU tersedia)
```python
# File: dbscan_experiments.ipynb
# Alternatif: Gunakan cuML untuk GPU acceleration
try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    import cupy as cp
    
    print("Using GPU cuML DBSCAN...")
    
    # Transfer ke GPU
    X_gpu = cp.asarray(semantic_vectors, dtype=cp.float32)
    
    # DBSCAN GPU
    dbscan_gpu = cuDBSCAN(eps=1.5, min_samples=50, metric='euclidean')
    labels_gpu = dbscan_gpu.fit_predict(X_gpu)
    
    # Transfer kembali ke CPU
    labels_full = cp.asnumpy(labels_gpu)
    
    print(f"✓ GPU DBSCAN complete in {time.time() - start:.2f}s")
    
except ImportError:
    print("cuML not available, using CPU sklearn DBSCAN")
    # Fallback ke snippet 2.2
```

### Snippet 2.4: Parameter Grid Search untuk DBSCAN
```python
# File: cluster_testing_pipeline.py & dbscan_experiments.ipynb
import pandas as pd
from tqdm import tqdm

# Tentukan range eps dari k-distance percentiles
eps_min = np.percentile(k_distances, 75)
eps_max = np.percentile(k_distances, 99)
eps_values = np.linspace(eps_min, eps_max, 6)  # 6 values

min_samples_values = [10, 20, 30]  # 3 values

results = []

# Test semua kombinasi
for eps in eps_values:
    for ms in min_samples_values:
        dbs = DBSCAN(eps=float(eps), min_samples=int(ms), 
                     algorithm='ball_tree', n_jobs=-1)
        labels = dbs.fit_predict(semantic_vectors)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_pct = (n_noise / len(labels)) * 100
        
        # Silhouette score (optional, untuk cluster quality)
        sil = -1
        if n_clusters > 1:
            sil = silhouette_score(semantic_vectors, labels)
        
        results.append({
            'eps': float(eps),
            'min_samples': int(ms),
            'n_clusters': n_clusters,
            'noise_pct': float(noise_pct),
            'silhouette': float(sil)
        })

df_results = pd.DataFrame(results)
print(df_results.sort_values('silhouette', ascending=False).head(10))
```

---

## 3️⃣ PROSES INFERENSI DENGAN FAISS (Pencarian Jarak)

### Snippet 3.1: FAISS Index Building
```python
# File: cluster_testing_pipeline.py - Line 2347+
import faiss

def fast_cluster_assignment_faiss(training_embeddings, training_cluster_labels, 
                                  test_embeddings, k=10, nlist=100):
    """
    FAST cluster assignment menggunakan FAISS IVF (Approximate k-NN)
    - 10-100x lebih cepat dari sklearn
    - ~98% accuracy vs exact k-NN
    """
    print(f"\n🚀 Menggunakan FAISS IVF untuk FAST approximate k-NN...")
    
    d = int(training_embeddings.shape[1])  # Dimensi
    n_train = len(training_embeddings)
    
    # Adaptive nlist based on dataset size
    effective_nlist = min(nlist, max(1, int(np.sqrt(n_train / 100))))
    print(f"   Building FAISS IVF index:")
    print(f"   - Dimensi: {d}")
    print(f"   - Training samples: {n_train:,}")
    print(f"   - nlist (Voronoi cells): {effective_nlist}")
    
    # Create FAISS IVF index
    quantizer = faiss.IndexFlatL2(d)  # Quantizer for Voronoi cells
    index = faiss.IndexIVFFlat(quantizer, d, effective_nlist, faiss.METRIC_L2)
    
    # Train index (learns Voronoi cells)
    print(f"   Training index...")
    index.train(training_embeddings.astype(np.float32))
    
    # Add training points
    print(f"   Adding {n_train:,} training samples to index...")
    index.add(training_embeddings.astype(np.float32))
    
    # Set nprobe for recall/speed tradeoff
    index.nprobe = max(1, effective_nlist // 10)  # Trade-off: ~90% recall
    print(f"   nprobe (cells to search): {index.nprobe}")
    
    # Search: find nearest training samples for each test sample
    print(f"   Searching {len(test_embeddings):,} test samples...")
    distances, indices = index.search(test_embeddings.astype(np.float32), k)
    
    # Get cluster labels for each neighbor
    neighbor_cluster_ids = training_cluster_labels[indices]  # (n_test, k)
    
    # Majority vote: tentukan cluster label untuk setiap test sample
    test_cluster_labels = []
    for i in range(len(test_embeddings)):
        neighbor_labels = neighbor_cluster_ids[i]
        # Majority vote (ignoring noise -1 if possible)
        votes = np.bincount(neighbor_labels[neighbor_labels >= 0])
        if len(votes) > 0:
            assigned_label = np.argmax(votes)
        else:
            assigned_label = neighbor_labels[0]  # Fallback
        test_cluster_labels.append(assigned_label)
    
    test_cluster_labels = np.array(test_cluster_labels)
    
    return {
        'labels': test_cluster_labels,
        'distances': distances,
        'neighbor_indices': indices
    }
```

### Snippet 3.2: Predict K-Means dari Centroids (Inference Mode)
```python
# File: cluster_testing_pipeline.py
def predict_kmeans_from_centroids(test_embeddings, cluster_ids, centroids,
                                  use_cosine=True, batch_size=20000):
    """
    Predict cluster IDs dengan nearest centroid dalam batch
    (Memory-efficient untuk test dataset besar)
    """
    n_test = len(test_embeddings)
    predictions = np.empty(n_test, dtype=np.int32)
    
    if use_cosine:
        # L2-normalize untuk cosine distance
        centroids_norm = normalize(centroids.astype(np.float32), norm='l2', copy=True)
    else:
        centroids_norm = centroids.astype(np.float32)
        centroids_sq = np.sum(centroids_norm * centroids_norm, axis=1)[None, :]
    
    # Process dalam batch untuk stabilitas RAM
    for start in tqdm(range(0, n_test, batch_size), desc="Predicting"):
        end = min(start + batch_size, n_test)
        batch = np.asarray(test_embeddings[start:end], dtype=np.float32)
        
        if use_cosine:
            # Cosine similarity: normalize batch, then dot product
            batch_norm = normalize(batch, norm='l2', copy=False)
            scores = batch_norm @ centroids_norm.T
            best = np.argmax(scores, axis=1)  # Find best centroid
        else:
            # Euclidean distance
            dot = batch @ centroids_norm.T
            batch_sq = np.sum(batch * batch, axis=1, keepdims=True)
            dist2 = batch_sq + centroids_sq - 2.0 * dot
            best = np.argmin(dist2, axis=1)
        
        predictions[start:end] = cluster_ids[best]
    
    return predictions
```

### Snippet 3.3: Hybrid Assignment (FAISS + Fallback)
```python
# File: cluster_testing_pipeline.py - Line 2863+
# Try FAISS first (10-100x faster, ~98% accuracy)
try:
    faiss_result = fast_cluster_assignment_faiss(
        training_embeddings, training_cluster_labels,
        test_embeddings, k=10, nlist=100
    )
    test_cluster_labels = faiss_result['labels']
    print(f"✓ FAISS assignment successful!")
    
except ImportError:
    print("\n⚠️ FAISS not available, falling back to sklearn...")
    # Use sklearn NearestNeighbors as fallback
    knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree', n_jobs=-1)
    knn.fit(training_embeddings)
    distances, indices = knn.kneighbors(test_embeddings)
    
    # Majority vote
    neighbor_labels = training_cluster_labels[indices]
    test_cluster_labels = np.array([
        np.bincount(labels[labels >= 0]).argmax() 
        for labels in neighbor_labels
    ])
    print(f"✓ Sklearn k-NN assignment complete (slower but exact)")
```

---

## 4️⃣ EKSTRAKSI VEKTOR SEMANTIK MENGGUNAKAN BERT

### Snippet 4.1: Single Log to Vector (Mean Pooling)
```python
# File: sentence_to_vector.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def log_to_vector(text: str, model_name: str = "bert-base-uncased",
                  tokenizer: AutoTokenizer = None, model: AutoModel = None,
                  device: torch.device = None) -> np.ndarray:
    """
    Konversi single raw log string → 1D numpy vector (sentence embedding)
    
    Metode: Mean pooling atas last_hidden_state dengan attention mask
    (Ignores padding tokens)
    """
    if not isinstance(text, str):
        raise ValueError("text must be a str")
    
    device = device or (torch.device("cuda") if torch.cuda.is_available() 
                        else torch.device("cpu"))
    
    # Load model if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModel.from_pretrained(model_name).to(device)
    
    model.eval()
    
    with torch.no_grad():
        # Tokenize (truncate ke 512 tokens max)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get hidden states
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
        
        # Mean pooling dengan attention mask
        attention_mask = inputs["attention_mask"]  # (1, seq_len)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        
        masked = last_hidden * mask
        sum_embedded = torch.sum(masked, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        
        sentence_vec = sum_embedded / sum_mask  # (1, 768)
        return sentence_vec.squeeze(0).cpu().numpy()
```

### Snippet 4.2: Batch Processing dengan GPU Optimization
```python
# File: bert.py - Line 133+
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader

# GPU Optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

# Batch processing
batch_size = 256  # Aggressive batch size untuk GTX 1660 6GB
embedding_list = []

# Tokenize in advance (pre-tokenization strategy - 3-4x lebih cepat!)
print("Pre-tokenizing logs...")
log_texts = [...]  # Your list of logs
all_inputs = tokenizer(
    log_texts,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True
)

# DataLoader untuk efficient batching
dataset = TensorDataset(all_inputs['input_ids'], all_inputs['attention_mask'])
loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

# Inference loop
with torch.no_grad():
    for batch_ids, batch_mask in loader:
        batch_ids = batch_ids.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)
        
        # Get embeddings
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        # Mean pooling
        mask_expanded = batch_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embedded = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        batch_embeddings = (sum_embedded / sum_mask).cpu().numpy()
        embedding_list.append(batch_embeddings)

# Combine all embeddings
all_embeddings = np.vstack(embedding_list)  # (n_logs, 768)
np.save('log_embeddings.npy', all_embeddings)
```

### Snippet 4.3: Streaming Processing untuk Dataset Sangat Besar
```python
# File: bert.py - Large file handling
def process_text_chunk(lines_chunk, batch_size, tokenizer, model, device):
    """
    Process chunk of text lines into embeddings
    Digunakan untuk file ultra-large (streaming mode)
    """
    embeddings_list = []
    
    for i in range(0, len(lines_chunk), batch_size):
        batch = lines_chunk[i:i+batch_size]
        
        # Tokenize batch
        batch_inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=64,  # Shorter max_length untuk streaming
            padding=True,
            return_attention_mask=True
        )
        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_inputs)
            last_hidden = outputs.last_hidden_state
            
            # Mean pooling
            attention_mask = batch_inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            
            sum_embedded = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            
            batch_emb = (sum_embedded / sum_mask).cpu().numpy()
            embeddings_list.append(batch_emb)
    
    return np.vstack(embeddings_list) if embeddings_list else np.array([])

# Usage: streaming large file
embeddings_all = []
with open('large_logfile.txt', 'r') as f:
    chunk = []
    for line in f:
        chunk.append(line.strip())
        if len(chunk) == 10000:  # Process every 10k lines
            emb = process_text_chunk(chunk, batch_size=256, 
                                    tokenizer=tokenizer, model=model, device=device)
            embeddings_all.append(emb)
            chunk = []
    
    # Handle remaining lines
    if chunk:
        emb = process_text_chunk(chunk, batch_size=256, 
                                tokenizer=tokenizer, model=model, device=device)
        embeddings_all.append(emb)

all_embeddings = np.vstack(embeddings_all)
```

---

## 5️⃣ LOGIKA KARAKTERISASI KLASTER (Semi-Supervised 70%)

### Snippet 5.1: Load Template Events untuk Metadata Labeling
```python
# File: cluster_testing_pipeline.py - Line 2427+
def load_template_events(template_path: Path) -> set:
    """
    Load Label set dari template TSV file
    (Untuk 70% semi-supervised labeling strategy)
    
    Returns: set of Labels (e.g., {'-', 'APPREAD', 'KERNDTLB', ...})
    """
    print(f"Loading template: {template_path.name}")
    
    if not template_path.exists():
        return set()
    
    # Try TSV format first
    try:
        df = pd.read_csv(template_path, sep='\t', dtype=str)
    except:
        df = None
    
    # Find 'Label' column (case-insensitive)
    if df is not None:
        for col in df.columns:
            if str(col).strip().lower() == 'label':
                label_set = set(df[col].unique())
                label_set = {str(v).strip().upper() for v in label_set}
                label_set.discard('')
                print(f"✓ Found {len(label_set)} unique Labels")
                return label_set
    
    # Fallback: parse file manually
    labels = set()
    with open(template_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                labels.add(parts[0].strip().upper())
    
    labels.discard('')
    print(f"✓ Inferred {len(labels)} unique Labels (fallback)")
    return labels
```

### Snippet 5.2: Metadata-Based Cluster Labeling (70% Semi-Supervised)
```python
# File: cluster_testing_pipeline.py - Line 1114+
def analyze_cluster_characteristics(cluster_labels, embeddings=None, 
                                    metadata_tsv_path=None,
                                    normal_template_path=None,
                                    nonnormal_template_path=None):
    """
    🔑 SEMI-SUPERVISED (70%): Analyze clusters dengan metadata voting
    
    Strategy:
    1. Untuk setiap cluster: sample 70% dari samples di dalamnya
    2. Check metadata (EventId/Label) untuk setiap sample
    3. Majority vote: Tentukan cluster label (NORMAL / NON-NORMAL / ANOMALY)
    
    Returns:
    - DataFrame dengan cluster stats
    - Dict dengan cluster_id → label mapping
    """
    print("\n🔍 Analyzing clusters (SEMI-SUPERVISED: Metadata-based)...")
    
    unique_clusters = sorted(set(cluster_labels))
    cluster_info = []
    
    # Load templates
    normal_events = load_template_events(normal_template_path)
    nonnormal_events = load_template_events(nonnormal_template_path)
    
    print(f"Templates: NORMAL={len(normal_events)}, NON-NORMAL={len(nonnormal_events)}")
    
    # Load metadata TSV (label column only)
    metadata_df = pd.read_csv(metadata_tsv_path, sep='\t', usecols=['label'], dtype=str)
    
    # MAGIC: 70% sampling untuk efficiency + robustness
    METADATA_SAMPLE_SIZE = int(len(metadata_df) * 0.7)  # 70% of training data
    MAJORITY_THRESHOLD = 0.5  # >50% vote → assign label
    
    # Analyze each cluster
    for cluster_id in tqdm(unique_clusters, desc="Analyzing clusters"):
        mask = cluster_labels == cluster_id
        n_samples = np.sum(mask)
        cluster_indices = np.where(mask)[0]
        
        if cluster_id == -1:
            # DBSCAN noise → ANOMALY
            cluster_label = 2
            label_name = 'ANOMALY'
            reason = 'dbscan_noise'
        else:
            # Sample 70% dari cluster untuk voting
            if n_samples > METADATA_SAMPLE_SIZE:
                sample_indices = np.random.choice(cluster_indices, 
                                                 METADATA_SAMPLE_SIZE, 
                                                 replace=False)
            else:
                sample_indices = cluster_indices
            
            # Count normal vs non-normal dari metadata
            count_normal = 0
            count_nonnormal = 0
            count_unknown = 0
            
            for idx in sample_indices:
                if idx < len(metadata_df):
                    label_val = metadata_df.iloc[idx]['label']
                    label_normalized = str(label_val).strip().upper()
                    
                    if label_normalized in normal_events:
                        count_normal += 1
                    elif label_normalized in nonnormal_events:
                        count_nonnormal += 1
                    else:
                        count_unknown += 1
            
            total_evidence = count_normal + count_nonnormal
            
            if total_evidence == 0:
                # No metadata evidence → ANOMALY (safe default)
                cluster_label = 2
                label_name = 'ANOMALY'
                reason = 'no_metadata'
                pct_normal = 0.0
                pct_nonnormal = 0.0
            else:
                pct_normal = count_normal / total_evidence
                pct_nonnormal = count_nonnormal / total_evidence
                
                # Assign label based on majority vote
                if pct_nonnormal >= MAJORITY_THRESHOLD:
                    cluster_label = 1
                    label_name = 'NON-NORMAL'
                    reason = 'nonnormal_majority'
                elif pct_normal >= MAJORITY_THRESHOLD:
                    cluster_label = 0
                    label_name = 'NORMAL'
                    reason = 'normal_majority'
                else:
                    # Mixed: default to NON-NORMAL (conservative)
                    cluster_label = 1
                    label_name = 'NON-NORMAL'
                    reason = 'mixed_ambiguous'
        
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_samples': n_samples,
            'cluster_label': cluster_label,
            'label_name': label_name,
            'labeling_reason': reason
        })
    
    df_clusters = pd.DataFrame(cluster_info)
    
    print(f"\n✓ Cluster Analysis Complete:")
    print(f"  NORMAL clusters: {(df_clusters['label_name']=='NORMAL').sum()}")
    print(f"  NON-NORMAL clusters: {(df_clusters['label_name']=='NON-NORMAL').sum()}")
    print(f"  ANOMALY clusters: {(df_clusters['label_name']=='ANOMALY').sum()}")
    
    return df_clusters
```

### Snippet 5.3: Direct Cluster Lookup untuk Prediction (Fast Inference)
```python
# File: cluster_testing_pipeline.py - Line 2545+
def hybrid_predict(test_cluster_labels, cluster_dict):
    """
    🚀 FAST PREDICTION: Langsung lookup cluster label
    
    Karena sudah ada 70% metadata voting saat training,
    inference hanya perlu simple dictionary lookup!
    
    Returns: predictions array (0=NORMAL, 1=NON-NORMAL, 2=ANOMALY)
    """
    predictions = np.zeros(len(test_cluster_labels), dtype=np.int32)
    confidence = np.zeros(len(test_cluster_labels), dtype=np.float32)
    
    for i, cluster_id in enumerate(tqdm(test_cluster_labels, desc="Predicting")):
        if cluster_id in cluster_dict:
            cluster_info = cluster_dict[cluster_id]
            predictions[i] = cluster_info['cluster_label']
            
            # Confidence based on labeling reason
            if cluster_info['labeling_reason'] in ['normal_majority', 'nonnormal_majority']:
                confidence[i] = 0.85  # High confidence
            else:
                confidence[i] = 0.60  # Medium confidence
        else:
            # Unknown cluster → ANOMALY (conservative)
            predictions[i] = 2
            confidence[i] = 0.50
    
    return predictions, confidence
```

### Snippet 5.4: Confusion Matrix & Detailed Metrics (Evaluation)
```python
# File: cluster_testing_pipeline.py - Line 2704+
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate 2x3 metrics: 2-class ground truth vs 3-class prediction
    
    Ground truth: 0=NORMAL, 1=NON-NORMAL (from test set name)
    Predictions:  0=NORMAL, 1=NON-NORMAL, 2=ANOMALY
    """
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix: 2x3 (ground truth rows, predictions cols)
    unique_true = sorted(set(y_true))
    cm_full = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm = cm_full[unique_true, :]  # Keep only ground truth rows
    
    print("="*70)
    print("EVALUATION: 2-CLASS GT vs 3-CLASS PRED")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Per-class breakdown
    class_names = ['NORMAL', 'NON-NORMAL', 'ANOMALY']
    true_names = [class_names[i] for i in unique_true]
    
    print(f"\n{' '*20} Predicted")
    print(f"{'':20s} NORMAL  NON-NOR  ANOMALY")
    for i, true_idx in enumerate(unique_true):
        print(f"GT {true_names[i]:12s} [{cm[i,0]:7d} {cm[i,1]:7d} {cm[i,2]:7d}]")
    
    # Error analysis
    print(f"\nError Breakdown:")
    for i, true_idx in enumerate(unique_true):
        total = cm[i].sum()
        correct = cm[i, true_idx]
        wrong = total - correct
        print(f"  {true_names[i]:12s}: {correct:,} correct, {wrong:,} errors ({wrong/total*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'true_labels': unique_true
    }
```

---

## 📊 Quick Reference: 5 Components Flow

```
┌─────────────────────────────────────────────────────────┐
│                ANOMALY DETECTION PIPELINE                │
├─────────────────────────────────────────────────────────┤
│                                                           │
│ 1️⃣  BERT Embedding                                       │
│    sentence_to_vector() → (n_logs, 768)                 │
│                           ↓                              │
│ 2️⃣  K-Means Training                                     │
│    KMeans(k=5) → cluster_labels, centroids             │
│                           ↓                              │
│ 3️⃣  DBSCAN Training                                      │
│    DBSCAN(eps=1.5) → cluster_labels, noise points      │
│                           ↓                              │
│ 4️⃣  Metadata Labeling (70% Semi-Supervised)             │
│    analyze_cluster_characteristics() → cluster_dict      │
│    (Maps cluster_id → label: NORMAL/NON-NORMAL/ANOMALY) │
│                           ↓                              │
│ 5️⃣  Inference (Test Phase)                               │
│    - FAISS for nearest neighbor assignment               │
│    - Direct cluster lookup for prediction                │
│    - Return: predictions, confidence                     │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Key Parameters Summary

| Component | Key Parameter | Typical Value | Impact |
|-----------|--------------|---------------|--------|
| BERT | max_length | 512 | Token truncation |
| K-Means | n_init | 10 | Better convergence |
| K-Means | init | 'k-means++' | Faster training |
| DBSCAN | eps | 0.5 - 2.0 | Cluster separation |
| DBSCAN | min_samples | 10 - 50 | Noise sensitivity |
| FAISS | nlist | √(n_train/100) | Speed/accuracy trade-off |
| Metadata | SAMPLE_RATIO | 70% | 70% semi-supervised |
| Metadata | MAJORITY_THRESHOLD | 50% | Label confidence |

---

## 💾 Memory Tips

- **BERT**: Use gradient checkpointing jika OOM
- **K-Means**: Normalisasi data (L2) untuk cosine distance  
- **DBSCAN**: Gunakan `algorithm='ball_tree'` untuk O(n·d) vs O(n²)
- **FAISS**: IVF index lebih cepat tapi approximate (~98% accuracy)
- **Metadata**: Gunakan memmap untuk file >2GB
