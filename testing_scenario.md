# Testing Scenario: Log Anomaly Detection Clustering

## 📋 Project Overview

**Objective:** Evaluasi performa clustering (K-means & DBSCAN) untuk deteksi anomali log menggunakan **Hybrid Prediction Strategy** yang dapat handle ratusan cluster dengan distribusi data yang merata.

**Datasets:**
- BGL (Blue Gene/L)
- Thunderbird

**Algorithms:**
- K-means (adaptif k, bisa puluhan hingga ratusan cluster)
- DBSCAN (dengan tuning eps & min_samples)

**Dataset Variations:**
- **Base (768 dim)**: Non-normalized embeddings dari BERT
- **PCA-256**: Reduced dimensionality menggunakan PCA
- **PCA-128**: Further reduced dimensionality

**Data Split:** Training & Testing untuk masing-masing dataset

**Ground Truth Source:** Template-based 3-way classification

```
┌─────────────────────────────────────────────────────────────────┐
│ GROUND TRUTH LABELING STRATEGY                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Input: Log line dengan EventId (e.g., E77, E33, E55)           │
│                                                                  │
│ Step 1: Load template files                                     │
│   - normal_template.txt     (EventId → Normal patterns)        │
│   - nonNormal_template.txt  (EventId → Expected errors)        │
│                                                                  │
│ Step 2: Match EventId to templates                             │
│                                                                  │
│   IF EventId in normal_template:                               │
│     → Ground Truth = NORMAL (0)                                │
│     → Example: E77 "instruction cache parity error corrected"  │
│                E3 "double-hummer alignment exceptions"          │
│                                                                  │
│   ELIF EventId in nonNormal_template:                          │
│     → Ground Truth = NON-NORMAL (1)                            │
│     → Example: E33 "APPREAD: failed to read message"           │
│                E55 "KERNDTLB: data TLB error interrupt"        │
│     → Known error patterns (expected, but not normal)          │
│                                                                  │
│   ELSE:                                                         │
│     → Ground Truth = ANOMALY (2)                               │
│     → Example: Unknown EventId, novel patterns                 │
│     → True outliers, security issues, corruption               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Template file locations:
  - log_processing/bgl/bgl_normal_template.txt
  - log_processing/bgl/bgl_nonNormal_template.txt
  - log_processing/thunderbird/thunderbird_normal_template.txt
  - log_processing/thunderbird/thunderbird_nonNormal_template.txt
```

**Critical Insight:**
- ❌ **OLD WRONG ASSUMPTION**: Label `-` = normal, all others = anomaly
- ✅ **CORRECT APPROACH**: Match EventId to templates
  - Normal templates = routine operations
  - Non-Normal templates = expected errors (FATAL but known)
  - No match = TRUE anomaly (unknown/novel patterns)

---

## 🎯 Testing Strategy: Semi-Supervised Metadata-based Cluster Labeling

### Large-Scale Memory Safety (Thunderbird)

Untuk dataset Thunderbird skala sangat besar (ratusan juta baris), implementasi pipeline menggunakan prinsip yang tetap konsisten dengan skenario ini, tetapi dengan proteksi memori:

- Metadata TSV diproses **streaming/chunked** menjadi label memmap (uint8) agar tidak OOM.
- Silhouette dihitung pada **sample non-noise** (bukan full dataset) untuk menjaga RAM/compute tetap aman.
- Untuk Thunderbird, sample silhouette dibatasi adaptif (contoh: maksimum 20k) agar proses stabil di server.

Catatan metodologi:
- Ini **tidak mengubah strategi evaluasi** (tetap semi-supervised metadata-based untuk cluster labeling).
- Ini hanya optimasi implementasi agar feasible untuk data skala sangat besar.

### Update Implementasi (April 2026)

Semua perubahan berikut sudah diadopsi di pipeline terbaru:

1. **Metadata loading dipisah per dataset:**
  - **Thunderbird:** metadata TSV diproses streaming menjadi `uint8` memmap (`0=Normal, 1=Non-Normal, 2=Anomaly`) dan bisa di-reuse.
  - **BGL:** tetap gunakan mode lama (load label-column ke RAM), dengan fallback chunked jika diperlukan.

2. **Template parser dibuat robust:**
  - Bisa handle file template yang punya baris metadata di awal (mis. `Total ...`).
  - Bisa cari header `Label` secara dinamis atau fallback ke kolom pertama.

3. **Silhouette dibuat memory-safe untuk dataset sangat besar:**
  - Tidak pernah melakukan slicing full `embeddings[mask]`.
  - Ambil indeks sample dulu, baru load embedding sample.
  - Untuk Thunderbird, sample silhouette dibatasi adaptif (default cap 20k).
  - Silhouette per-cluster dihitung dari mean sample cluster (aproksimasi terkontrol).

4. **Threshold K-Means dibuat hybrid-adaptive (lebih stabil lintas skala):**
  - Gabungan threshold berbasis `% total data` dan `% average cluster size`.
  - Final threshold memakai batas yang lebih ketat agar tidak over-labeling pada dataset sangat besar.

5. **Embedding loader lebih robust untuk format file beragam:**
  - Coba `np.load(..., mmap_mode='r')` terlebih dulu.
  - Jika gagal, fallback berjenjang (allow_pickle / raw memmap heuristic) untuk kompatibilitas file lama.

### **Evolution: From Unreliable Size-based to Data-driven Metadata-based**

**OLD APPROACH (Size-based - FAILED ❌):**
```
Cluster ID   Size      Strategy              Problem
---------------------------------------------------------------
0            125,000   Pure size-based       Cannot distinguish NORMAL vs NON-NORMAL
1             89,000   k-NN vote             Unreliable, slow, 25% accuracy
2              5,000   Small → NON-NORMAL    Arbitrary threshold
3                 45   Very small → ANOMALY  Size ≠ semantic meaning
```

**Result:** 25% accuracy (random guess level)

**NEW APPROACH (Metadata-based - MUCH BETTER ✅):**
```
Cluster ID   Size      Sample Metadata         Majority         Label
--------------------------------------------------------------------------
0            125,000   Sample 1K → 94.5% normal   → NORMAL (0)      ✓
1             89,000   Sample 1K → 96.8% non-normal → NON-NORMAL (1) ✓
2              5,000   Sample 1K → 88% normal     → NORMAL (0)      ✓
3                 45   < 50 samples              → ANOMALY (2)     ✓
4             15,600   Sample 1K → 45%/55% mix   → ANOMALY (2)     ✓
```

**Expected Result:** 65-80% accuracy (useful for production!)

---

### **New Strategy: Semi-Supervised Transductive Learning**

**Philosophy:**
1. ✅ **Clustering remains UNSUPERVISED** → DBSCAN/K-Means without ground truth
2. ✅ **Cluster characterization uses training metadata** → Data-driven labeling
3. ✅ **Test samples inherit cluster labels** → Transductive learning
4. ✅ **Research valid** → Standard practice in semi-supervised learning

**NOT cheating because:**
- Test samples are NOT labeled individually
- Only training metadata used for cluster-level labeling
- Classification based on natural log patterns from clustering
- Analogous to: image clustering → label clusters by dominant class → classify new images

---

### **Metadata-based Cluster Labeling Flow**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP A: TRAINING - Characterize Clusters with Metadata                 │
└─────────────────────────────────────────────────────────────────────────┘

FOR each training cluster:

  1️⃣ CHECK SIZE THRESHOLD:
    Hitung adaptive threshold (khusus K-Means):
    - base_anomaly = max(MIN_CLUSTER_SIZE_FOR_LABELING, ratio_total_data)
    - avg_anomaly = max(MIN_CLUSTER_SIZE_FOR_LABELING, ratio_avg_cluster)
    - final_anomaly_threshold = min(base_anomaly, avg_anomaly)

    IF cluster_size < final_anomaly_threshold:
        → cluster_label = ANOMALY (2)
        → reason = "too_small" (rare/suspicious)
        → DONE ✓
     
     ELSE: Continue to metadata check →

  2️⃣ SAMPLE CLUSTER DATA:
     IF cluster_size > 1000:
        → sample = random 1000 samples
     ELSE:
        → sample = all samples
  
  3️⃣ LOAD TRAINING METADATA:
      IF DATASET == Thunderbird:
        → build/reuse metadata label memmap via streaming
        → access label by index from memmap
      ELSE (BGL):
        → load label column directly (legacy path)
     
     FOR each sampel in cluster:
        event_label = metadata_label[sample_idx]
        
        IF event_label in normal_template:
           count_normal++
        
        IF event_label in nonnormal_template:
           count_nonnormal++
  
  4️⃣ CALCULATE MAJORITY:
     total = count_normal + count_nonnormal
     pct_normal = count_normal / total
     pct_nonnormal = count_nonnormal / total
  
  5️⃣ ASSIGN CLUSTER LABEL based on majority:
     
     IF pct_normal ≥ 70%:
        → cluster_label = NORMAL (0)
        → reason = "normal_majority"
        → confidence = pct_normal
     
     ELIF pct_nonnormal ≥ 70%:
        → cluster_label = NON-NORMAL (1)
        → reason = "nonnormal_majority"
        → confidence = pct_nonnormal
     
     ELSE (40-60% mixed):
        → cluster_label = ANOMALY (2)
        → reason = "mixed_ambiguous"
        → confidence = 0.50

  RESULT: cluster_df with [cluster_id, size, cluster_label, pct_normal, ...]

┌─────────────────────────────────────────────────────────────────────────┐
│ STEP B: TESTING - Predict New Samples (SIMPLE & FAST!)                 │
└─────────────────────────────────────────────────────────────────────────┘

FOR each test sample:

  1️⃣ ASSIGN TO CLUSTER:
     cluster_id = assign_to_cluster(test_sample)
     
     Methods:
     - K-means: model.predict(embedding)
     - DBSCAN: k-NN to find nearest training samples
  
  2️⃣ LOOKUP CLUSTER LABEL:
     cluster_info = cluster_df[cluster_id]
     prediction = cluster_info['cluster_label']  # 0, 1, or 2
     confidence = cluster_info['confidence']
  
  3️⃣ DONE! ✓
     → No k-NN needed
     → No complex voting
     → Just direct lookup!

EXAMPLE:
Test Sample #12345
  → Assigned to Cluster #7
  → Cluster #7 label = NORMAL (0)
  → Prediction = NORMAL (0) ✓
  → Confidence = 0.945 (94.5% normal in metadata)

┌─────────────────────────────────────────────────────────────────────────┐
│ WHY THIS WORKS BETTER                                                   │
└─────────────────────────────────────────────────────────────────────────┘

✅ DATA-DRIVEN: Based on actual training metadata, not arbitrary thresholds
✅ ACCURATE: 3-4x better than size-based (65-80% vs 25%)
✅ FAST: Simple lookup, no k-NN voting needed
✅ INTERPRETABLE: Can explain why cluster labeled X
✅ SCALABLE: Works with hundreds of clusters
✅ RESEARCH VALID: Semi-supervised transductive learning

❌ OLD (Size + k-NN): Slow, unreliable, arbitrary, 25% accuracy
✅ NEW (Metadata): Fast, accurate, data-driven, 65-80% accuracy

```

---

## ⚠️ **DEPRECATED: OLD APPROACH DOCUMENTATION (FOR REFERENCE ONLY)** ⚠️

**WARNING: The section below describes the OLD size + purity + k-NN approach that achieved only 25% accuracy.**

**DO NOT USE THIS APPROACH. See lines 76-240 above for the NEW metadata-based approach.**

<details>
<summary>📜 Click to view legacy documentation</summary>

### OLD Prediction Flowchart (Size + Purity + k-NN)

```
│     │   → Method = "noise"                                 │    │
│     │   → Reasoning: Outliers = Critical anomalies         │    │
│     │   → K-means: N/A (no noise concept)                 │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│     ┌─ Very small cluster (< 50 samples)? ────────────────┐    │
│     │   → Prediction = ANOMALY                             │    │
│     │   → Confidence = 0.75                                │    │
│     │   → Method = "very_small"                            │    │
│     │   → Reasoning: Rare critical patterns                │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│     ┌─ Small cluster (50-200 samples)? ──────────────────┐     │
│     │   → Prediction = NON-NORMAL                          │    │
│     │   → Confidence = 0.65                                │    │
│     │   → Method = "small"                                 │    │
│     │   → Reasoning: Suspicious/unusual patterns           │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│     ┌─ High purity cluster (> 95%)? ──────────────────────┐    │
│     │   IF dominant = Normal (label 0):                   │    │
│     │     → Prediction = NORMAL                            │    │
│     │     → Confidence = purity (e.g., 0.98)              │    │
│     │                                                       │    │
│     │   IF dominant = Anomaly (label 1):                  │    │
│     │     → Prediction = ANOMALY                           │    │
│     │     → Confidence = purity (e.g., 0.97)              │    │
│     │                                                       │    │
│     │   → Method = "pure"                                  │    │
│     │   → Reasoning: Clear separation, trust label        │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│     ┌─ Medium purity (70-95%)? ────────────────────────────┐   │
│     │   → Prediction = NON-NORMAL                          │    │
│     │   → Confidence = purity                              │    │
│     │   → Method = "medium_purity"                         │    │
│     │   → Reasoning: Mixed cluster = Suspicious zone       │    │
│     │   → Contains both normal & anomaly → Borderline     │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│     ┌─ Low purity (< 70%)? ────────────────────────────────┐   │
│     │   → Use k-NN vote (k=10) from training samples      │    │
│     │                                                       │    │
│     │   Vote breakdown:                                     │    │
│     │   • 8-10 normal → NORMAL (confidence = vote/10)     │    │
│     │   • 6-7  normal → NON-NORMAL (borderline)           │    │
│     │   • 5-5  split  → NON-NORMAL (ambiguous)            │    │
│     │   • 6-7  anomaly → NON-NORMAL (suspicious)          │    │
│     │   • 8-10 anomaly → ANOMALY (confidence = vote/10)   │    │
│     │                                                       │    │
│     │   → Method = "knn"                                   │    │
│     │   → Reasoning: Use individual similarity            │    │
│     └──────────────────────────────────────────────────────┘    │
│                                                                  │
│ [3] Output: prediction + confidence + method                    │
└─────────────────────────────────────────────────────────────────┘
```

### **3-Way Classification: Normal vs Non-Normal vs Anomaly**

**Konsep:**
```
┌─────────────────────────────────────────────────────────────────┐
│ NORMAL (0)         NON-NORMAL (1)        ANOMALY (2)           │
├─────────────────────────────────────────────────────────────────┤
│ • Clear normal     • Suspicious           • Critical errors    │
│ • High confidence  • Borderline           • Outliers           │
│ • Purity > 95%     • Mixed clusters       • Noise points       │
│ • Large clusters   • Medium purity        • Very rare patterns │
│                    • Unusual but not      • Security issues    │
│                      critical             •                     │
└─────────────────────────────────────────────────────────────────┘
```

**Perbedaan K-means vs DBSCAN:**

| Karakteristik | K-means | DBSCAN |
|---------------|---------|--------|
| **Noise concept** | ❌ Tidak ada | ✅ Cluster -1 = ANOMALY |
| **Small clusters** | Relative size | Absolute + relative |
| **Pure clusters** | ✅ NORMAL/ANOMALY | ✅ NORMAL/ANOMALY |
| **Mixed clusters** | NON-NORMAL or k-NN | NON-NORMAL or k-NN |
| **Advantage** | All points assigned | Detect outliers naturally |
| **Disadvantage** | Force assignment | Need tuning eps/min_samples |

**Decision Matrix:**

```
Cluster Type        | K-means Prediction | DBSCAN Prediction
--------------------|-------------------|-------------------
Noise (-1)          | N/A               | ANOMALY (0.85)
Very small (<50)    | ANOMALY (0.75)    | ANOMALY (0.75)
Small (50-200)      | NON-NORMAL (0.65) | NON-NORMAL (0.65)
Pure normal (>95%)  | NORMAL (purity)   | NORMAL (purity)
Pure anomaly (>95%) | ANOMALY (purity)  | ANOMALY (purity)
Medium purity (70-95%) | NON-NORMAL     | NON-NORMAL
Low purity (<70%)   | k-NN vote         | k-NN vote
```

### **Keunggulan Hybrid Strategy:**

1. ✅ **Handle Ratusan Cluster** - Tidak bergantung pada jumlah cluster
2. ✅ **Handle Distribusi Merata** - k-NN vote untuk cluster ambiguous
3. ✅ **3-Way Classification** - Normal / Non-Normal / Anomaly
4. ✅ **Confidence Score** - Tahu seberapa yakin prediction kita
5. ✅ **Method Tracking** - Bisa analisis per-method performance
6. ✅ **Algorithm-Specific** - Different strategy untuk K-means vs DBSCAN
7. ✅ **Ground Truth dari Metadata** - Automatic loading dari TSV


</details>

---

## 🔧 Implementation: `cluster_testing_pipeline.py`

### **Step 1: Configuration yang Perlu Diubah**

```python
# ============================================================
# A. Dataset Selection
# ============================================================
DATASET = "BGL"           # "BGL" atau "Thunderbird"
ALGORITHM = "kmeans"      # "kmeans" atau "dbscan"
EMBEDDING_TYPE = "base"   # "base", "pca256", atau "pca128"

# ============================================================
# B. Template Files (Ground Truth Definition)
# ============================================================
# Load EventId mappings from templates
if DATASET == "BGL":
    NORMAL_TEMPLATE_PATH = Path("log_processing/bgl/bgl_normal_template.txt")
    NONNORMAL_TEMPLATE_PATH = Path("log_processing/bgl/bgl_nonNormal_template.txt")
else:  # Thunderbird
    NORMAL_TEMPLATE_PATH = Path("log_processing/thunderbird/thunderbird_normal_template.txt")
    NONNORMAL_TEMPLATE_PATH = Path("log_processing/thunderbird/thunderbird_nonNormal_template.txt")

# ============================================================
# C. Path ke Metadata TSV (for EventId extraction)
# ============================================================
METADATA_TSV_PATH = Path(
    "/media/bioinfo04/Expansion/after_preprocessed_meta_data/"
    "after_preprocessed_bgl_meta.tsv"
)
# Format: Contains EventId column for matching to templates

# ============================================================
# C. Path ke Training Results
# ============================================================
if ALGORITHM == "kmeans":
    TRAINED_MODEL_PATH = Path("model_kmeans_log.pkl")
    TRAINING_LABELS_PATH = Path("cluster_labels.npy")
    TRAINING_EMBEDDINGS_PATH = Path(
        "/media/.../after_preprocessed_bgl_embeddings.npy"
    )
else:  # dbscan
    TRAINING_LABELS_PATH = Path("dbscan/dbscan_labels.npy")
    TRAINING_CONFIG_PATH = Path("dbscan/dbscan_config.npy")
    TRAINING_EMBEDDINGS_PATH = Path(
        "/media/.../after_preprocessed_bgl_embeddings.npy"
    )

# ============================================================
# D. Path ke Testing Data
# ============================================================
TESTING_EMBEDDINGS_PATHS = [
    Path("testing_error.npy"),
    Path("testing_warning.npy"),
    Path("testing_info.npy"),
]

TESTING_METADATA_TSV = Path(
    "/media/bioinfo04/Expansion/after_preprocessed_meta_data/"
    "testing_bgl_meta.tsv"
)

# ============================================================
# E. Semi-supervised Cluster Labeling Parameters (RECOMMENDED)
# ============================================================
# Metadata-based cluster characterization
USE_METADATA_LABELING = True        # Use training metadata (RECOMMENDED!)
METADATA_SAMPLE_SIZE = 1000         # Sample per cluster (or full if smaller)
MAJORITY_THRESHOLD = 0.70           # ≥70% majority → assign that label
MIN_CLUSTER_SIZE_FOR_LABELING = 50  # < 50 samples → auto ANOMALY

# K-Means hybrid adaptive thresholds
KMEANS_ANOMALY_CLUSTER_RATIO = 0.002
KMEANS_SMALL_CLUSTER_RATIO = 0.005
KMEANS_ANOMALY_AVG_CLUSTER_RATIO = 0.04
KMEANS_SMALL_AVG_CLUSTER_RATIO = 0.12

# Legacy: Size-based thresholds (if metadata disabled)
VERY_SMALL_CLUSTER_THRESHOLD = 50   # < 50 → ANOMALY (legacy)
SMALL_CLUSTER_THRESHOLD = 200       # 50-200 → NON-NORMAL (legacy)

USE_COSINE_DISTANCE = True          # Normalize for BERT embeddings
```

### **Parameter Explanation:**

**Metadata-based Labeling (NEW):**
```
USE_METADATA_LABELING = True        # Use training metadata to label clusters
  → Load training metadata (Thunderbird=streaming memmap, BGL=legacy label load)
  → Sample 1000 samples per cluster
  → Count normal vs non-normal events from templates
  → Majority vote (≥70%) → assign cluster label

MAJORITY_THRESHOLD = 0.70           # Require 70% majority
  ≥ 70% normal     → cluster_label = NORMAL (0)
  ≥ 70% non-normal → cluster_label = NON-NORMAL (1)
  < 70% both       → cluster_label = ANOMALY (2) - ambiguous

MIN_CLUSTER_SIZE_FOR_LABELING = 50  # Minimum samples to label
  < 50 samples → cluster_label = ANOMALY (2) - too rare/suspicious

KMEANS_ANOMALY_CLUSTER_RATIO / KMEANS_SMALL_CLUSTER_RATIO
  → Komponen threshold berbasis total training size

KMEANS_ANOMALY_AVG_CLUSTER_RATIO / KMEANS_SMALL_AVG_CLUSTER_RATIO
  → Komponen threshold berbasis ukuran rata-rata cluster (sensitif terhadap jumlah cluster)

FINAL K-Means thresholds
  → min(threshold_total_data, threshold_avg_cluster)
  → membuat threshold lebih robust untuk BGL (~4M) vs Thunderbird (~200M+)
```

**Why Metadata-based is Better:**
```
❌ OLD (Size-based):
   - Arbitrary thresholds
   - Cannot distinguish NORMAL vs NON-NORMAL in large clusters
   - 25% accuracy

✅ NEW (Metadata-based):
   - Data-driven from training metadata
   - Accurate cluster characterization
   - 65-80% accuracy (3-4x improvement!)
```

---

## **Step 2: Execution Pipeline**

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: LOAD GROUND TRUTH LABELS (TEMPLATE-BASED)              │
│                                                                  │
│ A. Load template files:                                         │
│    - normal_template.txt → extract EventId set                 │
│      Example: {E77, E3, E18, E2, ...}                          │
│                                                                  │
│    - nonNormal_template.txt → extract EventId set              │
│      Example: {E33, E55, APPREAD, KERNDTLB, ...}               │
│                                                                  │
│ B. Load metadata TSV (training & testing):                     │
│    - Read EventId column per log line                          │
│                                                                  │
│ C. Assign ground truth labels:                                 │
│    FOR each log line:                                           │
│      IF EventId in normal_set:                                 │
│        gt_label = 0  (NORMAL)                                  │
│      ELIF EventId in nonNormal_set:                            │
│        gt_label = 1  (NON-NORMAL)                              │
│      ELSE:                                                      │
│        gt_label = 2  (ANOMALY)                                 │
│                                                                  │
│ D. Output:                                                      │
│    - training_gt_labels (numpy array: 0/1/2)                   │
│    - testing_gt_labels (numpy array: 0/1/2)                    │
│                                                                  │
│ E. Statistics:                                                  │
│    - Count per class: N=?, NN=?, A=?                           │
│    - Verify distribution reasonable                            │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: LOAD TRAINING CLUSTER RESULTS                          │
│ - Load TRAINING_LABELS_PATH → cluster assignments              │
│ - Contoh: [1, 5, 3, 1, 2, ...]                                 │
│ - Verify length match dengan ground truth                      │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: ANALYZE CLUSTER CHARACTERISTICS                        │
│ Untuk setiap cluster, hitung:                                   │
│   cluster_id | n_samples | n_normal | n_anomaly | purity       │
│   ---------- | --------- | -------- | --------- | ------       │
│   0          | 245,678   | 240,123  | 5,555     | 0.9774       │
│   1          | 123,456   | 70,000   | 53,456    | 0.5670       │
│   5          | 120,000   | 117,600  | 2,400     | 0.9800       │
│   87         | 450       | 157      | 293       | 0.6511       │
│   142        | 45        | 12       | 33        | 0.7333       │
│   -1         | 5,420     | 1,234    | 4,186     | 0.7723       │
│                                                                  │
│ Klasifikasi cluster type:                                       │
│   - "noise"  → cluster_id = -1                                  │
│   - "small"  → n_samples < SMALL_CLUSTER_THRESHOLD              │
│   - "pure"   → purity > PURITY_THRESHOLD                        │
│   - "mixed"  → purity ≤ PURITY_THRESHOLD                        │
│                                                                  │
│ Save: cluster_analysis.csv                                      │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4-5: LOAD TESTING DATA                                    │
│ - Load test embeddings (multi-file vstack)                     │
│ - Load test metadata TSV → test_gt_labels                      │
│ - Verify dimension match dengan training                       │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: ASSIGN TEST SAMPLES TO CLUSTERS                        │
│                                                                  │
│ If K-Means:                                                     │
│   test_cluster_labels = model.predict(test_embeddings)         │
│                                                                  │
│ If DBSCAN:                                                      │
│   1. Build k-NN (k=1) pada training embeddings                 │
│   2. Find nearest training sample per test sample              │
│   3. Assign cluster dari nearest neighbor                      │
│                                                                  │
│ Output: test_cluster_labels (array of cluster IDs)             │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: HYBRID PREDICTION (CORE LOGIC)                         │
│                                                                  │
│ For each test sample with cluster assignment:                   │
│                                                                  │
│ ┌─ Check cluster_type ────────────────────────────────────┐    │
│ │                                                           │    │
│ │ case "noise":                                            │    │
│ │   prediction = 1 (ANOMALY)                               │    │
│ │   confidence = 0.8                                       │    │
│ │   method = "noise"                                       │    │
│ │                                                           │    │
│ │ case "small":                                            │    │
│ │   prediction = 1 (ANOMALY)                               │    │
│ │   confidence = 0.7                                       │    │
│ │   method = "small"                                       │    │
│ │                                                           │    │
│ │ case "pure":                                             │    │
│ │   prediction = cluster.dominant_label                    │    │
│ │   confidence = cluster.purity                            │    │
│ │   method = "pure"                                        │    │
│ │                                                           │    │
│ │ case "mixed":                                            │    │
│ │   neighbors = knn.find_k_nearest(sample, k=10)          │    │
│ │   vote = count_labels(neighbors)                         │    │
│ │   prediction = majority(vote)                            │    │
│ │   confidence = vote_ratio                                │    │
│ │   method = "knn"                                         │    │
│ │                                                           │    │
│ └───────────────────────────────────────────────────────────┘    │
│                                                                  │
│ Output:                                                         │
│   - predictions (numpy array)                                   │
│   - confidence (numpy array)                                    │
│   - methods (list of strings)                                  │
│                                                                  │
│ Save: predictions.npy                                           │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: CALCULATE METRICS                                      │
│                                                                  │
│ Compare predictions vs test_gt_labels:                         │
│   - Accuracy                                                    │
│   - Precision (anomaly detection precision)                    │
│   - Recall (anomaly detection rate)                            │
│   - F1-Score                                                   │
│   - Specificity (true negative rate)                           │
│   - Confusion Matrix: TN, FP, FN, TP                           │
│                                                                  │
│ Per-Method Analysis:                                            │
│   method     | samples | accuracy                              │
│   ---------- | ------- | --------                              │
│   noise      | 5,122   | 0.8234                                │
│   small      | 1,234   | 0.7456                                │
│   pure       | 1,950K  | 0.9456                                │
│   knn        | 345K    | 0.8123                                │
│                                                                  │
│ Save: metrics.txt                                               │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: SAVE DETAILED RESULTS                                  │
│                                                                  │
│ detailed_results.csv:                                           │
│   cluster_id | true_label | predicted | confidence | method    │
│   ---------- | ---------- | --------- | ---------- | ------    │
│   5          | 0          | 0         | 0.98       | pure      │
│   -1         | 1          | 1         | 0.80       | noise     │
│   23         | 1          | 1         | 0.70       | knn       │
│   87         | 0          | 1         | 0.65       | knn       │
│                                                                  │
│ Save: detailed_results.csv (per-row analysis)                  │
└─────────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 10: CREATE VISUALIZATIONS                                 │
│                                                                  │
│ 1. Cluster Purity Distribution (histogram)                     │
│    - X: Purity (0-1), Y: Number of clusters                    │
│    - Red line: PURITY_THRESHOLD = 0.95                         │
│                                                                  │
│ 2. Confusion Matrix Heatmap                                    │
│    - 2x2 matrix: TN, FP, FN, TP                                │
│    - Annotated with counts                                     │
│                                                                  │
│ 3. Cluster Type Distribution (bar chart)                       │
│    - X: Type (noise, small, pure, mixed)                       │
│    - Y: Number of clusters                                     │
│                                                                  │
│ 4. Cluster Size Distribution (log scale)                       │
│    - X: Cluster size, Y: Frequency (log scale)                 │
│    - Red line: SMALL_CLUSTER_THRESHOLD = 100                   │
│                                                                  │
│ Save:                                                           │
│   - analysis_overview.png (4-panel plot)                       │
│   - confusion_matrix.png (detailed heatmap)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Output Files & Interpretasi

### **1. `cluster_analysis.csv`** - Karakteristik Per Cluster

```csv
cluster_id,n_samples,n_normal,n_anomaly,pct_normal,pct_anomaly,purity,dominant_label,cluster_type,predicted_class
0,245678,240123,5555,97.74,2.26,0.9774,0,pure,NORMAL
1,123456,70000,53456,56.70,43.30,0.5670,0,low_purity,NON-NORMAL
5,120000,117600,2400,98.00,2.00,0.9800,0,pure,NORMAL
87,450,157,293,34.89,65.11,0.6511,1,low_purity,NON-NORMAL
102,180,95,85,52.78,47.22,0.5278,0,medium_purity,NON-NORMAL
142,35,8,27,22.86,77.14,0.7714,1,very_small,ANOMALY
-1,5420,1234,4186,22.77,77.23,0.7723,1,noise,ANOMALY
```

**Interpretasi:**
- **Pure clusters (purity > 0.95)**: 
  - Cluster 0, 5 → NORMAL (confident)
  - Jika ada pure anomaly → ANOMALY (confident)
  
- **Medium purity clusters (0.70-0.95)**: 
  - Cluster 102 → NON-NORMAL (borderline, suspicious)
  - Mixed distribution, tidak safe untuk binary classification
  
- **Low purity clusters (< 0.70)**: 
  - Cluster 1, 87 → NON-NORMAL (ambiguous)
  - Perlu k-NN vote untuk individual samples
  
- **Very small clusters (< 50)**: 
  - Cluster 142 → ANOMALY (rare critical patterns)
  
- **Noise (-1)**: 
  - DBSCAN outliers → ANOMALY (high confidence)

**3-Way Classification Interpretation:**
```
┌──────────────────────────────────────────────────────────────────┐
│ NORMAL (0): Expected routine operations                          │
│   - EventId matches normal_template.txt                         │
│   - Examples: E77 (cache parity corrected), E3 (alignment)      │
│   - Should cluster in large, pure clusters                      │
│   - Misclassification as NN/A = False Positive                  │
│                                                                  │
│ NON-NORMAL (1): Expected errors/warnings                         │
│   - EventId matches nonNormal_template.txt                      │
│   - Examples: E33 (APPREAD failed), E55 (TLB error)             │
│   - Should cluster in medium-sized, medium-purity groups        │
│   - Known issues but need attention                             │
│   - Misclassification as N = False Negative (missed warning)    │
│   - Misclassification as A = Over-reaction                      │
│                                                                  │
│ ANOMALY (2): Unknown/novel patterns                              │
│   - EventId NOT in either template                              │
│   - Security issues, data corruption, new failures              │
│   - Should be NOISE or very small clusters                      │
│   - Misclassification as N = CRITICAL MISS!                     │
│   - Misclassification as NN = Delayed response                  │
└──────────────────────────────────────────────────────────────────┘
```

**Cost-Sensitive Analysis:**
Most critical errors:
- **A → N (Anomaly missed as Normal)**: 1,234 cases - DANGEROUS!
  - True novel attacks/failures treated as routine
  - Need to lower confidence thresholds
  
- **NN → N (Non-Normal missed as Normal)**: 25,678 cases
  - Known errors not flagged
  - May indicate need for more sensitive clustering

Acceptable errors:
- **N → NN (Normal as suspicious)**: 35,678 cases
  - Conservative, better safe than sorry
  - Increases alert fatigue but no critical miss

### **2. `metrics.txt`** - Overall Performance

```
======================================================================
TESTING RESULTS - BGL KMEANS BASE (3-WAY CLASSIFICATION)
======================================================================

Overall Accuracy: 0.9234

Per-Class Metrics:
                  Precision  Recall   F1-Score  Support
  NORMAL (0)        0.9456    0.9678    0.9565   1,234,567
  NON-NORMAL (1)    0.7892    0.7345    0.7609     123,456
  ANOMALY (2)       0.8567    0.7123    0.7780      12,345

Macro Avg:          0.8638    0.8049    0.8318   1,370,368
Weighted Avg:       0.9201    0.9234    0.9212   1,370,368

Confusion Matrix (3x3):
                 Predicted
                 N      NN     A
    True  N  [1,194,123  35,678  4,766]
          NN [   25,678  90,678  7,100]
          A  [    1,234   2,345  8,766]

Error Analysis:
  N → NN:  35,678 (2.9%)  - Normal flagged as suspicious
  N → A:    4,766 (0.4%)  - Normal flagged as critical (false alarm)
  NN → N:  25,678 (20.8%) - Non-normal missed (treated as normal)
  NN → A:   7,100 (5.8%)  - Non-normal escalated to anomaly
  A → N:    1,234 (10.0%) - Critical anomaly missed (DANGEROUS!)
  A → NN:   2,345 (19.0%) - Anomaly downgraded to warning

======================================================================
PER-METHOD METRICS
======================================================================

NOISE (DBSCAN only): 5,122 samples (0.4%)
  N: 234 (4.6%)   NN: 1,234 (24.1%)   A: 3,654 (71.3%)
  Accuracy: 0.8234
  → Most noise points correctly identified as ANOMALY

VERY_SMALL (<50): 1,234 samples (0.1%)
  N: 123 (10.0%)   NN: 345 (28.0%)   A: 766 (62.0%)
  Accuracy: 0.7456
  → Rare patterns, mostly ANOMALY/NON-NORMAL

SMALL (50-200): 12,345 samples (0.9%)
  N: 2,345 (19.0%)   NN: 7,890 (63.9%)   A: 2,110 (17.1%)
  Accuracy: 0.8123
  → Correctly classified as NON-NORMAL

PURE (>95%): 1,950,432 samples (85.3%)
  N: 1,234,567 (63.3%)   NN: 98,765 (5.1%)   A: 617,100 (31.6%)
  Accuracy: 0.9456
  → High confidence, high accuracy

KNN (<70% purity): 345,678 samples (14.2%)
  N: 123,456 (35.7%)   NN: 178,900 (51.8%)   A: 43,322 (12.5%)
  Accuracy: 0.8123
  → Mixed clusters, moderate accuracy
```

**Interpretasi:**
- **High Accuracy pada Pure Method**: Cluster separation baik
- **Lower Accuracy pada KNN**: Mixed clusters sulit diprediksi (expected)
- **High Precision, Moderate Recall**: Conservative predictions

### **3. `detailed_results.csv`** - Per-Row Analysis

```csv
sample_idx,event_id,cluster_id,true_label,predicted_label,confidence,method,correct
0,E77,5,0,0,0.98,pure,1
1,E33,23,1,1,0.70,knn,1
2,UNKNOWN,-1,2,2,0.85,noise,1
3,E55,87,1,2,0.65,small,0
4,NOVEL142,142,2,2,0.75,very_small,1
5,E77,0,0,0,0.98,pure,1
6,E3,102,0,1,0.72,medium_purity,0
```

**Column Definitions:**
- `event_id`: EventId dari log (e.g., E77, E33, UNKNOWN)
- `cluster_id`: Cluster assignment dari K-means/DBSCAN
- `true_label`: Ground truth dari template matching
  - 0 = NORMAL (EventId in normal_template)
  - 1 = NON-NORMAL (EventId in nonNormal_template)
  - 2 = ANOMALY (EventId not in either template)
- `predicted_label`: Prediction dari hybrid strategy (0/1/2)
- `confidence`: Confidence score (0.0-1.0)
- `method`: Which strategy was used (noise/very_small/small/pure/medium_purity/knn)
- `correct`: 1 if prediction matches ground truth, 0 otherwise

**Use Cases:**
1. **Error Analysis**: Filter `correct=0` untuk lihat misclassifications
   ```python
   errors = df[df['correct'] == 0]
   errors.groupby(['true_label', 'predicted_label']).size()
   ```

2. **EventId Analysis**: Which EventIds are hard to classify?
   ```python
   error_events = errors['event_id'].value_counts()
   # E.g., E33 has high error rate → need special handling
   ```

3. **Method Performance**: Which strategy performs worst?
   ```python
   accuracy_per_method = df.groupby('method')['correct'].mean()
   # If 'knn' has low accuracy → increase KNN_NEIGHBORS
   ```

4. **Confidence Calibration**: Are low-confidence predictions actually wrong?
   ```python
   df.groupby(pd.cut(df['confidence'], bins=5))['correct'].mean()
   # If low correlation → confidence scores not calibrated
   ```

### **4. Visualizations**

**`analysis_overview.png`** (4-panel plot):
1. Cluster purity histogram → Assess separation quality
2. Confusion matrix → Overall classification performance
3. Cluster type distribution → Strategy distribution
4. Cluster size distribution → Identify rare vs common patterns

**`confusion_matrix.png`**:
- High TN + TP → Good predictions
- High FP → Too aggressive (many false alarms)
- High FN → Missing anomalies (need tuning)

---

## 🎓 Parameter Tuning Guide

### **Scenario A: Too Many False Positives (High FP)**

```python
# Make predictions more conservative
PURITY_THRESHOLD = 0.90  # Lower → More mixed clusters → More k-NN
SMALL_CLUSTER_THRESHOLD = 50  # Lower → Less aggressive "small=anomaly"
KNN_NEIGHBORS = 15  # More neighbors → Smoother decisions
```

### **Scenario B: Missing Anomalies (High FN)**

```python
# Make predictions more aggressive
PURITY_THRESHOLD = 0.98  # Higher → More pure clusters
SMALL_CLUSTER_THRESHOLD = 200  # Higher → More "small=anomaly"
KNN_NEIGHBORS = 5  # Fewer neighbors → More local sensitivity
```

### **Scenario C: Many Mixed Clusters (Low Purity)**

```
Problem: Clustering tidak optimal
Solution:
  - K-means: Increase k (more clusters = better separation)
  - DBSCAN: Tune eps/min_samples
  - Consider PCA dimensionality reduction
```

---

## ✅ Checklist Sebelum Run

- [ ] **Edit DATASET, ALGORITHM, EMBEDDING_TYPE**
- [ ] **Set METADATA_TSV_PATH** (training ground truth)
- [ ] **Set TRAINING_LABELS_PATH** (cluster assignments)
- [ ] **Set TRAINING_EMBEDDINGS_PATH** (training vectors)
- [ ] **Set TESTING_EMBEDDINGS_PATHS** (test vectors)
- [ ] **Set TESTING_METADATA_TSV** (test ground truth)
- [ ] **Adjust parameters** (optional: PURITY_THRESHOLD, etc.)
- [ ] **Run:** `python cluster_testing_pipeline.py`
- [ ] **Check output:** `testing_results/{dataset}_{algo}_{dim}/`

---

## 🔄 Complete Testing Matrix (24 Experiments)

```
For dataset in [BGL, Thunderbird]:
  For embedding in [base, pca256, pca128]:
    For algorithm in [kmeans, dbscan]:
      
      # Edit configuration
      DATASET = dataset
      ALGORITHM = algorithm
      EMBEDDING_TYPE = embedding
      
      # Update paths accordingly
      # ...
      
      # Run pipeline
      python cluster_testing_pipeline.py
      
      # Results saved to:
      # testing_results/{dataset}_{algorithm}_{embedding}/
```

**Total: 24 experiments**

**Output:** Comprehensive comparison table dengan F1-Score, Precision, Recall untuk setiap kombinasi.

---

## 📈 Success Criteria

### **Good Results:**
- F1-Score > 0.75
- Accuracy > 0.85
- Precision > 0.70 (low false alarm rate)
- Recall > 0.70 (detect most anomalies)

### **Excellent Results:**
- F1-Score > 0.85
- Accuracy > 0.92
- Balanced Precision & Recall (both > 0.80)
- Most predictions from "pure" method (> 80%)

### **Red Flags:**
- F1-Score < 0.60 → Re-cluster dengan parameter berbeda
- Most predictions from "knn" method (> 50%) → Poor cluster purity
- High FP rate (> 10%) → Too aggressive, tune thresholds

---

## 🎯 Thesis Contributions

1. **Novel Hybrid Prediction Strategy** untuk handle ratusan cluster
2. **Ground Truth Integration** dari metadata TSV
3. **Confidence Scoring** untuk prediction reliability
4. **Method-wise Performance Analysis** untuk interpretability
5. **Practical Guidelines** untuk parameter tuning

---

## 📝 Notes

### **Advantages vs Traditional Approaches:**

**Traditional Majority Vote:**
```python
# Simple but problematic
cluster_label = "Anomaly" if pct_anomaly > 50% else "Normal"
# Problem: 51% vs 49% treated sama dengan 98% vs 2%
```

**Our Hybrid Strategy:**
```python
# Intelligent multi-tier decision
if cluster_type == "pure":
    # Trust cluster label (high confidence)
elif cluster_type == "mixed":
    # Use k-NN vote (individual similarity)
elif cluster_type == "small":
    # Rare pattern (likely anomaly)
elif cluster_type == "noise":
    # Outlier (high prob anomaly)
```

### **Metadata TSV Format:**
```
label    unix_ts    date    ...
-        123456789  2023... ...  ← Normal
KERNERR  123456790  2023... ...  ← Anomaly
-        123456791  2023... ...  ← Normal
KERNCRIT 123456792  2023... ...  ← Anomaly
```

**Extraction:** First column, `-` → 0, else → 1

---

## 🔍 **3-Way Classification: K-means vs DBSCAN Deep Dive**

### **K-means Strategy:**

```python
# K-means memaksa semua points masuk cluster
# Tidak ada konsep "outlier" natural
# Strategi: Size + Purity based

if cluster_size < 50:
    # Very rare pattern
    prediction = ANOMALY
    confidence = 0.75
    
elif cluster_size < 200:
    # Unusual but not critical
    prediction = NON-NORMAL
    confidence = 0.65
    
elif purity > 0.95:
    # Clear separation
    if dominant_label == 0:
        prediction = NORMAL
    else:
        prediction = ANOMALY
    confidence = purity
    
elif 0.70 < purity <= 0.95:
    # Borderline cluster
    prediction = NON-NORMAL
    confidence = purity
    
else:  # purity <= 0.70
    # Very ambiguous, use k-NN
    neighbors = find_knn(sample, k=10)
    vote_normal = count(neighbors == 0)
    
    if vote_normal >= 8:
        prediction = NORMAL
        confidence = vote_normal / 10
    elif vote_normal >= 6:
        prediction = NON-NORMAL
        confidence = 0.60
    elif vote_normal >= 4:
        prediction = NON-NORMAL
        confidence = 0.50
    else:  # vote_normal <= 3
        if vote_normal <= 2:
            prediction = ANOMALY
            confidence = (10 - vote_normal) / 10
        else:
            prediction = NON-NORMAL
            confidence = 0.60
```

**K-means Advantages:**
- ✅ All points assigned (no "unknown" category)
- ✅ Predictable behavior (deterministic with fixed seed)
- ✅ Fast inference (distance to centroids)

**K-means Challenges:**
- ⚠️ Force assignment of true outliers
- ⚠️ Spherical cluster assumption
- ⚠️ Sensitive to k selection

---

### **DBSCAN Strategy:**

```python
# DBSCAN natural outlier detection via noise points
# Strategi: Noise + Density + Purity based

if cluster_id == -1:
    # Noise point = natural outlier
    prediction = ANOMALY
    confidence = 0.85
    # High confidence karena explicitly rejected by algorithm
    
elif cluster_size < 50:
    # Very rare dense region
    prediction = ANOMALY
    confidence = 0.75
    # Rare but density-connected
    
elif cluster_size < 200:
    # Unusual dense pattern
    prediction = NON-NORMAL
    confidence = 0.65
    
elif purity > 0.95:
    # Dense region dengan clear label
    if dominant_label == 0:
        prediction = NORMAL
    else:
        prediction = ANOMALY
    confidence = purity
    
elif 0.70 < purity <= 0.95:
    # Dense but mixed
    prediction = NON-NORMAL
    confidence = purity
    
else:  # purity <= 0.70
    # Dense but very ambiguous
    neighbors = find_knn(sample, k=10)
    # Same k-NN logic as K-means
```

**DBSCAN Advantages:**
- ✅ Natural outlier detection (noise points)
- ✅ Arbitrary shaped clusters
- ✅ No need to specify k
- ✅ Noise = ANOMALY with high confidence

**DBSCAN Challenges:**
- ⚠️ Sensitive to eps & min_samples
- ⚠️ Varying density = poor performance
- ⚠️ High dimensional curse

---

### **Comparison Example:**

**Scenario:** 1 test sample yang adalah true anomaly (rare error pattern)

```
┌─────────────────────────────────────────────────────────────┐
│ K-MEANS BEHAVIOR:                                           │
├─────────────────────────────────────────────────────────────┤
│ 1. Sample dipaksa masuk ke NEAREST centroid                │
│    → Assigned to cluster 45 (size=80)                      │
│                                                             │
│ 2. Cluster 45 characteristics:                             │
│    - Size: 80 (small threshold crossed)                    │
│    - Purity: 0.72 (medium)                                 │
│    - Dominant: Normal (58%)                                │
│                                                             │
│ 3. Decision:                                                │
│    cluster_size < 200 → NON-NORMAL                         │
│    Confidence: 0.65                                        │
│                                                             │
│ 4. Problem:                                                 │
│    True anomaly classified as NON-NORMAL                   │
│    (Conservative, missed critical anomaly)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ DBSCAN BEHAVIOR:                                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Sample evaluated for density-connectivity               │
│    → Not enough neighbors within eps                       │
│    → Assigned to cluster -1 (NOISE)                        │
│                                                             │
│ 2. Decision:                                                │
│    cluster_id == -1 → ANOMALY                              │
│    Confidence: 0.85                                        │
│                                                             │
│ 3. Result:                                                  │
│    ✅ True anomaly correctly detected                      │
│    ✅ High confidence score                                │
│    ✅ Natural outlier detection worked                     │
└─────────────────────────────────────────────────────────────┘
```

**Conclusion:**
- **K-means**: Better untuk balanced, well-separated data
- **DBSCAN**: Better untuk anomaly detection (natural outliers)

---

## 📊 **Evaluation Metrics untuk 3-Way Classification**

### **Confusion Matrix (3x3):**

```
                Predicted
              N    NN    A
True  N  [[  TN   FN2   FA  ]
      NN  [  FN   TNN   FA2 ]
      A   [  FN3  FN4   TA  ]]

Where:
  N  = Normal
  NN = Non-Normal
  A  = Anomaly
  
  TN  = True Normal
  TNN = True Non-Normal
  TA  = True Anomaly
  
  FN, FN2, FN3, FN4 = Different types of False Negatives
  FA, FA2 = Different types of False Alarms
```

### **Per-Class Metrics:**

```python
# Normal class (0)
Precision_Normal = TN / (TN + FN + FN3)
Recall_Normal = TN / (TN + FN2 + FA)
F1_Normal = 2 * (P * R) / (P + R)

# Non-Normal class (1)
Precision_NonNormal = TNN / (TNN + FN2 + FN4)
Recall_NonNormal = TNN / (FN + TNN + FA2)
F1_NonNormal = 2 * (P * R) / (P + R)

# Anomaly class (2)
Precision_Anomaly = TA / (TA + FA + FA2)
Recall_Anomaly = TA / (FN3 + FN4 + TA)
F1_Anomaly = 2 * (P * R) / (P + R)

# Macro Average
F1_Macro = (F1_Normal + F1_NonNormal + F1_Anomaly) / 3

# Weighted Average (by support)
F1_Weighted = (F1_Normal * support_N + 
               F1_NonNormal * support_NN + 
               F1_Anomaly * support_A) / total_samples
```

### **Cost-Sensitive Evaluation:**

```python
# Different misclassification costs
COST_MATRIX = {
    'N→NN':  1,   # Normal as Non-Normal (minor false alarm)
    'N→A':   5,   # Normal as Anomaly (major false alarm)
    'NN→N':  2,   # Non-Normal as Normal (missed warning)
    'NN→A':  2,   # Non-Normal as Anomaly (over-reaction)
    'A→N':   10,  # Anomaly as Normal (critical miss!)
    'A→NN':  5,   # Anomaly as Non-Normal (delayed response)
}

# Total cost
Total_Cost = sum(count[error_type] * COST_MATRIX[error_type] 
                 for error_type in COST_MATRIX)

# Cost-weighted F1
F1_Cost = harmonic_mean(Precision_Cost, Recall_Cost)
```

---

## 🎯 **Recommended Configuration per Algorithm**

### **For K-means (3-way):**

```python
ALGORITHM = "kmeans"
K = 150  # Or auto-determine via elbow/silhouette

# More conservative thresholds (force assignment)
PURITY_THRESHOLD_HIGH = 0.97      # Stricter for pure
PURITY_THRESHOLD_MEDIUM = 0.75    # Wider medium zone

VERY_SMALL_CLUSTER_THRESHOLD = 30  # Very aggressive
SMALL_CLUSTER_THRESHOLD = 150      # Conservative

KNN_NEIGHBORS = 15  # More neighbors for stability
KNN_HIGH_CONFIDENCE = 0.85  # 13/15 vote
KNN_MEDIUM_CONFIDENCE = 0.65  # 10/15 vote
```

### **For DBSCAN (3-way):**

```python
ALGORITHM = "dbscan"
# eps, min_samples auto-tuned dari experiments

# Less conservative (trust noise detection)
PURITY_THRESHOLD_HIGH = 0.93      # Relax a bit
PURITY_THRESHOLD_MEDIUM = 0.70    # Standard

VERY_SMALL_CLUSTER_THRESHOLD = 50  # Trust density
SMALL_CLUSTER_THRESHOLD = 200      # Standard

KNN_NEIGHBORS = 10  # Standard k
KNN_HIGH_CONFIDENCE = 0.80  # 8/10 vote
KNN_MEDIUM_CONFIDENCE = 0.60  # 6/10 vote

# DBSCAN-specific: noise handling
NOISE_AS_ANOMALY = True  # Always True for 3-way
NOISE_CONFIDENCE = 0.85  # High confidence for outliers
```

---

**Document Version:** 2.0  
**Last Updated:** March 7, 2026  
**Author:** Thesis Research - Log Anomaly Detection Project  
**Tool:** `cluster_testing_pipeline.py` - Hybrid Prediction Strategy
