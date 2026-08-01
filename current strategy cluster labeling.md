# Current Strategy — Mapping KMeans & DBSCAN (kode)

Dokumen ini memetakan alur implementasi labeling & prediksi di `cluster_testing_pipeline.py` untuk dua algoritma (KMeans, DBSCAN) dan dua dataset (Thunderbird, BGL).

Referensi utama: [cluster_testing_pipeline.py](cluster_testing_pipeline.py)

---

## Ringkasan alur utama (pipeline)

1. Load konfigurasi & path (variabel seperti `USE_METADATA_LABELING`, `METADATA_SAMPLE_SIZE`, `MAJORITY_THRESHOLD`, `MIN_CLUSTER_SIZE_FOR_LABELING`, `KMEANS_*` etc.).
2. Muat training labels/embeddings (paths: `TRAINING_LABELS_PATH`, `TRAINING_EMBEDDINGS_PATH`).
3. Analisis karakteristik cluster training → `analyze_cluster_characteristics(...)` → menghasilkan `cluster_df` dan `cluster_dict` (cluster-level stats + `cluster_label`).
4. Load testing sets → `load_multiple_testing_sets(...)`.
5. Assign test samples ke cluster training:
   - KMeans: `model.predict(test_embeddings)` atau centroid fallback `predict_kmeans_from_centroids(...)`.
   - DBSCAN: approximate/exact nearest-neighbour → `fast_cluster_assignment_faiss(...)` atau `fast_cluster_assignment_sklearn_batched(...)`.
6. Prediksi label 3-kelas via `hybrid_predict(...)` (jika `USE_METADATA_LABELING` True → direct lookup dari `cluster_dict`).
7. Hitung metrik → `calculate_metrics(...)`.
8. Simpan hasil dan visualisasi (`visualize_results`, `analyze_prediction_distribution`).

---

## KMeans — implementasi dan per-dataset notes

### Umum (kode terkait)
- Model load / centroid fallback: `load_kmeans_model_compat()`, `build_kmeans_centroids_from_labels()`, `predict_kmeans_from_centroids()`.
- Threshold/adaptive rules: `KMEANS_ANOMALY_CLUSTER_RATIO`, `KMEANS_SMALL_CLUSTER_RATIO`, `KMEANS_ANOMALY_AVG_CLUSTER_RATIO`, `KMEANS_SMALL_AVG_CLUSTER_RATIO`.
- Cluster analysis: `analyze_cluster_characteristics()` (menghasilkan `cluster_label`, `label_name`, `pct_normal`, `pct_nonnormal`, `labeling_reason`).

### KMeans — Thunderbird
- Training: pipeline mencoba load model (`TRAINED_MODEL_PATH`) — jika tidak ada, membangun centroid dari `TRAINING_EMBEDDINGS_PATH` + `TRAINING_LABELS_PATH` chunked via `build_kmeans_centroids_from_labels()`.
- Labeling cluster: jika `USE_METADATA_LABELING` True dan `METADATA_TSV_PATH` tersedia, pipeline membangun/men-load memmap label via `build_metadata_label_memmap()` (streaming, uint8) untuk Thunderbird. `analyze_cluster_characteristics()` melakukan sampling per-cluster (maks `METADATA_SAMPLE_SIZE`) lalu majority vote menggunakan `MAJORITY_THRESHOLD`.
- Silhouette: untuk Thunderbird silhouette dihitung pada sampel terbatas (adaptive cap, mis. 200k) agar aman memori.
- Test assignment: jika model ada → `model.predict(test_embeddings)`; else centroid nearest assignment (`predict_kmeans_from_centroids`).
- Prediksi: `hybrid_predict()` memakai `cluster_dict` hasil analisis untuk mapping cluster → label (NORMAL/NON-NORMAL/ANOMALY). Pada KMeans, tidak ada noise; ANOMALY ditentukan dari ukuran ultra-kecil, distance-tail, entropy campuran, atau siluet rendah.

### KMeans — BGL
- Metadata handling: BGL biasanya dimuat langsung ke RAM (pandas read) jika kecil; jika besar, fallback chunked streaming memanggil `_load_metadata_chunked()`.
- Thresholds: KMeans adaptive rules sama, namun memori lebih longgar sehingga sampling/silhouette bisa lebih besar.
- Giant-cluster handling: saat ini pipeline hanya memakai adaptive thresholds; sub-clustering giant cluster belum otomatis — direkomendasikan sebagai langkah berikutnya.

---

## DBSCAN — implementasi dan per-dataset notes

### Umum (kode terkait)
- Training labels: `TRAINING_LABELS_PATH` (cluster ids termasuk `-1` noise).
- Cluster analysis: `analyze_cluster_characteristics()` (cluster `-1` langsung diperlakukan sebagai `ANOMALY`).
- Test assignment: mapping test → nearest training vector via `fast_cluster_assignment_faiss()` (FAISS IVF, dengan subsampling/batching) atau fallback `fast_cluster_assignment_sklearn_batched()`.
- Prediction: `hybrid_predict()` akan menggunakan `cluster_dict` untuk langsung menetapkan label ketika `USE_METADATA_LABELING` True; legacy k-NN hanya jika metadata disabled.

### DBSCAN — Thunderbird
- Metadata: pipeline khusus untuk Thunderbird memakai streaming memmap (`build_metadata_label_memmap()`), sehingga analisis per-cluster (count normal/nonnormal) tidak memaksa RAM besar.
- Noise: karena DBSCAN menghasilkan `-1` untuk noise, pipeline langsung mengkategorikannya sebagai `ANOMALY` di `analyze_cluster_characteristics()`.
- Test assignment: fungsi FAISS yang di-patch agar aman memori (subsampling training jika sangat besar + batch-wise normalization) digunakan untuk meng-assign ratusan ribu/ juta sampel testing tanpa OOM.
- Prediksi: direct cluster lookup dari `cluster_dict` (metadata-based) — sangat cepat dan stabil jika metadata-labeled clusters berkualitas.

### DBSCAN — BGL
- Metadata: BGL biasanya dimuat sebagai `pandas` dataframe label-column (lebih kecil), sehingga per-cluster majority lebih mudah diterapkan.
- Noise handling sama: `-1` => `ANOMALY`.

---

## Perbedaan praktis KMeans vs DBSCAN di kode

- DBSCAN: mempunyai konsep noise (`-1`) → ANOMALY lebih eksplisit.
- KMeans: semua titik diberi cluster → pipeline menggunakan distance-tail, entropy, dan ukuran quantile untuk mendeteksi pseudo-anomaly.
- Metadata-based labeling: sama di keduanya (`analyze_cluster_characteristics()`), tetapi untuk KMeans threshold adaptif KMEANS_* berperan lebih besar.
- Test sample assignment: KMeans → model.predict() atau centroids; DBSCAN → nearest-neighbour ke training vectors (FAISS/Sklearn).

---

## Lokasi kode untuk perubahan cepat

- Ubah majority / fallback behavior: `analyze_cluster_characteristics()` in `cluster_testing_pipeline.py` (ganti `MAJORITY_THRESHOLD` logic atau tambahkan asymmetric thresholds).
- Ubah ambiguous fallback dari `ANOMALY` → `NON-NORMAL`: update branch where `labeling_reason=='mixed_ambiguous'` inside `analyze_cluster_characteristics()`.
- Implement sub-clustering giant cluster: implement in main flow after `analyze_cluster_characteristics()` detects cluster with `n_samples` >> (e.g., > 50% total) — call MiniBatchKMeans/HDBSCAN on sample and then re-run per-subcluster labeling.

---

## Opsi implementasi berikutnya

Jika mau, saya bisa sekarang:

- (A) Tambah asymmetric threshold config (`TH_NORMAL_HIGH`, `TH_NONNORMAL_MID`) dan ubah logic `analyze_cluster_characteristics()`.
- (B) Ubah fallback ambiguous → `NON-NORMAL` secara langsung.
- (C) Implement sub-clustering otomatis untuk giant cluster (lebih besar effort).

Pilih opsi (A/B/C) atau minta saya apply semua langkah bertahap.
