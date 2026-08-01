# Strategi Labeling Cluster untuk NonNormal dan Anomaly

## 1. Masalah Utama yang Terlihat

- Untuk kasus data sangat besar dan timpang (contoh Thunderbird 208 juta baris), ukuran cluster menjadi sinyal dominan.
- Jika 1 cluster menampung mayoritas data, aturan size-only akan bias:
  - cluster kecil terlalu mudah dilabel anomaly
  - cluster nonNormal sering tidak lolos mayoritas threshold
- Pada KMeans tidak ada label noise seperti DBSCAN (-1), jadi anomaly harus dibentuk dari sinyal lain (jarak, kepadatan relatif, kemurnian metadata, dan ketidakpastian).

## 2. Prinsip Umum (Berlaku untuk DBSCAN dan KMeans)

- Jangan pakai satu threshold global statis.
- Pakai threshold adaptif berbasis distribusi ukuran cluster, bukan angka absolut saja.
- Pisahkan threshold untuk NORMAL dan NONNORMAL (asymmetric threshold).
- Label ANOMALY hanya jika benar-benar ada sinyal kuat (rare, jauh dari centroid, campuran tinggi, atau noise).
- Untuk data sangat besar, gunakan sampling representatif per cluster agar hemat memori.

## 3. Tahap Diagnostik Wajib Sebelum Labeling

Lakukan profiling distribusi cluster terlebih dulu:

- Distribusi ukuran cluster: min, p1, p5, p10, p25, median, p75, p90, p95, p99, max.
- Rasio cluster terbesar terhadap total data.
- Silhouette per cluster (atau sample silhouette untuk skala besar).
- Untuk cluster besar, cek kemurnian metadata:
  - pct_normal
  - pct_nonnormal
  - unknown ratio

Jika cluster terbesar > 70% total data, aktifkan mode highly imbalanced.

## 4. Strategi Labeling DBSCAN

### 4.1 Aturan Dasar

- Cluster id = -1 -> ANOMALY.
- Cluster non-noise:
  - hitung pct_normal dan pct_nonnormal dari sample metadata per cluster
  - pakai threshold terpisah:
    - th_normal_high, misal 0.80 sampai 0.90
    - th_nonnormal_mid, misal 0.30 sampai 0.45

### 4.2 Aturan Keputusan yang Disarankan

- Jika pct_normal >= th_normal_high -> NORMAL.
- Jika pct_nonnormal >= th_nonnormal_mid -> NONNORMAL.
- Jika keduanya tidak terpenuhi:
  - jika cluster sangat kecil (berdasarkan quantile size terendah) -> ANOMALY
  - selain itu -> NONNORMAL (bukan langsung ANOMALY)

Alasan: pada praktik log data, NONNORMAL sering heterogen dan jarang mencapai mayoritas tinggi.

### 4.3 Threshold Size Adaptif untuk DBSCAN

- anomaly_size_threshold = max(50, p5 ukuran cluster)
- nonnormal_size_upper = p25 ukuran cluster

Interpretasi:
- ukuran < anomaly_size_threshold -> kandidat ANOMALY
- ukuran di antara anomaly_size_threshold dan nonnormal_size_upper -> kandidat NONNORMAL

## 5. Strategi Labeling KMeans (Paling Penting)

KMeans tidak punya noise, jadi butuh pseudo-anomaly scoring.

### 5.1 Sinyal yang Dipakai per Cluster

- Size signal:
  - size_ratio = ukuran_cluster / total_data
  - size_quantile_position terhadap semua ukuran cluster
- Distance signal:
  - median jarak ke centroid
  - tail ratio, contoh proporsi anggota dengan jarak > p95 global
- Metadata purity signal:
  - pct_normal
  - pct_nonnormal
  - entropy campuran label
- Separation signal:
  - mean silhouette cluster

### 5.2 Aturan Label Cluster KMeans (3 kelas)

- NORMAL:
  - pct_normal >= th_normal_high
  - dan distance tail rendah
- NONNORMAL:
  - pct_nonnormal >= th_nonnormal_mid
  - dan bukan cluster ultra-rare
- ANOMALY:
  - cluster ultra-kecil (quantile size paling bawah)
  - atau distance tail sangat tinggi
  - atau entropy sangat tinggi + silhouette rendah (cluster campur dan tidak stabil)

### 5.3 Asymmetric Majority Threshold (Kunci untuk kasusmu)

Gunakan dua threshold berbeda:

- th_normal_high = 0.80 sampai 0.90
- th_nonnormal_mid = 0.30 sampai 0.45

Jangan pakai satu MAJORITY_THRESHOLD untuk semua kelas, karena itu membuat NONNORMAL sering 0%.

### 5.4 Tangani Giant Cluster (misal 200 juta data di 1 cluster)

Lakukan dua tahap:

- Tahap 1: labeling kasar antar cluster.
- Tahap 2: khusus giant cluster, lakukan sub-clustering ulang:
  - MiniBatchKMeans atau HDBSCAN pada sampel giant cluster
  - lalu labeling ulang sub-cluster dengan aturan yang sama

Ini mencegah semua variasi nonNormal tenggelam di satu cluster besar.

## 6. Aturan Fallback Praktis untuk Mengurangi Over-Detection ANOMALY

Jika cluster ambigu (tidak lolos normal dan nonnormal):

- default ke NONNORMAL, bukan ANOMALY,
- kecuali ada bukti kuat anomaly:
  - ultra-rare size
  - distance tail ekstrem
  - silhouette sangat buruk

Tujuan: menekan false anomaly dan menaikkan recall NONNORMAL.

## 7. Evaluasi yang Harus Dipantau (Bukan Akurasi Saja)

Untuk tiap model, pantau:

- Recall NONNORMAL
- Precision NONNORMAL
- Persentase data yang diprediksi ANOMALY
- Distribusi prediksi per test set (normal vs nonnormal)
- Error breakdown:
  - NONNORMAL -> ANOMALY
  - NONNORMAL -> NORMAL
  - NORMAL -> ANOMALY

Target awal tuning:

- turunkan NONNORMAL -> ANOMALY (yang sekarang dominan)
- naikkan proporsi prediksi NONNORMAL tanpa meledakkan false positive pada normal

## 8. Rekomendasi Tuning Bertahap

Urutan tuning yang aman:

1. Aktifkan threshold asimetris normal vs nonnormal.
2. Ubah fallback ambiguous dari ANOMALY menjadi NONNORMAL.
3. Ganti size threshold fixed menjadi quantile-based.
4. Tambah distance-tail signal untuk KMeans anomaly.
5. Jika masih bias karena giant cluster, lakukan sub-clustering pada giant cluster.

## 9. Contoh Konfigurasi Awal untuk Thunderbird Besar

- th_normal_high = 0.85
- th_nonnormal_mid = 0.35
- anomaly_size_threshold = max(50, p5_size)
- nonnormal_size_upper = p25_size
- ambiguous_default = NONNORMAL
- cluster dengan distance_tail sangat tinggi tetap ANOMALY

## 10. Ringkasan Keputusan untuk 2 Model

- DBSCAN:
  - andalkan noise + size quantile + metadata purity asimetris
  - ambiguous lebih aman ke NONNORMAL
- KMeans:
  - wajib tambah distance/separation signal karena tidak ada noise
  - wajib asimetris threshold
  - wajib treatment khusus giant cluster

Dengan strategi ini, label NONNORMAL tidak lagi collapse ke 0%, dan ANOMALY tidak over-detected hanya karena ukuran cluster.
