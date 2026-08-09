
======================================================================
SECTION A: Missing Values & Outlier (setelah Cell 7)
======================================================================

## 4.5 Identifikasi Data Tidak Lengkap dan Outlier

### 4.5.1 Analisis Missing Values

Sebelum melakukan pemodelan, identifikasi kelengkapan data merupakan langkah kritis untuk memastikan integritas input model. Tabel berikut merangkum temuan missing values pada kedua dataset:

| Dataset | Kolom | Missing | Proporsi | Strategi Penanganan |
|---|---|---|---|---|
| **Data Klaim** | `Inpatient/Outpatient` | 37 | 0.80% | Dipertahankan; LightGBM menangani missing secara native melalui *default direction split* |
| | `Tanggal Pembayaran Klaim` | 37 | 0.80% | Fitur turunan `processing_days` diisi -999 sebagai *flag* |
| | `Lokasi RS` | 7 | 0.15% | Proporsi negligible, dipertahankan |
| | `ICD Diagnosis/Description` | 6 | 0.13% | Tidak digunakan sebagai fitur langsung |
| **Data Polis** | — | 0 | 0% | Tidak ada missing |

Secara keseluruhan, proporsi missing values sangat rendah (<1%) sehingga **tidak memerlukan imputasi agresif**. Pendekatan yang dipilih adalah mempertahankan seluruh record dan membiarkan model *tree-based* menangani missing values melalui mekanisme internal.

### 4.5.2 Analisis dan Penanganan Outlier

Dalam konteks asuransi, klaim bernilai sangat besar (*catastrophic claims*) merupakan fenomena yang **valid secara aktuaria** — bukan noise. Penghapusan outlier dapat menyebabkan **underestimasi risiko** yang berdampak pada *inadequate reserves*.

**Keputusan: Seluruh outlier DIPERTAHANKAN** dengan justifikasi:
1. Klaim ekstrem merepresentasikan risiko aktual portofolio yang harus diperhitungkan
2. Model *tree-based* (LightGBM) robust terhadap outlier karena menggunakan *histogram-based split*
3. Outlier terkonsentrasi pada klaim Singapore — konsisten dengan biaya medis internasional yang lebih tinggi

### 4.5.3 Normalisasi Data

Normalisasi (standardisasi, min-max scaling) **tidak diterapkan** dalam pipeline ini. Justifikasi:
- Model *tree-based* seperti LightGBM membuat keputusan berdasarkan **threshold split**, bukan jarak antar titik data
- Skala absolut fitur tidak mempengaruhi hasil split — berbeda dengan model berbasis jarak (SVM, KNN) atau model linear yang sensitif terhadap perbedaan skala
- Transformasi logaritmik pada target (`log1p`) telah dieksplorasi di EDA namun model dilatih pada **nilai asli** karena metrik evaluasi (MAPE) bersifat *scale-invariant*



======================================================================
SECTION B: Scatter Plot (setelah Cell 16)
======================================================================

### 5.9 Scatter Plot Analysis — Hubungan Antar Variabel

Visualisasi scatter plot digunakan untuk mengidentifikasi hubungan bivariat antara variabel prediktif dan target. Empat pasangan variabel berikut dipilih berdasarkan relevansi aktuaria:

| Plot | Variabel X | Variabel Y | Temuan |
|---|---|---|---|
| **Kiri Atas** | Usia Tertanggung | Nominal Klaim | Hubungan positif lemah; klaim Singapore (merah) mendominasi nilai ekstrem di semua kelompok usia |
| **Kanan Atas** | Lama Rawat Inap (LOS) | Nominal Klaim | Hubungan positif kuat; rawat inap (merah) memiliki LOS dan klaim lebih tinggi |
| **Kiri Bawah** | Waktu Proses Klaim | Nominal Klaim | Klaim bernilai besar memerlukan waktu verifikasi lebih panjang |
| **Kanan Bawah** | Biaya RS | Klaim Disetujui | Mayoritas mendekati garis y=x (100% approval); beberapa klaim di-cut oleh excess/deductible |



======================================================================
SECTION D: Model Comparison (setelah Cell 21)
======================================================================

## 12. Perbandingan Model Machine Learning

### 12.1 Desain Eksperimen

Untuk memenuhi persyaratan minimal 2 model ML berbeda, dilakukan benchmarking terhadap tiga algoritma *regression* yang umum digunakan untuk *tabular data*:

| Model | Arsitektur | Keunggulan Utama | Kelemahan |
|---|---|---|---|
| **LightGBM** | Gradient Boosting, *leaf-wise* | Efisiensi tinggi, native missing value handling | Risiko overfitting pada data kecil |
| **XGBoost** | Gradient Boosting, *level-wise* | Stabilitas dan regularisasi kuat | Lebih lambat dari LightGBM |
| **Random Forest** | Bagging, *independent trees* | Robust terhadap overfitting | Kurang menangkap interaksi kompleks |

Seluruh model dilatih dan dievaluasi menggunakan **fitur dan split data yang identik** untuk memastikan perbandingan yang *fair*:
- **Fitur**: 23 fitur (tanpa *data leakage*)
- **Split**: Time-based (Train=2024, Validation=2025)
- **Metrik**: MAPE, RMSE, R²

### 12.2 Hasil Perbandingan — Severity Model

### 12.3 Hasil Perbandingan — Frequency Model

### 12.4 Justifikasi Pemilihan Model Final

Berdasarkan hasil perbandingan, **LightGBM dipilih sebagai model final** untuk severity:
1. **MAPE terendah** pada validasi severity
2. **Efisiensi komputasi**: Waktu training lebih cepat
3. **Kompatibilitas SHAP**: Dukungan native TreeSHAP yang efisien
4. **Missing value handling**: Penanganan native tanpa perlu imputasi manual

Untuk frequency, meskipun Random Forest menunjukkan MAPE sedikit lebih baik, **LightGBM tetap dipilih** karena konsistensi framework dan kemampuan autoregressive yang lebih baik pada *rolling forecast*.



======================================================================
SECTION E: Cross-Validation (setelah Section D)
======================================================================

## 13. Validasi Model — Time Series Cross-Validation

### 13.1 Metodologi Expanding Window

Untuk memvalidasi stabilitas performa model, digunakan **Expanding Window Cross-Validation** yang menghormati urutan kronologis data:

| Fold | Training Period | Validation Period | # Train | # Val |
|---|---|---|---|---|
| **Fold 1** | 10 bulan pertama | 3 bulan berikutnya | ~2.482 | ~724 |
| **Fold 2** | 13 bulan pertama | 3 bulan berikutnya | ~3.206 | ~684 |
| **Fold 3** | 16 bulan pertama | Sisa data | ~3.890 | ~737 |

Pendekatan ini memastikan model **tidak pernah melihat data masa depan** — mensimulasikan skenario prediksi nyata.

### 13.2 Hasil Cross-Validation

### 13.3 Interpretasi

R² yang konsisten di kisaran 0.42–0.52 antar fold menunjukkan **stabilitas model yang baik**. Variasi MAPE yang lebih besar mengindikasikan bahwa model lebih sensitif terhadap perubahan *claim mix* antar periode.

> **Catatan — Frequency Model**: Cross-validation tradisional tidak feasible untuk model frequency karena hanya tersedia **19 observasi bulanan**. Pembagian ke fold-fold menghasilkan training set yang terlalu kecil. *Time-based holdout* (train=16 bulan, val=3 bulan) merupakan strategi terbaik yang tersedia.



======================================================================
SECTION F: Ablation Study (setelah Section E)
======================================================================

## 14. Ablation Study — Dampak Feature Engineering

### 14.1 Desain Eksperimen

Untuk mengukur kontribusi proses *feature engineering* terhadap performa model, dilakukan perbandingan:

| Konfigurasi | Fitur | Deskripsi |
|---|---|---|
| **Tanpa FE** | 7 fitur | Hanya kolom asli: bulan, tahun, tipe perawatan (encoded), metode klaim (encoded), gender (encoded), lokasi RS (encoded), plan code (encoded) |
| **Dengan FE** | 23 fitur | Kolom asli + fitur turunan: `los`, `age_at_claim`, `policy_tenure_years`, `month_sin/cos`, one-hot lokasi, plan, dan tipe perawatan |

### 14.2 Hasil Perbandingan

### 14.3 Interpretasi

Feature engineering memberikan peningkatan R² sebesar **+0.28** — sebuah peningkatan yang sangat substansial. Fitur-fitur derivatif seperti `los` (lama rawat inap), `age_at_claim` (usia saat klaim), dan `policy_tenure_years` (masa berlaku polis) menangkap informasi yang **tidak tersedia secara langsung** dari data mentah namun memiliki relevansi aktuaria yang tinggi.

Temuan ini mengkonfirmasi pentingnya *domain knowledge* aktuaria dalam proses rekayasa fitur — pengetahuan tentang faktor-faktor risiko asuransi kesehatan ditranslasikan menjadi fitur prediktif yang meningkatkan kemampuan model secara signifikan.



======================================================================
SECTION G: Forecast 2026 (setelah Cell 26)
======================================================================

## 15. Prediksi Klaim Tahun 2026

### 15.1 Metodologi

Prediksi klaim tahun 2026 merupakan **fokus utama analisis** sesuai case document. Pendekatan yang digunakan:

1. **Basis**: Output forecast Agustus–Desember 2025 digunakan sebagai lag features untuk bulan-bulan 2026
2. **Template**: Komposisi klaim menggunakan profil bulan yang bersesuaian dari 2024 (*seasonal template*)
3. **Autoregressive**: Setiap bulan diprediksi secara iteratif — output bulan sebelumnya menjadi input lag bulan berikutnya

### 15.2 Tabel Prediksi 2026

### 15.3 Visualisasi Timeline Lengkap

### 15.4 Analisis Trend dan Rekomendasi Bisnis

**Trend Kunci:**
- **Frequency**: Prediksi stabil di ~240 klaim/bulan sepanjang 2026, konsisten dengan pola historis
- **Severity**: Tertinggi di Q1 (Jan-Feb ~Rp66M), menurun di Q2-Q3 (~Rp50-55M), terendah di Q4 (~Rp49M)
- **Total Claim**: Estimasi tahunan ~Rp156.7 Miliar

**Rekomendasi:**
1. **Alokasi Cadangan Teknis**: Berdasarkan prediksi total klaim 2026 ~Rp156.7M, cadangan teknis perlu disiapkan dengan margin keamanan yang sesuai
2. **Seasonal Budgeting**: Alokasi klaim Q1 2026 harus lebih tinggi (~Rp32M) dibandingkan Q4 (~Rp35.6M) karena severity lebih tinggi di awal tahun
3. **Evaluasi Premi**: Jika tren klaim stabil, premi saat ini kemungkinan masih memadai — namun perlu monitoring terhadap perubahan *claim mix* (terutama proporsi Singapore)
4. **Manajemen Risiko**: Intensifkan *pre-authorization* untuk klaim Singapore yang menjadi *severity driver* utama



======================================================================
SECTION H: Total Claim SHAP (setelah Cell 35)
======================================================================

## 16. Analisis Faktor Total Nominal Klaim

### 16.1 Konteks

Selain severity (besaran per klaim) dan frequency (volume klaim), case document meminta identifikasi faktor yang mempengaruhi **total nominal klaim** (Total Claim = Frequency × Severity). Analisis ini penting karena faktor yang mendorong total klaim belum tentu sama dengan faktor yang mendorong severity atau frequency secara individual.

### 16.2 Pendekatan Dual

1. **Model SHAP**: Model LightGBM dilatih pada data agregat bulanan dengan target `total_claim`, menggunakan fitur komposisi, frequency, dan severity sebagai input
2. **Dekomposisi**: Perubahan total klaim didekomposisi menjadi kontribusi frequency vs severity — memungkinkan identifikasi apakah kenaikan total klaim disebabkan oleh **lebih banyak klaim** atau **klaim yang lebih mahal**

### 16.3 Hasil Analisis


