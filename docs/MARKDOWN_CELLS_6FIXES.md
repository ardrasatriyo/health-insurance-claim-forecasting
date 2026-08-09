
======================================================================
FIX6_MD
======================================================================
## 12. Penanganan Missing Values dan Outlier

### 12.1 Identifikasi Data Tidak Lengkap (*Missing Values*)

Sebelum melakukan pemodelan, identifikasi data tidak lengkap merupakan langkah wajib untuk memastikan integritas input model. Analisis berikut menampilkan jumlah dan proporsi *missing values* per kolom dalam dataset.

### 12.2 Strategi Penanganan Missing Values

| Kolom | Jumlah Missing | Strategi | Justifikasi |
|---|---|---|---|
| `Inpatient/Outpatient` | 37 (0.8%) | Dipertahankan, di-encode sebagai kategori tersendiri | Proporsi sangat kecil; menghapus 37 record mengurangi representativitas data |
| `Tanggal Pembayaran Klaim` | 37 (0.8%) | Diisi dengan -999 sebagai *flag* | LightGBM mampu menangani *missing value flag* secara native melalui *default direction* pada tree split |
| `Lokasi RS` | 7 (0.15%) | Dipertahankan | Proporsi negligible |
| `ICD Diagnosis / Description` | 6 (0.13%) | Dipertahankan | Tidak digunakan sebagai fitur langsung dalam model |

### 12.3 Analisis dan Penanganan Outlier

Dalam konteks asuransi kesehatan, **outlier merupakan klaim yang valid secara aktuaria** — klaim ekstrem (*catastrophic claims*) adalah fenomena yang nyata dan menjadi komponen penting dalam perhitungan *expected loss*. Penghapusan outlier dapat menyebabkan **underestimasi risiko** yang berdampak pada *inadequate pricing* dan *insufficient reserves*.

| Metode | Threshold | Jumlah Outlier | Proporsi |
|---|---|---|---|
| **IQR** (Q3 + 1.5×IQR) | — | 507 | 11.0% |
| **Persentil 95** | 223.3 Juta Rp | 232 | 5.0% |
| **Persentil 99** | 633.7 Juta Rp | 47 | 1.0% |

**Keputusan: Outlier DIPERTAHANKAN** dengan justifikasi berikut:
1. Klaim bernilai besar merupakan representasi valid dari risiko aktual portofolio
2. Model *tree-based* (LightGBM) relatif robust terhadap outlier karena menggunakan *histogram-based split* yang tidak dipengaruhi oleh magnitude absolut
3. Normalisasi/standarisasi **tidak diperlukan** untuk model *tree-based* — berbeda dengan model berbasis jarak (SVM, KNN) atau model linear yang sensitif terhadap skala fitur


======================================================================
FIX1_MD
======================================================================
## 13. Perbandingan Model (*Model Benchmarking*)

### 13.1 Latar Belakang

Untuk memastikan bahwa pemilihan LightGBM sebagai model utama didasarkan pada bukti empiris, dilakukan perbandingan dengan dua algoritma *machine learning* alternatif:

| Model | Arsitektur | Keunggulan |
|---|---|---|
| **LightGBM** | Gradient Boosting, *leaf-wise growth*, *histogram-based* | Efisiensi komputasi tinggi, native handling missing values |
| **XGBoost** | Gradient Boosting, *level-wise growth*, *exact greedy* | Stabilitas tinggi, regularisasi built-in yang kuat |
| **Random Forest** | Bagging, *independent trees* | Robust terhadap overfitting, tidak memerlukan tuning intensif |

### 13.2 Protokol Eksperimen

Seluruh model dilatih dan dievaluasi dengan kondisi yang **identik** untuk memastikan perbandingan yang *fair*:
- **Fitur**: 23 fitur yang sama (tanpa *data leakage*)
- **Split**: Time-based split (Train=2024, Validation=2025)
- **Metrik**: MAPE, RMSE, dan R²

### 13.3 Hasil Perbandingan — Severity Model


======================================================================
FIX1_MD_FREQ
======================================================================
### 13.4 Hasil Perbandingan — Frequency Model

### 13.5 Justifikasi Pemilihan Model Final

Berdasarkan hasil perbandingan, **LightGBM dipilih sebagai model final** dengan pertimbangan:

1. **Performa kompetitif**: LightGBM menunjukkan MAPE terendah untuk severity dan performa frequency yang kompetitif
2. **Efisiensi**: Waktu training LightGBM lebih cepat dibandingkan XGBoost dan Random Forest
3. **Kompatibilitas SHAP**: LightGBM memiliki dukungan native TreeSHAP yang efisien untuk analisis interpretabilitas
4. **Handling missing values**: Native support untuk missing values tanpa perlu imputasi manual


======================================================================
FIX2_MD
======================================================================
## 14. Validasi Model — Time Series Cross-Validation

### 14.1 Metodologi

Untuk memvalidasi stabilitas performa model terhadap variasi data temporal, digunakan **Expanding Window Cross-Validation** — varian *k-fold CV* yang menghormati urutan kronologis data:

| Fold | Training Set | Validation Set | Rasional |
|---|---|---|---|
| **Fold 1** | 10 bulan pertama | 3 bulan berikutnya | Model dilatih dengan data minimal |
| **Fold 2** | 13 bulan pertama | 3 bulan berikutnya | Training set diperluas |
| **Fold 3** | 16 bulan pertama | Sisa data | Training set maksimal |

**Mengapa Expanding Window?** Berbeda dengan *k-fold CV* standar yang mencampur data secara acak, pendekatan ini memastikan bahwa model **tidak pernah melihat data masa depan** — sesuai dengan skenario prediksi nyata (*forward-looking validation*).

### 14.2 Hasil Cross-Validation — Severity Model

### 14.3 Interpretasi

> **Catatan mengenai Frequency Model**: Cross-validation tradisional tidak feasible untuk model frequency karena hanya tersedia **19 observasi bulanan**. Membagi 19 observasi ke dalam fold-fold akan menghasilkan training set yang terlalu kecil untuk menghasilkan model yang stabil. Oleh karena itu, *time-based holdout* (train=16 bulan, val=3 bulan) yang telah digunakan merupakan pilihan terbaik mengingat keterbatasan data ini.


======================================================================
FIX3_MD
======================================================================
## 15. Ablation Study — Dampak Feature Engineering

### 15.1 Tujuan

Untuk mengukur kontribusi nyata proses *feature engineering* terhadap performa model, dilakukan eksperimen ablasi (*ablation study*) yang membandingkan:

| Konfigurasi | Jumlah Fitur | Fitur yang Digunakan |
|---|---|---|
| **Tanpa FE** | 7 fitur | Hanya kolom asli: bulan, tahun, lokasi RS (encoded), plan code (encoded), tipe perawatan (encoded), tipe klaim (encoded), gender (encoded) |
| **Dengan FE** | 23 fitur | Kolom asli + derived features: `los`, `age_at_claim`, `policy_tenure_years`, `month_sin/cos`, one-hot encoding lokasi & plan, dll. |

### 15.2 Hasil Ablation Study

### 15.3 Interpretasi

Feature engineering memberikan **peningkatan substansial** pada seluruh metrik evaluasi. Fitur-fitur derivatif seperti `los` (lama rawat inap), `age_at_claim` (usia saat klaim), dan `policy_tenure_years` (tenure polis) menangkap informasi yang **tidak tersedia secara langsung** dari data mentah, namun memiliki relevansi aktuaria yang tinggi terhadap besaran klaim.

Temuan ini mengkonfirmasi pentingnya *domain knowledge* dalam proses rekayasa fitur — pengetahuan tentang faktor-faktor risiko asuransi kesehatan ditranslasikan menjadi fitur prediktif yang meningkatkan kemampuan model secara signifikan.


======================================================================
FIX4_MD
======================================================================
## 16. Forecasting Klaim Tahun 2026

### 16.1 Metodologi

Prediksi klaim tahun 2026 dilakukan dengan memperpanjang pendekatan *autoregressive bottom-up forecasting* yang sama:
- **Basis**: Output forecast Agustus–Desember 2025 digunakan sebagai lag features untuk Januari 2026
- **Template**: Komposisi klaim menggunakan template bulan yang bersesuaian dari tahun 2024
- **Iteratif**: Setiap bulan 2026 diprediksi secara berurutan, dengan update lag features dari prediksi bulan sebelumnya

### 16.2 Hasil Prediksi 2026

### 16.3 Visualisasi Timeline Lengkap

### 16.4 Rekomendasi Bisnis Berdasarkan Prediksi 2026

Berdasarkan tren prediksi, beberapa rekomendasi strategis dapat dirumuskan:

1. **Alokasi Cadangan Teknis**: Total prediksi klaim 2026 menunjukkan tren yang stabil, mendukung strategi reserving yang konservatif namun tidak berlebihan
2. **Seasonal Budgeting**: Pola severity yang lebih tinggi di Q1 2026 (Jan-Feb) perlu diantisipasi dalam alokasi anggaran klaim per kuartal
3. **Premium Review**: Jika tren prediksi mengindikasikan peningkatan klaim, evaluasi kecukupan premi (*premium adequacy*) perlu dilakukan sebelum siklus renewal polis


======================================================================
FIX5_MD
======================================================================
## 17. Analisis Faktor Nominal Klaim (*Total Claim*)

### 17.1 Konteks

Selain menganalisis faktor-faktor yang mempengaruhi **severity** (besaran per klaim) dan **frequency** (volume klaim), case document juga meminta identifikasi faktor yang mempengaruhi **total nominal klaim** — yaitu Total Claim = Frequency × Severity.

### 17.2 Pendekatan

Analisis dilakukan melalui dua cara komplementer:

1. **Model SHAP**: Model LightGBM dilatih pada data bulanan dengan target `total_claim`, menggunakan fitur komposisi, severity, dan frequency sebagai input. SHAP values menunjukkan kontribusi relatif setiap faktor.

2. **Dekomposisi Naratif**: Total nominal klaim merupakan produk dari frequency dan severity. Perubahan total klaim dapat didekomposisi menjadi:
   - Perubahan **volume** (frequency): dipengaruhi oleh pola seasonal, pertumbuhan polis, dan *utilization rate*
   - Perubahan **intensitas** (severity): dipengaruhi oleh *case mix*, lokasi RS, dan *medical inflation*

### 17.3 Hasil Analisis

