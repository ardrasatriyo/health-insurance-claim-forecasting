# Prediksi Klaim Asuransi Kesehatan dengan GBDT + Explainable AI

Arsip kode dan hasil untuk **Data Science Competition — Mathematical Challenge Festival ITB 2026**.

**Tim TehTarikMakNani** — Ardra Chandra Satriyo, Muhammad Rayhan Ayman Mangku, Fakhrul Zidan Nurrohman
**Hasil:** MAPE portofolio 0,22877 di Kaggle — peringkat 18 dari 296 tim.

---

## Masalah

Memprediksi klaim asuransi kesehatan individu AXA Financial Indonesia untuk **Agustus–Desember 2025** (dinilai di Kaggle) dan memproyeksikan **2026**, pada tiga komponen: frekuensi klaim, severitas (rata-rata nilai per klaim), dan total nominal — semuanya di level agregat bulanan.

Data latih: 4.627 klaim dan 4.096 polis, Januari 2024 – Juli 2025 (19 bulan). **Dataset tidak ada di repo ini** — bersifat rahasia. Skemanya didokumentasikan di [`data/README.md`](data/README.md).

## Pendekatan

**Dekomposisi frequency–severity** — pendekatan standar aktuaria: total klaim = frekuensi × severitas, dengan kedua komponen dimodelkan terpisah karena dinamikanya berbeda.

| Model | Level | Observasi | Target |
|---|---|---|---|
| Severity | Per klaim individu | 4.627 (train 2024 = 2.990, val 2025 = 1.637) | `Nominal Klaim Yang Disetujui` |
| Frequency | Agregat bulanan | 19 bulan | Jumlah klaim per bulan |

Forecasting bersifat **autoregressive bottom-up**: prediksi bulan sebelumnya jadi *lag feature* bulan berikutnya. Untuk severitas, komposisi portofolio bulan yang sama tahun 2024 dipakai sebagai template.

**Feature engineering** menghasilkan 23 fitur dari 7 kolom mentah: temporal (`month`, `quarter`, `month_sin`, `month_cos`), medis (`los`, `processing_days`), demografis (`age_at_claim`, `policy_tenure_years`), tipe perawatan, lokasi RS (Indonesia/Singapura/Malaysia/overseas), dan plan code. Encoding siklik sin/cos dipakai supaya Desember dan Januari berdekatan secara numerik.

`Nominal Biaya RS Yang Terjadi` **sengaja dibuang** dari fitur — berkorelasi langsung dengan target, jadi *data leakage*.

Interpretasi memakai **SHAP** (TreeSHAP) di ketiga model.

## Hasil

### Perbandingan model severity (level klaim individu)

Fitur, split, dan hyperparameter identik untuk ketiganya. Sumber: [`outputs/model_comparison_severity.png`](outputs/model_comparison_severity.png).

| Model | MAPE | RMSE | R² |
|---|---|---|---|
| LightGBM | **792,66 %** | Rp 100,8 jt | 0,4664 |
| XGBoost | 810,47 % | **Rp 99,6 jt** | **0,4796** |
| Random Forest | 837,82 % | Rp 104,4 jt | 0,4280 |

LightGBM dipilih sebagai model utama karena unggul di MAPE — metrik yang dinilai kompetisi. XGBoost sebenarnya sedikit lebih baik di RMSE dan R².

> **Kenapa MAPE ~790 % tapi skor Kaggle 0,229?** Keduanya mengukur hal berbeda. MAPE di atas dihitung **per klaim individu**, dan distribusi klaim sangat *right-skewed*: klaim bernilai kecil menghasilkan error relatif raksasa meski selisih absolutnya sepele. Kompetisi menilai MAPE di **level portofolio bulanan**, di mana agregasi meredam variabilitas individual — di situlah angka 22,8 % berasal. R² ≈ 0,47 adalah indikator kualitas model yang jauh lebih jujur untuk kasus ini.

### Cross-validation (expanding window, 3 fold)

Sumber: [`outputs/cv_results.png`](outputs/cv_results.png). Split kronologis 10→3 bulan, 13→3 bulan, 16→sisa, sehingga tidak ada kebocoran temporal.

Rata-rata **R² = 0,4799** (per fold: 0,504 / 0,521 / 0,415) dan MAPE 769,28 %. Standar deviasi R² ≈ 0,05 — performa stabil antar periode, tidak ada indikasi overfitting ke satu rentang waktu tertentu.

### Ablation study — dampak feature engineering

Sumber: [`outputs/ablation_fe.png`](outputs/ablation_fe.png).

| Konfigurasi | Fitur | MAPE | RMSE | R² |
|---|---|---|---|---|
| Tanpa FE | 7 (bulan, tahun + 5 kategorikal label-encoded) | 1282,89 % | Rp 124,6 jt | 0,1855 |
| Dengan FE | 23 | 792,66 % | Rp 100,8 jt | 0,4664 |

R² naik **+0,28** — kontribusi terbesar datang dari `los`, `age_at_claim`, dan encoding lokasi RS.

### Faktor determinan (SHAP)

**Severity** — lama rawat inap (LOS) sebagai proxy kompleksitas medis, lokasi RS Singapura (severitas 3–5× klaim domestik), waktu proses klaim, usia tertanggung (kontribusi naik tajam setelah 50 tahun), dan tipe perawatan rawat inap.

**Frequency** — didominasi lag-1 frekuensi (persistensi temporal kuat), lalu encoding musiman dan rasio komposisi portofolio.

**Total klaim** — severitas jadi *driver* yang lebih dominan dibanding frekuensi, karena frekuensi relatif stasioner sepanjang 19 bulan observasi.

Grafik: `outputs/shap_*.png`, ringkasan naratif di [`outputs/xai_report.txt`](outputs/xai_report.txt) dan [`outputs/xai_business_report.txt`](outputs/xai_business_report.txt).

### Forecast

**Agustus–Desember 2025** — model menghasilkan frekuensi datar 239,9/bulan dengan severitas 53,3 / 56,4 / 49,1 / 50,8 / 48,6 juta (notebook sel 26).

**2026** — frekuensi stabil di ~238 klaim/bulan, severitas memuncak di Q1 (~78–79 juta) lalu turun ke kisaran 54–69 juta. Lihat [`outputs/forecast_2026_timeline.png`](outputs/forecast_2026_timeline.png).

### Submission

| File | Isi |
|---|---|
| [`submissions/smart_octnovdec_sev_trend.csv`](submissions/smart_octnovdec_sev_trend.csv) | **Submission final.** Severitas Okt–Des dikalibrasi mengikuti tren YoY +8,4 % yang terobservasi di H1 2025 |
| [`submissions/phase4_aug_freq104_sep122.csv`](submissions/phase4_aug_freq104_sep122.csv) | Baseline sebelum kalibrasi, hasil *leaderboard probing* |
| [`submissions/sample_submission.csv`](submissions/sample_submission.csv) | Skema format |

Kalibrasi berangkat dari temuan bahwa 5 dari 6 bulan di H1 2025 mencatat kenaikan severitas YoY (median +8,4 %), sementara model justru memprediksi penurunan di Q4 — konsekuensi dari data latih yang hanya mencakup 12 bulan penuh.

## Struktur repo

```
notebooks/pipeline_forecasting_xai.ipynb   Pipeline utama, 37 sel, output tersimpan
src/pipeline_forecasting_xai.py            Versi skrip dari notebook di atas
src/case_compliance_part1_eda.py           EDA missing value, outlier, scatter plot
src/case_compliance_part2_model.py         Model comparison, CV, ablation, forecast 2026, SHAP total claim
outputs/                                   38 artefak: PNG, model_info.json, laporan XAI
submissions/                               Submission final + baseline + sample
docs/                                      Laporan akhir (.docx), case study, catatan penyusunan
data/README.md                             Skema dataset (data-nya sendiri tidak disertakan)
```

### Peta laporan → kode

Laporan akhir bukan produk satu file — ia menggabungkan tiga eksekusi terpisah:

| Bab laporan | Dihasilkan oleh |
|---|---|
| EDA 3.1.1–3.1.5, model severity & frequency, forecast Agt–Des 2025, SHAP 3.5.1–3.5.2 | `notebooks/pipeline_forecasting_xai.ipynb` |
| 3.2 perbandingan model, 3.3 CV, 3.4 ablation, 3.5.3 SHAP total klaim, 3.8 forecast 2026 | `src/case_compliance_part2_model.py` |
| Analisis missing value & outlier, scatter plot | `src/case_compliance_part1_eda.py` |

## Menjalankan

```bash
pip install -r requirements.txt
```

Letakkan `Data_Klaim.csv` dan `Data_Polis.csv` di root repo, lalu:

```bash
python src/case_compliance_part2_model.py
```

Notebook dijalankan dari root repo juga (path data bersifat relatif terhadap direktori kerja, bukan terhadap lokasi file `.ipynb`).

## Catatan reproduksibilitas

Beberapa hal yang perlu diketahui siapa pun yang membaca repo ini bersama laporan akhirnya:

**Angka metrik di README ≠ angka di laporan.** README memakai nilai dari artefak yang benar-benar dihasilkan kode (`outputs/*.png`, output tersimpan di notebook). Laporan `.docx` mencantumkan angka yang berbeda:

| | Laporan | Artefak kode |
|---|---|---|
| LightGBM | MAPE 84,20 % · RMSE 54,5 jt · R² 0,4827 | 792,66 % · 100,8 jt · 0,4664 |
| XGBoost | 85,74 % · R² 0,4677 | 810,47 % · R² 0,4796 |
| Random Forest | 89,52 % · R² 0,4189 | 837,82 % · R² 0,4280 |
| Ablation tanpa FE | 91,45 % · R² 0,3521 | 1282,89 % · R² 0,1855 |

Selisih lain: laporan menyebut CV **4 fold**, kode mendefinisikan **3 fold** (`src/case_compliance_part2_model.py:189`) — meski nilai R² 0,48 ± 0,05 yang dikutip laporan cocok dengan hasil 3-fold tersebut. Laporan juga menyebut bootstrap 50 iterasi dengan CI 90 %; kode untuk itu tidak ada di repo. Dan Tabel 9 laporan memuat angka dari `phase4_aug_freq104_sep122.csv` (hasil probing leaderboard), bukan keluaran model di notebook sel 26.

**Bug yang diketahui — `is_reimburse` konstan nol.** Fitur dibentuk lewat `df['Reimburse/Cashless'] == 'Reimburse'`, padahal nilai kolomnya adalah `'R'` (2.722 baris) dan `'C'` (1.905 baris). Kondisi itu tidak pernah benar, sehingga fitur bernilai 0 di seluruh 4.627 baris meski terhitung sebagai salah satu dari 23 fitur. Muncul di `src/case_compliance_part2_model.py:49`, `src/pipeline_forecasting_xai.py`, dan sel 7 notebook.

**Kode tidak diperbaiki.** Repo ini arsip apa adanya dari kondisi saat kompetisi berakhir (Maret 2026) — temuan di atas dicatat, bukan ditambal, supaya isinya tetap sesuai dengan yang menghasilkan artefak.

**Eksekusi ulang belum tervalidasi.** Virtualenv aslinya kini tidak bisa dipakai: `pyexpat` pada Homebrew Python 3.14.6 tidak cocok dengan `libexpat` bawaan macOS, dan `import matplotlib` gagal sebelum kode proyek sempat berjalan. Kendala lingkungan, bukan kode. `requirements.txt` merekam versi persis yang dulu terpasang.

## Referensi

- Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
- Lundberg, S. M., et al. (2020). *From Local Explanations to Global Understanding with Explainable AI for Trees.* Nature Machine Intelligence 2(1).
- Klugman, S. A., Panjer, H. H., & Willmot, G. E. (2012). *Loss Models: From Data to Decisions* (4th ed.).

---

Repo private. Dataset milik AXA Financial Indonesia dan tidak didistribusikan.
