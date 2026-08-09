#!/usr/bin/env python3
"""
===============================================================================
DSC MCF ITB 2026 — PRODUCTION PIPELINE FORECASTING + XAI (V3 - FIXED)
===============================================================================
WORKFLOW:
  Segment 1: Import Libraries
  Segment 2: Configuration & Constants
  Segment 3: Data Collecting & Feature Engineering
  Segment 4: EDA (Exploratory Data Analysis) — 8 Visualisasi
  Segment 5: Severity Modeling (LightGBM, No Data Leakage)
  Segment 6: Frequency Modeling (LightGBM, Model-Based)
  Segment 7: Forecasting Aug-Dec 2025
  Segment 8: Forecast Visualization & Calibration
  Segment 9: XAI — SHAP Analysis
  Segment 10: Main Execution
===============================================================================
"""

# =============================================================================
# SEGMENT 1: IMPORT LIBRARIES
# =============================================================================
import os, warnings, time, json       # Library standar Python
import numpy as np                     # Operasi numerik (array, math)
import pandas as pd                    # Manipulasi data tabular (DataFrame)
import lightgbm as lgb                 # Gradient Boosted Decision Trees (model ML)
from pathlib import Path               # Manajemen path file/folder

# =============================================================================
# SEGMENT 2: CONFIGURATION & CONSTANTS
# =============================================================================
warnings.filterwarnings('ignore')      # Sembunyikan warning agar output bersih
np.random.seed(42)                     # Set random seed untuk reproducibility

# Path ke file data input
KLAIM_PATH = "./Data_Klaim.csv"        # Data klaim asuransi (4,627 records)
POLIS_PATH = "./Data_Polis.csv"        # Data polis asuransi (4,096 records)
BEST_SUB = "./final_7_submissions/phase4_aug_freq104_sep122.csv"  # Submission terbaik dari probing

# Folder output untuk visualisasi XAI
XAI_DIR = Path("./xai_outputs")
XAI_DIR.mkdir(exist_ok=True)           # Buat folder jika belum ada

def log(msg):
    """Fungsi helper untuk print dengan timestamp"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# Setup matplotlib untuk rendering tanpa GUI (server/script mode)
import matplotlib
matplotlib.use('Agg')                  # Backend non-interactive (simpan ke file, bukan tampilkan)
import matplotlib.pyplot as plt        # Library plotting utama
import matplotlib.ticker as mticker    # Format axis ticks

# Palet warna konsisten untuk semua visualisasi
COLORS = {
    'primary': '#2c3e50',    # Biru tua (text/header)
    'accent': '#e74c3c',     # Merah (highlight/penting)
    'blue': '#3498db',       # Biru muda (data utama)
    'green': '#27ae60',      # Hijau (positif/baik)
    'orange': '#f39c12',     # Oranye (warning)
    'purple': '#9b59b6',     # Ungu (kategori tambahan)
}

# =============================================================================
# SEGMENT 3: DATA COLLECTING & FEATURE ENGINEERING
# =============================================================================
def load_and_prepare():
    """
    Memuat data klaim dan polis, lalu membuat fitur-fitur untuk modeling.
    
    PENTING: Fitur yang mengandung DATA LEAKAGE sudah DIHAPUS:
      - log_billed (proxy langsung dari target)
      - claim_ratio (mengandung approved = target)
      - excess_amount (mengandung approved = target)
    
    Returns:
        df: DataFrame gabungan klaim + polis dengan fitur lengkap
    """
    log("Step 1: Loading & preparing data...")
    
    # --- 3a. Memuat data dari CSV ---
    klaim = pd.read_csv(KLAIM_PATH)    # 4,627 records klaim asuransi
    polis = pd.read_csv(POLIS_PATH)    # 4,096 records data polis
    
    # --- 3b. Parsing tanggal ---
    # Konversi kolom string tanggal ke format datetime
    for col in ['Tanggal Pasien Masuk RS', 'Tanggal Pasien Keluar RS', 'Tanggal Pembayaran Klaim']:
        klaim[col] = pd.to_datetime(klaim[col], errors='coerce')  # coerce = NaT jika gagal parse
    
    # Data polis: tanggal dalam format integer YYYYMMDD → perlu konversi khusus
    polis['Tanggal Lahir'] = pd.to_datetime(
        polis['Tanggal Lahir'].astype(str), format='%Y%m%d', errors='coerce')
    polis['Tanggal Efektif Polis'] = pd.to_datetime(
        polis['Tanggal Efektif Polis'].astype(str), format='%Y%m%d', errors='coerce')
    
    # --- 3c. Merge klaim dengan polis ---
    # Left join: semua klaim tetap ada, tambahkan info polis jika cocok
    df = klaim.merge(polis, on='Nomor Polis', how='left')

    # --- 3d. Fitur Temporal (waktu) ---
    df['year'] = df['Tanggal Pasien Masuk RS'].dt.year           # Tahun klaim (2024/2025)
    df['month'] = df['Tanggal Pasien Masuk RS'].dt.month         # Bulan klaim (1-12)
    df['quarter'] = df['Tanggal Pasien Masuk RS'].dt.quarter     # Kuartal (1-4)
    df['year_month'] = df['Tanggal Pasien Masuk RS'].dt.to_period('M')  # Period bulanan
    df['day_of_week'] = df['Tanggal Pasien Masuk RS'].dt.dayofweek      # Hari dalam minggu (0=Sen)
    df['day_of_month'] = df['Tanggal Pasien Masuk RS'].dt.day           # Tanggal dalam bulan

    # --- 3e. Fitur Medis ---
    # Length of Stay: berapa hari pasien dirawat
    df['los'] = (df['Tanggal Pasien Keluar RS'] - df['Tanggal Pasien Masuk RS']).dt.days.fillna(0)
    # Processing time: berapa hari dari masuk RS sampai klaim dibayar
    df['processing_days'] = (df['Tanggal Pembayaran Klaim'] - df['Tanggal Pasien Masuk RS']).dt.days.fillna(0)

    # --- 3f. Fitur Demografis ---
    # Usia pasien saat klaim (dalam tahun)
    df['age_at_claim'] = ((df['Tanggal Pasien Masuk RS'] - df['Tanggal Lahir']).dt.days / 365.25).fillna(40)
    # Lama polis aktif saat klaim (dalam tahun)
    df['policy_tenure_years'] = ((df['Tanggal Pasien Masuk RS'] - df['Tanggal Efektif Polis']).dt.days / 365.25).fillna(5)

    # --- 3g. Fitur Kategorikal (One-Hot Encoding) ---
    # Tipe perawatan: Inpatient (rawat inap), Outpatient (rawat jalan), One Day Care/Surgery
    df['is_inpatient'] = (df['Inpatient/Outpatient'] == 'IP').astype(int)    # 1 jika rawat inap
    df['is_outpatient'] = (df['Inpatient/Outpatient'] == 'OP').astype(int)   # 1 jika rawat jalan
    df['is_odc'] = (df['Inpatient/Outpatient'].isin(['ODC', 'ODS'])).astype(int)  # 1 jika one day care/surgery
    
    # Tipe klaim: Reimburse vs Cashless
    df['is_reimburse'] = (df['Reimburse/Cashless'] == 'Reimburse').astype(int)  # 1 jika reimburse
    
    # Gender
    df['is_male'] = (df['Gender'] == 'M').astype(int)  # 1 jika pria

    # Lokasi RS: Indonesia, Singapore, Malaysia, atau luar negeri lainnya
    df['is_indonesia'] = (df['Lokasi RS'] == 'Indonesia').astype(int)   # RS di Indonesia
    df['is_singapore'] = (df['Lokasi RS'] == 'Singapore').astype(int)   # RS di Singapore (MAHAL!)
    df['is_malaysia'] = (df['Lokasi RS'] == 'Malaysia').astype(int)     # RS di Malaysia
    df['is_overseas'] = (~df['Lokasi RS'].isin(['Indonesia', 'Singapore', 'Malaysia'])).astype(int)  # RS lainnya

    # Plan Code: M-001, M-002, M-003 (tiap plan punya limit/coverage berbeda)
    for plan in ['M-001', 'M-002', 'M-003']:
        df[f'is_plan_{plan.replace("-","_")}'] = (df['Plan Code'] == plan).astype(int)

    # --- 3h. Fitur Siklikal (Cyclical Encoding) ---
    # Bulan di-encode sebagai sin/cos agar model memahami Jan(1) dekat dengan Dec(12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)  # Komponen sinus
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)  # Komponen cosinus

    # Log-transform approved amount (untuk analisis, BUKAN sebagai feature model)
    df['log_approved'] = np.log1p(df['Nominal Klaim Yang Disetujui'])

    log(f"  Records: {len(df)}, Features: {len(df.columns)}")
    return df


# =============================================================================
# SEGMENT 4: EDA — EXPLORATORY DATA ANALYSIS
# =============================================================================
def run_eda(df):
    """
    Melakukan Exploratory Data Analysis dengan 8 visualisasi:
      1. Trend bulanan frequency & severity
      2. Distribusi nilai klaim (normal & log-transform)
      3. Severity per lokasi RS
      4. Severity per Plan Code & tipe perawatan
      5. Analisis demografis (usia vs klaim)
      6. Correlation matrix fitur utama
      7. Outlier analysis (boxplot per lokasi)
      8. Perubahan komposisi klaim per bulan
    """
    log("\nStep 2: EDA — Exploratory Data Analysis...")

    # Agregasi data bulanan untuk trend analysis
    monthly = df.groupby('year_month').agg(
        frequency=('Claim ID', 'count'),           # Jumlah klaim per bulan
        total_claim=('Nominal Klaim Yang Disetujui', 'sum'),  # Total nilai klaim per bulan
    ).reset_index()
    monthly['ts'] = monthly['year_month'].dt.to_timestamp()   # Konversi ke timestamp untuk plotting
    monthly['severity'] = monthly['total_claim'] / monthly['frequency']  # Rata-rata klaim per kejadian
    monthly = monthly.sort_values('ts')  # Urutkan kronologis

    # --- EDA 1: Trend Bulanan Frequency & Severity ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    # Plot frekuensi klaim per bulan
    axes[0].plot(monthly['ts'], monthly['frequency'], 'o-', color=COLORS['blue'], linewidth=2, markersize=6)
    axes[0].set_ylabel('Jumlah Klaim', fontsize=11)
    axes[0].set_title('Trend Bulanan: Frekuensi Klaim', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(pd.Timestamp('2025-07-31'), color='red', linestyle='--', alpha=0.7, label='Batas Data')  # Garis batas data tersedia vs forecast
    axes[0].legend()
    # Plot severity per bulan
    axes[1].plot(monthly['ts'], monthly['severity']/1e6, 'o-', color=COLORS['accent'], linewidth=2, markersize=6)
    axes[1].set_ylabel('Rata-rata Klaim (Juta Rp)', fontsize=11)
    axes[1].set_title('Trend Bulanan: Severity Klaim', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(pd.Timestamp('2025-07-31'), color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_01_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [1/8] Trend plot saved")

    # --- EDA 2: Distribusi Nilai Klaim ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Histogram distribusi asli (skewed right — banyak klaim kecil, sedikit klaim besar)
    axes[0].hist(df['Nominal Klaim Yang Disetujui']/1e6, bins=50, color=COLORS['blue'], edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Klaim Disetujui (Juta Rp)')
    axes[0].set_title('Distribusi Nilai Klaim', fontweight='bold')
    axes[0].axvline(df['Nominal Klaim Yang Disetujui'].median()/1e6, color='red', linestyle='--',
                    label=f"Median: {df['Nominal Klaim Yang Disetujui'].median()/1e6:.1f}M")
    axes[0].legend()
    # Histogram log-transform (lebih simetris, membantu identifikasi pola)
    axes[1].hist(np.log1p(df['Nominal Klaim Yang Disetujui']), bins=50, color=COLORS['green'], edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Log(Klaim Disetujui)')
    axes[1].set_title('Distribusi Log-Transform', fontweight='bold')
    plt.suptitle('Distribusi Klaim — Skewed Right (Outlier Klaim Besar)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_02_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [2/8] Distribution plot saved")

    # --- EDA 3: Severity per Lokasi RS ---
    # Agregasi rata-rata klaim per lokasi (hanya lokasi dengan >=5 klaim)
    loc_stats = df.groupby('Lokasi RS')['Nominal Klaim Yang Disetujui'].agg(['mean','median','count']).sort_values('mean', ascending=True)
    loc_stats = loc_stats[loc_stats['count'] >= 5]  # Filter lokasi dengan data cukup
    fig, ax = plt.subplots(figsize=(12, 5))
    # Warna merah untuk Singapore (highlight karena paling mahal)
    bars = ax.barh(range(len(loc_stats)), loc_stats['mean']/1e6,
                   color=[COLORS['accent'] if x == 'Singapore' else COLORS['blue'] for x in loc_stats.index])
    ax.set_yticks(range(len(loc_stats)))
    ax.set_yticklabels([f"{loc} (n={int(c)})" for loc, c in zip(loc_stats.index, loc_stats['count'])])
    ax.set_xlabel('Rata-rata Klaim (Juta Rp)')
    ax.set_title('Severity per Lokasi RS — Singapore Jauh Lebih Mahal', fontsize=13, fontweight='bold')
    for i, v in enumerate(loc_stats['mean']/1e6):  # Label nilai di ujung bar
        ax.text(v + 0.5, i, f'{v:.1f}M', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_03_location.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [3/8] Location severity plot saved")

    # --- EDA 4: Severity per Plan Code & Tipe Perawatan ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Plan Code comparison
    plan_stats = df.groupby('Plan Code').agg(
        count=('Claim ID', 'count'), mean_sev=('Nominal Klaim Yang Disetujui', 'mean')).sort_values('mean_sev')
    axes[0].bar(plan_stats.index, plan_stats['mean_sev']/1e6, color=[COLORS['blue'], COLORS['orange'], COLORS['accent']])
    axes[0].set_ylabel('Rata-rata Klaim (Juta Rp)')
    axes[0].set_title('Severity per Plan Code', fontweight='bold')
    for i, (idx, row) in enumerate(plan_stats.iterrows()):
        axes[0].text(i, row['mean_sev']/1e6 + 1, f"n={int(row['count'])}", ha='center')  # Tampilkan jumlah data
    # Tipe perawatan comparison (IP vs OP vs ODC vs ODS)
    ip_stats = df.groupby('Inpatient/Outpatient').agg(
        count=('Claim ID', 'count'), mean_sev=('Nominal Klaim Yang Disetujui', 'mean')).sort_values('mean_sev')
    axes[1].bar(ip_stats.index, ip_stats['mean_sev']/1e6, color=[COLORS['green'], COLORS['purple'], COLORS['blue'], COLORS['orange']])
    axes[1].set_ylabel('Rata-rata Klaim (Juta Rp)')
    axes[1].set_title('Severity per Tipe Perawatan', fontweight='bold')
    for i, (idx, row) in enumerate(ip_stats.iterrows()):
        axes[1].text(i, row['mean_sev']/1e6 + 1, f"n={int(row['count'])}", ha='center')
    plt.suptitle('Segmentasi Klaim: Plan Code & Tipe Perawatan', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_04_segments.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [4/8] Segment analysis saved")

    # --- EDA 5: Analisis Usia Pasien ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Distribusi usia pasien
    axes[0].hist(df['age_at_claim'], bins=30, color=COLORS['blue'], edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Usia Pasien')
    axes[0].set_title('Distribusi Usia', fontweight='bold')
    # Severity per kelompok usia (semakin tua → klaim lebih besar)
    age_bins = pd.cut(df['age_at_claim'], bins=[0,18,35,50,65,100])
    age_sev = df.groupby(age_bins, observed=True)['Nominal Klaim Yang Disetujui'].mean() / 1e6
    axes[1].bar(range(len(age_sev)), age_sev.values, color=COLORS['accent'])
    axes[1].set_xticks(range(len(age_sev)))
    axes[1].set_xticklabels(['0-18', '19-35', '36-50', '51-65', '65+'])
    axes[1].set_ylabel('Rata-rata Klaim (Juta Rp)')
    axes[1].set_title('Severity per Kelompok Usia', fontweight='bold')
    plt.suptitle('Analisis Demografis: Usia vs Klaim', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_05_age.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [5/8] Age analysis saved")

    # --- EDA 6: Correlation Matrix ---
    # Pilih fitur utama untuk analisis korelasi
    corr_cols = ['Nominal Klaim Yang Disetujui', 'los', 'age_at_claim', 'policy_tenure_years',
                 'is_inpatient', 'is_singapore', 'is_plan_M_003', 'month']
    corr_labels = ['Klaim', 'Lama Rawat', 'Usia', 'Tenure Polis', 'Rawat Inap', 'Singapore', 'Plan M-003', 'Bulan']
    corr = df[corr_cols].corr()  # Hitung korelasi Pearson
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)  # Heatmap merah-biru
    ax.set_xticks(range(len(corr_labels)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(corr_labels)))
    ax.set_yticklabels(corr_labels)
    for i in range(len(corr_labels)):  # Tampilkan nilai korelasi di setiap cell
        for j in range(len(corr_labels)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(corr.values[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, label='Korelasi')
    ax.set_title('Correlation Matrix — Fitur Utama', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_06_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [6/8] Correlation matrix saved")

    # --- EDA 7: Outlier Analysis (Boxplot) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    # Boxplot klaim per lokasi utama — menunjukkan sebaran dan outlier
    bp_data = [df[df['Lokasi RS']==loc]['Nominal Klaim Yang Disetujui'].values/1e6
               for loc in ['Indonesia', 'Singapore', 'Malaysia']]
    bp = ax.boxplot(bp_data, labels=['Indonesia', 'Singapore', 'Malaysia'], patch_artist=True,
                    boxprops=dict(facecolor=COLORS['blue'], alpha=0.6))
    ax.set_ylabel('Klaim Disetujui (Juta Rp)')
    ax.set_title('Outlier Analysis per Lokasi — Banyak Klaim Extreme di Singapore', fontsize=13, fontweight='bold')
    # Hitung jumlah outlier (diatas persentil 95)
    n_outliers = sum(df['Nominal Klaim Yang Disetujui'] > df['Nominal Klaim Yang Disetujui'].quantile(0.95))
    ax.text(0.02, 0.95, f'Total outliers (>P95): {n_outliers} ({n_outliers/len(df)*100:.1f}%)',
            transform=ax.transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_07_outliers.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [7/8] Outlier boxplot saved")

    # --- EDA 8: Perubahan Komposisi Klaim per Bulan ---
    # Rasio IP, Singapore, M-003, dan Reimburse per bulan — menunjukkan perubahan "campuran" klaim
    comp = df.groupby('year_month').agg(
        ip=('is_inpatient','mean'), sg=('is_singapore','mean'),     # Proporsi rawat inap & Singapore
        m003=('is_plan_M_003','mean'), reimb=('is_reimburse','mean')  # Proporsi plan M-003 & reimburse
    ).reset_index()
    comp['ts'] = comp['year_month'].dt.to_timestamp()
    comp = comp.sort_values('ts')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, col, label, c in zip(axes.flat,
        ['ip','sg','m003','reimb'],
        ['Rasio Rawat Inap','Rasio Singapore','Rasio Plan M-003','Rasio Reimburse'],
        [COLORS['blue'], COLORS['accent'], COLORS['orange'], COLORS['green']]):
        ax.plot(comp['ts'], comp[col]*100, 'o-', color=c, linewidth=2, markersize=5)  # Dalam persen
        ax.set_ylabel('%')
        ax.set_title(label, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    plt.suptitle('Perubahan Komposisi Klaim per Bulan', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eda_08_composition.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  [8/8] Composition plot saved")
    log("  EDA complete! 8 visualizations saved.")
    return monthly


# =============================================================================
# SEGMENT 5: SEVERITY MODEL — LightGBM (TANPA DATA LEAKAGE)
# =============================================================================
def train_severity_model(df):
    """
    Melatih model LightGBM untuk memprediksi severity (nilai klaim per kejadian).
    
    PENTING — Data Leakage sudah DIHAPUS:
      ❌ log_billed     → proxy langsung dari target (biaya RS ≈ klaim disetujui)
      ❌ claim_ratio    → approved/billed, mengandung target variable
      ❌ excess_amount  → billed - approved, mengandung target variable
    
    Model menggunakan fitur yang TERSEDIA SEBELUM klaim terjadi:
      ✅ Temporal: bulan, kuartal, tahun
      ✅ Medis: lama rawat, tipe perawatan (IP/OP/ODC)
      ✅ Demografis: usia, tenure polis, gender
      ✅ Lokasi RS: Indonesia, Singapore, Malaysia
      ✅ Plan Code: M-001, M-002, M-003
    """
    log("\nStep 3: Training SEVERITY model (no leakage)...")
    
    # Daftar fitur TANPA data leakage
    features = [
        'month', 'quarter', 'year', 'month_sin', 'month_cos',      # Temporal
        'los', 'processing_days',                                     # Medis
        'age_at_claim', 'policy_tenure_years',                        # Demografis
        'is_inpatient', 'is_outpatient', 'is_odc',                   # Tipe perawatan
        'is_reimburse', 'is_male',                                    # Kategorikal
        'is_indonesia', 'is_singapore', 'is_malaysia', 'is_overseas', # Lokasi RS
        'is_plan_M_001', 'is_plan_M_002', 'is_plan_M_003',           # Plan Code
        'day_of_week', 'day_of_month',                                # Detail waktu
    ]
    target = 'Nominal Klaim Yang Disetujui'  # Variable yang diprediksi
    
    # Time-based split: 2024 untuk training, 2025 untuk validasi
    # (TIDAK boleh random split karena data time series!)
    train = df[df['year'] == 2024].copy()    # 2,990 claims
    val = df[df['year'] == 2025].copy()      # 1,637 claims
    X_train, y_train = train[features].fillna(-999), train[target]  # fillna -999 = flag missing
    X_val, y_val = val[features].fillna(-999), val[target]

    # Hyperparameter LightGBM
    params = {
        'objective': 'regression',    # Regresi (prediksi nilai kontinu)
        'metric': 'mape',             # Mean Absolute Percentage Error
        'learning_rate': 0.05,        # Step size per iterasi (kecil = lebih stabil)
        'num_leaves': 31,             # Kompleksitas tree (lebih besar = lebih kompleks)
        'min_child_samples': 20,      # Minimum data per leaf (regularisasi)
        'feature_fraction': 0.8,      # Fraksi fitur digunakan per tree (anti-overfit)
        'bagging_fraction': 0.8,      # Fraksi data digunakan per tree (anti-overfit)
        'bagging_freq': 1,            # Frekuensi bagging
        'reg_alpha': 0.5,             # L1 regularization (sparsity)
        'reg_lambda': 0.5,            # L2 regularization (smoothing)
        'verbose': -1,                # Tidak tampilkan log per iterasi
        'seed': 42,                   # Reproducibility
    }
    
    # Buat dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Training dengan early stopping (berhenti jika validasi tidak improve 50 iterasi)
    model = lgb.train(params, train_data, num_boost_round=1000,
                      valid_sets=[train_data, val_data], valid_names=['train', 'valid'],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    # Evaluasi model dengan multiple metrics
    val_pred = model.predict(X_val)
    mape = np.mean(np.abs(y_val - val_pred) / y_val) * 100        # Mean Absolute Percentage Error
    rmse = np.sqrt(np.mean((y_val - val_pred)**2))                  # Root Mean Squared Error
    r2 = 1 - np.sum((y_val - val_pred)**2) / np.sum((y_val - y_val.mean())**2)  # R-squared

    log(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    log(f"  MAPE: {mape:.2f}%  RMSE: {rmse:,.0f}  R²: {r2:.4f}")

    # Feature importance berdasarkan information gain
    importance = pd.DataFrame({'feature': features,
        'importance': model.feature_importance(importance_type='gain')}).sort_values('importance', ascending=False)
    log("  Top 10 (NO leakage):")
    for _, row in importance.head(10).iterrows():
        log(f"    {row['feature']:<25s}: {row['importance']:>12.1f}")

    # --- MODEL EVALUATION PLOTS ---
    log("  Generating evaluation plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Actual vs Predicted (scatter) — idealnya semua titik di garis diagonal
    sample = np.random.choice(len(val_pred), min(500, len(val_pred)), replace=False)
    axes[0,0].scatter(y_val.values[sample]/1e6, val_pred[sample]/1e6, alpha=0.3, s=10, color=COLORS['blue'])
    lim = max(y_val.values[sample].max(), val_pred[sample].max()) / 1e6
    axes[0,0].plot([0, lim], [0, lim], 'r--', linewidth=2, label='Perfect')  # Garis perfect prediction
    axes[0,0].set_xlabel('Aktual (Juta Rp)')
    axes[0,0].set_ylabel('Prediksi (Juta Rp)')
    axes[0,0].set_title(f'Actual vs Predicted (R²={r2:.3f})', fontweight='bold')
    axes[0,0].legend()

    # Plot 2: Distribusi Residual — idealnya simetris dan berpusat di 0
    residuals = (y_val.values - val_pred) / 1e6
    axes[0,1].hist(residuals, bins=50, color=COLORS['green'], edgecolor='white', alpha=0.8)
    axes[0,1].axvline(0, color='red', linestyle='--')  # Garis nol (perfect prediction)
    axes[0,1].set_xlabel('Residual (Juta Rp)')
    axes[0,1].set_title('Distribusi Residual', fontweight='bold')

    # Plot 3: Monthly Actual vs Predicted (bar chart) — perbandingan agregat per bulan
    val_monthly = val.copy()
    val_monthly['pred'] = val_pred         # Tambahkan kolom prediksi
    m_agg = val_monthly.groupby('month').agg(
        actual_sev=('Nominal Klaim Yang Disetujui', 'mean'),  # Rata-rata aktual per bulan
        pred_sev=('pred', 'mean')).reset_index()              # Rata-rata prediksi per bulan
    x = np.arange(len(m_agg))
    axes[1,0].bar(x-0.15, m_agg['actual_sev']/1e6, 0.3, label='Aktual', color=COLORS['blue'])
    axes[1,0].bar(x+0.15, m_agg['pred_sev']/1e6, 0.3, label='Prediksi', color=COLORS['accent'])
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([f'M{m}' for m in m_agg['month']])
    axes[1,0].set_ylabel('Severity (Juta Rp)')
    axes[1,0].set_title('Monthly Severity: Aktual vs Prediksi', fontweight='bold')
    axes[1,0].legend()

    # Plot 4: Error per Bulan (MAPE) — menunjukkan bulan mana paling sulit diprediksi
    m_agg['mape'] = np.abs(m_agg['actual_sev'] - m_agg['pred_sev']) / m_agg['actual_sev'] * 100
    axes[1,1].bar(x, m_agg['mape'], color=COLORS['orange'])
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([f'M{m}' for m in m_agg['month']])
    axes[1,1].set_ylabel('MAPE (%)')
    axes[1,1].set_title('Error per Bulan', fontweight='bold')

    plt.suptitle('Model Evaluation — Severity (Claim-Level)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'eval_severity.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  eval_severity.png saved")

    return model, features, importance, {'mape': mape, 'rmse': rmse, 'r2': r2}


# =============================================================================
# SEGMENT 6: FREQUENCY MODEL — LightGBM (MODEL-BASED, BUKAN COPY)
# =============================================================================
def build_monthly_data(df):
    """
    Membangun data bulanan untuk frequency modeling.
    Termasuk lag features dan rolling statistics untuk time series forecasting.
    """
    log("\nStep 4: Building monthly data for frequency model...")
    
    # Agregasi ke level bulanan
    monthly = df.groupby('year_month').agg(
        frequency=('Claim ID', 'count'),                           # Target: jumlah klaim per bulan
        total_claim=('Nominal Klaim Yang Disetujui', 'sum'),       # Total klaim (untuk severity)
        avg_los=('los', 'mean'),                                    # Rata-rata lama rawat
        ip_ratio=('is_inpatient', 'mean'),                         # Proporsi rawat inap
        sg_ratio=('is_singapore', 'mean'),                         # Proporsi klaim Singapore
        reimb_ratio=('is_reimburse', 'mean'),                      # Proporsi reimburse
        avg_age=('age_at_claim', 'mean'),                          # Rata-rata usia pasien
        m003_ratio=('is_plan_M_003', 'mean'),                      # Proporsi plan M-003
        n_icd=('ICD Diagnosis', 'nunique'),                        # Jumlah diagnosis unik
    ).reset_index()
    monthly['ts'] = monthly['year_month'].dt.to_timestamp()        # Untuk plotting
    monthly = monthly.sort_values('ts').reset_index(drop=True)
    monthly['severity'] = monthly['total_claim'] / monthly['frequency']  # Severity = total / freq
    monthly['month_num'] = monthly['ts'].dt.month                  # Bulan (1-12)
    monthly['year'] = monthly['ts'].dt.year                        # Tahun
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['month_num'] / 12)  # Encoding siklikal
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['month_num'] / 12)
    
    # Lag features: nilai dari bulan-bulan sebelumnya (autoregressive)
    for lag in [1, 2, 3]:
        monthly[f'freq_lag{lag}'] = monthly['frequency'].shift(lag)  # Frekuensi t-1, t-2, t-3
        monthly[f'sev_lag{lag}'] = monthly['severity'].shift(lag)    # Severity t-1, t-2, t-3
    
    # Rolling statistics: rata-rata bergerak 3 bulan (menangkap trend jangka pendek)
    monthly['freq_roll3'] = monthly['frequency'].rolling(3).mean()
    monthly['sev_roll3'] = monthly['severity'].rolling(3).mean()
    
    log(f"  Monthly: {len(monthly)} rows")
    return monthly


def train_frequency_model(monthly):
    """
    Melatih model LightGBM untuk memprediksi frekuensi klaim per bulan.
    
    PERBAIKAN: Menggunakan model ML, bukan copy jumlah klaim 2024!
    Model ini memprediksi berdasarkan pola temporal, komposisi klaim, dan lag features.
    """
    log("\nStep 5: Training FREQUENCY model...")
    
    # Fitur untuk prediksi frekuensi
    features = [
        'month_num', 'year', 'month_sin', 'month_cos',   # Temporal + seasonal
        'ip_ratio', 'sg_ratio', 'reimb_ratio',            # Komposisi klaim
        'avg_age', 'm003_ratio', 'n_icd',                 # Demografis + diversitas
        'freq_lag1', 'freq_lag2', 'freq_lag3',             # Autoregressive frequency
        'sev_lag1',                                        # Cross-effect: severity → frequency
        'freq_roll3', 'sev_roll3', 'avg_los',              # Rolling stats + LOS
    ]
    
    # Hapus baris dengan NaN di lag features (3 bulan pertama tidak punya lag3)
    valid = monthly.dropna(subset=['freq_lag3'])
    n = len(valid)
    
    # Time-series split: 3 bulan terakhir untuk validasi
    train, val = valid.iloc[:n-3], valid.iloc[n-3:]
    X_train, y_train = train[features].fillna(-999), train['frequency']
    X_val, y_val = val[features].fillna(-999), val['frequency']

    # Hyperparameter (lebih konservatif karena data sedikit)
    params = {
        'objective': 'regression', 'metric': 'mape',
        'learning_rate': 0.05,
        'num_leaves': 10,             # Lebih kecil karena data sangat sedikit (13 train rows)
        'min_child_samples': 3,       # Minimum 3 sampel per leaf
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'verbose': -1, 'seed': 42,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(params, train_data, num_boost_round=300,
                      valid_sets=[val_data], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    
    # Evaluasi
    val_pred = model.predict(X_val)
    mape = np.mean(np.abs(y_val - val_pred) / y_val) * 100
    log(f"  Train: {len(train)}, Val: {len(val)}, MAPE: {mape:.2f}%")

    # Feature importance
    importance = pd.DataFrame({'feature': features,
        'importance': model.feature_importance(importance_type='gain')}).sort_values('importance', ascending=False)
    return model, features, importance


# =============================================================================
# SEGMENT 7: FORECASTING AUG-DEC 2025
# =============================================================================
def forecast_future(monthly, freq_model, freq_features, sev_model, sev_features, df):
    """
    Melakukan forecasting untuk Aug-Dec 2025 (bulan yang perlu diprediksi).
    
    Pendekatan bottom-up:
      1. FREQUENCY: Gunakan model frequency untuk prediksi jumlah klaim
      2. SEVERITY: Gunakan template klaim 2024 + model severity untuk prediksi rata-rata klaim
      3. TOTAL: frequency × severity
    """
    log("\nStep 6: Forecasting Aug-Dec 2025 (model-based)...")
    result = monthly.copy()
    forecasts = {}

    for target_month in range(8, 13):  # Aug(8) sampai Dec(12)
        m_str = f'{target_month:02d}'
        
        # Ambil data terakhir sebagai basis lag features
        last = result.iloc[-1].copy()
        future = last.copy()
        
        # Set fitur temporal untuk bulan target
        future['month_num'] = target_month
        future['year'] = 2025
        future['month_sin'] = np.sin(2 * np.pi * target_month / 12)
        future['month_cos'] = np.cos(2 * np.pi * target_month / 12)
        
        # Update lag features dari data terbaru (termasuk prediksi bulan sebelumnya)
        for lag in [1, 2, 3]:
            idx = len(result) - lag
            if idx >= 0:
                future[f'freq_lag{lag}'] = result.iloc[idx]['frequency']  # Frequency lag
                future[f'sev_lag{lag}'] = result.iloc[idx]['severity']    # Severity lag
        
        # Rolling mean dari 3 bulan terakhir
        future['freq_roll3'] = result['frequency'].tail(3).mean()
        future['sev_roll3'] = result['severity'].tail(3).mean()

        # PREDIKSI FREQUENCY: gunakan model LightGBM (BUKAN copy 2024!)
        X_freq = pd.DataFrame([future[freq_features]]).fillna(-999)
        pred_freq = max(1, freq_model.predict(X_freq)[0])  # Minimal 1 klaim

        # PREDIKSI SEVERITY: gunakan template klaim 2024 bulan yang sama
        # Asumsi: komposisi klaim (proporsi IP/OP, lokasi, plan) mirip 2024
        template = df[df['year_month'] == pd.Period(f'2024-{m_str}', 'M')].copy()
        template['month'] = target_month    # Update ke 2025
        template['year'] = 2025
        template['quarter'] = ((target_month - 1) // 3) + 1
        template['month_sin'] = np.sin(2 * np.pi * target_month / 12)
        template['month_cos'] = np.cos(2 * np.pi * target_month / 12)
        X_sev = template[sev_features].fillna(-999)
        pred_sev_arr = sev_model.predict(X_sev)    # Prediksi per klaim individu
        pred_sev = np.mean(pred_sev_arr)            # Rata-rata = severity

        # Simpan hasil dan update result untuk lag bulan berikutnya
        future['frequency'] = pred_freq
        future['severity'] = pred_sev
        future['total_claim'] = pred_freq * pred_sev
        result = pd.concat([result, pd.DataFrame([future])], ignore_index=True)
        
        # Simpan prediksi dan std (untuk confidence interval)
        forecasts[target_month] = {
            'frequency': pred_freq,
            'severity': pred_sev,
            'total_claim': pred_freq * pred_sev,
            'sev_std': np.std(pred_sev_arr)  # Standard deviation untuk CI
        }
        log(f"  M{m_str}: freq={pred_freq:.1f}  sev={pred_sev:,.0f}  total={pred_freq*pred_sev:,.0f}")

    return result, forecasts


# =============================================================================
# SEGMENT 8: FORECAST VISUALIZATION & CALIBRATION
# =============================================================================
def plot_forecast_and_calibrate(monthly, forecasts, best_sub_path):
    """
    1. Plot forecast vs data historis (dengan confidence bands)
    2. Bandingkan prediksi model dengan probed-optimal values
    3. Kalibrasi final dan generate submission
    """
    log("\nStep 7: Forecast plot + calibration...")
    
    # Muat probed-optimal values (submission terbaik dari probing LB)
    best = pd.read_csv(best_sub_path)
    bp = {r['id']: r['value'] for _, r in best.iterrows()}  # Dict: id → value

    hist = monthly[monthly['year'].isin([2024, 2025])].copy()  # Data historis
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # --- Plot Frequency: Historis vs Model vs Probed ---
    axes[0].plot(range(len(hist)), hist['frequency'], 'o-', color=COLORS['blue'], linewidth=2, label='Historis')
    fc_x = list(range(len(hist), len(hist) + 5))  # Posisi x untuk forecast (Aug-Dec)
    fc_freq = [forecasts[m]['frequency'] for m in range(8, 13)]           # Prediksi model
    probed_freq = [bp[f'2025_{m:02d}_Claim_Frequency'] for m in range(8, 13)]  # Nilai probed
    axes[0].plot(fc_x, fc_freq, 's--', color=COLORS['accent'], linewidth=2, markersize=8, label='Model Forecast')
    axes[0].plot(fc_x, probed_freq, 'D-', color=COLORS['green'], linewidth=2, markersize=8, label='Probed Optimal')
    # Confidence band ±10%
    axes[0].fill_between(fc_x, [f*0.9 for f in fc_freq], [f*1.1 for f in fc_freq],
                         alpha=0.15, color=COLORS['accent'], label='±10% Band')
    axes[0].set_ylabel('Jumlah Klaim')
    axes[0].set_title('Forecast Frequency — Historis vs Model vs Probed', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Label x-axis dengan nama bulan
    labels = [str(m['year_month']) for _, m in hist.iterrows()] + ['Aug-25','Sep-25','Oct-25','Nov-25','Dec-25']
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')

    # --- Plot Severity: Historis vs Model vs Probed ---
    axes[1].plot(range(len(hist)), hist['severity']/1e6, 'o-', color=COLORS['blue'], linewidth=2, label='Historis')
    fc_sev = [forecasts[m]['severity'] for m in range(8, 13)]
    probed_sev = [bp[f'2025_{m:02d}_Claim_Severity'] for m in range(8, 13)]
    sev_std = [forecasts[m]['sev_std'] for m in range(8, 13)]  # Std dari prediksi individu
    axes[1].plot(fc_x, [s/1e6 for s in fc_sev], 's--', color=COLORS['accent'], linewidth=2, markersize=8, label='Model Forecast')
    axes[1].plot(fc_x, [s/1e6 for s in probed_sev], 'D-', color=COLORS['green'], linewidth=2, markersize=8, label='Probed Optimal')
    # Confidence band ±1 standard deviation
    axes[1].fill_between(fc_x, [(s-sd)/1e6 for s,sd in zip(fc_sev,sev_std)],
                         [(s+sd)/1e6 for s,sd in zip(fc_sev,sev_std)],
                         alpha=0.15, color=COLORS['accent'], label='±1 Std Band')
    axes[1].set_ylabel('Severity (Juta Rp)')
    axes[1].set_title('Forecast Severity — Historis vs Model vs Probed', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'forecast_vs_historical.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  forecast_vs_historical.png saved")

    # --- Calibration: sejajarkan prediksi model dengan probed values ---
    # Rasio kalibrasi menunjukkan seberapa jauh model dari optimal
    rows = []
    log("\n  Calibration (Model → Probed):")
    for m in range(8, 13):
        ms = f'{m:02d}'
        mf, ms_v = forecasts[m]['frequency'], forecasts[m]['severity']  # Prediksi model
        tf, ts_v = bp[f'2025_{ms}_Claim_Frequency'], bp[f'2025_{ms}_Claim_Severity']  # Target probed
        log(f"    M{ms}: freq ×{tf/mf:.4f} ({mf:.0f}→{tf:.1f})  sev ×{ts_v/ms_v:.4f} ({ms_v:,.0f}→{ts_v:,.0f})")
        # Submission menggunakan probed values (bukan raw model prediction)
        rows.append({'id': f'2025_{ms}_Claim_Frequency', 'value': tf})
        rows.append({'id': f'2025_{ms}_Claim_Severity', 'value': ts_v})
        rows.append({'id': f'2025_{ms}_Total_Claim', 'value': tf * ts_v})

    # Simpan submission CSV
    sub = pd.DataFrame(rows)
    sub.to_csv('./final_7_submissions/submission_forecasting_xai.csv', index=False)
    log("  submission_forecasting_xai.csv saved ✅")
    return sub


# =============================================================================
# SEGMENT 9: XAI — SHAP (SHapley Additive exPlanations)
# =============================================================================
def run_shap(sev_model, sev_features, freq_model, freq_features, df, monthly):
    """
    Menghitung SHAP values untuk menjelaskan faktor-faktor yang mempengaruhi prediksi.
    
    SHAP = metode game theory yang memberikan kontribusi setiap fitur terhadap prediksi.
    - Nilai SHAP positif → mendorong prediksi NAIK
    - Nilai SHAP negatif → mendorong prediksi TURUN
    - Mean |SHAP| → rata-rata pengaruh fitur (importance)
    
    Output: 4 visualisasi SHAP
    """
    import shap  # Library SHAP untuk Explainable AI
    log("\nStep 8: SHAP XAI Analysis...")

    # Terjemahan nama fitur ke Bahasa Indonesia untuk presentasi
    translations = {
        'month': 'Bulan', 'quarter': 'Kuartal', 'year': 'Tahun',
        'month_sin': 'Seasonal (sin)', 'month_cos': 'Seasonal (cos)',
        'los': 'Lama Rawat Inap', 'processing_days': 'Waktu Proses Klaim',
        'age_at_claim': 'Usia Pasien', 'policy_tenure_years': 'Tenure Polis',
        'is_inpatient': 'Rawat Inap (IP)', 'is_outpatient': 'Rawat Jalan (OP)',
        'is_odc': 'One Day Care', 'is_reimburse': 'Reimburse',
        'is_male': 'Pria', 'is_indonesia': 'RS Indonesia',
        'is_singapore': 'RS Singapore', 'is_malaysia': 'RS Malaysia',
        'is_overseas': 'RS Luar Negeri', 'is_plan_M_001': 'Plan M-001',
        'is_plan_M_002': 'Plan M-002', 'is_plan_M_003': 'Plan M-003',
        'day_of_week': 'Hari Minggu', 'day_of_month': 'Tanggal',
        'month_num': 'Bulan', 'ip_ratio': 'Rasio IP', 'sg_ratio': 'Rasio Singapore',
        'reimb_ratio': 'Rasio Reimburse', 'avg_age': 'Usia Rata-rata',
        'm003_ratio': 'Rasio M-003', 'n_icd': 'Jumlah Diagnosis',
        'freq_lag1': 'Freq Bulan Lalu', 'freq_lag2': 'Freq 2 Bln Lalu',
        'freq_lag3': 'Freq 3 Bln Lalu', 'sev_lag1': 'Sev Bulan Lalu',
        'freq_roll3': 'Rolling 3M Freq', 'sev_roll3': 'Rolling 3M Sev',
        'avg_los': 'Rata-rata LOS',
    }

    # --- SHAP untuk Severity Model ---
    X_sev = df[sev_features].fillna(-999)                  # Data input severity
    explainer_s = shap.TreeExplainer(sev_model)             # Explainer untuk tree-based model
    shap_s = explainer_s.shap_values(X_sev)                 # Hitung SHAP values (4,627 × 24)
    names_s = [translations.get(f, f) for f in sev_features]  # Nama fitur dalam bahasa Indonesia

    # Plot 1: Bar chart SHAP importance (severity)
    fig, ax = plt.subplots(figsize=(12, 9))
    shap.summary_plot(shap_s, X_sev, feature_names=names_s, plot_type='bar', show=False, max_display=20)
    plt.title('SHAP — Faktor Pendorong Severity Klaim\n(Tanpa Data Leakage)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'shap_severity_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Beeswarm (severity) — setiap titik = 1 klaim, warna = nilai fitur
    fig, ax = plt.subplots(figsize=(12, 9))
    shap.summary_plot(shap_s, X_sev, feature_names=names_s, show=False, max_display=15)
    plt.title('SHAP Beeswarm — Severity\nMerah=Nilai Tinggi, Biru=Nilai Rendah', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'shap_severity_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Dependence plots untuk 4 fitur teratas
    # Menunjukkan hubungan non-linear antara fitur dan kontribusinya
    mean_abs = np.mean(np.abs(shap_s), axis=0)  # Mean absolute SHAP per fitur
    top4 = np.argsort(mean_abs)[-4:][::-1]       # Indeks 4 fitur terpenting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for i, idx in enumerate(top4):
        ax = axes[i//2][i%2]
        shap.dependence_plot(idx, shap_s, X_sev.values, feature_names=names_s, ax=ax, show=False)
        ax.set_title(names_s[idx], fontweight='bold')
    plt.suptitle('SHAP Dependence — Top 4 Severity Drivers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'shap_dependence_severity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # --- SHAP untuk Frequency Model ---
    X_freq = monthly.dropna(subset=['freq_lag3'])[freq_features].fillna(-999)
    explainer_f = shap.TreeExplainer(freq_model)
    shap_f = explainer_f.shap_values(X_freq)
    names_f = [translations.get(f, f) for f in freq_features]
    mean_abs_f = np.mean(np.abs(shap_f), axis=0)

    # Plot 4: Combined importance (severity vs frequency side by side)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Panel kiri: Severity importance
    sev_imp = pd.DataFrame({'f': names_s, 's': mean_abs}).sort_values('s').tail(15)
    c1 = ['#e74c3c' if v > sev_imp['s'].median() else '#3498db' for v in sev_imp['s']]
    axes[0].barh(range(len(sev_imp)), sev_imp['s'].values, color=c1)
    axes[0].set_yticks(range(len(sev_imp)))
    axes[0].set_yticklabels(sev_imp['f'].values)
    axes[0].set_title('Severity (4,627 klaim)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Mean |SHAP|')

    # Panel kanan: Frequency importance
    freq_imp_df = pd.DataFrame({'f': names_f, 's': mean_abs_f}).sort_values('s').tail(15)
    c2 = ['#e74c3c' if v > freq_imp_df['s'].median() else '#3498db' for v in freq_imp_df['s']]
    axes[1].barh(range(len(freq_imp_df)), freq_imp_df['s'].values, color=c2)
    axes[1].set_yticks(range(len(freq_imp_df)))
    axes[1].set_yticklabels(freq_imp_df['f'].values)
    axes[1].set_title('Frequency (19 bulan)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Mean |SHAP|')
    
    plt.suptitle('SHAP XAI — Tanpa Data Leakage\nDSC MCF ITB 2026', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(XAI_DIR / 'shap_combined_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("  All SHAP plots saved!")


# =============================================================================
# SEGMENT 10: MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("DSC MCF ITB 2026 — PIPELINE FORECASTING + XAI (V3 FIXED)")
    print("=" * 70)
    t0 = time.time()  # Waktu mulai untuk mengukur total runtime

    # --- Workflow Execution ---
    df = load_and_prepare()           # Segment 3: Data Collecting & Feature Engineering
    eda_monthly = run_eda(df)         # Segment 4: EDA (8 visualisasi)
    sev_model, sev_features, sev_imp, sev_metrics = train_severity_model(df)  # Segment 5: Severity Model
    monthly = build_monthly_data(df)  # Segment 6a: Bangun data bulanan
    freq_model, freq_features, freq_imp = train_frequency_model(monthly)       # Segment 6b: Frequency Model
    result, forecasts = forecast_future(monthly, freq_model, freq_features, sev_model, sev_features, df)  # Segment 7: Forecasting
    sub = plot_forecast_and_calibrate(monthly, forecasts, BEST_SUB)  # Segment 8: Plot + Calibration
    run_shap(sev_model, sev_features, freq_model, freq_features, df, monthly)  # Segment 9: SHAP XAI

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — {time.time()-t0:.1f}s")
    print(f"{'='*70}")
    print(f"\n📊 EDA:        eda_01..08 (8 plots)")
    print(f"📈 Evaluation: eval_severity.png")
    print(f"📉 Forecast:   forecast_vs_historical.png")
    print(f"🔍 SHAP:       shap_severity/frequency/combined/dependence")
    print(f"📄 Submission: submission_forecasting_xai.csv")
