#!/usr/bin/env python3
"""
CASE DOCUMENT COMPLIANCE SCRIPT — PART 1
=========================================
Sections to add to notebook (EDA Component - 20%):
  A. Missing Values & Outlier (setelah Cell 7)
  B. Scatter Plot Analysis (setelah Cell 16)
  C. Justifikasi Normalisasi (dalam markdown)

Assumes existing notebook cells 0-16 sudah dijalankan.
Variables yang diasumsikan sudah ada: df, COLORS, XAI_DIR
=========================================
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

# ===== RE-LOAD DATA (standalone) =====
klaim = pd.read_csv("./Data_Klaim.csv")
polis = pd.read_csv("./Data_Polis.csv")
for col in ['Tanggal Pasien Masuk RS','Tanggal Pasien Keluar RS','Tanggal Pembayaran Klaim']:
    klaim[col] = pd.to_datetime(klaim[col], errors='coerce')
polis['Tanggal Lahir'] = pd.to_datetime(polis['Tanggal Lahir'].astype(str), format='%Y%m%d', errors='coerce')
polis['Tanggal Efektif Polis'] = pd.to_datetime(polis['Tanggal Efektif Polis'].astype(str), format='%Y%m%d', errors='coerce')
df = klaim.merge(polis, on='Nomor Polis', how='left')
# FE
df['los'] = (df['Tanggal Pasien Keluar RS'] - df['Tanggal Pasien Masuk RS']).dt.days.fillna(0)
df['age_at_claim'] = ((df['Tanggal Pasien Masuk RS'] - df['Tanggal Lahir']).dt.days / 365.25).fillna(40)
df['processing_days'] = (df['Tanggal Pembayaran Klaim'] - df['Tanggal Pasien Masuk RS']).dt.days.fillna(0)
df['is_inpatient'] = (df['Inpatient/Outpatient'] == 'IP').astype(int)
df['is_singapore'] = (df['Lokasi RS'] == 'Singapore').astype(int)
target = 'Nominal Klaim Yang Disetujui'
XAI_DIR = Path("./xai_outputs"); XAI_DIR.mkdir(exist_ok=True)
COLORS = {'primary':'#2c3e50','accent':'#e74c3c','blue':'#3498db',
          'green':'#27ae60','orange':'#f39c12','purple':'#9b59b6'}

# =====================================================================
print("="*70)
print("SECTION A: MISSING VALUES & OUTLIER HANDLING")
print("="*70)
# =====================================================================
# --- A1: Missing Values ---
print("\n📊 Missing Values per Kolom (Data Klaim + Polis):")
missing_k = klaim.isnull().sum()
missing_k = missing_k[missing_k > 0].sort_values(ascending=False)
print("\nData Klaim:")
for col, n in missing_k.items():
    print(f"  {col:40s}: {n:5d} ({n/len(klaim)*100:.2f}%)")

missing_p = polis.isnull().sum()
missing_p = missing_p[missing_p > 0].sort_values(ascending=False)
print("\nData Polis:")
if len(missing_p) > 0:
    for col, n in missing_p.items():
        print(f"  {col:40s}: {n:5d} ({n/len(polis)*100:.2f}%)")
else:
    print("  Tidak ada missing values di Data Polis")

missing_df = df.isnull().sum()
missing_df = missing_df[missing_df > 0].sort_values(ascending=False)
print("\nData Gabungan (setelah merge):")
for col, n in missing_df.items():
    print(f"  {col:40s}: {n:5d} ({n/len(df)*100:.2f}%)")

# --- A2: Outlier Analysis ---
print("\n📊 Outlier Analysis:")
q1 = df[target].quantile(0.25)
q3 = df[target].quantile(0.75)
iqr = q3 - q1
upper_iqr = q3 + 1.5 * iqr
p95 = df[target].quantile(0.95)
p99 = df[target].quantile(0.99)
n_iqr = (df[target] > upper_iqr).sum()
n_p95 = (df[target] > p95).sum()
n_p99 = (df[target] > p99).sum()

print(f"  Q1: Rp {q1:,.0f}")
print(f"  Q3: Rp {q3:,.0f}")
print(f"  IQR: Rp {iqr:,.0f}")
print(f"  Upper bound (Q3+1.5×IQR): Rp {upper_iqr:,.0f}")
print(f"  Outlier (>IQR): {n_iqr} ({n_iqr/len(df)*100:.1f}%)")
print(f"  Outlier (>P95): {n_p95} ({n_p95/len(df)*100:.1f}%)")
print(f"  Outlier (>P99): {n_p99} ({n_p99/len(df)*100:.1f}%)")
print(f"\n  ✅ Keputusan: DIPERTAHANKAN (klaim ekstrem valid secara aktuaria)")

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Missing values bar
cols_miss = list(missing_df.head(8).index)
vals_miss = list(missing_df.head(8).values)
axes[0].barh(range(len(cols_miss)), vals_miss, color=COLORS['accent'])
axes[0].set_yticks(range(len(cols_miss)))
axes[0].set_yticklabels([c[:25] for c in cols_miss])
axes[0].set_xlabel('Jumlah Missing')
axes[0].set_title('Missing Values per Kolom', fontweight='bold')
for i, v in enumerate(vals_miss):
    axes[0].text(v+5, i, f'{v} ({v/len(df)*100:.1f}%)', va='center', fontsize=9)

# Outlier distribution
axes[1].hist(df[target]/1e6, bins=50, color=COLORS['blue'], edgecolor='white', alpha=0.8)
axes[1].axvline(upper_iqr/1e6, color='red', linestyle='--', linewidth=2, label=f'IQR: {upper_iqr/1e6:.0f}M')
axes[1].axvline(p95/1e6, color='orange', linestyle='--', linewidth=2, label=f'P95: {p95/1e6:.0f}M')
axes[1].axvline(p99/1e6, color='purple', linestyle='--', linewidth=2, label=f'P99: {p99/1e6:.0f}M')
axes[1].set_xlabel('Klaim Disetujui (Juta Rp)')
axes[1].set_title('Distribusi Klaim + Threshold Outlier', fontweight='bold')
axes[1].legend(fontsize=9)

# Outlier per lokasi
outlier_loc = df[df[target] > p95].groupby('Lokasi RS').size().sort_values(ascending=False)
c_loc = [COLORS['accent'] if x=='Singapore' else COLORS['blue'] for x in outlier_loc.index]
axes[2].bar(outlier_loc.index, outlier_loc.values, color=c_loc)
axes[2].set_title(f'Outlier (>{p95/1e6:.0f}M) per Lokasi RS', fontweight='bold')
axes[2].set_ylabel('Jumlah')
for i, (loc, v) in enumerate(outlier_loc.items()):
    axes[2].text(i, v+2, str(v), ha='center', fontweight='bold')

plt.suptitle('Identifikasi Missing Values dan Outlier', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(XAI_DIR / 'eda_missing_outlier.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: eda_missing_outlier.png")

# =====================================================================
print("\n" + "="*70)
print("SECTION B: SCATTER PLOT ANALYSIS")
print("="*70)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Age vs Claim (scatter)
sample_idx = np.random.RandomState(42).choice(len(df), min(2000, len(df)), replace=False)
colors_scatter = [COLORS['accent'] if s else COLORS['blue'] for s in df.iloc[sample_idx]['is_singapore']]
axes[0,0].scatter(df.iloc[sample_idx]['age_at_claim'], df.iloc[sample_idx][target]/1e6,
    c=colors_scatter, alpha=0.4, s=15, edgecolors='none')
axes[0,0].set_xlabel('Usia Tertanggung (Tahun)')
axes[0,0].set_ylabel('Klaim Disetujui (Juta Rp)')
axes[0,0].set_title('Usia vs Nominal Klaim\n(Merah=Singapore, Biru=Lainnya)', fontweight='bold')

# 2. LOS vs Claim (scatter)
colors_ip = [COLORS['accent'] if ip else COLORS['blue'] for ip in df.iloc[sample_idx]['is_inpatient']]
axes[0,1].scatter(df.iloc[sample_idx]['los'], df.iloc[sample_idx][target]/1e6,
    c=colors_ip, alpha=0.4, s=15, edgecolors='none')
axes[0,1].set_xlabel('Lama Rawat Inap (Hari)')
axes[0,1].set_ylabel('Klaim Disetujui (Juta Rp)')
axes[0,1].set_title('Lama Rawat vs Nominal Klaim\n(Merah=Inpatient, Biru=Outpatient)', fontweight='bold')

# 3. Processing Days vs Claim
axes[1,0].scatter(df.iloc[sample_idx]['processing_days'], df.iloc[sample_idx][target]/1e6,
    c=colors_scatter, alpha=0.4, s=15, edgecolors='none')
axes[1,0].set_xlabel('Waktu Proses Klaim (Hari)')
axes[1,0].set_ylabel('Klaim Disetujui (Juta Rp)')
axes[1,0].set_title('Waktu Proses vs Nominal Klaim', fontweight='bold')

# 4. Biaya RS vs Klaim Disetujui
axes[1,1].scatter(df.iloc[sample_idx]['Nominal Biaya RS Yang Terjadi']/1e6,
    df.iloc[sample_idx][target]/1e6, c=colors_scatter, alpha=0.4, s=15, edgecolors='none')
x_line = np.linspace(0, df['Nominal Biaya RS Yang Terjadi'].max()/1e6, 100)
axes[1,1].plot(x_line, x_line, '--', color='gray', alpha=0.5, label='y=x (100% approved)')
axes[1,1].set_xlabel('Biaya RS (Juta Rp)')
axes[1,1].set_ylabel('Klaim Disetujui (Juta Rp)')
axes[1,1].set_title('Biaya RS vs Klaim Disetujui\n(Garis = 100% approval)', fontweight='bold')
axes[1,1].legend()

plt.suptitle('Scatter Plot Analysis — Hubungan Antar Variabel', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(XAI_DIR / 'eda_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: eda_scatter_plots.png")

print("\n✅ Part 1 complete — 2 new visualizations ready")
