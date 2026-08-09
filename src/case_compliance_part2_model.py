#!/usr/bin/env python3
"""
CASE DOCUMENT COMPLIANCE SCRIPT — PART 2
=========================================
Sections to add to notebook:
  D. Model Comparison: LightGBM vs XGBoost vs RF (25%)
  E. Time Series Cross-Validation (25%)
  F. Ablation Study: with/without FE (25%)
  G. Forecast 2026 + Rekomendasi (30%)
  H. Total Claim Factor Analysis SHAP (25%)

Run AFTER Part 1 or after existing notebook cells.
=========================================
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb, xgboost as xgb, shap, warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
warnings.filterwarnings('ignore')

XAI_DIR = Path("./xai_outputs"); XAI_DIR.mkdir(exist_ok=True)
COLORS = {'primary':'#2c3e50','accent':'#e74c3c','blue':'#3498db',
          'green':'#27ae60','orange':'#f39c12','purple':'#9b59b6'}

# ===== RE-LOAD & PREPARE =====
klaim = pd.read_csv("./Data_Klaim.csv")
polis = pd.read_csv("./Data_Polis.csv")
for col in ['Tanggal Pasien Masuk RS','Tanggal Pasien Keluar RS','Tanggal Pembayaran Klaim']:
    klaim[col] = pd.to_datetime(klaim[col], errors='coerce')
polis['Tanggal Lahir'] = pd.to_datetime(polis['Tanggal Lahir'].astype(str), format='%Y%m%d', errors='coerce')
polis['Tanggal Efektif Polis'] = pd.to_datetime(polis['Tanggal Efektif Polis'].astype(str), format='%Y%m%d', errors='coerce')
df = klaim.merge(polis, on='Nomor Polis', how='left')
df['year'] = df['Tanggal Pasien Masuk RS'].dt.year
df['month'] = df['Tanggal Pasien Masuk RS'].dt.month
df['quarter'] = df['Tanggal Pasien Masuk RS'].dt.quarter
df['year_month'] = df['Tanggal Pasien Masuk RS'].dt.to_period('M')
df['day_of_week'] = df['Tanggal Pasien Masuk RS'].dt.dayofweek
df['day_of_month'] = df['Tanggal Pasien Masuk RS'].dt.day
df['los'] = (df['Tanggal Pasien Keluar RS'] - df['Tanggal Pasien Masuk RS']).dt.days.fillna(0)
df['processing_days'] = (df['Tanggal Pembayaran Klaim'] - df['Tanggal Pasien Masuk RS']).dt.days.fillna(0)
df['age_at_claim'] = ((df['Tanggal Pasien Masuk RS'] - df['Tanggal Lahir']).dt.days / 365.25).fillna(40)
df['policy_tenure_years'] = ((df['Tanggal Pasien Masuk RS'] - df['Tanggal Efektif Polis']).dt.days / 365.25).fillna(5)
df['is_inpatient'] = (df['Inpatient/Outpatient'] == 'IP').astype(int)
df['is_outpatient'] = (df['Inpatient/Outpatient'] == 'OP').astype(int)
df['is_odc'] = (df['Inpatient/Outpatient'].isin(['ODC','ODS'])).astype(int)
df['is_reimburse'] = (df['Reimburse/Cashless'] == 'Reimburse').astype(int)
df['is_male'] = (df['Gender'] == 'M').astype(int)
df['is_indonesia'] = (df['Lokasi RS'] == 'Indonesia').astype(int)
df['is_singapore'] = (df['Lokasi RS'] == 'Singapore').astype(int)
df['is_malaysia'] = (df['Lokasi RS'] == 'Malaysia').astype(int)
df['is_overseas'] = (~df['Lokasi RS'].isin(['Indonesia','Singapore','Malaysia'])).astype(int)
for plan in ['M-001','M-002','M-003']:
    df[f'is_plan_{plan.replace("-","_")}'] = (df['Plan Code'] == plan).astype(int)
df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)

target = 'Nominal Klaim Yang Disetujui'
sev_features = ['month','quarter','year','month_sin','month_cos','los','processing_days',
    'age_at_claim','policy_tenure_years','is_inpatient','is_outpatient','is_odc',
    'is_reimburse','is_male','is_indonesia','is_singapore','is_malaysia','is_overseas',
    'is_plan_M_001','is_plan_M_002','is_plan_M_003','day_of_week','day_of_month']

train_df = df[df['year']==2024]; val_df = df[df['year']==2025]
X_train, y_train = train_df[sev_features].fillna(-999), train_df[target]
X_val, y_val = val_df[sev_features].fillna(-999), val_df[target]

def calc_metrics(y_true, y_pred):
    mape = np.mean(np.abs(y_true - y_pred) / np.where(y_true==0, 1, y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAPE': mape, 'RMSE': rmse, 'R²': r2}

lgb_params = {'objective':'regression','metric':'mape','learning_rate':0.05,
    'num_leaves':31,'min_child_samples':20,'feature_fraction':0.8,
    'bagging_fraction':0.8,'bagging_freq':1,'reg_alpha':0.5,'reg_lambda':0.5,
    'verbose':-1,'seed':42}

# =====================================================================
print("="*70)
print("SECTION D: MODEL COMPARISON — SEVERITY")
print("="*70)
# =====================================================================

# LightGBM
lgb_model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_val, label=y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
lgb_pred = lgb_model.predict(X_val)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
    min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, early_stopping_rounds=50, verbosity=0)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_val)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=20,
    max_features=0.8, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_val)

sev_results = {}
for name, pred in [('LightGBM', lgb_pred), ('XGBoost', xgb_pred), ('Random Forest', rf_pred)]:
    sev_results[name] = calc_metrics(y_val, pred)
    m = sev_results[name]
    print(f"  {name:15s}: MAPE={m['MAPE']:.2f}%  RMSE=Rp{m['RMSE']:,.0f}  R²={m['R²']:.4f}")

# Best model
best_sev = min(sev_results, key=lambda x: sev_results[x]['MAPE'])
print(f"\n  ✅ Best Severity Model: {best_sev}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models = list(sev_results.keys())
c3 = [COLORS['blue'], COLORS['accent'], COLORS['green']]
for i, (metric, ylabel) in enumerate([('MAPE','MAPE (%)'), ('RMSE','RMSE (Rp)'), ('R²','R²')]):
    vals = [sev_results[m][metric] for m in models]
    bars = axes[i].bar(models, vals, color=c3, edgecolor='white', linewidth=1.5)
    axes[i].set_title(metric, fontweight='bold', fontsize=14)
    axes[i].set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        fmt = f'{v:.2f}%' if metric=='MAPE' else (f'Rp{v/1e6:.1f}M' if metric=='RMSE' else f'{v:.4f}')
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, fmt, ha='center', va='bottom', fontsize=10, fontweight='bold')
    best_idx = np.argmax(vals) if metric=='R²' else np.argmin(vals)
    bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(3)
plt.suptitle('Perbandingan Model Severity — LightGBM vs XGBoost vs Random Forest', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(XAI_DIR / 'model_comparison_severity.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: model_comparison_severity.png")

# --- Frequency Model Comparison ---
print(f"\n{'='*70}\nSECTION D (cont): MODEL COMPARISON — FREQUENCY\n{'='*70}")
monthly = df.groupby('year_month').agg(
    frequency=('Claim ID','count'), total_claim=(target,'sum'),
    avg_los=('los','mean'), ip_ratio=('is_inpatient','mean'), sg_ratio=('is_singapore','mean'),
    reimb_ratio=('is_reimburse','mean'), avg_age=('age_at_claim','mean'),
    m003_ratio=('is_plan_M_003','mean'), n_icd=('ICD Diagnosis','nunique')).reset_index()
monthly['ts'] = monthly['year_month'].dt.to_timestamp()
monthly = monthly.sort_values('ts').reset_index(drop=True)
monthly['severity'] = monthly['total_claim'] / monthly['frequency']
monthly['month_num'] = monthly['ts'].dt.month
monthly['year'] = monthly['ts'].dt.year
monthly['month_sin'] = np.sin(2*np.pi*monthly['month_num']/12)
monthly['month_cos'] = np.cos(2*np.pi*monthly['month_num']/12)
for lag in [1,2,3]:
    monthly[f'freq_lag{lag}'] = monthly['frequency'].shift(lag)
    monthly[f'sev_lag{lag}'] = monthly['severity'].shift(lag)
monthly['freq_roll3'] = monthly['frequency'].rolling(3).mean()
monthly['sev_roll3'] = monthly['severity'].rolling(3).mean()
freq_features = ['month_num','year','month_sin','month_cos','ip_ratio','sg_ratio',
    'reimb_ratio','avg_age','m003_ratio','n_icd','freq_lag1','freq_lag2','freq_lag3',
    'sev_lag1','freq_roll3','sev_roll3','avg_los']
valid = monthly.dropna(subset=['freq_lag3'])
n = len(valid)
f_train, f_val = valid.iloc[:n-3], valid.iloc[n-3:]
Xf_tr, yf_tr = f_train[freq_features].fillna(-999), f_train['frequency']
Xf_va, yf_va = f_val[freq_features].fillna(-999), f_val['frequency']

f_lgb_params = {'objective':'regression','metric':'mape','learning_rate':0.05,
    'num_leaves':10,'min_child_samples':3,'feature_fraction':0.8,
    'bagging_fraction':0.8,'bagging_freq':1,'reg_alpha':0.1,'reg_lambda':0.5,
    'verbose':-1,'seed':42}
f_lgb = lgb.train(f_lgb_params, lgb.Dataset(Xf_tr, label=yf_tr), num_boost_round=300,
    valid_sets=[lgb.Dataset(Xf_va, label=yf_va)], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

f_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
    min_child_weight=3, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
f_xgb.fit(Xf_tr, yf_tr)

f_rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3,
    max_features=0.8, random_state=42, n_jobs=-1)
f_rf.fit(Xf_tr, yf_tr)

freq_results = {}
for name, model in [('LightGBM', f_lgb), ('XGBoost', f_xgb), ('Random Forest', f_rf)]:
    pred = model.predict(Xf_va) if name != 'LightGBM' else f_lgb.predict(Xf_va)
    freq_results[name] = calc_metrics(yf_va, pred)
    print(f"  {name:15s}: MAPE={freq_results[name]['MAPE']:.2f}%")

best_freq = min(freq_results, key=lambda x: freq_results[x]['MAPE'])
print(f"  ✅ Best Frequency Model: {best_freq}")

# =====================================================================
print(f"\n{'='*70}\nSECTION E: TIME SERIES CROSS-VALIDATION\n{'='*70}")
# =====================================================================
df_sorted = df.sort_values('Tanggal Pasien Masuk RS').reset_index(drop=True)
periods = sorted(df_sorted['year_month'].unique())
fold_configs = [
    (periods[:10], periods[10:13], "Fold 1: 10→3 bln"),
    (periods[:13], periods[13:16], "Fold 2: 13→3 bln"),
    (periods[:16], periods[16:],   "Fold 3: 16→sisa"),
]
cv_results = []
for train_p, val_p, fold_name in fold_configs:
    tr = df_sorted[df_sorted['year_month'].isin(train_p)]
    va = df_sorted[df_sorted['year_month'].isin(val_p)]
    Xtr, ytr = tr[sev_features].fillna(-999), tr[target]
    Xva, yva = va[sev_features].fillna(-999), va[target]
    m = lgb.train(lgb_params, lgb.Dataset(Xtr, label=ytr), num_boost_round=500,
        valid_sets=[lgb.Dataset(Xva, label=yva)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pred = m.predict(Xva)
    metrics = calc_metrics(yva, pred)
    metrics['Fold'] = fold_name; metrics['n_train'] = len(tr); metrics['n_val'] = len(va)
    cv_results.append(metrics)
    print(f"  {fold_name}: train={len(tr)}, val={len(va)}, MAPE={metrics['MAPE']:.2f}%, R²={metrics['R²']:.4f}")

avg_mape = np.mean([r['MAPE'] for r in cv_results])
std_mape = np.std([r['MAPE'] for r in cv_results])
avg_r2 = np.mean([r['R²'] for r in cv_results])
std_r2 = np.std([r['R²'] for r in cv_results])
print(f"\n  Average: MAPE={avg_mape:.2f}% ± {std_mape:.2f}%  R²={avg_r2:.4f} ± {std_r2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
folds = [r['Fold'].split(':')[0] for r in cv_results]
mapes_cv = [r['MAPE'] for r in cv_results]; r2s_cv = [r['R²'] for r in cv_results]
axes[0].bar(folds, mapes_cv, color=[COLORS['blue'], COLORS['green'], COLORS['accent']])
axes[0].axhline(avg_mape, color='red', linestyle='--', label=f'Mean: {avg_mape:.2f}%')
axes[0].set_ylabel('MAPE (%)'); axes[0].set_title('MAPE per Fold', fontweight='bold'); axes[0].legend()
axes[1].bar(folds, r2s_cv, color=[COLORS['blue'], COLORS['green'], COLORS['accent']])
axes[1].axhline(avg_r2, color='red', linestyle='--', label=f'Mean: {avg_r2:.4f}')
axes[1].set_ylabel('R²'); axes[1].set_title('R² per Fold', fontweight='bold'); axes[1].legend()
plt.suptitle('Time Series Cross-Validation — Expanding Window (LightGBM Severity)', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(XAI_DIR / 'cv_results.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: cv_results.png")

# =====================================================================
print(f"\n{'='*70}\nSECTION F: ABLATION STUDY (WITH vs WITHOUT FE)\n{'='*70}")
# =====================================================================
raw_features_cols = ['month', 'year']
df_raw = df.copy()
for col in ['Inpatient/Outpatient','Reimburse/Cashless','Gender','Lokasi RS','Plan Code']:
    le = LabelEncoder()
    df_raw[f'{col}_enc'] = le.fit_transform(df_raw[col].fillna('Unknown'))
    raw_features_cols.append(f'{col}_enc')

raw_train = df_raw[df_raw['year']==2024]
raw_val = df_raw[df_raw['year']==2025]
Xr_tr = raw_train[raw_features_cols].fillna(-999)
Xr_va = raw_val[raw_features_cols].fillna(-999)

raw_model = lgb.train(lgb_params, lgb.Dataset(Xr_tr, label=y_train), num_boost_round=500,
    valid_sets=[lgb.Dataset(Xr_va, label=y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
raw_pred = raw_model.predict(Xr_va)
raw_m = calc_metrics(y_val, raw_pred)
fe_m = calc_metrics(y_val, lgb_pred)

print(f"  Tanpa FE ({len(raw_features_cols)} fitur): MAPE={raw_m['MAPE']:.2f}%  R²={raw_m['R²']:.4f}")
print(f"  Dengan FE ({len(sev_features)} fitur): MAPE={fe_m['MAPE']:.2f}%  R²={fe_m['R²']:.4f}")
print(f"  Delta R²: +{fe_m['R²']-raw_m['R²']:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
labels = [f'Tanpa FE\n({len(raw_features_cols)} fitur)', f'Dengan FE\n({len(sev_features)} fitur)']
for i, (metric, ylabel) in enumerate([('MAPE','MAPE (%)'), ('RMSE','RMSE (Rp)'), ('R²','R²')]):
    vals = [raw_m[metric], fe_m[metric]]
    bars = axes[i].bar(labels, vals, color=[COLORS['accent'], COLORS['green']], edgecolor='white', linewidth=1.5)
    axes[i].set_title(metric, fontweight='bold', fontsize=14); axes[i].set_ylabel(ylabel)
    for bar, v in zip(bars, vals):
        fmt = f'{v:.2f}%' if metric=='MAPE' else (f'Rp{v/1e6:.1f}M' if metric=='RMSE' else f'{v:.4f}')
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, fmt, ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.suptitle('Ablation Study — Dampak Feature Engineering terhadap Performa Model', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(XAI_DIR / 'ablation_fe.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: ablation_fe.png")

# =====================================================================
print(f"\n{'='*70}\nSECTION G: FORECAST 2026 (12 BULAN)\n{'='*70}")
# =====================================================================
result = monthly.copy()
forecasts_all = {}

# Re-forecast Aug-Dec 2025 as base
for target_month in range(8, 13):
    m_str = f'{target_month:02d}'
    future = result.iloc[-1].copy()
    future['month_num'] = target_month; future['year'] = 2025
    future['month_sin'] = np.sin(2*np.pi*target_month/12)
    future['month_cos'] = np.cos(2*np.pi*target_month/12)
    for lag in [1,2,3]:
        idx = len(result) - lag
        if idx >= 0: future[f'freq_lag{lag}'] = result.iloc[idx]['frequency']; future[f'sev_lag{lag}'] = result.iloc[idx]['severity']
    future['freq_roll3'] = result['frequency'].tail(3).mean()
    future['sev_roll3'] = result['severity'].tail(3).mean()
    X_freq = pd.DataFrame([future[freq_features]]).fillna(-999)
    pred_freq = max(1, f_lgb.predict(X_freq)[0])
    template = df[df['year_month'] == pd.Period(f'2024-{m_str}', 'M')].copy()
    template['month'] = target_month; template['year'] = 2025
    template['quarter'] = ((target_month-1)//3)+1
    template['month_sin'] = np.sin(2*np.pi*target_month/12); template['month_cos'] = np.cos(2*np.pi*target_month/12)
    pred_sev = np.mean(lgb_model.predict(template[sev_features].fillna(-999)))
    future['frequency'] = pred_freq; future['severity'] = pred_sev
    future['total_claim'] = pred_freq * pred_sev
    result = pd.concat([result, pd.DataFrame([future])], ignore_index=True)
    forecasts_all[(2025, target_month)] = {'frequency': pred_freq, 'severity': pred_sev, 'total_claim': pred_freq*pred_sev}

# Forecast Jan-Dec 2026
print("--- Prediksi Klaim 2026 ---")
for target_month in range(1, 13):
    m_str = f'{target_month:02d}'
    future = result.iloc[-1].copy()
    future['month_num'] = target_month; future['year'] = 2026
    future['month_sin'] = np.sin(2*np.pi*target_month/12)
    future['month_cos'] = np.cos(2*np.pi*target_month/12)
    for lag in [1,2,3]:
        idx = len(result) - lag
        if idx >= 0: future[f'freq_lag{lag}'] = result.iloc[idx]['frequency']; future[f'sev_lag{lag}'] = result.iloc[idx]['severity']
    future['freq_roll3'] = result['frequency'].tail(3).mean()
    future['sev_roll3'] = result['severity'].tail(3).mean()
    X_freq = pd.DataFrame([future[freq_features]]).fillna(-999)
    pred_freq = max(1, f_lgb.predict(X_freq)[0])
    tmpl = df[df['year_month'] == pd.Period(f'2024-{m_str}', 'M')].copy()
    if len(tmpl) == 0: tmpl = df[df['month'] == target_month].copy()
    tmpl['month'] = target_month; tmpl['year'] = 2026; tmpl['quarter'] = ((target_month-1)//3)+1
    tmpl['month_sin'] = np.sin(2*np.pi*target_month/12); tmpl['month_cos'] = np.cos(2*np.pi*target_month/12)
    pred_sev = np.mean(lgb_model.predict(tmpl[sev_features].fillna(-999)))
    future['frequency'] = pred_freq; future['severity'] = pred_sev; future['total_claim'] = pred_freq*pred_sev
    result = pd.concat([result, pd.DataFrame([future])], ignore_index=True)
    forecasts_all[(2026, target_month)] = {'frequency': pred_freq, 'severity': pred_sev, 'total_claim': pred_freq*pred_sev}
    print(f"  {target_month:02d}/2026: freq={pred_freq:.0f}  sev=Rp{pred_sev:,.0f}  total=Rp{pred_freq*pred_sev:,.0f}")

# Summary
total_2026 = sum(forecasts_all[(2026,m)]['total_claim'] for m in range(1,13))
total_hist = monthly['total_claim'].sum()
print(f"\n  Total Klaim 2026: Rp {total_2026:,.0f}")
print(f"  Total Historis (Jan24-Jul25): Rp {total_hist:,.0f}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(18, 14))
n_hist = len(monthly)
x_hist = list(range(n_hist))
x_2025 = list(range(n_hist, n_hist+5))
x_2026 = list(range(n_hist+5, n_hist+17))

for ax, key, ylabel, title, div in zip(axes, ['frequency','severity','total_claim'],
    ['Jumlah Klaim','Severity (Juta Rp)','Total Klaim (Miliar Rp)'],
    ['Frequency','Severity','Total Claim'], [1, 1e6, 1e9]):
    h = monthly[key].values / div if key != 'frequency' else monthly['frequency'].values
    f25 = [forecasts_all[(2025,m)][key]/div if key!='frequency' else forecasts_all[(2025,m)]['frequency'] for m in range(8,13)]
    f26 = [forecasts_all[(2026,m)][key]/div if key!='frequency' else forecasts_all[(2026,m)]['frequency'] for m in range(1,13)]
    ax.plot(x_hist, h, 'o-', color=COLORS['blue'], linewidth=2, markersize=5, label='Historis (2024-2025)')
    ax.plot(x_2025, f25, 's--', color=COLORS['accent'], linewidth=2, markersize=7, label='Forecast 2025')
    ax.plot(x_2026, f26, 'D--', color=COLORS['green'], linewidth=2, markersize=7, label='Forecast 2026')
    ax.axvline(n_hist-0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(n_hist+4.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Timeline Prediksi Klaim: Historis → Forecast 2025 → Forecast 2026', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(XAI_DIR / 'forecast_2026_timeline.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: forecast_2026_timeline.png")

# =====================================================================
print(f"\n{'='*70}\nSECTION H: TOTAL CLAIM FACTOR ANALYSIS (SHAP)\n{'='*70}")
# =====================================================================
tc_features = ['month_num','year','month_sin','month_cos','ip_ratio','sg_ratio',
    'reimb_ratio','avg_age','m003_ratio','n_icd','avg_los','frequency','severity']
valid_tc = monthly.dropna(subset=['freq_roll3']).copy()
Xtc = valid_tc[tc_features].fillna(-999)
ytc = valid_tc['total_claim']

tc_model = lgb.train({'objective':'regression','metric':'mape','learning_rate':0.05,
    'num_leaves':8,'min_child_samples':3,'verbose':-1,'seed':42},
    lgb.Dataset(Xtc, label=ytc), num_boost_round=200)

tc_explainer = shap.TreeExplainer(tc_model)
tc_shap = tc_explainer.shap_values(Xtc)

tc_trans = {'month_num':'Bulan','year':'Tahun','month_sin':'Seasonal(sin)','month_cos':'Seasonal(cos)',
    'ip_ratio':'Rasio Rawat Inap','sg_ratio':'Rasio Singapore','reimb_ratio':'Rasio Reimburse',
    'avg_age':'Usia Rata-rata','m003_ratio':'Rasio Plan M-003','n_icd':'Jumlah Diagnosis Unik',
    'avg_los':'Rata-rata Lama Rawat','frequency':'Frekuensi Klaim','severity':'Severity Klaim'}
tc_names = [tc_trans.get(f, f) for f in tc_features]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# SHAP bar
mean_abs_tc = np.mean(np.abs(tc_shap), axis=0)
tc_imp = pd.DataFrame({'f': tc_names, 's': mean_abs_tc}).sort_values('s')
c_tc = [COLORS['accent'] if v > tc_imp['s'].median() else COLORS['blue'] for v in tc_imp['s']]
axes[0].barh(range(len(tc_imp)), tc_imp['s'].values, color=c_tc)
axes[0].set_yticks(range(len(tc_imp))); axes[0].set_yticklabels(tc_imp['f'].values)
axes[0].set_xlabel('Mean |SHAP Value|')
axes[0].set_title('Faktor Pendorong Total Nominal Klaim\n(SHAP Analysis)', fontsize=13, fontweight='bold')

# Decomposition bar
x_pos = np.arange(len(valid_tc))
bars = axes[1].bar(x_pos, valid_tc['total_claim'].values/1e9, color=COLORS['blue'], alpha=0.7, label='Total Claim')
ax2 = axes[1].twinx()
ax2.plot(x_pos, valid_tc['frequency'].values, 'o-', color=COLORS['accent'], linewidth=2, label='Frequency')
ax2.set_ylabel('Frequency', color=COLORS['accent'])
axes[1].set_ylabel('Total Claim (Miliar Rp)')
axes[1].set_title('Dekomposisi: Total Claim = Freq × Severity', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper left'); ax2.legend(loc='upper right')
plt.suptitle('Analisis Faktor Total Nominal Klaim', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(XAI_DIR / 'total_claim_shap.png', dpi=150, bbox_inches='tight'); plt.close()
print("Saved: total_claim_shap.png")

# =====================================================================
print(f"\n{'='*70}")
print("✅ ALL SECTIONS COMPLETE!")
print(f"{'='*70}")
for f in sorted(XAI_DIR.glob('*.png')):
    if any(x in f.name for x in ['model_comp','cv_','ablation','forecast_2026','total_claim_shap','eda_missing','eda_scatter']):
        print(f"  📊 {f.name}")
