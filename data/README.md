# Data

**Dataset tidak disertakan di repo ini.**

`Data_Klaim.csv` dan `Data_Polis.csv` merupakan data klaim asuransi kesehatan individu milik **AXA Financial Indonesia**, disediakan khusus untuk peserta Data Science Competition MCF ITB 2026. Data bersifat rahasia dan tidak boleh didistribusikan ulang, sehingga di-*ignore* secara eksplisit lewat `.gitignore`.

Halaman ini mendokumentasikan **skema** kedua file agar kode di `src/` dan `notebooks/` tetap bisa dibaca dan dijalankan ulang oleh pihak yang memang berhak memegang datanya.

## Cara menempatkan data

Skrip dan notebook membaca lewat *path* relatif `./Data_Klaim.csv` dan `./Data_Polis.csv` — yaitu **direktori kerja saat eksekusi**, bukan folder `data/` ini. Letakkan kedua CSV di root repo sebelum menjalankan apa pun:

```bash
cp /lokasi/aman/Data_Klaim.csv /lokasi/aman/Data_Polis.csv .
```

Kalau ingin menyimpannya di `data/`, ubah konstanta `KLAIM_PATH` / `POLIS_PATH` di [`src/pipeline_forecasting_xai.py`](../src/pipeline_forecasting_xai.py) (sekitar baris 36–37) dan sel konfigurasi notebook.

## `Data_Klaim.csv` — 4.627 baris × 13 kolom

Granularitas: satu baris per kejadian klaim. Periode masuk RS: Januari 2024 – Juli 2025.

| Kolom | Tipe | Catatan |
|---|---|---|
| `Claim ID` | string | Unik, 4.627 nilai |
| `Nomor Polis` | string | 1.210 polis unik; kunci join ke `Data_Polis.csv` |
| `Reimburse/Cashless` | string | **2 nilai: `R` dan `C`** — bukan kata penuh (lihat catatan bug di README utama) |
| `Inpatient/Outpatient` | string | 4 nilai; **37 null** |
| `ICD Diagnosis` | string | 753 nilai unik; 6 null |
| `ICD Description` | string | 946 nilai unik; 6 null |
| `Status Klaim` | string | Konstan (1 nilai) |
| `Tanggal Pembayaran Klaim` | string tanggal | **37 null** — klaim berstatus pending |
| `Tanggal Pasien Masuk RS` | string tanggal | Dasar seluruh agregasi bulanan |
| `Tanggal Pasien Keluar RS` | string tanggal | Bersama kolom di atas membentuk fitur `los` |
| `Nominal Klaim Yang Disetujui` | float | **Target** model severity |
| `Nominal Biaya RS Yang Terjadi` | float | Sengaja **tidak dipakai** sebagai fitur — data leakage |
| `Lokasi RS` | string | 10 nilai (Indonesia, Singapore, Malaysia, dll.); 7 null |

## `Data_Polis.csv` — 4.096 baris × 6 kolom

Granularitas: satu baris per polis.

| Kolom | Tipe | Catatan |
|---|---|---|
| `Nomor Polis` | string | Unik, 4.096 nilai |
| `Plan Code` | string | 3 nilai: `M-001` (Worldwide), `M-002` (Asia), `M-003` (Indonesia) |
| `Gender` | string | 2 nilai |
| `Tanggal Lahir` | int | Format `YYYYMMDD`, perlu dikonversi ke datetime |
| `Tanggal Efektif Polis` | int | Format `YYYYMMDD`, perlu dikonversi |
| `Domisili` | string | 21 nilai |

Integrasi memakai **left join** pada `Nomor Polis` dengan klaim sebagai sisi kiri, sehingga seluruh 4.627 klaim dipertahankan meski data polisnya tidak cocok.

## Format submission

`submissions/sample_submission.csv` (15 baris × 2 kolom, semua `value` = 0) menunjukkan skema yang diminta: kolom `id` dan `value`, dengan id berpola

```
2025_{08..12}_{Claim_Frequency|Claim_Severity|Total_Claim}
```

Contoh submission berisi angka asli ada di `submissions/` — nilainya sudah berupa agregat bulanan tingkat portofolio, bukan data klaim per individu.
