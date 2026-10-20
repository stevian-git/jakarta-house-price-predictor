# HANDOFF — Proyek ML `indonesian-house-price-predictor`

## 1. Konteks Proyek

Melanjutkan pengembangan proyek Machine Learning untuk prediksi harga rumah di Indonesia pada repository BARU.

- **Nama proyek:** `indonesian-house-price-predictor`
- **Lokasi lokal:** `~/python/indonesian-house-price-predictor`
- **Status repository:** Baru, repository publik di GitHub
- **Tujuan:** Portfolio Machine Learning / tugas kuliah
- **Bahasa kode dan dokumentasi:** Python; README dalam bahasa Inggris
- **Gaya kerja:** Step-by-step, dipandu melalui terminal
- **Aturan utama:** User yang mengetik dan menjalankan command. Berhenti di checkpoint setiap fase. Jangan memberikan seluruh kode proyek sekaligus.
- **Environment:** Python 3.12, `uv`
- **Aplikasi:** Streamlit
- **Model deployment:** scikit-learn Pipeline yang disimpan menggunakan joblib

### Prinsip pendampingan

1. Mulai dari Fase 0 dan jangan melompat ke fase berikutnya.
2. Berikan command yang diperlukan untuk satu langkah kecil pada satu waktu.
3. Jelaskan tujuan command secara singkat.
4. Minta user menjalankan command dan mengirimkan output aktual.
5. Verifikasi output sebelum melanjutkan.
6. Jika terjadi error, selesaikan error tersebut terlebih dahulu.
7. Jangan mengarang hasil command, metrik, grafik, struktur dataset, atau status deployment.
8. Jangan menulis semua file dan kode sekaligus.
9. Setiap fase memiliki kriteria selesai yang eksplisit.
10. Jangan meminta user mengulang langkah yang telah diverifikasi berhasil.

---

## 2. Tujuan dan Batasan Proyek

Membangun pipeline ML end-to-end yang mencakup:

1. Pengambilan dataset dari Kaggle.
2. Audit dan eksplorasi data (EDA).
3. Pembersihan data dan feature engineering.
4. Pembuatan baseline model.
5. Perbandingan dan evaluasi model regresi.
6. Cross-validation dan hyperparameter tuning.
7. Penyimpanan model dan preprocessing dalam satu artefak.
8. Aplikasi prediksi interaktif menggunakan Streamlit.
9. Dokumentasi, visualisasi hasil, dan publikasi repository.

### Definisi masalah

- **Task:** Supervised learning — regression.
- **Unit prediksi:** Satu listing rumah.
- **Target:** Harga listing dalam Rupiah.
- **Target transformasi training:** `log1p(price)`.
- **Output aplikasi:** Estimasi harga dalam Rupiah.
- **Evaluasi utama:** MAE dalam Rupiah.
- **Evaluasi tambahan:** RMSE dalam Rupiah dan R².

Model memperkirakan harga berdasarkan pola listing historis. Hasilnya bukan jaminan harga transaksi, penilaian resmi, atau estimasi harga pasar terkini.

### Cakupan geografis — keputusan penting

Dataset yang dipilih pada handoff ini adalah dataset harga rumah Jakarta. Dengan demikian:

- Cakupan awal adalah wilayah yang benar-benar terwakili oleh dataset tersebut.
- Jangan mengklaim model mencakup seluruh Jabodetabek sebelum memeriksa nilai aktual pada kolom `city`.
- Lakukan audit cakupan kota pada Fase 1.
- Jika data hanya mencakup Jakarta, pertahankan proyek sebagai **Jakarta House Price Predictor**.
- Untuk mengembangkan versi Jabodetabek, dibutuhkan dataset Kaggle yang memang mencakup Jakarta, Bogor, Depok, Tangerang, dan Bekasi, serta proses validasi tambahan.
- Jangan menambahkan kota yang tidak didukung oleh data training ke dalam UI seolah-olah model telah belajar dari kota tersebut.

---

## 3. Dataset

### Sumber utama

**Jakarta House Price Dataset — Kaggle**

- Dataset identifier: `abiyyurasyiq/jakarta-house-price-dataset`
- Metode download: `kagglehub.dataset_download(...)`
- Dataset publik; gunakan metode akses tanpa token sebagaimana direncanakan, lalu verifikasi pada environment aktual.
- Lisensi yang tercatat dalam handoff: MIT. Verifikasi metadata dan file lisensi dari sumber dataset sebelum publikasi.
- Jumlah baris yang dilaporkan: sekitar 10.000.
- Seluruh informasi di atas merupakan asumsi awal yang harus diverifikasi dari hasil download, bukan hasil inspeksi yang sudah terkonfirmasi pada sesi ini.

### Kolom yang dilaporkan

| Kolom | Peran |
|---|---|
| `index` | Kandidat identifier; audit sebelum dihapus |
| `price` | Target harga dalam Rupiah |
| `district` | Informasi lokasi |
| `city` | Informasi kota |
| `bed_rooms` | Jumlah kamar tidur |
| `bath_rooms` | Jumlah kamar mandi |
| `carport` | Kapasitas carport |
| `land_area` | Luas tanah |
| `building_area` | Luas bangunan |

### Audit awal wajib

Pada Fase 1, verifikasi:

- Nama file dan nama kolom aktual.
- Jumlah baris dan kolom.
- Tipe data setiap kolom.
- Missing values.
- Duplikasi penuh dan kemungkinan duplikasi listing.
- Distribusi target `price`.
- Harga minimum, maksimum, median, kuantil, dan rentang harga.
- Jumlah kategori unik pada `city` dan `district`.
- Kota dan distrik yang benar-benar ada.
- Nilai tidak masuk akal pada luas bangunan, luas tanah, kamar, dan carport.
- Lisensi dan metadata dataset.

Jangan mengasumsikan seluruh 10.000 baris bersih atau valid sebelum audit.

### Penyimpanan data

- Raw dataset: `data/raw/jakarta_house.csv`
- Clean dataset: `data/processed/train_clean.csv`

File dataset lokal tidak dimasukkan ke Git.

---

## 4. Spesifikasi Fitur Model

Versi pertama menggunakan tepat lima input fitur, sesuai rancangan UI.

### Input model

```python
FEATURES = [
    "city",
    "building_area",
    "bed_rooms",
    "bath_rooms",
    "carport",
]
```

### Target

```python
TARGET = "price"
```

Target transformasi:

```python
y_train = np.log1p(price)
```

Prediksi dikembalikan ke skala Rupiah:

```python
predicted_price = np.expm1(predicted_log_price)
```

Gunakan inverse transform hanya sekali. Pastikan hasil prediksi finite dan tidak negatif sebelum ditampilkan.

### Keputusan fitur

- `city`: fitur kategorikal utama untuk lokasi.
- `building_area`: fitur ukuran utama.
- `bed_rooms`: jumlah kamar tidur.
- `bath_rooms`: jumlah kamar mandi.
- `carport`: kapasitas carport.
- `district`: tidak digunakan pada baseline karena cardinality berpotensi tinggi; boleh dianalisis pada EDA.
- `land_area`: tidak digunakan pada versi pertama agar kontrak input tetap tepat lima fitur. Dapat diuji sebagai eksperimen lanjutan yang terdokumentasi.
- `index`: tidak digunakan sebagai fitur model kecuali ada bukti kuat bahwa kolom tersebut memiliki makna prediktif yang sah; identifier biasanya harus dikeluarkan.

Catatan: lima fitur harus diverifikasi pada dataset aktual. Jika ada nama atau tipe kolom yang berbeda, perbaiki mapping data secara eksplisit dan dokumentasikan.

---

## 5. Strategi Cleaning dan EDA

### Target skewness dan outlier

Target `price` dilaporkan memiliki distribusi sangat skewed dan rentang yang ekstrem, dari sekitar Rp1,8 juta hingga Rp900 miliar.

Angka tersebut harus diverifikasi dari CSV aktual.

`log1p(price)` wajib diuji sebagai transformasi target untuk training, tetapi tidak boleh diasumsikan otomatis menghasilkan distribusi normal. Buktikan melalui histogram dan statistik deskriptif.

### Prinsip cleaning

1. Konversi `price` menjadi numerik.
2. Tangani harga missing, nol, negatif, atau tidak valid.
3. Konversi fitur numerik dengan error handling.
4. Audit nilai tidak masuk akal pada `building_area`, kamar, kamar mandi, dan carport.
5. Identifikasi duplikasi dan listing yang mungkin sama.
6. Analisis outlier dengan kuantil, IQR, histogram, dan domain knowledge.
7. Hapus atau koreksi hanya nilai yang dapat dibuktikan tidak valid.
8. Jangan menghapus semua rumah mahal hanya karena harganya ekstrem.
9. Dokumentasikan jumlah baris sebelum dan sesudah setiap keputusan cleaning.
10. Hindari imputasi atau transformasi yang menggunakan statistik dari test set.

Jika harga sangat ekstrem membuat model tidak stabil, bandingkan hasil sebelum dan sesudah penanganan outlier. Setiap aturan filtering harus didokumentasikan agar hasil eksperimen dapat direproduksi.

### Analisis EDA yang harus tersedia

- Distribusi target pada skala asli dan log.
- Statistik harga per kota.
- Jumlah listing per kota dan distrik.
- Hubungan `building_area` dengan `price`.
- Hubungan kamar dan carport dengan harga.
- Missing values dan duplikasi.
- Korelasi fitur numerik.
- Outlier dan pemeriksaan kewajaran data.
- Keterwakilan geografis.
- Keterbatasan dataset historis.

Simpan grafik akhir di `results/figures/`.

### File hasil cleaning

`data/processed/train_clean.csv`

File ini harus berisi data yang sudah dibersihkan dan tetap mempertahankan lima fitur input beserta target. Dokumentasikan jika ada kolom tambahan yang sengaja dipertahankan untuk analisis.

---

## 6. Arsitektur Pipeline

Gunakan `ColumnTransformer` dan scikit-learn Pipeline agar preprocessing dan model tersimpan bersama.

### Preprocessing

Untuk `city`:

- Imputasi missing values bila diperlukan.
- `OneHotEncoder(handle_unknown="ignore")`.

Untuk fitur numerik:

- Imputasi median jika diperlukan.
- Scaling hanya jika sesuai dengan model yang dipakai.
- Tree-based models tidak memerlukan scaling, tetapi preprocessing tetap harus konsisten.

### Target transformation

Gunakan `TransformedTargetRegressor` atau implementasi ekuivalen yang menjamin:

1. Target ditransformasi menggunakan `log1p` saat training.
2. Prediksi otomatis dikembalikan ke skala asli oleh estimator.
3. Metric dihitung dari prediksi dan target dalam Rupiah.
4. Tidak ada inverse transform ganda.

### Penyimpanan model

Simpan seluruh preprocessing dan estimator sebagai satu artefak:

`models/model.joblib`

Jangan menyimpan model mentah yang membutuhkan preprocessing manual terpisah pada saat inference.

---

## 7. Strategi Training dan Eksperimen

### Model baseline

Bandingkan minimal empat model:

1. `DummyRegressor` — baseline sederhana.
2. `LinearRegression` atau `Ridge` — baseline linear.
3. `RandomForestRegressor` — model utama kandidat.
4. `GradientBoostingRegressor` — pembanding boosting.

Pemilihan model final tidak boleh didasarkan pada asumsi bahwa Random Forest pasti terbaik.

### Pembagian data

Default:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)
```

Test set harus disisihkan sebelum proses tuning dan tidak boleh digunakan untuk memilih hyperparameter.

### Cross-validation

- Gunakan cross-validation pada training set.
- Gunakan `GridSearchCV` atau pencarian parameter yang sesuai untuk Random Forest.
- Batasi ruang pencarian agar tetap layak pada dataset sekitar 10.000 baris.
- Gunakan preprocessing pipeline di dalam cross-validation untuk menghindari leakage.
- Catat parameter terbaik dan skor validasi.

Jika ditemukan listing duplikat atau kelompok rumah yang sangat mirip, pertimbangkan split berdasarkan grup. Jika tujuan penelitian menilai generalisasi ke lokasi baru, pertimbangkan evaluasi berdasarkan kota atau distrik, dengan memperhatikan ukuran data setiap kelompok.

### Custom scorer

Gunakan custom scorer untuk menghitung MAE atau RMSE pada skala asli Rupiah meskipun target training berada pada skala log.

Pastikan scorer:

- Menggunakan prediksi yang sudah dikembalikan ke skala Rupiah.
- Tidak menerapkan `expm1` dua kali.
- Konsisten dengan kontrak estimator yang digunakan.
- Diuji dengan beberapa contoh prediksi sebelum dipakai untuk tuning.

Jika menggunakan `TransformedTargetRegressor`, prediksi publiknya sudah berada pada skala target asli. Scorer harus mengikuti kontrak ini.

### Pemeriksaan overfitting

Catat:

- Training MAE dan RMSE.
- Validation MAE dan RMSE.
- Test MAE dan RMSE.
- R².
- Selisih performa train dan test.
- Waktu training jika relevan.

Model dengan training error sangat rendah tetapi test error jauh lebih tinggi perlu ditinjau dari sisi overfitting.

---

## 8. Evaluasi Model

### Metrik wajib

**MAE — Mean Absolute Error**

Metrik utama karena langsung menggambarkan rata-rata selisih absolut prediksi dalam Rupiah.

**RMSE — Root Mean Squared Error**

Memberi penalti lebih besar terhadap kesalahan yang sangat besar.

**R²**

Mengukur seberapa baik prediksi menjelaskan variasi target. Nilai negatif dimungkinkan.

Semua metrik utama harus dihitung pada skala Rupiah.

### Analisis tambahan

- Actual vs. predicted plot.
- Residual plot.
- Distribusi absolute error.
- Error berdasarkan kota.
- Error berdasarkan rentang harga.
- Perbandingan baseline dan model final.
- Feature importance atau permutation importance.
- Analisis jumlah sampel per kategori lokasi.

Jangan menjanjikan nilai akurasi tertentu sebelum eksperimen dilakukan.

### Pemilihan model

Pilih model final berdasarkan performa validation/CV dan pertimbangan generalisasi, bukan test set.

Test set digunakan untuk evaluasi final setelah keputusan model selesai.

### Artefak hasil

`results/metrics.json`

Isi setidaknya:

- Dataset identifier dan jumlah sampel.
- Nama model.
- Daftar fitur.
- Transformasi target.
- Random seed.
- Strategi split dan CV.
- Parameter model terbaik.
- MAE, RMSE, dan R².
- Versi library utama.
- Tanggal eksperimen.

Jangan memasukkan angka metrik contoh atau placeholder sebagai hasil eksperimen aktual.

---

## 9. Struktur Repository

```text
rumah-price-predictor/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── jakarta_house.csv
│   └── processed/
│       └── train_clean.csv
├── models/
│   └── model.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_evaluate_tune.ipynb
├── results/
│   ├── metrics.json
│   └── figures/
├── src/
│   ├── fetch_data.py
│   ├── data_prep.py
│   ├── train.py
│   ├── predict.py
│   └── make_figures.py
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

### Tanggung jawab modul

- `fetch_data.py`: download dan menyimpan dataset raw.
- `data_prep.py`: validasi, cleaning, dan penulisan data processed.
- `train.py`: split, training, cross-validation, tuning, evaluasi, dan penyimpanan model.
- `predict.py`: helper inference, validasi input, pemuatan model, dan formatting Rupiah.
- `make_figures.py`: menghasilkan ulang grafik final dari data dan hasil eksperimen.
- `app/app.py`: UI Streamlit dan integrasi helper inference.
- Notebook: eksplorasi dan analisis; tidak menjadi satu-satunya tempat implementasi pipeline.

Gunakan path berbasis `Path(__file__).resolve()` agar script dapat dipanggil dari berbagai working directory.

---

## 10. Dependencies dan Environment

Gunakan Python 3.12 dan `uv`.

### Main dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `streamlit`
- `kagglehub`

### Development dependencies

- `jupyter`
- `ipykernel`

Pin dan lock dependency melalui `uv.lock`. Setelah environment siap, verifikasi versi Python dan kemampuan import package.

Jangan memasukkan token, API key, kredensial pribadi, atau file konfigurasi rahasia ke repository publik.

### `.gitignore`

```gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
data/raw/
data/processed/
models/*.joblib
.DS_Store
.vscode/
```

Tambahkan file atau folder lain yang berisi artefak lokal maupun rahasia jika ditemukan selama pengembangan.

---

## 11. Spesifikasi Aplikasi Streamlit

### Lima input UI

1. `city`
2. `building_area`
3. `bed_rooms`
4. `bath_rooms`
5. `carport`

Opsi `city` harus diambil dari kota yang valid dan didukung dataset. Jangan menawarkan kota di luar cakupan training.

Validasi input:

- Luas bangunan harus positif.
- Jumlah kamar dan carport harus mengikuti batas input yang masuk akal.
- Kategori kota harus valid.
- Input kosong atau tidak valid harus menghasilkan pesan yang jelas.
- Prediksi harus berupa angka finite dan nonnegatif.

### Output

- Estimasi harga dalam Rupiah.
- Format yang mudah dibaca, misalnya `Rp 1,5 Miliar`.
- Penjelasan singkat bahwa hasil merupakan estimasi model berbasis listing historis.
- Informasi model atau metrik evaluasi bila tersedia.

Contoh formatting:

- Rp 750 Juta
- Rp 1,5 Miliar
- Rp 12 Miliar

Pastikan pembulatan konsisten dan nilai asli tetap digunakan dalam perhitungan.

### Batasan UI

UI hanya menerima lima fitur yang digunakan pada training. Jangan menambahkan field UI yang tidak digunakan oleh model, atau menghilangkan fitur wajib saat inference.

---

## 12. README dan Portfolio

README ditulis dalam bahasa Inggris dan mencakup:

1. Project overview.
2. Problem statement.
3. Dataset source dan license.
4. Geographic scope dan periode dataset.
5. Feature definitions.
6. Data cleaning dan EDA.
7. Modeling methodology.
8. Baseline comparison.
9. Evaluation metrics.
10. Best model dan hasil aktual.
11. Feature importance.
12. Cara instalasi dan menjalankan training.
13. Cara menjalankan aplikasi Streamlit.
14. Screenshot atau grafik hasil.
15. Limitations dan potensi bias.
16. Future improvements.

Jangan memasukkan data mentah yang dilarang lisensi atau mengandung informasi sensitif. Periksa lisensi dan syarat distribusi sebelum mempublikasikan artefak dataset.

README tidak boleh menyatakan hasil model atau deployment berhasil sebelum diverifikasi.

---

## 13. Alur Fase dan Checkpoint

### Fase 0 — Setup repository

Tujuan:

- Buat repository lokal baru.
- Inisialisasi `uv` dengan Python 3.12.
- Tambahkan dependencies.
- Buat struktur folder.
- Buat `.gitignore`.
- Inisialisasi Git.
- Verifikasi environment.
- Buat commit pertama.

Checkpoint wajib:

- Output versi Python.
- Output `uv --version`.
- Struktur folder.
- Status Git.
- Isi dependency awal.

**Jangan lanjut ke Fase 1 sebelum user mengirim output aktual dan setup terverifikasi.**

### Fase 1 — Dataset acquisition

Tujuan:

- Implementasikan `src/fetch_data.py`.
- Download menggunakan `kagglehub`.
- Simpan file ke `data/raw/jakarta_house.csv`.
- Verifikasi shape, kolom, missing values, distribusi harga, kategori kota, dan lisensi.

Checkpoint wajib:

- Shape aktual.
- Daftar kolom aktual.
- Missing values.
- Statistik harga.
- `city.nunique()` dan daftar kota.
- `district.nunique()`.
- Konfirmasi lisensi yang ditemukan.

### Fase 2 — EDA dan cleaning

Tujuan:

- Lengkapi notebook `01_eda.ipynb`.
- Audit outlier dan missing values.
- Bandingkan distribusi harga asli dan log.
- Dokumentasikan setiap keputusan cleaning.
- Implementasikan `src/data_prep.py`.
- Simpan data processed.

Checkpoint wajib:

- Grafik distribusi.
- Statistik sebelum dan sesudah cleaning.
- Jumlah baris yang dipertahankan.
- Alasan penghapusan atau koreksi data.
- Daftar kolom final dan validasi lima fitur.

### Fase 3 — Baseline models

Tujuan:

- Implementasikan split data yang konsisten.
- Bangun preprocessing pipeline.
- Latih Dummy, Linear/Ridge, Random Forest, dan Gradient Boosting.
- Hitung MAE, RMSE, dan R² dalam Rupiah.
- Simpan hasil perbandingan.

Checkpoint wajib:

- Tabel metrik aktual.
- Jumlah sampel train dan test.
- Konfirmasi bahwa tidak terjadi data leakage.
- Model baseline terbaik berdasarkan validasi.

### Fase 4 — Evaluation dan tuning

Tujuan:

- Cross-validation.
- GridSearchCV Random Forest.
- Pemeriksaan overfitting.
- Analisis error per kota dan rentang harga.
- Pilih model final berdasarkan validation/CV.
- Evaluasi final pada test set.
- Simpan `models/model.joblib`.
- Simpan `results/metrics.json`.

Checkpoint wajib:

- Parameter terbaik.
- Metrik CV.
- Metrik test final.
- Grafik actual vs. predicted dan residual.
- Validasi bahwa model tersimpan dan dapat dimuat ulang.

### Fase 5 — Streamlit

Tujuan:

- Implementasikan `src/predict.py`.
- Implementasikan `app/app.py`.
- Buat lima input yang sesuai dengan model.
- Format hasil dalam Rupiah.
- Tangani input invalid dan exception.
- Jalankan aplikasi secara lokal.

Checkpoint wajib:

- Output prediksi untuk input valid.
- Bukti model berhasil dimuat.
- Verifikasi bahwa input inference sesuai dengan fitur training.
- Screenshot atau hasil pengujian UI.

### Fase 6 — README dan GitHub

Tujuan:

- Buat grafik final melalui `make_figures.py`.
- Dokumentasikan hasil aktual.
- Tambahkan screenshot aplikasi.
- Lengkapi README bahasa Inggris.
- Verifikasi `.gitignore`.
- Pastikan dataset mentah dan artefak model tidak terunggah.
- Commit dan push ke repository GitHub publik.

Checkpoint wajib:

- Status Git.
- Pemeriksaan file yang akan dipublikasikan.
- README lengkap.
- Link repository yang benar-benar telah dibuat dan di-push.

---

## 14. Aturan Kualitas Teknis

1. Reproducibility: random seed dan konfigurasi eksperimen terdokumentasi.
2. No leakage: preprocessing dan tuning dilakukan hanya pada training data.
3. Single artifact: model dan preprocessing disimpan bersama.
4. Honest evaluation: metrik hanya berasal dari eksperimen yang benar-benar dijalankan.
5. Feature consistency: lima fitur UI harus identik dengan kontrak inference model.
6. Robustness: input invalid, kategori tidak dikenal, dan nilai prediksi nonfinite ditangani.
7. Reproducible paths: gunakan path absolut berbasis lokasi script.
8. Data integrity: keputusan cleaning dicatat.
9. Scope integrity: jangan klaim seluruh Jabodetabek jika dataset hanya mewakili Jakarta.
10. Public repository hygiene: tidak mengunggah secrets, data lokal, atau artefak yang seharusnya diabaikan Git.

---

## 15. Mulai dari Fase 0

Mulai hanya dengan Fase 0.

Jangan menulis seluruh kode proyek.

Langkah pertama adalah memeriksa environment dan memastikan lokasi kerja sesuai dengan handoff:

`~/python/rumah-price-predictor`

Berikan satu langkah command kecil, minta user menjalankannya, lalu periksa output sebelum melanjutkan. Setelah Fase 0 selesai dan diverifikasi, lanjutkan ke Fase 1.

**Status awal:** Fase 0 belum dijalankan atau diverifikasi dalam handoff ini.