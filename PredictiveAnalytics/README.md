# Laporan Proyek Machine Learning - Kresna Devara
***
# HR Analytics Turn Over Rate

## Domain Proyek

Karyawan adalah salah satu aset berharga bagi suatu perusahaan. Baik atau buruknya suatu perusahaan dan juga perkembangannya salah satu faktor terbesarnya adalah karyawan, terutama karyawan yang memiliki kualitas baik.

Namun ditengah perkembangan jaman yang dinamik dan juga penuh kompetisi, sering kali karyawan berpindah-pindah perusahaan. Akibatnya perusahaan terkena dampak buruknya. Berdasarkan Work Institue pada tahun 2019, _turn over_ karyawan memiliki nilai _cost_ bisnis lebih dari 600 miliar dollar per tahunnya [1]. _Cost_ tersebut bukan hanya soal biaya, namun juga waktu. Akhirnya berbagai macam projek menjadi terhambat sehingga perkembangan perusahaan juga menjadi imbasnya.

Telah terdapat banyak studi yang dilakukan untuk mengetahui penyebab terjadinya _turn over_ atau _attrition_ pada karyawan seperti pada Rabiyathul, et.al [2] dan juga Walid, et.al [3]. Dimana faktor-faktor penyebabnya bisa terjadi akibat dari stres, kepuasaan pekerjaan (_satisfication_), lingkungan kerja, _rewards_, gaji dan lain sebagainya.

Permasalahan ini merupakan salah satu pekerjaan terberat yang harus dilakukan bagian Human Resource (HR) agar tetap dapat membuat karyawan-karyawan terbaiknya bekerja secara penuh untuk perusahaan (loyal). Sehingga HR perlu mengetahui faktor-faktor terbesar yang menentukan seorang karyawan pergi meninggalkan perusahaan. Dengan menggunakan _Machine Learning_, faktor-faktor tersebut dapat dianalisa dengan lebih mudah. Sehingga HR dapat mencegah _turn over_ atau setidaknya meminimalisir hal tersebut dengan meningkatkan faktor-faktor penyebabnya.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang tersebut, terdapat beberapa masalah yang harus diselesaikan diantaranya:
- Bagaimana cara memvisualisasikan data-data referensi?
- Bagaimana cara mengolah data (_preprocess_) untuk bisa digunakan dalam model _Machine Learning_
- Model _Machine Learning_ apa yang dapat digunakan pada masalah ini?
- Bagaimana cara mengetahui model _Machine Learning_ sudah berjalan dengan baik?
- Apa faktor-faktor penyebab terjadinya _turn over_ berdasarkan hasil dari _Machine Learning?_

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Melakukan _Exploratory Data Analysis_ (EDA) secara _univariate_ dan _multivariate_ untuk mengetahui data kasar secara grafik.
- Melakukan tahapan-tahapan _preprocessing_ pada data berdasarkan hasil EDA dan juga analisa awal data.
- Menggunakan model _Machine Learning_ umum seperti **KNN**, **SVM** dan juga **Random Forest**, dan juga melakukan _hyper parameter tunning_ pada masing-masing model.
- Mengetahui metrik evaluasi apa yang paling tepat untuk mendapatkan akurasi yang terbaik.
- Melakukan _Feature Importance_ setelah model _Machine Learning_ terbaik ditemukan.

### Solution Statement

Solusi yang dapat dilakukan sebagai berikut:
- Membandingkan hasil dari tiga algoritma yang ada, dan juga melakukan _tunning_ parameter baik secara konvensional (mencari nilai K dan C yang terbaik pada KNN dan SVM), ataupun menggunakan metode **GridSearchCV** pada algoritma Random Forest.
- Jika data merupakan data klasifikasi yang tidak seimbang (_imbalance dataset_) metrik evaluasi akan menggunakan nilai **f1 score**

## Data Understanding

Dataset yang digunakan adalah HR Analytics yang berasal dari [Kaggle](https://www.kaggle.com/datasets/giripujar/hr-analytics)

Berikut rangkuman umum mengenai dataset

Tabel 1. Rangkuman Dataset HR Analytics

Jenis | Keterangan 
--- | ---
Sumber | [Kaggle Dataset - HR Analytics](https://www.kaggle.com/datasets/giripujar/hr-analytics)
Lisensi | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Author | Giri Pujar
Size | 567 kB

Data dengan nama `HR_comma_sep.csv` berisikan 14999 baris dan 10 kolom. Dimana 10 kolom tersebut menjelaskan bebearap fitur yaitu:
- satisfaction_level: Angka kepuasan yang dibeirkan karyawan (0-1)
- last_evaluation: Angka penilaian dari manager (0-1)
- number_project: Jumlah projek yang pernah dikerjakan karyawan
- average_monthly_hours: Total jam kerja per bulan
- time_spend_company: Total masa kerja dalam tahun
- Work_accident: Dummy variable terjadinya kecelakaan: Ya (1), Tidak (0)
- left: Dummy variable: Keluar (1), Tetap (0)
- promoted_last_5years: Dummy variable, Dipromosikan (1), Tidak dipromosikan(0)
- sales: Nama department (sales,technical,support,IT, product,marketing, other)
- salary: 3-level kategori sallary (low, medium, high)

***
__Persiapan Data dan Exploratory Data Analysis (EDA)__
1. Melakukan _import_ data dari csv dengan menggunakan pandas
2. Melakukan pengecekan data secara umum, dan juga data yang kosong (null dan NA)
3. Melakukan pengecekan Outlier
4. Melakukan Univariate Analysis pada fitur numerik dan kategorik
5. Melakukan Multivariate Analysis pada fitur numerik dan kategorik
6. Melakukan Korelasi Matrix untuk masing-masing fitur

### 1. Import data dengan Pandas

Pada proyek ini library pandas digunakan, dimana data yang berjenis CSV diubah kedalam bentuk Pandas Data Frame. 5 data awalnya adalah sebagai berikut

Tabel 2. 5 Data awal Dataset

|  Index  |   satisfaction_level |   last_evaluation |   number_project |   average_montly_hours |   time_spend_company |   Work_accident |   left |   promotion_last_5years | Department   | salary   |
|---:|---------------------:|------------------:|-----------------:|-----------------------:|---------------------:|----------------:|-------:|------------------------:|:-------------|:---------|
|  0 |                 0.38 |              0.53 |                2 |                    157 |                    3 |               0 |      1 |                       0 | sales        | low      |
|  1 |                 0.8  |              0.86 |                5 |                    262 |                    6 |               0 |      1 |                       0 | sales        | medium   |
|  2 |                 0.11 |              0.88 |                7 |                    272 |                    4 |               0 |      1 |                       0 | sales        | medium   |
|  3 |                 0.72 |              0.87 |                5 |                    223 |                    5 |               0 |      1 |                       0 | sales        | low      |
|  4 |                 0.37 |              0.52 |                2 |                    159 |                    3 |               0 |      1 |                       0 | sales        | low      |


### 2. Pengencekan Data Secara Umum

dengan menggunakan `df.info()` kita bisa mengetahui jenis-jenis dari data yaitu:

![datatype](https://user-images.githubusercontent.com/60245989/201567069-7a0361f0-62dc-4a44-990e-d6760b551cad.PNG)

Gambar 1. Jenis-jenis Type pada Dataset

Jika dilihat lebih detail dengan menggunakan `df.nunique()` maka data data tersebut menjadi seperti berikut:

Tabel 3. Unique Value Dataset

| Fitur                 |   Unique Value |
|:----------------------|----:|
| satisfaction_level    |  92 |
| last_evaluation       |  65 |
| number_project        |   6 |
| average_montly_hours  | 215 |
| time_spend_company    |   8 |
| Work_accident         |   2 |
| left                  |   2 |
| promotion_last_5years |   2 |
| Department            |  10 |
| salary                |   3 |


Data-data tersebut memiliki 3 jenis tipe data yang berbeda yaitu float64, int64, dan juga object. Namun secara garis besar data-data tersebut dapat dikelompokkan secara numerik dan kategorik:
- Numerik: satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company 
- Kategorik: Work_accident, promoted_last_5_years, left, Department, salary

Setelah dilakukan pengecekan dengan menggunakan `df.isnull()` dan juga `df.isna()` _tidak ada data yang kosong_

Secara statistik datanya adalah sebagai berikut:

Tabel 4. Statistik Dataset

|       |   satisfaction_level |   last_evaluation |   number_project |   average_montly_hours |   time_spend_company |   Work_accident |         left |   promotion_last_5years |
|:------|---------------------:|------------------:|-----------------:|-----------------------:|---------------------:|----------------:|-------------:|------------------------:|
| count |         14999        |      14999        |      14999       |             14999      |          14999       |    14999        | 14999        |           14999         |
| mean  |             0.612834 |          0.716102 |          3.80305 |               201.05   |              3.49823 |        0.14461  |     0.238083 |               0.0212681 |
| std   |             0.248631 |          0.171169 |          1.23259 |                49.9431 |              1.46014 |        0.351719 |     0.425924 |               0.144281  |
| min   |             0.09     |          0.36     |          2       |                96      |              2       |        0        |     0        |               0         |
| 25%   |             0.44     |          0.56     |          3       |               156      |              3       |        0        |     0        |               0         |
| 50%   |             0.64     |          0.72     |          4       |               200      |              3       |        0        |     0        |               0         |
| 75%   |             0.82     |          0.87     |          5       |               245      |              4       |        0        |     0        |               0         |
| max   |             1        |          1        |          7       |               310      |             10       |        1        |     1        |               1         |

### 3. Pengecekan Outlier

Data numerik pada dataset hanyalah: 
- satisfaction_level
- last_evaluation
- number_project
- average_montly_hours
- time_spend_company 

Sehingga hanya data tersebut saja yang akan dilakukan pengecekan. Diagram boxplot digunakan untuk melakukan pengecekan pada outliers, gambar berikut menunjukan boxplot untuk setiap kategori numerik

![Outliers](https://user-images.githubusercontent.com/60245989/201568021-24de7a87-8845-4761-96f7-444f248d1108.png)

Gambar 2. Pengecekan Outliers pada Fitur Numerik

Dari data di atas hanya `time_spend_company` saja yang memiliki __outliers__ walaupun datanya sedikit, namun data lamanya tahun pada perusahaan berpengaruh terhadap keputusan pergi atau tidaknya karyawan dari suatu perusahaan, sehingga data `time_spend_company` __tidak dibuang__


### 4. Univariate Analysis

Pada analisa univariate, setiap kategori akan dilihat secara masing-masing.

#### 4.1 Univerate Analysis Fitur Kategorik

Untuk melakukan analisa Univariate pada fitur kategorik, diagram bar digunakan. Gambarnya adalah sebagai berikut:

![Kategorik](https://user-images.githubusercontent.com/60245989/201568739-ed827a81-58df-4289-8693-96f0b4d5a6e2.png)

Gambar 3. Univariate Analysis Fitur Kategorik

Dari data-data tersebut dapat diambil analisa bahwa:
1. Jumlah data karyawan yang tidak mengalami work accident lebih banyak dan tidak seimbang (imbalance)
2. Karyawan yang meninggalkan perusahaan lebih sediki dibanding yang tetap tinggal (imbalance)
3. Jumlah karyawan yang belum dipromosikan leibh banyak dan tidak seimbang
4. Departemen **sales** memiliki karyawan yang paling banyak
5. Karyawan dengan kategori tinggi memiliki jumlah paling rendah

#### 4.2 Univerate Analysis Fitur Numerik

Untuk melakukan analisa Univariate pada fitur numerik, sebuah historgram digunakan dengan melakukan binning 20. Diagramnya adalah sebagai berikut:

![numeriks](https://user-images.githubusercontent.com/60245989/201568918-dfa4f62a-8a39-420b-8be4-adf2894a468b.png)

Gambar 3. Univariate Analysis Fitur Numerik

Dari data-data tersebut dapat diambil analisa bahwa:

1. Satisfication level, Last Evaluation, Number Project, danAvarage Montly hours karyawan beragam
2. Jumlah karyawan dengan jumlah tahun kerja 3 tahun paling dominan

### 5. Multivariate Analysis

Pada analisa multivariate 2 kategori ada dibandingkan. Kategori `left` akan menjadi acuan terhadap kategori lainnya.

#### 5.1 Multivariate Analysis Fitur Kategorik

Untuk melakukan analisa Multivariate pada fitur kategorik, diagram bar digunakan. Gambarnya adalah sebagai berikut:

![MultivariateKategorik](https://user-images.githubusercontent.com/60245989/201570153-85f189bd-22e5-4e27-b511-5458bfca2419.png)

Gambar 4. Multivariate Analysis Fitur Kategorik

Dari data-data tersebut dapat dianalisa bahwa:

1. Jumlah data karywan yang pergi dan stay di perusahaan tidak seimbang (imbalance dataset)
2. Karyawan yang **belum** pernah mengalami work accident lebih banyak meninggalkan perusahaan
3. Karyawan yang **belum** dipromosikan dalam 5 tahun terakhir lebih banyak meninggalkan perusahaan
4. Departemen **sales** memiliki karyawan yang paling banyak meninggalkan perusahaan
5. Karyawan dengan gaji kategori **rendah** paling banyak meninggalkan karyawan

#### 5.1 Multivariate Analysis Fitur Kategorik

Untuk melakukan analisa Multivariate pada fitur numerik, Histogram digunakan. Pada analisa ini hanya dilihat data-data dari setiap fitur dimana bersinggunan juga dengan `left` (Hanya data karyawan yang telah pergi meninggal perusahaan yang ditampilkan). Diagramnya adalah sebagai berikut:

![Multivariatenumeriks](https://user-images.githubusercontent.com/60245989/201570687-089e70c5-1008-4ba7-8160-806b02279000.png)

Gambar 5. Multivariate Analysis Fitur Numerik

Dari data-data tersebut dapat dianalisa bahwa:
1. Karyawan yang pergi meninggalkan perusahaan adalah karyawan dengan satisfication level rendah (0.1) dan menengah  (0.4)
2. Karyawan dengan penilain buruk (<0.5) dan penilaian baik (>0.8) sama-sama pergi dari perusahaan
3. Karyawan dengan jumlah projek **terendah (2)** pergi meninggalkan perusahaan
4. Karyawan dengan jumlah jam paling sedikit setiap bulannya (<160 jam) paling banyak pergi meninggalkan perusahaan
5. Karyawan dengan waktu bersama perusahaan **3 tahun** paling banyak meninggalkan perusahaan

### 6. Korelasi Matrix Fitur

Korelasi antar fitur dapat dilakukan dengan menggunakan fitur `corr()` pada Pandas. Dengan korelasi matrix dapat dilihat hubungan antara dua jenis fitur, dimana nilainya akan berkisar dari -1 hingga +1. Semakin mendekati +1 berartikan bahwa kedua kategori memiliki hubungan kuat, sedangkan semakin mendekati -1 berarti kedua kategori memiliki hubungan kuat namun dalam arah yang berkebalikan (Semakin kecil nilainya semakin besar korelasinya). Berikut adalah korelasi matrix pada masing-masing fitur.

![corrmatrix](https://user-images.githubusercontent.com/60245989/201571258-694fd718-65a1-4057-b7e1-5b3b187552cc.png)

Gambar 6. Matrix Korelasi Fitur

Dari gambar diatas dapat dilihat bahwa `satisfication level` merupakan kategorik yang paling berpengaruh terhadap perginya karyawan (`left`) disusul dengan `Work_accident` dan juga `time_spend_company`


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
