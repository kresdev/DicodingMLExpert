# Laporan Proyek Machine Learning - Kresna Devara
***
# Movie Recommendation

***
## Domain Proyek

Pada pekermbangan jaman saat ini, dimana dunia entertainment berkembang dengan saat pesat, berbagai hiburan seperti: permainan (game) dan juga film banyak diminati oleh berbagai kalangan. Saat ini layangan streaming berbayar sangatlah menjamur. Selain kualitas film yang disuguhkan salah satu faktor utama lainnya adalah relevansi antara pengguna dan juga konten dari film tersebut. Layanan streaming berbayar memiliki berbagai macam demografi pengguna, baik itu dengan usia anak-anak hinga sampai lansia. Umumnya para pengguna yang sudah lansia dan juga anak-anak tidak dapat memilih filmnya secara lebih mudah, bahkan pengguna dengan usia produktifpun terkadang masih kesusahan dalam menentukan film pilihannya. Sehingga rekomendasi film merupakan salah satu yang dapat meningkatkan tingkat minat pengguna dalam melakukan layanan streaming berbayar.

Penelitian telah menunjukan rekomendasi film (_movie recommendation_) merupakan faktor penting dalam sebuah aplikasi streaming berbayar. Berbagai peneliti mencoba menemukan teknik-teknik yang efektif untuk melakukan rekomendasi film, seperti yang dilakukan oleh Halder, et.al dengan melakukan _Movie Recommendation System Based on Movie Swarm_ [1](https://ieeexplore.ieee.org/document/6382910), ataupun yang berbasis Content Based Filtering seperti yang dilakukan oleh Reddy, et.al [2](https://www.researchgate.net/publication/331966843_Content-Based_Movie_Recommendation_System_Using_Genre_Correlation) ataupun yang berbasis Collaborative Filtering yang dilakukan oleh Schafer, et.al [3](https://www.researchgate.net/publication/200121027_Collaborative_Filtering_Recommender_Systems). Berbagai teknik ini memiliki tujuan utama yang pada akhirnya adalah dapat meningkatan nilai bisnis dari suatu layanan berbayar.

Permasalahan ini merupakan salah satu hal terberat bagi suatu perusahaan layanan streaming berbayar agar dapat memberikan rekomendasi film terbaik kepada penggunanya. Semakin relevan rekomendasi yang diberikan, penggunaan aplikasi akan semakin meningkat sehingga dapat meningkatnya juga jumlah pengguna secara jangka panjang. Dengan menggunakan _Machine Learning_ masalah ini dapat dengan lebih baik diselesaikan. Teknik-teknik dengan berbasis Content ataupun dengan menggunakan Collaborative Filtering dapat digunakan sehingga pengguna baru ataupun pengguna yang bahkan belum pernah menyaksikan film tersebut bisa mendapatkan rekomendasi yang relevan sesuai _behaviour_nya.

***
## Business Understanding

### Problem Statements
Berdasarkan latar belakang tersebut, terdapat beberapa masalah yang harus diselesaikan diantaranya:
- Bagaimana cara memvisualisasikan data-data referensi?
- Bagaimana cara mengolah data (_preprocess_) untuk bisa digunakan dalam model _Machine Learning_?
- Model _Machine Learning_ apa saja yang dapat digunakan pada masalah ini?
- Bagaimana cara mengetahui model _Machine Learning_ sudah berjalan dengan baik?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Melakukan _Exploratory Data Analysis_ (EDA) untuk mengetahui data kasar secara grafik.
- Melakukan tahapan-tahapan _preprocessing_ pada data berdasarkan hasil EDA dan juga analisa awal data.
- Menggunakan model _Machine Learning_ yang diggunakan untuk sistem rekomendasi seperti _Content-Based Filtering_ dan juga _Colaborative Filtering_.
- Mengetahui metrik evaluasi apa yang paling tepat untuk mendapatkan akurasi yang terbaik.

### Solution Statement

Solusi yang dapat dilakukan sebagai berikut:
- Membandingkan hasil rekomendasi dari dua jenis algoritma yang berbeda yaitu _Content-Based Filtering_ dan juga _Colaborative Filtering_.
- Melihat parameter-parameter apa saja yang dapat digunakan sebagai feature dari kedua model tersebut.
- Melakukan evaluasi dengan menggunakan metrik yang relevan terhadap kedua model tersebut baik menggunakan Presisi ataupun Root Mean Square.

***
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
- satisfaction_level: Angka kepuasan yang diberikan karyawan (0-1)
- last_evaluation: Angka penilaian dari manager (0-1)
- number_project: Jumlah projek yang pernah dikerjakan karyawan
- average_monthly_hours: Total jam kerja per bulan
- time_spend_company: Total masa kerja dalam tahun
- work_accident: Dummy variable terjadinya kecelakaan: Ya (1), Tidak (0)
- left: Dummy variable: Keluar (1), Tetap (0)
- promoted_last_5years: Dummy variable, Dipromosikan (1), Tidak dipromosikan(0)
- department: Nama department (sales,technical,support,IT, product,marketing, other)
- salary: 3-level kategori salary (low, medium, high)

__Persiapan Data dan Exploratory Data Analysis (EDA)__
1. Melakukan _import_ data dari csv dengan menggunakan Pandas.
2. Melakukan pengecekan data secara umum, dan juga data yang kosong (null dan NA).
3. Melakukan pengecekan _Outlier_.
4. Melakukan _Univariate Analysis_ pada fitur numerik dan kategorik.
5. Melakukan _Multivariate Analysis_ pada fitur numerik dan kategorik.
6. Melakukan Korelasi Metrik untuk masing-masing fitur.

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
- Numerik: satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company.
- Kategorik: Work_accident, promoted_last_5_years, left, Department, salary.

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

### 3. Pengecekan _Outlier_

Data numerik pada dataset hanyalah: 
- satisfaction_level
- last_evaluation
- number_project
- average_montly_hours
- time_spend_company 

Sehingga hanya data tersebut saja yang akan dilakukan pengecekan. Diagram boxplot digunakan untuk melakukan pengecekan pada _outliers_, gambar berikut menunjukan boxplot untuk setiap kategori numerik.

![Outliers](https://user-images.githubusercontent.com/60245989/201568021-24de7a87-8845-4761-96f7-444f248d1108.png)

Gambar 2. Pengecekan Outliers pada Fitur Numerik

Dari data di atas hanya `time_spend_company` saja yang memiliki ___outliers___ walaupun datanya sedikit, namun data _time_spend_company_ pada perusahaan berpengaruh terhadap keputusan pergi atau tidaknya karyawan dari suatu perusahaan, sehingga data `time_spend_company` __tidak dibuang__.


### 4. _Univariate Analysis_

Pada analisa _Univariate_, setiap kategori akan dilihat secara masing-masing.

#### 4.1 _Univerate Analysis_ Fitur Kategorik

Untuk melakukan analisa _Univariate_ pada fitur kategorik, diagram bar digunakan. Gambarnya adalah sebagai berikut:

![Kategorik](https://user-images.githubusercontent.com/60245989/201568739-ed827a81-58df-4289-8693-96f0b4d5a6e2.png)

Gambar 3. _Univariate Analysis_ Fitur Kategorik

Dari data-data tersebut dapat diambil analisa bahwa:
1. Jumlah data karyawan yang tidak mengalami work accident lebih banyak dan tidak seimbang (imbalance).
2. Karyawan yang meninggalkan perusahaan lebih sedikiT dibanding yang tetap tinggal (imbalance).
3. Jumlah karyawan yang belum dipromosikan lebih banyak dan tidak seimbang.
4. Departemen **sales** memiliki karyawan yang paling banyak.
5. Karyawan dengan kategori gaji paling tinggi memiliki jumlah paling rendah.

#### 4.2 _Univerate Analysis_ Fitur Numerik

Untuk melakukan analisa Univariate pada fitur numerik, sebuah historgram digunakan dengan melakukan _binning_ sebesar 20. Diagramnya adalah sebagai berikut:

![numeriks](https://user-images.githubusercontent.com/60245989/201568918-dfa4f62a-8a39-420b-8be4-adf2894a468b.png)

Gambar 3. _Univariate Analysis_ Fitur Numerik

Dari data-data tersebut dapat diambil analisa bahwa:

1. Satisfication level, Last Evaluation, Number Project, danAvarage Montly hours karyawan beragam.
2. Jumlah karyawan dengan jumlah tahun kerja 3 tahun paling dominan.

### 5. _Multivariate Analysis_

Pada analisa _Multivariate_ 2 kategori ada dibandingkan. Kategori `left` akan menjadi acuan terhadap kategori lainnya.

#### 5.1 _Multivariate Analysis_ Fitur Kategorik

Untuk melakukan analisa _Multivariate_ pada fitur kategorik, diagram bar digunakan. Gambarnya adalah sebagai berikut:

![MultivariateKategorik](https://user-images.githubusercontent.com/60245989/201570153-85f189bd-22e5-4e27-b511-5458bfca2419.png)

Gambar 4. _Multivariate Analysis_ Fitur Kategorik

Dari data-data tersebut dapat dianalisa bahwa:

1. Jumlah data karywan yang pergi dan stay di perusahaan tidak seimbang (imbalance dataset).
2. Karyawan yang **belum** pernah mengalami work accident lebih banyak meninggalkan perusahaan.
3. Karyawan yang **belum** dipromosikan dalam 5 tahun terakhir lebih banyak meninggalkan perusahaan.
4. Departemen **sales** memiliki karyawan yang paling banyak meninggalkan perusahaan.
5. Karyawan dengan gaji kategori **rendah** paling banyak meninggalkan karyawan.

#### 5.2 _Multivariate Analysis_ Fitur Numerik

Untuk melakukan analisa _Multivariate_ pada fitur numerik, Histogram digunakan. Pada analisa ini hanya dilihat data-data dari setiap fitur dimana bersinggungan juga dengan `left` (Hanya data karyawan yang telah pergi meninggal perusahaan yang ditampilkan). Diagramnya adalah sebagai berikut:

![Multivariatenumeriks](https://user-images.githubusercontent.com/60245989/201570687-089e70c5-1008-4ba7-8160-806b02279000.png)

Gambar 5. _Multivariate Analysis_ Fitur Numerik

Dari data-data tersebut dapat dianalisa bahwa:
1. Karyawan yang pergi meninggalkan perusahaan adalah karyawan dengan satisfication level rendah (0.1) dan menengah  (0.4).
2. Karyawan dengan penilain buruk (<0.5) dan penilaian baik (>0.8) sama-sama pergi dari perusahaan.
3. Karyawan dengan jumlah projek **terendah (2)** pergi meninggalkan perusahaan.
4. Karyawan dengan jumlah jam paling sedikit setiap bulannya (<160 jam) paling banyak pergi meninggalkan perusahaan.
5. Karyawan dengan waktu bersama perusahaan **3 tahun** paling banyak meninggalkan perusahaan.

### 6. Korelasi Metrik Fitur

Korelasi antar fitur dapat dilakukan dengan menggunakan fitur `corr()` pada Pandas. Dengan korelasi metrik dapat dilihat hubungan antara dua jenis fitur, dimana nilainya akan berkisar dari -1 hingga +1. Semakin mendekati +1 berartikan bahwa kedua kategori memiliki hubungan kuat, sedangkan semakin mendekati -1 berarti kedua kategori memiliki hubungan kuat namun dalam arah yang berkebalikan (Semakin kecil nilainya semakin besar korelasinya). Berikut adalah korelasi matrix pada masing-masing fitur.

![corrmatrix](https://user-images.githubusercontent.com/60245989/201571258-694fd718-65a1-4057-b7e1-5b3b187552cc.png)

Gambar 6. Metrik Korelasi Fitur

Dari gambar diatas dapat dilihat bahwa `satisfication level` merupakan kategorik yang paling berpengaruh terhadap perginya karyawan (`left`) disusul dengan `Work_accident` dan juga `time_spend_company`.

***
## _Data Preparation_

Pada tahap ini akan dilakukan _preprocessing_ terhadap data yang akan dimasukkan ke dalam model _Machine Learning_, ada beberapa tahapan yang dilakukan, yaitu:
1. Melakukan _encoding_ pada fitur kategorik.
2. Memisahkan dataset untuk training dan juga test (Dataset splitting).
3. Melakukan _scaling_/normalisasi terhadap dataset.

Pada proses _Data Preparation_ saat ini proses reduksi fitur dengan menggunakan PCA tidak dilakukan karena berdasarkan korelasi metrik, tidak ada fitur selain fitur target yang memiliki korelasi yang kuat.

### _Encoding_ Fitur

Mesin tidak mampu memproses data berupa string secara langsung, sehingga data string atau fitur kategorik perlu dilakukan proses yang disebut `encoding`. Pada projek kali ini encoding berjenis `One Hot Encoding` akan digunakan. Dimana dari fitur yang ada, masing-masing memiliki representasinya sendiri. Contohnya Jika terdapat fitur gender: male dan female, maka hasil One Hot Encodingnya adalah sebagai berikut:

|Index|Sex_female|Sex_male|
|:----|:---------|-------:|
|0    |1         | 0      |
|1    |0         | 1      |

Dari dataset yang ada, hanya terdapat 2 fitur dengan nama kategori yang beragam, yaitu:
- Department
- Salary

Library scikit-learn dan pandas memiliki fitur untuk melakukan Encoding secara mudah dengan menggunakan method `get_dummies`. Maka dataset setelah dilakukan One Hot Encoding menjadi seperti berikut

Tabel 5. Data Encoded

|    |   satisfaction_level |   last_evaluation |   number_project |   average_montly_hours |   time_spend_company |   Work_accident |   left |   promotion_last_5years | Department   | salary   |
|---:|---------------------:|------------------:|-----------------:|-----------------------:|---------------------:|----------------:|-------:|------------------------:|:-------------|:---------|
|  0 |                 0.38 |              0.53 |                2 |                    157 |                    3 |               0 |      1 |                       0 | sales        | low      |
|  1 |                 0.8  |              0.86 |                5 |                    262 |                    6 |               0 |      1 |                       0 | sales        | medium   |
|  2 |                 0.11 |              0.88 |                7 |                    272 |                    4 |               0 |      1 |                       0 | sales        | medium   |
|  3 |                 0.72 |              0.87 |                5 |                    223 |                    5 |               0 |      1 |                       0 | sales        | low      |
|  4 |                 0.37 |              0.52 |                2 |                    159 |                    3 |               0 |      1 |                       0 | sales        | low      |

### Dataset _Splitting_

Untuk setiap proyek _Machine Learning_ perlu dilakukannya proses pemisahan antara data untuk `Training` dan juga untuk `Test` agar tidak terjadi _Overfit_ ataupun _data leakage_ ketika model _Machine Learning_ selesai dibuat. Lebih lanjut fitur _Cross Validation_ juga perlu dilakukan agar datanya lebih konsisten, _Cross Validation_ akan digunakan ketika melakukan _Hyperparameter tunning_ dengan menggunakan `GridSearchCV`.

Jumlah dataset ini cukup banyak, totalnya berjumlah 14999 dataset, sehingga pembagian dataset dengan porsi 80% training : 20% testing, sudahlah cukup. Dengan menggunakan Library Scikit-learn proses train test split dapat dengan mudah dilakukan. Random_state yang digunakan adalah _42_. Sehingga Jumlahnya menjadi:
- Data train: 11999
- Data test: 3000

### Scaling dan Normalisasi

Dalam pemprosesan data pada _Machine Learning_ melakukan normalisasi terhadap data sangatlah penting, agar tidak terjadi ketidak seimbangan terhadap _weight_/bobot pada data dengan nilai yang tinggi dibandingkan nilai yang rendah. Terdapat beberapa jenis teknik normalisasi/_scaling_ yang sering digunakan. MinMax scaler dan Standard Scaler adalah dua teknik normalisasi yang paling populer.

MinMax Scaler bekerja dengan melakukan normalisasi data menjadi pada rentang tertentu (umumnya 0 hingga 1, atau -1 hingga 1). Sedangkan Standard scaler melakukan proses standarisasi fitur dengan menghilangkan mean dan membuat standard deviasinya data menjadi 1.

Untuk menghindari kebocoran data (_data leakage_) proses standarisasi haruslah terpisah. Dengan menggunakan library scikit-learn prosesnya dalam dilakukan seperti berikut:

Pertama-tama scaler akan melihat persebaran data training dengan menggunakan metode `fit`

`scaler.fit(X_train)`

Selanjutnya data training akan dilakukan normalisasi dengan method `transform`

`scaler.transform(X_train)`

Agar tidak terjadi _data leakage_ data test __tidak boleh__ diikut sertakan dalam proses fit. Dan hanya digunakan ketika melakukan transform

`scaler.transform(X_test)`

Berikut merupakan contoh data train setelah dilakukan proses normalisasi:

Tabel 6. Normalisasi Data

|       |   satisfaction_level |   last_evaluation |   number_project |   average_montly_hours |   time_spend_company |
|------:|---------------------:|------------------:|-----------------:|-----------------------:|---------------------:|
| 12896 |            0.474481  |         -0.562644 |         0.162568 |               0.921578 |             0.342509 |
| 12545 |            0.675614  |          1.65763  |         0.97399  |               0.701313 |             1.02861  |
| 14833 |           -2.05979   |         -0.971642 |        -1.46027  |               0.921578 |             0.342509 |
|  8335 |           -0.0886901 |         -1.20536  |        -1.46027  |              -1.50134  |            -0.343595 |
|  2724 |            0.273349  |         -1.38064  |         0.162568 |               1.00167  |             0.342509 |

***
## Modeling

Pada projek ini terdapat 3 macam algoritma _Machine Learning_ yang digunakan yaitu:
1. KNN
2. SVM
3. Random Forest

Semua model dilatih dengan menggunakan parameter dasar. Selanjutnya akan dilakukan _tunning_ pada beberapa parameter baik secara konvensional ataupun dengan menggunakan `GridSearchCV`.

### K-Nearest Neighbors (KNN)

KNN adalah salah satu algoritma _Machine Learning_ bertipe _supervised_ dimana dibutuhkannya suatu label. KNN bekerja dengan cara mengidentifikasikan jarak antara 'tetangga', dimana `K` pada KNN adalah jumlah tetangga yang akan dilihat. Tahapan cara kerja KNN adalah sebagai berikut:

1. Menentukan banyaknya jumlah tetangga yang akan dipakai (K).
2. Menghitung jarak antara dokumen testing dan training dengan menggunakan metode distance (umumnya Euclidean atau Manhatan).
3. Mengurutkan data berdasarkan jarak terkecil.
4. Menentukan kelompok testing berdasarkan jumlah tetangga (K) yang telah dipilih.

__Keuntungan__

1. Algortima KNN sederhana dan mudah diimplementasikan.
2. Algoritma multifungsi dapat digunakan untuk kasus klasifikasi, regresi, dan pencarian.
3. Tidak ada periode training, sehingga data baru lebih mudah ditambahkan.

__Kekurangan__

1. Tidak berjalan secara baik pada dataset yang besar. Perlu dilakukannya perhitungan jarak dapat meningkatkan beban komputasi mesin.
2. Tidak berjalan baik pada data berdimensi tinggi. Karena KNN berdasarkan perhitungan jarak, akan lebih sulit melakukannya pada dimensi tinggi.
3. Membutuhkan _scaling_ fitur. Perhitungan jarak dengan data acuan yang berbeda-beda akan membuat hasilnya tidak akurat, sehingga perlu dilakukannya _scaling_/normalisasi data.

Pada proyek ini, pertama-tama KNN akan menggunakan __n_neighbors=10__ tetangga terdekat. Dan setelahnya akan dilakukan proses _tunning_ secara manual untuk mencari nilai K terbaik dari 1-20 tetangga hingga mendapatkan akurasi testing yang paling tinggi. Perhitungan jarak antar tetangga yang dipakai adalah Euclidean distance.

### Support Vector Machine (SVM)

Sama dengan KNN, SVM merupakan metode _supervised learning_. Dimana terdapat dua tipe SVM yang umum digunakan yaitu _Support Vector Classificatioin (SVC)_ dan juga _Support Vector Regression (SVR)_. SVM juga dapat mengatasi masalah klasifikasi dan regresi baik secara _linear_ ataupun _non-linear_

Cara kerja SVM adalah dengan menemukan _hyperplace_ (jalan raya) terbaik dengan memaksimalkan jarak antar kelas sehingga dapat memisahkan titik-titik pada input. SVM dulunya dikenal sebagai _maximum margin classifier_. Ternyata alternatif untuk memperoleh _maximum margin_ adalah dengan mencari _support vector_ . Itulah mengapa nama algoritma ini juga disebut sebagai Support Vector Machine, yakni mesin pencari _support vector_.Gambar berikut menunjukan ilustrasi pada SVM.

![SVM](https://user-images.githubusercontent.com/60245989/201678523-204899d3-af83-4e12-991f-b20719c80124.png)

Gambar 7. Ilustrasi SVM

Salah satu parameter penting pada SVM adalah __C-Penalty Parameter__ dimana:
- Semakin besar __C__ -> Semakin besar penalty terhadap kesalah -> lebih sensitif
- Semakin kecil __C__ -> Semakin toleran terhadap kesalahan -> lebih smooth

Selain itu pada data yang lebih kompleks, data-data dapat ditransformasikan ke dalam dimensi lain dengan mengubah jenis kernelnya. Terdapat parameter lainnya yaitu $\gamma$ kernel coefficient, dimana:

- Semakin besar $\gamma$ -> semakin detail oriented -> lebih sensitif
- Semakin kecil $\gamma$ -> semakin melihat big picture -> lebih smooth

__Keuntungan__

1. SVM bekerja relatif baik ketika terdapat pemisahan yang cukup jelas antar kelas.
2. SVM lebih efektif pada data berdimensi tinggi.
3. SVM relatif lebih efektif secara memori.

__Kekurangan__

1. SVM tidak cocok untuk dataset yang besar.
2. SVM tidak bekerja secara baik ketika dataset terdapat banyak _noise_ contohnya pada target kelas yang overlap.
3. SVM Membutuhkan waktu training yang relatif lebih lama.

Pada proyek ini, pertama-tama SVM akan menggunakan nilai default dimana nilai C=1. Dan setelahnya akan dilakukan proses _tunning_ secara manual untuk mencari nilai C terbaik dari 1-1000 dengan skala pengali 10 hingga mendapatkan akurasi testing yang paling tinggi. Kernel yang digunakan bernilai default.

### Random Forest

Random Forest merupakan salah satu metode dalam _Decision Tree_ yang menggunakan teknik bagging, dimana beberapa model bekerja bersama-sama (_ensemble_) untuk mendapatkan suatu keputusan, sehingga tingkat keberhasilannya akan semakin tinggi. 

_Decision Tree_ sendiri adalah suatu diagram alir yang terdiri dari _ node_ yang berisikan _root_ dan juga _leaf_, dimana masing-masingnya akan memecahkan masalah secara independen lalu menggabungkanya. Berikut adalah ilustrasi dari Random Forest.

![Random Forest](https://user-images.githubusercontent.com/60245989/201685936-7cf555bf-1b51-4e89-9519-18ec8a040a6d.gif)

Gambar 8. Ilustrasi Random Forest (source:  Tensorflow Blog)

Terdapat beberapa parameter pada Random Forest diantaranya:
- n_estimator -> jumlah tress (pohon).
- max_depth -> kedalaman suatu cabang pohon (percabangan/pembelahan pohon).
- min_sample_leaf -> minimal leaf node yang ada.

__Kelebihan__

1. Random Forest berbasis _bagging_ dan menggunakan _ensemble learning_ dimana dapat mengurangi masalah _overfitting_ pada _decision tree_ dan juga mengurangi _variance_ sehingga dapat meningkatkan akurasai.
2. Random Forest dapat menangani _missing value_.
3. Random Forest tidak membutuhkan fitur _scaling_.
4. Random Forest robust terhadap _outliers_.
5. Random Forest memiliki efek yang rendah terhadap _noise_ dan relatif stabil.

__Kekurangan__

1. Random Forest memiliki kompleksitas yang tinggi.
2. Random Forest memiliki training periode yang lama, sehingga membutuhkan mesin komputasi yang lebih baik.

Pada projek ini nilai parameter awal Random Forest Classifier adalah sebagai berikut: n_estimators=5, max_depth=5, random_state=42, n_jobs=-1. 

Setelahnya dilakukan _tunning_ dengan menggunakan metode GridSearchCV. Metode ini digunakan untuk melakukan _tunning_ parameter-parameter algoritma secara otomatis, dan juga ditambah dengan melakukan _k fold cross validation_ untuk mendapatkan parameter terbaik. Kelebihan dan kekurangan dari GridSearch adalah sebagai berikut:

- Input berupa semua kombinasi hyperparameter yang ingin dicoba.
- Menjamin score terbaik dari semua kombinasi.
- Komputasi berat karena semua kombinasi dilakukan.
- Tidak cocok dipakai untuk algoritma dengan banyak hyperparameter.

Pada projek kali ini parameter yang digunakan pada Random Forest adalah sebagai berikut:
- "n_estimators": [25, 50, 100],
- "max_depth": [10, 20, 30],
- "min_samples_leaf": [1, 5, 10]

GridSearch akan secara otomatis menentukan model dengan hyperparameter terbaik setelah juga melakukan _cross validation_.

## Evaluation

Pada proyek ini permasalahan berjenis klasifikasi, sehinga metrik evaluasinya adalah precision, recall, dan F1 score. Metrik ini biasa dikenal dengan sebutan __Confusion Matrix__ Secara matematis metrik evaluasi _Confusion Matrix_ pada _binary classification_ (Yes/No) adalah sebagai berikut

![1_M0Ex70vbOhV9eHKAdk7Ekg](https://user-images.githubusercontent.com/60245989/201802447-b60b9c0c-1fee-4944-9f45-340867d77b3d.png)

Gambar 9. _Confusion Matrix_ [res. stevkarta](https://stevkarta.medium.com/membicarakan-precision-recall-dan-f1-score-e96d81910354)

0 untuk label negatif dan 1 untuk label positif. Istilah-istilah pada _confusion matrix_:
- True Negative (TN): Model memprediksi hasil __Negatif__ dan data sebenarnya adalah __Negatif__.
- True Positive (TP): Model memprediksi hasil __Positif__ dan data sebenarnya adalah __Positif__.
- False Negative (FN): Model memprediksi hasil __Negatif__, namun data sebenarnya adalah __Positif__.
- False Positive (FP): Model memprediksi hasil __Positif__, namun data sebenarnya adalah __Negatif__.

Mendapatkan nilai TN dan TP terbanyak adalah tujuan dari model. Namun, dalam kenyataanya prediksi FN atau FP seringlah terjadi. Pada beberapa kasus, kita dituntut untuk mendapatkan model yang akan lebih toleran pada FN dan dikasus lain akan lebih toleran pada FP. Contohnya pada model prediksi pasin kanker, FP lebih toleran dari pada FN, karena lebih baik pasien terprediksi terkena kanker (meskipun aslinya tidak) dibandingkan pasien diprediksi __tidak__ terkena kanker dan ternyata aslinya pasien mengidap kanker (False Negative).

Pemilihan toleransi antara FP dan FN akan bergantung pada pemilihan tipe error, lebih diutamakan __Precision__ atau __Recall__.

Precision, secara definisi adalah perbandingan antara True Positive (TP) dengan banyaknya data yang diprediksi positif. Atau secara matematis

![0_fD_fCLwvjNnp2Nel](https://user-images.githubusercontent.com/60245989/201803735-3a27c74f-0b5b-4daf-8811-4784fa448560.gif)

Recall, secara definisi adalah perbandingan antara True Positive (TP) dengan banyaknya data yang sebenarnya positif. Atau secara matematis

![0_jnTAutFpHEVqqBHJ](https://user-images.githubusercontent.com/60245989/201803902-798b61bf-92c8-47d3-952a-435880a92ab2.gif)

Jika dilihat dari dua persamaan di atas maka terlihta pada _precision_ semakin kecil nilai False Positive maka nilainya _precision_ akan semakin besar. Begitu juga pada _recall_. Pada contoh kasus di kanker di atas, FN yang kecil lebih baik ketimbang FP sehingga acuan _recall_ bisalah digunakan.

Namun terkadang terdapat dilema dalam memilih _precision_ dan juga _recall_, dan kita perlu score yang sama-sama tinggi. Untuk mendapatkan hasil yang seimbang terdapat metode lain yaitu __F1-Score__ yang secara definisi adalah harmonic mean dari _precision_ dan _recall_. Secara matematis ditulis sebagia berikut

![0_RdOj9EZ6TbVmwWmi](https://user-images.githubusercontent.com/60245989/201804716-a4ab55fb-73ad-4029-95b2-b8a223681043.gif)

Terlebih F1-Score cocok untuk data klasifikasi yang tidak seimbang (_imbalance dataset_). Sehingga F1-Score akan dijadikan acuan pada projek kali ini.

Berikut adalah hasil _Consufion Matrix_ pada setiap model

__KNN__

![KNN_F1](https://user-images.githubusercontent.com/60245989/201805289-18da32a9-4ce7-4017-96f0-073c044d2f09.PNG)

Gambar 10. Confusion Matrix KNN

__SVM__

![SVM_F1](https://user-images.githubusercontent.com/60245989/201805347-c83d6939-7a63-43d6-9902-e6c28593e14e.PNG)

Gambar 11. Confusion Matrix SVM

__Random Forest__

![RF_F1](https://user-images.githubusercontent.com/60245989/201805392-d2f41cbe-76c1-4f95-be59-dfb49459072a.PNG)

Gambar 12. Confusion Matrix Random Forest

__Random Forest dengan GridSearch__

![GSV_F1](https://user-images.githubusercontent.com/60245989/201805430-b437f2b6-fb67-4842-a054-f78a7505f301.PNG)

Gambar 13. Confusion Matrix Random Forest dengan GridSearch


Terlihat bahwa model Random Forest dengan _Hyperparameter_ yang telah ditunning dengan menggunakan GridSearch memiliki nilai F1-Score yang paling tinggi. Sehingga model inilah yang akan digunakan dalam proyek ini.

Selanjutnya prediksi dengan menggunakan Real Data digunakan pada 10 data acak. Hasilnya adalah sebagai berikut

Tabel 7. Prediksi Real Data

|       |   y_true |   KNN |   SVM |   RandomForest |   GridSearch |
|------:|---------:|------:|------:|---------------:|-------------:|
|  4278 |        0 |     0 |     0 |              0 |            0 |
|   244 |        1 |     1 |     1 |              0 |            1 |
| 10646 |        0 |     0 |     0 |              0 |            0 |
|  1818 |        1 |     1 |     1 |              0 |            1 |
|  2950 |        0 |     0 |     0 |              0 |            0 |
|  9241 |        0 |     0 |     0 |              0 |            0 |
|  1153 |        1 |     1 |     1 |              1 |            1 |
|  7535 |        0 |     0 |     0 |              0 |            0 |
| 13541 |        0 |     0 |     0 |              0 |            0 |
| 11167 |        0 |     0 |     0 |              0 |            0 |

Random Forest basic memiliki kekeliruan dalam memprediksi nilai 1 (True), ini selaras dengan hasil pada confusion matrix, dimana F1-Scorenya hanya sebesar 0.81

### Feature Importance

Tujuan akhir dari proyek ini bukan hanya menemukan model dengan tingkat akurasi terbaik. Tapi mengetahui fitur apa yang paling berpengaruh terhadap perginya seorang karyawan dari suatu perusahaan. Setelah menentukan model terbaik yaitu Random Forest dengan GridSearch, hal selanjutnya adalah mengetahui fitur yang paling penting untuk direkomendasikan kepada bagian HR. Random Forest sudah memiliki data fitur pada bagian `feature_importances_` kita dapat langsung memanggilnya. Hasilnya adalah sebagai berikut

![Feature Importance](https://user-images.githubusercontent.com/60245989/201806521-fce71354-d5ef-435a-b971-250defa1169b.png)

Gambar 14. Feature Importance

Dari data diketahui __satisfaction_level__ lah nilai yang sangat mempengaruhi untuk pengunduran diri karyawan dimana hasil ini sesuai dengan EDA pada bagian sebelumnya. Sehingga HR perlu  menjaga kepuasaan setiap karyawannya. Selain itu banyaknya projek yang diterima, dan jumlah jam kerja setiap bulannya, waktu yang dihabiskan di perusahaan, dan evaluasi terakhir juga mempengaruhi perginya seorang karyawan dari suatu perusahaan.

***
## Kesimpulan

Dari data HR Analytics didapatkan model terbaik adalah Random Forest dengan melakukan _tunning_ parameter menggunakan GridSearch dengan nilai F1-Score pada True sebesar 0.98. Selain itu juga Random Forest dapat mengetahui _Feature Importance_ untuk mengetahui fitur paling penting yang mempengaruhi prediksi. Dari Proses ini diketahui bahwa `satisfication_level` atau kepuasaan seorang karyawan terhadap suatu perusahaan adalah faktor terbesar yang menentukan seorang karyawan dapat pergi atau tetap tingal di suatu perusahaan. Selanjutnya data-data ini dapat digunakan HR sebagai acuan untuk menjaga kestabilan perusahaan.


## Referensi

[1] [Work Institue. Retention Report 2019. Report. 2019.](https://info.workinstitute.com/hubfs/2019%20Retention%20Report/Work%20Institute%202019%20Retention%20Report%20final-1.pdf)

[2] [Al-suraihi, Walid Abdullah; et.al. Employee Turnover: Causes, Importance and Retention Strategies. European Journal of Business Management and Research, Vol. 6, Issue 3, June 2021.](https://www.researchgate.net/publication/352390912_Employee_Turnover_Causes_Importance_and_Retention_Strategies)

[3] [Basariya, s. Rabiyathul; Ahmed, Ramyar Rzgar. A STUDY ON ATTRITION – TURNOVER INTENTIONS OF EMPLOYEES. International Journal of Civil Engineering and Technology, Volume 10, Issue 01, January 2019, pp. 2594–2601.](https://www.researchgate.net/publication/333104347_A_STUDY_ON_ATTRITION_-_TURNOVER_INTENTIONS_OF_EMPLOYEES)
