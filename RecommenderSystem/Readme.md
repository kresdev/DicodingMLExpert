# Laporan Proyek Machine Learning - Kresna Devara
***
# Movie Recommendation

***
## Domain Proyek

Pada pekermbangan jaman saat ini, dimana dunia entertainment berkembang dengan saat pesat, berbagai hiburan seperti: permainan (game) dan juga film banyak diminati oleh berbagai kalangan. Saat ini layangan streaming berbayar sangatlah menjamur. Selain kualitas film yang disuguhkan salah satu faktor utama lainnya adalah relevansi antara pengguna dan juga konten dari film tersebut. Layanan streaming berbayar memiliki berbagai macam demografi pengguna, baik itu dengan usia anak-anak hinga sampai lansia. Umumnya para pengguna yang sudah lansia dan juga anak-anak tidak dapat memilih filmnya secara lebih mudah, bahkan pengguna dengan usia produktifpun terkadang masih kesusahan dalam menentukan film pilihannya. Sehingga rekomendasi film merupakan salah satu yang dapat meningkatkan tingkat minat pengguna dalam melakukan layanan streaming berbayar.

Penelitian telah menunjukan rekomendasi film (_movie recommendation_) merupakan faktor penting dalam sebuah aplikasi streaming berbayar. Berbagai peneliti mencoba menemukan teknik-teknik yang efektif untuk melakukan rekomendasi film, seperti yang dilakukan oleh Halder, et.al dengan melakukan _Movie Recommendation System Based on Movie Swarm_ [1](https://ieeexplore.ieee.org/document/6382910), ataupun yang berbasis Content Based Filtering seperti yang dilakukan oleh Reddy, et.al [2](https://www.researchgate.net/publication/331966843_Content-Based_Movie_Recommendation_System_Using_Genre_Correlation) ataupun yang berbasis Collaborative Filtering yang dilakukan oleh Schafer, et.al [3](https://www.researchgate.net/publication/200121027_Collaborative_Filtering_Recommender_Systems). Berbagai teknik ini memiliki tujuan utama yang pada akhirnya adalah dapat meningkatan nilai bisnis dari suatu layanan berbayar.

Permasalahan ini merupakan salah satu hal terberat bagi suatu perusahaan layanan streaming berbayar agar dapat memberikan rekomendasi film terbaik kepada penggunanya. Semakin relevan rekomendasi yang diberikan, penggunaan aplikasi akan semakin meningkat sehingga dapat meningkatnya juga jumlah pengguna secara jangka panjang. Dengan menggunakan _Machine Learning_ masalah ini dapat dengan lebih baik diselesaikan. Teknik-teknik dengan berbasis Content ataupun dengan menggunakan Collaborative Filtering dapat digunakan sehingga pengguna baru ataupun pengguna yang bahkan belum pernah menyaksikan film tersebut bisa mendapatkan rekomendasi yang relevan sesuai _behaviour_.

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
Sumber | [Kaggle Dataset - Movie Lens Small Latest Datasets](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset)
Author | SHUBHAM MEHTA
Size | 3.3 MB

Data dengan nama `movies.csv` berisikan 9742 baris dan 3 kolom. Dimana 3 kolom tersebut menjelaskan beberapa fitur yaitu:
- MovieId: Unique id dari film
- Title: Judul dari film
- Genres: Tipe genre dari film

Sedangkan data dengan nama `ratings.csv` berisikan 100836 baris dan 4 kolom, dimana fiturnya adalah:
- UserId: Unique id dari user
- MovieId: Unique id dari film
- rating: Penilaian film dari user
- timestamp: waktu melakukan submit survey

__Persiapan Data dan Exploratory Data Analysis (EDA)__
1. Melakukan _import_ data dari csv dengan menggunakan Pandas.
2. Melakukan pengecekan data secara umum, dan juga data yang kosong (null dan NA).
3. Melakukan _Univariate Analysis_ pada jenis genre dan juga jumlah rating.

### 1. Import data dengan Pandas

Pada proyek ini library pandas digunakan, dimana data yang berjenis CSV diubah kedalam bentuk Pandas Data Frame. 5 data awalnya movies dan ratings adalah sebagai berikut

Tabel 2. 5 Data awal Movies Dataset

|  # |   movieId | title                              | genres                                      |
|---:|----------:|:-----------------------------------|:--------------------------------------------|
|  0 |         1 | Toy Story (1995)                   | Adventure|Animation|Children|Comedy|Fantasy |
|  1 |         2 | Jumanji (1995)                     | Adventure|Children|Fantasy                  |
|  2 |         3 | Grumpier Old Men (1995)            | Comedy|Romance                              |
|  3 |         4 | Waiting to Exhale (1995)           | Comedy|Drama|Romance                        |
|  4 |         5 | Father of the Bride Part II (1995) | Comedy                                      |

Tabel 3. 5 Data awal Ratings Dataset

|  # |   userId |   movieId |   rating |   timestamp |
|---:|---------:|----------:|---------:|------------:|
|  0 |        1 |         1 |        4 | 9.64983e+08 |
|  1 |        1 |         3 |        4 | 9.64981e+08 |
|  2 |        1 |         6 |        4 | 9.64982e+08 |
|  3 |        1 |        47 |        5 | 9.64984e+08 |
|  4 |        1 |        50 |        5 | 9.64983e+08 |

### 2. Pengencekan Data Secara Umum

dengan menggunakan `df.info()` kita bisa mengetahui jenis-jenis dari data yaitu:

Tabel 4. Dataset Movies Info

|   # |   Column  |   Non-Null Count    |   Dtype   |
|----:|----------:|--------------------:|----------:|
|   0 | MovieId   | 9742 non-null       | int64     |
|   1 | title     | 9742 non-null       | object    |
|   2 | genres    | 9742 non-null       | object    |

Tabel 5. Dataset Ratings Info

|   # |   Column  |   Non-Null Count    |   Dtype   |
|----:|----------:|--------------------:|----------:|
|   0 | userId    | 100836 non-null     | int64     |
|   1 | MovieId   | 100836 non-null     | int64     |
|   2 | rating    | 100836 non-null     | float64   |
|   3 | timestamp | 100836 non-null     | int64     |

Jika dilihat lebih detail dengan menggunakan `df.nunique()` maka data data tersebut menjadi seperti berikut:

Tabel 6. Unique Value Movies Dataset

| Column  |Unique|
|:--------|-----:|
| movieId | 9742 |
| title   | 9737 |
| genres  |  951 |

Tabel 7. Unique Value Ratings Dataset

| Column    | Unique|
|:----------|------:|
| userId    |   610 |
| movieId   |  9724 |
| rating    |    10 |
| timestamp | 85043 |


Dari Tabel 6 terlihat bahwa terdapat MovieId dengan judul yang sama sebanyak 5 Movie, Nantinya data-data dengan title yang sama namun Id yang berbeda akan didrop (dibuang) sehingga tidak membingungkan mesin. 

Selanjutnya untuk melakukan pengecekan apakah terdapat data yang kosong dengan menggunakan `isnull` dan juga `isna`:

Total Null movies dataframe: 0
Total NA movies dataframe: 0
Total Null ratings dataframe: 0
Total NA ratings dataframe: 0

Tidak terdapat data yang kosong. Data-data statistik pada dataset ratings adalah sebagai berikut:

Tabel 8. Statistik Data Movies Dataset

|       |     userId |   movieId |       rating |        timestamp |
|:------|-----------:|----------:|-------------:|-----------------:|
| count | 100836     |  100836   | 100836       | 100836           |
| mean  |    326.128 |   19435.3 |      3.50156 |      1.20595e+09 |
| std   |    182.618 |   35531   |      1.04253 |      2.16261e+08 |
| min   |      1     |       1   |      0.5     |      8.28125e+08 |
| 25%   |    177     |    1199   |      3       |      1.01912e+09 |
| 50%   |    325     |    2991   |      3.5     |      1.18609e+09 |
| 75%   |    477     |    8122   |      4       |      1.43599e+09 |
| max   |    610     |  193609   |      5       |      1.5378e+09  |

### 3. _Univariate Analysis_

Pada analisa _univariate_ akan dilihat total dari masing-masing genres pada movies dan jumlah mayoritas rating pada dataset ratings.

#### 3.1 _Univariate Analysis_ Fitur Genre

Kolom genres pada dataset movies tidak hanya terdiri dari 1 jenis genres, tapi memiliki banyak genres sekaligus. Maka dari itu, genre-genre akan dihitung secara terpisah, sehingga memungkinkan total genre lebih dari jumlah filmnya (karena 1 film bisa lebih dari 1 genre). Setelah dilakukan filter didapatkan bahwa jumlah genre adalah sebagai berikut

![download](https://user-images.githubusercontent.com/60245989/204174201-a50b8a5b-5bb1-44b1-bc93-3b0ea67d392d.png)

Gambar 1. _Univariate Analysis_ Jumlah Genre pada Film

Dari data-data tersebut dapat diambil analisa bahwa:
1. Film yang terdapat pada dataset mayoritas memiliki genre __Drama__ dan juga __Comedy__ .
2. Terdapat film yang tidak memiliki genre (no genres listed). Selanjutnya film-film tanpa genres akan dibuang.

### 3.2 _Univariate Analysis_ Jumlah Rating

Pada dataset ratings dilakukan analisa untuk mengetahui jumlah rating mayoritas dan juga minoritas yang diberikan oleh user. Data-datanya adalah sebagai berikut:

![download](https://user-images.githubusercontent.com/60245989/204174661-6bb6f2d9-23ea-4ad8-bdbc-a7bb519398da.png)

Gambar 2. _Univariate Analysis_ Jumlah Rating

Dari data-data tersebut dapat diambil analisa bahwa:
1. Rating yang paling banyak diberikan oleh user adalah 4.0.
2. Rating 0.5 memiliki jumlah rating paling sedikit.

***
## _Data Preparation_

Pada tahap ini akan dilakukan _preprocessing_ terhadap data yang akan dimasukkan ke dalam model _Machine Learning_, ada beberapa tahapan yang dilakukan, yaitu:
1. Membuang data yang duplikasi (pada movie dataset)
2. Membuang data yang tidak diperlukan. Pada movie dataset yang tidak memiliki genre, pada rating dataset kolom time stamp.
3. Menggabungkan movie dataset dengan rating dataset (merge dataset) untuk melakukan _collaborative filtering_.
4. Melakukan tokenisasi dan membuat bank token untuk _content-based filtering_.
5. Melakukan dataset split untuk mendapat data training dan testing pada merge dataset untuk _collaborative filtering_.

### Membuang duplikasi dataset

Pada bagian sebelumnya telah dijelaskan bahwa terdapat duplikasi pada movies dataset. Dimana film dengan judul yang sama memiliki movieId yang berbeda. Data-datanya adalah sebagai berikut:

Tabel 9. Film Duplikat dengan Judul yang Sama

|      |   movieId | title                                  | genres                              |
|-----:|----------:|:---------------------------------------|:------------------------------------|
| 5601 |     26958 | Emma (1996)                            | Romance                             |
| 6932 |     64997 | War of the Worlds (2005)               | Action|Sci-Fi                       |
| 9106 |    144606 | Confessions of a Dangerous Mind (2002) | Comedy|Crime|Drama|Romance|Thriller |
| 9135 |    147002 | Eros (2004)                            | Drama|Romance                       |
| 9468 |    168358 | Saturn 3 (1980)                        | Sci-Fi|Thriller                     |

Data-data ini akan dibuang agar tidak membingungkan mesin.

### Membuang Data yang tidak diperlukan

Diketahui dari proses sebelumnya bahwa, tidak ada data yang kosong (Null/NA). Namun terdapat film yang memiliki Genre **(not listed)** sehingga film dengan genre tersebut akan didrop, timestamp pada dataframe rating juga akan didrop (karena tidak digunakan). Sehingga total dari dataset movies menjadi berikut

Tabel 10. Total Movies Dataset Setelah Dropout

| Column  |Total |
|:--------|-----:|
| movieId | 9703 |
| title   | 9703 |
| genres  |  9703 |

### Merge Dataset

Untuk melakukan _Collaborative Filtering_ perlu dilakukannya kombinasi (merge) antara dataset movie dan juga dataset rating. Data datanya adalah sebagai berikut:

Tabel 11. Merge Dataset

|    |   userId |   movieId |   rating | title                       | genres                                      |
|---:|---------:|----------:|---------:|:----------------------------|:--------------------------------------------|
|  0 |        1 |         1 |        4 | Toy Story (1995)            | Adventure|Animation|Children|Comedy|Fantasy |
|  1 |        1 |         3 |        4 | Grumpier Old Men (1995)     | Comedy|Romance                              |
|  2 |        1 |         6 |        4 | Heat (1995)                 | Action|Crime|Thriller                       |
|  3 |        1 |        47 |        5 | Seven (a.k.a. Se7en) (1995) | Mystery|Thriller                            |
|  4 |        1 |        50 |        5 | Usual Suspects, The (1995)  | Crime|Mystery|Thriller                      |

### Tokenisasi dan Bank Token untuk Content Based Filtering

Pada content based filtering kita akan melakukan rekomendasi berdasarkan kemiripan genres.Library sklearn `CountVectorizer` dan `word_tokenize` digunakan untuk melakukan ekstrasi fitur dan pembuatan token. Selanjutnya semua genres pada movie dataset akan difit untuk mendapatkan bank token.

### Dataset Split untuk Collaborative Filtering

Merge dataset yang telah didapatkan pada proses sebelumnya selanjutnya akan dipisahkan data training dan juga data testnya. Library `surprise` digunakan pada fase ini. Berdasarkan EDA pada sub bab sebelumnya rating yang akan digunakan berskala 0.5 sampai dengan 5. Testsize sebesar 0.3 dari total data dan random_state yang digunakan adalah 42

`data = Dataset.load_from_df(df_combine_new, Reader(rating_scale=(0.5, 5)))`
`trainset, testset = train_test_split(data, test_size=0.3, random_state=42)`

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
