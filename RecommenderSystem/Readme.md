# Laporan Proyek Machine Learning - Kresna Devara
***
# Movie Recommendation

***
## Domain Proyek

Pada pekermbangan jaman saat ini, dimana dunia entertainment berkembang dengan saat pesat, berbagai hiburan seperti: permainan (game) dan juga film banyak diminati oleh berbagai kalangan. Saat ini layangan streaming berbayar sangatlah menjamur. Selain kualitas film yang disuguhkan salah satu faktor utama lainnya adalah relevansi antara pengguna dan juga konten dari film tersebut. Layanan streaming berbayar memiliki berbagai macam demografi pengguna, baik itu dengan usia anak-anak hinga sampai lansia. Umumnya para pengguna yang sudah lansia dan juga anak-anak tidak dapat memilih filmnya secara lebih mudah, bahkan pengguna dengan usia produktifpun terkadang masih kesusahan dalam menentukan film pilihannya. Sehingga rekomendasi film merupakan salah satu yang dapat meningkatkan tingkat minat pengguna dalam melakukan layanan streaming berbayar.

Penelitian telah menunjukan rekomendasi film (_movie recommendation_) merupakan faktor penting dalam sebuah aplikasi streaming berbayar. Berbagai peneliti mencoba menemukan teknik-teknik yang efektif untuk melakukan rekomendasi film, seperti yang dilakukan oleh Halder, et.al dengan melakukan _Movie Recommendation System Based on Movie Swarm_ [[1]](https://ieeexplore.ieee.org/document/6382910), ataupun yang berbasis Content Based Filtering seperti yang dilakukan oleh Reddy, et.al [[2]](https://www.researchgate.net/publication/331966843_Content-Based_Movie_Recommendation_System_Using_Genre_Correlation) ataupun yang berbasis Collaborative Filtering yang dilakukan oleh Schafer, et.al [[3]](https://www.researchgate.net/publication/200121027_Collaborative_Filtering_Recommender_Systems). Berbagai teknik ini memiliki tujuan utama yang pada akhirnya adalah dapat meningkatan nilai bisnis dari suatu layanan berbayar.

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
|  0 |         1 | Toy Story (1995)                   | Adventure;Animation;Children;Comedy;Fantasy |
|  1 |         2 | Jumanji (1995)                     | Adventure;Children;Fantasy                  |
|  2 |         3 | Grumpier Old Men (1995)            | Comedy;Romance                              |
|  3 |         4 | Waiting to Exhale (1995)           | Comedy;Drama;Romance                        |
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
| 6932 |     64997 | War of the Worlds (2005)               | Action;Sci-Fi                       |
| 9106 |    144606 | Confessions of a Dangerous Mind (2002) | Comedy;Crime;Drama;Romance;Thriller |
| 9135 |    147002 | Eros (2004)                            | Drama;Romance                       |
| 9468 |    168358 | Saturn 3 (1980)                        | Sci-Fi;Thriller                     |

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
|  0 |        1 |         1 |        4 | Toy Story (1995)            | Adventure;Animation;Children;Comedy;Fantasy |
|  1 |        1 |         3 |        4 | Grumpier Old Men (1995)     | Comedy;Romance                              |
|  2 |        1 |         6 |        4 | Heat (1995)                 | Action;Crime;Thriller                       |
|  3 |        1 |        47 |        5 | Seven (a.k.a. Se7en) (1995) | Mystery;Thriller                            |
|  4 |        1 |        50 |        5 | Usual Suspects, The (1995)  | Crime;Mystery;Thriller                      |

### Tokenisasi dan Bank Token untuk Content Based Filtering

Pada content based filtering kita akan melakukan rekomendasi berdasarkan kemiripan genres. Library sklearn `CountVectorizer` dan `word_tokenize` digunakan untuk melakukan ekstrasi fitur dan pembuatan token berbasis Bag of Words. Selanjutnya semua genres pada movie dataset akan difit untuk mendapatkan bank token.

### Dataset Split untuk Collaborative Filtering

Merge dataset yang telah didapatkan pada proses sebelumnya selanjutnya akan dipisahkan data training dan juga data testnya. Library `surprise` digunakan pada fase ini. Berdasarkan EDA pada sub bab sebelumnya rating yang akan digunakan berskala 0.5 sampai dengan 5. Testsize sebesar 0.3 dari total data dan random_state yang digunakan adalah 42

`data = Dataset.load_from_df(df_combine_new, Reader(rating_scale=(0.5, 5)))`

`trainset, testset = train_test_split(data, test_size=0.3, random_state=42)`

***
## Modeling

Pada projek ini terdapat 2 macam _Recommender System_ yang digunakan yaitu:
1. Content Based Filtering
2. Collaborative Filtering

### Content-Based Filtering

Content based filtering menggunakan fitur dari item untuk merekomendasikan item yang mirip yang user sukai berdasarkan aksi sebelumnya atau dari feedback pengguna. Content based filtering bertujuan untuk memberikan rekomendasi dari kesamaan fitur yang didapat dari pengguna. Contohnya ketika menonton suatu film pada layanan streaming tertentu, pengguna bisa yang terbiasa menonton film dengan tema horor untuk selanjutnya ketika membuka aplikasi tersebut akan direkomendasikan film-film bertema horor.

Pada penerapannya biasanya content based filtering akan menggunakan perhitungan vektor, dimana item-item yang akan dilakukan perhitungan sebelumnya akan diubah kedalam bentuk token baik itu dengan Bag of Words ataupun TFIDF. Langkah-langkahnya adalah sebagai berikut:
1. Tentukan fitur yang akan digunakan pada dataset.
2. Ubah fitur menjadi bentuk token (contohnya dengan menggunakan CountVectorizer menjadi Bag of Words).
3. Membuat bank token yang berisikan semua kumpulan token pada fitur.
4. Melakukan perhitungan vektor dengan menggunakan similarity formula seperti contohnya cosine similarity.
5. Mencari top-n rekomendasi berdasarkan kemiripan dari cosine similarity.

__Keuntungan__

1. Model tidak membutuhkan data dari user lainnya, karena rekomendasi spesifik terhadap user tersebut. Hal ini membuat lebih mudah dalam melakukan scale up dengan jumlah user yang lebih banyak.
2. Model dapat menangkap minat khusus pengguna, dan apat merekomendasikan item khusus yang sangat sedikit diminati oleh pengguna lain.

__Kekurangan__

1. Model dibuat berdasarkan ketrampilan proses dari engineer, sehingga membutuhkan domain knowledge yang lebih luas untuk mendapatkan rekomendasi yang sesuai.
2. Model hanya bisa membuat rekomendasi berdasarkan user yang ada. Sehingga dengan user yang terbatas model tidak mampu bekerja secara maksimal.

Pada proyek ini setelah dilakukan pembuatan token pada data preparation. Selanjutnya akan dipilih 1 film yaitu Jumanji dengan genres Adventure|Children|Fantasy. Data genres akan diubah kedalam bentuk token dengan menggunakan class Bag of Words yang telah dibuat sebelumnya.

`code = bow.transform([content])`

selanjutnya hasil encode token akan dibandingkan dengan bank token dengan menggunakan cosine similarity

`dist = cosine_distances(code, bank)` 

dan akan diambil hasil 10 rekomendasi termirip yang didapatkan dari cosine similarity, 

Tabel 12. Rekomendasi Content Based Filtering

|      |   movieId | title                                                                                          | genres                     |
|-----:|----------:|:-----------------------------------------------------------------------------------------------|:---------------------------|
|   53 |        60 | Indian in the Cupboard, The (1995)                                                             | Adventure;Children;Fantasy |
| 7476 |     82152 | Beastly (2011)                                                                                 | Drama;Fantasy;Romance      |
| 8228 |    104017 | 3 dev adam (Three Giant Men) (1973)                                                            | Action;Adventure;Sci-Fi    |
| 9275 |    157312 | The Boss (2016)                                                                                | Comedy                     |
| 6628 |     56169 | Awake (2007)                                                                                   | Drama;Thriller             |
|  109 |       126 | NeverEnding Story III, The (1994)                                                              | Adventure;Children;Fantasy |
| 1556 |      2093 | Return to Oz (1985)                                                                            | Adventure;Children;Fantasy |
| 3574 |      4896 | Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) | Adventure;Children;Fantasy |
| 1799 |      2399 | Santa Claus: The Movie (1985)                                                                  | Adventure;Children;Fantasy |
| 1618 |      2162 | NeverEnding Story II: The Next Chapter, The (1990)                                             | Adventure;Children;Fantasy |

Dari 10 rekomendasi yang diberikan. terdapat 8 film dengan genre yang hampir sama. Hanya 2 film saja yang memiliki genre yang berbeda.

### Collaborative Filtering

Collaborative filtering memanfaatkan kesamaan ketertarikan/perilaku antar pengguna dan item. Sehingga pengguna bisa mendapatkan rekomendasi berdasarkan kesamaan dari pengguna lainnya, meskipun pengguna tersebut belum pernah memilih item tersebut. Collaborative filtering umumnya terbagi menjadi dua yaitu:

- User Based Collaborative Filtering (UB-CF)
- Item Based Collaborative Filtering (IB-CF)

User Based merekomendasikan item berdasarkan pengguna yang mirip dengan dirinya. Contohnya berdasarkan historynya user A dan user B memiliki ketertarikan yang sama dengan film-film bertema pahlawan dan action, user B sudah menonton film spiderman, sehingga sistem merekomendasi A untuk menonton film spiderman karena memiliki kemiripan yang sama dengan user B.

Item Based merekomendasikan kesamaan item yang berkeaitan dengan pengguna dan item lainnya. Contohnya jika seorang pengguna menyukai suatu grup idol K-POP, sistem akan merekomendasikan barang-barang yang berkaitan dengan idol K-POP tersebut seperti baju, foto, dan marchandise lainnya.

Tahapan-tahapan dalam collaborative filltering pada proyek ini adalah sebagai berikut:
1. Menggabungkan dataset user dan juga content yang akan diproses.
2. Memisahkan antara trainset dan juga test set.
3. Melakukan training dengan menggukana SVD.
4. Melakukan rekomendasi (top-n) film-film yang belum pernah ditonton oleh user berdasarkan model SVD.

__Keuntungan__

1. Tidak memerlukan domain knowledge. Karena Embedding akan dilakukan otomatis oleh Machine Learning.
2. Model dapat membuat pengguna menemukan minat baru. Model ML mungkin saja tidak mengetahui secara pasti rekomendasinya, namun berdasarkan kesamaan pengguna bisa jadi model ML memberikan ketertarikan baru pada pengguna.
3. Tidak membutuhkan detail mendalam dari suatu produk.

__Kekurangan__

1. Keterbatasan data. Akan sulit melakukan rekomendasi kepada user yang baru.
2. Skalabilitas yang terbatas. Dengan bertambahnya jumlah pengguna peningkatkan perhitungan juga akan berdampak.
3. Sinonim dan keberagamaan. Collaborative filtering cenderung lebih sulit dalam membedakan sinonim dan juga kebaragaman dalam suatu item.

Pada proyek ini model Machine Learning yang digunakan adalah SVD. setelah melakukan dataset splitting selanjutnya model akan ditrain menggunakan training dataset. Dan model akan melakukan prediksi film-film yang belum pernah ditonton oleh user yang mungkin diminati oleh user. Contoh hasilnya adalah sebagai berikut:

Tabel 13. Rekomendasi Collaborative Filtering

|      |   movieId | title                                                                       | genres                      |   pred_score |
|-----:|----------:|:----------------------------------------------------------------------------|:----------------------------|-------------:|
|  232 |       318 | Shawshank Redemption, The (1994)                                            | Crime;Drama                 |      4.39145 |
| 2395 |      1204 | Lawrence of Arabia (1962)                                                   | Adventure;Drama;War         |      4.37225 |
|  722 |       750 | Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964) | Comedy;War                  |      4.31638 |
| 1027 |       858 | Godfather, The (1972)                                                       | Crime;Drama                 |      4.25985 |
| 1110 |      1276 | Cool Hand Luke (1967)                                                       | Drama                       |      4.2485  |
| 1184 |      3275 | Boondock Saints, The (2000)                                                 | Action;Crime;Drama;Thriller |      4.24442 |
|  332 |       904 | Rear Window (1954)                                                          | Mystery;Thriller            |      4.24153 |
|   74 |      1213 | Goodfellas (1990)                                                           | Crime;Drama                 |      4.24099 |
|  192 |      2959 | Fight Club (1999)                                                           | Action;Crime;Drama;Thriller |      4.24052 |
| 2158 |      3451 | Guess Who's Coming to Dinner (1967)                                         | Drama                       |      4.24031 |

Data di atas merupakan hasil rekomendasi dari user 0. Dapat dilihat user 0 direkomendasikan film-film dengan tema drama, action dan adventure.

***
## Evaluation

Pada proyek ini terdapat 2 jenis recommender system yang digunakan. Kedua hal tersebut tidak bisa dievaluasi dengan menggunakan metrik yang sama. Pada Content based filtering, metrik evaluasi yang digunakan adalah presisi sedangkan pada collaborative filtering metrik evaluasi yang digunakaan adalah Root Mean Square Error (RMSE) dan juga Mean Absolute Error (MAE).

### Content Based Filtering

Untuk mengetahui presisi dari rekomendasi formula yang digunakan adalah sebagai berikut:

![dos_819311f78d87da1e0fd8660171fa58e620211012160253 (1)](https://user-images.githubusercontent.com/60245989/204700359-da5cc590-2940-4780-a97b-f66cca6b215c.png)

Dimana jumlah rekomendasi yang relevan akan dibagi dengan total jumlah yang direkomendasikan. Pada proyek ini akan dilakukan rekomendasi sebanyak 20 buah, dan akan dilihat sebarap banyak rekomendasi yang masih relevan terhadap film yang diberikan. Film yang akan dicari rekomendasinya adalah jumanji dengan genres Adventure|Children|Fantasy rekomendasinya adalah sebagai berikut:

Tabel 14. Top 20 Rekomendasi Jumanji

|      |   movieId | title                                                                                          | genres                                |
|-----:|----------:|:-----------------------------------------------------------------------------------------------|:--------------------------------------|
|   53 |        60 | Indian in the Cupboard, The (1995)                                                             | Adventure;Children;Fantasy            |
| 7476 |     82152 | Beastly (2011)                                                                                 | Drama;Fantasy;Romance                 |
| 8228 |    104017 | 3 dev adam (Three Giant Men) (1973)                                                            | Action;Adventure;Sci-Fi               |
| 9275 |    157312 | The Boss (2016)                                                                                | Comedy                                |
| 6628 |     56169 | Awake (2007)                                                                                   | Drama;Thriller                        |
|  109 |       126 | NeverEnding Story III, The (1994)                                                              | Adventure;Children;Fantasy            |
| 1556 |      2093 | Return to Oz (1985)                                                                            | Adventure;Children;Fantasy            |
| 3574 |      4896 | Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001) | Adventure;Children;Fantasy            |
| 1799 |      2399 | Santa Claus: The Movie (1985)                                                                  | Adventure;Children;Fantasy            |
| 1618 |      2162 | NeverEnding Story II: The Next Chapter, The (1990)                                             | Adventure;Children;Fantasy            |
| 6388 |     50514 | After the Wedding (Efter brylluppet) (2006)                                                    | Drama                                 |
| 9531 |    172253 | The Night Before (1988)                                                                        | Comedy                                |
| 1514 |      2043 | Darby O'Gill and the Little People (1959)                                                      | Adventure;Children;Fantasy            |
| 1617 |      2161 | NeverEnding Story, The (1984)                                                                  | Adventure;Children;Fantasy            |
| 7424 |     80693 | It's Kind of a Funny Story (2010)                                                              | Comedy;Drama                          |
|    1 |         2 | Jumanji (1995)                                                                                 | Adventure;Children;Fantasy            |
| 6654 |     56908 | Dedication (2007)                                                                              | Comedy;Drama;Romance                  |
| 8794 |    130050 | Digging Up the Marrow (2014)                                                                   | Drama;Fantasy;Horror;Mystery;Thriller |
| 8638 |    119155 | Night at the Museum: Secret of the Tomb (2014)                                                 | Adventure;Children;Comedy;Fantasy     |
|  767 |      1009 | Escape to Witch Mountain (1975)                                                                | Adventure;Children;Fantasy            |

Dapat dilihat bahwa genres yang tidak relevan terdapat 6 buah yaitu:

- The Boss (2016) = Comedy
- Awake (2007) = Drama;Thriller
- After the Wedding (Efter brylluppet) (2006) = Drama
- The Night Before (1988) = Comedy
- It's Kind of a Funny Story (2010) = Comedy;Dram
- Dedication (2007) = Comedy;Drama;Romance

Sehingga Presisinya adalah `14/20= 0.7 = 70%`

### Collaborative Filtering

Metrik evaluasi yang digunakan pada Collaborative filtering adalah RMSE dan juga MAE. Dimana kedua formula tersebut membandingan nilai actual dan juga nilai prediksi untuk mengetahui errornya. Formulanya adalah sebagai berikut

![download](https://user-images.githubusercontent.com/60245989/204703396-a1d9c85b-d229-4042-a530-f573914755a0.png)

![download](https://user-images.githubusercontent.com/60245989/204703412-0da899e0-ed44-43ff-be39-f56f854a9445.png)

Keterangan:
- y: nilai aktual
- y^: nilai prediksi
- n: jumlah total sampel

Pada collaborative filtering sebelumnya data telah dibagi menjadi data training dan juga data test. Untuk melakukan evaluasi metrik data test akan dilakukan. Hasil datanya dalah sebagai berikut:

Tabel 15. 5 Data perbandingan actual dan prediksi score pada Collaborative Filtering

|    |   userId |   movieId |   Rating_actual |   Rating_predictions |
|---:|---------:|----------:|----------------:|---------------------:|
|  0 |      217 |      1287 |             3   |              2.90435 |
|  1 |      594 |      7032 |             4   |              3.96098 |
|  2 |      117 |       697 |             3   |              3.20588 |
|  3 |      610 |     43928 |             2   |              3.11187 |
|  4 |      414 |      3986 |             1.5 |              2.9064  |

Setelah dilakukan perhitungan untuk semua data hasilnya adalah sebagai berikut: 

- MAE   : 0.678
- MSE   : 0.775
- RMSE  : 0.880

Dari data di atas diketahui bahwa model SVD memiliki nilai MAE 0.678 dan RMSE 0.880, dimana nilai tersebut sudah cukup baik untuk sistem rekomendasi berbasis Machine Learning. Untuk dapat meningkatkan performa model berbasis Neural Network dapat dilakukan. 

## Referensi

[1] [Halder, Sajal; et.al. Movie Recommendation System Based on Movie Swarm. Second International Conference on Cloud and Green Computing. 2012.](https://ieeexplore.ieee.org/document/6382910)

[2] [Reddy; et.al. Content-Based Movie Recommendation System Using Genre Correlation. Smart Innovation, Systems and Technologies. 2018.](https://www.researchgate.net/publication/331966843_Content-Based_Movie_Recommendation_System_Using_Genre_Correlation)

[3] [Schafer, Ben; et.al. Collaborative Filtering Recommender Systems. The Adaptive Web, LNCS 4321, pp. 291 – 324. 2007.](https://www.researchgate.net/publication/200121027_Collaborative_Filtering_Recommender_Systems)
