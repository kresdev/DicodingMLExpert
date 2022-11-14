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
- Menggunakan model _Machine Learning_ umum seperti **KNN** dan juga **Random Forest**, dan juga melakukan _hyper parameter tunning_ pada masing-masing model.
- Mengetahui metrik evaluasi apa yang paling tepat untuk mendapatkan akurasi yang terbaik.
- Melakukan _Feature Importance_ setelah model _Machine Learning_ terbaik ditemukan.

### Solution Statement

Solusi yang dapat dilakukan sebagai berikut:
- Membandingkan hasil dari dua algoritma yang ada, dan juga melakukan _tunning_ parameter baik secara konvensional (mencari nilai K yang terbaik pada KNN), ataupun menggunakan metode **GridSearchCV** pada algoritma Random Forest.
- Jika data merupakan data klasifikasi yang tidak seimbang (_imbalance dataset_) metrik evaluasi akan menggunakan nilai **f1 score**

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

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
