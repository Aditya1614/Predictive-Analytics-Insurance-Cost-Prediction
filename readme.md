# Laporan Proyek Machine Learning – Aditya Candra Gumilang

## Domain Proyek
Asuransi adalah kesepakatan yang melibatkan dua pihak, yaitu perusahaan asuransi sebagai penanggung dan nasabah sebagai tertanggung. Nasabah berkewajiban membayar sejumlah premi asuransi sesuai dengan produk atau polis yang dipilih. Sedangkan perusahaan asuransi berkewajiban memberikan ganti rugi sesuai pertanggungan yang telah disepakati dalam polis.

Penanggungan dari produk atau polis asuransi dapat berupa biaya kesehatan, kecelakaan, kematian, kerusakan, kehilangan dsb.

Dalam praktiknya, biaya asuransi kesehatan bisa berbeda tiap individu. Hal ini dikarenakan oleh beberapa hal seperti kebiasaan merokok dapat memiliki resiko kesehatan yang lebih besar. Maka dari itu, penelitian ini bertujuan menentukan biaya asuransi kesehatan yang tepat tiap individu sesuai profil calon nasabah.

Referensi penelitian terdahulu: https://jurnal.untan.ac.id/index.php/jepin/article/view/48822/75676592879 

## Business Understanding
### Problem Statements
-	Bagaimana menyiapkan data untuk melatih model?
-	Bagaimana membuat model untuk memprediksi biaya asuransi dengan baik?
-	Bagaimana membagi data latih dan data uji?

### Goals
-	Menyiapkan data untuk melatih model
-	Membuat model dengan algoritma KNN, RandomForest dan Boosting algorithm
-	Membagi data latih dan data uji dengan proporsi 80:20

### Solution Statement
- Menyiapkan data dengan melakukan One Hot Encoding, menghapus data outlier, standarisasi dan train-test split.
-	Membandingkan hasil dari algoritma KNN, RandomForest dan Boosting algorithm
-	Membagi data latih dan data uji dengan menggunakan library scikit-learn

## Data Understanding
Data yang saya gunakan dalam proyek ini bersumber dari situs kaggle yang dapat diakses di : https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset

Variable yang ada pada datasets adalah sebagai berikut :
-	age : merupakan umur dari nasabah
-	sex : merupakan jenis kelamin dari nasabah
-	bmi : Body mass index (BMI) atau index massa tubuh.
-	children : jumlah anak/tanggungan nasabah
-	smoker : apakah nasabah mempunyai kebiasaan merokok atau tidak
-	region : daerah asal nasabah
-	charges : biaya asuransi

**Tahapan pra-pemrosesan data**
1.	Memuat data kedalam pandas dataframe menggunakan fungsi read_csv()

|   | age | sex | bmi | children | smoker | region | charges |
|---|---|---|---|---|---|---|---|
| 0 | 19 | female | 27.900 | 0 | yes | southwest | 16884.92400 |
| 1 | 18 | male | 33.770 | 1 | no | southeast | 1725.55230 |
| 2 | 28 | male | 33.000 | 3 | no | southeast | 4449.46200 |
| 3 | 33 | male | 22.705 | 0 | no | northeast | 21984.47061 |
| 4 | 32 | male | 28.880 | 0 | no | northeast | 3866.85520 |
| ... | ... | ... | ... | ... | ... | ... | ... |
  
2.	cek type data dari variable/fitur menggunakan fungsi info()

RangeIndex: 1338 entries, 0 to 1337

Data columns (total 7 columns) :

| # | Column | Non-Null | Count | Dtype |
|---|---|---|---|---|
| 0 |	age |	1338 | non-null | int64 |
| 1 |	sex |	1338 | non-null | object |
| 2 |	bmi |	1338 | non-null | float64 |
| 3 |	childen |	1338 | non-null | int64 |
| 4 |	smoker |	1338 | non-null | object | 
| 5 |	region |	1338 | non-null | object | 
| 6 |	charges |	1338 | non-null | float64 |

dtypes: float64(2), int64(2), object(3)

memory usage: 73.3+ KB
 
  Dari info tersebut terlihat jumlah data sebanyak 1338 dan memiliki 7 kolom/fitur, 4 fitur numerik dan 3 fitur kategorikal.
  
3.	melihat statistic dataset menggunakan fungsi describe()

|  | age | bmi | children | charges |
|---|---|---|---|---|
| count | 1338.000000 | 1338.000000 | 1338.000000 | 1338.000000 |
| mean | 39.207025 | 30.663397 | 1.094918 | 13270.422265 |
| std | 14.049960 | 6.098187 | 1.205493 | 12110.011237 |
| min	| 18.000000	| 15.960000	| 0.000000 | 1121.873900 |
| 25%	| 27.000000	| 26.296250	| 0.000000 | 4740.287150 |
| 50%	| 39.000000	| 30.400000	| 1.000000 | 9382.033000 |
| 75%	| 51.000000	| 34.693750	| 2.000000 | 16639.912515 |
| max	| 64.000000	| 53.130000	| 5.000000 | 63770.428010 |

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
-	Count  adalah jumlah sampel pada data.
-	Mean adalah nilai rata-rata.
-	Std adalah standar deviasi.
-	Min yaitu nilai minimum setiap kolom. 
-	25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
-	50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
-	75% adalah kuartil ketiga.
-	Max adalah nilai maksimum.

4.	mengecek apakah ada missing value menggunakan fungsi isnull()

|   |   |
|---|---|
| age | 0 |
| sex | 0 |
| bmi | 0 |
| children | 0 |
| smoker | 0 |
| region | 0 |
| charges | 0 |

 dtype: int64

Dari fungsi isnull() dapat dipastikan bahwa dataset tidak memiliki missing value sehingga data siap untuk lanjut ke proses berikutnya.

**Exploratory data analysis (EDA)**
1.	Menangani outlier pada fitur numerik
- Fitur age

<img src="https://user-images.githubusercontent.com/93992324/201516286-d374109b-09ea-4d01-99ad-d851c187e311.png" width="400"/>

-	Fitur bmi

<img src="https://user-images.githubusercontent.com/93992324/201516312-9fea8e0c-49ba-4197-9fa3-2eaca4da6702.png" width="400"/>

-	Fitur children

<img src="https://user-images.githubusercontent.com/93992324/201516327-a1179cfb-a45e-4429-b640-c98994a5b613.png" width="400"/>

- Fitur charges

<img src="https://user-images.githubusercontent.com/93992324/201518825-ed4dc082-e7c5-4d65-855f-4b42c4fd9b3a.png" width="400"/>

Berdasarkan plot diatas terdapat outlier pada fitur bmi dan fitur charges, maka untuk menghasilkan prediksi yang baik data outlier harus dihapuskan dari dataset. Dalam proyek ini, saya menggunakan metode IQR (Interquartile range), metode IQR mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.

Berikut persamaannya:

Batas bawah = Q1 - 1.5 * IQR

Batas atas = Q3 + 1.5 * IQR

Setelah menghapus data outlier jumlah total data menjadi 1193 dengan 7 kolom/fitur

2.	Univariate Analysis pada fitur kategorikal
-	Fitur sex

<img src="https://user-images.githubusercontent.com/93992324/201519123-f4ed6114-d265-409d-bb0f-92c18909569f.png" width="400" />

Dari chart bar diatas dapat disimpulkan bahwa data female (perempuan) lebih banyak dibandingkan data male (laki-laki)

-	Fitur smoker

<img src="https://user-images.githubusercontent.com/93992324/201519174-fe5a31df-d1bb-4bb3-9f4e-c6393d23f35d.png" width="400" />

Dari chart bar diatas dapat disimpukan bahwa data no (nasabah tidak merokok) lebih banyak dari data yes (nasabah merokok)

-	Fitur region

<img src="https://user-images.githubusercontent.com/93992324/201519275-564a8647-f47f-4c77-aaa2-5403c214bef3.png" width="400" />

Dari chart bar diatas dapat disimpukan bahwa semua region berjumlah relatif sama.

3.	Univariate Analysis pada fitur numerik

![numerik](https://user-images.githubusercontent.com/93992324/201519481-0e0ef706-8268-4da5-aa41-1a71f09c013d.png)

Dari histogram charges, bisa diperoleh beberapa informasi, antara lain:

-	Peningkatan biaya (charges) sebanding dengan penurunan jumlah sampel. Hal ini dapat dilihat jelas dari histogram charges yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).
-	Distribusi biaya miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

4.	Multivariate Analysis pada fitur kategorikal

![multi kategori](https://user-images.githubusercontent.com/93992324/201519924-47ebf95a-97e5-49be-a832-8277c2673485.png)

Dengan mengamati rata-rata biaya (charges) relatif terhadap fitur kategori di atas, dapat diperoleh insight sebagai berikut:
-	Pada fitur sex, rata-rata biaya cenderung mirip. Rentangnya berada +- 10000
-	Pada fitur smoker, biaya akan lebih mahal pada data yes (nasabah yang merokok)
-	Pada fitur region, rata-rata biaya cenderung mirip. Rentangnya berada antara 8000 hingga 12000
-	Kesimpulan akhir, fitur smoker memiliki pengaruh lebih besar terhadap biaya asuransi dibandingkan fitur kategorikal lainnya.

5.	Multivariate Analysis pada fitur numerik

![multi numerik](https://user-images.githubusercontent.com/93992324/201520049-2d7f1eee-2817-4772-b80c-1cc6f1c58eb2.png)

Membuat matriks korelasi untuk fitur numerik

![korelasi](https://user-images.githubusercontent.com/93992324/201520092-a535879d-b8f7-492a-87e5-669a8c1daa79.png)

Matriks korelasi berkisar antara -1 dan +1. Ia mengukur kekuatan hubungan antara dua variabel serta arahnya (positif atau negatif). Mengenai kekuatan hubungan antar variabel, semakin dekat nilainya ke 1 atau -1, korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke 0, korelasinya semakin lemah.

Arah korelasi antara dua variabel bisa bernilai positif (nilai kedua variabel cenderung meningkat bersama-sama) maupun negatif (nilai salah satu variabel cenderung meningkat ketika nilai variabel lainnya menurun).

Dari matriks diatas dapat disimpulkan bahwa fitur bmi memiliki skor korelasi yang sangat kecil (-0.06) terhadap fitur target charges. Sehingga, fitur tersebut dapat di-drop.

## Data Preparation
1.	One-hot-encoding

  Model machine learning tidak dapat mengolah data kategorik, sehingga perlu melakukan konversi data kategorik menjadi data numerik. Salah satu teknik untuk mengubah data kategorik menjadi data numerik adalah dengan menggunakan One Hot Encoding atau yang juga dikenal sebagai dummy variables. One Hot Encoding mengubah data kategorik dengan membuat kolom baru untuk setiap kategori seperti gambar di bawah.

<img src="https://user-images.githubusercontent.com/93992324/201522565-0fe411df-1a66-4818-ac8b-b3667f2d56d4.png" width="700" />

setelah mengubah fitur kategorik sex, smoker, dan region maka dataframe akan menjadi seperti ini 

|   | age | children | charges | sex_female | sex_male | smoker_no | smoker_yes | region_northeast | region_northwest | region_southeast | region_southwest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 19 | 0 | 16884.92400 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| 1 | 18 | 1 | 1725.55230 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| 2 | 28 | 3 | 4449.46200 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| 3 | 33 | 0 | 21984.47061 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| 4 | 32 | 0 | 3866.85520 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |

2.	Train-Test split

Train-test split adalah membagi dataset menjadi 2 bagian yaitu data training dan data testing. Dengan demikian, kita bisa melakukan pelatihan model pada train set, kemudian mengujinya pada test set.

Data testing diambil dengan proporsi tertentu. Pada praktiknya, pembagian data training dan data testing yang paling umum adalah 80:20, 70:30, atau 60:40, tergantung dari ukuran atau jumlah data. Namun, untuk dataset berukuran besar, proporsi pembagian 90:10 atau 99:1 juga umum dilakukan. Misal jika ukuran dataset sangat besar berisi lebih dari 1 juta record, maka dapat mengambil sekitar 10 ribu data saja untuk testing alias sebesar 1% saja. Dalam penelitian ini akan membagi data training dan data testing dengan proporsi 80:20.

Untuk membagi data bisa dilakukan dengan menggunakan fungsi train_test_split yang disediakan oleh library scikit-learn, dengan kode sebagai berikut:

```
from sklearn.model_selection import train_test_split
 
X = asuransi.drop(["charges"],axis =1)
y = asuransi["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```

- X : berfungsi untuk drop/menghapus kolom charges
- y : berfungsi untuk menampilkan nilai dari kolom charge
- test_size : adalah ukuran test (0.2 berarti 20% dari total dataset)
- random_state : untuk menyeting random seed yang bertujuan supaya dapat memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan.

Total data setelah dibagi yaitu : 

<img src="https://user-images.githubusercontent.com/93992324/201522832-fc44d75b-cc91-4c00-bdc6-40ff318b617a.png" width="300" />

3.	Standarisasi

Standarisasi adalah proses konversi nilai-nilai dari suatu fitur sehingga nilai-nilai tersebut memiliki skala yang sama. Z score adalah metode paling populer untuk standardisasi di mana setiap nilai pada sebuah atribut numerik akan dikurangi dengan rata-rata dan dibagi dengan standar deviasi dari seluruh nilai pada sebuah kolom atribut.

<img src="https://user-images.githubusercontent.com/93992324/201522892-afe4f86d-0169-4887-9281-027f5e16de65.png" width="300" />

Standarisasi dilakukan pada fitur data numerik selain fitur target, dalam penelitian ini stadarisasi akan dilakukan pada fitur age dan children yang ada di dataset train (latih) sehingga menjadi seperti ini

<img src="https://user-images.githubusercontent.com/93992324/201522929-bcbf0ce7-6617-46f4-9924-fa97dd79fd31.png" width="300" />

## Modelling
Penelitian ini akan menggunakan 3 algoritma berbeda yaitu : 

1.	KNN
2.	RandomForest
3.	Boosting Algorithm

Berikut adalah penjelasan mengenai ketiga algoritma tersebut.

1.	KNN

Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.

KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat).
Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika memilih k yang terlalu rendah, maka akan menghasilkan model yang overfit dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k terlalu tinggi, maka model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi.

KNN menggunakan perhitungan ukuran jarak untuk menentukan titik mana dalam data yang paling mirip dengan input baru. Metrik ukuran jarak yang digunakan secara default pada library sklearn adalah Minkowski distance. Beberapa metrik ukuran jarak yang juga sering dipakai antara lain: Euclidean distance dan Manhattan distance. Sebagai contoh, jarak Euclidean dihitung sebagai akar kuadrat dari jumlah selisih kuadrat antara titik a dan titik b. Dirumuskan sebagai berikut:

![image](https://user-images.githubusercontent.com/93992324/201523019-099c8050-25e4-41ee-a6b9-b920e7a36091.png)

Sedangkan, Minkowski distance merupakan generalisasi dari Euclidean dan Manhattan distance. Dirumuskan sebagai berikut:

![image](https://user-images.githubusercontent.com/93992324/201523062-4c7cddb2-de7b-4e1d-80ab-d43c56c54177.png)

Dalam penelitian ini penulis menggunakan metrik Euclidean dan parameter k = 10.

Setiap algoritma tentu memiliki kelebihan dan kekurangan berikut adalah  kelebihan dan kekurangan dari algoritma KNN.

Kelebihan :

-	Algoritma K-NN kuat dalam mentraining data yang noisy.
-	Algoritma K-NN sangat efektif jika datanya besar.
-	Mudah diimplementasikan.

Kekurangan :

-	Algoritma K-NN perlu menentukan nilai parameter K.
-	Tidak efektif pada dataset yang memiliki jumlah fitur yang banyak
-	Rentan pada variabel yang non-informatif.

2.	RandomForest

RandomForest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Esemble learning merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir.

RandomForest menggunakan Teknik pendekatan bagging dalam membuat modelnya. Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random. Dalam teknik bagging, sejumlah model dilatih dengan teknik sampling with replacement (proses sampling dengan penggantian). Ketika melakukan sampling with replacement, sampel dengan nilai yang berbeda bersifat independen. Artinya, nilai suatu sampel tidak mempengaruhi sampel lainnya. Akibatnya, model yang dilatih akan berbeda antara satu dan lainnya.

Ada beberapa parameter yang digunakan dalam algoritma RandomForest yaitu:

-	n_estimator: jumlah trees (pohon) di forest. Di penelitian ini penulis menggunakan n_estimator=50.
-	max_depth: kedalaman atau panjang pohon. Merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Di penelitian ini penulis set max_depth =16.
-	random_state: digunakan untuk mengontrol random number generator yang digunakan. Di penelitian ini penulis menggunakan random_state =55. 
-	n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. Penulis menggunakan n_jobs=-1 yang artinya semua proses berjalan secara paralel.
-	
Setiap algoritma tentu memiliki kelebihan dan kekurangan berikut adalah  kelebihan dan kekurangan dari algoritma RandomForest.

Kelebihan: 

-	Bekerja dengan baik dengan data non-linear.
-	Risiko overfitting lebih rendah.
-	Berjalan secara efisien pada kumpulan data yang besar.

Kekurangan:

-	Random Forest cenderung bias saat berhadapan dengan variabel kategorikal.
-	Waktu komputasi pada dataset berskala besar relatif lambat.
-	Tidak cocok untuk metode linier dengan banyak fitur sparse.

3.	Boosting Algorithm

Boosting Algorithm algorithm juga merupakan salah satu dari metode ensemble learning. Teknik boosting bekerja dengan membangun model dari data latih. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner).

Dilihat dari caranya memperbaiki kesalahan pada model sebelumnya, algoritma boosting terdiri dari dua metode:

-	Adaptive boosting
-	Gradient boosting
-	
Penelitian, penulis akan menggunakan metode adaptive boosting yaitu algoritma AdaBoost. Berikut merupakan parameter-parameter yang digunakan pada algoritma AdaBoost:

-	learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting. Dalam penelitian ini penulis menggunakan learning_rate=0.05.
-	random_state: digunakan untuk mengontrol random number generator yang digunakan. Dalam penelitian ini penulis menggunakan random_state=55

Setiap algoritma tentu memiliki kelebihan dan kekurangan berikut adalah  kelebihan dan kekurangan dari algoritma RandomForest.

Kelebihan:

-	Hasil pemodelan yang lebih akurat
-	Metode ensemble dapat digunakan untuk menangkap hubungan linier maupun non-linier dalam data.
Kekurangan:

-	Waktu komputasi dan desain tinggi.
-	Pengurangan kemampuan interpretasi model

## Evaluation
Karena penelitian ini adalah model regresi maka prediksi yang mendekati nilai sebenarnya, mempunyai performa yang baik. Sedangkan jika tidak, performanya buruk.
Metrik yang digunakan pada penelitian ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut

![image](https://user-images.githubusercontent.com/93992324/201523312-81244b9d-0b5e-414f-8d6d-63d110404217.png)

Keterangan:

-	N = jumlah dataset
-	yi = nilai sebenarnya
-	y_pred = nilai prediksi

Hasil perhitungan metrik mse terhadap model yaitu sebagai berikut

<img src="https://user-images.githubusercontent.com/93992324/201523323-d448b470-c0cf-40f2-939e-608f046a5411.png" width="300" />

Untuk mempermudah mempermudah membaca data, lihat plot metrik tersebut dengan chart bar berikut

<img src="https://user-images.githubusercontent.com/93992324/201523330-544c6ab3-6db1-4e0b-b614-acc58c9449cf.png" width="400" />

Uji model dengan data test

<img src="https://user-images.githubusercontent.com/93992324/201523341-fa37ca71-1f90-4958-b404-435dc669430c.png" width="400" />

Dapat disimpulkan bahwa, model RandomForest(RF) membuat prediksi paling mendekati dibandingkan algoritma lainnya.
