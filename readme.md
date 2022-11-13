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

  ![memuat](https://user-images.githubusercontent.com/93992324/201515920-c3e22237-b710-45f7-94c9-e17564ad2481.png)
  
2.	cek type data dari variable/fitur menggunakan fungsi info()

  ![image](https://user-images.githubusercontent.com/93992324/201516013-afa098ab-bea5-4012-a758-f09d1d1485e2.png)
  
  Dari info tersebut terlihat jumlah data sebanyak 1338 dan memiliki 7 kolom/fitur, 4 fitur numerik dan 3 fitur kategorikal.
  
3.	melihat statistic dataset menggunakan fungsi describe()

![image](https://user-images.githubusercontent.com/93992324/201516183-2ce96038-0180-412a-a81f-174b74564b98.png)

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

![image](https://user-images.githubusercontent.com/93992324/201516230-01c156e4-e187-42e0-a7f6-5ec9a2200c4e.png)

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

![image](https://user-images.githubusercontent.com/93992324/201518915-200bdad9-4049-48f7-b515-c5073f75c13d.png)

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
