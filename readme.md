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
-	Membandingkan hasil dari algoritma KNN, RandomForest dan Boosting algorithm
-	Menggunakan algoritma Mean squared error(MSE) untuk evaluasi model
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
