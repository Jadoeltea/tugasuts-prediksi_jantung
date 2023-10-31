# Laporan Proyek Machine Learning
### Nama  : **Irfan Zulkarnaen**
### Nim   : 231352002
### Kelas : Malam B

# Memahami Kesehatan Jantung Dengan Teknologi : Aplikasi Data Mining dalam memprediksi Penyakit Jantung menggunakan Python
### **Summary**

Penyakit jantung adalah salah satu masalah kesehatan yang paling umum di seluruh dunia. Menurut Organisasi Kesehatan Dunia (World Health Organization), penyakit jantung adalah penyebab kematian nomor satu di seluruh dunia. Oleh karena itu, penting untuk dapat mendeteksi dan memprediksi risiko penyakit jantung dengan cepat dan akurat.    


### **Business Understanding**
Teknologi dan metode ilmu data (data mining) telah menjadi alat yang sangat berguna dalam pemahaman dan prediksi penyakit jantung. Dalam konteks ini, data mining digunakan untuk menganalisis sejumlah besar data medis yang mencakup informasi tentang pasien, faktor risiko, gejala, dan hasil tes medis. 


### **Goals**
Dengan menggunakan teknik-teknik data mining, kita dapat mengidentifikasi pola dan hubungan dalam data ini untuk memprediksi risiko atau kemungkinan seseorang terkena penyakit jantung. Dalam  Aplikasi ini akan memungkinkan pengguna untuk memasukkan informasi pribadi mereka, seperti usia, jenis kelamin, nilai kolesterol, tekanan darah, dan gejala lainnya, dan akan memberikan prediksi tentang apakah mereka berisiko terkena penyakit jantung.


### **Solution statements**
Saya akan mencoba mengembangkan sebuah aplikasi menggunakan bahasa pemrograman Python dan teknik data mining untuk memprediksi penyakit jantung.

### **Data Understanding**
Dalam proyek ini, saya menggunakan dataset dari [Kaggle](https://www.kaggle.com). Dataset ini, yang disebut [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)
, berisi data kesehatan yang kita gunakan untuk menganalisis dan memprediksi penyakit jantung. dataset [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland) ini dipilih karena relevansi atribut-atributnya dalam proyek ini."

###  **Variabel-variabel pada Aplikasi Data Mining dalam memprediksi Penyakit Jantung menggunakan Python** 
Selanjutnya variabel atau fitur pada data ini adalah sebagai berikut :  

    -) age/umur (umur dalam tahun)
    -) sex/Jenis Kelamin (1 = laki-laki, 0 = perempuan)
    -) cp/Jenis Nyeri dada (nyeri dada dengan nilai 0(tidak ada), 1(ringan), 2(signifikan), dan 3(heabt))
    -) trestbps/Tekanan darah (tekanan darah)
    -) chol/Nilai kolesterol (konsentrasi kolesterol dalam mg/dl)
    -) fbs/Gula darah (konsentrasi gula setelah puasa dalam 8 jam dengan nilai 1(> 120 mg/dl) dan 0(< 120 mg/dL)
    -) restecg/Nilai ECG (hasil ECG dengan nilai 0(tidak ada kelainan), 1(kelainan minor), 2(kelainan signifikan)
    -) thalach/Detak Jantung maksimal (detak jantung maksimal setelah latihan fisik intens)
    -) exang/Induksi Angina ( ada tidaknya rasa sakit atau tekanan pada dada akibat kurangnya pasokan darah dan oksigen yang cukup ke jantung)
    -) oldpeak/Penurunan segmen ST pada EKG (penurunan segmen ST pada elektrokardiogram (EKG) setelah latihan fisik)
    -) slope/Tingkat Penurunan ST segmen (Tingkat penurunan ST segment dengan nilai 0(tidak ada), 1(lambat), 2(cepat))
    -) ca/Jumlah Sumbatan Arteri Koroner (jumlah sumbatan arteri koroner utama dengan nilai 0(tidak ada), 1(1 arteri), 2(2 arteri), 3(3 arteri)
    -) thal (aliran darah ke otot jantung dengan nilai 0 = normal, 1 = fixed defect, 2 = reversable defect)
    -) target (mengidap penyakit jantung atau tidak)

Adapun tipe data dalam dataset [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland) yaitu:

 No|  Column  |  Non-Null Count | Dtype  |
---| ---------| ----------------|--------| 
 0 | age      | 303 non-null    | int64  |
 1 | sex      | 303 non-null    | int64  |
 2 | cp       | 303 non-null    | int64  |
 3 | trestbps | 303 non-null    | int64  |
 4 | chol     | 303 non-null    | int64  |
 5 | fbs      | 303 non-null    | int64  |
 6 | restecg  | 303 non-null    | int64  |
 7 | thalach  | 303 non-null    | int64  |
 8 | exang    | 303 non-null    | int64  |
 9 | oldpeak  | 303 non-null    | float64|
 10| slope    | 303 non-null    | int64  |
 11| ca       | 303 non-null    | int64  |
 12| thal     | 303 non-null    | int64  |
 13| target   | 303 non-null    | int64  |
dtypes: float64(1), int64(13)

**Visualisasi Data**:
- Dalam Melakukan pemahaman terhadap dataset [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland), saya membuat empat grafik [Histogram](https://drive.google.com/file/d/1lmHKm_LYWlYxfqwbLz03c3m4cxt65v1B/view?usp=sharing) yang berbeda dimana, pada grafik [Histogram](https://drive.google.com/file/d/1lmHKm_LYWlYxfqwbLz03c3m4cxt65v1B/view?usp=sharing) yang pertama menunjukkan jumlah penderita penyakit jantung didalam dataset [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland), yang kedua jumlah penderita berdasarkan jenis kelamin, berdasarkan golongan Usia, dan terakhir berdasarkan jumlah penderita jantung berdasarkan jumlah kolesterol pasien.
- Lalu saya membuat [scatter plot](https://drive.google.com/file/d/1wXx6T6lOrC2ocTTLwQhlitWNPNy3vIV7/view?usp=sharing) untuk membandingkan jumlah penderita jantung berdasarkan usia vs data detak jantung maksimal pasien
- Dan terakhir dalam preprocesing data saya membuat matriks korelasi dan menampilkannya dalam bentuk [heatmap](https://drive.google.com/file/d/16ZjLnqzlWMvDJC9qQrzh-E0YhKRbFby4/view?usp=sharing). karena Matriks korelasi memberikan informasi tentang hubungan antara berbagai atribut numerik dalam dataset [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland)


## Data Preparation
Teknik data preparation yang dilakukan adalah :
- Konfigurasi API Kaggle dengan vscode
- mendownload file dari Kaggle
- Menentukan library yang akan digunakan
- Membaca dataset [Heart Disease Cleveland](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland) yang telah didownload, yaitu file Heart_disease_cleveland_new.csv kemudian saya rename menjadi heart.csv agar memudahkan 
- Cek missing value didalam data heart_dataset
- Visualisasi Data
- Preprocessing
- Memisahkan data dan label
- Menstandarisasikan data
- Memisahkan data training dan testing
- Membuat data latih menggunakan Algoritma SVM
- Membuat model evaluasi
- Membuat model prediksi
- Simpan model untuk proses deploy ke streamlit


## Modeling
Prediksi Aplikasi Data Mining dalam memprediksi Penyakit Jantung menggunakan Python menggunakan algoritma Support Vector Machine (SVM), Alasan mengapa menggunakan algoritma Support Vector Machine (SVM) untuk model prediksi penyakit jantung dalam proyek ini adalah sebagai berikut:

    1. Kemampuan Memisahkan Data yang Kompleks: SVM adalah algoritma yang sangat efektif dalam memisahkan data yang kompleks dan tidak linier. Data medis yang digunakan dalam prediksi penyakit jantung seringkali memiliki hubungan yang kompleks antara atribut-atributnya. SVM dapat menangani hubungan ini dengan baik, bahkan dalam kasus data yang tidak linier.

    2. Kemampuan Generalisasi: SVM memiliki kemampuan yang baik untuk generalisasi, artinya kemampuan untuk menghasilkan model yang tidak hanya baik pada data pelatihan, tetapi juga mampu menggeneralisasi dan memberikan prediksi yang baik pada data baru yang belum pernah dilihat sebelumnya. Ini penting dalam aplikasi medis di mana kita ingin dapat memprediksi penyakit jantung pada pasien baru.

    3. Kemampuan Menangani Data yang Tidak Seimbang: Data medis sering kali tidak seimbang, yaitu jumlah pasien dengan penyakit jantung mungkin jauh lebih sedikit dari pada yang sehat. SVM memiliki teknik penalti yang dapat mengatasi masalah ini dan memberikan prediksi yang seimbang.

    4. Kemampuan Menangani Data Berkualitas Rendah: Data medis sering kali memiliki data yang hilang atau berkualitas rendah. SVM dapat mengatasi masalah ini dengan baik melalui teknik penanganan data yang hilang dan pemrosesan data.

    5. Interpretabilitas: SVM memberikan hasil yang relatif mudah diinterpretasikan. Ini penting dalam konteks medis di mana profesional kesehatan ingin memahami faktor-faktor yang berkontribusi pada prediksi penyakit jantung.

    6. Kinerja yang Baik: SVM seringkali memberikan kinerja yang baik dalam banyak kasus, termasuk dalam prediksi penyakit jantung. Dengan parameter yang disesuaikan dengan baik, SVM dapat memberikan akurasi yang tinggi.

    7. Pendukung Optimal: SVM menghasilkan "support vectors" yang merupakan contoh data yang paling penting untuk pemisahan kelas. Ini berarti kita dapat fokus pada subset data yang paling relevan, yang mengurangi kompleksitas model.

    8. Pengendalian Overfitting: SVM memiliki parameter penalti yang dapat mengontrol overfitting, sehingga kita dapat menghasilkan model yang tidak terlalu rumit dan lebih mungkin untuk menggeneralisasi dengan baik.

    9. Kemampuan Mengatasi Noise: SVM memiliki toleransi terhadap noise dalam data, yang sering kali ada dalam data medis.

    10. Kemampuan Menangani Data Multivariat: Data medis sering kali memiliki banyak atribut yang berkorelasi. SVM dapat menangani data multivariat ini dan mempertimbangkan korelasi antara atribut untuk meningkatkan prediksi.

## Evaluation
Sebelum membuat model prediksi saya mencoba membuat evaluasi tingkat akurasi data menggunakan algoritma SVM. berikut :

1. Matrik [konfusi](https://drive.google.com/file/d/1cut75Gr_3mGGf5MEqDlU-WY0sYIbR09S/view?usp=sharing), Matriks konfusi adalah alat evaluasi yang umum digunakan dalam pemodelan klasifikasi untuk mengukur sejauh mana model klasifikasi dapat memprediksi dengan benar kelas-kelas target. Matriks konfusi biasanya dibagi menjadi empat sel atau komponen, yang mencakup True Positives (TP), False Positives (FP), True Negatives (TN), dan False Negatives (FN). hasil matriks konfusi untuk data pelatihan dan data uji menggunakan algoritma SVM adalah :
    a.) Matriks Konfusi pada Data Pelatihan:

        - True Positives (TP): Jumlah kasus di mana model dengan benar memprediksi pasien menderita penyakit jantung (positif) ketika mereka sebenarnya menderita penyakit jantung. Dalam kasus ini, ada 90 pasien yang benar-benar menderita penyakit jantung dan telah diprediksi dengan benar oleh model.

        - False Positives (FP): Jumlah kasus di mana model salah memprediksi pasien menderita penyakit jantung (positif) ketika sebenarnya mereka tidak menderita penyakit jantung (negatif). Dalam kasus ini, ada 14 pasien yang tidak menderita penyakit jantung, tetapi model secara keliru memprediksi mereka sebagai positif.

        - True Negatives (TN): Jumlah kasus di mana model dengan benar memprediksi pasien tidak menderita penyakit jantung (negatif) ketika mereka sebenarnya tidak menderita penyakit jantung. Dalam kasus ini, ada 117 pasien yang benar-benar tidak menderita penyakit jantung dan telah diprediksi dengan benar oleh model.

        - False Negatives (FN): Jumlah kasus di mana model salah memprediksi pasien tidak menderita penyakit jantung (negatif) ketika sebenarnya mereka menderita penyakit jantung (positif). Dalam kasus ini, ada 21 pasien yang sebenarnya menderita penyakit jantung, tetapi model secara keliru memprediksi mereka sebagai negatif.

    b.) Matriks Konfusi pada Data Uji:

        - True Positives (TP): Jumlah kasus di mana model dengan benar memprediksi pasien menderita penyakit jantung (positif) ketika mereka sebenarnya menderita penyakit jantung. Dalam kasus ini, ada 22 pasien yang benar-benar menderita penyakit jantung dan telah diprediksi dengan benar oleh model.

        - False Positives (FP): Jumlah kasus di mana model salah memprediksi pasien menderita penyakit jantung (positif) ketika sebenarnya mereka tidak menderita penyakit jantung (negatif). Dalam kasus ini, ada 6 pasien yang tidak menderita penyakit jantung, tetapi model secara keliru memprediksi mereka sebagai positif.

        - True Negatives (TN): Jumlah kasus di mana model dengan benar memprediksi pasien tidak menderita penyakit jantung (negatif) ketika mereka sebenarnya tidak menderita penyakit jantung. Dalam kasus ini, ada 27 pasien yang benar-benar tidak menderita penyakit jantung dan telah diprediksi dengan benar oleh model.

        - False Negatives (FN): Jumlah kasus di mana model salah memprediksi pasien tidak menderita penyakit jantung (negatif) ketika sebenarnya mereka menderita penyakit jantung (positif). Dalam kasus ini, ada 6 pasien yang sebenarnya menderita penyakit jantung, tetapi model secara keliru memprediksi mereka sebagai negatif.

2. dan hasil menggunakan metrik [**akurasi, precision, recall, dan F1 score**](https://drive.google.com/file/d/1ohVKDf1sJquRSYdyne83HxGpwfeavDvd/view?usp=sharing) :

        - Precision (presisi):
        Untuk kelas 0 (label 0): Presisi adalah 0.82, yang berarti bahwa dari semua contoh yang diprediksi sebagai kelas 0, sekitar 82% dari prediksi tersebut adalah benar (True Positives) dan sekitar 18% adalah kesalahan (False Positives).
        Untuk kelas 1 (label 1): Presisi adalah 0.79, yang berarti bahwa dari semua contoh yang diprediksi sebagai kelas 1, sekitar 79% dari prediksi tersebut adalah benar (True Positives) dan sekitar 21% adalah kesalahan (False Positives).

        - Recall (sensitivitas):
        Untuk kelas 0 (label 0): Recall adalah 0.82, yang berarti bahwa dari semua contoh yang sebenarnya adalah kelas 0, model dapat mengidentifikasinya dengan benar sekitar 82% (True Positives), sedangkan sekitar 18% dari kasus kelas 0 tidak terdeteksi (False Negatives).
        Untuk kelas 1 (label 1): Recall adalah 0.79, yang berarti bahwa dari semua contoh yang sebenarnya adalah kelas 1, model dapat mengidentifikasinya dengan benar sekitar 79% (True Positives), sedangkan sekitar 21% dari kasus kelas 1 tidak terdeteksi (False Negatives).

        - F1-Score (skor F1):
        F1-Score adalah rata-rata harmonik dari presisi dan recall. Untuk kelas 0, skor F1 adalah sekitar 0.82, dan untuk kelas 1, skor F1 adalah sekitar 0.79. Skor F1 menggabungkan presisi dan recall ke dalam satu metrik yang memberikan gambaran tentang sejauh mana model dapat memprediksi kelas target dengan baik.

        - Support (dukungan):
        Dukungan adalah jumlah contoh dalam data uji yang termasuk dalam masing-masing kelas. Terdapat 33 contoh kelas 0 dan 28 contoh kelas 1 dalam data uji.

        - Accuracy (akurasi):
        Akurasi adalah persentase total prediksi yang benar dibandingkan dengan jumlah total contoh dalam data uji. Akurasi adalah sekitar 0.80 atau 80%, yang berarti model dapat memprediksi dengan benar sekitar 80% dari semua contoh dalam data uji.

        - Macro Average (Rata-rata Makro):
        Rata-rata makro adalah rata-rata dari metrik-metrik evaluasi untuk setiap kelas secara terpisah. Dalam kasus ini, rata-rata makro untuk presisi, recall, dan F1-score adalah sekitar 0.80, yang merupakan rata-rata dari metrik-metrik tersebut untuk kedua kelas.

        - Weighted Average (Rata-rata Berbobot):
        Rata-rata berbobot memberikan bobot yang lebih besar kepada kelas dengan dukungan yang lebih tinggi. Dalam kasus ini, rata-rata berbobot untuk presisi, recall, dan F1-score adalah sekitar 0.80, yang juga merupakan rata-rata dari metrik-metrik tersebut dengan mempertimbangkan jumlah dukungan masing-masing kelas.

Dari hasil Laporan klasifikasi ini sesuai dengan konteks data yang memberikan gambaran tentang sejauh mana model klasifikasi berhasil dalam memprediksi kelas target(pasien penderita penyakit jantung), baik untuk kelas 0(tidak) maupun kelas 1(ya). 


## Deployment
Berikut hasil dari akhir proyek [APLIKASI PREDIKSI PENYAKIT JANTUNG](https://s.id/TugasMLIrfanZulkarnaen)
ada pun [screenshoot](https://drive.google.com/file/d/17oeDQ0akI0VExSBDqQjWl9WDLlMrLJQR/view?usp=sharing) nya.





**---Ini adalah bagian akhir laporan---**



