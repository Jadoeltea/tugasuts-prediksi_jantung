#library
import pickle
import numpy as np
import streamlit as st

classifier = pickle.load(open('prediksi_jantung123.sav', 'rb'))

st.title('PREDIKSI PENYAKIT JANTUNG')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.text_input('Umur')
    trestbps = st.text_input('Tekanan Darah')
    restecg = st.selectbox('Hasil ECG', ('0', '1', '2'))
    oldpeak = st.text_input('Penurunan segmen ST pada EKG')
with col2:
    sex = st.selectbox('Jenis Kelamin', ('0', '1'))
    chol = st.text_input('Nilai Kolesterol')
    thalach = st.text_input('Detak Jantung Maksimal')
    slope = st.selectbox('Tingkat Penurunan ST Segment', ('0', '1', '2'))
    thal = st.selectbox('Aliran Darah ke Otot Jantung', ('0', '1', '2'))
with col3:
    cp = st.selectbox('Jenis Nyeri Dada', ('0', '1', '2', '3'))
    fbs = st.selectbox('Gula Darah', ('0', '1'))
    exang = st.selectbox('Induksi Angina', ('0', '1'))
    ca = st.selectbox('Jumlah Sumbatan Arteri Koroner', ('0', '1', '2', '3'))

heart_diagnosis = ''

if st.button('Prediksi Penyakit Jantung'):
    # memeriksa jika ada data yang kosong
    if age == '' or trestbps == '' or chol == '' or thalach == '' or oldpeak == '':
        st.error("Data tidak boleh kosong. Harap isi semua kolom input.")
    else:
        # Konversi data input ke tipe data numerik
        age = int(age)
        trestbps = float(trestbps)
        chol = float(chol)
        thalach = float(thalach)
        oldpeak = float(oldpeak)

        # melakukan prediksi
        heart_prediction = classifier.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'Pasien Terkena Penyakit Jantung'
        else:
            heart_diagnosis = 'Pasien Tidak Terkena Penyakit Jantung'

st.success(heart_diagnosis)
