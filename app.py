import streamlit as st
import joblib
import numpy as np

# Load model dan preprocessing
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('random_forest_pca.pkl')  # Pilih model yang akan digunakan

st.title("Diabetes Prediction App")

# Input pengguna
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 0, 120)

# Prediksi saat tombol ditekan
if st.button("Predict"):
    # Membuat array input
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Preprocessing input
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_scaled)
    
    # Prediksi
    prediction = model.predict(input_pca)
    
    # Menampilkan hasil
    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    st.success(f"Prediction: {result}"
)
