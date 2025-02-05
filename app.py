from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan preprocessing tools
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('random_forest_pca.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Memuat halaman HTML

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form HTML
    input_data = [float(request.form[key]) for key in ['pregnancies', 'glucose', 'blood_pressure', 
                                                       'skin_thickness', 'insulin', 'bmi', 'dpf', 'age']]
    
    # Preprocessing input
    input_scaled = scaler.transform([input_data])
    input_pca = pca.transform(input_scaled)
    
    # Prediksi
    prediction = model.predict(input_pca)
    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    
    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == '__main__':
    app.run(debug=True)
