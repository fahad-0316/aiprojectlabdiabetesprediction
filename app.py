from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler safely (allow app to start even if files are missing)
model = None
scaler = None
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception:
        model = None
        scaler = None

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['bloodpressure']),
            float(request.form['skinthickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        data = np.array([features])
        if scaler is None or model is None:
            return render_template('index.html', prediction="Model or scaler not found. Run training script.", color="orange")

        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)[0]

        if result == 1:
            prediction = "⚠️ Diabetic"
            color = "red"
        else:
            prediction = "✅ Non-Diabetic"
            color = "green"

        return render_template('index.html', prediction=prediction, color=color)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", color="orange")

if __name__ == '__main__':
    app.run(debug=True)