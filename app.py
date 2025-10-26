from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from src.data_preprocessing import load_data, preprocess_data

app = Flask(__name__)

# Load model and encoders
model = joblib.load('models/Random_Forest.pkl')
scaler = joblib.load('models/scaler.pkl')
le_gender = joblib.load('models/le_gender.pkl')
le_family = joblib.load('models/le_family.pkl')
le_extra = joblib.load('models/le_extra.pkl')

df = load_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    age = int(request.form['age'])
    study_time = float(request.form['study_time'])
    attendance = float(request.form['attendance'])
    prior_grade = float(request.form['prior_grade'])
    family_support = request.form['family_support']
    extracurricular = request.form['extracurricular']
    
    gender_enc = le_gender.transform([gender])[0]
    family_enc = le_family.transform([family_support])[0]
    extra_enc = le_extra.transform([extracurricular])[0]
    
    student_data = np.array([[gender_enc, age, study_time, attendance, prior_grade, family_enc, extra_enc]])
    student_data_scaled = scaler.transform(student_data)
    
    predicted_grade = model.predict(student_data_scaled)[0]
    
    return render_template('index.html', result=f'Predicted Final Grade: {predicted_grade:.2f}')

@app.route('/all_students')
def all_students():
    X, y, _, _, _, _ = preprocess_data(df)
    df_copy = df.copy()
    df_copy['Predicted_Grade'] = model.predict(X)
    return render_template('all_students.html', tables=[df_copy.to_html(classes='table table-striped', index=False)])

if __name__ == '__main__':
    import os
    port=int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)