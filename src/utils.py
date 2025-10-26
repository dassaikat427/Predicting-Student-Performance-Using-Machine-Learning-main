import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_data(file_path='data/student_data.csv'):
    return pd.read_csv(file_path)

def encode_features(df, le_gender=None, le_family=None, le_extra=None):
    if le_gender is None: le_gender = LabelEncoder().fit(df['gender'])
    if le_family is None: le_family = LabelEncoder().fit(df['family_support'])
    if le_extra is None: le_extra = LabelEncoder().fit(df['extracurricular'])
    
    df['gender'] = le_gender.transform(df['gender'])
    df['family_support'] = le_family.transform(df['family_support'])
    df['extracurricular'] = le_extra.transform(df['extracurricular'])
    
    return df, le_gender, le_family, le_extra

def scale_features(X, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def predict_student(model, scaler, le_gender, le_family, le_extra, student_data):
    data = [
        le_gender.transform([student_data['gender']])[0],
        student_data['age'],
        student_data['study_time'],
        student_data['attendance'],
        student_data['prior_grade'],
        le_family.transform([student_data['family_support']])[0],
        le_extra.transform([student_data['extracurricular']])[0]
    ]
    data_scaled = scaler.transform([data])
    return model.predict(data_scaled)[0]