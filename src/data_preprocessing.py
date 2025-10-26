import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path='data/student_data.csv'):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    le_gender = LabelEncoder()
    le_family = LabelEncoder()
    le_extra = LabelEncoder()
    
    df['gender'] = le_gender.fit_transform(df['gender'])
    df['family_support'] = le_family.fit_transform(df['family_support'])
    df['extracurricular'] = le_extra.fit_transform(df['extracurricular'])
    
    X = df.drop(['student_id', 'final_grade'], axis=1)
    y = df['final_grade']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, le_gender, le_family, le_extra