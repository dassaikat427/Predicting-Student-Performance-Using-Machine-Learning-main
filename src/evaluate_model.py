import joblib
from data_preprocessing import load_data, preprocess_data
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

def evaluate_model(model_path='models/Random_Forest.pkl', data_path='data/student_data.csv'):
    model = joblib.load(model_path)
    df = load_data(data_path)
    X, y, scaler, le_gender, le_family, le_extra = preprocess_data(df)
    
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    try:
        importances = model.feature_importances_
        features = df.drop(['student_id', 'final_grade'], axis=1).columns
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
        print("\nFeature Importances:")
        print(feat_imp)
    except AttributeError:
        print("Model does not support feature_importances_")

if __name__ == '__main__':
    evaluate_model()