import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from data_preprocessing import load_data, preprocess_data

# Create models folder if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load and preprocess data
df = load_data()
X, y, scaler, le_gender, le_family, le_extra = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Save model and encoders
joblib.dump(model, 'models/Random_Forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_gender, 'models/le_gender.pkl')
joblib.dump(le_family, 'models/le_family.pkl')
joblib.dump(le_extra, 'models/le_extra.pkl')