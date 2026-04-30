import numpy as np
import xgboost as xgb
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import random
import os

os.environ['PYTHONHASHSEED'] = '40'
random.seed(40)
np.random.seed(40)

print("Training Peak Demand XGBoost Forecasting Model")
print("=" * 60)

# Load prepared data
X_train = np.load('X_train_peak.npy')
X_test = np.load('X_test_peak.npy')
y_train = np.load('y_train_peak.npy')
y_test = np.load('y_test_peak.npy')

with open('scaler_y_peak.pkl', 'rb') as f:
    scaler_y_peak = pickle.load(f)

print(f"Training set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}\n")

# Model Architecture for flat tabular data
model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    random_state=40
)

# Train Model
print("\nStarting Training...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Evaluate
print("\nEvaluating on Test Data...")
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y_peak.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_true = scaler_y_peak.inverse_transform(y_test.reshape(-1, 1))

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print("\nModel Performance Summary:")
print(f"  Test Loss (MSE, unscaled): {mse:.4f}")
print(f"  Test MAE (unscaled): {mae:.4f}")
print(f"  Test R2 Score: {r2:.4f}")

with open('peak_demand_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nPeak Demand Model saved to: peak_demand_model.pkl")
