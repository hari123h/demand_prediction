import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import r2_score

print("=" * 60)
print("Evaluating Demand Forecasting Models")
print("=" * 60)

# ---------------------------------------------------------
# Evaluate LSTM Demand Forecasting Model
# ---------------------------------------------------------
print("\n[1] LSTM Demand Forecasting Model")

try:
    # Load data
    X_test_lstm = np.load('X_test_lstm.npy')
    y_test_lstm = np.load('y_test_lstm.npy')

    with open('scaler_y_lstm.pkl', 'rb') as f:
        scaler_y_lstm = pickle.load(f)

    # Load model
    lstm_model = load_model('electricity_lstm_model.keras')

    # Evaluate
    loss_lstm, mae_lstm = lstm_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
    
    y_pred_scaled_lstm = lstm_model.predict(X_test_lstm, verbose=0)
    y_pred_lstm = scaler_y_lstm.inverse_transform(y_pred_scaled_lstm.reshape(-1, 1))
    y_true_lstm = scaler_y_lstm.inverse_transform(y_test_lstm.reshape(-1, 1))

    r2_lstm = r2_score(y_true_lstm, y_pred_lstm)

    print(f"  Test Loss (MSE): {loss_lstm:.4f}")
    print(f"  Test MAE (scaled): {mae_lstm:.4f}")
    print(f"  R2 Score: {r2_lstm:.4f}")

except Exception as e:
    print(f"  Error evaluating LSTM model: {e}")

# ---------------------------------------------------------
# Evaluate Peak Demand FNN Model
# ---------------------------------------------------------
print("\n[2] Peak Demand FNN Model")

try:
    # Load data
    X_test_peak = np.load('X_test_peak.npy')
    y_test_peak = np.load('y_test_peak.npy')

    with open('scaler_y_peak.pkl', 'rb') as f:
        scaler_y_peak = pickle.load(f)

    # Load model
    peak_model = load_model('peak_demand_model.keras')

    # Evaluate
    loss_peak, mae_peak = peak_model.evaluate(X_test_peak, y_test_peak, verbose=0)
    
    y_pred_scaled_peak = peak_model.predict(X_test_peak, verbose=0)
    y_pred_peak = scaler_y_peak.inverse_transform(y_pred_scaled_peak.reshape(-1, 1))
    y_true_peak = scaler_y_peak.inverse_transform(y_test_peak.reshape(-1, 1))

    r2_peak = r2_score(y_true_peak, y_pred_peak)

    print(f"  Test Loss (MSE): {loss_peak:.4f}")
    print(f"  Test MAE (scaled): {mae_peak:.4f}")
    print(f"  R2 Score: {r2_peak:.4f}")

except Exception as e:
    print(f"  Error evaluating Peak model: {e}")
