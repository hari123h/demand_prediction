# Project Codebase

## `app.py`

```python
"""
Electricity Demand Forecasting - Premium Web API
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)



class PeakDemandForecaster:
    """Predict peak electricity demand for a given date."""

    def __init__(self):
        print("Loading peak demand FNN model and preprocessing objects...")
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model('peak_demand_model.keras')
            
            with open('scaler_X_peak.pkl', 'rb') as f:
                self.scaler_X = pickle.load(f)
            with open('scaler_y_peak.pkl', 'rb') as f:
                self.scaler_y = pickle.load(f)
            with open('features_peak.pkl', 'rb') as f:
                self.features = pickle.load(f)
            try:
                y_train_scaled = np.load('y_train_peak.npy')
                y_train_scaled_2d = y_train_scaled.reshape(-1, 1)
                y_train_real = self.scaler_y.inverse_transform(y_train_scaled_2d).flatten()
                self.default_demand = float(np.mean(y_train_real))
            except FileNotFoundError:
                self.default_demand = 3000.0
        except Exception as e:
            print(f"Error loading peak model file: {e}")
            raise
            
        print("Peak model loaded successfully!")

    def predict_peak(self, date_str):
        dt = pd.to_datetime(date_str)
        month = dt.month
        day_of_week = dt.dayofweek
        day_of_month = dt.day
        quarter = dt.quarter
        is_weekend = 1 if day_of_week >= 5 else 0
        
        values = {
            'Month': month,
            'DayOfWeek': day_of_week,
            'DayOfMonth': day_of_month,
            'Quarter': quarter,
            'IsWeekend': is_weekend,
            'Month_sin': np.sin(2 * np.pi * month / 12),
            'Month_cos': np.cos(2 * np.pi * month / 12),
            'Dow_sin': np.sin(2 * np.pi * day_of_week / 7),
            'Dow_cos': np.cos(2 * np.pi * day_of_week / 7)
        }
        
        if 'PeakDemand_lag1' in self.features:
            values['PeakDemand_lag1'] = self.default_demand
        if 'PeakDemand_lag7' in self.features:
            values['PeakDemand_lag7'] = self.default_demand
        if 'PeakDemand_roll7' in self.features:
            values['PeakDemand_roll7'] = self.default_demand
        
        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        feature_vector = np.array([values[f] for f in self.features]).reshape(1, -1)
        feature_scaled = self.scaler_X.transform(feature_vector)
        y_pred_scaled = self.model.predict(feature_scaled, verbose=0).reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]
        
        return {
            'date': dt.strftime('%Y-%m-%d'),
            'predicted_peak_demand_mw': round(float(y_pred), 2)
        }


class LSTMDemandForecaster:
    """Predict electricity demand for any date/time using LSTM and sequences."""

    def __init__(self):
        print("Loading trained LSTM model and preprocessing objects...")
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model('electricity_lstm_model.keras')

            with open('scaler_X_lstm.pkl', 'rb') as f:
                self.scaler_X = pickle.load(f)

            with open('scaler_y_lstm.pkl', 'rb') as f:
                self.scaler_y = pickle.load(f)

            with open('features_lstm.pkl', 'rb') as f:
                self.features = pickle.load(f)

            try:
                y_train_scaled = np.load('y_train_lstm.npy')
                y_train_scaled_2d = y_train_scaled.reshape(-1, 1)
                y_train_real = self.scaler_y.inverse_transform(y_train_scaled_2d).flatten()
                self.default_demand = float(np.mean(y_train_real))
            except FileNotFoundError:
                self.default_demand = 1500.0

        except Exception as e:
            print(f"Error loading LSTM file: {e}")
            raise

        self.time_steps = 24
        print("LSTM Model loaded successfully!")

    def create_sequence(self, dt, temp, humidity, rainfall, wind_speed):
        sequence = []
        for i in range(self.time_steps, 0, -1):
            past_dt = dt - timedelta(hours=i)
            hour = past_dt.hour
            day_of_week = past_dt.dayofweek
            month = past_dt.month
            day_of_month = past_dt.day
            quarter = (month - 1) // 3 + 1
            is_weekend = 1 if day_of_week >= 5 else 0
            
            values = {
                'Demand': self.default_demand,
                'Temperature': temp,
                'Humidity': humidity,
                'Rain': rainfall,
                'WindSpeed': wind_speed,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'Month': month,
                'DayOfMonth': day_of_month,
                'Quarter': quarter,
                'IsWeekend': is_weekend,
                'Hour_sin': np.sin(2 * np.pi * hour / 24),
                'Hour_cos': np.cos(2 * np.pi * hour / 24),
                'Dow_sin': np.sin(2 * np.pi * day_of_week / 7),
                'Dow_cos': np.cos(2 * np.pi * day_of_week / 7)
            }
            
            if 'Rainfall' in self.features and 'Rain' not in self.features:
                 values['Rainfall'] = rainfall
            
            missing = [f for f in self.features if f not in values]
            if missing:
                raise ValueError(f"Missing features: {missing}")

            feature_vector = [values[f] for f in self.features]
            sequence.append(feature_vector)
            
        sequence_array = np.array(sequence)
        sequence_scaled = self.scaler_X.transform(sequence_array)
        return sequence_scaled.reshape(1, self.time_steps, len(self.features))

    def predict_demand(self, datetime_str, temp, humidity, rainfall, wind_speed):
        dt = pd.to_datetime(datetime_str)
        X_seq_scaled = self.create_sequence(dt, temp, humidity, rainfall, wind_speed)
        
        y_pred_scaled = self.model.predict(X_seq_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]

        return {
            'datetime': datetime_str,
            'predicted_demand_mw': round(float(y_pred), 2),
            'temperature': temp,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': wind_speed
        }

    def predict_hourly_forecast(self, start_datetime, temp, humidity, rainfall, wind_speed, hours=24):
        forecasts = []
        dt = pd.to_datetime(start_datetime)
        
        for i in range(hours):
            current_dt = dt + timedelta(hours=i)
            temp_adj = temp + 5 * np.sin(np.pi * current_dt.hour / 12 - np.pi/2) - 5 * np.sin(np.pi * dt.hour / 12 - np.pi/2)
            
            result = self.predict_demand(
                current_dt.strftime('%Y-%m-%d %H:%M:%S'),
                temp_adj, humidity, rainfall, wind_speed
            )
            forecasts.append({
                'hour': current_dt.strftime('%H:%M'),
                'date': current_dt.strftime('%Y-%m-%d'),
                'demand': result['predicted_demand_mw']
            })
            
        return forecasts

# Initialize forecaster
try:
    forecaster = LSTMDemandForecaster()
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    forecaster = None

try:
    peak_forecaster = PeakDemandForecaster()
except Exception as e:
    print(f"Warning: Could not load peak model - {e}")
    peak_forecaster = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/peak-demand')
def peak_demand():
    return render_template('peak.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if forecaster is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.json
        
        datetime_str = f"{data['date']} {data['time']}:00"
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        rainfall = float(data['rainfall'])
        wind_speed = float(data['windSpeed'])
        
        result = forecaster.predict_demand(
            datetime_str, temperature, humidity, rainfall, wind_speed
        )
        
        # Get hourly forecast
        hourly = forecaster.predict_hourly_forecast(
            datetime_str, temperature, humidity, rainfall, wind_speed, hours=24
        )
        
        result['hourly_forecast'] = hourly
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict-peak', methods=['POST'])
def predict_peak():
    if peak_forecaster is None:
        return jsonify({'error': 'Peak Model not loaded. Please train the peak model first.'}), 500
    
    try:
        data = request.json
        date_str = data['date']
        
        result = peak_forecaster.predict_peak(date_str)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/model-info')
def model_info():
    if forecaster is None:
        return jsonify({'loaded': False})
    
    return jsonify({
        'loaded': True,
        'features': forecaster.features,
        'default_demand': forecaster.default_demand
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)

```

## `evaluate_models.py`

```python
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
```

## `prepare_lstm_data.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("Data Preparation for LSTM")
print("=" * 60)

# ===================== LOAD DATA =====================
df = pd.read_csv('electricity_data.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')
df = df.sort_values('DateTime').reset_index(drop=True)

# ===================== BASIC TIME FEATURES =====================
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek
df['Month'] = df['DateTime'].dt.month
df['DayOfMonth'] = df['DateTime'].dt.day
df['Quarter'] = df['DateTime'].dt.quarter
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# ===================== CYCLICAL ENCODING FOR TIME =====================
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Dow_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['Dow_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# We are NOT computing explicit rolling or lag columns here like Demand_lag1.
# The LSTM will handle the "lag" naturally by looking at the whole 24h sequence.

df = df.dropna().reset_index(drop=True)

# ===================== SELECT FEATURES & TARGET =====================
features = [
    'Demand', # Keeping past demand in the sequence
    'Temperature',
    'Humidity',
    'Rain',
    'WindSpeed',
    'Hour',
    'DayOfWeek',
    'Month',
    'DayOfMonth',
    'Quarter',
    'IsWeekend',
    'Hour_sin',
    'Hour_cos',
    'Dow_sin',
    'Dow_cos'
]

target = 'Demand'

X_df = df[features].copy()
y_df = df[target].copy()

# ===================== NORMALIZE DATA =====================
scaler_X_lstm = MinMaxScaler()
scaler_y_lstm = MinMaxScaler()

X_scaled = scaler_X_lstm.fit_transform(X_df)
y_scaled = scaler_y_lstm.fit_transform(y_df.values.reshape(-1, 1)).flatten()

# ===================== CREATE SEQUENCES FOR LSTM =====================
# For LSTM, we need to create a 3D array: [samples, time_steps, features]
TIME_STEPS = 24 # Use past 24 hours to predict the next hour

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)]) # Past 24 hours of features
        ys.append(y[i + time_steps])     # 25th hour target demand
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

print(f"Original X shape: {X_scaled.shape}")
print(f"Sequence X shape: {X_seq.shape}")
print(f"Sequence y shape: {y_seq.shape}")

# ===================== TRAIN-TEST SPLIT =====================
split_idx = int(len(X_seq) * 0.8)

X_train_lstm = X_seq[:split_idx]
X_test_lstm = X_seq[split_idx:]
y_train_lstm = y_seq[:split_idx]
y_test_lstm = y_seq[split_idx:]

print(f"\nTrain-Test Split:")
print(f"  Training: {len(X_train_lstm):,} samples")
print(f"  Testing:  {len(X_test_lstm):,} samples")

# ===================== SAVE ARTIFACTS =====================
np.save('X_train_lstm.npy', X_train_lstm)
np.save('X_test_lstm.npy', X_test_lstm)
np.save('y_train_lstm.npy', y_train_lstm)
np.save('y_test_lstm.npy', y_test_lstm)

with open('scaler_X_lstm.pkl', 'wb') as f:
    pickle.dump(scaler_X_lstm, f)

with open('scaler_y_lstm.pkl', 'wb') as f:
    pickle.dump(scaler_y_lstm, f)

with open('features_lstm.pkl', 'wb') as f:
    pickle.dump(features, f)

print("\nSaved LSTM sequence files and scalers successfully!")
```

## `prepare_peak_data.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("Preparing Peak Demand Data (from Main Dataset)")
print("=" * 60)

# ===================== LOAD DATA =====================
df = pd.read_csv('electricity_data.csv')

# DateTime format in electricity_data.csv is 'DD-MM-YYYY HH:MM'
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')
df['Date'] = df['DateTime'].dt.date

print(f"Loaded main electricity data: {df.shape}")

# ===================== AGGREGATE DAILY PEAK =====================
daily_df = df.groupby('Date').agg(
    PeakDemand=('Demand', 'max')
).reset_index()

daily_df['Date'] = pd.to_datetime(daily_df['Date'])

print(f"Aggregated daily peak data: {daily_df.shape}")

# Sort by date
daily_df = daily_df.sort_values('Date').reset_index(drop=True)

# ===================== ENGINEER FEATURES =====================
daily_df['Month'] = daily_df['Date'].dt.month
daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek  # Monday=0
daily_df['DayOfMonth'] = daily_df['Date'].dt.day
daily_df['Quarter'] = daily_df['Date'].dt.quarter
daily_df['IsWeekend'] = (daily_df['DayOfWeek'] >= 5).astype(int)

# Cyclical encoding for Month and DayOfWeek
daily_df['Month_sin'] = np.sin(2 * np.pi * daily_df['Month'] / 12)
daily_df['Month_cos'] = np.cos(2 * np.pi * daily_df['Month'] / 12)
daily_df['Dow_sin'] = np.sin(2 * np.pi * daily_df['DayOfWeek'] / 7)
daily_df['Dow_cos'] = np.cos(2 * np.pi * daily_df['DayOfWeek'] / 7)

# Lag capabilities
daily_df['PeakDemand_lag1'] = daily_df['PeakDemand'].shift(1)
daily_df['PeakDemand_lag7'] = daily_df['PeakDemand'].shift(7)
daily_df['PeakDemand_roll7'] = daily_df['PeakDemand'].rolling(window=7, min_periods=7).mean()

# Drop NaNs
daily_df = daily_df.dropna().reset_index(drop=True)

features = [
    'Month',
    'DayOfWeek',
    'DayOfMonth',
    'Quarter',
    'IsWeekend',
    'Month_sin',
    'Month_cos',
    'Dow_sin',
    'Dow_cos',
    'PeakDemand_lag1',
    'PeakDemand_lag7',
    'PeakDemand_roll7'
]

target = 'PeakDemand'

X = daily_df[features].copy()
y = daily_df[target].copy()

print(f"\nFeatures ({len(features)}):")
for i, feat in enumerate(features, 1):
    print(f"  {i:2d}. {feat}")

# ===================== NORMALIZE DATA =====================
scaler_X_peak = MinMaxScaler()
scaler_y_peak = MinMaxScaler()

X_scaled = scaler_X_peak.fit_transform(X)
y_scaled = scaler_y_peak.fit_transform(y.values.reshape(-1, 1)).flatten()

# ===================== TRAIN-TEST SPLIT =====================
# chronological split
split_idx = int(len(X_scaled) * 0.8)

X_train_peak = X_scaled[:split_idx]
X_test_peak = X_scaled[split_idx:]
y_train_peak = y_scaled[:split_idx]
y_test_peak = y_scaled[split_idx:]

print(f"\nTrain-Test Split (Peak Demand):")
print(f"  Training: {len(X_train_peak):,} samples")
print(f"  Testing:  {len(X_test_peak):,} samples")

# ===================== SAVE ARTIFACTS =====================
np.save('X_train_peak.npy', X_train_peak)
np.save('X_test_peak.npy', X_test_peak)
np.save('y_train_peak.npy', y_train_peak)
np.save('y_test_peak.npy', y_test_peak)

with open('scaler_X_peak.pkl', 'wb') as f:
    pickle.dump(scaler_X_peak, f)

with open('scaler_y_peak.pkl', 'wb') as f:
    pickle.dump(scaler_y_peak, f)

with open('features_peak.pkl', 'wb') as f:
    pickle.dump(features, f)

print(f"\nPeak data preparation complete! Files saved.")
```

## `train_gb_models.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_demand_gb():
    print("Training Demand Forecasting GB Model...")
    df = pd.read_csv('electricity_data.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')
    df = df.sort_values('DateTime').reset_index(drop=True)

    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['Month'] = df['DateTime'].dt.month
    df['DayOfMonth'] = df['DateTime'].dt.day
    df['Quarter'] = df['DateTime'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Dow_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['Dow_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    features = [
        'Temperature', 'Humidity', 'Rain', 'WindSpeed',
        'Hour', 'DayOfWeek', 'Month', 'DayOfMonth', 'Quarter', 'IsWeekend',
        'Hour_sin', 'Hour_cos', 'Dow_sin', 'Dow_cos'
    ]
    target = 'Demand'

    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    X = df[features]
    y = df[target]

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    
    train_preds = gb.predict(X_train)
    test_preds = gb.predict(X_test)
    
    print("\n--- Demand GB Model Evaluation ---")
    print(f"Train MAE:  {mean_absolute_error(y_train, train_preds):.2f}")
    print(f"Test MAE:   {mean_absolute_error(y_test, test_preds):.2f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.2f}")
    print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_test, test_preds)):.2f}")
    print(f"Train R2:   {r2_score(y_train, train_preds):.4f}")
    print(f"Test R2:    {r2_score(y_test, test_preds):.4f}\n")

    joblib.dump(gb, 'demand_model_gb.pkl')
    with open('scaler_X_demand_gb.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('features_demand_gb.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("Demand GB Model saved successfully.")

def train_peak_gb():
    print("Training Peak Demand GB Model...")
    df = pd.read_csv('electricity_data.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')
    df['Date'] = df['DateTime'].dt.date

    daily_df = df.groupby('Date').agg(PeakDemand=('Demand', 'max')).reset_index()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df = daily_df.sort_values('Date').reset_index(drop=True)

    daily_df['Month'] = daily_df['Date'].dt.month
    daily_df['DayOfWeek'] = daily_df['Date'].dt.dayofweek
    daily_df['DayOfMonth'] = daily_df['Date'].dt.day
    daily_df['Quarter'] = daily_df['Date'].dt.quarter
    daily_df['IsWeekend'] = (daily_df['DayOfWeek'] >= 5).astype(int)

    daily_df['Month_sin'] = np.sin(2 * np.pi * daily_df['Month'] / 12)
    daily_df['Month_cos'] = np.cos(2 * np.pi * daily_df['Month'] / 12)
    daily_df['Dow_sin'] = np.sin(2 * np.pi * daily_df['DayOfWeek'] / 7)
    daily_df['Dow_cos'] = np.cos(2 * np.pi * daily_df['DayOfWeek'] / 7)

    daily_df['PeakDemand_lag1'] = daily_df['PeakDemand'].shift(1)
    daily_df['PeakDemand_lag7'] = daily_df['PeakDemand'].shift(7)
    daily_df['PeakDemand_roll7'] = daily_df['PeakDemand'].rolling(window=7, min_periods=7).mean()

    daily_df = daily_df.dropna().reset_index(drop=True)

    features = [
        'Month', 'DayOfWeek', 'DayOfMonth', 'Quarter', 'IsWeekend',
        'Month_sin', 'Month_cos', 'Dow_sin', 'Dow_cos',
        'PeakDemand_lag1', 'PeakDemand_lag7', 'PeakDemand_roll7'
    ]
    target = 'PeakDemand'

    X = daily_df[features]
    y = daily_df[target]

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)

    train_preds = gb.predict(X_train)
    test_preds = gb.predict(X_test)
    
    print("\n--- Peak GB Model Evaluation ---")
    print(f"Train MAE:  {mean_absolute_error(y_train, train_preds):.2f}")
    print(f"Test MAE:   {mean_absolute_error(y_test, test_preds):.2f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, train_preds)):.2f}")
    print(f"Test RMSE:  {np.sqrt(mean_squared_error(y_test, test_preds)):.2f}")
    print(f"Train R2:   {r2_score(y_train, train_preds):.4f}")
    print(f"Test R2:    {r2_score(y_test, test_preds):.4f}\n")

    joblib.dump(gb, 'peak_model_gb.pkl')
    with open('scaler_X_peak_gb.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('features_peak_gb.pkl', 'wb') as f:
        pickle.dump(features, f)
        
   

if __name__ == '__main__':
    train_demand_gb()
    train_peak_gb()
```

## `train_lstm_model.py`

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle
import random
import os

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print("Training LSTM Demand Forecasting Model")
print("=" * 60)

# Load data
X_train = np.load('X_train_lstm.npy')
X_test = np.load('X_test_lstm.npy')
y_train = np.load('y_train_lstm.npy')
y_test = np.load('y_test_lstm.npy')

with open('scaler_y_lstm.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Model Architecture
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('electricity_lstm_model.keras', save_best_only=True, monitor='val_loss')

# Train Model
print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate
print("\nEvaluating on Test Data...")
loss, mae = model.evaluate(X_test, y_test)

# Calculate R2 Score
from sklearn.metrics import r2_score
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

r2 = r2_score(y_true, y_pred)

print("\nModel Performance Summary:")
print(f"  Test Loss (MSE): {loss:.4f}")
print(f"  Test MAE (scaled): {mae:.4f}")
print(f"  R2 Score: {r2:.4f}")

print("\nModel saved to: electricity_lstm_model.keras")
```

## `train_peak_model.py`

```python


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from sklearn.metrics import r2_score
import random
import os

os.environ['PYTHONHASHSEED'] = '40'
random.seed(40)
np.random.seed(40)
tf.random.set_seed(40)

print("Training Peak Demand FNN Forecasting Model")
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
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('peak_demand_model.keras', save_best_only=True, monitor='val_loss')

# Train Model
print("\nStarting Training...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# Evaluate
print("\nEvaluating on Test Data...")
loss, mae = model.evaluate(X_test, y_test)

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y_peak.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_true = scaler_y_peak.inverse_transform(y_test.reshape(-1, 1))

r2 = r2_score(y_true, y_pred)

print("\nModel Performance Summary:")
print(f"  Test Loss (MSE): {loss:.4f}")
print(f"  Test MAE (scaled): {mae:.4f}")
print(f"  Test R2 Score: {r2:.4f}")

print("\nPeak Demand Model saved to: peak_demand_model.keras")
```

## `use_gb_models.py`

```python
import pandas as pd
import numpy as np
import pickle
import joblib

def predict_demand_gb(temperature, humidity, rain, wind_speed, datetime_str):
    print(f"\n--- Predicting Hourly Demand (GB Model) for {datetime_str} ---")
    
    # 1. Load the model and preprocessing objects
    model = joblib.load('demand_model_gb.pkl')
    with open('scaler_X_demand_gb.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('features_demand_gb.pkl', 'rb') as f:
        features = pickle.load(f)
        
    # 2. Extract time features
    dt = pd.to_datetime(datetime_str)
    hour = dt.hour
    day_of_week = dt.dayofweek
    month = dt.month
    day_of_month = dt.day
    quarter = dt.quarter
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 3. Create feature dictionary
    values = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Rain': rain,
        'WindSpeed': wind_speed,
        'Hour': hour,
        'DayOfWeek': day_of_week,
        'Month': month,
        'DayOfMonth': day_of_month,
        'Quarter': quarter,
        'IsWeekend': is_weekend,
        'Hour_sin': np.sin(2 * np.pi * hour / 24),
        'Hour_cos': np.cos(2 * np.pi * hour / 24),
        'Dow_sin': np.sin(2 * np.pi * day_of_week / 7),
        'Dow_cos': np.cos(2 * np.pi * day_of_week / 7)
    }
    
    # 4. Prepare and scale feature array
    feature_df = pd.DataFrame([[values[f] for f in features]], columns=features)
    feature_scaled = scaler_X.transform(feature_df)
    
    # 5. Predict
    prediction = model.predict(feature_scaled)[0]
    print(f"Predicted Demand: {prediction:.2f} MW")
    return prediction


def predict_peak_gb(date_str, default_peak_demand=3000.0):
    print(f"\n--- Predicting Peak Demand (GB Model) for {date_str} ---")
    
    # 1. Load the model and preprocessing objects
    model = joblib.load('peak_model_gb.pkl')
    with open('scaler_X_peak_gb.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('features_peak_gb.pkl', 'rb') as f:
        features = pickle.load(f)
        
    # 2. Extract time features
    dt = pd.to_datetime(date_str)
    month = dt.month
    day_of_week = dt.dayofweek
    day_of_month = dt.day
    quarter = dt.quarter
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 3. Create feature dictionary (Using default average value for lag features)
    values = {
        'Month': month,
        'DayOfWeek': day_of_week,
        'DayOfMonth': day_of_month,
        'Quarter': quarter,
        'IsWeekend': is_weekend,
        'Month_sin': np.sin(2 * np.pi * month / 12),
        'Month_cos': np.cos(2 * np.pi * month / 12),
        'Dow_sin': np.sin(2 * np.pi * day_of_week / 7),
        'Dow_cos': np.cos(2 * np.pi * day_of_week / 7),
        'PeakDemand_lag1': default_peak_demand,  
        'PeakDemand_lag7': default_peak_demand,  
        'PeakDemand_roll7': default_peak_demand  
    }
    
    # 4. Prepare and scale feature array
    feature_df = pd.DataFrame([[values[f] for f in features]], columns=features)
    feature_scaled = scaler_X.transform(feature_df)
    
    # 5. Predict
    prediction = model.predict(feature_scaled)[0]
    print(f"Predicted Daily Peak Demand: {prediction:.2f} MW")
    return prediction


if __name__ == '__main__':
    print("=== Electricity Demand Forecasting (GB Models) ===")
    
    print("\n--- Input Parameters for Hourly Demand Models ---")
    date_input = input("Enter Date (YYYY-MM-DD) ")
    if not date_input.strip(): date_input = "2026-04-15"
    
    time_input = input("Enter Time (HH:MM) ")
    if not time_input.strip(): time_input = "14:00"
    
    datetime_str = f"{date_input} {time_input}"
    
    try:
        temp_input = input("Enter Temperature (°C) [Default 28.5]: ")
        temperature = float(temp_input) if temp_input.strip() else 28.5
        
        hum_input = input("Enter Humidity (%) [Default 45.0]: ")
        humidity = float(hum_input) if hum_input.strip() else 45.0
        
        rain_input = input("Enter Rainfall (mm) [Default 0.0]: ")
        rain = float(rain_input) if rain_input.strip() else 0.0
        
        wind_input = input("Enter Wind Speed (km/h) [Default 12.5]: ")
        wind_speed = float(wind_input) if wind_input.strip() else 12.5
        
        predict_demand_gb(
            temperature=temperature, 
            humidity=humidity, 
            rain=rain, 
            wind_speed=wind_speed, 
            datetime_str=datetime_str
        )
    except ValueError:
        print("Invalid number entered. Please run again and enter valid numbers.")
    except Exception as e:
        print(f"Error predicting demand: {e}")
        
    print("\n--- Input Parameters for Daily Peak Model ---")
    peak_date_input = input("Enter Date for Peak Prediction (YYYY-MM-DD) [Default 2026-04-15]: ")
    if not peak_date_input.strip(): peak_date_input = "2026-04-15"
    
    try:
        predict_peak_gb(date_str=peak_date_input)
    except Exception as e:
        print(f"Error predicting peak demand: {e}")

```

## `use_models.py`

```python

from app import LSTMDemandForecaster, PeakDemandForecaster

def print_header(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)

def main():
    print("Initializing CLI... Loading models, please wait.")
    
    # Load forecasters
    try:
        # Redirect stdout briefly to suppress model loading messages if desired, 
        # but the classes print loading messages which is fine.
        lstm_forecaster = LSTMDemandForecaster()
        peak_forecaster = PeakDemandForecaster()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    while True:
        print_header("Demand Forecasting - Interactive CLI")
        print("1. Predict Hourly Demand (LSTM)")
        print("2. Predict Peak Demand (FNN)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print_header("LSTM Hourly Demand Prediction")
            try:
                date_str = input("Enter Date (YYYY-MM-DD): ").strip()
                time_str = input("Enter Time (HH:MM): ").strip()
                datetime_str = f"{date_str} {time_str}:00"
                
                temp = float(input("Enter Temperature (°C): ").strip())
                humidity = float(input("Enter Humidity (%): ").strip())
                rainfall = float(input("Enter Rainfall (mm): ").strip())
                wind_speed = float(input("Enter Wind Speed (km/h): ").strip())
                
                print("\nGenerating prediction...")
                result = lstm_forecaster.predict_demand(
                    datetime_str, temp, humidity, rainfall, wind_speed
                )
                
                print("\n--- PREDICTION RESULT ---")
                print(f"Date & Time: {result['datetime']}")
                print(f"Predicted Demand: {result['predicted_demand_mw']:.2f} MW")
                print("-------------------------")
                
            except ValueError as ve:
                print(f"\n[!] Error: Invalid numeric input or date format. Details: {ve}")
            except Exception as e:
                print(f"\n[!] Error during prediction: {e}")
                
        elif choice == '2':
            print_header("FNN Peak Demand Prediction")
            try:
                date_str = input("Enter Date (YYYY-MM-DD): ").strip()
                
                print("\nGenerating prediction...")
                result = peak_forecaster.predict_peak(date_str)
                
                print("\n--- PREDICTION RESULT ---")
                print(f"Date: {result['date']}")
                print(f"Predicted Daily Peak Demand: {result['predicted_peak_demand_mw']:.2f} MW")
                print("-------------------------")
                
            except ValueError as ve:
                print(f"\n[!] Error: Invalid date format. Details: {ve}")
            except Exception as e:
                print(f"\n[!] Error during prediction: {e}")
                
        elif choice == '3':
            print("\nExiting. Goodbye!")
            break
        else:
            print("\n[!] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()
```

## `test_seeds.py`

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from sklearn.metrics import r2_score
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*60)
print("Testing Seeds for Models")

def test_peak_fnn():
    print("Testing FNN (Peak Demand)...")
    X_train = np.load('X_train_peak.npy')
    X_test = np.load('X_test_peak.npy')
    y_train = np.load('y_train_peak.npy')
    y_test = np.load('y_test_peak.npy')

    with open('scaler_y_peak.pkl', 'rb') as f:
        scaler_y_peak = pickle.load(f)

    for seed in range(40, 50):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = scaler_y_peak.inverse_transform(model.predict(X_test, verbose=0))
        y_true = scaler_y_peak.inverse_transform(y_test.reshape(-1, 1))
        
        r2 = r2_score(y_true, y_pred)
        print(f"FNN Seed: {seed} - R2: {r2:.4f}")
        with open('seed_results.txt', 'a') as f: f.write(f"FNN Seed: {seed} - R2: {r2:.4f}\n")

def test_lstm_demand():
    print("Testing LSTM (Demand)...")
    X_train = np.load('X_train_lstm.npy')
    X_test = np.load('X_test_lstm.npy')
    y_train = np.load('y_train_lstm.npy')
    y_test = np.load('y_test_lstm.npy')

    with open('scaler_y_lstm.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    for seed in range(40, 45): # Just 5 seeds for LSTM because it's slower
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        r2 = r2_score(y_true, y_pred)
        print(f"LSTM Seed: {seed} - R2: {r2:.4f}")
        with open('seed_results.txt', 'a') as f: f.write(f"LSTM Seed: {seed} - R2: {r2:.4f}\n")

if __name__ == '__main__':
    test_peak_fnn()
    test_lstm_demand()
```

## `templates/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PowerGrid AI | Electricity Demand Forecasting</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <!-- Animated Background -->
    <div class="bg-grid"></div>
    <div class="bg-glow bg-glow-1"></div>
    <div class="bg-glow bg-glow-2"></div>
    <div class="bg-glow bg-glow-3"></div>
    
    <!-- Floating Particles -->
    <div class="particles" id="particles"></div>

    <!-- Header -->
    <header class="header">
        <div class="logo">
            <div class="logo-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                </svg>
            </div>
            <span class="logo-text">PowerGrid<span class="logo-accent">AI</span></span>
        </div>
        <div class="header-status">
            <div class="status-dot"></div>
            <span>Model Active</span>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <section class="hero">
            <h1 class="hero-title">
                Electricity Demand <span class="gradient-text">Forecasting</span>
           </h1>
            

            <div class="nav-tabs" style="display: flex; gap: 15px; justify-content: center; margin-top: 25px;">
                <a href="/" class="nav-btn" style="padding: 10px 20px; border-radius: 8px; background: rgba(99, 102, 241, 0.2); color: #fff; text-decoration: none; border: 1px solid rgba(99, 102, 241, 0.5);">Hourly Forecast</a>
                <a href="/peak-demand" class="nav-btn" style="padding: 10px 20px; border-radius: 8px; background: rgba(255, 255, 255, 0.05); color: #a1a1aa; text-decoration: none; border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s;" onmouseover="this.style.background='rgba(255, 255, 255, 0.1)'" onmouseout="this.style.background='rgba(255, 255, 255, 0.05)'">Daily Peak Demand</a>
            </div>
        </section>

        <!-- Dashboard Grid -->
        <div class="dashboard">
            <!-- Input Panel -->
            <section class="panel input-panel">
                <div class="panel-header">
                    <div class="panel-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <path d="M12 6v6l4 2"/>
                        </svg>
                    </div>
                    <h2>Forecast Parameters</h2>
                </div>
                
                <form id="forecastForm" class="forecast-form">
                    <!-- Date Time Section -->
                    <div class="form-section">
                        <h3 class="section-title">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="4" width="18" height="18" rx="2"/>
                                <line x1="16" y1="2" x2="16" y2="6"/>
                                <line x1="8" y1="2" x2="8" y2="6"/>
                                <line x1="3" y1="10" x2="21" y2="10"/>
                            </svg>
                            Date & Time
                        </h3>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="date">Date</label>
                                <input type="date" id="date" name="date" required>
                            </div>
                            <div class="form-group">
                                <label for="time">Time</label>
                                <input type="time" id="time" name="time" required>
                            </div>
                        </div>
                    </div>

                    <!-- Weather Section -->
                    <div class="form-section">
                        <h3 class="section-title">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
                            </svg>
                            Weather Conditions
                        </h3>
                        
                        <div class="form-group slider-group">
                            <div class="slider-header">
                                <label for="temperature">Temperature</label>
                                <span class="slider-value"><span id="tempValue">25</span>°C</span>
                            </div>
                            <input type="range" id="temperature" name="temperature" min="0" max="60" value="25" class="premium-slider">
                            <div class="slider-labels">
                                <span>0°C</span>
                                <span>60°C</span>
                            </div>
                        </div>

                        <div class="form-group slider-group">
                            <div class="slider-header">
                                <label for="humidity">Humidity</label>
                                <span class="slider-value"><span id="humidityValue">50</span>%</span>
                            </div>
                            <input type="range" id="humidity" name="humidity" min="0" max="100" value="50" class="premium-slider">
                            <div class="slider-labels">
                                <span>0%</span>
                                <span>100%</span>
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label for="rainfall">Rainfall</label>
                                <div class="input-with-unit">
                                    <input type="number" id="rainfall" name="rainfall" min="0" step="0.1" value="0" placeholder="0">
                                    <span class="unit-suffix">mm</span>
                                </div>
                            </div>
                            <div class="form-group">
                                <label for="windSpeed">Wind Speed</label>
                                <div class="input-with-unit">
                                    <input type="number" id="windSpeed" name="windSpeed" min="0" max="100" step="0.1" value="10" placeholder="10">
                                    <span class="unit-suffix">km/h</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </form>
                
                <button type="submit" form="forecastForm" class="predict-btn" id="predictBtn">
                    <span class="btn-content">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                        </svg>
                        Generate Forecast
                    </span>
                    <span class="btn-loader">
                        <svg class="spinner" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" fill="none" stroke-dasharray="31.4 31.4"/>
                        </svg>
                    </span>
                </button>
            </section>

            <!-- Results Panel -->
            <section class="panel results-panel" id="resultsPanel">
                <div class="panel-header">
                    <div class="panel-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 3v18h18"/>
                            <path d="M18 9l-5 5-4-4-3 3"/>
                        </svg>
                    </div>
                    <h2>Prediction Results</h2>
                </div>

                <!-- Empty State -->
                <div class="empty-state" id="emptyState">
                    <div class="empty-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                        </svg>
                    </div>
                    <h3>Ready to Forecast</h3>
                    <p>Configure the parameters and generate your electricity demand prediction</p>
                </div>

                <!-- Results Content -->
                <div class="results-content" id="resultsContent" style="display: none;">
                    <!-- Main Prediction Card -->
                    <div class="prediction-card">
                        <div class="prediction-label">Predicted Demand</div>
                        <div class="prediction-value">
                            <span class="value-number" id="demandValue">0</span>
                            <span class="value-unit">MW</span>
                        </div>
                        <div class="prediction-meta" id="predictionMeta">
                            --
                        </div>
                    </div>

                    <!-- Stats Grid -->
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-icon temp-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/>
                                </svg>
                            </div>
                            <div class="stat-info">
                                <span class="stat-value" id="statTemp">--</span>
                                <span class="stat-label">Temperature</span>
                            </div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon humidity-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>
                                </svg>
                            </div>
                            <div class="stat-info">
                                <span class="stat-value" id="statHumidity">--</span>
                                <span class="stat-label">Humidity</span>
                            </div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon rain-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/>
                                    <path d="M11 21v-1"/>
                                    <path d="M7 21v-1"/>
                                    <path d="M15 21v-1"/>
                                </svg>
                            </div>
                            <div class="stat-info">
                                <span class="stat-value" id="statRain">--</span>
                                <span class="stat-label">Rainfall</span>
                            </div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-icon wind-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/>
                                    <path d="M9.6 4.6A2 2 0 1 1 11 8H2"/>
                                    <path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>
                                </svg>
                            </div>
                            <div class="stat-info">
                                <span class="stat-value" id="statWind">--</span>
                                <span class="stat-label">Wind Speed</span>
                            </div>
                        </div>
                    </div>

                    <!-- Chart -->
                    <div class="chart-container">
                        <h3 class="chart-title">24-Hour Forecast</h3>
                        <canvas id="forecastChart"></canvas>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <!-- Footer -->
    <!-- <footer class="footer">
        <p>Powered by Gradient Boosting Machine Learning Algorithm</p>
    </footer> -->

    <script src="/static/js/app.js"></script>
</body>
</html>

```

## `templates/peak.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PowerGrid AI | Electricity Demand Forecasting</title>
    
    <!-- Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <!-- Animated Background -->
    <div class="bg-grid"></div>
    <div class="bg-glow bg-glow-1"></div>
    <div class="bg-glow bg-glow-2"></div>
    <div class="bg-glow bg-glow-3"></div>
    
    <!-- Floating Particles -->
    <div class="particles" id="particles"></div>

    <!-- Header -->
    <header class="header">
        <div class="logo">
            <div class="logo-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                </svg>
            </div>
            <span class="logo-text">PowerGrid<span class="logo-accent">AI</span></span>
        </div>
        <div class="header-status">
            <div class="status-dot"></div>
            <span>Model Active</span>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-badge">
                <span class="badge-pulse"></span>
                Machine Learning Powered
            </div>
            <h1 class="hero-title">
                Electricity Demand
                <span class="gradient-text">Forecasting</span>
            </h1>
            <!-- <p class="hero-subtitle">
                Advanced AI-powered predictions using our best-fit Regression algorithms to forecast daily peak electricity demand.
            </p> -->
            <div class="nav-tabs" style="display: flex; gap: 15px; justify-content: center; margin-top: 25px;">
                <a href="/" class="nav-btn" style="padding: 10px 20px; border-radius: 8px; background: rgba(255, 255, 255, 0.05); color: #a1a1aa; text-decoration: none; border: 1px solid rgba(255, 255, 255, 0.1); transition: all 0.3s;" onmouseover="this.style.background='rgba(255, 255, 255, 0.1)'" onmouseout="this.style.background='rgba(255, 255, 255, 0.05)'">Hourly Forecast</a>
                <a href="/peak-demand" class="nav-btn" style="padding: 10px 20px; border-radius: 8px; background: rgba(99, 102, 241, 0.2); color: #fff; text-decoration: none; border: 1px solid rgba(99, 102, 241, 0.5);">Daily Peak Demand</a>
            </div>
        </section>

        <!-- Dashboard Grid -->
        <div class="dashboard">
            <!-- Input Panel -->
            <section class="panel input-panel">
                <div class="panel-header">
                    <div class="panel-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"/>
                            <path d="M12 6v6l4 2"/>
                        </svg>
                    </div>
                    <h2>Forecast Parameters</h2>
                </div>
                
                <form id="forecastForm" class="forecast-form">
                    <!-- Date Time Section -->
                    <div class="form-section">
                        <h3 class="section-title">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="4" width="18" height="18" rx="2"/>
                                <line x1="16" y1="2" x2="16" y2="6"/>
                                <line x1="8" y1="2" x2="8" y2="6"/>
                                <line x1="3" y1="10" x2="21" y2="10"/>
                            </svg>
                            Forecast Date
                        </h3>
                        <div class="form-row">
                            <div class="form-group" style="width: 100%;">
                                <label for="date">Date</label>
                                <input type="date" id="date" name="date" required>
                                
                                <!-- Removed time input and weather section for Peak Demand -->
                            </div>
                        </div>
                    </div>

                    <button type="submit" class="predict-btn" id="predictBtn">
                        <span class="btn-content">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                            </svg>
                            Generate Forecast
                        </span>
                        <span class="btn-loader">
                            <svg class="spinner" viewBox="0 0 24 24">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" fill="none" stroke-dasharray="31.4 31.4"/>
                            </svg>
                        </span>
                    </button>
                </form>
            </section>

            <!-- Results Panel -->
            <section class="panel results-panel" id="resultsPanel">
                <div class="panel-header">
                    <div class="panel-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M3 3v18h18"/>
                            <path d="M18 9l-5 5-4-4-3 3"/>
                        </svg>
                    </div>
                    <h2>Prediction Results</h2>
                </div>

                <!-- Empty State -->
                <div class="empty-state" id="emptyState">
                    <div class="empty-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                        </svg>
                    </div>
                    <h3>Ready to Forecast</h3>
                    <p>Configure the parameters and generate your electricity demand prediction</p>
                </div>

                <!-- Results Content -->
                <div class="results-content" id="resultsContent" style="display: none;">
                    <!-- Main Prediction Card -->
                    <div class="prediction-card">
                        <div class="prediction-label">Predicted Demand</div>
                        <div class="prediction-value">
                            <span class="value-number" id="demandValue">0</span>
                            <span class="value-unit">MW</span>
                        </div>
                        <div class="prediction-meta" id="predictionMeta">
                            --
                        </div>
                    </div>

                    <!-- Removed Stats Grid and Chart for Peak Demand -->
                </div>
            </section>
        </div>
    </main>

    <!-- Footer -->
    <!-- <footer class="footer">
        <p>Powered by Gradient Boosting Machine Learning Algorithm</p>
    </footer> -->

    <script src="/static/js/peak.js"></script>
</body>
</html>

```

## `static/css/style.css`

```css
/* ===================================================
   PowerGrid AI - Premium Electricity Demand Forecasting
   Premium Dark Theme with Elegant Animations
   =================================================== */

/* CSS Variables */
:root {
    /* Colors - Deep Dark Neumorphic Palette */
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-tertiary: #1a1a25;
    --bg-card: #15151e;
    --bg-input: #0e0e14;
    
    /* Accent Colors */
    --accent-primary: #8b5cf6;
    --accent-secondary: #a855f7;
    --accent-tertiary: #c084fc;
    --accent-cyan: #22d3ee;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    
    /* Gradients */
    --gradient-primary: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
    --gradient-glow: linear-gradient(135deg, rgba(139, 92, 246, 0.3) 0%, rgba(168, 85, 247, 0.1) 100%);
    
    /* Text */
    --text-primary: #ffffff;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    
    /* Borders */
    --border-color: rgba(255, 255, 255, 0.05);
    --border-glow: rgba(139, 92, 246, 0.4);
    
    /* Shadows */
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 32px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 16px 64px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.3);
    
    /* Spacing */
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --radius-xl: 24px;
    
    /* Transitions */
    --transition-fast: 0.15s ease;
    --transition-base: 0.3s ease;
    --transition-slow: 0.5s ease;
}

/* Reset & Base */
*, *::before, *::after {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
    line-height: 1.6;
}

/* Animated Background */
.bg-grid {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
    animation: gridPulse 20s ease-in-out infinite;
}

@keyframes gridPulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.5; }
}

.bg-glow {
    position: fixed;
    border-radius: 50%;
    filter: blur(120px);
    pointer-events: none;
    z-index: 0;
    animation: glowFloat 15s ease-in-out infinite;
}

.bg-glow-1 {
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    top: -200px;
    left: -200px;
    animation-delay: 0s;
}

.bg-glow-2 {
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.12) 0%, transparent 70%);
    bottom: -100px;
    right: -100px;
    animation-delay: -5s;
}

.bg-glow-3 {
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(34, 211, 238, 0.08) 0%, transparent 70%);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    animation-delay: -10s;
}

@keyframes glowFloat {
    0%, 100% { transform: translate(0, 0) scale(1); }
    25% { transform: translate(30px, -30px) scale(1.05); }
    50% { transform: translate(-20px, 20px) scale(0.95); }
    75% { transform: translate(-30px, -20px) scale(1.02); }
}

/* Particles */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
    overflow: hidden;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: var(--accent-primary);
    border-radius: 50%;
    opacity: 0;
    animation: particleFloat 8s ease-in-out infinite;
}

@keyframes particleFloat {
    0% {
        opacity: 0;
        transform: translateY(100vh) scale(0);
    }
    10% {
        opacity: 0.6;
    }
    90% {
        opacity: 0.6;
    }
    100% {
        opacity: 0;
        transform: translateY(-100vh) scale(1);
    }
}

/* Header */
.header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    background: rgba(10, 10, 15, 0.8);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-color);
    z-index: 100;
    animation: slideDown 0.6s ease-out;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-100%);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.logo-icon {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--gradient-primary);
    border-radius: var(--radius-md);
    color: white;
    animation: logoPulse 3s ease-in-out infinite;
}

@keyframes logoPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.4); }
    50% { box-shadow: 0 0 20px 5px rgba(99, 102, 241, 0.2); }
}

.logo-icon svg {
    width: 22px;
    height: 22px;
}

.logo-text {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.logo-accent {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.header-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-radius: 100px;
    font-size: 0.875rem;
    color: var(--accent-emerald);
}

.status-dot {
    width: 8px;
    height: 8px;
    background: var(--accent-emerald);
    border-radius: 50%;
    animation: statusPulse 2s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* Main Content */
.main-content {
    position: relative;
    z-index: 2;
    padding: 100px 40px 40px;
    max-width: 1400px;
    margin: 0 auto;
}

/* Hero Section */
.hero {
    text-align: center;
    margin-bottom: 60px;
    animation: fadeInUp 0.8s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 100px;
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--accent-primary);
    margin-bottom: 24px;
    animation: fadeInUp 0.8s ease-out 0.1s backwards;
}

.badge-pulse {
    width: 8px;
    height: 8px;
    background: var(--accent-primary);
    border-radius: 50%;
    animation: badgePulse 2s ease-in-out infinite;
}

@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.6); }
    50% { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }
}

.hero-title {
    font-size: clamp(2.5rem, 6vw, 4rem);
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 20px;
    animation: fadeInUp 0.8s ease-out 0.2s backwards;
}

.gradient-text {
    display: inline-block;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.125rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
    animation: fadeInUp 0.8s ease-out 0.3s backwards;
}

/* Dashboard Grid */
.dashboard {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 30px;
    animation: fadeInUp 0.8s ease-out 0.4s backwards;
}

@media (max-width: 1024px) {
    .dashboard {
        grid-template-columns: 1fr;
    }
}

/* Panels */
.panel {
    background: var(--bg-card); /* Solid inner box look */
    border: 1px solid var(--border-color); /* Subtle inner shadow/border */
    border-radius: var(--radius-xl);
    padding: 32px; 
    transition: all var(--transition-base);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.panel:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-glow);
}

.panel-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
}

.panel-icon {
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--gradient-primary);
    border-radius: var(--radius-md);
    color: white;
}

.panel-icon svg {
    width: 22px;
    height: 22px;
}

.panel-header h2 {
    font-size: 1.25rem;
    font-weight: 600;
}

/* Form Styles */
.forecast-form {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.form-section {
    padding: 24px;
    background: #0e0e14; /* Inner box dark background */
    border-radius: var(--radius-xl);
    border: 1px solid rgba(255, 255, 255, 0.02);
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
}

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 18px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title svg {
    width: 18px;
    height: 18px;
    color: var(--accent-primary);
}

.form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.form-group label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.form-group input[type="date"],
.form-group input[type="time"],
.form-group input[type="number"] {
    width: 100%;
    padding: 12px 16px;
    background: #08080c; /* Deep pill background */
    border: 1px solid rgba(255, 255, 255, 0.02);
    border-radius: 12px;
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    transition: all var(--transition-base);
    outline: none;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

.form-group input:focus {
    border-color: rgba(139, 92, 246, 0.5);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2), 0 0 0 2px rgba(139, 92, 246, 0.15);
}

.form-group input::placeholder {
    color: var(--text-muted);
}

/* Input with Unit Suffix */
.input-with-unit {
    position: relative;
    display: flex;
    align-items: center;
}

.input-with-unit input {
    width: 100%;
    padding-right: 60px; /* Space for the suffix */
}

.input-with-unit .unit-suffix {
    position: absolute;
    right: 16px;
    color: var(--text-muted);
    font-size: 0.875rem;
    font-weight: 500;
    pointer-events: none;
}

/* Premium Slider */
.slider-group {
    gap: 12px;
}

.slider-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.slider-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
    padding: 4px 12px;
    border-radius: var(--radius-sm);
}

.premium-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 8px;
    background: rgba(99, 102, 241, 0.15);
    border-radius: 100px;
    outline: none;
    cursor: pointer;
    transition: all var(--transition-base);
}

.premium-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 24px;
    height: 24px;
    background: var(--gradient-primary);
    border-radius: 50%;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    transition: all var(--transition-base);
}

.premium-slider::-webkit-slider-thumb:hover {
    transform: scale(1.15);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
}

.premium-slider::-moz-range-thumb {
    width: 24px;
    height: 24px;
    background: var(--gradient-primary);
    border-radius: 50%;
    cursor: pointer;
    border: none;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}

.slider-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Predict Button */
.predict-btn {
    position: relative;
    width: 100%;
    padding: 16px 32px;
    background: var(--gradient-primary);
    border: none;
    border-radius: 16px; /* Pill shape */
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    overflow: hidden;
    transition: all var(--transition-base);
    margin-top: 24px;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.2);
}

.predict-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: all 0.5s ease;
}

.predict-btn:hover::before {
    left: 100%;
}

.predict-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(139, 92, 246, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.2);
}

.predict-btn:active {
    transform: translateY(0);
}

.btn-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
}

.btn-content svg {
    width: 22px;
    height: 22px;
}

.btn-loader {
    display: none;
}

.predict-btn.loading .btn-content {
    display: none;
}

.predict-btn.loading .btn-loader {
    display: flex;
    align-items: center;
    justify-content: center;
}

.spinner {
    width: 24px;
    height: 24px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}


/* Empty State */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 60px 40px;
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.empty-icon {
    width: 80px;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(139, 92, 246, 0.1);
    border-radius: 50%;
    margin-bottom: 24px;
}

.empty-icon svg {
    width: 50px;
    height: 50px;
    color: var(--accent-primary);
    opacity: 0.6;
}

.empty-state h3 {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.empty-state p {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

/* Results Content */
.results-content {
    animation: fadeInUp 0.6s ease-out;
}

/* Prediction Card */
.prediction-card {
    text-align: center;
    padding: 32px;
    background: #0e0e14; /* Solid dark background */
    border: 1px solid rgba(139, 92, 246, 0.4); /* Purple glow border */
    border-radius: var(--radius-xl);
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.05) 0%, transparent 60%);
    animation: cardGlow 6s ease-in-out infinite;
}

@keyframes cardGlow {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(10%, 10%); }
}

.prediction-label {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    position: relative;
}

.prediction-value {
    display: flex;
    align-items: baseline;
    justify-content: center;
    gap: 8px;
    margin-bottom: 16px;
    position: relative;
}

.value-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(3rem, 8vw, 4.5rem);
    font-weight: 700;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    text-shadow: 0 0 20px rgba(99, 102, 241, 0.4); /* Subtle outer glow matching primary brand color */
}

.value-unit {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--accent-primary);
}

.prediction-meta {
    font-size: 0.9rem;
    color: var(--text-muted);
    position: relative;
}

/* Stats Grid */
.stats-grid {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 24px;
}

.stat-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px 12px 12px;
    background: #0e0e14;
    border: 1px solid rgba(255, 255, 255, 0.02);
    border-radius: 100px; /* Pill shape */
    transition: all var(--transition-base);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    flex: 1; /* Stretch properly */
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-glow);
}

.stat-icon {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px; /* Slightly rounded icons inside pill */
}

.stat-icon svg {
    width: 18px;
    height: 18px;
}

.temp-icon {
    background: rgba(244, 63, 94, 0.15);
    color: var(--accent-rose);
    transition: all var(--transition-base);
}

.temp-icon.hot {
    background: rgba(245, 158, 11, 0.2);
    color: var(--accent-amber);
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.4);
}

.humidity-icon {
    background: rgba(34, 211, 238, 0.15);
    color: var(--accent-cyan);
}

.rain-icon {
    background: rgba(99, 102, 241, 0.15);
    color: var(--accent-primary);
}

.wind-icon {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-emerald);
}

.stat-info {
    display: flex;
    flex-direction: column;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
}

.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* Chart Container */
.chart-container {
    padding: 24px;
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
}

.chart-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 20px;
    color: var(--text-secondary);
}

/* Footer */
.footer {
    text-align: center;
    padding: 40px;
    color: var(--text-muted);
    font-size: 0.875rem;
}

/* Animations for Results */
@keyframes countUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.stat-card:nth-child(1) { animation: fadeInUp 0.4s ease-out 0.1s backwards; }
.stat-card:nth-child(2) { animation: fadeInUp 0.4s ease-out 0.2s backwards; }
.stat-card:nth-child(3) { animation: fadeInUp 0.4s ease-out 0.3s backwards; }
.stat-card:nth-child(4) { animation: fadeInUp 0.4s ease-out 0.4s backwards; }

/* Responsive */
@media (max-width: 768px) {
    .header {
        padding: 0 20px;
    }
    
    .main-content {
        padding: 90px 20px 20px;
    }
    
    .form-row {
        grid-template-columns: 1fr;
    }
    
    .stats-grid {
        grid-template-columns: 1fr;
    }
    
    .hero-title {
        font-size: 2rem;
    }
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
    background: var(--accent-primary);
    border-radius: 100px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-secondary);
}

/* Selection */
::selection {
    background: var(--accent-primary);
    color: white;
}

```

## `static/js/app.js`

```javascript
/**
 * PowerGrid AI - Electricity Demand Forecasting
 * Premium Interactive UI Script
 */

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    createParticles();
    setDefaultDateTime();
    initializeSliders();
    initializeForm();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Check model status
    checkModelStatus();
    
    // Add smooth reveal animations
    observeElements();
}

/**
 * Check if the ML model is loaded
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        const statusEl = document.querySelector('.header-status');
        if (data.loaded) {
            statusEl.innerHTML = `
                <div class="status-dot"></div>
                <span>Model Active</span>
            `;
            statusEl.style.borderColor = 'rgba(16, 185, 129, 0.2)';
            statusEl.style.background = 'rgba(16, 185, 129, 0.1)';
            statusEl.style.color = '#10b981';
        } else {
            statusEl.innerHTML = `
                <div class="status-dot" style="background: #f59e0b;"></div>
                <span>Model Not Loaded</span>
            `;
            statusEl.style.borderColor = 'rgba(245, 158, 11, 0.2)';
            statusEl.style.background = 'rgba(245, 158, 11, 0.1)';
            statusEl.style.color = '#f59e0b';
        }
    } catch (error) {
        console.log('Could not check model status');
    }
}

/**
 * Create floating particles animation
 */
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random properties
        const size = Math.random() * 4 + 2;
        const left = Math.random() * 100;
        const delay = Math.random() * 8;
        const duration = Math.random() * 4 + 6;
        const opacity = Math.random() * 0.5 + 0.2;
        
        // Random color from palette
        const colors = ['#6366f1', '#8b5cf6', '#22d3ee', '#10b981', '#a855f7'];
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${left}%;
            background: ${color};
            animation-delay: ${delay}s;
            animation-duration: ${duration}s;
            opacity: ${opacity};
        `;
        
        particlesContainer.appendChild(particle);
    }
}

/**
 * Set default date and time to current
 */
function setDefaultDateTime() {
    const now = new Date();
    
    // Format date as YYYY-MM-DD
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    document.getElementById('date').value = `${year}-${month}-${day}`;
    
    // Format time as HH:MM
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    document.getElementById('time').value = `${hours}:${minutes}`;
}

/**
 * Initialize premium sliders
 */
function initializeSliders() {
    // Temperature slider
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    
    tempSlider.addEventListener('input', (e) => {
        tempValue.textContent = e.target.value;
        updateSliderBackground(e.target);
    });
    updateSliderBackground(tempSlider);
    
    // Humidity slider
    const humiditySlider = document.getElementById('humidity');
    const humidityValue = document.getElementById('humidityValue');
    
    humiditySlider.addEventListener('input', (e) => {
        humidityValue.textContent = e.target.value;
        updateSliderBackground(e.target);
    });
    updateSliderBackground(humiditySlider);
}

/**
 * Update slider track fill based on value
 */
function updateSliderBackground(slider) {
    const min = slider.min || 0;
    const max = slider.max || 100;
    const value = slider.value;
    const percentage = ((value - min) / (max - min)) * 100;
    
    // Update thermometer icon if it's the temperature slider
    if (slider.id === 'temperature') {
        const tempIcon = document.querySelector('.temp-icon');
        if (tempIcon) {
            if (value > 30) {
                tempIcon.classList.add('hot');
            } else {
                tempIcon.classList.remove('hot');
            }
        }
    }
    
    slider.style.background = `linear-gradient(to right, 
        rgba(99, 102, 241, 0.8) 0%, 
        rgba(139, 92, 246, 0.8) ${percentage}%, 
        rgba(99, 102, 241, 0.15) ${percentage}%, 
        rgba(99, 102, 241, 0.15) 100%)`;
}

/**
 * Initialize form submission
 */
function initializeForm() {
    const form = document.getElementById('forecastForm');
    const predictBtn = document.getElementById('predictBtn');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        
        // Collect form data
        const formData = {
            date: document.getElementById('date').value,
            time: document.getElementById('time').value,
            temperature: document.getElementById('temperature').value,
            humidity: document.getElementById('humidity').value,
            rainfall: document.getElementById('rainfall').value,
            windSpeed: document.getElementById('windSpeed').value
        };
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                displayResults(result);
            } else {
                showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            showError('Connection error. Please make sure the server is running.');
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });
}

/**
 * Display prediction results with animations
 */
function displayResults(data) {
    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');
    
    // Hide empty state, show results
    emptyState.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Animate demand value
    animateCounter('demandValue', data.predicted_demand_mw, 0, 1500);
    
    // Update prediction meta
    const predictionDate = new Date(data.datetime);
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    document.getElementById('predictionMeta').textContent = 
        predictionDate.toLocaleDateString('en-US', options);
    
    // Update stats with animation
    setTimeout(() => {
        document.getElementById('statTemp').textContent = `${data.temperature}°C`;
    }, 200);
    setTimeout(() => {
        document.getElementById('statHumidity').textContent = `${data.humidity}%`;
    }, 300);
    setTimeout(() => {
        document.getElementById('statRain').textContent = `${data.rainfall} mm`;
    }, 400);
    setTimeout(() => {
        document.getElementById('statWind').textContent = `${data.wind_speed} km/h`;
    }, 500);
    
    // Update chart
    if (data.hourly_forecast) {
        updateChart(data.hourly_forecast);
    }
    
    // Scroll to results on mobile
    if (window.innerWidth < 1024) {
        document.getElementById('resultsPanel').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }
}

/**
 * Animate counter from start to end
 */
function animateCounter(elementId, endValue, startValue = 0, duration = 1000) {
    const element = document.getElementById(elementId);
    const startTime = performance.now();
    const range = endValue - startValue;
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out-expo)
        const easeOutExpo = 1 - Math.pow(2, -10 * progress);
        const currentValue = startValue + (range * easeOutExpo);
        
        element.textContent = currentValue.toFixed(2);
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

/**
 * Custom Chart.js Plugin for vertical hover line
 */
const verticalLinePlugin = {
    id: 'verticalLine',
    afterDraw: (chart) => {
        if (chart.tooltip?._active?.length) {
            const ctx = chart.ctx;
            const x = chart.tooltip._active[0].element.x;
            const topY = chart.scales.y.top;
            const bottomY = chart.scales.y.bottom;

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(x, topY);
            ctx.lineTo(x, bottomY);
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.restore();
        }
    }
};

/**
 * Update the forecast chart
 */
let forecastChart = null;

function updateChart(hourlyData) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    const labels = hourlyData.map(d => d.hour);
    const data = hourlyData.map(d => d.demand);
    
    // Confidence Interval (+/- 5%)
    const upperData = data.map(v => v * 1.05);
    const lowerData = data.map(v => v * 0.95);
    // Destroy existing chart
    if (forecastChart) {
        forecastChart.destroy();
    }
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');
    
    const datasets = [
        {
            label: 'Predicted Demand (MW)',
            data: data,
            borderColor: '#6366f1',
            backgroundColor: gradient,
            borderWidth: 3,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 8,
            pointHoverBackgroundColor: '#6366f1',
            pointHoverBorderColor: '#ffffff',
            pointHoverBorderWidth: 3,
            order: 2
        },
        {
            label: 'Upper Bound',
            data: upperData,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            borderWidth: 0,
            fill: false,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 0,
            order: 1
        },
        {
            label: 'Lower Bound',
            data: lowerData,
            borderColor: 'transparent',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderWidth: 0,
            fill: '-1', // Fill to previous dataset (Upper Bound)
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 0,
            order: 3
        }
    ];
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#ffffff',
                    titleColor: '#000000',
                    bodyColor: '#1a1a1a',
                    borderColor: 'rgba(255, 255, 255, 1)',
                    borderWidth: 0,
                    cornerRadius: 100, /* Pill shape */
                    padding: {
                        top: 8,
                        bottom: 8,
                        left: 16,
                        right: 16
                    },
                    titleFont: {
                        family: 'Outfit',
                        size: 13,
                        weight: '600'
                    },
                    bodyFont: {
                        family: 'Outfit',
                        size: 13
                    },
                    displayColors: false,
                    caretSize: 0, /* No pointer triangle */
                    yAlign: 'bottom',
                    filter: function(item) {
                       return !item.dataset.label.includes('Bound');
                    },
                    callbacks: {
                        title: () => null, /* Hide default title */
                        label: (item) => `Predicted Load: ${item.raw.toFixed(2)} MW | Time: ${item.label} | Temp: ${document.getElementById('tempValue').textContent}°C`
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.02)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.3)',
                        font: {
                            family: 'Outfit',
                            size: 11
                        },
                        maxRotation: 45
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.02)',
                        drawBorder: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.3)',
                        font: {
                            family: 'Outfit',
                            size: 11
                        },
                        callback: (value) => value.toFixed(0) + ' MW'
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        },
        plugins: [verticalLinePlugin]
    });
}

/**
 * Show error message
 */
function showError(message) {
    // Create error notification
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>${message}</span>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 24px;
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        color: #f43f5e;
        font-size: 0.95rem;
        z-index: 1000;
        animation: slideInRight 0.4s ease-out;
    `;
    
    notification.querySelector('svg').style.cssText = `
        width: 20px;
        height: 20px;
        flex-shrink: 0;
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease-out forwards';
        setTimeout(() => notification.remove(), 400);
    }, 5000);
}

// Add keyframe animations via JavaScript
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
`;
document.head.appendChild(style);

/**
 * Intersection Observer for scroll animations
 */
function observeElements() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.panel, .stat-card').forEach(el => {
        observer.observe(el);
    });
}

```

## `static/js/peak.js`

```javascript
/**
 * PowerGrid AI - Peak Demand Forecasting
 * Premium Interactive UI Script
 */

// Initialize application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    createParticles();
    setDefaultDate();
    initializeForm();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Check model status
    checkModelStatus();
    
    // Add smooth reveal animations
    observeElements();
}

/**
 * Check if the ML model is loaded
 */
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        const statusEl = document.querySelector('.header-status');
        if (data.loaded) {
            statusEl.innerHTML = `
                <div class="status-dot"></div>
                <span>Peak Model Active</span>
            `;
            statusEl.style.borderColor = 'rgba(16, 185, 129, 0.2)';
            statusEl.style.background = 'rgba(16, 185, 129, 0.1)';
            statusEl.style.color = '#10b981';
        } else {
            statusEl.innerHTML = `
                <div class="status-dot" style="background: #f59e0b;"></div>
                <span>Model Not Loaded</span>
            `;
            statusEl.style.borderColor = 'rgba(245, 158, 11, 0.2)';
            statusEl.style.background = 'rgba(245, 158, 11, 0.1)';
            statusEl.style.color = '#f59e0b';
        }
    } catch (error) {
        console.log('Could not check model status');
    }
}

/**
 * Create floating particles animation
 */
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 4 + 2;
        const left = Math.random() * 100;
        const delay = Math.random() * 8;
        const duration = Math.random() * 4 + 6;
        const opacity = Math.random() * 0.5 + 0.2;
        
        const colors = ['#6366f1', '#8b5cf6', '#22d3ee', '#10b981', '#a855f7'];
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${left}%;
            background: ${color};
            animation-delay: ${delay}s;
            animation-duration: ${duration}s;
            opacity: ${opacity};
        `;
        
        particlesContainer.appendChild(particle);
    }
}

/**
 * Set default date to current
 */
function setDefaultDate() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    document.getElementById('date').value = `${year}-${month}-${day}`;
}

/**
 * Initialize form submission
 */
function initializeForm() {
    const form = document.getElementById('forecastForm');
    const predictBtn = document.getElementById('predictBtn');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        predictBtn.classList.add('loading');
        predictBtn.disabled = true;
        
        // Collect form data (only date needed for peak)
        const formData = {
            date: document.getElementById('date').value,
        };
        
        try {
            const response = await fetch('/api/predict-peak', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                displayResults(result);
            } else {
                showError(result.error || 'Prediction failed');
            }
        } catch (error) {
            showError('Connection error. Please make sure the server is running.');
        } finally {
            predictBtn.classList.remove('loading');
            predictBtn.disabled = false;
        }
    });
}

/**
 * Display prediction results with animations
 */
function displayResults(data) {
    const emptyState = document.getElementById('emptyState');
    const resultsContent = document.getElementById('resultsContent');
    
    // Hide empty state, show results
    emptyState.style.display = 'none';
    resultsContent.style.display = 'block';
    
    // Animate demand value
    animateCounter('demandValue', data.predicted_peak_demand_mw, 0, 1500);
    
    // Update prediction meta
    const predictionDate = new Date(data.date);
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric'
    };
    document.getElementById('predictionMeta').textContent = 
        predictionDate.toLocaleDateString('en-US', options);
    
    // Scroll to results on mobile
    if (window.innerWidth < 1024) {
        document.getElementById('resultsPanel').scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    }
}

/**
 * Animate counter from start to end
 */
function animateCounter(elementId, endValue, startValue = 0, duration = 1000) {
    const element = document.getElementById(elementId);
    const startTime = performance.now();
    const range = endValue - startValue;
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out-expo)
        const easeOutExpo = 1 - Math.pow(2, -10 * progress);
        const currentValue = startValue + (range * easeOutExpo);
        
        element.textContent = currentValue.toFixed(2);
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

/**
 * Show error message
 */
function showError(message) {
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <span>${message}</span>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 24px;
        background: rgba(244, 63, 94, 0.15);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 12px;
        color: #f43f5e;
        font-size: 0.95rem;
        z-index: 1000;
        animation: slideInRight 0.4s ease-out;
    `;
    
    notification.querySelector('svg').style.cssText = `
        width: 20px;
        height: 20px;
        flex-shrink: 0;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.4s ease-out forwards';
        setTimeout(() => notification.remove(), 400);
    }, 5000);
}

const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideOutRight {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100px); }
    }
`;
document.head.appendChild(style);

function observeElements() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.panel, .stat-card').forEach(el => {
        observer.observe(el);
    });
}
```

## `requirements.txt`

```text
pandas
numpy
scikit-learn
matplotlib
flask
flask-cors
tensorflow
```

## `read_me.md`

```markdown

```bash
pip install pandas numpy scikit-learn matplotlib flask flask-cors tensorflow
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

# run
```bash
python prepare_lstm_data.py
python train_lstm_model.py
python train_peak_model.py
python app.py
```

