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
        print("Loading peak demand XGBoost model and preprocessing objects...")
        try:
            with open('peak_demand_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
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
        if 'PeakDemand_std7' in self.features:
            values['PeakDemand_std7'] = 200.0  # reasonable default standard deviation
            
        if 'Temp_max' in self.features:
            values['Temp_max'] = 25.0
        if 'Temp_mean' in self.features:
            values['Temp_mean'] = 20.0
        if 'Humidity_mean' in self.features:
            values['Humidity_mean'] = 70.0
        if 'HeatIndex_max' in self.features:
            values['HeatIndex_max'] = 25.0
        
        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        feature_vector = np.array([values[f] for f in self.features]).reshape(1, -1)
        feature_scaled = self.scaler_X.transform(feature_vector)
        y_pred_scaled = self.model.predict(feature_scaled).reshape(-1, 1)
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

        self.time_steps = 168
        print("LSTM Model loaded successfully!")

    def create_sequence(self, dt, temp, humidity, wind_speed):
        sequence = []
        heat_index = temp + 0.33 * humidity - 0.7 * wind_speed - 4.0
        
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
                'WindSpeed': wind_speed,
                'Rain': 0.0,
                'Hour': hour,
                'DayOfWeek': day_of_week,
                'Month': month,
                'DayOfMonth': day_of_month,
                'Quarter': quarter,
                'IsWeekend': is_weekend,
                'Hour_sin': np.sin(2 * np.pi * hour / 24),
                'Hour_cos': np.cos(2 * np.pi * hour / 24),
                'Dow_sin': np.sin(2 * np.pi * day_of_week / 7),
                'Dow_cos': np.cos(2 * np.pi * day_of_week / 7),
                'HeatIndex': heat_index,
                'Demand_t-1': self.default_demand,
                'Demand_t-24': self.default_demand,
                'Demand_t-168': self.default_demand,
                'RollingMean_24': self.default_demand,
                'RollingMean_168': self.default_demand
            }
            
            missing = [f for f in self.features if f not in values]
            if missing:
                raise ValueError(f"Missing features: {missing}")

            feature_vector = [values[f] for f in self.features]
            sequence.append(feature_vector)
            
        sequence_array = np.array(sequence)
        sequence_scaled = self.scaler_X.transform(sequence_array)
        return sequence_scaled.reshape(1, self.time_steps, len(self.features))

    def predict_demand(self, datetime_str, temp, humidity, wind_speed):
        dt = pd.to_datetime(datetime_str)
        X_seq_scaled = self.create_sequence(dt, temp, humidity, wind_speed)
        
        y_pred_scaled = self.model.predict(X_seq_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        return {
            'datetime': datetime_str,
            'predicted_demand_mw': round(float(y_pred[0]), 2),
            'temperature': temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'forecast_array': y_pred.tolist()
        }

    def predict_hourly_forecast(self, start_datetime, temp, humidity, wind_speed):
        # We predict 24 hours in a single pass now
        result = self.predict_demand(start_datetime, temp, humidity, wind_speed)
        forecasts = []
        dt = pd.to_datetime(start_datetime)
        
        for i, pred_val in enumerate(result['forecast_array']):
            current_dt = dt + timedelta(hours=i)
            forecasts.append({
                'hour': current_dt.strftime('%H:%M'),
                'date': current_dt.strftime('%Y-%m-%d'),
                'demand': round(pred_val, 2)
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
        wind_speed = float(data['windSpeed'])
        
        result = forecaster.predict_demand(
            datetime_str, temperature, humidity, wind_speed
        )
        
        # Get hourly forecast (which now simply uses the direct 24 output)
        hourly = forecaster.predict_hourly_forecast(
            datetime_str, temperature, humidity, wind_speed
        )
        
        result['hourly_forecast'] = hourly
        del result['forecast_array']  # Don't send this raw array back
        
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

