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


class DemandForecaster:
    """Predict electricity demand for any date/time with weather."""

    def __init__(self):
        """Load the trained model, scalers, features, and compute lag defaults."""
        print("Loading trained model and preprocessing objects...")

        try:
            with open('gradient_boosting_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            with open('scaler_X.pkl', 'rb') as f:
                self.scaler_X = pickle.load(f)

            with open('scaler_y.pkl', 'rb') as f:
                self.scaler_y = pickle.load(f)

            with open('features.pkl', 'rb') as f:
                self.features = pickle.load(f)

            try:
                y_train_scaled = np.load('y_train.npy')
                y_train_scaled_2d = y_train_scaled.reshape(-1, 1)
                y_train_real = self.scaler_y.inverse_transform(y_train_scaled_2d).flatten()
                self.default_demand = float(np.mean(y_train_real))
            except FileNotFoundError:
                self.default_demand = 1500.0

        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            raise

        print("Model loaded successfully!")

    def create_features(self, dt, temp, humidity, rainfall, wind_speed):
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month
        day_of_month = dt.day
        quarter = (month - 1) // 3 + 1
        is_weekend = 1 if day_of_week >= 5 else 0

        values = {}

        if 'Hour' in self.features:
            values['Hour'] = hour
        if 'DayOfWeek' in self.features:
            values['DayOfWeek'] = day_of_week
        if 'Month' in self.features:
            values['Month'] = month
        if 'DayOfMonth' in self.features:
            values['DayOfMonth'] = day_of_month
        if 'Quarter' in self.features:
            values['Quarter'] = quarter
        if 'IsWeekend' in self.features:
            values['IsWeekend'] = is_weekend

        if 'Hour_sin' in self.features:
            values['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
        if 'Hour_cos' in self.features:
            values['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
        if 'Dow_sin' in self.features:
            values['Dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        if 'Dow_cos' in self.features:
            values['Dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

        if 'Temperature' in self.features:
            values['Temperature'] = temp
        if 'Humidity' in self.features:
            values['Humidity'] = humidity

        if 'Rain' in self.features:
            values['Rain'] = rainfall
        if 'Rainfall' in self.features:
            values['Rainfall'] = rainfall

        if 'WindSpeed' in self.features:
            values['WindSpeed'] = wind_speed

        if 'AQI' in self.features:
            values['AQI'] = 100.0

        if 'Demand_lag1' in self.features:
            values['Demand_lag1'] = self.default_demand
        if 'Demand_lag24' in self.features:
            values['Demand_lag24'] = self.default_demand
        if 'Demand_roll24' in self.features:
            values['Demand_roll24'] = self.default_demand

        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        feature_vector = np.array([values[f] for f in self.features]).reshape(1, -1)
        feature_scaled = self.scaler_X.transform(feature_vector)
        return feature_scaled

    def predict_demand(self, datetime_str, temp, humidity, rainfall, wind_speed):
        dt = pd.to_datetime(datetime_str)
        X_scaled = self.create_features(dt, temp, humidity, rainfall, wind_speed)
        y_pred_scaled = self.model.predict(X_scaled).reshape(-1, 1)
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
        """Generate hourly forecast for next N hours."""
        forecasts = []
        dt = pd.to_datetime(start_datetime)
        
        for i in range(hours):
            current_dt = dt + timedelta(hours=i)
            result = self.predict_demand(
                current_dt.strftime('%Y-%m-%d %H:%M:%S'),
                temp, humidity, rainfall, wind_speed
            )
            forecasts.append({
                'hour': current_dt.strftime('%H:%M'),
                'date': current_dt.strftime('%Y-%m-%d'),
                'demand': result['predicted_demand_mw']
            })
        
        return forecasts


# Initialize forecaster
try:
    forecaster = DemandForecaster()
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    forecaster = None


@app.route('/')
def index():
    return render_template('index.html')


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

