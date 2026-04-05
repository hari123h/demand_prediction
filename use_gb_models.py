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

