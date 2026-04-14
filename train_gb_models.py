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
        'Temperature', 'Humidity', 'WindSpeed',
        'Month', 'IsWeekend',
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
        'IsWeekend',
        'Month_sin', 'Month_cos', 'Dow_sin', 'Dow_cos',
        'PeakDemand_lag1', 'PeakDemand_lag7'
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
