import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("Data Preparation")
print("=" * 60)

# ===================== LOAD DATA =====================
df = pd.read_csv('electricity_data.csv')

# Adjust this format string if your DateTime looks different
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')

print(f"Loaded data: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Make sure data is sorted in time
df = df.sort_values('DateTime').reset_index(drop=True)

# ===================== BASIC TIME FEATURES =====================
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # Monday=0
df['Month'] = df['DateTime'].dt.month
df['DayOfMonth'] = df['DateTime'].dt.day
df['Quarter'] = df['DateTime'].dt.quarter
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# ===================== LAG & ROLLING FEATURES =====================
# Demand one hour ago
df['Demand_lag1'] = df['Demand'].shift(1)

# Demand 24 hours ago (same hour previous day)
df['Demand_lag24'] = df['Demand'].shift(24)

# Average demand over the last 24 hours
df['Demand_roll24'] = df['Demand'].rolling(window=24, min_periods=24).mean()

# ===================== CYCLICAL ENCODING FOR TIME =====================
# Hour of day: 0–23
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

# Day of week: 0–6
df['Dow_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['Dow_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# Drop rows with NaNs created by shift/rolling
df = df.dropna().reset_index(drop=True)

print(f"After feature engineering and dropping NaNs: {df.shape}")

# ===================== SELECT FEATURES & TARGET =====================
# Note: using 'Rain' (not 'Rainfall'), and no 'AQI'
features = [
    'Hour',
    'DayOfWeek',
    'Month',
    'Temperature',
    'Humidity',
    'Rain',
    'WindSpeed',
    'DayOfMonth',
    'Quarter',
    'IsWeekend',
    'Demand_lag1',
    'Demand_lag24',
    'Demand_roll24',
    'Hour_sin',
    'Hour_cos',
    'Dow_sin',
    'Dow_cos'
]

target = 'Demand'

missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing feature columns in DataFrame: {missing}")

X = df[features].copy()
y = df[target].copy()

print(f"\nFeatures ({len(features)}):")
for i, feat in enumerate(features, 1):
    print(f"  {i:2d}. {feat}")

# ===================== NORMALIZE DATA =====================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print(f"\nData normalized to 0-1 range")
print(f"  X shape: {X_scaled.shape}")
print(f"  y shape: {y_scaled.shape}")

# ===================== TRAIN-TEST SPLIT (TIME-BASED) =====================
split_idx = int(len(X_scaled) * 0.8)

X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y_scaled[:split_idx]
y_test = y_scaled[split_idx:]

print(f"\nTrain-Test Split:")
print(f"  Training: {len(X_train):,} samples ({len(X_train)/len(X_scaled)*100:.1f}%)")
print(f"  Testing:  {len(X_test):,} samples ({len(X_test)/len(X_scaled)*100:.1f}%)")
print(f"  Train dates: {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[split_idx-1]}")
print(f"  Test dates:  {df['DateTime'].iloc[split_idx]} to {df['DateTime'].iloc[-1]}")

# ===================== SAVE ARTIFACTS =====================
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)

with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

print(f"\nAll files saved!")
print(f"  - X_train.npy, X_test.npy")
print(f"  - y_train.npy, y_test.npy")
print(f"  - scaler_X.pkl, scaler_y.pkl")
print(f"  - features.pkl")
