import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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

# ===================== NEW FEATURES =====================
df['HeatIndex'] = df['Temperature'] + 0.33 * df['Humidity'] - 0.7 * df['WindSpeed'] - 4.0

df['Demand_t-1'] = df['Demand'].shift(1)
df['Demand_t-24'] = df['Demand'].shift(24)
df['Demand_t-168'] = df['Demand'].shift(168)

df['RollingMean_24'] = df['Demand'].rolling(window=24).mean()
df['RollingMean_168'] = df['Demand'].rolling(window=168).mean()

# Drop rows with NaN values created by shifts and rolling means
df = df.dropna().reset_index(drop=True)

# ===================== SELECT FEATURES & TARGET =====================
features = [
    'Demand',
    'Temperature',
    'Humidity',
    'WindSpeed',
    'Rain',
    'Month',
    'IsWeekend',
    'Hour_sin',
    'Hour_cos',
    'Dow_sin',
    'Dow_cos',
    'HeatIndex',
    'Demand_t-1',
    'Demand_t-24',
    'Demand_t-168',
    'RollingMean_24',
    'RollingMean_168'
]

target = 'Demand'

# Ensure all features exist (e.g., Rain might be missing if previously removed, but should be kept according to instructions)
if 'Rain' not in df.columns:
    df['Rain'] = 0.0 # fallback if not present, though electricity_data.csv has it

X_df = df[features].copy()
y_df = df[target].copy()

# ===================== NORMALIZE DATA =====================
scaler_X_lstm = MinMaxScaler()
scaler_y_lstm = MinMaxScaler()

X_scaled = scaler_X_lstm.fit_transform(X_df)
y_scaled = scaler_y_lstm.fit_transform(y_df.values.reshape(-1, 1)).flatten()

# ===================== CREATE SEQUENCES FOR LSTM =====================
TIME_STEPS = 168 # 7 days
FORECAST_STEPS = 24 # Predict next 24 hours directly

def create_sequences(X, y, time_steps, forecast_steps):
    Xs, ys, dates = [], [], []
    for i in range(len(X) - time_steps - forecast_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps : i + time_steps + forecast_steps])
        dates.append(df['DateTime'].iloc[i + time_steps])
    return np.array(Xs), np.array(ys), pd.Series(dates)

X_seq, y_seq, target_dates = create_sequences(X_scaled, y_scaled, TIME_STEPS, FORECAST_STEPS)

print(f"Original X shape: {X_scaled.shape}")
print(f"Sequence X shape: {X_seq.shape}")
print(f"Sequence y shape: {y_seq.shape}")

# ===================== TRAIN-TEST SPLIT (MONTHLY) =====================
# We will split such that the first 80% of months are in training, and 20% in testing.
target_months = target_dates.dt.to_period('M')
unique_months = target_months.unique()
split_month_idx = int(len(unique_months) * 0.8)
split_month = unique_months[split_month_idx]

train_mask = target_months < split_month
test_mask = target_months >= split_month

X_train_lstm = X_seq[train_mask]
X_test_lstm = X_seq[test_mask]
y_train_lstm = y_seq[train_mask]
y_test_lstm = y_seq[test_mask]

print(f"\nTrain-Test Split (Split at {split_month}):")
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
