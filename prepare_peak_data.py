import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

print("Preparing Peak Demand Data (from Main Dataset)")
print("=" * 60)

# ===================== LOAD DATA =====================
df = pd.read_csv('electricity_data.csv')

# DateTime format in electricity_data.csv is 'DD-MM-YYYY HH:MM'
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d-%m-%Y %H:%M')
df['Date'] = df['DateTime'].dt.date

print(f"Loaded main electricity data: {df.shape}")

# Calculate HeatIndex on hourly data before aggregation
df['HeatIndex'] = df['Temperature'] + 0.33 * df['Humidity'] - 0.7 * df['WindSpeed'] - 4.0

# ===================== AGGREGATE DAILY PEAK =====================
daily_df = df.groupby('Date').agg(
    PeakDemand=('Demand', 'max'),
    Temp_max=('Temperature', 'max'),
    Temp_mean=('Temperature', 'mean'),
    Humidity_mean=('Humidity', 'mean'),
    HeatIndex_max=('HeatIndex', 'max')
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
daily_df['PeakDemand_std7'] = daily_df['PeakDemand'].rolling(window=7, min_periods=7).std()

# Drop NaNs
daily_df = daily_df.dropna().reset_index(drop=True)

features = [
    'PeakDemand_lag1',
    'PeakDemand_lag7',
    'Temp_max',
    'Temp_mean',
    'Humidity_mean',
    'HeatIndex_max',
    'PeakDemand_roll7',
    'PeakDemand_std7',
    'IsWeekend',
    'Month_sin',
    'Month_cos'
]

target = 'PeakDemand'

X = daily_df[features].copy()
y = daily_df[target].copy()

# ===================== CORRELATION MATRIX =====================
corr_df = daily_df[features + [target]].copy()
corr = corr_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Peak Demand Correlation Matrix")
# plt.show()

corr.to_csv("peak_correlation_matrix.csv")

print(f"\nFeatures ({len(features)}):")
for i, feat in enumerate(features, 1):
    print(f"  {i:2d}. {feat}")

# ===================== NORMALIZE DATA =====================
scaler_X_peak = MinMaxScaler()
scaler_y_peak = MinMaxScaler()

X_scaled = scaler_X_peak.fit_transform(X)
y_scaled = scaler_y_peak.fit_transform(y.values.reshape(-1, 1)).flatten()

# ===================== TRAIN-TEST SPLIT (MONTHLY) =====================
target_months = daily_df['Date'].dt.to_period('M')
unique_months = target_months.unique()
split_month_idx = int(len(unique_months) * 0.8)
split_month = unique_months[split_month_idx]

train_mask = target_months < split_month
test_mask = target_months >= split_month

X_train_peak = X_scaled[train_mask]
X_test_peak = X_scaled[test_mask]
y_train_peak = y_scaled[train_mask]
y_test_peak = y_scaled[test_mask]

print(f"\nTrain-Test Split (Peak Demand - Split at {split_month}):")
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
