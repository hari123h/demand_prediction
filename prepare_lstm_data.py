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

# We are NOT computing explicit rolling or lag columns here like Demand_lag1.
# The LSTM will handle the "lag" naturally by looking at the whole 24h sequence.

df = df.dropna().reset_index(drop=True)

# ===================== SELECT FEATURES & TARGET =====================
features = [
    'Demand', # Keeping past demand in the sequence
    'Temperature',
    'Humidity',
    'WindSpeed',
    'Month',
    'IsWeekend',
    'Hour_sin',
    'Hour_cos',
    'Dow_sin',
    'Dow_cos'
]

target = 'Demand'

X_df = df[features].copy()
y_df = df[target].copy()
#correlation matrix
# ===================== CORRELATION MATRIX =====================
corr = X_df.corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Full Correlation Matrix")
plt.show()

# Optional: Save correlation matrix
corr.to_csv("correlation_matrix.csv")
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
