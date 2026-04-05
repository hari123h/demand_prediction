

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
