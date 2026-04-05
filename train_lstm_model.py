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
