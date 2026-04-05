import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from sklearn.metrics import r2_score
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*60)
print("Testing Seeds for Models")

def test_peak_fnn():
    print("Testing FNN (Peak Demand)...")
    X_train = np.load('X_train_peak.npy')
    X_test = np.load('X_test_peak.npy')
    y_train = np.load('y_train_peak.npy')
    y_test = np.load('y_test_peak.npy')

    with open('scaler_y_peak.pkl', 'rb') as f:
        scaler_y_peak = pickle.load(f)

    for seed in range(40, 50):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
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
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = scaler_y_peak.inverse_transform(model.predict(X_test, verbose=0))
        y_true = scaler_y_peak.inverse_transform(y_test.reshape(-1, 1))
        
        r2 = r2_score(y_true, y_pred)
        print(f"FNN Seed: {seed} - R2: {r2:.4f}")
        with open('seed_results.txt', 'a') as f: f.write(f"FNN Seed: {seed} - R2: {r2:.4f}\n")

def test_lstm_demand():
    print("Testing LSTM (Demand)...")
    X_train = np.load('X_train_lstm.npy')
    X_test = np.load('X_test_lstm.npy')
    y_train = np.load('y_train_lstm.npy')
    y_test = np.load('y_test_lstm.npy')

    with open('scaler_y_lstm.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    for seed in range(40, 45): # Just 5 seeds for LSTM because it's slower
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        r2 = r2_score(y_true, y_pred)
        print(f"LSTM Seed: {seed} - R2: {r2:.4f}")
        with open('seed_results.txt', 'a') as f: f.write(f"LSTM Seed: {seed} - R2: {r2:.4f}\n")

if __name__ == '__main__':
    test_peak_fnn()
    test_lstm_demand()
