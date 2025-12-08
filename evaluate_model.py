import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("STEP 5: Model Evaluation")
print("=" * 60)

# Load data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load model and scaler
with open('gradient_boosting_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Make predictions
y_pred_scaled = model.predict(X_test)

# Convert to actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"""
MODEL EVALUATION ON TEST DATA
────────────────────────────

MAE (Mean Absolute Error):     {mae:.2f} MW
RMSE (Root Mean Squared):      {rmse:.2f} MW
R² Score:                       {r2:.4f}

Interpretation:
  • Average prediction error: ±{mae:.0f} MW
  • Model explains {r2*100:.1f}% of demand variation
  • Accuracy: {r2*100:.1f}%
""")

# Show sample predictions
print(f"Sample Predictions (First 10 test samples):")
print(f"{'Actual':>10} | {'Predicted':>10} | {'Error':>10}")
print("─" * 35)

for i in range(min(10, len(y_actual))):
    actual = y_actual[i]
    pred = y_pred[i]
    error = abs(actual - pred)
    print(f"{actual:10.2f} | {pred:10.2f} | {error:10.2f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_actual, y_pred, alpha=0.5, s=10)
plt.plot([y_actual.min(), y_actual.max()], 
         [y_actual.min(), y_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Demand (MW)')
plt.ylabel('Predicted Demand (MW)')
plt.title('Actual vs Predicted Demand')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
errors = y_actual - y_pred
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (MW)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=100)
print(f"\n✓ Chart saved: model_evaluation.png")

plt.show()
