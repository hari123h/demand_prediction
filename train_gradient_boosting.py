import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("Train Gradient Boosting Model")

# UNDERSTAND GRADIENT BOOSTING 
print("""
• Builds many decision trees, one after another
• Each new tree corrects errors of previous trees
• Like learning from mistakes!
• Great for predicting numbers (regression)

""")

# Load prepared data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}")

# CREATE MODEL
print(f"\nCreating Gradient Boosting model...")

model = GradientBoostingRegressor(
    n_estimators=100,        # Build 100 trees
    max_depth=5,             # Each tree has max depth 5
    learning_rate=0.1,       # Learning speed
    subsample=0.8,           # Use 80% of samples for each tree
    random_state=42,         # For reproducibility
    verbose=0
)

# TRAIN MODEL
print(f"Training model on {len(X_train):,} samples...")
print("This may take a minute...")

model.fit(X_train, y_train)

print(f"Model trained!")

# TEST MODEL
print(f"\nMaking predictions...")

# Predictions on training data (to see if overfitting)
y_pred_train = model.predict(X_train)

# Predictions on testing data (real performance)
y_pred_test = model.predict(X_test)

#CALCULATE METRICS
# MAE: Mean Absolute Error (average error magnitude)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

# RMSE: Root Mean Square Error (penalizes large errors)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# R²: Coefficient of determination (0-1, higher is better)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

#DISPLAY RESULTS
print(f"""   GRADIENT BOOSTING MODEL PERFORMANCE

TRAINING SET 
  MAE:  {mae_train:.4f}  (average error)
  RMSE: {rmse_train:.4f}  (penalizes large errors)
  R²:   {r2_train:.4f}  (fit quality)

TESTING SET 
  MAE:  {mae_test:.4f}  (average error)
  RMSE: {rmse_test:.4f}  (penalizes large errors)
  R²:   {r2_test:.4f}  (fit quality) ← THIS IS THE REAL SCORE!

INTERPRETATION:
  • R² = {r2_test:.1%} accuracy on unseen data
  • Model explains {r2_test*100:.1f}% of demand variation
""")

# FEATURE IMPORTANCE
print(f"\nFeature Importance (what affects demand most?):")

features_importance = model.feature_importances_

with open('features.pkl', 'rb') as f:
    import pickle
    features = pickle.load(f)

importance_df = list(zip(features, features_importance))
importance_df.sort(key=lambda x: x, reverse=True)

for i, (feat, imp) in enumerate(importance_df, 1):
    bar = "█" * int(imp * 100)
    print(f"  {i:2d}. {feat:15s} {imp:7.2%} {bar}")

# SAVE MODEL
print(f"\nSaving model..")

with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to: gradient_boosting_model.pkl")

# Save predictions for later analysis
np.save('gb_y_pred_test.npy', y_pred_test)
print(f"Predictions saved")
