import pickle
import numpy as np
import pandas as pd


class DemandForecaster:
    """Predict electricity demand for any date/time with weather."""

    def __init__(self):
        """Load the trained model, scalers, features, and compute lag defaults."""
        print("Loading trained model and preprocessing objects...\n")

        try:
            with open('gradient_boosting_model.pkl', 'rb') as f:
                self.model = pickle.load(f)

            with open('scaler_X.pkl', 'rb') as f:
                self.scaler_X = pickle.load(f)

            with open('scaler_y.pkl', 'rb') as f:
                self.scaler_y = pickle.load(f)

            with open('features.pkl', 'rb') as f:
                self.features = pickle.load(f)

            # Optional: load y_train to derive a reasonable default demand
            try:
                y_train_scaled = np.load('y_train.npy')
                # y_train is scaled; inverse-transform to MW
                y_train_scaled_2d = y_train_scaled.reshape(-1, 1)
                y_train_real = self.scaler_y.inverse_transform(y_train_scaled_2d).flatten()
                self.default_demand = float(np.mean(y_train_real))
            except FileNotFoundError:
                # Fallback if y_train.npy not present
                self.default_demand = 1500.0  # arbitrary reasonable default
                print("Warning: y_train.npy not found. Using fixed default demand for lags.")

        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            print("Make sure you have run data preparation and model training first.")
            raise

        print("Loaded features used by the model:")
        for i, f in enumerate(self.features, 1):
            print(f"  {i:2d}. {f}")
        print(f"\nDefault demand value for lag features: {self.default_demand:.2f} MW")
        print("\nModel is ready.\n")

    def create_features(self, dt, temp, humidity, rainfall, wind_speed):
        """
        Create feature vector from datetime and user inputs,
        matching exactly the feature order used during training.

        Parameters
        ----------
        dt : datetime
        temp : float
        humidity : float
        rainfall : float
        wind_speed : float

        Returns
        -------
        np.ndarray
            Normalized feature vector with shape (1, n_features).
        """

        # Time components
        hour = dt.hour
        day_of_week = dt.dayofweek  # 0=Monday
        month = dt.month
        day_of_month = dt.day
        quarter = (month - 1) // 3 + 1
        is_weekend = 1 if day_of_week >= 5 else 0

        # Base feature dictionary
        values = {}

        # Common time-based features (only if model expects them)
        if 'Hour' in self.features:
            values['Hour'] = hour
        if 'DayOfWeek' in self.features:
            values['DayOfWeek'] = day_of_week
        if 'Month' in self.features:
            values['Month'] = month
        if 'DayOfMonth' in self.features:
            values['DayOfMonth'] = day_of_month
        if 'Quarter' in self.features:
            values['Quarter'] = quarter
        if 'IsWeekend' in self.features:
            values['IsWeekend'] = is_weekend

        # Cyclical encodings if present in the model
        if 'Hour_sin' in self.features:
            values['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
        if 'Hour_cos' in self.features:
            values['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
        if 'Dow_sin' in self.features:
            values['Dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        if 'Dow_cos' in self.features:
            values['Dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

        # Weather features
        if 'Temperature' in self.features:
            values['Temperature'] = temp
        if 'Humidity' in self.features:
            values['Humidity'] = humidity

        # Rain / Rainfall mapping
        if 'Rain' in self.features:
            values['Rain'] = rainfall
        if 'Rainfall' in self.features:
            values['Rainfall'] = rainfall

        if 'WindSpeed' in self.features:
            values['WindSpeed'] = wind_speed

        # AQI if model expects it (you likely removed this, but we support it)
        if 'AQI' in self.features:
            # If you want to use AQI, extend the input to collect it.
            # For now, default to a moderate AQI value.
            values['AQI'] = 100.0

        # Lag features: fill with default demand (approximation)
        if 'Demand_lag1' in self.features:
            values['Demand_lag1'] = self.default_demand
        if 'Demand_lag24' in self.features:
            values['Demand_lag24'] = self.default_demand
        if 'Demand_roll24' in self.features:
            values['Demand_roll24'] = self.default_demand

        # Final check: ensure we have everything required
        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(
                f"Still missing values for features required by the model: {missing}"
            )

        # Build feature vector in correct order
        feature_vector = np.array([values[f] for f in self.features]).reshape(1, -1)

        # Scale with the same scaler used during training
        feature_scaled = self.scaler_X.transform(feature_vector)
        return feature_scaled

    def predict_demand(self, datetime_str, temp, humidity, rainfall, wind_speed):
        """
        Predict electricity demand.

        Parameters
        ----------
        datetime_str : str
            Datetime string 'YYYY-MM-DD HH:MM:SS'.
        temp : float
        humidity : float
        rainfall : float
        wind_speed : float

        Returns
        -------
        dict
            Prediction result including demand in MW.
        """

        dt = pd.to_datetime(datetime_str)

        X_scaled = self.create_features(
            dt=dt,
            temp=temp,
            humidity=humidity,
            rainfall=rainfall,
            wind_speed=wind_speed
        )

        y_pred_scaled = self.model.predict(X_scaled).reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0, 0]

        return {
            'datetime': datetime_str,
            'predicted_demand_mw': round(float(y_pred), 2),
            'temperature': temp,
            'humidity': humidity,
            'rainfall': rainfall,
            'wind_speed': wind_speed
        }


def get_user_input():
    """Get input from user and return as parameters."""

    print("=" * 70)
    print("ELECTRICITY DEMAND FORECASTING SYSTEM")
    print("=" * 70)

    # Date/time
    print("\nDATE AND TIME")
    print("-" * 70)
    date_str = input("Enter date (YYYY-MM-DD): ").strip()
    time_str = input("Enter time (HH:MM:SS): ").strip()
    datetime_str = f"{date_str} {time_str}"

    # Weather parameters
    print("\nWEATHER INFORMATION")
    print("-" * 70)

    while True:
        try:
            temperature = float(input("Enter temperature (0-60°C): "))
            if not (0 <= temperature <= 60):
                print("Temperature must be between 0 and 60°C.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            humidity = float(input("Enter humidity (0-100%): "))
            if not (0 <= humidity <= 100):
                print("Humidity must be between 0 and 100%.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            rainfall = float(input("Enter rainfall (≥0 mm): "))
            if rainfall < 0:
                print("Rainfall cannot be negative.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            wind_speed = float(input("Enter wind speed (0-100 km/h): "))
            if not (0 <= wind_speed <= 100):
                print("Wind speed must be between 0 and 100 km/h.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    return datetime_str, temperature, humidity, rainfall, wind_speed


def display_prediction(result):
    """Format and print the prediction."""

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    print(f"\nDate/Time:       {result['datetime']}")
    print("\nInput conditions:")
    print(f"  Temperature:   {result['temperature']} °C")
    print(f"  Humidity:      {result['humidity']} %")
    print(f"  Rainfall:      {result['rainfall']} mm")
    print(f"  Wind Speed:    {result['wind_speed']} km/h")

    print("\n" + "-" * 70)
    print(f"Predicted demand: {result['predicted_demand_mw']} MW")
    print("-" * 70 + "\n")


def main():
    forecaster = DemandForecaster()

    while True:
        try:
            datetime_str, temp, humidity, rainfall, wind_speed = get_user_input()

            # Validate datetime format
            try:
                pd.to_datetime(datetime_str)
            except Exception:
                print("Invalid date/time format. Use: YYYY-MM-DD HH:MM:SS")
                continue

            result = forecaster.predict_demand(
                datetime_str=datetime_str,
                temp=temp,
                humidity=humidity,
                rainfall=rainfall,
                wind_speed=wind_speed
            )
            display_prediction(result)

        except ValueError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue


if __name__ == "__main__":
    main()
