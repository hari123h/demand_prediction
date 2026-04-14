
from app import LSTMDemandForecaster, PeakDemandForecaster

def print_header(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)

def main():
    print("Initializing CLI... Loading models, please wait.")
    
    # Load forecasters
    try:
        # Redirect stdout briefly to suppress model loading messages if desired, 
        # but the classes print loading messages which is fine.
        lstm_forecaster = LSTMDemandForecaster()
        peak_forecaster = PeakDemandForecaster()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    while True:
        print_header("Demand Forecasting - Interactive CLI")
        print("1. Predict Hourly Demand (LSTM)")
        print("2. Predict Peak Demand (FNN)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print_header("LSTM Hourly Demand Prediction")
            try:
                date_str = input("Enter Date (YYYY-MM-DD): ").strip()
                time_str = input("Enter Time (HH:MM): ").strip()
                datetime_str = f"{date_str} {time_str}:00"
                
                temp = float(input("Enter Temperature (°C): ").strip())
                humidity = float(input("Enter Humidity (%): ").strip())
                wind_speed = float(input("Enter Wind Speed (km/h): ").strip())
                
                print("\nGenerating prediction...")
                result = lstm_forecaster.predict_demand(
                    datetime_str, temp, humidity, wind_speed
                )
                
                print("\n--- PREDICTION RESULT ---")
                print(f"Date & Time: {result['datetime']}")
                print(f"Predicted Demand: {result['predicted_demand_mw']:.2f} MW")
                print("-------------------------")
                
            except ValueError as ve:
                print(f"\n[!] Error: Invalid numeric input or date format. Details: {ve}")
            except Exception as e:
                print(f"\n[!] Error during prediction: {e}")
                
        elif choice == '2':
            print_header("FNN Peak Demand Prediction")
            try:
                date_str = input("Enter Date (YYYY-MM-DD): ").strip()
                
                print("\nGenerating prediction...")
                result = peak_forecaster.predict_peak(date_str)
                
                print("\n--- PREDICTION RESULT ---")
                print(f"Date: {result['date']}")
                print(f"Predicted Daily Peak Demand: {result['predicted_peak_demand_mw']:.2f} MW")
                print("-------------------------")
                
            except ValueError as ve:
                print(f"\n[!] Error: Invalid date format. Details: {ve}")
            except Exception as e:
                print(f"\n[!] Error during prediction: {e}")
                
        elif choice == '3':
            print("\nExiting. Goodbye!")
            break
        else:
            print("\n[!] Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()
