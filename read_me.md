# PowerGrid AI - Electricity Demand Forecasting

Advanced AI-powered electricity demand prediction using Gradient Boosting machine learning algorithm.

## Features

- Machine Learning model trained on historical electricity consumption data
- Weather-based prediction (temperature, humidity, rainfall, wind speed)
- Beautiful modern web interface with premium animations
- 24-hour demand forecast visualization
- Real-time predictions

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib flask flask-cors
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare and Train the Model

```bash
python prepare_data.py
python train_gradient_boosting.py
python evaluate_model.py
```

### 2. Run the Web Interface

```bash
python app.py
```

Then open your browser and navigate to: **http://localhost:5000**

## Usage

1. **Date & Time**: Select the date and time for your prediction
2. **Weather Conditions**: 
   - Adjust temperature (0-60°C) using the slider
   - Adjust humidity (0-100%) using the slider
   - Enter rainfall amount (mm)
   - Enter wind speed (km/h)
3. Click **"Generate Forecast"** to see the prediction

## Files

- `prepare_data.py` - Data preprocessing and feature engineering
- `train_gradient_boosting.py` - Model training
- `evaluate_model.py` - Model evaluation and visualization
- `prediction_system.py` - Command-line prediction interface
- `app.py` - Flask web server and API
- `templates/index.html` - Web UI
- `static/css/style.css` - Premium dark theme styles
- `static/js/app.js` - Interactive frontend functionality

## Model Performance

The Gradient Boosting model is trained on electricity consumption data with features including:
- Time features (hour, day of week, month, etc.)
- Weather data (temperature, humidity, rainfall, wind speed)
- Lag features (demand from previous hours)

## Tech Stack

- **Backend**: Python, Flask, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **ML Algorithm**: Gradient Boosting Regressor
