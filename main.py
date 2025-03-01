import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ Force TensorFlow to use CPU

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import os

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Load trained models
def load_trained_model(model_name):
    """Loads an existing trained LSTM model."""
    if os.path.exists(f"{model_name}.h5"):
        print(f"✅ Loading model: {model_name}.h5")
        return load_model(f"{model_name}.h5")
    else:
        print(f"⚠️ Model {model_name}.h5 not found.")
        return None  # Return None if model is missing

solar_model = load_trained_model("solar_model")
wind_model = load_trained_model("wind_model")

# ✅ API Input Schema
class ForecastRequest(BaseModel):
    plant_type: str
    last_30_days: list

# ✅ Health Check
@app.get("/")
def home():
    return {"message": "Energy Forecasting API is running!"}

# ✅ Forecast Function (Next 30 Days)
def forecast_next_month(model, last_30_days):
    last_30_days = np.array(last_30_days).reshape(-1, 1)  # Convert to NumPy array
    predictions = []
    
    for _ in range(30):  # Predict for 30 days
        future_array = np.array(last_30_days[-30:]).reshape(1, 30, 1)
        pred = model.predict(future_array, verbose=0)[0][0]
        predictions.append(float(pred))  # Convert NumPy float32 → Python float
        last_30_days = np.append(last_30_days, pred).reshape(-1, 1)  # Append prediction

    return predictions  # ✅ Now returns only 30 days of forecast

# ✅ Prediction Endpoint
@app.post("/predict")
def predict_energy(request: ForecastRequest):
    if request.plant_type == "solar" and solar_model:
        forecast = forecast_next_month(solar_model, request.last_30_days)
    elif request.plant_type == "wind" and wind_model:
        forecast = forecast_next_month(wind_model, request.last_30_days)
    else:
        return {"error": "Invalid plant type or model not found. Use 'solar' or 'wind'."}

    return {"forecast": forecast}
