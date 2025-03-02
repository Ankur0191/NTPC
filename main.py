from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import os
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError  # ✅ Fix import issue

# ✅ Initialize FastAPI App
app = FastAPI()

# 🔹 Allow CORS for All Domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow ALL origins (public access)
    allow_credentials=True,
    allow_methods=["*"],  # 👈 Allow ALL HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # 👈 Allow ALL headers
)

# ✅ Force TensorFlow to use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ✅ Load trained models
def load_trained_model(model_name):
    """Loads an existing trained LSTM model and registers 'mse' as a valid loss function."""
    model_path = f"{model_name}.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None

    try:
        print(f"✅ Loading model: {model_path}")
        return load_model(model_path, custom_objects={"mse": MeanSquaredError()})
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None

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
    """Generates a 30-day energy forecast based on the last 30 days."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")

    last_30_days = np.array(last_30_days).reshape(-1, 1)  # Convert to NumPy array
    predictions = []

    for _ in range(30):  # Predict for 30 days
        future_array = np.array(last_30_days[-30:]).reshape(1, 30, 1)
        pred = model.predict(future_array, verbose=0)[0][0]
        predictions.append(float(pred))  # ✅ Convert NumPy float32 → Python float
        last_30_days = np.append(last_30_days, pred).reshape(-1, 1)  # Append prediction

    return predictions  # ✅ Now returns only 30 days of forecast

# ✅ Prediction Endpoint
@app.post("/predict")
def predict_energy(request: ForecastRequest):
    """API to predict next 30 days of energy generation."""
    if request.plant_type == "solar":
        if solar_model:
            forecast = forecast_next_month(solar_model, request.last_30_days)
        else:
            raise HTTPException(status_code=500, detail="Solar model not loaded.")
    elif request.plant_type == "wind":
        if wind_model:
            forecast = forecast_next_month(wind_model, request.last_30_days)
        else:
            raise HTTPException(status_code=500, detail="Wind model not loaded.")
    else:
        raise HTTPException(status_code=400, detail="Invalid plant type. Use 'solar' or 'wind'.")

    return {"forecast": forecast}

# ✅ Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
