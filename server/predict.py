import sys
import json
import joblib
import pandas as pd
import numpy as np

# Load trained models
arima_model = joblib.load("arima_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")

# Get input data from Node.js
input_json = json.loads(sys.argv[1])

# Convert input JSON to DataFrame
features = ["GRT", "ship_loa", "power", "wdir", "wspd", "Sensor_Temp", "Sensor_Humidity", "Vehicle"]
for key in features:
    if not key in input_json.keys():
        print(json.dumps({"err": "Input fields incomplete"}))
        exit(1)
input_data = pd.DataFrame([input_json], columns=features)

# Predict ARIMA base trend
last_known_co2 = arima_model.predict(start=len(arima_model.fittedvalues), end=len(arima_model.fittedvalues))
last_known_co2.reset_index(drop=True, inplace=True)

# Predict XGBoost residuals
xgb_residual = xgb_model.predict(input_data)[0]

# Final CO2 prediction
final_prediction = last_known_co2[0] + xgb_residual

# Output result
print(json.dumps({"CO2_prediction": final_prediction}))
