import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



# Load dataset (replace 'your_file.csv' with the actual file)
df = pd.read_csv("input.csv", parse_dates=["Timestamp"], index_col="Timestamp")

# Fill missing values (if any)
df = df.fillna(method="ffill")  # Forward-fill method

# Visualize the first few rows
# print(df.head())

# Define prediction target
target = "avg_CO2"

# Define feature set (excluding target and timestamp)
features = ['GRT', 'ship_loa', 'power', 'wdir', 'wspd', 'Sensor_Temp', 'Sensor_Humidity', 'Vehicle']

# Train-test split (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

print(f"Train size: {len(train)}, Test size: {len(test)}")

# Train ARIMA model (AutoRegressive Integrated Moving Average)
arima_model = ARIMA(train[target], order=(5,1,2))  # (p,d,q) values may need tuning
arima_fit = arima_model.fit()

# Forecast using ARIMA
train["arima_pred"] = arima_fit.fittedvalues
test["arima_pred"] = arima_fit.forecast(steps=len(test))

plt.figure(figsize=(10,5))
plt.plot(train[target], label="Actual (Train)")
plt.plot(test[target], label="Actual (Test)", color='black')
plt.plot(test["arima_pred"], label="ARIMA Prediction", linestyle="dashed")
plt.legend()
plt.show()

# Use ARIMA residuals (error) as the new target for XGBoost
train["residual"] = train[target] - train["arima_pred"]
test["residual"] = test[target] - test["arima_pred"]

# Train XGBoost on feature set
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

xgb_model.fit(train[features], train["residual"])

# Predict residuals using XGBoost
test["xgb_pred_residual"] = xgb_model.predict(test[features])

# Final prediction: ARIMA + XGBoost residual correction
test["final_pred"] = test["arima_pred"] + test["xgb_pred_residual"]

# Evaluate model
mae = mean_absolute_error(test[target], test["final_pred"])
rmse = np.sqrt(mean_squared_error(test[target], test["final_pred"]))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot Final Prediction
plt.figure(figsize=(10,5))
plt.plot(test[target], label="Actual CO2", color='black')
plt.plot(test["final_pred"], label="Final Prediction (ARIMA+XGBoost)", linestyle="dashed", color="red")
plt.legend()
plt.show()
