# Hackpompey 2025: The backend of The Lonely Dashboard

A backend server that applies a machine learning approach to predict the CO2 metrics.

## Components

### Prediction Server

Under `server/` you can find a minimal Node.js server that predicts the CO2 metric of a certain time. 

#### Endpoints:

`/predict`, _GET_

**Input** : a JSON string containing the following fields:
| field | data type | explanation |
|---|---|---|
| GRT | int | The sum of the gross tonnage of the ships, i.e., it is the total gross tonnage of all the ships present at Portsmouth Port during the timestamp (Linkspan 1, Linkspan 2, etc.). |
| ship_loa | float | The sum of every ship’s length present at port (Linkspan 1, Linkspan 2, etc.). |
| power | int | The sum of every ship’s power present at port (Linkspan 1, Linkspan 2, etc.). | 
| wdir | float | the average wind direction in degrees (°) |
| wspd | float | The average wind speed in km/h |
| Sensor_Temp | float | The average temperature detected by the B4T sensor at port during timestamp." |
| Sensor_Humidity | float | The average Humidity detected by the B4T sensor at port during timestamp. |
| Vehicle | int | The total number of vehicles at M275 during the timestamp |

**Output**:
```json
{
    "CO2_prediction": "<result>" // float
}
```

## Machine Learning
[ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) + [XGBoost](https://en.wikipedia.org/wiki/XGBoost) are the two algorithms implemented in this project, due to their suitability for time-series data and ease of implementation. 


## Usage

### Setup

Prerequisite: Python > 3.8

1. Clone this repository:

    ```sh
    git clone https://github.com/CrzongA/hackpompey2025-data-server.git
    cd hackpompey2025-data-server
    ```

2. Setup a virtual environment for Python, and activate it

    ```sh
    python -m pip venv .venv
    python -m pip install -r requirements.txt
    call .venv/Scripts/activate # windows
    source .venv/bin/activate # mac / linux
    ```

3. After completing the above, you can now use `train.ipynb` to modify and train the ML model, or run `cd server && node server.js` to start the prediction server.

