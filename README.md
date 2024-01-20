# VIX-Prediction
ML Project to predict VIX 1 day-ahead using Economic Data from FRED

# Description
* Uses yfinance library to grab historical VIX data since 2016.
* Target variable is Next_Close, which we try to predict tomorrow's VIX level

# Features Implemented
* Using FREDAPI, we add Economic Data as indicators for our VIX forecast:
* CPI, FEDFUNDS rate, Unemployment Rate, NonFarm Payments, FedFunds Effective Rate, TSY3M/10M, St Louis Financial Stress Index and WTI
* For variables that are only populated monthly, we forward fill to daily values so that they can be used for forecasting daily VIX closes.

# Time Series Splitting
To test our time-series model, we have to implement time-series-splitting in order to backtest the model's performance against multiple folds. We train it from 2016 and test the model's performance multiple times until 12/31/2023.

# Models Implemented
We test model performances with their R2 score/Mean Absolute Errors

XGBOOST: R2 - 0.88 / MAE - 0.67
LSTM: R2 - -0.98 / MAE - 3.61
Sequential: R2 - -102.02 / MAE - 17.72
ARIMA: R2 - 0.90 / MAE - 0.74
SARIMAX: R2 - 0.88 / MAE - 0.79

What was interesting is that the Neural Network models, while not only took much longer to train/test, also performed significantly worse than the XGBOOST and ARIMA-based models. I also found it interesting that ARIMA performed better than SARIMAX, which means the features added to the dataset made the model perform worse. I was expecting SARIMAX to perform better due to having these features that would help forecast VIX.

# Parameter Tuning
I used parameter tuning and tested which had the lowest MAE
XGBOOST - Tuned parameters: Max Depth, Learning Rate and Min Child Weight
LSTM - Tuned Parameters: LSTM Units, Dense Units and Learning Rate
Sequential - Used the same parameters as LSTM
ARIMA - Tuned the Order parameter
SARIMAX - Used same parameters as ARIMA

# Using the Model
1) Use Pipenv to install the included Piplock file
2) Run train.py to generate the updated Model.bin trained on the latest data
3) Build using Dockerfile by running "docker build -t vix-pred ." in the folder directory
4) Run the model using "docker run -it --rm -p 8000:8000 vix-pred"
5) Connect via http://localhost:8000/predict

