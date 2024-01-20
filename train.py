import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from fredapi import Fred

#Final Model
from fredapi import Fred
fred = Fred(api_key = 'fd4285a201de47c1efaa681faea2cddd')


symbol = '^VIX'
df = yf.download(symbol, start="2016-01-01")
df['Next_Close'] = df['Close'].shift(-1)
df_spx = yf.download('^GSPC', start="2016-01-01")

data = {}
series = ['CPIAUCNS', 'FEDFUNDS', 'UNRATE', 'PAYEMS', 'DFF', 'TB3MS', 'GS10','STLFSI4','DCOILWTICO']
for code in series:
    data[code] = fred.get_series(code)
data = pd.DataFrame(data)
data.index.name = 'Date'

daily_dates = pd.date_range(start=data.index.min(), end=df.index.max(), freq='D')
data = data.reindex(daily_dates).asfreq('D').fillna(method='ffill')
data.index.name = 'Date'

df_spx.rename(columns={'Close':'SPX'},inplace=True)
df = pd.merge(df,df_spx['SPX'],on='Date')
df = pd.merge(df, data, on ='Date', how = 'left')

features = ['Volume', 'SPX',
       'CPIAUCNS', 'FEDFUNDS', 'UNRATE', 'PAYEMS', 'DFF', 'TB3MS', 'GS10',
       'STLFSI4', 'DCOILWTICO']

target = 'Next_Close'

X_test = df.iloc[[-1]][features]

df.dropna(inplace=True)
X_train = df[features]
X_train.dropna(inplace=True)
y_train = df[target]
y_train.dropna(inplace=True)

model = xgb.XGBRegressor(booster='gbtree',
                        n_estimators=1000,
                        objective='reg:squarederror',
                        max_depth=3,
                        learning_rate=0.1,
                        min_child_weight = 1,
                        n_jobs = -1)
model.fit(X_train, y_train)


with open('model.bin', 'wb') as f_out:
   pickle.dump((model), f_out)
f_out.close()

with open('X_test.pkl', 'wb') as file:
    pickle.dump(X_test,file)
file.close()