import numpy as np
import pandas as pd
import sklearn
import yfinance as yf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler as mms
import joblib

start_date="2019-01-01"
end_date="2024-02-02"
data_1=yf.download("AAPL", start=start_date, end=end_date)
data_2=yf.download("TSLA", start=start_date, end=end_date)

# print("data download")

open_data1=data_1['Open'].tolist()
window=50
s = round(len(open_data1)*0.7)

train= open_data1[:s]
test= open_data1[s:-1]

train = np.array(train)
test = np.array(test)

scaler=mms(feature_range=(0,1))
train_data=scaler.fit_transform(train.reshape(-1,1))
test_data=scaler.fit_transform(test.reshape(-1,1))

# print("data scaled ")

x,y=[],[]
for i in range(len(train_data)-window-1):
  x.append(train_data[i:window +i])
  y.append(train_data[i+window])
x=np.array(x)
y=np.array(y)

x_test,y_test=[],[]
for i in range(len(test_data)-window-1):
  x_test.append(test_data[i:window +i])
  y_test.append(test_data[i+window])
x_test=np.array(x_test)
y_test=np.array(y_test)

# print("model training started")

model1=Sequential()
model1.add(LSTM(128, return_sequences=True, input_shape= (x.shape[1], 1)))
model1.add(LSTM(64, return_sequences=False))
model1.add(Dense(25))
model1.add(Dense(1))

model1.compile(loss='mean_squared_error', optimizer='adam')

history=model1.fit(x, y, epochs=50, batch_size=32)

y_pred=model1.predict(x_test)
y_pred=scaler.inverse_transform(y_pred)

def pred(days, base_series1):
  
  for i in range(0,days):
    temp = model1.predict(base_series1)
    base_series1 = np.append(base_series, temp[0][0])
    base_series1 = base_series1[1:].reshape(1,50,1)

  return base_series1

joblib.dump(model1, 'model.pkl')

# print("SDF")