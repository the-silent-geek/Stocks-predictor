import yfinance as yf
from sklearn.preprocessing import MinMaxScaler as mms
import numpy as np

start_date="2019-01-01"
end_date="2024-02-02"
data_1=yf.download("AAPL", start=start_date, end=end_date)

scaler=mms(feature_range=(0,1))
base_series_test = data_1['Open'].tolist()
base_series_test = np.array(base_series_test[:50])
base_series_test = scaler.fit_transform(base_series_test.reshape(-1,1))
base_series_test = base_series_test.reshape(1,50,1)
# print(base_series_test)