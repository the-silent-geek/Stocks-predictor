import numpy as np
import joblib

model1 = joblib.load("model.pkl")

def pred(days, base_series1, test):
  
  for i in range(0,days):
    temp = model1.predict(base_series1)
    base_series1 = np.append(base_series1, temp[0][0])
    base_series1 = base_series1[1:].reshape(1,50,1)
    test = np.append(test,temp[0][0])

  return test