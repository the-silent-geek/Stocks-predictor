from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
import json
from prediction import pred
from base_series import base_series_test

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            b = request.json
            series = pred(b[0]["Days"], base_series_test)
            
            l = []
            for i in series[0]:
                l.append(i[0])
            
            print(l)             
            return jsonify(l)
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print("DF")

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8000
        
model = joblib.load("model.pkl")
print("loaded")    

app.run(port=port, debug=True)