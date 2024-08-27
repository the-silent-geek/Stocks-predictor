from flask import Flask, render_template
from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import traceback
import numpy as np
import json
from prediction import pred
from base_series import base_series_test

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template(template_name_or_list='index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            b = request.json
            l = []
            series = pred(b["Days"], base_series_test, l)
            
            # l = [i[0] for i in series[0]]
                      
            return jsonify(list(series))
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print("Error")

@app.route('/results')
def results():
    data = request.args.get('data')
    data = json.loads(data)
    label1 = [
        1,2,3,4,5,6,7,8,9,10
    ]
    return render_template(template_name_or_list='results.html', data=data, label=label1)
    #     return jsonify({'error':'No data provided'})
model = joblib.load("model.pkl")
print("loaded")    

if __name__=="__main__":
    app.run(debug=True)
    
