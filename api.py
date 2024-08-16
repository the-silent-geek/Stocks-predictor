from flask import Flask, request, jsonify
import joblib
import traceback
import numpy as np
import json
from prediction import pred
from base_series import base_series_test, test
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Any

class Item(BaseModel):
    Days: int

app = FastAPI()

@app.post('/predict')
async def create_item(item: Item = Body(...))-> Any:

    if model:
        try:
            b = item.Days
            series = pred(b, base_series_test, test)
            
            l = [i for i in series]
                        
            return {"result":l}
        except:
            return jsonify({'trace': traceback.format_exc()})
        
model = joblib.load("model.pkl")
print("loaded")    
