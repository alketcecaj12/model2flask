
"""
Created on Sun June 28 20:06:35 2020

@author: alketcecaj
"""

from keras.models import load_model
from flask import Flask, request
from flasgger import Swagger
from flask import jsonify, make_response
import numpy as np
import pandas as pd
import json

mlp = load_model('./modelMLP.h5')
cnn = load_model('./modelCNN.h5')
lstm = load_model('./modelLSTM.h5')

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/mlp')
def predict_mlp():
    """Multistep forecasting with MLP
    ---
    parameters:
      - name: p
        in: query
        type: number
        required: true
      - name: s
        in: query
        type: number
        required: true
      - name: t
        in: query
        type: number
        required: true
      - name: q
        in: query
        type: number
        required: true
      - name: qt
        in: query
        type: number
        required: true
      - name: st
        in: query
        type: number
        required: true
    responses:
        200:
           description: The output values

    """
    p = int(request.args.get("p"))
    s = int(request.args.get("s"))
    t = int(request.args.get("t"))
    q = int(request.args.get("q"))
    qt = int(request.args.get("qt"))
    st = int(request.args.get("st"))

    prediction = mlp.predict(np.array([[p, s, t, q, qt, st]]))
    print('--------------------->>>', type(prediction))
    prediction = prediction.tolist()
    res = str(prediction)
    return res

@app.route('/cnn')
def predict_cnn():
    """Multistep forecasting with CNN
    ---
    parameters:
      - name: p
        in: query
        type: number
        required: true
      - name: s
        in: query
        type: number
        required: true
      - name: t
        in: query
        type: number
        required: true
      - name: q
        in: query
        type: number
        required: true
      - name: qt
        in: query
        type: number
        required: true
      - name: st
        in: query
        type: number
        required: true
    responses:
        200:
           description: The output values

    """
    p = int(request.args.get("p"))
    s = int(request.args.get("s"))
    t = int(request.args.get("t"))
    q = int(request.args.get("q"))
    qt = int(request.args.get("qt"))
    st = int(request.args.get("st"))
    cnn = mlp
    prediction = cnn.predict(np.array([[p, s, t, q, qt, st]]))
    prediction = prediction.tolist()
    res = json.dumps(prediction)
    return res

@app.route('/lstm')
def predict_lstm():
    """Multistep forecasting with LSTM
    ---
    parameters:
      - name: p
        in: query
        type: number
        required: true
      - name: s
        in: query
        type: number
        required: true
      - name: t
        in: query
        type: number
        required: true
      - name: q
        in: query
        type: number
        required: true
      - name: qt
        in: query
        type: number
        required: true
      - name: st
        in: query
        type: number
        required: true
    responses:
        200:
           description: The output values

    """
    p = int(request.args.get("p"))
    s = int(request.args.get("s"))
    t = int(request.args.get("t"))
    q = int(request.args.get("q"))
    qt = int(request.args.get("qt"))
    st = int(request.args.get("st"))
    lstm = mlp
    prediction = lstm.predict(np.array([[p, s, t, q, qt, st]]))
    prediction = prediction.tolist()
    res = json.dumps(prediction)
    return res

if __name__ == '__main__':
    app.run()
