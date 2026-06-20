from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# loading model
import os
import pickle

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))

# flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']
    features = features.split(',')
    np_features = np.asarray(features, dtype=np.float32)

    pred = model.predict(np_features.reshape(1, -1))

    # prediction
    pred = model.predict(np_features.reshape(1, -1))
    output = ['Cancer Detected' if pred[0] == 1 else 'No Cancer Detected']
    return render_template('index.html', message=output)


if __name__ == '__main__':
    app.run(debug=True)
