from flask import Flask, render_template, request, jsonify, url_for
import requests
import pickle
import numpy as np
import sklearn


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html',)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    PM2_5 = int(float(request.form['PM2.5']))
    NO2 = int(float(request.form['NO2']))
    CO = int(float(request.form['CO']))
    SO2 = int(float(request.form['SO2']))
    O3 = int(float(request.form['O3']))
    data = np.array([[PM2_5, NO2, CO, SO2, O3]])
    output = model.predict(data)
    prediction = round(float(output[0]), 2)
    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
