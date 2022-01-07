#This function will just do routing of request and response
from flask import Flask, request, jsonify, render_template
import json
import pickle
import numpy as np
app = Flask(__name__)
columns = None
model = None
with open("columns.json", "r") as f:
    columns = json.load(f)["data_columns"]
    
with open("banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/location')
def location():
    return jsonify(columns[8:-1])

@app.route('/area')
def area():
    return jsonify(columns[4:8])

@app.route('/predict', methods=['POST'])
def predict():
    area_type = request.form['area_type']
    location = request.form['location']
    bhk = int(request.form['bhk'])
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    balcony = int(request.form['balcony'])

    tokens = area_type.split(" ")
    if(len(tokens) == 2):
        area_type = tokens[0] + "  " + tokens[1]
    else:
        area_type = tokens[0] + " " + tokens[1] + "  " + tokens[2]
    area_type_ind = columns.index(area_type.lower())
    
    location_ind = columns.index(location.lower())

    data = np.zeros(len(columns))
    data[0] = bhk
    data[1] = total_sqft
    data[2] = bath
    data[3] = balcony
    data[area_type_ind] = 1
    data[location_ind] = 1

    output = np.round(model.predict([data])[0],2)
    return jsonify(output)
    
if __name__ == '__main__':
    print("Starting flask server")
    app.run(debug=True)