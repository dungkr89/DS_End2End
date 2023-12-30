import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#Load model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

#Test with JSON in postman
#json body: 
# {
# "data":{
# "MedInc":2.33526315,
# "HouseAge":0.98504972,
# "AveRooms":0.63012521,
# "AveBedrms":-0.16586931,
# "Population":-0.96995366,
# "AveOccup":-0.04555657,
# "Latitude":1.04385626,
# "Longitude":-1.32105914
# }
# }

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

#Test with home html
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)
    print(output[0])
    return render_template('home.html', prediction_text = "The house price (only for test flow) is {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)