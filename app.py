from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# import lasso model and standard scaler pickle
lasso_model = pickle.load(open('models/lasso.pkl', 'rb'))
standard_scalar = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Temperature=float(request.form.get('Temperature'))
        # RH = float(request.form.get('RH'))
        # Ws = float(request.form.get('Ws'))
        # Rain = float(request.form.get('Rain'))
        # FFMC = float(request.form.get('FFMC'))
        # DMC = float(request.form.get('DMC'))
        # ISI = float(request.form.get('ISI'))
        # Classes = float(request.form.get('Classes'))
        # Region = float(request.form.get('Region'))

        # concise and alternative way to do the same above thing
        feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
        input_data = [float(request.form.get(feature)) for feature in feature_names]

        # new_data_scaled = [standard_scalar.transform([input_data])]
        new_data_scaled = standard_scalar.transform(np.array(input_data).reshape(1, -1)) # same as using [[]] as standard scalar expects 2d array
        result = lasso_model.predict(new_data_scaled)

        #When you use .predict with machine learning models, it often returns an array or list containing the predicted values for multiple instances, even if you're just predicting a single instance. In this case, by using result[0], you're accessing the first (and presumably the only) element of that array to get the single predicted value.
        return render_template('home.html', result=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
