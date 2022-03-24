import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
#from sklearn.preprocessing import StandardScaler

#deploy Artificially Neural Network breast cancer model in a web application
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    #final_features = scaler.transform(final_features)
    features_name = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32']
    df = pd.DataFrame(final_features, columns=features_name)
    prediction = model.predict(df)
    y_probabilities_test = model.predict_proba(df)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    print(output)

    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
    else:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}'.format(y_prob))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
