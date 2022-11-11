import numpy as np
from flask import Flask, request, render_template
import pickle

# create flask app
app = Flask(__name__)
# load model
model = pickle.load(open('model.pkl', 'rb'))


# create route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features_test = [np.array(float_features)]
    prediction = model.predict(features_test)

    return render_template('index.html', prediction_text='Value of your car is $ {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
