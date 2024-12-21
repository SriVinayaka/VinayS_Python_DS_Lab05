from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open('./models/model.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

# Load the label encoder for "CarName" (if you have saved it)
car_name_encoder = pickle.load(open('./models/car_name_encoder.pkl', 'rb'))  # If you saved the encoder


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        car_name = request.form['CarName']
        kms_driven = float(request.form['Kms_Driven'])
        age_of_the_car = float(request.form['age_of_the_car'])

        # Encode car name if necessary (e.g., using LabelEncoder)
        car_name_encoded = car_name_encoder.transform([car_name])[0]  # Encode the car name

        # Prepare the features for scaling
        features = [car_name_encoded, kms_driven, age_of_the_car]

        # Scale the features
        features_scaled = scaler.transform([features])

        # Make the prediction
        prediction = model.predict(features_scaled)

        # Display the input values and prediction result
        return render_template('index.html',
                               car_name=car_name,
                               kms_driven=kms_driven,
                               age_of_the_car=age_of_the_car,
                               prediction_text=f'Selling Price: ₹{prediction[0]:,.2f} Lakhs')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
