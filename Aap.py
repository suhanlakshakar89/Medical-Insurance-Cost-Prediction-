from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the machine learning model
model_path = "model_ML_2.pkl" 
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask application
app = Flask(__name__)

# Home route to display the web page
@app.route('/')
def home():
    return render_template('index.html')

# Predict route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    region = request.form['region']
    sex = request.form['sex']
    smoker = request.form['smoker']

    # Convert categorical variables to numerical format if necessary
    region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
    sex_map = {'male': 0, 'female': 1}
    smoker_map = {'yes': 1, 'no': 0}

    region_num = region_map.get(region.lower(), 0)
    sex_num = sex_map.get(sex.lower(), 0)
    smoker_num = smoker_map.get(smoker.lower(), 0)

    # Create input array for prediction
    input_features = np.array([[age, bmi, children, region_num, sex_num, smoker_num]])

    # Make prediction
    prediction = model.predict(input_features)
    predicted_cost = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted Medical Insurance Cost: ₹{predicted_cost}')

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
