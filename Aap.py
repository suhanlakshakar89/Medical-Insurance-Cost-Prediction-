import streamlit as st
import pickle
import numpy as np

# Load the pre-trained machine learning model
model_path = "model_ML_2.pkl"
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict insurance cost
def predict_cost(age, bmi, children, region, sex, smoker):
    # Convert categorical data to numeric
    sex = 1 if sex == 'Male' else 0
    smoker = 1 if smoker == 'Yes' else 0

    # Region encoding (example - match your model's encoding)
    regions = {'Southwest': 0, 'Southeast': 1, 'Northwest': 2, 'Northeast': 3}
    region = regions[region]

    # Prepare the features for the model
    features = np.array([[age, bmi, children, region, sex, smoker]])

    # Make a prediction using the model
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app code
st.title("Medical Insurance Cost Prediction (in INR)")

# Create form inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
region = st.selectbox("Region", ["Southwest", "Southeast", "Northwest", "Northeast"])
sex = st.selectbox("Sex", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])

# Button to make predictions
if st.button("Predict"):
    prediction = predict_cost(age, bmi, children, region, sex, smoker)

    # Convert prediction from USD to INR (assuming 1 USD = 83 INR)
    prediction_inr = prediction * 83

    # Display the result in INR
    st.success(f"The predicted medical insurance cost is ₹{round(prediction_inr, 2)}")
