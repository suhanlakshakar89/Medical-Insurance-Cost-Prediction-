import streamlit as st
import pickle
import numpy as np

# Load the machine learning model
model_path = 'model_ML_2.pkl'  # Change this path to the actual path of your model file
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define a function for prediction
def predict_insurance_cost(age, bmi, children, region, sex, smoker):
    # Convert categorical variables to numerical format
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
    return round(prediction[0], 2)

# Streamlit app
st.title('Medical Insurance Cost Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=30)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
region = st.selectbox('Region', ('southwest', 'southeast', 'northwest', 'northeast'))
sex = st.selectbox('Sex', ('male', 'female'))
smoker = st.selectbox('Smoker', ('yes', 'no'))

# Prediction button
if st.button('Predict'):
    predicted_cost = predict_insurance_cost(age, bmi, children, region, sex, smoker)
    st.success(f'Predicted Medical Insurance Cost: ₹{predicted_cost}')
