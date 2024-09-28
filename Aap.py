import streamlit as st
import pickle
import numpy as np

# Load the machine learning model
model_path = 'model_Medical.pkl'  # Change this path to the actual path of your model file
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

    # Ensure there are no missing or NaN values
    if np.isnan(input_features).any():
        raise ValueError("Input features contain NaN values. Please check the inputs.")

    # Make prediction
    try:
        prediction = model.predict(input_features)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

    return round(prediction[0], 2)

# Streamlit app
st.title('Medical Insurance Cost Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=30)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=28.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=2)
region = st.selectbox('Region', ('northwest',  'northeast','southwest', 'southeast'))
sex = st.selectbox('Sex', ('female', 'male'))
smoker = st.selectbox('Smoker', ('yes', 'no'))

# Prediction button
if st.button('Predict'):
    try:
        predicted_cost = predict_insurance_cost(age, bmi, children, region, sex, smoker)
        if predicted_cost is not None:
            st.success(f'Predicted Medical Insurance Cost: ₹{predicted_cost}')
    except ValueError as ve:
        st.error(f"Input error: {ve}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
