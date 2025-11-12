# import streamlit as st
# import requests
# import os

# st.title('IRIS Prediction')

# sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.1)
# sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
# petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
# petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

# if st.button('Predict'):
#     data = {
#         'sepal_length': sepal_length,
#         'sepal_width': sepal_width,
#         'petal_length': petal_length,
#         'petal_width': petal_width
#     }
#     try:
#         # response = requests.post('https://iris-app-155173250771.us-central1.run.app/predict', json=data)
#         response = requests.post('http://127.0.0.1:5000/predict', json=data)
#         if response.status_code == 200:
#             prediction = response.json()['prediction']
#             st.success(f'Predicted species: {prediction}')
#         else:
#             st.error(f'Error occurred during prediction. Status code: {response.status_code}')
#     except requests.exceptions.RequestException as e:
#         st.error(f'Error occurred during prediction: {str(e)}')

# src/streamlit_app.py
import streamlit as st
import requests
from predict import FEATURE_NAMES

st.title("Wine Quality Class Prediction")
st.write("Predict wine class (0, 1, or 2) using 13 chemical features.")

# Create input fields
inputs = {}
for col in FEATURE_NAMES:
    clean_name = col.replace('_', ' ').title()
    default = 0.0
    if 'alcohol' in col: default = 13.0
    elif 'proline' in col: default = 700.0
    inputs[col] = st.number_input(clean_name, value=default, step=0.1)

if st.button("Predict"):
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=inputs)
        if response.status_code == 200:
            pred = response.json()['prediction']
            st.success(f"**Predicted Wine Class: {pred}**")
            st.balloons()
        else:
            st.error(f"API Error: {response.json().get('error', 'Unknown')}")
    except Exception as e:
        st.error(f"Connection failed: {e}")



# # Final changes

# import streamlit as st
# import requests
# from predict import FEATURE_NAMES 

# st.title("Wine Quality Class Prediction")
# st.write("Predict wine class (0, 1, or 2) using 13 chemical features.")

# inputs = {}
# for col in FEATURE_NAMES:
#     clean_name = col.replace('_', ' ').title()
#     default = 0.0
#     if 'alcohol' in col: default = 13.0
#     elif 'proline' in col: default = 700.0
#     inputs[col] = st.number_input(clean_name, value=default, step=0.1)

# if st.button("Predict"):
#     try:
#         response = requests.post("http://127.0.0.1:5000/predict", json=inputs)
#         if response.status_code == 200:
#             pred = response.json()['prediction']
#             st.success(f"**Predicted Wine Class: {pred}**")
#             st.balloons()
#         else:
#             st.error(f"API Error: {response.json().get('error', 'Unknown')}")
#     except Exception as e:
#         st.error(f"Connection failed: {e}")