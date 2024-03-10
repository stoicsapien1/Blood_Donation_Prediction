import streamlit as st
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
import pandas as pd
# Header
st.title("Blood Donation Prediction")
st.header("Made by Belal Ahmed Siddiqui")
st.image(r"C:\Users\lucius seneca\CampusX\Project\Blood Donation Prediction\project\Give Life_ Predict Blood Donations\datasets\xco2_8jtl_220606.jpg", use_column_width=True)

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to predict blood donation
def predict_donation(recency, frequency, time, monetary_log):
    prediction = model.predict([[recency, frequency, time, monetary_log]])
    if prediction[0] == 1:
        return "Likely to Donate Blood"
    else:
        return "Not Likely to Donate Blood"

# Input fields for user input
st.sidebar.header("Input Features")
recency = st.sidebar.slider("Recency (months)", 0, 100, 50)
frequency = st.sidebar.slider("Frequency (times)", 0, 50, 25)
time = st.sidebar.slider("Time (months)", 0, 100, 50)
monetary_log = st.sidebar.slider("Monetary Log", 0.0, 5.0, 2.5)

# Prediction
if st.sidebar.button("Predict"):
    prediction = predict_donation(recency, frequency, time, monetary_log)
    st.write("Prediction:", prediction)