import streamlit as st
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
import pandas as pd
# Header
st.title("Blood Donation Prediction")
st.header("Made by Belal Ahmed Siddiqui")
st.write('''Features:

Recency: Months since the last donation (numerical)

Frequency: Total number of blood donations made (numerical)

Monetary: Total volume of blood donated in cubic centimeters (numerical)

Time: Months since the first donation (numerical)''')
st.write('''VARIOUS STEPS INCORPORATED 
Data Loading and Preprocessing:
Loaded the dataset from a CSV file (transfusion.data).
Renamed the target column to "target".
Checked the data types and basic information about the dataset.

Data Splitting:

Split the dataset into training and testing sets using train_test_split() from sklearn.model_selection.

Model Training with TPOT:
Imported TPOTClassifier from tpot.
Trained the TPOTClassifier model using the training data (X_train, y_train) to find the best pipeline.
Used ROC AUC score (roc_auc_score) to evaluate the model's performance on the testing data.

Log Transformation:Normalized the specified column ("Monetary (c.c. blood)") using log transformation.
Checked the variance of the normalized data.

Model Training with Logistic Regression:

Imported LogisticRegression from sklearn.linear_model.

Trained a logistic regression model using the normalized training data (X_train_normed, y_train).
Used ROC AUC score to evaluate the logistic regression model's performance on the testing data.

Model Comparison:Compared the performance of the TPOT model and logistic regression model based on their AUC scores.
Model Serialization:

Serialized the trained logistic regression model using pickle and saved it to a file (logistic_regression_model.pkl).
Demonstrated loading the saved model from the file for future use.''')
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
