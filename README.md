Blood Donation Prediction using Machine Learning

This repository contains code for predicting blood donation likelihood using machine learning models. The dataset used in this project (transfusion.data) contains information about individuals' blood donation history.

Steps Taken:

Data Loading and Preprocessing:

Loaded the dataset from a CSV file (transfusion.data).
Renamed the target column to "target".
Checked the data types and basic information about the dataset.

Data Splitting:

Split the dataset into training and testing sets using train_test_split() from sklearn.model_selection.

Model Training with TPOT:

Trained the TPOTClassifier model to find the best pipeline for predicting blood donation likelihood.
Evaluated the model's performance on the testing data using ROC AUC score.

Log Transformation:

Normalized the specified column ("Monetary (c.c. blood)") using log transformation.

Checked the variance of the normalized data.

Model Training with Logistic Regression:

Trained a logistic regression model using the normalized training data.

Evaluated the logistic regression model's performance on the testing data using ROC AUC score.

Model Comparison:

Compared the performance of the TPOT model and logistic regression model based on their AUC scores.

Model Serialization:

Serialized the trained logistic regression model using pickle and saved it to a file (logistic_regression_model.pkl).
Demonstrated loading the saved model from the file for future use.

Requirements:

Python 3

Libraries: numpy, pandas, streamlit, scikit-learn, tpot
