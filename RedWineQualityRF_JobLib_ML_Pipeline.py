#
# Author: Jamey Johnston
# Title: SciKit Learn ML Pipeline Example with joblib
# Date: 2020/01/16
# Email: jameyj@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

# Train models for Detecting Wine Quality
# Save model to file using joblib and use ML Pipelines
# Load model and make predictions
#

# Import OS and set CWD
import os
from settings import APP_ROOT

import numpy as np
from numpy import loadtxt, vstack, column_stack

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Import joblib to save ML models
import joblib


# Load the Wine Data
dataset = np.loadtxt(os.path.join(APP_ROOT, "winequality-red.csv"), delimiter=';', skiprows=1)

# Headers of Data
# "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"

# Split the wine data into X (independent variable) and y (dependent variable)
X = dataset[:,0:11].astype(float)
Y = dataset[:,11].astype(int)

# Split wine data into train and validation sets
seed = 7
test_size = 0.3
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Create a Pipeline with data scaler and classifier
pipe_RF = Pipeline([('scl', StandardScaler()),
			('clf', RandomForestRegressor())])

# Fit model on Wine Training Data using Random Forest save model to joblib file
pipe_RF.fit(X_train, y_train)

# Make predictions for Validation data
y_predRF = pipe_RF.predict(X_valid)
predictionsRF = [round(value) for value in y_predRF]

# Evaluate predictions
accuracyRF = accuracy_score(y_valid, predictionsRF)
print("Accuracy of Random Forest: %.2f%%" % (accuracyRF * 100.0))

# Create Dataset with Prediction and Inputs
predictionResultRF = column_stack(([X_valid, vstack(y_valid), vstack(y_predRF)]))

# save model to file
joblib.dump(pipe_RF, "winequality-red.joblibRF.sav")

# Load model from joblib file
loaded_pipe_RF = joblib.load("winequality-red.joblibRF.sav")

# Predict a Wine Quality (Class) from inputs
loaded_pipe_RF.predict([[6.8, .47, .08, 2.2, .0064, 18.0, 38.0, .999933, 3.2, .64, 9.8, ]])

