#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Summer 2020
@author: Jordan Gross
@title: REI and Linear Regression (Stat 656, Dr. Jones, Assignment 1)
"""

# Import OS and set CWD
import os

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

try:
    encoded_df = joblib.load(open('./Data/encoded_df.joblibRF.sav', 'rb')).dropna().drop(columns='obs')
except (OSError, IOError, EOFError):
    df = pd.read_excel("Data/diamondswmissing.xls")
    df['cut'] = df['cut'].replace(np.nan, 'Unknown', regex=True)
    df['color'] = df['color'].replace(np.nan, 'Unknown', regex=True)
    df['clarity'] =df['clarity'].replace(np.nan, 'Unknown', regex=True)
        
    categorical_columns = ['cut', 'color', 'clarity']

    ohe = LabelEncoder()
    df['cut'] = ohe.fit_transform(df['cut'])
    df['color'] = ohe.fit_transform(df['color'])
    df['clarity'] = ohe.fit_transform(df['clarity'])

    encoded_df = df.dropna().drop(columns='obs')

    with open('./Data/encoded_df.joblibRF.sav', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        joblib.dump(encoded_df, f) 

X = encoded_df[['color', 'cut', 'x', 'y', 'z', 'Carat', 'clarity', 'depth', 'table']].astype(float)
Y = encoded_df['price'].astype(int)

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
joblib.dump(pipe_RF, "diamonds_model.joblibRF.sav")

# Load model from joblib file
loaded_pipe_RF = joblib.load("diamonds_model.joblibRF.sav")

# Predict a Wine Quality (Class) from inputs
loaded_pipe_RF.predict([[6.8, .47, .08, 2.2, .0064, 18.0, 38.0, .999933, 3.2, .64, 9.8, ]])
