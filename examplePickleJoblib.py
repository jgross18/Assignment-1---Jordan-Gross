#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Summer 2020
@author: Jordan Gross
@title: REI and Linear Regression (Stat 656, Dr. Jones, Assignment 1)
"""

import numpy  as np
import pandas as pd
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# Import the statsmodels package
import statsmodels.api as sm
#  classes provided for the course




try:
    encoded_df = joblib.load(open('./Data/encoded_df.joblibRF.sav', 'rb'))
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

    encoded_df = df

    with open('./Data/encoded_df.joblibRF.sav', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        joblib.dump(encoded_df, f) 


print("***** REI & Linear Regression *****")
print(encoded_df.iloc[0:5])


    
try:
    results = pickle.load(open('./Data/results.pickle', 'rb'))
except (OSError, IOError, EOFError):
    X = np.asarray(encoded_df.drop('price', axis=1))
    y = np.asarray(encoded_df['price'])

    Xc = sm.add_constant(X)
    ols_model = sm.OLS(y, Xc, missing = 'drop')
    results   = ols_model.fit()
    with open('./Data/results.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL) 


print(results.summary())

