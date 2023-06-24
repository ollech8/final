# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 01:06:38 2023

@author: ASUS VIVOBOOK
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from madlan_data_prep import prepare_data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import seaborn as sns
from datetime import datetime, timedelta
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn
import subprocess
from madlan_data_prep import prepare_data

df = pd.read_excel('output_all_students_Train_v10.xlsx')
df= prepare_data(df)
dft = pd.read_excel('Dataset_for_test.xlsx')
dft= prepare_data(dft)


# Select the features and target column
features = ['City', 'type', 'room_number', 'Area', 'city_area', 'hasElevator',
       'hasParking', 'hasBars', 'hasStorage', 'condition', 'hasAirCondition',
       'hasBalcony', 'hasMamad', 'handicapFriendly', 'floor']
target = 'price'

X_train = df[features]
y_train = df[target]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_test = dft[features]
y_test = dft[target]

cat_cols = ['City', 'type' , 'hasElevator',
            'hasParking',  'hasStorage', 'condition', 
            'hasBalcony', 'hasMamad', 'handicapFriendly','city_area']
num_cols = ['room_number', 'Area', 'floor']

cat_pipeline = Pipeline([
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore',drop='if_binary'))
])

num_pipeline = Pipeline([('numerical_imputation', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler())
])
column_transformer = ColumnTransformer([
    ('numerical_preprocessing', num_pipeline, num_cols),
    ('categorical_preprocessing', cat_pipeline, cat_cols)
], remainder='drop')



X_train = df[features]
y_train = df[target]

X_test = dft[features]
y_test = dft[target]

model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Perform 10-fold cross-validation on the training data
cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_squared_error')

# Convert the negative MSE scores to positive
mse_scores = -cv_scores

# Calculate RMSE scores
rmse_scores = mse_scores ** 0.5

# Calculate R2 scores
r2_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')

# Print the average scores
print("Cross-Validation Performance:")
print(f"Average RMSE: {rmse_scores.mean():.2f}")
print(f"Average MSE: {mse_scores.mean():.2f}")
print(f"Average R2 score: {r2_scores.mean():.2f}")

# Define the pipeline
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
#mse KFold
kfold_mse  = np.abs(cv_scores.mean())
kfold_std = cv_scores.std()
print(f"MSE KFold: {np.round(kfold_mse, 1)} kFold std {kfold_std}")
# Fit the pipeline to the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance on the test set
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance on Test Set:")
print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2 score: {r2:.2f}")

joblib.dump(model, 'trained_model.pkl')

