# ocean-health_prediction
XGBoost ML model predicting ocean health indicators from real-world buoy sensor data such as temperature, salinity, pH, dissolved oxygen.

# Ocean Health Prediction Model

A machine learning pipeline that predicts ocean health indicators 
from real-world oceanographic buoy sensor data.

## What it does
- Merges buoy data across 4 parameters: temperature, salinity, pH, 
  and dissolved oxygen
- Engineers time-based features with cyclical encoding
- Trains an XGBoost model for both regression and classification tasks
- Outputs feature importance plots and actual vs predicted evaluation charts

## Stack
Python, XGBoost, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Data
Open-source oceanographic buoy datasets covering sensor readings 
across multiple buoy stations and timestamps.

## Results
Model evaluated on RMSE, MAE, and R² for regression tasks.
Confusion matrix and classification report for classification tasks.
