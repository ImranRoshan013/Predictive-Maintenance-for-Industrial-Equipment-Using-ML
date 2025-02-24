# Predictive Maintenance for Industrial Equipment Using Machine Learning

## Overview
This project focuses on building a **predictive maintenance system** to detect equipment failures based on sensor data. The goal is to enable proactive maintenance, reduce downtime, and save costs. The dataset used is the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).

## Problem Statement
Predict equipment failures based on sensor data to enable proactive maintenance and reduce downtime.

## Dataset
The dataset contains **10,000 samples** with the following features:
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`
- `Machine failure` (Target variable: 0 = No Failure, 1 = Failure)

## Approach
1. **Exploratory Data Analysis (EDA)**:
   - Analyzed the distribution of features and target variable.
   - Visualized relationships between features and target.
2. **Data Preprocessing**:
   - Encoded categorical variables using one-hot encoding.
   - Standardized numerical features.
   - Handled class imbalance using **SMOTE**.
3. **Model Building**:
   - Trained a **Random Forest** model.
   - Evaluated the model using precision, recall, F1-score, and ROC-AUC.
4. **Threshold Tuning**:
   - Adjusted the decision threshold to balance precision and recall.
5. **Feature Importance**:
   - Analyzed the most important features for predicting failures.

## Results
- **Precision (Class 1 - Failure)**: 67%
- **Recall (Class 1 - Failure)**: 97%
- **F1-Score (Class 1 - Failure)**: 79%
- **ROC-AUC Score**: 0.98

