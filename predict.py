import pandas as pd
import joblib
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict machine failure using the trained model.")
parser.add_argument('--input_data', type=str, required=True, help="Path to the input CSV file.")
args = parser.parse_args()

# Load the input data
input_data = pd.read_csv(args.input_data)

# Preprocessing
input_data = pd.get_dummies(input_data, columns=['Type'], drop_first=True)
input_data = input_data.drop(['UDI', 'Product ID'], axis=1, errors='ignore')

# Load the trained model and scaler
rf_model = joblib.load('random_forest_predictive_maintenance_model.pkl')
scaler = joblib.load('scaler.pkl')

# Standardize numerical features
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Make predictions
predictions = rf_model.predict(input_data)

# Save predictions to a file
input_data['Predicted_Machine_Failure'] = predictions
input_data.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'.")
