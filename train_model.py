import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
data = pd.read_csv(url)

# Preprocessing
data = pd.get_dummies(data, columns=['Type'], drop_first=True)
data = data.drop(['UDI', 'Product ID'], axis=1)

# Split into features and target
X = data.drop('Machine failure', axis=1)
y = data['Machine failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Standardize numerical features
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
scaler = StandardScaler()
X_train_res[numerical_features] = scaler.fit_transform(X_train_res[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)

# Save the model and scaler
joblib.dump(rf_model, 'random_forest_predictive_maintenance_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model trained and saved successfully!")
