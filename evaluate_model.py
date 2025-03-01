import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

# Load the trained model and scaler
rf_model = joblib.load('random_forest_predictive_maintenance_model.pkl')
scaler = joblib.load('scaler.pkl')

# Standardize numerical features
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_auc}")
