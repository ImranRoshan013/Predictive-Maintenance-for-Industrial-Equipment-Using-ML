# Predictive Maintenance for Machine Failure

This project aims to predict machine failures using machine learning techniques. The dataset used is the **AI4I 2020 Predictive Maintenance Dataset**, which contains information about machine operating conditions and failure types. The goal is to build a model that can accurately predict machine failures to enable proactive maintenance and reduce downtime.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building](#model-building)
6. [Results](#results)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

---

## Introduction
Predictive maintenance is a critical application of machine learning in industrial settings. By predicting machine failures, businesses can reduce downtime, save costs, and improve operational efficiency. This project uses the **AI4I 2020 Predictive Maintenance Dataset** to build a machine learning model that predicts machine failures based on operational data.

---

## Dataset
The dataset used in this project is the **AI4I 2020 Predictive Maintenance Dataset**, which can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset). It contains 10,000 rows and 14 columns, including features like air temperature, process temperature, rotational speed, torque, and tool wear.

### Dataset Summary
- **Total Rows**: 10,000
- **Columns**: 14
- **Target Variable**: `Machine failure` (0 = No Failure, 1 = Failure)
- **Key Features**:
  - `Air temperature [K]`
  - `Process temperature [K]`
  - `Rotational speed [rpm]`
  - `Torque [Nm]`
  - `Tool wear [min]`
  - `Type` (Machine type: L, M, H)

### Dataset Preview
```plaintext
   UDI Product ID Type  Air temperature [K]  Process temperature [K]  \
0    1     M14860    M                298.1                    308.6   
1    2     L47181    L                298.2                    308.7   
2    3     L47182    L                298.1                    308.5   
3    4     L47183    L                298.2                    308.6   
4    5     L47184    L                298.2                    308.7   

   Rotational speed [rpm]  Torque [Nm]  Tool wear [min]  Machine failure  TWF  \
0                    1551         42.8                0                0    0   
1                    1408         46.3                3                0    0   
2                    1498         49.4                5                0    0   
3                    1433         39.5                7                0    0   
4                    1408         40.0                9                0    0   

   HDF  PWF  OSF  RNF  
0    0    0    0    0  
1    0    0    0    0  
2    0    0    0    0  
3    0    0    0    0  
4    0    0    0    0  
```

---

## Exploratory Data Analysis (EDA)
### Key Insights
1. **Target Variable Distribution**:
   - The dataset is highly imbalanced, with **9,661 samples (96.61%)** representing no machine failure and **339 samples (3.39%)** representing machine failures.
   - Visualized using a count plot with annotations.

2. **Numerical Feature Distributions**:
   - Features like `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]` were analyzed using histograms.

3. **Feature-Target Relationships**:
   - Boxplots were used to visualize the relationship between numerical features and the target variable (`Machine failure`).

---

## Data Preprocessing
### Key Steps
1. **Encoding Categorical Variables**:
   - The `Type` column was one-hot encoded to convert it into numerical format.

2. **Feature Selection**:
   - Unnecessary columns like `UDI` and `Product ID` were dropped.

3. **Train-Test Split**:
   - The dataset was split into training (80%) and testing (20%) sets.

4. **Handling Class Imbalance**:
   - **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the dataset.

5. **Feature Scaling**:
   - Numerical features were standardized using `StandardScaler`.

---

## Model Building
### Models Used
1. **Logistic Regression**:
   - Accuracy: 97%
   - ROC-AUC Score: 0.96

2. **Random Forest**:
   - Accuracy: 98%
   - ROC-AUC Score: 0.98

### Evaluation Metrics
- **Accuracy**: Overall correctness.
- **Precision**: Correctness of positive predictions.
- **Recall**: Ability to capture positive cases.
- **F1-Score**: Balance between precision and recall.
- **ROC-AUC**: Model's discriminative ability.

---

## Results
### What You Did
- **Data Preprocessing**: Cleaned the dataset by handling missing values, encoding categorical variables, and balancing the dataset using SMOTE.
- **Exploratory Data Analysis (EDA)**: Analyzed the dataset to understand the distribution of features and their relationships with the target variable.
- **Model Development**: Trained and evaluated two models: **Logistic Regression** and **Random Forest**.
- **Feature Importance**: Identified the most important features using the Random Forest model.
- **Threshold Tuning**: Adjusted the classification threshold to optimize precision and recall.

### Why You Did It
- **Business Problem**: Machine failures can lead to significant downtime and costs. Predicting failures allows for proactive maintenance, reducing these risks.
- **Class Imbalance**: The dataset was highly imbalanced, so SMOTE was used to ensure the model learns from all classes equally.
- **Model Performance**: By using Random Forest, we aimed to achieve high accuracy while capturing complex relationships in the data.

### What Were the Results
- **High Accuracy**: The Random Forest model achieved a **test accuracy of 98%** and an **ROC-AUC score of 0.98**, demonstrating its ability to predict machine failures effectively.
- **Feature Importance**: The most important features for predicting machine failures were:
  - `Rotational speed [rpm]`: 22.49%
  - `Torque [Nm]`: 21.33%
  - `Tool wear [min]`: 14.20%
  - `PWF`: 7.76%
  - `HDF`: 6.34%
  - `TWF`: 5.78%
  - `Air temperature [K]`: 5.43%
  - `OSF`: 5.19%
  - `Type_L`: 4.32%
  - `Process temperature [K]`: 3.85%
  - `Type_M`: 3.28%
  - `RNF`: 0.03%
- **Precision-Recall Analysis**:
  - Precision and recall intersect at a threshold of **0.3**, where both metrics are around **0.9 to 0.8**.
  - This threshold was chosen to balance precision and recall for the minority class (machine failures).

---

## Usage
To use this model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost
   ```

3. **Train the Model**:
   ```bash
   python train_model.py
   ```

4. **Evaluate the Model**:
   ```bash
   python evaluate_model.py
   ```

5. **Make Predictions**:
   ```bash
   python predict.py --input_data path/to/data.csv
   ```

---

## Contributing
Contributions to this project are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
