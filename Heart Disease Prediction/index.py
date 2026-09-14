# Machine learning model to predict if a person has heart disease or not

# Import libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load training data
df = pd.read_csv("heart_disease.csv")

# Preprocessing the data
df = df.apply(pd.to_numeric, errors='coerce').dropna()
df['target'] = (df['num'] > 0).astype(int)

# Prepare features and target
feature_columns = [column for column in df.columns if column not in ['num', 'target']]
X = df[feature_columns]
y = df['target']

# Training-Testing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Standard Scaling of data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train support vector machine model
model = SVC(kernel='rbf')
model.fit(X_train_scaled, y_train)

# Testing Accuracy Score
test_accuracy = model.score(X_test_scaled, y_test)
print(f"Testing Accuracy: {test_accuracy:.2%}")

# Get new heart disease data from user
new_patient_data = [[
    float(input("Enter the age: ")),
    float(input("Enter the sex (1 = male, 0 = female): ")),
    float(input("Enter the chest pain type (1-4): ")),
    float(input("Enter the resting blood pressure: ")),
    float(input("Enter the cholesterol level: ")),
    float(input("Enter fasting blood sugar (1 = true, 0 = false): ")),
    float(input("Enter the resting ECG result (0-2): ")),
    float(input("Enter the maximum heart rate: ")),
    float(input("Enter exercise-induced angina (1 = yes, 0 = no): ")),
    float(input("Enter the ST depression value: ")),
    float(input("Enter the slope of the peak exercise ST segment (1-3): ")),
    float(input("Enter the number of major vessels (0-3): ")),
    float(input("Enter the thalassemia result (3, 6, or 7): "))
]]

# Convert input to DataFrame for model
new_patient_data = pd.DataFrame(new_patient_data, columns=feature_columns)

# Make prediction
prediction = int(model.predict(scaler.transform(new_patient_data))[0])

# Display result
if prediction == 1:
    print("The person has heart disease.")
else:
    print("The person does not have heart disease.")

# Save the new heart disease record to CSV
new_row = new_patient_data.copy()
new_row['num'] = prediction
new_row = new_row[feature_columns + ['num']]

new_row.to_csv("heart_disease.csv", mode="a", header=False, index=False, lineterminator='\n')