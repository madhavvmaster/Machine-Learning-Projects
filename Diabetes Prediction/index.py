# Machine learning model to predict if a patient has diabetes or not

# Import libraries
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load training data
df = pd.read_csv("diabetes.csv")

# Prepare features and target
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

# Train logistic regression model
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))
model.fit(X, y)

# Get new patient data from user
new_pregnancies = input("Enter the number of pregnancies: ")
new_glucose = input("Enter the glucose level: ")
new_blood_pressure = input("Enter the blood pressure level: ")
new_skin_thickness = input("Enter the skin thinkness: ")
new_insulin = input("Enter the insulin level: ")
new_bmi = input("Enter the BMI: ")
new_diabetes_pedigree_function = input("Enter the Diabetes Predigree Function: ")
new_age = input("Enter the age: ")

# Convert input to DataFrame for model
new_patient_data = pd.DataFrame([[new_pregnancies, new_glucose, new_blood_pressure, new_skin_thickness, new_insulin, new_bmi, new_diabetes_pedigree_function, new_age]],
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Make prediction
prediction = model.predict(new_patient_data)[0]

# Display result (1 = Positive, 0 = Negative)
if (prediction):
    print("The patient has Diabetes.")
else:
    print("The patient does not have Diabetes.")

# Save new patient record to CSV
new_row = [new_pregnancies, new_glucose, new_blood_pressure, new_skin_thickness, new_insulin, new_bmi, new_diabetes_pedigree_function, new_age, prediction]

with open("diabetes.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(new_row)