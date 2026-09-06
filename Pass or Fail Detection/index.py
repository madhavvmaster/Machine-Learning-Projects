# Machine learning model to predict if a student will pass or fail

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load training data
df = pd.read_csv("Pass-Fail-Data.csv")

# Prepare features and target
X = df[['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week']]
y = df['pass']

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Get new student data from user
new_attendance = input("Enter the attendance (in between 0-100): ")
new_homework = input("Enter the homework done (in between 0-100): ")
new_midterm_score = input("Enter the midterm score (in between 0-100): ")
new_studyhours = input("Enter the studyhours per week: ")

# Convert input to DataFrame for model
new_student_data = pd.DataFrame([[new_attendance, new_homework, new_midterm_score, new_studyhours]],
                                columns=['attendance_pct', 'homework_pct', 'midterm_score', 'study_hours_per_week'])

# Make prediction
prediction = model.predict(new_student_data)[0]

# Display result (1 = Pass, 0 = Fail)
if (prediction):
    print("The student will pass.")
else:
    print("The student will fail.")

# Save new student record to CSV
last_id = df["student_id"].max()

new_row = pd.DataFrame([
    {
        'student_id': last_id + 1,
        'attendance_pct': new_attendance,
        'homework_pct': new_homework,
        'midterm_score': new_midterm_score,
        'study_hours_per_week' : new_studyhours,
        'pass': prediction
    }
])

new_row.to_csv("Pass-Fail-Data.csv", mode="a", header=False, index=False, lineterminator='\n')