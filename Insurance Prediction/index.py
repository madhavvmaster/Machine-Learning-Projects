# Machine learning model to predict if a person has insurance or not

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load training data
df = pd.read_csv("insurance_data.csv")

# Prepare features and target
X = df[['age']]
y = df['has_insurance']

# Training-Testing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Testing R2 Score
test_r2 = model.score(X_test, y_test)
print(f"R2 Score: {test_r2}")

# Get new insurance data from user
new_age = input("Enter the age: ")

# Convert input to DataFrame for model
new_insurance_data = pd.DataFrame([[new_age]],
                                columns=['age'])

# Make prediction
prediction = model.predict(new_insurance_data)[0]

# Display result (1 = Has Insurance, 0 = Doesn't Have Insurance)
if (prediction):
    print("The person has insurance.")
else:
    print("The person doesn't have insurance.")

# Save new insurance record to CSV
new_row = pd.DataFrame([
    {
        'age': new_age,
        'has_insurance': prediction
    }
])

new_row.to_csv("insurance_data.csv", mode="a", header=False, index=False, lineterminator='\n')