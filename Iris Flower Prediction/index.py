# Machine learning model to predict if a flower is setosa, virginica or versicolor

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load training data
df = pd.read_csv("Iris.csv")

# Preprocessing the data
le = LabelEncoder()
encoded_species = le.fit_transform(df['Species'])

# Prepare features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = encoded_species

# Training-Testing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)

# Testing R2 Score
test_r2 = model.score(X_test, y_test)
print(f"R2 Score: {test_r2}")

# Get new flower data from user
new_sepal_length = float(input("Enter the sepal length (in cm): "))
new_sepal_width = float(input("Enter the sepal width (in cm): "))
new_petal_length = float(input("Enter the petal length (in cm): "))
new_petal_width = float(input("Enter the petal width (in cm): "))

# Convert input to DataFrame for model
new_flower_data = pd.DataFrame([[new_sepal_length, new_sepal_width, new_petal_length, new_petal_width]],
                                columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# Make prediction
prediction = model.predict(new_flower_data)[0]
predicted_species = le.inverse_transform([prediction])[0]

# Display result
print(f"The flower is {predicted_species}.")

# Save the new flower record to CSV
last_id = df["Id"].max()

new_row = pd.DataFrame([
    {
        'Id': last_id + 1,
        'SepalLengthCm': new_sepal_length,
        'SepalWidthCm': new_sepal_width,
        'PetalLengthCm': new_petal_length,
        'PetalWidthCm' : new_petal_width,
        'Species': predicted_species
    }
])

new_row.to_csv("Iris.csv", mode="a", header=False, index=False, lineterminator='\n')