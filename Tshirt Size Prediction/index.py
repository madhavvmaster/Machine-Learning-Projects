# Machine learning model to predict if a flower is setosa, virginica or versicolor

# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load training data
df = pd.read_csv("tshirt_sizes.csv")

# Preprocessing the data
le = LabelEncoder()
encoded_sizes = le.fit_transform(df['size'])

# Prepare features and target
X = df[['height_cm', 'weight_kg']]
y = encoded_sizes

# Training-Testing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=250)
model.fit(X_train, y_train)

# Testing R2 Score
test_r2 = model.score(X_test, y_test)
print(f"R2 Score: {test_r2}")

# Get new tshirt size data from user
new_height = float(input("Enter the height (in cm): "))
new_weight = float(input("Enter the weight (in cm): "))

# Convert input to DataFrame for model
new_tshirt_data = pd.DataFrame([[new_height, new_weight]],
                                columns=['height_cm', 'weight_kg'])

# Make prediction
prediction = model.predict(new_tshirt_data)[0]
predicted_sizes = le.inverse_transform([prediction])[0]

# Display result
print(f"The tshirt size is {predicted_sizes}.")

# Save the new tshirt size record to CSV
new_row = pd.DataFrame([
    {
        'height_cm': new_height,
        'weight_cm': new_weight,
        'size': predicted_sizes 
    }
])

new_row.to_csv("tshirt_sizes.csv", mode="a", header=False, index=False, lineterminator='\n')