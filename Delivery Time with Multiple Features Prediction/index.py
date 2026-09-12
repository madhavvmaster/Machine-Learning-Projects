# Machine learning model to predict delivery time using multiple features

# Import libraries
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load training data
df = pd.read_csv("delivery_time_multi.csv")

# Prepare features and target
X = df[['distance_km', 'traffic_level', 'raining']]
y = df['delivery_time_min']

# Training-Testing the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Testing R2 Score
test_r2 = model.score(X_test, y_test)
print(f"R2 Score: {test_r2}")

# Get new delivery data from user
new_distance_km = input("Enter the distance (in km): ")
new_traffic_level = input("Enter the traffic level: ")
new_raining = input("Enter 1 if its raining otherwise 0: ")

# Convert input to DataFrame for model
new_delivery_data = pd.DataFrame([[new_distance_km, new_traffic_level, new_raining]],
                                columns=['distance_km', 'traffic_level', 'raining'])

# Make prediction
prediction = round(model.predict(new_delivery_data)[0])

# Display result
print(f"The time taken to deliver will be {prediction} min.")

# Save new delivery record to CSV
new_row = [new_distance_km, new_traffic_level, new_raining, prediction]

with open("delivery_time_multi.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(new_row)