# Machine learning model to predict the delivery time using km

# Import libraries
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression

# Load training data
df = pd.read_csv("delivery_time.csv")

# Prepare features and target
X = df[['distance']]
y = df['time_taken']

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Get new delivery data from user
new_distance = input("Enter the distance (in km): ")

# Convert input to DataFrame for model
new_delivery_data = pd.DataFrame([[new_distance]],
                                columns=['distance'])

# Make prediction
prediction = round(model.predict(new_delivery_data)[0])

# Display result
print(f"The time taken to deliver will be {prediction} min.")

# Save new delivery record to CSV
new_row = [new_distance, prediction]

with open("delivery_time.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(new_row)