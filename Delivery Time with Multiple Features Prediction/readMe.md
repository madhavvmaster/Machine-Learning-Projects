# Delivery Time Prediction with Multiple Features

A small machine learning project that predicts delivery time using linear regression and three delivery conditions: distance, traffic level, and rain.

## Features

The model uses the following inputs:

- `distance_km`: Delivery distance in kilometers
- `traffic_level`: Traffic level, represented as a numeric value
- `raining`: Weather indicator (`1` for raining, `0` for not raining)

The target value is `delivery_time_min`, the delivery time in minutes.

## How it works

1. Loads training data from `delivery_time_multi.csv`.
2. Splits the data into training and testing sets using a fixed random state.
3. Trains a scikit-learn `LinearRegression` model.
4. Prints the model's R² score on the test set.
5. Reads distance, traffic, and rain information from the user.
6. Predicts and prints the delivery time rounded to the nearest minute.
7. Appends the new input and prediction to the CSV file.

## Requirements

- Python 3.8 or newer
- pandas
- scikit-learn

Install the dependencies with:

```bash
pip install pandas scikit-learn
```

## Run the program

Run the script from this project directory so it can find the CSV file:

```bash
cd "Delivery Time with Multiple Features Prediction"
python index.py
```

Example interaction:

```text
R2 Score: 0.85
Enter the distance (in km): 6.5
Enter the traffic level: 7
Enter 1 if its raining otherwise 0: 1
The time taken to deliver will be 37 min.
```

The R² score will vary if the dataset or train-test configuration changes.

## Data format

`delivery_time_multi.csv` must contain these columns:

| Column | Description |
| --- | --- |
| `distance_km` | Delivery distance in kilometers |
| `traffic_level` | Numeric traffic level |
| `raining` | `1` if it is raining, otherwise `0` |
| `delivery_time_min` | Actual delivery time in minutes |

## Important behavior

After making a prediction, the script appends the entered values and predicted delivery time to `delivery_time_multi.csv`. The new row may therefore be used as training data during future runs. Keep a backup of the original dataset if needed.

This project is intended for educational purposes. Predictions are estimates and may be affected by route conditions, traffic patterns, weather severity, and other factors that are not included in the model.
