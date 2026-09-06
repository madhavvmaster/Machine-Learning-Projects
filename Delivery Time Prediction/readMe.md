# Delivery Time Prediction

A small machine learning project that predicts delivery time from delivery distance using linear regression.

## How it works

The model is trained with the `distance` and `time_taken` columns from `delivery_time.csv`:

- `distance`: delivery distance in kilometers
- `time_taken`: delivery time in minutes

After training, the program asks for a new delivery distance, predicts the delivery time, rounds the result to the nearest minute, and prints it.

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
cd "Delivery Time Prediction"
python index.py
```

Enter the distance in kilometers when prompted:

```text
Enter the distance (in km): 6.5
```

Example output:

```text
The time taken to deliver will be 42 min.
```

## Data format

`delivery_time.csv` must contain these columns:

| Column | Description |
| --- | --- |
| `distance` | Delivery distance in kilometers |
| `time_taken` | Delivery time in minutes |

## Important behavior

After making a prediction, the script appends the entered distance and predicted time to `delivery_time.csv`. The new row may therefore be included in future training runs. Keep a backup if you want to preserve the original training data.

This project is intended for educational purposes. Predictions are estimates and may be affected by traffic, weather, route conditions, and other delivery factors that are not included in the model.
