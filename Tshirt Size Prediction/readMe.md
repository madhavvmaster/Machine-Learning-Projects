# T-Shirt Size Prediction

A machine learning program that predicts a person's T-shirt size (`S`, `M`, or
`L`) from their height and weight.

## Features

- Loads training data from `tshirt_sizes.csv`.
- Encodes the size labels for model training.
- Trains a logistic regression classification model.
- Reports the model score on the test data.
- Predicts a T-shirt size from new measurements.
- Appends the new prediction to the CSV file.

## Requirements

- Python 3.8 or later
- pandas
- scikit-learn

Install the required packages with:

```bash
pip install pandas scikit-learn
```

## Project Structure

```text
Tshirt Size Prediction/
|-- index.py
|-- tshirt_sizes.csv
|-- readMe.md
```

## Dataset

The `tshirt_sizes.csv` file contains these columns:

- `height_cm`: Height in centimeters
- `weight_kg`: Weight in kilograms
- `size`: T-shirt size label (`S`, `M`, or `L`)

## How to Run

Open a terminal in the project folder and run:

```bash
python index.py
```

Enter the requested measurements:

```text
Enter the height (in cm): 170
Enter the weight (in kg): 65
```

The program prints the predicted T-shirt size and appends the new record to
`tshirt_sizes.csv`.

## Model

The program uses `LogisticRegression` from scikit-learn. The input features are
height and weight, and the target is the T-shirt size.

The printed score is the model's classification accuracy on the test subset.
It is not an R2 regression score.
