# Iris Flower Prediction

A machine learning program that predicts whether an iris flower is **setosa**,
**versicolor**, or **virginica** from four measurements.

## Features

- Loads flower data from `Iris.csv`.
- Encodes the species labels for model training.
- Trains a logistic regression classifier.
- Displays the model score on the test data.
- Predicts a species from new flower measurements.
- Saves the new prediction to `Iris.csv`.

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
Iris Flower Prediction/
|-- index.py
|-- Iris.csv
|-- readMe.md
```

## Dataset

`Iris.csv` contains these columns:

- `Id`
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species`

The supported species labels are `Iris-setosa`, `Iris-versicolor`, and
`Iris-virginica`.

## How to Run

Open a terminal in the project folder and run:

```bash
python index.py
```

The program asks for four measurements in centimeters:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

For example:

```text
Enter the sepal length (in cm): 5.1
Enter the sepal width (in cm): 3.5
Enter the petal length (in cm): 1.4
Enter the petal width (in cm): 0.2
```

The program then prints the predicted species and appends the new record to
`Iris.csv`.

## Model

The program uses `LogisticRegression` from scikit-learn. The input features are
the four sepal and petal measurements, and the target is the flower species.

The printed test score is the model's classification accuracy on the test
subset, not an R2 regression score.
