# T-Shirt Size Prediction

A NiceGUI web application that predicts a person's T-shirt size (`Small`, `Medium`, or
`Large`) from height and weight measurements.

## Features

- Loads the training data from `tshirt_sizes.csv`.
- Encodes T-shirt size labels.
- Trains a logistic regression classification model.
- Displays the classification accuracy.
- Provides a five-block responsive interface:
  - Height input in two grid blocks.
  - Weight input in two grid blocks.
  - Accuracy in one grid block.
  - Predicted size across the full width below the inputs.
- Displays predictions as Small, Medium, or Large.
- Validates that height and weight are positive numbers.
- Appends each valid prediction to `tshirt_sizes.csv`.
- Uses a light, white-background theme with configurable `THEME_COLORS`.

## Requirements

- Python 3.8 or later
- pandas
- scikit-learn
- nicegui

Install the dependencies with:

```bash
pip install pandas scikit-learn nicegui
```

## Project Structure

```text
Tshirt Size Prediction/
|-- index.py
|-- tshirt_sizes.csv
|-- readMe.md
```

## Dataset

`tshirt_sizes.csv` contains:

- `height_cm`: Height in centimeters
- `weight_kg`: Weight in kilograms
- `size`: T-shirt size label (`S`, `M`, or `L`)

## How to Run

From the project directory, run:

```bash
python index.py
```

Then open the application at:

```text
http://localhost:8080
```

Enter a positive height in centimeters and weight in kilograms, then click
**Predict T-shirt Size**. The predicted size appears below the input grid.

## Model Metrics

The application uses `LogisticRegression(max_iter=250)` with height and weight
as input features and the encoded size as the target.

It calculates and prints both:

- R² score: `model.score(X_test, y_test)`
- Accuracy: `accuracy_score(y_test, predictions)`

The displayed test accuracy is the percentage of correctly classified test
samples. The R² value is printed in the console because logistic regression is
a classifier, so this value is not an R² regression metric in the usual sense.

## UI Theme

The interface uses the `THEME_COLORS` dictionary in `index.py`. Colors can be
adjusted centrally, including the primary blue, light blue background, borders,
text, card surface, and page background.
