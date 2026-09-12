# Insurance Prediction

A small machine learning project that predicts whether a person has insurance based on their age using logistic regression.

## How it works

The model is trained with the `age` and `has_insurance` columns from `insurance_data.csv`:

- `age`: the person's age
- `has_insurance`: `1` if the person has insurance, or `0` otherwise

The dataset is split into training and testing data. After training, the program asks for a new person's age, predicts whether they have insurance, and prints the result.

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
cd "Insurance Prediction"
python index.py
```

Enter an age when prompted:

```text
Enter the age: 35
```

The program first prints the test score, then displays a prediction such as:

```text
R2 Score: 0.7
The person doesn't have insurance.
```

## Data format

`insurance_data.csv` must contain these columns:

| Column | Description |
| --- | --- |
| `age` | The person's age |
| `has_insurance` | Insurance status: `1` for yes or `0` for no |

## Important behavior

After making a prediction, the script appends the entered age and predicted insurance status to `insurance_data.csv`. The new row may therefore be included in future training runs. Keep a backup if you want to preserve the original dataset.

The value printed as `R2 Score` is the model's classification accuracy on the test set, as returned by scikit-learn's `model.score()` method. Predictions are estimates based only on age and are intended for educational purposes, not real insurance decisions.
