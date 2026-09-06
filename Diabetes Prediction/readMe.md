# Diabetes Prediction

A small machine learning project that predicts whether a patient may have diabetes using a standardized logistic regression model.

## Features

The model uses these patient details:

- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- Body Mass Index (BMI)
- Diabetes pedigree function
- Age

The training data is stored in `diabetes.csv`. The `Outcome` column is the target, where `1` indicates a positive diabetes result and `0` indicates a negative result.

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
cd "Diabetes Prediction"
python index.py
```

Enter numeric patient information when prompted:

```text
Enter the number of pregnancies: 2
Enter the glucose level: 120
Enter the blood pressure level: 70
Enter the skin thinkness: 25
Enter the insulin level: 100
Enter the BMI: 30
Enter the Diabetes Predigree Function: 0.5
Enter the age: 30
```

The program displays one of these results:

```text
The patient has Diabetes.
```

or:

```text
The patient does not have Diabetes.
```

## Data format

`diabetes.csv` must contain these columns:

| Column | Description |
| --- | --- |
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | Two-hour serum insulin level |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes hereditary risk score |
| `Age` | Patient age |
| `Outcome` | Training label: `1` for positive, `0` for negative |

## Important behavior

After making a prediction, the script appends the entered patient data and prediction to `diabetes.csv`. This means each run modifies the dataset and the new row may be included in future training runs. Keep a backup if you want to preserve the original training data.

This project is for educational purposes only. Its predictions are not a medical diagnosis and should not replace advice from a qualified healthcare professional.
