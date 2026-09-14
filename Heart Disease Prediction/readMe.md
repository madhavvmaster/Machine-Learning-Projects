# Heart Disease Prediction

A small machine learning project that predicts whether a person may have heart disease using a support vector machine (SVM) model.

## Features

The model uses these patient details:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol level
- Fasting blood sugar
- Resting ECG result
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Slope of the peak exercise ST segment
- Number of major vessels
- Thalassemia result

The training data is stored in `heart_disease.csv`. The original `num` column is converted into a binary target:

- `0`: no heart disease
- Any value greater than `0`: heart disease

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
cd "Heart Disease Prediction"
python index.py
```

Enter the patient information when prompted:

```text
Enter the age: 63
Enter the sex (1 = male, 0 = female): 1
Enter the chest pain type (1-4): 1
Enter the resting blood pressure: 145
Enter the cholesterol level: 233
Enter fasting blood sugar (1 = true, 0 = false): 1
Enter the resting ECG result (0-2): 2
Enter the maximum heart rate: 150
Enter exercise-induced angina (1 = yes, 0 = no): 0
Enter the ST depression value: 2.3
Enter the slope of the peak exercise ST segment (1-3): 3
Enter the number of major vessels (0-3): 0
Enter the thalassemia result (3, 6, or 7): 6
```

The program first prints the testing accuracy, then displays one of these results:

```text
The person has heart disease.
```

or:

```text
The person does not have heart disease.
```

## Data format

`heart_disease.csv` must contain these columns:

| Column | Description |
| --- | --- |
| `age` | Patient age |
| `sex` | Sex indicator: `1` for male or `0` for female |
| `cp` | Chest pain type, from `1` to `4` |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol level |
| `fbs` | Fasting blood sugar indicator: `1` or `0` |
| `restecg` | Resting ECG result, from `0` to `2` |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator: `1` or `0` |
| `oldpeak` | ST depression caused by exercise |
| `slope` | Slope of the peak exercise ST segment, from `1` to `3` |
| `ca` | Number of major vessels, from `0` to `3` |
| `thal` | Thalassemia result, usually `3`, `6`, or `7` |
| `num` | Original disease severity label; `0` means no disease and values above `0` mean disease |

## Important behavior

After making a prediction, the script appends the entered patient data and predicted `num` value to `heart_disease.csv`. The new row may therefore be included in future training runs. Keep a backup if you want to preserve the original dataset.

Rows with missing or non-numeric values, such as `?`, are ignored during training.

This project is for educational purposes only. Its predictions are not a medical diagnosis and should not replace advice from a qualified healthcare professional.
