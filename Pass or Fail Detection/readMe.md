# Pass or Fail Detection

A small machine learning project that predicts whether a student will pass or fail using logistic regression.

## Features

The model uses these student details:

- Attendance percentage
- Homework completion percentage
- Midterm score
- Study hours per week

The training data is stored in `Pass-Fail-Data.csv`. The `pass` column is the target, where `1` means pass and `0` means fail.

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
cd "Pass or Fail Detection"
python index.py
```

Enter the requested values when prompted:

```text
Enter the attendance (in between 0-100): 90
Enter the homework done (in between 0-100): 85
Enter the midterm score (in between 0-100): 80
Enter the studyhours per week: 10
```

The program displays either:

```text
The student will pass.
```

or:

```text
The student will fail.
```

## Data format

`Pass-Fail-Data.csv` must contain these columns:

| Column | Description |
| --- | --- |
| `student_id` | Unique student identifier |
| `attendance_pct` | Attendance percentage |
| `homework_pct` | Homework completion percentage |
| `midterm_score` | Midterm score |
| `study_hours_per_week` | Weekly study hours |
| `pass` | Training label: `1` for pass, `0` for fail |

## Important behavior

After making a prediction, the script appends the new student and prediction to `Pass-Fail-Data.csv`. This allows future runs to include previous predictions in the training data, but it also modifies the original dataset. Keep a backup if you want to preserve the initial training data.

The model is intended for demonstration and educational purposes. Its predictions depend on the quality and size of the training dataset and should not be used as the sole basis for academic decisions.
