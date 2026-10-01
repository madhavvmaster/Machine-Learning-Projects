# Titanic Survival Prediction

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster based on features such as passenger class, sex, age, fare, family size, and embarkation port.

## Overview

The project uses the classic Titanic dataset and trains two classifiers:

- Decision Tree Classifier
- Random Forest Classifier

The script performs data cleaning, feature engineering, model training, evaluation, and a feature importance visualization.

## Dataset

The dataset is stored in `titanic.csv` and contains passenger information including:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

## Project Structure

- `index.py` — main Python script that loads the dataset, prepares features, trains models, and prints evaluation metrics.
- `titanic.csv` — Titanic passenger dataset used for training and testing.
- `readMe.md` — project documentation.

## Features in the Model

During preprocessing, the script:

- removes non-useful columns like `PassengerId`, `Name`, `Ticket`, and `Cabin`
- fills missing values in `Age` and `Embarked`
- converts categorical variables into numeric form using one-hot encoding
- splits the data into training and testing sets with stratification

## Machine Learning Workflow

The script follows this workflow:

1. Load the Titanic dataset.
2. Clean and preprocess the data.
3. Encode category values.
4. Split into train/test data.
5. Train a Decision Tree model.
6. Train a Random Forest model.
7. Print accuracy and classification reports for both models.
8. Plot feature importance for the Random Forest model.

## Technologies Used

- Python
- pandas
- matplotlib
- scikit-learn

## Requirements

Install the required dependencies with:

```bash
pip install pandas matplotlib scikit-learn
```

## Run the Project

From the project directory, run:

```bash
python index.py
```

This will print model performance metrics and display a feature-importance chart.

## Output

The script produces:

- decision tree accuracy metrics
- random forest accuracy metrics
- detailed classification reports
- a bar chart showing which features most influence survival prediction

## Notes

This is a beginner-friendly machine learning project that demonstrates:

- data preprocessing
- categorical encoding
- model evaluation
- feature importance analysis

It is useful for learning how supervised classification models work on a real-world dataset.
