# Breast Cancer Prediction

A small machine-learning project that trains a decision tree classifier to predict whether a breast tumor is malignant or benign. The script reports training and test accuracy, prints a classification report, and plots the trained tree.

## Project files

- `index.py` - loads the data, trains and evaluates the classifier, and creates the tree plot.
- `breast_cancer.csv` - input dataset. It must include a `target` column and feature columns.
- `decision_tree.png` - generated visualization of the trained decision tree.

The target values are interpreted as `0 = malignant` and `1 = benign`.

## Requirements

Python 3 and the following packages are required:

```bash
python -m pip install pandas scikit-learn matplotlib
```

## Run

From this project directory, run:

```bash
python index.py
```

The script uses a stratified 80/20 train-test split with a fixed random seed, prints accuracy and precision/recall/F1 metrics in the terminal, and saves the decision-tree plot as `decision_tree.png`. It also opens the plot window when a graphical Matplotlib backend is available. Running the script again replaces the existing plot.
