# Machine learning model to predict the breast cancer is there or not (Just Classification Report)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load training data
df = pd.read_csv("breast_cancer.csv")

# Prepare features and target
X = df.drop(columns='target')
y = df['target']

# Splitting Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train linear regression model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print('Decision Tree — train accuracy:', round(model.score(X_train, y_train), 3))
print('Decision Tree — test accuracy: ', round(model.score(X_test, y_test), 3))
print(classification_report(y_test, model.predict(X_test),
                            target_names=['malignant', 'benign']))

fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(model, feature_names=X.columns.tolist(), class_names=['malignant', 'benign'],
          filled=True, rounded=True, fontsize=8, ax=ax)
fig.tight_layout()
plt.show()