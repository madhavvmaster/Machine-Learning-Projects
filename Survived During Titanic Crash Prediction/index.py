# Machine learning model to predict the survival of people during Titanic Crash

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load training data
df = pd.read_csv("titanic.csv")

# Dropping unhelpful columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Filling missing values 
df['Age'] = df['Age'].fillna(df['Age'].median()) 
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode() [0]) 

# Turning text columns into numeric 0/1 columns 
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Prepare features and target
X = df.drop(columns="Survived")
y = df['Survived']

# Splitting into Training & Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Decision Tree Classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Getting classification report
print('Decision Tree - train accuracy:', round(model.score(X_train, y_train), 3)) 
print('Decision Tree - test accuracy: ', round(model.score(X_test, y_test), 3)) 
print(classification_report(y_test, model.predict(X_test), 
                            target_names=['died', 'survived' ]))

# Train Random Forest Classifier model
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

# Getting classification report
print('Random Forest - train accuracy:', round(forest.score(X_train, y_train), 3)) 
print('Random Forest - test accuracy: ', round(forest.score(X_test, y_test), 3)) 
print(classification_report(y_test, forest.predict(X_test), 
                            target_names=['died', 'survived' ]))

# Generating graph of feature importance
importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values()

importances.plot(kind="barh")
plt.xlabel("Importance")
plt.title("What predicted survival on the Titanic?")
plt.show()