import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets


#Create Features / Target Variables
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=50)

model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    report = classification_report(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return report, matrix

report, matrix = evaluate_model(model, X_test, y_test)
print(report)
#Plot

def plot_model(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Setosa", "Versicolor", "Virginica"]
                ,yticklabels=["Setosa", "Versicolor", "Virginica"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.show()
plot_model(matrix)