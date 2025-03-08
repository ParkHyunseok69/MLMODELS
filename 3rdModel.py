import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets


#Create Features / Target Variables
diabetes_data = datasets.load_diabetes()
diabetes = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
diabetes.drop(columns=['age', 'sex', 's1', 's2', 's3'], inplace=True)
X = diabetes
threshold = np.percentile(diabetes_data.target, 75)
y = np.where(diabetes_data.target > threshold, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=100)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=500, random_state=42)
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
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Have Diabetes",  "Dont Have Diabetes", ]
                ,yticklabels=["Have Diabetes", "Dont Have Diabetes"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Value")
    plt.show()
plot_model(matrix)