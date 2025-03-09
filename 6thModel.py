import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

housing_data = pd.read_csv("housing.csv")
enc = LabelEncoder()
housing_data['ocean_proximity'] = enc.fit_transform(housing_data['ocean_proximity'])
housing_data['total_bedrooms'] = housing_data['total_bedrooms'].fillna(housing_data['total_bedrooms'].median())
X = housing_data.drop(['median_house_value'], axis=1)
y = housing_data['median_house_value']


pipe = Pipeline([('scale', RobustScaler()),
                 ('model',RandomForestRegressor(random_state=11))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
pipe.fit(X_train, y_train)
def evaluate_model(pipe, X_test, y_test):
    prediction = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction) ** 0.5  # RMSE
    r2 = r2_score(y_test, prediction)
    return mae, mse, rmse, r2

mae, mse, rmse, r2 = evaluate_model(pipe, X_test, y_test)

print(f"MAE: {mae:.2f}")
#🔹 Good: Below $50,000
#🔹 Average: Around $50,000 - $80,000
#🔹 Poor: Above $80,000
print(f"MSE: {mse:.2f}")
#🔹 Good: Below 3e+9
#🔹 Average: Around 3e+9 - 5e+9
#🔹 Poor: Above 5e+9
print(f"RMSE: {rmse:.2f}")
#🔹 Good: Below $55,000
#🔹 Average: Around $55,000 - $85,000
#🔹 Poor: Above $85,000
print(f"R² Score: {r2:.2f}")
#🔹 Good: 0.80 - 1.00 (Great fit)
#🔹 Average: 0.60 - 0.79 (Decent fit, could improve)
#🔹 Poor: Below 0.60 (Model isn’t explaining much)