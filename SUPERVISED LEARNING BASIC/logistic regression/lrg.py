import sklearn.linear_model as logistic_regression
import numpy as np

X = [[0], [1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1, 1]
# Creates a Logistic Regression model object.
model = logistic_regression.LogisticRegression()
# Trains the model by learning the relationship between X and y.
model.fit(X, y)

input_value = float(input("Enter a value to predict its class (0 or 1): "))
predicted_class = model.predict([[input_value]])
print(f"Predicted class for input {input_value}: {predicted_class[0]}")
# Shows weights learned by the model for each feature.
print("Model Coefficients:", model.coef_[0])   
# Shows the bias term of the model.     
print("Model Intercept:", model.intercept_)   