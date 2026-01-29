from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40,50,60,70,90]

model = LinearRegression()

model.fit(X, y)

hours = float(input("Enter number of study hours: "))
predicted_score = model.predict([[hours]])

print(f"Predicted score for studying {hours} hours: {predicted_score[0]}")
print("Model Coefficient:", model.coef_[0])
print("Model Intercept:", model.intercept_)


