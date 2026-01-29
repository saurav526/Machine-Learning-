import sklearn.tree as tree

X = [
    [7,3],
    [8,4],
    [9,6],
    [5,2],
    [10,7],
    [4,1],  
    [6,3],
    [11,8]
]
y = [0, 0, 1, 0, 1, 0, 0, 1]  # 0: Apple , 1: Orange

model = tree.DecisionTreeClassifier()
model.fit(X, y)

weight = float(input("Enter weight in kg: "))
size = float(input("no of fruits : "))

model_prediction = model.predict([[weight, size]])[0]   
if model_prediction == 0:
    print("The fruit is an Apple.")
else:
    print("The fruit is an Orange.")
print(f"Predicted class for weight {weight} kg and size {size}: {model_prediction}")
print("Model Feature Importances:", model.feature_importances_)