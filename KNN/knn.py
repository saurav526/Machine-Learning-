import sklearn.neighbors as knn
import numpy as np

X = [
    [80,5],
    [90,6],
    [75,6],
    [60,5],
    [60,7],    
    [40,5],
    [50,6],
    [85,6] 
    
]
# 0: Not an athlete, 1: Athlete
y = [0, 0, 1, 1, 1, 0, 0, 1]

model = knn.KNeighborsClassifier(n_neighbors=6)
model.fit(X, y)

weight = float(input("Enter weight in kg: "))
height = float(input("Enter height in feet: "))

prediction = model.predict([[weight, height]])[0]

if prediction == 0:
    print("The person is not an athlete.")
else:
    print("The person is an athlete.")


print(f"Predicted class for weight {weight} lbs and height {height} ft: {prediction}")
print("Model Neighbors:", model.n_neighbors)            