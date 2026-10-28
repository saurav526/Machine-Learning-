import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.to_csv("iris.csv", index=False)

data = pd.read_csv("iris.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:", list(y_test))
print("Predicted Values:", list(y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

plt.scatter(X["sepal length (cm)"], X["sepal width (cm)"], c=y, cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Naive Bayes Classification - Iris Dataset")  
plt.show()
