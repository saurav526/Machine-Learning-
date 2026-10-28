from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Predicted Values:", list(y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=data.target_names
)

plt.title("Logistic Regression - Breast Cancer")
plt.show()
