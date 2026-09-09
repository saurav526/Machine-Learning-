import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)


data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

y = pd.Series(data.target)

print("Dataset Shape:")
print(X.shape)

print("\nFirst 5 Rows:")
print(X.head())

print("\nMissing Values:")
print(X.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# 6. Create KNN Model
# ============================================================

knn = KNeighborsClassifier(n_neighbors=5,metric="euclidean")


# ============================================================
# 7. Train Model
# ============================================================

knn.fit(X_train, y_train)

print("\nKNN Model trained successfully.")


# ============================================================
# 8. Make Predictions
# ============================================================

y_pred = knn.predict(X_test)

print("\nPredictions:")
print(y_pred)


# ============================================================
# 9. Model Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test,y_pred,average="weighted")

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n============================")
print("       KNN RESULTS")
print("============================")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 10. Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# ============================================================
# 11. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("KNN Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ============================================================
# 12. Test One New Sample
# ============================================================

sample = X_test[0].reshape(1, -1)

prediction = knn.predict(sample)

print("\n============================")
print("      NEW PREDICTION")
print("============================")

print(
    "Predicted Class:",
    data.target_names[prediction[0]]
)