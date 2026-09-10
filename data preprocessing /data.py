import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Amit", "Vikas", "Neha", "Rohan", "Kiran", "Anjali"],
    "Age": [21, 22, np.nan, 24, 21, 25, 23, np.nan, 22, 24],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Male", "Female", "Male", "Female", "Female"],
    "City": ["Pune", "Mumbai", "Pune", "Delhi", "Pune", "Mumbai", "Delhi", "Pune", "Mumbai", "Delhi"],
    "Salary": [25000, 30000, 28000, np.nan, 25000, 35000, 32000, 150000, 29000, 31000],
    "Experience": [1, 2, 2, 3, 1, 4, 3, 5, 2, 3],
    "Purchased": ["No", "Yes", "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Salary"] = df["Salary"].fillna(df["Salary"].median())

df = df.drop_duplicates()

print("\nDataset After Removing Duplicates:")
print(df)

df["Salary"] = df["Salary"].clip(
    lower=df["Salary"].quantile(0.05),
    upper=df["Salary"].quantile(0.95)
)

label_encoder = LabelEncoder()

df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Purchased"] = label_encoder.fit_transform(df["Purchased"])

df = pd.get_dummies(df, columns=["City"], dtype=int)

df = df.drop("Name", axis=1)

print("\nDataset After Encoding:")
print(df)

X = df.drop("Purchased", axis=1)
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

print("\nScaled Training Data:")
print(X_train)

print("\nScaled Testing Data:")
print(X_test)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nTarget Training Data:")
print(y_train.values)

print("\nTarget Testing Data:")
print(y_test.values)
