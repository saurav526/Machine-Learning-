import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'High', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

def gini(target):
    values, counts = np.unique(target, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob ** 2)

def gini_split(data, feature, target="PlayTennis"):
    values, counts = np.unique(data[feature], return_counts=True)
    
    gini_total = 0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        gini_total += (c / len(data)) * gini(subset[target])
    
    return gini_total

# Find best feature using Gini
features = df.columns[:-1]
gini_values = [gini_split(df, f) for f in features]

best_gini_feature = features[np.argmin(gini_values)]
print("Best Feature (using Gini):", best_gini_feature)