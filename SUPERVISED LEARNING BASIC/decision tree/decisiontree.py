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

# Entropy function
def entropy(target):
    values, counts = np.unique(target, return_counts=True)
    prob = counts / counts.sum()
    return -np.sum(prob * np.log2(prob))

# Information Gain
def info_gain(data, feature, target="PlayTennis"):
    total_entropy = entropy(data[target])
    
    values, counts = np.unique(data[feature], return_counts=True)
    
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        weighted_entropy += (c / len(data)) * entropy(subset[target])
    
    return total_entropy - weighted_entropy

# Build simple tree (1 level split)
def best_feature(data):
    features = data.columns[:-1]
    gains = [info_gain(data, f) for f in features]
    return features[np.argmax(gains)]

# Find best split
best = best_feature(df)
print("Best Feature (using Information Gain):", best)