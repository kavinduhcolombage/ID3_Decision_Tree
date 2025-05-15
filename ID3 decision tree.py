import numpy as np
import pandas as pd

# calculating entropy
def entropy(y):  
    counts = np.bincount(y)  # Count occurrences of each label
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

# calculating information gain
def information_gain(feature, labels):
    label_array = np.array([1 if l == 'Yes' else 0 for l in labels])
    total_entropy = entropy(label_array)

    categories = set(feature)
    total = len(feature)
    weighted_entropy = 0

    for cat in categories:
        subset = [1 if labels[i] == 'Yes' else 0 for i in range(len(labels)) if feature[i] == cat]
        subset_entropy = entropy(np.array(subset))
        weighted_entropy += (len(subset) / total) * subset_entropy

    return total_entropy - weighted_entropy

def best_feature_gain(features_dict, labels):
    gains = {}
    for name, feat in features_dict.items():
        gains[name] = information_gain(feat, labels)
    return gains

outlook = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']



df = pd.DataFrame({
    'outlook':    outlook,
    'temperature': temp,
    'humidity':   humidity,
    'wind':       wind,
    'play':       play
})


def id3(df, features, label_col):
    # If all labels are the same, return that label
    if len(df[label_col].unique()) == 1:
        return df[label_col].iloc[0]
    # If no features left, return the most common label
    if not features:
        return df[label_col].mode()[0]
    
    # Calculate information gain for each available feature and select the best one
    gains = best_feature_gain({f: df[f].tolist() for f in features}, df[label_col].tolist())
    best = max(gains, key=gains.get)
    
    # Create a new subtree with the best feature as the root
    tree = {best: {}}
    for val in df[best].unique():    # For each unique value of the best feature, create a branch
        subset = df[df[best] == val]    # Create a subset of the data where the best feature has the current value
        # Remove the used feature
        new_features = [f for f in features if f != best]
        # Recursively build the subtree for this branch
        tree[best][val] = id3(subset, new_features, label_col)
    return tree


features_list = ['outlook', 'temperature', 'humidity', 'wind']
tree = id3(df, features_list, 'play')
print("Decision Tree:")
print(tree)