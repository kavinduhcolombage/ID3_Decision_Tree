import numpy as np

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

outlook = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain']
temp = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong']
play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

print("Information Gain for wind:", information_gain(wind, play))