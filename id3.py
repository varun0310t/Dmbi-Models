import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def information_gain(X, y, threshold):
    parent_entropy = entropy(y)
    left_indices = X < threshold
    right_indices = X >= threshold
    n, n_left, n_right = len(y), np.sum(left_indices), np.sum(right_indices)
    if n_left == 0 or n_right == 0:
        return 0
    child_entropy = (n_left / n) * entropy(y[left_indices]) + (n_right / n) * entropy(y[right_indices])
    return parent_entropy - child_entropy

def best_split(X, y):
    best_feature, best_threshold, best_gain = None, None, -1
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            gain = information_gain(X[:, feature], y, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold

def build_tree(X, y, depth=0, max_depth=10):
    n_samples, n_features = X.shape
    if n_samples <= 1 or depth >= max_depth or len(np.unique(y)) == 1:
        leaf_value = np.argmax(np.bincount(y)) if len(y) > 0 else None
        return Node(value=leaf_value)
    feature, threshold = best_split(X, y)
    if feature is None:
        leaf_value = np.argmax(np.bincount(y)) if len(y) > 0 else None
        return Node(value=leaf_value)
    left_indices = X[:, feature] < threshold
    right_indices = X[:, feature] >= threshold
    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
        leaf_value = np.argmax(np.bincount(y)) if len(y) > 0 else None
        return Node(value=leaf_value)
    left = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth)
    right = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth)
    return Node(feature=feature, threshold=threshold, left=left, right=right)

def predict_tree(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature] < node.threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)

class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = build_tree(X, y, max_depth=self.max_depth)

    def predict(self, X):
        return np.array([predict_tree(self.tree, x) for x in X])

# Load the dataset
df = pd.read_csv('breast-cancer.csv')

# Preprocess the data
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)

# Evaluate the decision tree on the training set
train_predictions = clf.predict(X_train)
train_accuracy = np.mean(train_predictions == y_train)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Evaluate the decision tree on the testing set
test_predictions = clf.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)
print(f'Testing Accuracy: {test_accuracy:.2f}')

# Add this code to your script to get entropy and information gain details
print(f"Overall dataset entropy: {entropy(y):.4f}")

# Print entropy for each class
for class_val in np.unique(y):
    class_indices = y == class_val
    class_count = np.sum(class_indices)
    class_probability = class_count / len(y)
    print(f"Class {class_val} probability: {class_probability:.4f}, Count: {class_count}")

# Print top features by information gain
feature_gains = []
for feature in range(X.shape[1]):
    # Find best threshold for this feature
    best_gain = -1
    best_threshold = None
    thresholds = np.unique(X[:, feature])[:10]  # Limit to first 10 unique values for demonstration
    for threshold in thresholds:
        gain = information_gain(X[:, feature], y, threshold)
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold
    
    feature_gains.append((feature, best_gain, best_threshold))

# Sort features by information gain
feature_gains.sort(key=lambda x: x[1], reverse=True)

# Print top 5 features by information gain
print("\nTop 5 Features by Information Gain:")
for i, (feature, gain, threshold) in enumerate(feature_gains[:5]):
    print(f"Rank {i+1}: Feature {feature}, Gain: {gain:.4f}, Best Threshold: {threshold:.4f}")