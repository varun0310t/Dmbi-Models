# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data
df = pd.read_csv("breast-cancer.csv")
df.head()

df.drop(["id", "Unnamed: 32"], inplace=True, axis=1)
df.head()

# Prepare target variable - convert to numerical (0 for M, 1 for B)
y_original = df["diagnosis"]
y = pd.get_dummies(y_original).values
x = df.drop(["diagnosis"], axis=1)

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=69)

# Scale data
scaler = RobustScaler()
scaled_xtrain = scaler.fit_transform(xtrain)
scaled_xtest = scaler.transform(xtest)

# Create and train the neural network
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), 
                    activation='relu', 
                    solver='adam', 
                    alpha=0.1,
                    batch_size='auto', 
                    learning_rate='constant', 
                    learning_rate_init=0.001,
                    max_iter=15, 
                    random_state=42)

mlp.fit(scaled_xtrain, ytrain)

# Evaluate the model
y_pred = mlp.predict(scaled_xtest)
accuracy = accuracy_score(ytest.argmax(axis=1), y_pred.argmax(axis=1))
print(f"Accuracy: {accuracy:.4f}")

# Get confusion matrix
y_true = np.argmax(ytest, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_true, y_pred_class)
print("Confusion Matrix:")
print(cm)