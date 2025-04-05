import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and preprocess the data
df = pd.read_csv("breast-cancer.csv")
df = df.drop(["Unnamed: 32", "id"], axis=1)

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = nb_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance (using mean probabilities for each class)
feature_importance = np.abs(nb_model.theta_[1] - nb_model.theta_[0])
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Most Important Features - Naive Bayes')
plt.tight_layout()
plt.show()

# Function to predict for a single case
def predict_single_case(row_num):
    if row_num < 0 or row_num >= len(X):
        return "Invalid row number"
    
    row_data = X.iloc[row_num].values.reshape(1, -1)
    scaled_data = scaler.transform(row_data)
    prediction = nb_model.predict(scaled_data)
    probability = nb_model.predict_proba(scaled_data)
    
    return {
        'prediction': prediction[0],
        'probability': max(probability[0])
    }

# Example usage
print("\nExample prediction:")
result = predict_single_case(10)
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.4f}")
