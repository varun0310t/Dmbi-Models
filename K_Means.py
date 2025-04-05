import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix


# Load and preprocess data
df = pd.read_csv("breast-cancer.csv")
df = df.drop(["Unnamed: 32", "id"], axis=1)

# Separate features and encode diagnosis
X = df.drop("diagnosis", axis=1)
y = pd.get_dummies(df["diagnosis"], drop_first=True)  # M=1, B=0

# Scale the features (preserve feature names)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# Find optimal k using elbow method
inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.tight_layout()
plt.show()

# Apply K-means with k=2 (since we know there are 2 diagnosis categories)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # Explicitly set n_init
clusters = kmeans.fit_predict(X_scaled)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-means Clusters Visualization (PCA)')
plt.colorbar(scatter)
plt.show()

# Compare with actual diagnosis
cm = confusion_matrix(y, clusters)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Clustering Results vs Actual Diagnosis')
plt.xlabel('Predicted Clusters')
plt.ylabel('Actual Diagnosis')
plt.show()

# Feature importance based on cluster centers
feature_importance = np.abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Most Important Features for Clustering')
plt.tight_layout()
plt.show()

# Print clustering performance metrics
print("Silhouette Score:", silhouette_score(X_scaled, clusters))
print("\nCluster Distribution:")
print(pd.Series(clusters).value_counts())
print("\nActual Distribution:")
print(y.value_counts())

# Calculate and print accuracy
from sklearn.metrics import accuracy_score

# Convert diagnosis labels to numeric (0 and 1)
true_labels = pd.get_dummies(df["diagnosis"], drop_first=True).values

# Calculate accuracy (note: might need to flip labels if clusters are inversely assigned)
accuracy = max(
    accuracy_score(true_labels, clusters),
    accuracy_score(true_labels, 1 - clusters)
)
print("\nClustering Accuracy: {:.2f}%".format(accuracy * 100))

def predict_cluster(row_num):
    if row_num < 0 or row_num >= len(X):
        return "Invalid row number"
    
    row_data = X.iloc[row_num].values.reshape(1, -1)
    scaled_data = pd.DataFrame(
        scaler.transform(row_data),
        columns=X.columns
    )
    cluster = kmeans.predict(scaled_data)[0]
    
    return {
        'cluster': cluster,
        'actual_diagnosis': df.iloc[row_num]['diagnosis']
    }

# Example prediction
print("\nExample cluster prediction:")
result = predict_cluster(10)
print(f"Predicted Cluster: {result['cluster']}")
print(f"Actual Diagnosis: {result['actual_diagnosis']}")
