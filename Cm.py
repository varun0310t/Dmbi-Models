import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.gridspec as gridspec
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to create a consistent confusion matrix plot
def plot_confusion_matrix(cm, title, ax, accuracy):
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        cbar=False,
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant'],
        ax=ax,
        annot_kws={"size": 16}
    )
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    
    # Add accuracy text
    ax.text(0.5, -0.2, f'Accuracy: {accuracy:.4f}', 
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Create the confusion matrices manually (REPLACE THESE WITH YOUR ACTUAL VALUES)
# ANN confusion matrix
ann_cm = np.array([
    [59, 1],   # True Negatives, False Positives
    [12, 42]    # False Negatives, True Positives
])
ann_accuracy = (ann_cm[0,0] + ann_cm[1,1]) / np.sum(ann_cm)

# CNN confusion matrix
cnn_cm = np.array([
    [71, 0],
    [20, 23]
])
cnn_accuracy = (cnn_cm[0,0] + cnn_cm[1,1]) / np.sum(cnn_cm)

# Random Forest confusion matrix
rf_cm = np.array([
    [97, 11],
    [7, 56]
])
rf_accuracy = (rf_cm[0,0] + rf_cm[1,1]) / np.sum(rf_cm)

# Create figure with better styling
plt.figure(figsize=(20, 7))
plt.subplots_adjust(wspace=0.3)

# Plot each confusion matrix with better spacing
ax1 = plt.subplot(1, 3, 1)
plot_confusion_matrix(ann_cm, 'ANN Model', ax1, ann_accuracy)

ax2 = plt.subplot(1, 3, 2)
plot_confusion_matrix(cnn_cm, 'CNN Model', ax2, cnn_accuracy)

ax3 = plt.subplot(1, 3, 3)
plot_confusion_matrix(rf_cm, 'Random Forest Model', ax3, rf_accuracy)

# Add a common title for the entire figure
plt.suptitle('Comparison of Model Performance on Breast Cancer Dataset', 
             fontsize=20, fontweight='bold', y=0.98)

# Add a footnote
plt.figtext(0.5, 0.01, 
            'Benign: Non-cancerous | Malignant: Cancerous', 
            ha='center', fontsize=12, fontstyle='italic')

# Save with higher quality and show
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for the suptitle
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
models = ['ANN', 'CNN', 'Random Forest']
accuracies = [ann_accuracy, cnn_accuracy, rf_accuracy]

print("\nModel Performance Summary:")
print("=" * 50)
print(f"{'Model':<15} {'Accuracy':<10}")
print("-" * 50)
for model, acc in zip(models, accuracies):
    print(f"{model:<15} {acc:.4f}")
print("=" * 50)
print(f"Best Model: {models[np.argmax(accuracies)]} (Accuracy: {max(accuracies):.4f})")

# Calculate metrics for each model
models = ['ANN', 'CNN', 'Random Forest']

# Define function to calculate metrics from confusion matrix
def get_metrics(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1 Score': f1
    }

# Get metrics for each model
metrics_ann = get_metrics(ann_cm)
metrics_cnn = get_metrics(cnn_cm)
metrics_rf = get_metrics(rf_cm)

# Create a DataFrame for visualization
metrics_df = pd.DataFrame({
    'ANN': metrics_ann,
    'CNN': metrics_cnn,
    'Random Forest': metrics_rf
})

# Plot the metrics comparison
plt.figure(figsize=(12, 8))
metrics_df.T.plot(kind='bar', figsize=(12, 8), rot=0, width=0.7)
plt.title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Add value labels on bars
for i, container in enumerate(plt.gca().containers):
    plt.gca().bar_label(container, fmt='%.2f', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_metrics_comparison.png', dpi=300)
plt.show()

# Print detailed metrics table
print("\nDetailed Model Performance Metrics:")
print("=" * 70)
print(metrics_df)
print("=" * 70)