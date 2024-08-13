import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_results(loss_hist, metric_hist):
    if not {'train', 'val'}.issubset(loss_hist.keys()) or not {'train', 'val'}.issubset(metric_hist.keys()):
        raise ValueError("Missing keys in loss_hist or metric_hist. Expected keys: 'train', 'val'.")
    
    epochs = len(loss_hist["train"])
    
    # Check if lengths match
    if len(loss_hist["val"]) != epochs or len(metric_hist["val"]) != epochs:
        raise ValueError("Mismatch in lengths of training and validation data.")
    
    df_loss = pd.DataFrame({
        'Epoch': np.arange(1, epochs + 1),
        'Train Loss': loss_hist["train"],
        'Validation Loss': loss_hist["val"]
    })
    
    df_metric = pd.DataFrame({
        'Epoch': np.arange(1, epochs + 1),
        'Train Accuracy': metric_hist["train"],
        'Validation Accuracy': metric_hist["val"]
    })
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))

    # Tracer la perte
    sns.lineplot(x='Epoch', y='Train Loss', data=df_loss, ax=ax[0], label='Train Loss', marker='o')
    sns.lineplot(x='Epoch', y='Validation Loss', data=df_loss, ax=ax[0], label='Validation Loss', marker='o')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Tracer l'exactitude
    sns.lineplot(x='Epoch', y='Train Accuracy', data=df_metric, ax=ax[1], label='Train Accuracy', marker='o')
    sns.lineplot(x='Epoch', y='Validation Accuracy', data=df_metric, ax=ax[1], label='Validation Accuracy', marker='o')
    ax[1].set_title('Accuracy Curve')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()



def plot_prediction_histogram(predictions, labels):
    plt.figure(figsize=(10, 6))
    
    # Histogram of predicted class labels
    plt.hist(predictions, bins=np.arange(-0.5, 2, 1), alpha=0.7, label='Predicted Labels', rwidth=0.8)
    
    # Histogram of true class labels
    plt.hist(labels, bins=np.arange(-0.5, 2, 1), alpha=0.7, label='True Labels', rwidth=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted and True Labels')
    plt.legend(loc='upper right')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.show()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_class_probabilities(y_scores, y_true):
    df = pd.DataFrame({
        'True Label': y_true,
        'Predicted Probability': y_scores
    })
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Predicted Probability', hue='True Label', multiple='stack', bins=30, palette='viridis')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities by True Labels')
    plt.legend(title='True Label')
    plt.show()

def load_and_analyze_results():
    # Load predictions and ground truth
    y_out = np.load("predictions.npy")
    y_gt = np.load("ground_truth.npy")
    
    # Convert logits to predicted class indices
    y_pred = np.argmax(y_out, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_gt, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_gt, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_gt, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot histograms of predictions
    plot_prediction_histogram(y_pred, y_gt)
    
    # Plot ROC curve
    y_scores = np.exp(y_out[:, 1])  # Probabilities for class 1
    plot_roc_curve(y_gt, y_scores)
    
    # Plot class probabilities
    plot_class_probabilities(y_scores, y_gt)
    
    # Load loss and accuracy histories
    with open('loss_hist.json', 'r') as f:
        loss_hist = json.load(f)
    with open('metric_hist.json', 'r') as f:
        metric_hist = json.load(f)
    
    plot_results(loss_hist, metric_hist)

if __name__ == "__main__":
    load_and_analyze_results()
