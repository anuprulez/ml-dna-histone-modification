import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    precision_recall_curve, average_precision_score
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, labels=None, output_path="confusion_matrix.png"):
    """
    Plots a confusion matrix using seaborn heatmap.
    
    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - labels: List of class labels (optional)
    - output_path: Path to save the confusion matrix image
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)


def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_path="metrics.png"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.grid(True)
    plt.title("Accuracy")
    
    plt.tight_layout()
    plt.savefig(output_path)
    

def create_precision_recall_curve(y_true, y_scores, output_path="precision_recall_curve.png"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(output_path)


def compute_accuracy(preds, labels):
    preds = (preds > 0.5).int()
    return accuracy_score(labels.cpu(), preds.cpu())


def deduplicate_datasets(path_tr, path_te, path_val):
    """
    Deduplicates training, validation, and test datasets.
    """
    tr_set = set(pd.read_csv(path_tr)['query_subseq'].tolist())
    te_set = set(pd.read_csv(path_te)['query_subseq'].tolist())
    val_set = set(pd.read_csv(path_val)['query_subseq'].tolist())

    tr_te = tr_set.intersection(te_set)
    tr_val = tr_set.intersection(val_set)
    te_val = te_set.intersection(val_set)

    print(f"Training and Test intersection: {len(tr_te)}")
    print(f"Training and Validation intersection: {len(tr_val)}")
    print(f"Test and Validation intersection: {len(te_val)}")

    if len(tr_te) > 0 or len(tr_val) > 0:
        raise ValueError("Training set has overlaps with Test or Validation sets. Please check your datasets.")


def show_attention_matrix(attention_weights, save_path, num_heads=4, num_tokens=1000, tokens=None):
    """
    Visualizes the attention matrix using seaborn heatmap.
    
    Parameters:
    - attention_weights: 2D numpy array of shape (num_tokens, num_tokens)
    - tokens: List of token labels for x and y axes (optional)
    """
    #attention_matrix = np.random.rand(10, 10)

    # (Optional) Example token labels
    # Simulate attention data for 4 heads, each with a 10x10 matrix
    #num_heads = 4
    #num_tokens = 1000
    #attention_matrices = np.random.rand(num_heads, num_tokens, num_tokens)

    # Optional: token labels
    tokens = [f"T{i}" for i in range(num_tokens)]

    # Create subplots
    cols = 2
    rows = (num_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i < num_heads:
            sns.heatmap(attention_weights[i], ax=ax, xticklabels=tokens, yticklabels=tokens,
                        cmap="viridis", annot=False)
            ax.set_title(f"Head {i+1}")
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
        else:
            ax.axis('off')  # Turn off unused subplots

    plt.tight_layout()
    plt.savefig(save_path)
