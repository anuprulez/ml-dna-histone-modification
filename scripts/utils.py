import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns

def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_path="metrics.png"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    
    plt.tight_layout()
    plt.savefig(output_path)

def compute_accuracy(preds, labels):
    preds = (preds > 0.5).int()
    return accuracy_score(labels.cpu(), preds.cpu())


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
