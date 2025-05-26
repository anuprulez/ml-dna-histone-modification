import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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