import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import mlflow
import mlflow.pytorch

from dataset import DNADataset
from model import DNAClassifier
from utils import plot_metrics, compute_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 10
D_PATH = "../data/reads_dataframes/"
O_PATH = "../data/outputs/"


def train():
    mlflow.set_experiment("DNA_Transformer_Classification")
    #mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run():
        train_data = DNADataset(D_PATH + "H3K27me3_train_read.csv")
        val_data = DNADataset(D_PATH + "H3K27me3_val_read.csv")
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        model = DNAClassifier().to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=1e-4)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(N_EPOCHS):
            model.train()
            total_loss, total_acc = 0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += compute_accuracy(torch.sigmoid(logits), y)

            train_losses.append(total_loss / len(train_loader))
            train_accs.append(total_acc / len(train_loader))

            # Evaluation
            model.eval()
            val_loss, val_acc = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device).float()
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    val_loss += loss.item()
                    val_acc += compute_accuracy(torch.sigmoid(logits), y)

            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc / len(val_loader))

            print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}")

        # Plot
        plot_metrics(train_losses, val_losses, train_accs, val_accs, O_PATH + "metrics.png")

        # Log MLflow artifacts
        mlflow.log_param("model", "Transformer")
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("val_accuracy", val_accs[-1])
        mlflow.log_artifact(O_PATH + "metrics.png")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    
    train()
