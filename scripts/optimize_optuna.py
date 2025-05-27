import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import optuna
from sklearn.metrics import roc_auc_score, accuracy_score

from transformer_model import CustomTransformerClassifier
from dataset import DNADataset
from utils import compute_accuracy


N_EPOCHS = 5
D_PATH = "../data/reads_dataframes/"
O_PATH = "../data/outputs/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Trial hyperparameters
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    ffn_dim = trial.suggest_categorical("ffn_dim", [128, 256, 512])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    patience = 3

    train_dataset = DNADataset(D_PATH + "H3K27me3_train_read.csv", k=6)
    val_dataset = DNADataset(D_PATH + "H3K27me3_val_read.csv", k=6)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    vocab_size = max(train_dataset.get_vocab_size(), val_dataset.get_vocab_size())
    seq_len = len(train_dataset[0][0])

    model = CustomTransformerClassifier(
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_auc = 0
    epochs_no_improve = 0

    for epoch in range(N_EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        auc = roc_auc_score(val_labels, val_preds)
        acc = accuracy_score(val_labels, (torch.tensor(val_preds) > 0.5).int())

        # Early stopping
        if auc > best_auc:
            best_auc = auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        # Prune underperforming trials
        trial.report(auc, step=epoch)   
        if trial.should_prune():
            raise optuna.TrialPruned()
    print(f"Trial finished with AUC: {best_auc:.4f}, Accuracy: {acc:.4f}")
    return best_auc

def run_optimization(n_trials=25):

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    print("Best Trial:")
    print("  Value:", study.best_value)
    print("  Params:", study.best_params)


if __name__ == "__main__":
    run_optimization(n_trials=25)
