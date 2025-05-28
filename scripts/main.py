import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

from omegaconf.omegaconf import OmegaConf

from dataset import DNADataset
from utils import plot_metrics, compute_accuracy, show_attention_matrix
from transformer_model import CustomTransformerClassifier as DNAClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = OmegaConf.load("../configs/train.yaml")

def train():

    train_data = DNADataset(cfg.train_file)
    val_data = DNADataset(cfg.val_file)
    test_data = DNADataset(cfg.test_file)

    print(f"Train data: {len(train_data)}, Validation data: {len(val_data)}, Test data:  {len(test_data)}")
    
    #train_data = DNADataset(D_PATH + "H3K27me3_train_read.csv", k=1)
    #val_data = DNADataset(D_PATH + "H3K27me3_val_read.csv", k=1)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size)

    #vocab_size = max(train_data.get_vocab_size(), val_data.get_vocab_size())
    #seq_len = len(train_data[0][0])

    model = DNAClassifier(seq_len=cfg.seq_len, 
                          vocab_size=cfg.vocab_size, 
                          d_model=cfg.d_model, 
                          n_heads=cfg.n_heads, 
                          ffn_dim=cfg.ffn_dim, 
                          n_layers=cfg.n_layers, 
                          dropout=cfg.dropout).to(device)
    print(f"Model: {model}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(cfg.n_epochs):
        model.train()
        total_loss, total_acc = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            logits, _ = model(x)
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
                logits, _ = model(x)
                loss = loss_fn(logits, y)
                val_loss += loss.item()
                val_acc += compute_accuracy(torch.sigmoid(logits), y)

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Train Acc={train_accs[-1]:.4f}, Val Acc={val_accs[-1]:.4f}")

    # Plot
    print("Evaluting on test data")
    plot_metrics(train_losses, val_losses, train_accs, val_accs, cfg.plot_acc_loss)
    #evaluate_test(model, test_loader, cfg)


def evaluate_test(model, te_data_loader, cfg):
    model.eval()
    with torch.no_grad():
        ix = 0
        for x, y in te_data_loader:
            x, y = x.to(device), y.to(device).float()
            logits, att_matrices = model(x)
            print(f"Plotting attention matrix of shape: {att_matrices.shape}")
            show_attention_matrix(att_matrices[0].cpu().numpy(), cfg.plot_attention_mat)
            ix += 1
            break


if __name__ == "__main__":
    train()
