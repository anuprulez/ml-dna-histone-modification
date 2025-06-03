
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=4096, embed_dim=128, lstm_hidden=128, lstm_layers=1, num_classes=1):
        super(CNNLSTMClassifier, self).__init__()

        # Embedding layer for k-mers
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        # CNN to learn local motifs
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 995 -> 497

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 497 -> 248

            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 248 -> 124
        )

        # LSTM to learn long-range dependencies
        self.lstm = nn.LSTM(
            input_size=128,  # Must match CNN output channels
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch_size, 995) — sequence of k-mer indices
        x = self.embedding(x)              # (batch_size, 995, embed_dim)
        x = x.permute(0, 2, 1)             # (batch_size, embed_dim, 995)
        x = self.cnn(x)                    # (batch_size, 128, ~124)
        x = x.permute(0, 2, 1)             # (batch_size, 124, 128)
        lstm_out, _ = self.lstm(x)         # (batch_size, 124, 2*lstm_hidden)
        x = lstm_out[:, -1, :]             # (batch_size, 2*lstm_hidden)
        out = self.classifier(x)           # (batch_size, num_classes)
        return out.squeeze(1)
