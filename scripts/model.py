import torch
import torch.nn as nn

class DNAClassifier(nn.Module):
    def __init__(self, seq_len=1000, vocab_size=4, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x).squeeze(1)
