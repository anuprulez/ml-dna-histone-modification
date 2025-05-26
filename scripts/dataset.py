import pandas as pd
import torch
from torch.utils.data import Dataset

DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

class DNADataset(Dataset):
    def __init__(self, csv_file):
        
        df = pd.read_csv(csv_file)
        self.sequences = df['query_subseq'].apply(self.encode).tolist()
        self.labels = df['labels'].astype(int).tolist()

    def encode(self, seq):
        return [DNA_VOCAB[nt] for nt in seq]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

