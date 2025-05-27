import torch
from torch.utils.data import Dataset
import pandas as pd

def kmer_tokenizer(seq, k=6):
    vocab = {}
    tokens = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer not in vocab:
            vocab[kmer] = len(vocab)
        tokens.append(vocab[kmer])
    return tokens, vocab

class DNADataset(Dataset):
    def __init__(self, csv_file, k=3):
        df = pd.read_csv(csv_file)
        self.k = k
        self.labels = df['labels'].astype(int).tolist()
        self.vocab = {}
        self.sequences = [self.encode(seq) for seq in df['query_subseq']]

    def encode(self, seq):
        tokens = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if kmer not in self.vocab:
                self.vocab[kmer] = len(self.vocab)
            tokens.append(self.vocab[kmer])
        return tokens

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

    def get_vocab_size(self):
        return len(self.vocab)
