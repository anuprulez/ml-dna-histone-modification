import torch
from torch.utils.data import Dataset
from typing import List, Dict
from itertools import product

import pandas as pd
from pathlib import Path


def generate_all_kmers(k: int) -> List[str]:
    """Generate all possible DNA k-mers using A, C, G, T."""
    return [''.join(p) for p in product("ACGT", repeat=k)]


def generate_kmers_from_sequence(sequence: str, k: int) -> List[str]:
    """Split a DNA sequence into overlapping k-mers."""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def build_fixed_kmer_vocab(k: int) -> Dict[str, int]:
    """Generate fixed vocab with all possible k-mers."""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    all_kmers = generate_all_kmers(k)
    vocab.update({kmer: idx + 2 for idx, kmer in enumerate(all_kmers)})
    return vocab


def encode_sequence(sequence: str, k: int, vocab: Dict[str, int]) -> List[int]:
    kmers = generate_kmers_from_sequence(sequence, k)
    return [vocab.get(kmer, vocab['<UNK>']) for kmer in kmers]


class DNAKmerDataset(Dataset):
    def __init__(self, path_sequences: Path, k: int, vocab: Dict[str, int]):
        df = pd.read_csv(path_sequences)
        self.sequences = df['query_subseq'].tolist()
        self.encoded_sequences = [encode_sequence(seq, k, vocab) for seq in self.sequences]
        self.labels = torch.tensor(df['labels'].astype(int).tolist(), dtype=torch.long)
        

    def __len__(self):
        return len(self.encoded_sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.encoded_sequences[idx], dtype=torch.long)
        label = self.labels[idx]
        return sequence, label