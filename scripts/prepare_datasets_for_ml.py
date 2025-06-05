import os
import time
import random
from fireducks.pandas import pandas as pd
import glob
import pysam
from pyspark.sql import SparkSession
import numpy as np
from sklearn.model_selection import train_test_split
from pybedtools import BedTool
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import pairwise2
from Bio import SeqIO


from omegaconf.omegaconf import OmegaConf

cfg = OmegaConf.load("../configs/data_preprocess.yaml")
cfg_train = OmegaConf.load("../configs/train.yaml")

cfg.base_path = "../data/reads_dataframes/BAM_1_100/"


def prepare_datasets_for_ml():
    """
    Prepare datasets for machine learning.
    """
    # Load positive and negative reads
    df_p_reads = pd.read_csv(cfg.base_path + "all_p_read.csv")

    # Filter duplicates based on 'query_name' and 'reference_start'
    df_p_reads = df_p_reads.drop_duplicates(subset=['query_subseq'])

    print(df_p_reads)

    df_n_reads = pd.read_csv(cfg.base_path + "all_n_read.csv")
    # Filter duplicates based on 'query_name' and 'reference_start'
    df_n_reads = df_n_reads.drop_duplicates(subset=['query_subseq'])

    n_positive = len(df_p_reads.index)
    print(f"Number of positive reads: {n_positive}")
    print(f"Number of negative reads before sampling: {len(df_n_reads)}")

    # sample negative reads to match the number of positive reads
    df_n_reads = df_n_reads.sample(n=n_positive, random_state=42)
    print(f"Positive reads: {len(df_p_reads)}, Negative reads: {len(df_n_reads)}")
    # Reset index for both DataFrames
    # Combine positive and negative reads
    df_p_reads["labels"] = 1  # Positive samples
    df_n_reads["labels"] = 0  # Negative samples
    df_p_reads.reset_index(drop=True, inplace=True)
    df_combined = pd.concat([df_p_reads, df_n_reads], ignore_index=True)

    # Shuffle the dataset
    df_combined = df_combined.sample(frac=1).reset_index(drop=True)

    # Take only the first 1000 bases of the sequence
    df_combined['query_subseq'] = df_combined['query_subseq'].str[:cfg_train.seq_len]

    print(len(df_combined['query_subseq'][0]))
    print(df_combined.head())

    # Save the combined dataset
    df_combined.to_csv(cfg.base_path + "PAS56325_pass_e7d20a27_dca18cab_1_200_H3K27me3_combined_read.csv", index=False)

    df_ml = df_combined[['query_subseq']]
    df_y = df_combined['labels']

    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        df_ml, df_y, test_size=0.2, random_state=42, stratify=df_y
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Optionally recombine into DataFrames
    train_df = X_tr.copy()
    train_df['labels'] = y_tr

    test_df = X_te.copy()
    test_df['labels'] = y_te

    val_df = X_val.copy()
    val_df['labels'] = y_val

    train_df.to_csv(cfg.base_path + "H3K27me3_train_read.csv", index=False)
    val_df.to_csv(cfg.base_path + "H3K27me3_val_read.csv", index=False)
    test_df.to_csv(cfg.base_path + "H3K27me3_test_read.csv", index=False)

    print("Combined dataset saved.")


def analyze_peaks():
    """
    Analyze H3K27me3 peaks.
    """
    # Load the dataset
    df_h3k27me3_peaks = pd.read_csv("../data/H3K27me3_narrow_peaks.bed", sep="\t", header=None)

    print(df_h3k27me3_peaks.head())

    df_chr = df_h3k27me3_peaks[df_h3k27me3_peaks[0] == cfg.choromosome_name]

    peak_range = df_h3k27me3_peaks[2] - df_h3k27me3_peaks[1]

    print("Min, Max, Median, Mean", min(peak_range), max(peak_range), np.median(peak_range), np.mean(peak_range))

    df_chr["peak_range"] = peak_range

    print(df_chr.head())

    df_chr.columns = ["chr_name", "peak_start", "peak_end", "peak_range"]

    plt.figure(figsize=(10, 6))
    # Plot histogram
    sns.histplot(df_chr["peak_range"], bins=20, kde=True)  # `kde=True` adds a smooth density curve
    plt.xlabel("Peak Range (bp)")
    plt.ylabel("Frequency")
    plt.title("Histogram of peak ranges for H3K27me3 peaks on chr1")
    plt.savefig("../data/outputs/h3k27me3_peaks_ranges_histogram.png")

    plt.figure(figsize=(10, 6))
    categories = df_chr.index.tolist()
    plt.bar(categories, df_chr["peak_range"].tolist(), color='skyblue')

    # Customize labels and title
    plt.xlabel("Number of Peaks")
    plt.ylabel("Peak Range (bp)")
    plt.title("Peak Range for H3K27me3 Peaks on chr1")
    plt.savefig("../data/outputs/h3k27me3_peaks_ranges_barplot.png")


def qc_sequences():
    """
    Check sequence similarities.
    """
    # Load the dataset
    df_neg_sequences = pd.read_csv(cfg.base_path + "all_n_read.csv", sep=",")
    print(df_neg_sequences.head())

    df_pos_sequences = pd.read_csv(cfg.base_path + "all_p_read.csv", sep=",")
    
    with open(cfg.base_path + "all_p_read.fasta", "w") as fasta_out:
        for in_id, row in df_pos_sequences.iterrows():
            fasta_out.write(f">{in_id}\n{row['query_subseq']}\n")

    with open(cfg.base_path + "all_n_read.fasta", "w") as fasta_out:
        for in_id, row in df_neg_sequences.iterrows():
            fasta_out.write(f">{in_id}\n{row['query_subseq']}\n")


def find_sequence_similarity(seq_path, postive):
    sequences = [str(record.seq) for record in SeqIO.parse(seq_path, "fasta")]
    
    sequences = sequences[0:50]
    print(sequences)
    # Pairwise global alignment
    overall_score = []
    for i in range(len(sequences)):
        row_score  = []
        for j in range(0, len(sequences)):
            alignments = pairwise2.align.globalxx(sequences[i], sequences[j])
            score = alignments[0].score
            normalized_score = score / max(len(sequences[i]), len(sequences[j]))
            row_score.append(normalized_score)
            print(f"Normalized alignment score between Seq{i+1} and Seq{j+1}: {normalized_score:.3f}")
        overall_score.append(row_score)

    print(overall_score)
    matrix = np.array(overall_score)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.title("Sequence Similarity Heatmap")
    plt.tight_layout()
    plt.savefig(cfg.base_path + f"sequence_similarity_heatmap_{postive}.png")
    


if __name__ == "__main__":
    
    #qc_sequences()
    #find_sequence_similarity(cfg.base_path + "all_p_read.fasta", "positive")
    #find_sequence_similarity(cfg.base_path + "all_n_read.fasta", "neagative")
    # Prepare datasets for machine learning
    prepare_datasets_for_ml()
