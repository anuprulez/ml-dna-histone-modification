from fireducks.pandas import pandas as pd
import pysam


def load_preprocess():
    """
    Load and preprocess the dataset.
    """
    # Load the dataset
    df_h3k27me3_peaks = pd.read_csv("data/H3K27me3_narrow_peaks.bedgraph" , sep="\t", header=None)

    # Preprocess the dataset
    df_h3k27me3_peaks.columns = ["chrom", "start", "end", "name", "score", "strand"]

    return df_h3k27me3_peaks


def load_BAM_files():
    """
    Load BAM files.
    """

    # Load the BAM files

    filterd_BAM = pysam.AlignmentFile("ex1.bam", "rb")

    print(filterd_BAM.header)

    return filterd_BAM



if __name__ == "__main__":
    # Load and preprocess the dataset
    df_h3k27me3_peaks = load_preprocess()

    # Print the first 5 rows of the dataset
    print(df_h3k27me3_peaks.head())