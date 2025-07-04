from fireducks.pandas import pandas as pd
import glob
import pysam
import numpy as np
from sklearn.model_selection import train_test_split
from pybedtools import BedTool
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf.omegaconf import OmegaConf


cfg = OmegaConf.load("../configs/data_preprocess.yaml")

def load_preprocess():
    """
    Load and preprocess the dataset.
    """
    # Load the dataset
    df_h3k27me3_peaks = pd.read_csv("../data/H3K27me3_narrow_peaks.bedgraph" , sep="\t", header=None)
    # Preprocess the dataset
    df_h3k27me3_peaks.columns = ["chrom", "start", "end", "name", "score", "strand", "signalValue", "pValue", "qValue", "peak"]

    # Load BedGraph
    df = df_h3k27me3_peaks #pd.read_csv("input.bedgraph", sep="\t", header=None)

    # Select first 3 columns for BED
    df_bed = df.iloc[:, :3]

    # analyse H3K27me3 peaks
    #analyze_peaks()

    # Save as BED
    df_bed.to_csv("../data/H3K27me3_narrow_peaks.bed", sep="\t", header=False, index=False)


def filter_peaks(i_read, chip_peaks):
    """
    Filter peaks based on the reads.
    """
    w_p_reads = []
    w_n_reads = []
    aligned_pairs = i_read.get_aligned_pairs(matches_only=True)

    # Only keep mappings where both query and ref positions are present
    aligned_pairs = [(qpos, rpos) for qpos, rpos in aligned_pairs if qpos is not None and rpos is not None]

    # Convert to DataFrame for easier slicing
    df_pairs = pd.DataFrame(aligned_pairs, columns=["query_pos", "ref_pos"])

    # Slide a 1kb window over the reference positions
    for i in range(0, len(df_pairs) - cfg.window_size + 1, cfg.step_size):
        window = df_pairs.iloc[i:i + cfg.window_size]

        # Ensure the window is contiguous on reference (optional check)
        if window["ref_pos"].iloc[-1] - window["ref_pos"].iloc[0] != cfg.window_size - 1:
            continue  # skip non-contiguous spans

        ref_start = window["ref_pos"].iloc[0]
        ref_end = window["ref_pos"].iloc[-1] + 1
        query_start = window["query_pos"].iloc[0]
        query_end = window["query_pos"].iloc[-1] + 1

        query_subseq = i_read.query_sequence[query_start: query_end]

        # Make temporary BED line
        window_bed = BedTool(f"{i_read.reference_name}\t{ref_start}\t{ref_end}\n", from_string=True)

        overlap = window_bed.intersect(chip_peaks, wo=True)

        df_overlap = overlap.to_dataframe()
        
        '''if overlap:
            print("Overlap found")
            print(df_overlap.head())
            #print(window_bed, query_start, query_end)
            print(overlap, df_overlap["thickStart"], type(overlap), dir(overlap))
            print("================\n")'''

        # Check if it overlaps ChIP peak
        #if window_bed.intersect(chip_peaks, u=True):
        if overlap and df_overlap["thickStart"].loc[0] >= cfg.min_overlap:
            print(df_overlap["thickStart"])
            print("---------")
            # If it overlaps, create a positive sample
            #print(f"Positive sample: {i_read.query_name}, {i_read.reference_name}:{ref_start}-{ref_end}, Query: {query_subseq}")
            w_p_reads.append({
                'chromosome': i_read.reference_name,
                'query_name': i_read.query_name,
                'reference_start': ref_start,
                'reference_end': ref_end,
                'query_start': query_start,
                'query_end': query_end,
                'query_subseq': query_subseq,
                'mapping_quality': i_read.mapping_quality,
                'query_length': i_read.query_length,
            })
        else:
            # If it doesn't overlap, create a negative sample
            #print(f"Negative sample: {i_read.query_name}, {i_read.reference_name}:{ref_start}-{ref_end}, Query: {query_subseq}")
            w_n_reads.append({
                'chromosome': i_read.reference_name,
                'query_name': i_read.query_name,
                'reference_start': ref_start,
                'reference_end': ref_end,
                'query_start': query_start,
                'query_end': query_end,
                'query_subseq': query_subseq,
                'mapping_quality': i_read.mapping_quality,
                'query_length': i_read.query_length,
            })
    return w_p_reads, w_n_reads


def load_BAM_files():
    """
    Load BAM files.
    """
    # Get all .bam files in the current directory
    # BAM_files = glob.glob("../data/alignments/*.bam")
    BAM_files = glob.glob("../data/alignments/PAS56325_pass_e7d20a27_dca18cab_602.fastq.gz.bam")
    chip_peaks = BedTool("../data/H3K27me3_narrow_peaks.bed")

    print(f"Found {len(BAM_files)} BAM files.")
    f_p_reads = []
    f_n_reads = []
    for i, file in enumerate(BAM_files):
        print(f"Processing file {i+1}/{len(BAM_files)}: {file}")
        filterd_BAM = pysam.AlignmentFile(file, "rb")
        reads = []
        p_reads = []
        n_reads = []
        index_read = 0
        for read in filterd_BAM.fetch(until_eof=True):
            #print(read.reference_name, read.query_name, read.mapping_quality, read.query_length, read.reference_start, read.reference_end)
            if read.query_sequence in [None, "None"] or read.mapping_quality < cfg.min_mapping_quality or read.query_length < cfg.window_size or read.reference_name != cfg.choromosome_name:
                continue
            start = read.reference_start
            end = read.reference_end
            #print(f"{read.reference_name}\t{start}\t{end}")
            #print(f"Read name: {read.query_sequence}, Length: {read.query_length}, Mapping quality: {read.mapping_quality}")

            reads.append({
                'chromosome': read.reference_name,
                'query_name': read.query_name,
                'reference_start': read.reference_start,
                'reference_end': read.reference_end,
                'mapping_quality': read.mapping_quality,
                'query_length': read.query_length,
                'reference_length': read.reference_length,
                'is_reverse': read.is_reverse,
            })
            ##################### create positive/negative samples based on Chip-seq peaks #####################
            p_rd, n_rd = filter_peaks(read, chip_peaks)
            #print(f"Read {index_read} processed.")
            index_read += 1
            if index_read % 200 == 0:
                print(f"Processed {index_read} reads so far.")
                #break
            p_reads.extend(p_rd)
            n_reads.extend(n_rd)
        # Extend the main lists with the positive and negative reads
        f_p_reads.extend(p_reads)
        f_n_reads.extend(n_reads)
        # Read 10 BAM files
        if i == 10:
           break
    # save positive and negative reads to DataFrames
    df_p_reads = pd.DataFrame(f_p_reads)
    df_n_reads = pd.DataFrame(f_n_reads)
    print(f"Positive reads: {len(df_p_reads)}, Negative reads: {len(df_n_reads)}")
    # Save positive and negative reads to CSV files
    df_p_reads.to_csv(cfg.base_path + "all_p_read.csv", index=False)
    df_n_reads.to_csv(cfg.base_path + "all_n_read.csv", index=False)


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
    df_combined['query_subseq'] = df_combined['query_subseq'].str[:1000]

    # Save the combined dataset
    df_combined.to_csv(cfg.base_path + "PAS56325_pass_e7d20a27_dca18cab_602_H3K27me3_combined_read.csv", index=False)

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

    df_chr.columns = ["chr_name", "peak_start", "peank_end", "peak_range"]

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


if __name__ == "__main__":
    
    # Load and preprocess the dataset
    load_preprocess()

    # Load BAM files
    load_BAM_files()

    # Prepare datasets for machine learning
    # prepare_datasets_for_ml()