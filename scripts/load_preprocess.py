from fireducks.pandas import pandas as pd
import glob
import pysam
from pybedtools import BedTool


window_size = 1000
step_size = 10

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

    # Save as BED
    df_bed.to_csv("../data/H3K27me3_narrow_peaks.bed", sep="\t", header=False, index=False)

    return df_h3k27me3_peaks, df_bed


def filter_peaks(i_read, chip_peaks):
    """
    Filter peaks based on the reads.
    """
    

    aligned_pairs = i_read.get_aligned_pairs(matches_only=True)

    # Only keep mappings where both query and ref positions are present
    aligned_pairs = [(qpos, rpos) for qpos, rpos in aligned_pairs if qpos is not None and rpos is not None]

    # Convert to DataFrame for easier slicing
    df_pairs = pd.DataFrame(aligned_pairs, columns=["query_pos", "ref_pos"])

    #print(df_pairs)
    w_p_reads = []
    w_n_reads = []
    # Slide a 1kb window over the reference positions
    for i in range(0, len(df_pairs) - window_size + 1, step_size):
        window = df_pairs.iloc[i:i + window_size]

        # Ensure the window is contiguous on reference (optional check)
        if window["ref_pos"].iloc[-1] - window["ref_pos"].iloc[0] != window_size - 1:
            continue  # skip non-contiguous spans

        ref_start = window["ref_pos"].iloc[0]
        ref_end = window["ref_pos"].iloc[-1] + 1
        query_start = window["query_pos"].iloc[0]
        query_end = window["query_pos"].iloc[-1] + 1

        query_subseq = i_read.query_sequence[query_start: query_end]

        # Make temporary BED line
        window_bed = BedTool(f"{i_read.reference_name}\t{ref_start}\t{ref_end}\n", from_string=True)

        # Check if it overlaps ChIP peak
        if window_bed.intersect(chip_peaks, u=True):
            # If it overlaps, create a positive sample
            print(f"Positive sample: {i_read.query_name}, {i_read.reference_name}:{ref_start}-{ref_end}, Query: {query_subseq}")
            w_p_reads.append({
                'chromosome': i_read.reference_name,
                'query_name': i_read.query_name,
                'query_sequence': i_read.query_sequence,
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
                'query_sequence': i_read.query_sequence,
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
    BAM_files = glob.glob("../data/alignments/*.bam")
    chip_peaks = BedTool("../data/H3K27me3_narrow_peaks.bed")

    print(f"Found {len(BAM_files)} BAM files.")

    for file in BAM_files:
        filterd_BAM = pysam.AlignmentFile(file, "rb")
        reads = []
        p_reads = []
        n_reads = []
        index_read = 0
        for read in filterd_BAM.fetch(until_eof=True):
            if read.query_name != "None" and read.mapping_quality < 20 and read.query_length < 1000:
                continue
            start = read.reference_start
            end = read.reference_end
            print(f"{read.reference_name}\t{start}\t{end}")
            print(f"Read name: {read.query_sequence}, Length: {read.query_length}, Mapping quality: {read.mapping_quality}")

            reads.append({
                'chromosome': read.reference_name,
                'query_name': read.query_sequence,
                'reference_start': read.reference_start + 1,
                'reference_end': read.reference_end,
                'mapping_quality': read.mapping_quality,
                'query_length': read.query_length,
                'sequence_length': len(read.query_sequence),
                'reference_length': read.reference_length,
                'is_reverse': read.is_reverse,
            })

            ##################### create positive/negative samples based on Chip-seq peaks #####################
            p_rd, n_rd = filter_peaks(read, chip_peaks)
            print(f"Read {index_read} processed.")
            index_read += 1
            if index_read % 300 == 0:
                print(f"Processed {index_read} reads so far.")
                break
            p_reads.extend(p_rd)
            n_reads.extend(n_rd)

        # save positive and negative reads to DataFrames
        df_p_reads = pd.DataFrame(p_reads)
        df_n_reads = pd.DataFrame(n_reads)
        print(f"Positive reads: {len(df_p_reads)}, Negative reads: {len(df_n_reads)}")
        # Save positive and negative reads to CSV files
        df_p_reads.to_csv(f"../data/reads_dataframes/{file.split('/')[-1].split('.')[0]}_p_read.csv", index=False)
        df_n_reads.to_csv(f"../data/reads_dataframes/{file.split('/')[-1].split('.')[0]}_n_read.csv", index=False)


        # Now convert to DataFrame
        df_read = pd.DataFrame(reads)
        print(df_read)
        df_read.to_csv(f"../data/reads_dataframes/{file.split('/')[-1].split('.')[0]}_read.csv", index=False)
        break


if __name__ == "__main__":
    # Load and preprocess the dataset
    df_h3k27me3_peaks, df_bed = load_preprocess()

    # Print the first 5 rows of the dataset
    print(df_h3k27me3_peaks.head())

    # Load BAM files
    filterd_BAM = load_BAM_files()