import os
import re
import sys
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from omegaconf.omegaconf import OmegaConf
from dataset_kmer import DNAKmerDataset, build_fixed_kmer_vocab
from utils import deduplicate_datasets

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "zhihan1996/DNABERT-2-117M" #"zhihan1996/DNA_bert_6"
MAX_LENGTH = 500  # Length of DNA sequence
KMER = 6           # Must match model's k-mer
NUM_LABELS = 2     # Binary classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

cfg = OmegaConf.load("../configs/train.yaml")

# ----------------------------
# Tokenization using K-mers
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


print(tokenizer("AATATTTAAAATAAAATAATTAGGTAAATGTAATGGGATAAATACTTGTACACAAACTTGT"))

def tokenize_sequences(examples):
    """Convert DNA sequences to token IDs"""
    return tokenizer(
        examples['sequence'],
        padding='longest',    # Pad to longest sequence in batch
        truncation=True,       # Truncate to model's max length (512)
        #return_tensors='pt',   # Return PyTorch tensors
        max_length=cfg.seq_len         # DNABERT-2 max context
    )

def sanitize_dna(sequence):
    """Ensure uppercase ACGT only"""
    sequence = sequence.upper()
    sequence = re.sub(r'[^ACGT]', '', sequence)  # Remove non-ACGT
    return sequence

def load_dataset_from_csv(file_path):
    dataset = pd.read_csv(file_path, sep=",")
    #dataset = load_dataset('csv', data_files=file_path)

    print(dataset.head())

    data_dict = Dataset.from_dict({
        "sequence": dataset["query_subseq"].to_list(),
        "label": dataset["labels"].to_list()
    })
    #data_dict = data_dict.map(lambda x: {'sequence': sanitize_dna(x['sequence'])})
    tokenised_data = data_dict.map(tokenize_sequences, batched=True)
    print(tokenised_data)
    return tokenised_data

deduplicate_datasets(cfg.train_file, cfg.test_file, cfg.val_file)

tr_dataset = load_dataset_from_csv(cfg.train_file)
te_dataset = load_dataset_from_csv(cfg.test_file)
val_data = load_dataset_from_csv(cfg.val_file)

#print(tr_dataset)
#print(tr_dataset["sequence"][:1])
#print(tr_dataset["label"][:1])
#print( tr_dataset["input_ids"][:1])

#sys.exit()

# ----------------------------
# Model & Trainer Setup
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           trust_remote_code=True,
                                                           use_safetensors=True,
                                                           use_flash_attention_2=False)

#config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
#model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config, use_safetensors=True)

print(model.config)
model.config.use_flash_attn = False
model.config.attn_implementation = "eager"
#model.encoder.layer[0].attention.flash_attn = False
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation metric
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir="./dnabert2-finetune",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.n_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_accumulation_steps=64
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_dataset,
    eval_dataset=te_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ----------------------------
# Training
# ----------------------------
trainer.train()

# ----------------------------
# Evaluation
# ----------------------------
eval_results = trainer.evaluate()
print("Evaluation:", eval_results)
