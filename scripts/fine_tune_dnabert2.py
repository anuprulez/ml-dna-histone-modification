import os
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from omegaconf.omegaconf import OmegaConf
from dataset_kmer import DNAKmerDataset, build_fixed_kmer_vocab
from utils import deduplicate_datasets

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "zhihan1996/DNA_bert_6"
MAX_LENGTH = 1000  # Length of DNA sequence
KMER = 6           # Must match model's k-mer
NUM_LABELS = 2     # Binary classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = OmegaConf.load("../configs/train.yaml")

# ----------------------------
# Tokenization using K-mers
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(seq):
    return tokenizer(
        seq["sequence"],            # or "text" or whatever your column is called
        padding="max_length",
        truncation=True,
        max_length=cfg.seq_len                  # DNABERT-2 max length; adjust if needed
    )

def load_dataset_from_csv(file_path):
    dataset = pd.read_csv(file_path, sep=",")
    print(dataset.head())
    data_dict = Dataset.from_dict({
        "sequence": dataset["query_subseq"].to_list(),
        "label": dataset["labels"].to_list()
    })
    tokenised_data = data_dict.map(tokenize_function, batched=True)
    return tokenised_data

deduplicate_datasets(cfg.train_file, cfg.test_file, cfg.val_file)

k = cfg.kmer_size
vocab = build_fixed_kmer_vocab(k)
print(f"Vocabulary size: {len(vocab)}")

cfg.vocab_size = len(vocab)

tr_dataset = load_dataset_from_csv(cfg.train_file)
te_dataset = load_dataset_from_csv(cfg.test_file)
val_data = load_dataset_from_csv(cfg.val_file)

print(tr_dataset)

#tokenized_train = train_dataset.map(tokenize_function, batched=True)
#tokenized_test = test_dataset.map(tokenize_function, batched=True)

# ----------------------------
# Model & Trainer Setup
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
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
    learning_rate=2e-5,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.n_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_dataset,
    eval_dataset=te_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
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
