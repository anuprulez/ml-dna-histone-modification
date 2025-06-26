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

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from omegaconf.omegaconf import OmegaConf
from dataset_kmer import DNAKmerDataset, build_fixed_kmer_vocab
from utils import deduplicate_datasets

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "zhihan1996/DNABERT-2-117M" #"zhihan1996/DNA_bert_6"
#MAX_LENGTH = 500  # Length of DNA sequence
#KMER = 6           # Must match model's k-mer
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
        padding='max_length',
        truncation=True,
        max_length=cfg.seq_len
    )

def load_dataset_from_csv(file_path):
    dataset = pd.read_csv(file_path, sep=",")
    print(dataset.head())

    data_dict = Dataset.from_dict({
        "sequence": dataset["query_subseq"].to_list(),
        "label": dataset["labels"].to_list()
    })
    tokenised_data = data_dict.map(tokenize_sequences, batched=True)
    print(tokenised_data["sequence"][:1], len(tokenised_data["sequence"][:1][0]))
    print(tokenised_data["label"][:1])
    print(tokenised_data["input_ids"][:1], len(tokenised_data["input_ids"][:1][0]))
    return tokenised_data

deduplicate_datasets(cfg.train_file, cfg.test_file, cfg.val_file)

tr_dataset = load_dataset_from_csv(cfg.train_file)
te_dataset = load_dataset_from_csv(cfg.test_file)
val_data = load_dataset_from_csv(cfg.val_file)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           trust_remote_code=True,
                                                           use_safetensors=True)
print(model.config)
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    print(predictions[0].shape, predictions[1].shape)
    print(labels)
    print("Labels shape:", labels.shape)

    if predictions[0].ndim == 2:
        preds = np.argmax(predictions[0], axis=1)
    # If shape is not as expected, raise error or handle accordingly
    else:
        raise ValueError(f"Unexpected predictions shape: {predictions[0].shape}")

    print("Predictions shape:", preds.shape)
    print(preds)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }

training_args = TrainingArguments(
    output_dir="./dnabert2-finetune",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.n_epochs,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=0,
    load_best_model_at_end=False,
    metric_for_best_model="f1",
)

class NoSaveTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        pass

trainer = NoSaveTrainer(
    model=model,
    args=training_args,
    train_dataset=tr_dataset,
    eval_dataset=te_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

'''trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_dataset,
    eval_dataset=te_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)'''

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation:", eval_results)
