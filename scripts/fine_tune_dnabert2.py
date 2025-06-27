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
from utils import plot_metrics, compute_accuracy, show_attention_matrix, \
    deduplicate_datasets, plot_confusion_matrix, create_precision_recall_curve


MODEL_NAME = "zhihan1996/DNA_bert_6" #"zhihan1996/DNABERT-2-117M"
NUM_LABELS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

cfg = OmegaConf.load("../configs/train.yaml")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

#print(tokenizer("AATATTTAAAATAAAATAATTAGGTAAATGTAATGGGATAAATACTTGTACACAAACTTGT"))

data_collator = DataCollatorWithPadding(tokenizer)

def kmerize(sequence, k=6):
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def tokenize_sequences(examples):
    kmer_seqs = [kmerize(seq, k=cfg.kmer_size) for seq in examples["sequence"]]
    return tokenizer(kmer_seqs, padding="max_length", truncation=True, max_length=cfg.seq_len)


'''def tokenize_sequences(examples):
    return tokenizer(
        examples['sequence'],
        padding='max_length',
        #truncation=True,
        max_length=cfg.seq_len
    )'''

def load_dataset_from_csv(file_path):
    dataset = pd.read_csv(file_path, sep=",")
    print(dataset.head())
    print("Unique labels:", set(dataset["labels"]))
    data_dict = Dataset.from_dict({
        "sequence": dataset["query_subseq"].to_list(),
        "label": dataset["labels"].astype(int).to_list()
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

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    print(type(predictions), type(labels))
    print(predictions[0].shape, predictions[1].shape)
    preds = np.argmax(predictions[0], axis=1)
    print("Labels shape:", labels.shape)
    print("Predictions shape:", preds.shape)
    print(labels)
    print(preds)
    plot_confusion_matrix(preds, labels, labels=[0, 1], output_path=cfg.plot_confusion_matrix)
    create_precision_recall_curve(labels, predictions[0][:, 1], output_path=cfg.plot_precision_recall_curve)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }

def compute_metrics_kmer_model(eval_preds):
    predictions, labels = eval_preds
    print(type(predictions), type(labels))
    print(predictions.shape)
    preds = np.argmax(predictions, axis=1)
    print("Labels shape:", labels.shape)
    print("Predictions shape:", preds.shape)
    print(labels)
    print(preds)
    plot_confusion_matrix(preds, labels, labels=[0, 1], output_path=cfg.plot_confusion_matrix)
    create_precision_recall_curve(labels, predictions[:, 1], output_path=cfg.plot_precision_recall_curve)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }

training_args = TrainingArguments(
    output_dir="./dnabert2-finetune",
    eval_strategy="steps",
    save_strategy="no",
    eval_steps=cfg.eval_steps,
    learning_rate=cfg.learning_rate,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.n_epochs,
    weight_decay=cfg.weight_decay,
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
    data_collator=data_collator,
    compute_metrics=compute_metrics_kmer_model
)

trainer.train()
eval_results = trainer.evaluate()
print("Evaluation:", eval_results)