import os
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "zhanglab/bert-base-dna-6"
MAX_LENGTH = 1000  # Length of DNA sequence
KMER = 6           # Must match model's k-mer
NUM_LABELS = 2     # Binary classification

# ----------------------------
# Sample Dataset (for demonstration)
# ----------------------------
def generate_dummy_dna_dataset(num_samples=1000, seq_len=1000):
    def random_dna(length):
        return ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=length))

    sequences = [random_dna(seq_len) for _ in range(num_samples)]
    labels = np.random.randint(0, 2, size=num_samples).tolist()
    return Dataset.from_dict({"sequence": sequences, "label": labels})

dataset = generate_dummy_dna_dataset()

# Split
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ----------------------------
# Tokenization using K-mers
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def kmer_tokenize(example):
    seq = example["sequence"]
    kmer_seq = [seq[i:i+KMER] for i in range(len(seq) - KMER + 1)]
    tokenized = tokenizer(" ".join(kmer_seq), truncation=True, max_length=MAX_LENGTH)
    return tokenized

train_dataset = train_dataset.map(kmer_tokenize, batched=False)
test_dataset = test_dataset.map(kmer_tokenize, batched=False)

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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
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
