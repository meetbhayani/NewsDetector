# ---- Training & Evaluation ----

import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# Load data
columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title',
           'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
           'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']

train_df = pd.read_csv("/home/bm-3/Desktop/WORK/fakenews-env/train.tsv", sep='\t', header=None, names=columns)
val_df = pd.read_csv("/home/bm-3/Desktop/WORK/fakenews-env/valid.tsv", sep='\t', header=None, names=columns)
test_df = pd.read_csv("/home/bm-3/Desktop/WORK/fakenews-env/test.tsv", sep='\t', header=None, names=columns)

def label_to_binary(label):
    return 0 if label in ['pants-fire', 'false', 'barely-true'] else 1

for df in [train_df, val_df, test_df]:
    df['statement'] = df['statement'].astype(str)
    df['binary_label'] = df['label'].apply(label_to_binary).astype(int)
    df.dropna(subset=['statement'], inplace=True)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class LiarDataset(TorchDataset):
    def __init__(self, df, tokenizer):
        encodings = tokenizer(list(df['statement']), padding="max_length", truncation=True, max_length=128)
        self.inputs = encodings
        self.labels = list(df['binary_label'])

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.inputs['input_ids'][idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.inputs['attention_mask'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = LiarDataset(train_df, tokenizer)
val_dataset = LiarDataset(val_df, tokenizer)
test_dataset = LiarDataset(test_df, tokenizer)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Evaluate
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Save model
model.save_pretrained("./saved_fake_news_model")
tokenizer.save_pretrained("./saved_fake_news_model")
