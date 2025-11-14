from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# Load dataset
ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]

# Convert to DataFrame
records = [{"text": r["body"], "priority": r["priority"]} for r in ds if r["body"] and r["priority"]]
df = pd.DataFrame(records)

# Encode priority labels
label_map = {"low": 0, "medium": 1, "high": 2}
df = df[df["priority"].isin(label_map)]
df["label"] = df["priority"].map(label_map)

# Train/test split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

from datasets import Dataset
train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Model setup
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

args = TrainingArguments(
    output_dir="./model",
    eval_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("ticket_priority_model")
tokenizer.save_pretrained("ticket_priority_model")

print("✅ Model trained and saved.")
