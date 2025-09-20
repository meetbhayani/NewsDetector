import os
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer, TrainingArguments
)

MODEL_DIR = "./saved_fake_news_model"

# ✦ 1) Load existing model if it’s there ----------------------------
if os.path.isdir(MODEL_DIR):
    print(f"✅  Found {MODEL_DIR}.  Loading without retraining…")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)

# ✦ 2) Otherwise, train and then save -------------------------------
else:
    print("🔄  No saved model found – starting training…")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=2
                )

    # ─── your dataset code exactly as before ───
    # train_dataset, val_dataset = …

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        do_train=True, do_eval=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    # ── SAVE ──
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    trainer.save_state()               # optional: optimizer/scheduler state
    print(f"💾  Model saved to {MODEL_DIR}")
