"""
DistilBERT fine-tune comparison (stretch goal from the project brief).

Uses the exact same train/test split logic as scripts/train_classifier.py
(test_size=0.2, random_state=42, plain-split fallback when a class has <2
members) so the F1/inference-time numbers are directly comparable to the
TF-IDF + LogisticRegression baseline.
"""
import sys, os, time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "data/labeled/notams_labeled.csv"
MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 6
BATCH_SIZE = 16
MAX_LEN = 128


class NotamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        enc = tokenizer(list(texts), padding=True, truncation=True,
                         max_length=MAX_LEN, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def split(X, y):
    class_counts = Counter(y)
    can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
    return train_test_split(X, y, test_size=0.2, random_state=42,
                             stratify=y if can_stratify else None)


def train_one_task(X, y, label_name):
    print(f"\n{'='*50}\nDistilBERT: {label_name}\n{'='*50}")
    classes = sorted(set(y))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    y_ids = [label2id[label] for label in y]

    X_train, X_test, y_train, y_test = split(X, y_ids)
    print(f"train={len(X_train)} test={len(X_test)} classes={classes}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes)
    ).to(DEVICE)

    train_ds = NotamDataset(X_train, y_train, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        print(f"  epoch {epoch + 1}/{EPOCHS} loss={total_loss / len(train_loader):.4f}")

    model.eval()
    test_enc = tokenizer(list(X_test), padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=test_enc["input_ids"].to(DEVICE),
            attention_mask=test_enc["attention_mask"].to(DEVICE),
        ).logits
    preds = logits.argmax(dim=-1).cpu().numpy()

    y_test_labels = [id2label[i] for i in y_test]
    pred_labels = [id2label[i] for i in preds]
    print(classification_report(y_test_labels, pred_labels, zero_division=0))

    # single-example inference time, matching scripts/train_classifier.py's methodology
    single = tokenizer([X_test[0]], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(input_ids=single["input_ids"].to(DEVICE),
                  attention_mask=single["attention_mask"].to(DEVICE))
        start = time.perf_counter()
        for _ in range(100):
            model(input_ids=single["input_ids"].to(DEVICE),
                  attention_mask=single["attention_mask"].to(DEVICE))
        elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    print(f"Avg inference time ({DEVICE}): {elapsed_ms:.2f} ms")

    report = classification_report(y_test_labels, pred_labels, zero_division=0, output_dict=True)
    return {
        "label_name": label_name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "inference_ms": elapsed_ms,
        "device": str(DEVICE),
    }


def main():
    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
    X = df["text"].tolist()

    results = []
    results.append(train_one_task(X, df["category"].tolist(), "CATEGORY"))
    results.append(train_one_task(X, df["severity"].tolist(), "SEVERITY"))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in results:
        print(f"{r['label_name']:10} accuracy={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f} "
              f"inference={r['inference_ms']:.2f}ms ({r['device']})")

    import json
    with open("distilbert_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
