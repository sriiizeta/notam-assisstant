"""
Cross-validated evaluation + rule-vs-ML head-to-head.

Why this exists: scripts/train_classifier.py reports metrics on a single 20%
train/test split. With only ~346 NOTAMs (and ~100 `critical` severity
examples), a single split is noisy -- 5-fold CV shows severity macro-F1
swinging from ~0.62 to ~0.92 across folds, i.e. the fold-to-fold variance is
larger than most of the single-split deltas the docs used to treat as
findings. This script reports mean +/- std across folds instead, which is the
honest way to state how well the pipeline actually classifies.

It also runs the zero-ML Q-code lookup baseline (src/qcode_baseline.py) on the
*same folds*, so the README's "rule vs ML tradeoffs, measured" claim is backed
by an actual head-to-head rather than only ML-vs-ML comparisons.

Rare classes (fewer than N_SPLITS members) are excluded from the CV with a
printed note, because stratified k-fold needs at least one member per class
per fold; their counts are too small to produce a meaningful per-class metric
anyway (see failure_analysis.md on the `admin` singleton problem).
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.qcode_baseline import QCodeLookupClassifier
from src.severity_guard import guard_severity

DATA_PATH = "data/labeled/notams_labeled.csv"
RESULTS_PATH = "cross_validation_results.json"
N_SPLITS = 5


def build_tfidf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def summarize(scores):
    arr = np.array(scores)
    return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3),
            "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3)}


def cv_for_task(df, task):
    y_all = df[task].to_numpy()
    counts = Counter(y_all)
    keep = {c for c, n in counts.items() if n >= N_SPLITS}
    dropped = sorted(c for c in counts if c not in keep)
    mask = np.array([label in keep for label in y_all])

    text = df["text"].to_numpy()[mask]
    e_field = df["e_field"].fillna("").to_numpy()[mask]
    d_field = df["d_field"].fillna("").to_numpy()[mask]
    q_code = df["q_code"].fillna("").to_numpy()[mask]
    y = y_all[mask]

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    tfidf_f1, tfidf_acc = [], []
    guard_f1, guard_acc = [], []
    rule_f1, rule_acc = [], []

    for train_idx, test_idx in skf.split(text, y):
        y_tr, y_te = y[train_idx], y[test_idx]

        model = build_tfidf().fit(text[train_idx], y_tr)
        preds = model.predict(text[test_idx])
        tfidf_f1.append(f1_score(y_te, preds, average="macro", zero_division=0))
        tfidf_acc.append(accuracy_score(y_te, preds))

        if task == "severity":
            guarded = [guard_severity(p, e, d)
                       for p, e, d in zip(preds, e_field[test_idx], d_field[test_idx])]
            guard_f1.append(f1_score(y_te, guarded, average="macro", zero_division=0))
            guard_acc.append(accuracy_score(y_te, guarded))

        rule_key = "subject" if task == "category" else "condition"
        rule = QCodeLookupClassifier(key=rule_key).fit(q_code[train_idx], y_tr)
        rule_preds = rule.predict(q_code[test_idx])
        rule_f1.append(f1_score(y_te, rule_preds, average="macro", zero_division=0))
        rule_acc.append(accuracy_score(y_te, rule_preds))

    result = {
        "task": task,
        "n_splits": N_SPLITS,
        "n_evaluated": int(mask.sum()),
        "dropped_rare_classes": dropped,
        "tfidf_ml": {"macro_f1": summarize(tfidf_f1), "accuracy": summarize(tfidf_acc)},
        "qcode_rule_baseline": {"macro_f1": summarize(rule_f1), "accuracy": summarize(rule_acc)},
    }
    if task == "severity":
        result["tfidf_ml_plus_guard"] = {"macro_f1": summarize(guard_f1),
                                         "accuracy": summarize(guard_acc)}
    return result


def print_result(r):
    print(f"\n{'='*66}\n{r['task'].upper()}  ({r['n_evaluated']} NOTAMs, "
          f"{r['n_splits']}-fold CV)")
    if r["dropped_rare_classes"]:
        print(f"  excluded rare classes (<{r['n_splits']} members): {r['dropped_rare_classes']}")
    print(f"{'='*66}")
    print(f"{'model':30} {'macro-F1 (mean+/-std)':24} {'range':16}")
    print("-" * 66)

    def row(name, block):
        f1 = block["macro_f1"]
        print(f"{name:30} {f1['mean']:.3f} +/- {f1['std']:<14.3f} "
              f"{f1['min']:.3f}-{f1['max']:.3f}")

    row("Q-code lookup (zero ML)", r["qcode_rule_baseline"])
    row("TF-IDF + LR", r["tfidf_ml"])
    if "tfidf_ml_plus_guard" in r:
        row("TF-IDF + LR + severity guard", r["tfidf_ml_plus_guard"])


def main():
    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
    results = [cv_for_task(df, "category"), cv_for_task(df, "severity")]
    for r in results:
        print_result(r)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
