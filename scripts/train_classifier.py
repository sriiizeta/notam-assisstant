import sys, os, time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH     = "data/labeled/notams_labeled.csv"
CAT_MODEL_OUT = "model_category.joblib"
SEV_MODEL_OUT = "model_severity.joblib"


def build_pipeline():
    # class_weight="balanced" matters a lot here: the corpus is dominated by
    # `navaid`/`advisory` (routine IAP-amendment text), which without
    # reweighting pulls the decision boundary toward the majority class and
    # tanks recall on `critical` -- see failure_analysis.md for the measured
    # before/after (critical recall 0.38 -> higher with this weighting).
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def train_and_report(X, y, label_name, model_out):
    print(f"\n{'='*50}\nTraining: {label_name}\n{'='*50}")
    print(f"Total samples: {len(X)} | Classes: {sorted(set(y))}")

    if len(X) < 10:
        print("Dataset too small for a real split -- training on full set (get more labeled data).")
        model = build_pipeline()
        model.fit(X, y)
        preds = model.predict(X)
        print(classification_report(y, preds, zero_division=0))
    else:
        # stratify needs >=2 members per class (one for train, one for
        # test); real hand-labeled data has singleton classes (e.g. only
        # one "admin" example), so fall back to a plain split when that
        # happens instead of letting train_test_split raise
        from collections import Counter
        class_counts = Counter(y)
        can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
        if not can_stratify:
            print("Stratified split not possible (a class has <2 members) -- using a plain split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if can_stratify else None
        )
        model = build_pipeline()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(classification_report(y_test, preds, zero_division=0))

        labels = sorted(set(y_test) | set(preds))
        cm = confusion_matrix(y_test, preds, labels=labels)
        print("Confusion matrix (rows=true, cols=pred):")
        print("labels:", labels)
        print(cm)

        # dump misclassified examples for failure analysis
        mis_path = f"misclassified_{label_name.lower()}.csv"
        mis_df = pd.DataFrame({"text": X_test, "true": y_test, "pred": preds})
        mis_df = mis_df[mis_df["true"] != mis_df["pred"]]
        mis_df.to_csv(mis_path, index=False)
        print(f"Misclassified examples saved -> {mis_path} ({len(mis_df)} rows)")

    start = time.perf_counter()
    for _ in range(100):
        model.predict([X[0]])
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    print(f"Avg inference time: {elapsed_ms:.2f} ms")

    joblib.dump(model, model_out)
    print(f"Saved -> {model_out}")
    return model


def main():
    df = pd.read_csv(DATA_PATH).fillna("")
    X = df["text"].tolist()
    train_and_report(X, df["category"].tolist(), "CATEGORY", CAT_MODEL_OUT)
    train_and_report(X, df["severity"].tolist(),  "SEVERITY", SEV_MODEL_OUT)


if __name__ == "__main__":
    main()
