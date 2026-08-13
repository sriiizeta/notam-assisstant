import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.summarizer import summarize_notam
from src.faithfulness import check_faithfulness

DATA_PATH = "data/labeled/notams_labeled.csv"
FIELDS = ["a_field", "b_field", "c_field", "d_field", "e_field", "q_code"]


def main():
    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")

    n_faithful = 0
    issue_counts = {}
    for _, row in df.iterrows():
        parsed = {k: (row[k] if row[k] else None) for k in FIELDS}
        summary = summarize_notam(parsed, row["category"], row["severity"])
        result = check_faithfulness(parsed, summary)
        if result["faithful"]:
            n_faithful += 1
        else:
            for issue in result["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

    total = len(df)
    print(f"Faithfulness rate: {n_faithful}/{total} = {100 * n_faithful / total:.1f}%")
    print("Issue breakdown:", issue_counts if issue_counts else "none")


if __name__ == "__main__":
    main()
