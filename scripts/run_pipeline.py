import sys, os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import models_exist, load_models, load_raw_notams, process_notams
from src.summarizer import summarize_notam
from src.faithfulness import check_faithfulness
from src.ranker import rank_notams


def main():
    if not models_exist():
        print("Models not found. Run scripts/train_classifier.py first.")
        sys.exit(1)

    cat_model, sev_model = load_models()
    processed = process_notams(load_raw_notams(), cat_model, sev_model)

    route_airports = sys.argv[1].split(",") if len(sys.argv) > 1 else ["VIDP", "VABB"]

    for n in processed:
        summary = summarize_notam(n, n["pred_category"], n["pred_severity"])
        faith = check_faithfulness(n, summary)
        n["summary"] = summary
        n["faithful"] = faith["faithful"]
        n["faithfulness_issues"] = ", ".join(faith["issues"]) if faith["issues"] else "none"

    ranked = rank_notams(processed, route_airports)
    df = pd.DataFrame(ranked)

    print(f"\n=== RANKED NOTAMS for route {route_airports} (highest relevance first) ===\n")
    for _, row in df.iterrows():
        flag = "OK" if row["faithful"] else "FAITHFULNESS ISSUE"
        print(f"[{row['relevance_score']:5.2f}] {row['id']} | {row['pred_category']:10} | {row['pred_severity']:13} | {flag}")
        print(f"        {row['summary']}")
        if row["faithfulness_issues"] != "none":
            print(f"        Issues: {row['faithfulness_issues']}")
        print()


if __name__ == "__main__":
    main()
