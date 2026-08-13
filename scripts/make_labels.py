import json, sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.parser import parse_notam_to_dict

INPUT_PATH  = "data/raw/notams.json"
OUTPUT_PATH = "data/labeled/notams_labeled.csv"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        parsed = parse_notam_to_dict(item["text"])
        rows.append({
            "id": item["id"], "text": item["text"],
            "route_airports": ",".join(item.get("route_airports", [])),
            **parsed, "category": "", "severity": "",
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved label template -> {OUTPUT_PATH}")
    print("Categories: runway | taxiway | navaid | airspace | lighting | admin | other")
    print("Severity:   critical | advisory | informational")


if __name__ == "__main__":
    main()
