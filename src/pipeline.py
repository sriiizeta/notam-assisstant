"""
Shared pipeline glue: load the trained models once, and turn raw NOTAM
records into fully-processed dicts (parsed fields + guarded category/severity
predictions).

Extracted because run_pipeline.py, evaluate_ranker.py, notam_qa.py and app.py
were each carrying their own copy of the same parse -> predict -> guard loop,
which is exactly the kind of duplication that drifts out of sync (e.g. one
caller forgetting to apply guard_severity, or passing d_field). Everything
downstream of prediction (summarize, faithfulness, rank, retrieve) stays in
the individual entry points -- this only owns the shared prefix.
"""
import json
import os
from typing import List, Dict, Tuple

import joblib

from src.parser import parse_notam_to_dict
from src.severity_guard import guard_severity

NOTAM_PATH = "data/raw/notams.json"
CAT_MODEL_PATH = "model_category.joblib"
SEV_MODEL_PATH = "model_severity.joblib"


def models_exist() -> bool:
    return os.path.exists(CAT_MODEL_PATH) and os.path.exists(SEV_MODEL_PATH)


def load_models() -> Tuple[object, object]:
    return joblib.load(CAT_MODEL_PATH), joblib.load(SEV_MODEL_PATH)


def load_raw_notams(path: str = NOTAM_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_notams(raw: List[Dict], cat_model, sev_model) -> List[Dict]:
    """Parse + classify each raw NOTAM. `pred_severity` has the rule-based
    guard applied, matching how severity is used everywhere in the pipeline."""
    processed = []
    for item in raw:
        text = item["text"]
        parsed = parse_notam_to_dict(text)
        pred_category = cat_model.predict([text])[0]
        pred_severity = guard_severity(
            sev_model.predict([text])[0], parsed["e_field"], parsed["d_field"]
        )
        processed.append({
            "id": item["id"],
            "text": text,
            "route_airports": item.get("route_airports", []),
            **parsed,
            "pred_category": pred_category,
            "pred_severity": pred_severity,
        })
    return processed
