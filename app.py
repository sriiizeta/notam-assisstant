import streamlit as st
import json

from src.pipeline import load_models, process_notams
from src.summarizer import summarize_notam
from src.faithfulness import check_faithfulness
from src.ranker import rank_notams

st.title("NOTAM Analysis Pipeline")

uploaded = st.file_uploader("Upload notams.json", type=["json"])

if uploaded:
    raw = json.load(uploaded)
    cat_model, sev_model = load_models()

    route = st.text_input("Route airports (comma-separated)", "KJFK,KBOS")
    route_airports = [x.strip().upper() for x in route.split(",") if x.strip()]

    processed = process_notams(raw, cat_model, sev_model)
    for n in processed:
        summary = summarize_notam(n, n["pred_category"], n["pred_severity"])
        n["summary"] = summary
        n["faithful"] = check_faithfulness(n, summary)["faithful"]

    ranked = rank_notams(processed, route_airports)
    for n in ranked:
        st.subheader(f"{n['id']} | {n['pred_category']} | score={n['relevance_score']}")
        st.write(n["summary"])
        st.write("Faithful:", n["faithful"])
        st.code(n["text"])
