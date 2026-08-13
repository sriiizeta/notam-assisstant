"""
RAG over NOTAMs + live weather (README Section 8 stretch goal).

Usage: python scripts/notam_qa.py "<question>" <ROUTE_AIRPORT,ROUTE_AIRPORT,...>
Example: python scripts/notam_qa.py "Is the ILS working at JFK?" KJFK,KBOS

This script does the retrieval half of RAG for real: it fetches live METAR/
TAF from aviationweather.gov's confirmed-working endpoints (src/weather.py)
and retrieves the most relevant NOTAMs for the question (src/rag.py). It
prints the assembled context and stops there -- the generation half needs
an LLM API call, and there's no API key available in this environment (see
README, Section 8). If ANTHROPIC_API_KEY is set, it also calls the API to
generate a real answer; otherwise it prints the context for a human (or a
separate LLM session) to answer from.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import models_exist, load_models, load_raw_notams, process_notams
from src.rag import retrieve, format_context
from src.weather import fetch_weather


def try_generate_answer(context: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(api_key=api_key)
    # Haiku is the right tier for this cheap, grounded extraction task; the id
    # is overridable via NOTAM_QA_MODEL so this doesn't rot as models roll.
    model = os.environ.get("NOTAM_QA_MODEL", "claude-haiku-4-5-20251001")
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "Answer the pilot's question using ONLY the NOTAM and weather "
                "context below. If the context doesn't support an answer, say so "
                "explicitly rather than guessing.\n\n" + context
            ),
        }],
    )
    return response.content[0].text


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/notam_qa.py \"<question>\" <ROUTE_AIRPORT,...>")
        sys.exit(1)

    query = sys.argv[1]
    route_airports = sys.argv[2].split(",")

    if not models_exist():
        print("Models not found. Run scripts/train_classifier.py first.")
        sys.exit(1)

    cat_model, sev_model = load_models()
    notams = process_notams(load_raw_notams(), cat_model, sev_model)
    weather = [fetch_weather(icao) for icao in route_airports]
    retrieved = retrieve(query, notams, route_airports, top_k=5)
    context = format_context(query, route_airports, weather, retrieved)

    print(context)
    print()

    answer = try_generate_answer(context)
    if answer:
        print("=== GENERATED ANSWER ===")
        print(answer)
    else:
        print("=== NO LLM API KEY AVAILABLE ===")
        print("Retrieval is real and complete (above). Generation needs an LLM")
        print("call this environment has no API key for -- feed the context above")
        print("to any LLM to complete the RAG loop. See README, Section 8.")


if __name__ == "__main__":
    main()
