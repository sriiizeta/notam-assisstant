from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.ranker import rank_notams


def retrieve(query: str, notams: List[Dict], route_airports: List[str], top_k: int = 5) -> List[Dict]:
    """
    Retrieval for RAG over the NOTAM corpus: combines query-text similarity
    (TF-IDF cosine, so a question about "ILS" or "closed" pulls in NOTAMs
    using that vocabulary) with the same route-relevance ranking used
    elsewhere in the pipeline (src/ranker.py) -- a question about a route
    should still prioritize NOTAMs actually on that route over a
    lexically-similar one somewhere else.

    Deliberately TF-IDF, not an embeddings model: the corpus is small
    (hundreds of NOTAMs) and the vocabulary is a fixed, narrow ICAO/FAA
    abbreviation set, which is exactly the case where a heavier embedding
    model wouldn't earn its cost over a simple, fast, auditable baseline --
    the same "use the simplest thing that fits the vocabulary" principle
    this project applies everywhere else.
    """
    ranked = rank_notams([dict(n) for n in notams], route_airports)
    texts = [n.get("text", "") for n in ranked]

    if not query.strip() or not any(texts):
        return ranked[:top_k]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform(texts + [query])
    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    similarities = cosine_similarity(query_vec, doc_vecs)[0]

    max_relevance = max((n["relevance_score"] for n in ranked), default=1.0) or 1.0
    for n, sim in zip(ranked, similarities):
        normalized_relevance = n["relevance_score"] / max_relevance
        n["query_similarity"] = round(float(sim), 3)
        n["retrieval_score"] = round(0.6 * float(sim) + 0.4 * normalized_relevance, 3)

    return sorted(ranked, key=lambda n: n["retrieval_score"], reverse=True)[:top_k]


def format_context(query: str, route_airports: List[str], weather: List[Dict], retrieved: List[Dict]) -> str:
    """Assembles a plain-text context block ready to hand to an LLM's
    generation step. Building this is the retrieval half of RAG and is
    pure code; the generation half needs an actual LLM call, which this
    project's environment doesn't have an API key for -- see README,
    Section 8. This function stops exactly where that call would begin."""
    lines = [f"QUESTION: {query}", f"ROUTE: {', '.join(route_airports)}", ""]

    if weather:
        lines.append("WEATHER:")
        for w in weather:
            lines.append(f"  {w['icao']} METAR: {w['metar'] or '(unavailable)'}")
            lines.append(f"  {w['icao']} TAF: {w['taf'] or '(unavailable)'}")
        lines.append("")

    lines.append(f"RELEVANT NOTAMS (top {len(retrieved)}, retrieved for this question):")
    for n in retrieved:
        lines.append(
            f"  [{n['id']}] {n.get('pred_category', '?')}/{n.get('pred_severity', '?')} "
            f"(similarity={n.get('query_similarity', 0):.2f}): {n.get('e_field', '')}"
        )
    return "\n".join(lines)
