from src.rag import retrieve, format_context


def make_notam(nid, a_field, e_field, pred_category="taxiway", pred_severity="advisory"):
    return {
        "id": nid, "a_field": a_field, "e_field": e_field,
        "b_field": "2501010000", "c_field": "2601010000",
        "text": e_field, "pred_category": pred_category, "pred_severity": pred_severity,
    }


CORPUS = [
    make_notam("KOSH-1", "KOSH", "TWY C3 CLSD"),
    make_notam("KOSH-2", "KOSH", "RWY 18 SIGNS NOT STD"),
    make_notam("KJFK-1", "KJFK", "TWY A CLSD FOR MAINTENANCE"),
]


def test_retrieve_returns_top_k():
    results = retrieve("taxiway closed", CORPUS, ["KOSH"], top_k=2)
    assert len(results) == 2


def test_query_vocabulary_overlap_surfaces_matching_notam_first():
    results = retrieve("TWY CLSD", CORPUS, ["KOSH"], top_k=3)
    assert results[0]["id"] == "KOSH-1"


def test_empty_query_falls_back_to_rule_based_ranking():
    results = retrieve("", CORPUS, ["KOSH"], top_k=3)
    assert results[0]["a_field"] == "KOSH"


def test_format_context_includes_question_route_and_weather():
    weather = [{"icao": "KOSH", "metar": "METAR KOSH TEST", "taf": "TAF KOSH TEST"}]
    retrieved = retrieve("TWY CLSD", CORPUS, ["KOSH"], top_k=2)
    context = format_context("any taxiway closures?", ["KOSH"], weather, retrieved)
    assert "any taxiway closures?" in context
    assert "KOSH" in context
    assert "METAR KOSH TEST" in context
    assert "KOSH-1" in context
