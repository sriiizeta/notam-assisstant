from src.ranker import score_notam, rank_notams


def make_notam(a_field="KJFK", b_field="2501010000", c_field="2601010000", e_field="RWY 09 CLSD"):
    return {"a_field": a_field, "b_field": b_field, "c_field": c_field, "e_field": e_field}


def test_severity_ordering_critical_beats_advisory_beats_informational():
    n = make_notam()
    critical = score_notam(n, ["KJFK"], "runway", "critical")
    advisory = score_notam(n, ["KJFK"], "runway", "advisory")
    informational = score_notam(n, ["KJFK"], "runway", "informational")
    assert critical > advisory > informational


def test_on_route_always_outranks_off_route_regardless_of_severity():
    # regression test for failure_analysis.md Ranker Example 4: an off-route
    # critical NOTAM used to be able to outrank an on-route advisory one
    off_route_critical = score_notam(
        make_notam(a_field="KBOS"), ["KJFK"], "navaid", "critical"
    )
    on_route_informational = score_notam(
        make_notam(a_field="KJFK", e_field="AP SFC COND NOT REP"),
        ["KJFK"], "other", "informational",
    )
    assert on_route_informational > off_route_critical


def test_high_impact_category_scores_higher_than_low_impact_at_same_severity():
    n = make_notam()
    runway_score = score_notam(n, ["KJFK"], "runway", "advisory")
    lighting_score = score_notam(n, ["KJFK"], "lighting", "advisory")
    assert runway_score > lighting_score


def test_restriction_keyword_tiebreaker_beats_routine_text_at_same_bucket():
    # regression test for failure_analysis.md Ranker Example 3
    restrictive = make_notam(e_field="TWY C3 CLSD")
    routine = make_notam(e_field="ILS OR LOC RWY 22R, AMDT 4...")
    restrictive_score = score_notam(restrictive, ["KJFK"], "taxiway", "advisory")
    routine_score = score_notam(routine, ["KJFK"], "taxiway", "advisory")
    assert restrictive_score > routine_score


def test_time_window_adds_bonus():
    with_window = make_notam(b_field="2501010000", c_field="2601010000")
    without_window = make_notam(b_field=None, c_field=None)
    assert score_notam(with_window, ["KJFK"], "runway", "advisory") > \
        score_notam(without_window, ["KJFK"], "runway", "advisory")


def test_route_match_is_case_and_whitespace_insensitive():
    n = make_notam(a_field="kjfk ")
    score = score_notam(n, [" KJFK"], "runway", "advisory")
    off_route = score_notam(n, ["KBOS"], "runway", "advisory")
    assert score > off_route


def test_multi_code_a_field_matches_on_any_member():
    # a multi-FIR/multi-airport A field must count as on-route if ANY of its
    # codes is on the route (previously exact whole-string equality meant a
    # multi-code a_field could never match)
    n = make_notam(a_field="KZDV KZMP")
    on_route = score_notam(n, ["KZMP"], "airspace", "advisory")
    off_route = score_notam(n, ["KJFK"], "airspace", "advisory")
    assert on_route > off_route


def test_rank_notams_sorts_descending_by_score():
    notams = [
        {"id": "low", "pred_category": "other", "pred_severity": "informational", **make_notam(a_field="KBOS")},
        {"id": "high", "pred_category": "runway", "pred_severity": "critical", **make_notam(a_field="KJFK")},
        {"id": "mid", "pred_category": "taxiway", "pred_severity": "advisory", **make_notam(a_field="KJFK")},
    ]
    ranked = rank_notams(notams, ["KJFK"])
    assert [n["id"] for n in ranked] == ["high", "mid", "low"]
    assert ranked[0]["relevance_score"] >= ranked[1]["relevance_score"] >= ranked[2]["relevance_score"]
