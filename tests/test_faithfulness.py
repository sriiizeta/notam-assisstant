from src.faithfulness import check_faithfulness
from src.summarizer import summarize_notam


def test_faithful_summary_from_real_summarizer_output():
    parsed = {
        "a_field": "KJFK", "b_field": "2506180334", "c_field": "2609052300",
        "d_field": None, "e_field": "TERMINAL 1 RAMP WIP CONST LGTD AND BARRICADED",
        "q_code": "QMNHW",
    }
    summary = summarize_notam(parsed, "other", "advisory")
    result = check_faithfulness(parsed, summary)
    assert result["faithful"] is True
    assert result["issues"] == []


def test_missing_start_time_detected():
    parsed = {"b_field": "2506180334", "c_field": "2609052300", "e_field": "RWY CLSD"}
    summary = "At KJFK. RWY CLSD. Valid from ???? to 2609052300."
    result = check_faithfulness(parsed, summary)
    assert "missing_start_time" in result["issues"]


def test_missing_end_time_detected():
    parsed = {"b_field": "2506180334", "c_field": "2609052300", "e_field": "RWY CLSD"}
    summary = "At KJFK. RWY CLSD. Valid from 2506180334 to ????."
    result = check_faithfulness(parsed, summary)
    assert "missing_end_time" in result["issues"]


def test_e_field_not_reflected_when_summary_omits_it_entirely():
    parsed = {"e_field": "TWY C3 CLSD FOR MAINTENANCE"}
    summary = "At KOSH. Category: taxiway. Severity: critical."
    result = check_faithfulness(parsed, summary)
    assert "e_field_not_reflected" in result["issues"]


def test_e_field_reflected_when_one_token_survives():
    parsed = {"e_field": "TWY C3 CLSD FOR MAINTENANCE"}
    summary = "Something mentions MAINTENANCE happening."
    result = check_faithfulness(parsed, summary)
    assert "e_field_not_reflected" not in result["issues"]


def test_hallucinated_term_flagged_when_not_in_source():
    parsed = {"e_field": "RWY 09 CLSD", "a_field": "KJFK", "q_code": "QMRLC"}
    summary = "Flight will be delayed due to RWY 09 CLSD at KJFK."
    result = check_faithfulness(parsed, summary)
    assert "hallucinated_term:delay" in result["issues"]


def test_no_hallucination_flag_when_trigger_word_is_in_source():
    parsed = {"e_field": "FLIGHT DIVERT PROCEDURE CHANGE", "a_field": "KJFK", "q_code": ""}
    summary = "Divert procedure changed at KJFK."
    result = check_faithfulness(parsed, summary)
    assert not any(i.startswith("hallucinated_term") for i in result["issues"])


def test_no_time_checks_when_b_or_c_field_missing():
    parsed = {"b_field": None, "c_field": None, "e_field": "RWY CLSD"}
    summary = "RWY CLSD, no dates given."
    result = check_faithfulness(parsed, summary)
    assert "missing_start_time" not in result["issues"]
    assert "missing_end_time" not in result["issues"]


def test_reformatted_date_with_separators_still_matches():
    # regression test for failure_analysis.md Summarizer Example 1: a future
    # summarizer reformatting "2506180334" with separators shouldn't trip
    # missing_start_time/missing_end_time as long as the digits are intact
    parsed = {"b_field": "2506180334", "c_field": "2609052300EST", "e_field": "RWY CLSD"}
    summary = "At KJFK. RWY CLSD. Valid from 25-06-18 03:34 to 26-09-05 23:00 EST."
    result = check_faithfulness(parsed, summary)
    assert "missing_start_time" not in result["issues"]
    assert "missing_end_time" not in result["issues"]


def test_permanent_special_value_does_not_false_positive():
    # b_field/c_field can be non-numeric special values (PERM/UFN/WIE);
    # these have no digits to look for and must not be flagged missing
    parsed = {"b_field": "PERM", "c_field": "UFN", "e_field": "RWY CLSD"}
    summary = "At KJFK. RWY CLSD. Valid from PERM to UFN."
    result = check_faithfulness(parsed, summary)
    assert "missing_start_time" not in result["issues"]
    assert "missing_end_time" not in result["issues"]


def test_heavily_paraphrased_e_field_flagged_below_coverage_threshold():
    # regression test for failure_analysis.md Summarizer Example 2: dropping
    # most of a multi-token E-field must be caught, not just total omission
    parsed = {"e_field": "TWY ALPHA BRAVO CHARLIE DELTA CLOSED FOR MAINTENANCE WORK"}
    summary = "Something mentions MAINTENANCE happening."
    result = check_faithfulness(parsed, summary)
    assert "e_field_not_reflected" in result["issues"]


def test_e_field_passes_at_exactly_half_coverage():
    parsed = {"e_field": "TWY C3 CLSD FOR MAINTENANCE"}
    summary = "Something mentions MAINTENANCE happening."
    result = check_faithfulness(parsed, summary)
    assert "e_field_not_reflected" not in result["issues"]
