import re
from typing import Dict, List

MIN_E_FIELD_TOKEN_COVERAGE = 0.5


def _digits_only(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def check_faithfulness(parsed: Dict[str, str], summary: str) -> Dict[str, object]:
    """
    NOTE: checking whether a_field appears in the summary is a trivial pass
    (the summarizer always inserts it), so that check is deliberately excluded.
    Instead this checks time-window consistency, E-field token coverage, and a
    hallucination trigger list -- these can actually catch real problems.
    """
    summary_lower = summary.lower()
    issues: List[str] = []

    if parsed.get("b_field") and parsed.get("c_field"):
        # Compare on digits only, not the raw string, so this survives a
        # future summarizer reformatting "2506180334" into e.g.
        # "2506-18-0334" or a different date rendering that preserves the
        # same underlying digits in the same order (see failure_analysis.md,
        # Summarizer Example 1). b_field/c_field can also be non-numeric
        # special values ("PERM", "UFN", "WIE"), which yield an empty digit
        # string and are skipped rather than producing a false positive.
        b_digits = _digits_only(parsed["b_field"])
        c_digits = _digits_only(parsed["c_field"])
        summary_digits = _digits_only(summary)
        if b_digits and b_digits not in summary_digits:
            issues.append("missing_start_time")
        if c_digits and c_digits not in summary_digits:
            issues.append("missing_end_time")

    if parsed.get("e_field"):
        tokens = parsed["e_field"].split()
        key_tokens = [t.lower() for t in tokens if len(t) > 3]
        if key_tokens:
            # Require a minimum *fraction* of key tokens to survive, not
            # just one -- a single surviving token is too weak a bar (a
            # summarizer could drop 90% of a long E-field and still pass).
            # This only matters once the summarizer stops being a pure
            # field-echo template; see failure_analysis.md, Summarizer
            # Example 2.
            matched = sum(1 for tok in key_tokens if tok in summary_lower)
            coverage = matched / len(key_tokens)
            if coverage < MIN_E_FIELD_TOKEN_COVERAGE:
                issues.append("e_field_not_reflected")

    hallucination_triggers = ["delay", "cancel", "divert", "emergency"]
    all_source_text = " ".join([
        (parsed.get("e_field") or ""),
        (parsed.get("a_field") or ""),
        (parsed.get("q_code") or ""),
    ]).lower()
    for trigger in hallucination_triggers:
        if trigger in summary_lower and trigger not in all_source_text:
            issues.append(f"hallucinated_term:{trigger}")

    return {"faithful": len(issues) == 0, "issues": issues}
