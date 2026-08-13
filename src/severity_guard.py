import re

CONDITIONAL = re.compile(r"\bEXC\b|\bPPR\b|WINGSPAN|ACFT WITH|WEF\b")
REAL_TRIGGER = re.compile(r"\bCLSD\b|\bU/S\b|\bUNUSABLE\b")
SIGN_LIGHT = re.compile(r"SIGN.*\bLGT\b.*U/S|\bLGT\b.*U/S.*SIGN")
DAY_OR_TIME_RANGE = re.compile(
    r"\bDLY\b|\bMON\b|\bTUE\b|\bWED\b|\bTHU\b|\bFRI\b|\bSAT\b|\bSUN\b|"
    r"\bSR-SS\b|\d{4}-\d{4}"
)


def guard_severity(predicted_severity: str, e_field: str, d_field: str = "") -> str:
    """
    Caps a model's `critical` prediction back down to `advisory` when the
    E-field (and D-field) text itself doesn't support it. Mirrors the same
    rule set used to produce the training labels in
    scripts/apply_labels.py's classify_severity, applied as a post-hoc
    guard on the model's raw prediction rather than only baked into
    training -- these are all fixed, checkable ICAO/FAA vocabulary
    patterns, which is exactly the case for a rule rather than a model per
    this project's core design principle. See failure_analysis.md,
    Classifier Example 1 and "Zero-shot LLM comparison," for the specific
    false positives each check fixes:

    1. Conditional closures ("CLSD EXC TAX", "CLSD ... PPR ...") -- the
       classifier over-weights the bare CLSD token and misses the
       exception clause in short texts.
    2. Route-qualification text that merely resembles a genuinely unusable
       navaid ("NA EXCEPT FOR AIRCRAFT EQUIPPED WITH SUITABLE RNAV SYSTEM
       WITH GPS") without actually containing CLSD/U-S/UNUSABLE anywhere.
    3. A schedule in the D) field (e.g. "MON-FRI 0400-0930") makes an
       otherwise-unconditional-looking E-field closure conditional.
    4. A sign's illumination being out ("... SIGN ... LGT U/S") is a
       degraded marking, not the same as the fixture itself being closed.
    """
    if predicted_severity != "critical":
        return predicted_severity

    e = (e_field or "").upper()
    d = (d_field or "").upper()

    if SIGN_LIGHT.search(e):
        return "advisory"
    if CONDITIONAL.search(e) or DAY_OR_TIME_RANGE.search(d):
        return "advisory"
    if not REAL_TRIGGER.search(e):
        return "advisory"
    return predicted_severity
