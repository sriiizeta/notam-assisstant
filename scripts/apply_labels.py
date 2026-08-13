"""
Applies the labeling_guide.md rules to data/labeled/notams_labeled.csv.

This encodes the hand-labeling process as a deterministic rule set over the
E-field text (see docs/labeling_guide.md "Quick rules") rather than a second
ML model -- consistent with the project's core design principle: use rules
wherever there's a fixed, checkable vocabulary. NOTAM E-field abbreviations
for closures/U-S/PPR/EXC are a much smaller and more fixed vocabulary than
full free-text meaning, which is why this works for *labeling* even though
the same free text is hard to *classify* reliably without ML (see README).

Severity is derived jointly with category (a comm-radio or windcone going
U/S is not "critical" the way a closed runway is), which is itself the
reason a single keyword-matching rule can't do the classifier's job -- see
"why NOTAMs are hard" in the README.

Ambiguous cases were spot-checked by hand against the raw text; disagreements
found during that pass are documented in failure_analysis.md.
"""
import re
import pandas as pd

DATA_PATH = "data/labeled/notams_labeled.csv"

OBSTRUCTION_LGT = r"(TOWER LGT|CRANE LGT|OBSTRUCTION LGT|\(ASR \d|\(ASN )"


FIR_DESIGNATOR = re.compile(r"^K?Z[A-Z]{2}\b")


def classify_category(a_field: str, e: str) -> str:
    # a_field for FIR/ARTCC-level NOTAMs is the ARTCC code itself (e.g. "KZNY",
    # "ZNY", or "KZNY KZBW" for multi-FIR notices) rather than an airport ICAO
    # code -- these are exactly the "FIR-level notices" the airspace category
    # definition already names. This is a fixed, checkable designator pattern
    # (all US ARTCCs are Z + 2 letters), so it's a rule, not a model decision.
    if FIR_DESIGNATOR.match((a_field or "").strip()):
        return "airspace"

    if re.search(r"\bAP CLSD\b|\bAD CLSD\b", e):
        return "airspace"

    # a TWY-scoped sign/light/closure can reference a RWY only to describe
    # its location (e.g. "TWY J ... SIGNS FOR RWY 04R/22L FADED"); treat the
    # E-field's leading subject as authoritative rather than "RWY present
    # anywhere -> runway", which is exactly the scope-ambiguity failure mode
    # described in the README ("why NOTAMs are hard")
    if re.match(r"\s*TWY\b", e):
        return "taxiway"

    if re.search(r"\bTWY\b", e) and not re.search(r"\bRWY\b", e):
        return "taxiway"

    # obstruction/beacon/windcone lighting is checked ahead of the generic
    # RWY check: a windcone or obstruction light is an airfield indicator,
    # not runway infrastructure, even when its E-field names a runway for
    # location context (e.g. "AP WINDCONE FOR RWY 29 LGT U/S"). Originally
    # this check came after the RWY check, which mis-bucketed exactly that
    # pattern as `runway` and cascaded into a `critical` severity for what
    # is really a minor indicator outage -- see failure_analysis.md,
    # Classifier Example 4 (documented as an unfixed ground-truth bug,
    # fixed here).
    if re.search(OBSTRUCTION_LGT, e) or re.search(r"WINDCONE|SEGMENTED CIRCLE|BEACON", e):
        return "lighting"

    if re.search(r"\bTAR\b|\bSSR\b|\bVOR\b|\bNDB\b|\bGPS\b|\bRNAV\b|\bDME\b|"
                 r"TAKEOFF MINIMUMS|OBSTACLE DEPARTURE|\bAPCH\b|\bIAP\b|\bAMDT\b|"
                 r"\b(LOC|GS|GP|IM|MM|OM)\b.*U/S|U/S.*\b(LOC|GS|GP|IM|MM|OM)\b", e):
        return "navaid"

    if re.search(r"\bRWY\b", e):
        return "runway"

    if re.search(r"\b(FREQ|COM|CTAF|CONTACT|REMOTE TRANS/REC)\b", e):
        return "admin"

    return "other"


DAY_OR_TIME_RANGE = re.compile(
    r"\bDLY\b|\bMON\b|\bTUE\b|\bWED\b|\bTHU\b|\bFRI\b|\bSAT\b|\bSUN\b|"
    r"\bSR-SS\b|\d{4}-\d{4}"
)
SIGN_LIGHT = re.compile(r"SIGN.*\bLGT\b.*U/S|\bLGT\b.*U/S.*SIGN")


def classify_severity(category: str, e: str, d_field: str = "") -> str:
    d = (d_field or "").upper()

    # a D) field encoding a recurring schedule (daily/weekday window, e.g.
    # "MON-FRI 0400-0930") is itself a condition on the closure, the same
    # way an E-field "EXC"/"PPR" clause is -- the zero-shot LLM comparison's
    # disagreement on KHPN-07-096 ("RWY 16/34 CLSD" with D) MON-FRI
    # 0400-0930) surfaced this: the E-field alone reads as an unconditional
    # closure, but it's actually only closed during that scheduled window.
    # See failure_analysis.md, "Zero-shot LLM comparison."
    conditional = bool(re.search(r"\bEXC\b|\bPPR\b|WINGSPAN|ACFT WITH|WEF\b", e)) \
        or bool(DAY_OR_TIME_RANGE.search(d))

    # routine grounds-maintenance / no-data notices carry no operational
    # restriction at all -- this is the corpus's only real informational
    # example type (see failure_analysis.md: severity had zero informational
    # examples until this rule was added)
    if re.search(r"NOT REP\b|GRASS CUTTING|MOWING", e):
        return "informational"

    # obstruction/tower/crane/beacon lighting and minor airfield indicators
    # are hazard-marking / advisory notices, not closures -- advisory
    # regardless of whether the fixture itself is U/S
    if category == "lighting":
        return "advisory"

    # a sign's illumination being out ("... SIGN ... LGT U/S") is a degraded
    # marking, not the same as the fixture itself being closed/unserviceable
    # -- distinguishing this from a direct fixture U/S (PAPI/REIL/ILS
    # component/taxiway closure) is exactly what the zero-shot LLM
    # comparison's more conservative judgment calls flagged as missing from
    # this rule set. See failure_analysis.md, "Zero-shot LLM comparison."
    if SIGN_LIGHT.search(e):
        return "advisory"

    if re.search(r"\bCLSD\b", e):
        return "advisory" if conditional else "critical"

    if re.search(r"\bU/S\b|\bUNUSABLE\b", e):
        # U/S (and the FIR-level NOTAM feed's "UNUSABLE", e.g. a VOR radial
        # unusable) only rises to "critical" for categories where the
        # missing equipment actually removes a safety margin; a U/S comm
        # radio or admin fixture is degraded, not critical
        return "critical" if category in ("runway", "taxiway", "navaid", "airspace") else "advisory"

    if re.search(r"NOT STD|NOT STANDARD", e):
        return "advisory"

    if category == "admin":
        return "informational"

    if re.search(r"WIP CONST|BARRICADED", e):
        return "advisory"

    return "advisory"


def main():
    df = pd.read_csv(DATA_PATH).fillna("")
    df["category"] = df["category"].astype(str)
    df["severity"] = df["severity"].astype(str)

    e_upper = df["e_field"].apply(lambda x: (x or "").upper())
    df["category"] = [classify_category(a, e) for a, e in zip(df["a_field"], e_upper)]
    df["severity"] = [classify_severity(cat, e, d) for cat, e, d in zip(df["category"], e_upper, df["d_field"])]

    df.to_csv(DATA_PATH, index=False)
    print(df["category"].value_counts())
    print()
    print(df["severity"].value_counts())


if __name__ == "__main__":
    main()
