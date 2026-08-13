import re
from typing import Dict, List

SEVERITY_WEIGHT = {"critical": 3.0, "advisory": 1.5, "informational": 1.0}
HIGH_IMPACT_CATEGORIES = {"runway", "taxiway", "navaid", "airspace"}

# Large enough that route relevance always dominates severity+category+time+
# keyword combined (max possible non-route score is 3.0+1.5+0.5+0.3=5.3) --
# see failure_analysis.md, Ranker Example 4: an off-route critical NOTAM
# used to be able to outrank an on-route advisory one (5.0 vs 4.5), which
# defeats the point of a route-specific ranker.
ROUTE_MATCH_BONUS = 6.0

# A restriction/degradation keyword in the E-field (CLSD, PPR, U/S, NOT STD)
# is a real relevance signal that category+severity alone can't capture --
# a routine chart-amendment NOTAM and a genuine closure can land in the same
# category+severity bucket and tie on score. This keeps ties from falling
# back to arbitrary raw-feed order. See failure_analysis.md, Ranker Example 3.
RESTRICTION_KEYWORDS = re.compile(r"\bCLSD\b|\bPPR\b|\bU/S\b|NOT STD")


def score_notam(notam: Dict, route_airports: List[str], category: str, severity: str) -> float:
    score = 0.0
    sev = (severity or "informational").lower().strip()
    score += SEVERITY_WEIGHT.get(sev, 1.0)

    # an A) field can name more than one location (e.g. a multi-FIR route
    # NOTAM "KZDV KZMP"), so match on set membership of the individual codes
    # rather than exact whole-string equality -- otherwise any multi-code
    # a_field could never be "on route"
    a_codes = {tok for tok in (notam.get("a_field") or "").upper().split()}
    route_upper = {x.upper().strip() for x in route_airports}
    if a_codes & route_upper:
        score += ROUTE_MATCH_BONUS

    cat = (category or "other").lower().strip()
    if cat in HIGH_IMPACT_CATEGORIES:
        score += 1.5

    if notam.get("b_field") and notam.get("c_field"):
        score += 0.5

    if RESTRICTION_KEYWORDS.search((notam.get("e_field") or "").upper()):
        score += 0.3

    return round(score, 3)


def rank_notams(notams: List[Dict], route_airports: List[str]) -> List[Dict]:
    for n in notams:
        n["relevance_score"] = score_notam(
            n, route_airports, n.get("pred_category", ""), n.get("pred_severity", "")
        )
    return sorted(notams, key=lambda x: x["relevance_score"], reverse=True)
