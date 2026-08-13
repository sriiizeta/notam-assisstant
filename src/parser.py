import re
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ParsedNOTAM:
    q_line: Optional[str]
    fir: Optional[str]
    q_code: Optional[str]
    traffic: Optional[str]
    purpose: Optional[str]
    scope: Optional[str]
    lower_limit: Optional[str]
    upper_limit: Optional[str]
    coords_radius: Optional[str]
    a_field: Optional[str]
    b_field: Optional[str]
    c_field: Optional[str]
    d_field: Optional[str]
    e_field: Optional[str]


FIELD_PATTERNS = {
    "Q": r"Q\)\s*(.*?)(?=\s+[ABCDEN]\)|$)",
    "A": r"A\)\s*(.*?)(?=\s+[BCDEN]\)|$)",
    "B": r"B\)\s*(.*?)(?=\s+[ACDEN]\)|$)",
    "C": r"C\)\s*(.*?)(?=\s+[ABDEN]\)|$)",
    "D": r"D\)\s*(.*?)(?=\s+[ABCEN]\)|$)",
    "E": r"E\)\s*(.*)$",
}

# A field marker must be preceded by whitespace or the start of the string --
# without this, a parenthetical abbreviation that happens to end in a field
# letter (e.g. "(TDD)", "(2.0NM SW BED)") gets mistaken for a "D)" marker.
# This bites the D field specifically because it's optional/rare, so there's
# often no genuine earlier "D)" to match first (see failure_analysis.md,
# Parser section).
FIELD_BOUNDARY = r"(?:(?<=\s)|(?<=^))"


# Multi-part NOTAMs (a single notice split across several messages) carry a
# "PART n OF m" continuation marker that the FAA feed jams inline right after
# the airport code in the A) field, e.g. "A) KZNY PART 1 OF 2". That marker
# isn't part of the location and shouldn't survive into a_field -- it produces
# junk like "At KZNY PART 1 OF 2" in summaries and breaks any downstream
# grouping/route-matching on the airport code. Strip it (only the A field can
# meaningfully contain it).
PART_MARKER = re.compile(r"\s*\bPART\s+\d+\s+OF\s+\d+\b", flags=re.IGNORECASE)


def extract_field(text: str, letter: str) -> Optional[str]:
    pattern = FIELD_BOUNDARY + FIELD_PATTERNS[letter]
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    value = m.group(1).strip()
    if letter == "A":
        value = PART_MARKER.sub("", value).strip()
    return value


def parse_q_field(q_text: Optional[str]) -> Dict[str, Optional[str]]:
    if not q_text:
        return {
            "fir": None, "q_code": None, "traffic": None,
            "purpose": None, "scope": None, "lower_limit": None,
            "upper_limit": None, "coords_radius": None,
        }
    parts = [p.strip() for p in q_text.split("/")]
    return {
        "fir":           parts[0] if len(parts) > 0 else None,
        "q_code":        parts[1] if len(parts) > 1 else None,
        "traffic":       parts[2] if len(parts) > 2 else None,
        "purpose":       parts[3] if len(parts) > 3 else None,
        "scope":         parts[4] if len(parts) > 4 else None,
        "lower_limit":   parts[5] if len(parts) > 5 else None,
        "upper_limit":   parts[6] if len(parts) > 6 else None,
        "coords_radius": parts[7] if len(parts) > 7 else None,
    }


def parse_notam(text: str) -> ParsedNOTAM:
    q_line  = extract_field(text, "Q")
    a_field = extract_field(text, "A")
    b_field = extract_field(text, "B")
    c_field = extract_field(text, "C")
    d_field = extract_field(text, "D")
    e_field = extract_field(text, "E")
    q = parse_q_field(q_line)
    return ParsedNOTAM(
        q_line=q_line, fir=q["fir"], q_code=q["q_code"],
        traffic=q["traffic"], purpose=q["purpose"], scope=q["scope"],
        lower_limit=q["lower_limit"], upper_limit=q["upper_limit"],
        coords_radius=q["coords_radius"],
        a_field=a_field, b_field=b_field, c_field=c_field,
        d_field=d_field, e_field=e_field,
    )


def parse_notam_to_dict(text: str) -> Dict[str, Any]:
    p = parse_notam(text)
    return {
        "q_line": p.q_line, "fir": p.fir, "q_code": p.q_code,
        "traffic": p.traffic, "purpose": p.purpose, "scope": p.scope,
        "lower_limit": p.lower_limit, "upper_limit": p.upper_limit,
        "coords_radius": p.coords_radius,
        "a_field": p.a_field, "b_field": p.b_field, "c_field": p.c_field,
        "d_field": p.d_field, "e_field": p.e_field,
    }
