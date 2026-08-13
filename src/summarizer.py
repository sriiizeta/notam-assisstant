from typing import Dict


def summarize_notam(parsed: Dict[str, str], category: str, severity: str) -> str:
    """Intentionally simple: only uses fields already extracted by the parser.
    This constraint is what makes the faithfulness checker meaningful."""
    parts = []
    if parsed.get("a_field"):
        parts.append(f"At {parsed['a_field']}")
    if parsed.get("e_field"):
        parts.append(parsed["e_field"].rstrip("."))
    if parsed.get("b_field") and parsed.get("c_field"):
        parts.append(f"Valid from {parsed['b_field']} to {parsed['c_field']}")
    if category:
        parts.append(f"Category: {category}")
    if severity:
        parts.append(f"Severity: {severity}")
    return ". ".join(parts) + "."
