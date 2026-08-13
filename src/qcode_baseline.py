"""
A zero-ML rule baseline for the classifier, built on the NOTAM Q-code.

The ICAO Q-code (field Q, the 5-letter token after the FIR, e.g. `QMRLC`)
encodes the subject and condition of a NOTAM in a fixed, standardised
vocabulary (ICAO Annex 15 / Doc 8126 "NOTAM Selection Criteria"):

    Q + <subject: letters 2-3> + <condition: letters 4-5>

    subject examples: MR=runway  MX=taxiway  MM=movement area
                      IC=ILS  PI=instrument approach proc  LP=PAPI
                      OL=obstacle light  AR=ATS route  FA=aerodrome
    condition examples: LC=closed  AS=unserviceable  CH=changed
                        XX=plain-language (no standard condition)

Because that vocabulary is fixed, a NOTAM's category (and, more weakly, its
severity) is largely recoverable from the Q-code with no machine learning at
all -- which is exactly the "use a rule where the vocabulary is fixed"
principle this project is built on. This module makes that rule concrete so
it can be measured head-to-head against the TF-IDF classifier (see
scripts/cross_validate.py). The honest result of that comparison is in the
README: category is *mostly* a lookup, so ML only buys a few points there;
severity is where free-text ML genuinely earns its place, because the
condition sub-code can't see the EXC/PPR/schedule conditionality that
separates `critical` from `advisory`.

Rather than hard-code an ICAO decode table (which could be subtly wrong and
wouldn't respect train/test discipline), this learns a majority-vote lookup
from the training split only: for each Q-code group seen in training, predict
that group's most common label; fall back to the global majority for unseen
groups. It's still a "rule" in the sense that there are no learned weights and
the decision is a transparent table lookup -- just a data-derived one.
"""
from collections import Counter, defaultdict
from typing import List, Sequence


def qcode_subject(q_code: str) -> str:
    """Letters 2-3 of the Q-code (the subject group), or '' if unavailable."""
    q = (q_code or "").strip().upper()
    return q[1:3] if len(q) >= 3 else ""


def qcode_condition(q_code: str) -> str:
    """Letters 4-5 of the Q-code (the condition group), or '' if unavailable."""
    q = (q_code or "").strip().upper()
    return q[3:5] if len(q) >= 5 else ""


class QCodeLookupClassifier:
    """A transparent majority-vote lookup keyed on part of the Q-code.

    sklearn-style fit/predict so it drops straight into the same
    cross-validation harness as the TF-IDF pipeline. `key` selects which
    part of the Q-code to key on: "subject" (letters 2-3, best for category)
    or "condition" (letters 4-5, the natural rule signal for severity).
    """

    def __init__(self, key: str = "subject"):
        if key not in ("subject", "condition"):
            raise ValueError("key must be 'subject' or 'condition'")
        self._extract = qcode_subject if key == "subject" else qcode_condition
        self.table_ = {}
        self.fallback_ = None

    def fit(self, q_codes: Sequence[str], y: Sequence[str]) -> "QCodeLookupClassifier":
        votes = defaultdict(Counter)
        for q, label in zip(q_codes, y):
            votes[self._extract(q)][label] += 1
        self.table_ = {k: c.most_common(1)[0][0] for k, c in votes.items()}
        self.fallback_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, q_codes: Sequence[str]) -> List[str]:
        return [self.table_.get(self._extract(q), self.fallback_) for q in q_codes]
