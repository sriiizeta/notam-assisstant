# NOTAM Labeling Guide

Every NOTAM in `data/labeled/notams_labeled.csv` is labeled on two independent axes:

## category

| Value | Meaning |
|---|---|
| `runway` | Runway *physical infrastructure*: surface, markings, signs, and lighting fixtures (PAPI/REIL/threshold/runway-end-ID lights), plus takeoff/landing restrictions. **Not** ILS/approach-procedure text even when it names a runway — see `navaid`. |
| `taxiway` | Taxiway closures, taxiway lighting, taxiway signage |
| `navaid` | VOR/NDB/DME/GPS/RNAV/ILS *procedures and equipment* — approach-procedure-amendment text (`AMDT`, `APCH`, `IAP`), takeoff-minimums/ODP text, and ILS-component (`LOC`/`GS`/`IM`/`MM`/`OM`) unserviceability. This applies even when the text names a specific runway (e.g. `ILS OR LOC RWY 20R, AMDT 13...`), because the NOTAM's actual subject is the navigation procedure, not the runway's physical infrastructure. |
| `airspace` | Airspace restrictions, TFRs, obstacles (cranes, towers), FIR-level notices, whole-aerodrome closures. Detected structurally: any NOTAM whose `A)` field is an ARTCC/FIR designator (`Z` + 2 letters, e.g. `ZNY`, `KZBW`) rather than an airport ICAO code is `airspace` regardless of E-field content — see "FIR-level search" below. |
| `lighting` | Airport lighting not scoped to a runway or taxiway (e.g. beacon, obstruction lights not on a runway/taxiway) |
| `admin` | Frequency changes, paperwork, PPR contact numbers, administrative procedure text |
| `other` | Anything that doesn't fit the above |

**Note on this distinction**: an earlier version of this table said "ILS/localizer
glideslope tied to a specific runway" belonged under `runway`, which
contradicted the actual rule implementation in `scripts/apply_labels.py`
(which checks for `AMDT`/`APCH`/`IAP` and routes that text to `navaid`
*before* checking for `RWY`). A zero-shot LLM given the old wording applied
it literally and disagreed with the ground truth on exactly this pattern
(see `failure_analysis.md`, "Zero-shot LLM comparison"). The table above is
the corrected, implementation-consistent version.

## severity

| Value | Meaning |
|---|---|
| `critical` | Unconditional closure of a runway/taxiway, a navaid/ILS component fully unserviceable (`U/S`) with no workaround, or (for `airspace`) a route/radial marked `UNUSABLE`. **Not** conditional on a schedule in the `D)` field, and **not** a sign's illumination being out — see the "Second labeling pass" note below. |
| `advisory` | Partial/conditional restriction — closed except under some condition (PPR, time window, wingspan limit, `D)`-field schedule), degraded (not-to-standard) equipment, obstacles, or a sign/marking's light being unserviceable while the fixture itself remains usable |
| `informational` | Admin/paperwork, frequency changes, non-operational notices |

## Quick rules (applied consistently across the corpus)

- Runway/taxiway `CLSD` with **no exception clause** -> `critical`.
- Runway/taxiway `CLSD EXC ...`, `PPR`, or conditioned on aircraft specs (wingspan, weight) -> `advisory` — the closure is not absolute, so blanket "CLSD = critical" is wrong (see project README, "why NOTAMs are hard").
- ILS component (`IM`, `LOC`, `GS`) unserviceable (`U/S`) -> `critical`, category `navaid` if not scoped to lights, else `runway` if it's an approach-lighting system (PAPI/REIL/runway-end lights).
- `NOT STD` (not standard) on lights/markings/signs -> `advisory` (degraded, not closed).
- Obstacles (crane/tower lit/flagged) -> `airspace`, `advisory` (they are a hazard to avoid, not a closure).
- Takeoff minimums / obstacle departure procedure text -> `navaid`, `advisory`.
- Frequency, contact numbers, procedural admin text with no physical restriction -> `admin`, `informational`.
- When in doubt, prefer the more specific category: `runway` > `taxiway` > `navaid` > `airspace` > `lighting` > `admin` > `other`.
- Routine grounds-maintenance / no-data notices (`AP SFC COND NOT REP`, `GRASS CUTTING`, `MOWING`) carry no operational restriction at all -> `informational`.

## FIR-level search fixed the `airspace` gap (corrected from an earlier wrong assumption)

An earlier version of this doc claimed the `airspace` gap (1 example even
after expanding to 18 airports) was structurally unfixable, because
`designatorsForLocation` supposedly only returns *aerodrome*-scoped NOTAMs
for a single airport ICAO code. That assumption was never actually tested
and turned out to be wrong: the same search parameter also accepts a bare
ARTCC/FIR designator (e.g. `ZNY`, `ZBW`) and returns genuinely different,
area-wide route/airspace NOTAMs — TFR-adjacent route restrictions, VOR
radials marked `UNUSABLE`, oceanic clearance procedures, etc. Pulling from 6
FIRs (`ZNY`, `ZBW`, `ZAU`, `ZTL`, `ZME`, `ZDV` — matching the existing
airports' home ARTCCs, see `scripts/fetch_notams.py`'s `FIRS` list) took
`airspace` from 1 example to 67. A small number of FIR-level records
(0-4 per FIR pulled) render as broken template output on the FAA's side
(literal `undefined` tokens in the text) and are filtered out at fetch time
— a real upstream data-quality issue, not something to parse around.

`admin` is still stuck at 2 examples; that one really does look like a
genuine gap in this feed (frequency-change/comm-equipment notices are just
rare in both the aerodrome and FIR searches tried so far) rather than a
wrong assumption about the search API.

## A real parser bug this labeling pass surfaced

Testing the parser against the FIR-sourced records found that `D)` field
extraction had a genuine bug: any parenthetical abbreviation ending in the
letter D followed by `)` (e.g. `THEDFORD (TDD)`, `(2.0NM SW BED)`) was
mistaken for a `D)` field marker, because the regex had no requirement that
a field marker be preceded by whitespace. This specifically affects `D)`
(rarely present, so there's often no genuine earlier match to win the
leftmost-match race) rather than `A)`/`B)`/`C)`/`E)` (always present, so the
real marker is always found first). Fixed in `src/parser.py` by requiring a
whitespace-or-start-of-string boundary before every field letter; this
corrected 12 of 17 previously-extracted `D)` values from spurious text back
to `None`. See `tests/test_parser.py`.

## Second labeling pass on severity (D-field + fixture-vs-light distinction)

Two refinements to `classify_severity` in `scripts/apply_labels.py`, both
motivated by real disagreements from the zero-shot LLM comparison (see
`failure_analysis.md`):

1. **D-field schedule conditions.** A `D)` field encoding a recurring
   schedule (`DLY`, a day-of-week range like `MON-FRI`, or a time range like
   `0400-0930`) makes an otherwise-unconditional-looking `CLSD` in the
   E-field conditional, the same way an `EXC`/`PPR` clause does. Example:
   `KHPN-07-096`, `E) RWY 16/34 CLSD` with `D) MON-FRI 0400-0930`, was
   `critical` and is now `advisory`.
2. **Sign illumination vs. fixture closure.** `SIGN ... LGT U/S` (a sign's
   light element unserviceable) is now `advisory` rather than inheriting the
   category's default `U/S -> critical` rule — a dark sign is degraded, not
   the same as the fixture itself being closed. This is distinct from a
   *direct* fixture U/S (`RWY 36 PAPI U/S`, `ILS RWY 27L IM U/S`), which
   still correctly reads as `critical`. Also fixed as part of this pass: the
   `WINDCONE`/`SEGMENTED CIRCLE` check in `classify_category` now runs
   *before* the generic `RWY` check (previously documented as an unfixed bug
   in `failure_analysis.md`), so `AP WINDCONE FOR RWY 29 LGT U/S` correctly
   lands in `lighting`/`advisory` instead of `runway`/`critical`.

Both refinements are also applied as a post-hoc guard on the ML model's
predictions at inference time (`src/severity_guard.py`), not just baked into
training labels — see `failure_analysis.md` for the measured before/after.

**This changed `ranking_scenarios.json`'s `relevant_ids`.** 5 NOTAMs that
were included as "relevant" because they were `critical` at the time
(`KJFK-03-408`, `KBOS-06-436`, `KBOS-07-095`, `KBOS-07-096`, `KEGE-07-036` —
all `SIGN ... LGT U/S` cases) are now `advisory` and were removed from the
affected scenarios' `relevant_ids` to keep the methodology below internally
consistent. This is a real, documented tradeoff: it makes the severity
ground truth more accurate, but real-world "does a pilot want to know about
this" relevance and "is this severity=critical" aren't quite the same axis
-- see `failure_analysis.md`, Ranker section, for the resulting (lower,
more honest) precision@k numbers and what they reveal.

## Process

Labels were assigned by systematically applying the rules above against the
Q-code and E-field text extracted by `src/parser.py`, then spot-checked by hand
against the raw NOTAM text for ambiguous cases (see `failure_analysis.md` for
documented disagreements and edge cases discovered during this process).

### Honest caveat: the labels are rule-derived, so mind what the F1 measures

Because the ground truth comes from a deterministic rule set
(`scripts/apply_labels.py`) rather than independent human annotation, the
classifier's precision/recall/F1 primarily measure **how well TF-IDF+LR
reproduces / generalises those rules**, not how well it matches an
independent human's judgment. Three things keep this from being a circular
"model learns to imitate the regex it was trained on" exercise, but it's
worth stating plainly:

1. It doesn't just memorise the rule — if it did, severity F1 would be ~1.0;
   it's 0.84 (5-fold), because a bag-of-words model errs differently than the
   regex.
2. The rule-vs-ML baseline in `failure_analysis.md` uses a *different* signal
   (the Q-code) than the labeling rule (E-field regex), so that head-to-head
   is genuinely independent of how the labels were made.
3. Several documented cases are ones where the classifier *disagreed with the
   rule labels and was arguably more correct* (the zero-shot LLM comparison
   surfaced two labeling-rule bugs this way, both since fixed).

The clean upgrade, if this went further, would be a small independently
hand-labeled test set (no rules involved) to measure real-world accuracy
rather than rule-reproduction accuracy — noted in the README's "what's left."

## Ranking scenario relevance (data/labeled/ranking_scenarios.json)

For each of the 18 scenarios, `relevant_ids` is the hand-judged set of NOTAMs
at the scenario's route airport(s) that a pilot briefing for that specific
flight should surface: critical-severity closures/outages (`CLSD` with no
exception, ILS/PAPI/runway-lighting `U/S`) plus genuine conditional closures
(`CLSD EXC`, `PPR`, wingspan-restricted taxiway closures). Routine
chart-amendment `navaid`-category NOTAMs (IAP/RNAV procedure AMDT text with
no U/S/CLSD) and pure obstruction-lighting notices (crane/tower lights) are
deliberately excluded from `relevant_ids` even though they are technically
active NOTAMs for that airport — they are exactly the "trivial" NOTAMs the
project's README describes the critical ones as being "buried among."
