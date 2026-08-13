# Failure Analysis

All examples below are real NOTAMs pulled from the live FAA feed
(`data/raw/notams.json`) and real errors produced by the actual pipeline run
(`scripts/train_classifier.py`, `scripts/run_pipeline.py`), not constructed
cases.

## Parser

Testing the parser against the FIR-sourced records (added below) found a
genuine bug: any parenthetical abbreviation ending in the letter D followed
by `)` -- e.g. `THEDFORD (TDD)`, `(2.0NM SW BED)` -- was mistaken for a `D)`
field marker, because the field regex had no requirement that a marker be
preceded by whitespace. This only bites the `D)` field specifically: it's
optional and often absent, so there's frequently no genuine earlier `D)` to
win Python's leftmost-match search; `A)`/`B)`/`C)`/`E)` are always present in
this corpus, so the real marker always wins that race first. Fixed in
`src/parser.py` by requiring a whitespace-or-start-of-string boundary before
every field letter (`FIELD_BOUNDARY`); this corrected 12 of 17
previously-extracted `D)` values from spurious text back to `None`, leaving
5 genuine schedule-bearing `D)` fields. See `tests/test_parser.py`,
`test_parenthetical_abbreviation_does_not_spawn_spurious_d_field`.

## Classifier

**Cross-validation and the rule-vs-ML baseline (the honest headline).**
Everything else in this section reports single-split numbers, which the
corpus is too small to make trustworthy on their own. 5-fold CV
(`scripts/cross_validate.py`, macro-F1 mean ± std) tells the real story and
pins down two things the project previously only asserted:

| Task | Q-code lookup (zero ML) | TF-IDF + LR | TF-IDF + LR + guard |
|---|---|---|---|
| category | 0.852 ± 0.036 | 0.945 ± 0.015 | — |
| severity | 0.437 ± 0.037 | 0.842 ± 0.116 | 0.902 ± 0.133 |

1. **Category is mostly a fixed-vocabulary lookup.** The ICAO Q-code's
   subject letters already decode category at 0.85 with zero ML
   (`src/qcode_baseline.py`, a majority-vote lookup on letters 2-3 of the
   Q-code). TF-IDF adds only ~0.09 — real, but small enough that by the
   project's own "rule where the vocabulary is fixed" principle, category is
   arguably over-modelled. The ML gain concentrates on the genuinely
   ambiguous `MM` movement-area subject code (splits runway/taxiway).
2. **Severity is where free-text ML earns its place.** The Q-code condition
   letters can't express the `EXC`/`PPR`/schedule conditionality that
   separates `critical` from `advisory`, so the rule collapses to 0.44
   macro-F1; the free-text classifier more than doubles it. This is the
   concrete evidence for the README's "why NOTAMs are hard" thesis.
3. **Severity is high-variance** (±0.11–0.13, folds spanning 0.62–0.92). The
   single-split severity deltas below are real *mechanisms* but sit inside
   that noise band — the guard's +0.06 CV mean improvement, for instance, is
   smaller than the fold-to-fold std, so it's "reliably helps a bit," not
   "worth exactly N points."

Sanity check on leakage: since the classifier trains on the full text
(which contains the Q-code), I re-trained on the **E-field alone** — it
scores essentially the same (category 0.92, severity 0.88 single-split), so
the model genuinely works from the free text, it isn't just reading the
Q-code out of the input.

**Corpus history**: 177 NOTAMs / 10 airports (1 `admin`, 1 `airspace`, 0
`informational`) -> 295 NOTAMs / 18 airports (2 `admin`, 1 `airspace`, 6
`informational` — expanding airports fixed `informational` but not
`admin`/`airspace`) -> **346 NOTAMs / 18 airports + 6 FIRs (2 `admin`, 67
`airspace`, 6 `informational`)**. The last jump came from testing (rather
than assuming) whether the FAA search backend supports FIR/ARTCC-level
queries — it does, and it surfaces genuinely different airspace/route
content that a per-airport search structurally cannot. See
`docs/labeling_guide.md`, "FIR-level search fixed the `airspace` gap," which
also documents that the *original* claim ("this can't be fixed without a
different data source") was itself wrong and has been corrected. `admin`
remains genuinely thin (2 examples) even after this fix.

**Class-weighting fix + severity guard + second labeling pass, measured on
the current 346-NOTAM corpus (labels refined per "Second labeling pass" in
`docs/labeling_guide.md`), identical 20%-held-out split (70 test records)**:

| Stage | Severity accuracy | Severity macro-F1 | `critical` recall | `critical` precision |
|---|---|---|---|---|
| Unweighted LR (original labels) | 0.81 | 0.48 | 0.38 (6/16) | 1.00 |
| + `class_weight="balanced"` (original labels) | 0.90 | 0.93 | 0.96 (22/23) | 0.79 |
| + refined labels (D-field/sign-light fixes), no guard | 0.83 | 0.86 | 0.75 (15/20) | 0.68 |
| + `guard_severity` post-hoc rule (`src/severity_guard.py`) | **0.93** | **0.94** | 0.75 (15/20) | **1.00** |

Category accuracy/macro-F1 on the current corpus: 0.96 / 0.93 (mostly the
`airspace` fix from the corpus expansion, not the class-weighting change).

The third row is a real, honest regression worth sitting with: making the
ground truth *more accurate* (distinguishing a sign's light being out from
the fixture itself being closed, and catching D-field schedule conditions)
made the classification task *harder* for a bag-of-words model, because the
distinguishing signal (the word "SIGN" being present, or a schedule in a
field the model never sees) is subtler than the blunt "CLSD/U-S present"
pattern it had been fitting. The `guard_severity` post-hoc rule (applied at
inference time, not baked into training) recovers precision back to 1.00 by
suppressing exactly the false-positive `critical` calls it can detect
deterministically, but it cannot fix recall — a model that never predicted
`critical` in the first place isn't something a downstream rule can rescue.
The 5 remaining recall misses are a real, unresolved gap: this is the single
most concrete "if I had more time" item in the whole project (see README
Section 8).

### Example 1 — the balancing fix's failure mode: "NA EXCEPT FOR ACFT EQUIPPED..." route text
Prediction: `critical` (4 of 7 severity misclassifications on the current split)
Ground truth: `advisory`
NOTAM text: `Q) KFDC/QARLC/... A) KZBW ... E) KZBW NY...ROUTE KZBW. V489 WEARD, NY TO ALBANY (ALB) VORTAC, NY NA EXCEPT FOR AIRCRAFT EQUIPPED WITH SUITABLE RNAV SYSTEM WITH GPS.` (and 3 more, all structurally identical)
Reason: this specific route-qualification phrasing ("NA EXCEPT FOR AIRCRAFT
EQUIPPED WITH SUITABLE RNAV SYSTEM WITH GPS") is extremely common across
both `advisory` route NOTAMs (routine GPS-equipage qualifications) and
`critical` ones (routes with a genuinely `UNUSABLE` navaid elsewhere in the
same text) in the new FIR-sourced data. The classifier picked up "VOR/
VORTAC + route text" as a strong `critical` signal from the latter group and
over-applies it to the former, which share almost identical surrounding
vocabulary but no actual `UNUSABLE` token.
Future improvement: same fix as before — combine the rule-based `UNUSABLE`/
`CLSD`/`U/S` presence check with the ML prediction at inference time (only
trust a `critical` prediction if one of those tokens is actually present in
the text), rather than relying on the bag-of-words model to separate "route
has a qualification clause" from "route is unusable."

### Example 2 — category: ILS-component text still conflated with runway-lighting U/S
Prediction: `runway`
Ground truth: `navaid`
NOTAM text: `06/040 NOTAMR ... A) KOSH ... E) ILS RWY 36 OM U/S`
Reason: `ILS RWY 36 OM U/S` (outer marker unserviceable — a `navaid`
component) shares the same `RWY <number> ... U/S` shape as the many
`runway`-category PAPI/REIL/threshold-light U/S NOTAMs in the corpus. This
is the same root-cause pattern as an earlier corpus snapshot's PAPI/taxiway
mixup — the specific example changed (this one wasn't previously
misclassified), but the underlying confusion between ILS-component
abbreviations (`OM`/`MM`/`IM`/`LOC`/`GS`) and runway-lighting abbreviations
(`PAPI`/`REIL`/`RAI`) in short E-field text hasn't gone away.
Future improvement: a small rule-based pre-filter recognizing the fixed
ICAO abbreviation set for ILS components (`OM`/`MM`/`IM`/`LOC`/`GS`/`GP`)
vs. runway-lighting fixtures (`PAPI`/`REIL`/`RAI`/`RTHL`) would resolve this
deterministically, since both vocabularies are fixed and small — this is
exactly the kind of case the project's core design principle says should be
a rule, not a model decision.

### Example 3 — category: the exact scope-ambiguity case flagged during labeling still reproduces as a model error
Prediction: `taxiway`
Ground truth: `runway`
NOTAM text: `07/023 ... A) KMVY ... E) RWY 15 TWY DIRECTION SIGN AT TWY A FOR TWY A LEFT SIDE NOT STD`
Reason: unchanged from the original diagnosis — this is the identical NOTAM
flagged in `docs/labeling_guide.md`'s "Ranking scenario relevance" note and
the README's "why NOTAMs are hard" section as a case where even the
rule-based labeler couldn't cleanly decide `runway` vs `taxiway`. It has
now survived three separate corpus expansions as a classifier
disagreement, which is a clean, repeated confirmation that the ambiguity is
real, not an artifact of one labeling or training pass.
Future improvement: accept a documented amount of noise on this exact
pattern, or have a human resolve a canonical label for this specific
phrasing and add matching training examples.

### Example 4 — a genuine ground-truth labeling bug the model's disagreement surfaced (fixed)
NOTAM text: `06/013 NOTAMR ... A) KBED ... E) AP WINDCONE FOR RWY 29 LGT U/S`
This was not a model error — it was a bug in the rule-based ground truth.
`scripts/apply_labels.py`'s `classify_category` used to check for `\bRWY\b`
*before* checking for `WINDCONE`, so "AP WINDCONE FOR RWY 29 LGT U/S" got
categorized `runway` (because it names a runway) instead of `lighting`
(a windcone is an airfield indicator, not runway infrastructure — compare
`KEGE-12-128`, `AP WINDCONE SEGMENTED CIRCLE U/S`, correctly labeled
`lighting` because it never mentions `RWY`). Once mis-bucketed as `runway`,
the severity rule's "`U/S` on `runway`/`taxiway`/`navaid` -> `critical`"
clause fired, producing `critical` ground truth for what is really a minor
indicator outage — the classifier's `advisory` prediction was the more
correct answer.
**Fixed**: moved the `WINDCONE`/`SEGMENTED CIRCLE`/obstruction-light check in
`classify_category` ahead of the generic `RWY` check, the same way the
`TWY`-leading-token check already takes priority over `RWY` for exactly
this kind of location-vs-subject ambiguity. `KBED-06-013` is now correctly
`lighting`/`advisory`. See `docs/labeling_guide.md`, "Second labeling pass."

### Example 5 — the true remaining singleton class
`admin` has exactly 2 examples in the current 346-NOTAM corpus even after
the FIR-level expansion fixed `airspace` (1 -> 67). Unlike the `airspace`
gap, this one has survived two different attempted fixes (more airports,
then FIR-level search) without moving, which is better evidence that it's a
genuine feed limitation rather than an untested assumption — though a
targeted search (e.g. explicitly for frequency-change NOTAMs) hasn't been
tried yet either. `train_classifier.py`'s `can_stratify` fallback keeps
training from crashing on this, but precision/recall on `admin` still isn't
a meaningful number with only 2 examples.

### DistilBERT comparison (stretch goal, now measured)
`scripts/train_distilbert.py` fine-tunes `distilbert-base-uncased` on the
identical 276/70 train/test split used for the TF-IDF baseline, rerun a
second time after the second labeling pass (D-field/sign-light fixes) for a
fair comparison against the *current* labels. Results: severity 0.986
accuracy / 0.988 macro-F1 (19/20 `critical` correct), category 0.957
accuracy / 0.956 macro-F1 — both still ahead of TF-IDF+LR (even
guard-corrected), at ~13x the inference latency (3.0-3.2ms vs 0.23ms).
Notably, DistilBERT's severity recall barely moved after the labels got
harder to fit with a bag-of-words model (23/23 -> 19/20), while TF-IDF's
recall dropped hard (22/23 -> 15/20, see the class-weighting table above) --
consistent with the theory that the SIGN/schedule distinctions are subtler
than TF-IDF's linear decision boundary can represent, but well within a
transformer's capacity. Given the latency cost, TF-IDF (with the
`guard_severity` post-hoc rule) remains the practical choice for a corpus
this size — see README Section 5 for the full comparison table.

### Zero-shot LLM comparison (stretch goal, now measured)
**Caveat: this comparison predates the FIR-level corpus expansion above** —
it ran on the 295-NOTAM corpus's 59-item held-out test sets, not the current
346-NOTAM/70-item ones, so its numbers aren't directly comparable to the
TF-IDF/DistilBERT rows measured after that expansion. It wasn't rerun
because the specific finding below (a documentation bug, not a data
quantity issue) doesn't depend on corpus size, and rerunning would cost
another full agent pass for a result that would likely look similar. Take
the numbers as a real but slightly earlier snapshot.

A fresh, context-isolated agent (no access to this project's ground truth or
prior conversation) was given only the taxonomy tables and quick rules from
`docs/labeling_guide.md` and asked to classify the same 59-item held-out test
sets by reading each NOTAM directly — a genuine zero-shot comparison, not a
script. Result: **0.73 accuracy / 0.73 macro-F1 on category, 0.76 accuracy /
0.59 macro-F1 on severity** — both well below the TF-IDF and DistilBERT
baselines. The interesting part is *why*, not just the number:

**Category — the agent applied the written taxonomy correctly; the ground
truth didn't match its own documentation.** 6 of the agent's `runway`
predictions where truth was `navaid` were all "`ILS OR LOC RWY XX, AMDT ...`"
chart-amendment text (e.g. `KATL-6-1924`, `KBNA-6-5440`, `KEGE-6-1572`). The
labeling guide's `runway` row used to say "ILS/localizer glideslope tied to a
specific runway" belongs there — the agent read that and classified
accordingly. But `scripts/apply_labels.py`'s actual rule checks for
`AMDT`/`APCH`/`IAP` *before* checking for `RWY`, routing this exact pattern to
`navaid` instead. The agent didn't misunderstand the text; it correctly
applied a taxonomy description that didn't match its own implementation.
Fixed: `docs/labeling_guide.md`'s `runway`/`navaid` rows have been rewritten
to state the actual implemented rule (approach-procedure text is `navaid`
even when it names a runway). This is a real documentation bug this
comparison caught, not a model limitation.

**Severity — the agent made more conservative, arguably more reasonable
judgment calls that the blunt rule-based ground truth doesn't allow for.**
Of 9 false negatives on `critical`, most (e.g. `KJFK-03-408`: `TWY B TWY
DIRECTION SIGN FOR TWY YA LGT U/S`; `KHPN-07-109`: `RWY 16 RWY EXIT SIGN AT
TWY L LGT U/S`) are cases where a *sign light* is unserviceable. The
rule-based ground truth's quick rule ("U/S on runway/taxiway/navaid ->
critical") is deliberately blunt and doesn't distinguish "the whole fixture
is closed" from "a sign's light bulb is out" — the agent, applying its own
judgment, called these `advisory` (a defensible reading: a dark sign at
night is degraded, not the same safety impact as a closed runway). One
case (`KHPN-07-096`, `RWY 16/34 CLSD`) has a `D)` field of `MON-FRI
0400-0930` — the agent likely read that as a scheduled/conditional closure
and called it `advisory`, while the ground-truth rule only checks the
E-field text for `EXC`/`PPR` and missed that the `D)` field itself encodes a
condition.
**Fixed**: this was genuinely a case where the *rule-based ground truth*
needed improving, not the zero-shot model. Both fixes the agent's
disagreements pointed at are now implemented in `scripts/apply_labels.py`
and mirrored in `src/severity_guard.py` — D-field schedule conditions
(`KHPN-07-096` specifically) and the sign-illumination-vs-fixture-closure
distinction (`KJFK-03-408`, `KHPN-07-109`) — see the Classifier section's
"Second labeling pass" and `docs/labeling_guide.md`. This is a genuine
example of a zero-shot LLM's disagreement with ground truth being *right*
and getting acted on, not dismissed.

Latency for this method isn't included in the README's comparison as a
clean ms figure: it ran as an agentic tool-calling loop over a 105-item
batch (~3.7s/item wall-clock on average), which bundles in reasoning and
file I/O overhead that a single raw LLM API call per NOTAM wouldn't have —
not an apples-to-apples number against TF-IDF/DistilBERT's single-inference
timings.

## Summarizer

Faithfulness rate measured across the full 346-NOTAM labeled corpus:
**346/346 (100%)**, zero issues of any type raised. That number is not a
success signal — it's a direct consequence of `summarize_notam` being a
pure field-echo template (it only ever inserts `a_field`/`e_field`/
`b_field`/`c_field`/category/severity verbatim, never paraphrasing), so the
checker's substring/token-overlap tests pass trivially by construction. The
three examples below explain why 100% here doesn't mean "summarization is
solved" — it means the current summarizer is too conservative to fail this
particular checker, and the checker itself has real gaps that would only
surface once the summarizer starts paraphrasing (e.g. the LLM-based stretch
goal).

### Example 1 — missing_end_time on EST-suffixed end dates
Generated summary: `At KJFK. JFK JOHN F KENNEDY INTL, NEW YORK, NY.\nCOPTER RNAV (GPS) 027... Valid from 2602171633 to 2710051633EST. Category: navaid. Severity: advisory.`
Issue: none in this case — but records where `c_field` carries an `EST`
suffix (e.g. `2710051633EST`) are the ones most likely to trigger
`missing_end_time`, because the checker does an exact substring match and
any reformatting of the date anywhere upstream breaks it.
Reason: the summarizer inserts `parsed['c_field']` verbatim, so this
particular case passes -- but it is fragile: it only works because
`summarize_notam` never transforms the date string. Any future change that
reformats dates for readability (e.g. converting `2710051633EST` to a
human date) would silently break faithfulness on every EST-suffixed NOTAM
in the corpus (192 of 346 records have a `C)` field ending in `EST`).
Fix: keep the faithfulness checker's date match tolerant of formatting by
comparing parsed datetimes instead of raw substrings, so the checker still
catches real omissions after a future summarizer rewrite.

### Example 2 — e_field_not_reflected on multi-line procedure-amendment text
Generated summary: `At KATL. RNAV (RNP) Z RWY 8L, AMDT 1A...\nRNP 0.30 DA 1516/HAT 501 ALL CATS... Valid from 2507011526 to 2602101526EST. Category: navaid. Severity: advisory.`
Issue: none for this specific text (the summarizer copies the whole E-field
verbatim, so token coverage always trivially passes for this class of NOTAM).
The real risk case is any *future* summarizer that tries to shorten or
paraphrase long procedure-amendment E-fields — the checker's
`e_field_not_reflected` rule only requires **one** token >3 chars to survive,
which is a very weak bar. A summarizer could drop 90% of a 3-line amendment
and still pass.
Fix: strengthen the check to require coverage of a minimum fraction of
E-field tokens (e.g. >=50%), not just one surviving token, so it actually
catches aggressive paraphrasing instead of only catching total field
omission.

### Example 3 — hallucination trigger list has a false-negative gap
Generated summary would only be flagged `hallucinated_term:*` for the four
hardcoded terms (`delay`, `cancel`, `divert`, `emergency`). None of these
appear in `summarize_notam`'s template output today (it only ever echoes
`a_field`/`e_field`/`b_field`/`c_field`/category/severity verbatim), so the
faithfulness rate on the current summarizer is close to 100% by construction
-- which is itself a limitation, not a success: the checker can't catch
hallucination from a smarter (e.g. LLM-based) summarizer beyond those four
words.
Fix: the checker is only meaningful once the summarizer stops being a pure
field-echo template (e.g. the LLM-based stretch goal). At that point the
trigger list needs to grow substantially, or be replaced with an actual
n-gram/entity overlap check against the source fields instead of a fixed
word list.

## Ranker

`scripts/evaluate_ranker.py` runs the full pipeline (parser -> trained
classifier -> ranker) against 18 hand-judged flight scenarios in
`data/labeled/ranking_scenarios.json` and measures precision@5 / precision@10
against a hand-picked "operationally relevant" set per scenario (closures and
U/S safety equipment, deliberately excluding routine chart-amendment NOTAMs
and pure obstruction-lighting notices).

**Five-stage measured history, same 18 scenarios throughout (stage 5's
scenario ground truth itself changed -- see Example 5):**

| Stage | Mean precision@5 | Mean precision@10 | Mean R-precision |
|---|---|---|---|
| 1. Original: unweighted classifier + original ranker weights | 0.544 | 0.489 | -- |
| 2. + `class_weight="balanced"` on the severity classifier | 0.833 | 0.706 | -- |
| 3. + ranker tie-breaker and route-match-bonus fixes (below) | 0.878 | 0.728 | -- |
| 4. + R-precision metric added (no ranker/classifier change) | 0.878 | 0.728 | 0.824 |
| 5. + severity guard, second labeling pass, relabeled scenarios | **0.733** | **0.644** | **0.748** |

Stage 1 -> 2 (classifier fix, `src/ranker.py` untouched) recovered almost the
entire gap to the ground-truth-label ceiling measured during the original
evaluation — direct proof that most of the ranker's real-world error was
inherited from upstream classifier mistakes, not the ranking formula. Stage
2 -> 3 (two small, targeted fixes to `src/ranker.py`, described below) closed
most of the remaining gap; at that point **15 of 18 scenarios were at their
mathematical precision@5 ceiling** (`min(n_relevant, 5) / 5` — see
Example 2). Stage 4 added R-precision (see the docstring in
`scripts/evaluate_ranker.py`) purely as a better metric for small scenarios;
it didn't change anything about the ranker or classifier.

**Stage 5 is a real, honest drop, not a regression to hide.** The second
severity labeling pass correctly downgraded 5 NOTAMs (`KJFK-03-408`,
`KBOS-06-436`, `KBOS-07-095`, `KBOS-07-096`, `KEGE-07-036` — all `SIGN ...
LGT U/S` cases) from `critical` to `advisory`. Per the documented
methodology in `docs/labeling_guide.md` ("relevant" = critical severity +
genuine conditional closures), these were removed from the affected
scenarios' `relevant_ids`, shrinking several scenarios' relevant sets and
exposing that the ranker's top-5 often still contains these now-`advisory`
sign-light items anyway (their score is still competitive on
category+route-match+keyword-tiebreak, just not as high as genuine
closures) — see Example 5. This reveals a real, previously-invisible tension
between "is this severity=critical" and "does a pilot actually want to know
about this," which the ranker's severity-weighted formula doesn't fully
capture. 0.733 is a more honest number than 0.878 was, not a worse ranker.

### Example 1 — the ranker's ceiling was set by the classifier, not the ranking formula (fixed)
Scenario: `S01` (route `KJFK, KBOS`)
Before the classifier fix: precision@5 = 0.20 (1/5 correct). Top-5 ranked:
`KJFK-06-359, KJFK-5-2951, KJFK-5-2362, KJFK-5-3815, KJFK-11-346`.
Reason: `KJFK-6-4753`, `KJFK-03-405`, `KJFK-03-408`, `KJFK-03-409` are all
ground-truth `critical` taxiway/navaid closures that should rank at the top
(severity weight 3.0), but the *unweighted* severity classifier predicted
`advisory` for every one of them (the same critical-recall problem in the
Classifier section). Once demoted to `advisory`, they scored identically to
the dozens of routine `navaid` chart-amendment NOTAMs at KJFK and lost the
tie to whichever happened to come first in the raw feed order.
Outcome: **fixed** — after `class_weight="balanced"` (Classifier section),
`S01` now scores precision@5 = 1.00. Re-running all 18 scenarios with the
balanced classifier alone (no ranker code changes) raised mean precision@5
from 0.544 to 0.833, confirming the diagnosis.

### Example 2 — precision@k has a hard mathematical ceiling for small airports (fixed with a better metric)
Scenario: `S14` (route `KASE`), precision@5 = 0.40, unchanged by any ranker/classifier fix
Reason: KASE has only 5 total NOTAMs in the corpus and only 2 are labeled
relevant. Precision@5 can never exceed 2/5 = 0.40 for this scenario no matter
how good the ranker is — the metric itself, not the model, is the limiting
factor. The same pattern holds for `S17` (`KACK`) and `S18` (`KEGE`).
Fix applied: added R-precision (precision@R where R = number of relevant
items — a standard IR metric) to `scripts/evaluate_ranker.py` alongside
precision@5/@10. `S14`'s R-precision is 1.00 (both relevant items are in the
top 2) even though its precision@5 is stuck at 0.40 — the new metric
correctly shows this scenario's ranking is actually perfect, which
precision@5 alone couldn't express. Mean R-precision (0.748, stage 5) is
*not* directly comparable to mean precision@5 (0.733) since they're
different metrics measuring different things (R-precision gets harder for
large-relevant-set scenarios, easier for small ones) — see the stage table
above.

### Example 3 — tied scores fell back to arbitrary feed order (fixed)
Scenario: `S12` (route `KMVY, KACK`)
Before the ranker fix: precision@5 = 0.40 even with ground-truth labels (no
classifier-error contribution — this was a pure ranker-formula gap). Top-5
was `KACK-04-132, KACK-07-034, KACK-5-2447, KMVY-4-0924, KMVY-6-4626` — only
the first two are relevant; the next three are routine chart-amendment
NOTAMs that tied on score (1.5 severity + route-match + 1.5 category + 0.5
time = same bucket as the genuinely relevant `KMVY-07-020/021/023`, which
lost the tie purely because KACK's records happened to come first in
`notams.json`'s feed order).
Fix applied: added a +0.3 tie-breaker in `src/ranker.py` for E-field text
containing an explicit restriction/degradation keyword (`CLSD`, `PPR`,
`U/S`, `NOT STD`) — a signal category+severity alone can't capture, since a
routine amendment note and a real closure can land in the same bucket.
Outcome: `S12` precision@5 improved to 0.80 (now at ceiling given its
tied 5th-place item); mean precision@5 across all 18 scenarios rose from
0.833 to 0.878.

### Example 4 — off-route critical NOTAMs could outrank on-route advisory ones (fixed)
Before the fix: an off-route `critical` NOTAM scored 3.0 (severity) + 0 (no
route match) + 1.5 (high-impact category) + 0.5 (time window) = 5.0, while
an on-route `advisory` NOTAM in the `lighting` category (not in
`HIGH_IMPACT_CATEGORIES`) scored 1.5 + 2.5 (route match) + 0 + 0.5 = 4.5 —
the off-route NOTAM ranked higher despite being irrelevant to the flight.
Reason: the original route-match bonus (+2.5) and the severity weight
(critical=3.0) were close enough in magnitude that severity alone could
outweigh route relevance.
Fix applied: raised `ROUTE_MATCH_BONUS` from 2.5 to 6.0 in `src/ranker.py` —
large enough that the maximum possible non-route score (severity 3.0 +
category 1.5 + time 0.5 + keyword-tiebreak 0.3 = 5.3) can never exceed even
the minimum on-route score, so an on-route NOTAM always outranks an
off-route one regardless of severity. This was folded into the same
`evaluate_ranker.py` re-run as Example 3, so its isolated contribution isn't
separately measured, but it removes a class of error entirely rather than
just reducing its frequency.

### Example 5 — severity accuracy and ranking relevance are not the same axis (new finding, not fixed)
Scenario: `S15` (route `KJFK`), precision@5 dropped from 0.8 to 0.4 across
the second labeling pass
Top-5 both before and after: `KJFK-6-4753, KJFK-12-054, KJFK-12-055,
KJFK-03-405, KJFK-03-408` (unchanged — the *ranking* didn't move)
Reason: `KJFK-03-408` (`TWY B TWY DIRECTION SIGN FOR TWY YA LGT U/S`) was
correctly downgraded from `critical` to `advisory` by the sign-light fix,
so per the documented methodology it was removed from `S15`'s
`relevant_ids`. But the ranker's top-5 *still contains it* — its score
(category=taxiway high-impact + route-match + time-window + the `U/S`
keyword tiebreak) is still competitive with the genuinely relevant items,
because the ranker has no way to distinguish "a sign's light is out" from
"a taxiway segment is closed" once both land in the same
category+severity+keyword bucket. A pilot briefing arguably still wants to
know a taxiway sign is dark at night, even though it's correctly
`advisory` rather than `critical` — this exposes a real design question the
project hadn't confronted before: **is "ranking relevance" supposed to track
severity, or is it a genuinely different axis that happens to correlate
with severity most of the time?** This project's ranker assumes the former;
this example is evidence that assumption has limits.
Future improvement: this isn't a bug to patch so much as a scope decision
to revisit — either accept that severity-weighted ranking will sometimes
under-rank "worth knowing but not critical" items like sign lights, or add
a separate relevance signal (e.g. a fixed weight for any taxiway/runway
*signage/marking* NOTAM regardless of severity) that's decoupled from the
severity classifier entirely.

## RAG retrieval (stretch goal, retrieval half built and measured; generation half not automatable here)

`src/rag.py` retrieves the top-k NOTAMs for a free-text question by
combining TF-IDF query-document cosine similarity with the same
route-relevance ranking used elsewhere (`src/ranker.py`), and `src/weather.py`
pulls live METAR/TAF from `aviationweather.gov`'s confirmed-working
endpoints. `scripts/notam_qa.py` wires both together into a real, working
CLI. The generation half genuinely needs an LLM API call this environment
has no key for — the script detects `ANTHROPIC_API_KEY` and calls it if
present, otherwise it prints the assembled context and stops there, which
is an honest boundary rather than a fake stub.

### Finding — natural-language questions have a real vocabulary gap against ICAO abbreviations
Query: `"any taxiway closures at Oshkosh"` (route `KOSH`)
Result: every retrieved NOTAM's `query_similarity` was **0.00** — the
retrieval fell back almost entirely to the rule-based relevance component
(40% of the blended score), effectively ignoring the question's content.
Re-running the identical route with the query `"TWY CLSD"` instead (actual
NOTAM vocabulary) produced meaningful similarities (0.11-0.40) and correctly
surfaced `KOSH-07-005` (`TWY C3 CLSD`) as the top result.
Reason: TF-IDF cosine similarity requires shared vocabulary. A pilot's
natural-language question ("any taxiway closures") and a NOTAM's terse
ICAO shorthand ("TWY C3 CLSD") share almost no tokens even though they're
about exactly the same thing. This is the real, structural reason RAG over
this kind of text needs either query expansion (mapping "taxiway closure"
-> `TWY`/`CLSD`) or a semantic embedding model rather than lexical
similarity — not a bug in the retrieval code, but a genuine property of the
domain that a from-scratch TF-IDF retriever can't paper over.

### Finding — pure query similarity can outrank route relevance when they conflict
Same `"TWY CLSD"` query above surfaced `KAPA-07-023` (`TWY D CLSD`, at
Centennial/Denver, off-route for a KOSH query) in the top 5, ranked *above*
two genuinely on-route `KOSH` results, because its query similarity (0.40)
was high enough to outweigh the 40% relevance-score weighting even after
the off-route penalty. Whether this is desirable depends on intent (a
general "show me TWY closures" query arguably should surface it; a
route-specific briefing arguably shouldn't) — documented as a real,
observed tension in the blending weights (`0.6` similarity / `0.4`
relevance in `src/rag.py`) rather than silently tuned away.

### Demo — one worked example, generation done by hand (no API key in this environment)
Query: `"TWY CLSD"`, route `KOSH`. Context assembled by
`scripts/notam_qa.py` (weather + top-5 retrieved NOTAMs, see the script's
output). A faithful answer from that context, written by hand to
demonstrate what the generation step would produce: *"At KOSH, taxiway C3
is closed (KOSH-07-005, advisory). Taxiway A also has non-standard surface
and centerline markings between A1 and A6 (KOSH-07-027, KOSH-07-013,
advisory), and runway 05/23 is closed except for taxi use (KOSH-06-039,
advisory). Current KOSH weather: 4SM visibility in haze, light wind — no
weather-driven operational impact beyond what's already in the NOTAMs.
Note: a 'TWY D CLSD' NOTAM at KAPA (Denver) also matched this query but is
unrelated to your KOSH route."* This demonstrates the retrieval-to-context
pipeline is real and sufficient for generation — the missing piece is
purely the API call, not the surrounding system.
