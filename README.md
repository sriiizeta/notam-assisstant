# NOTAM Analysis Pipeline

An evaluated NLP pipeline for interpreting NOTAMs, comparing classical ML and
LLM-based approaches while measuring accuracy, latency, and faithfulness.

Commercial and hobby NOTAM decoders already exist (Notamify is the most
serious production tool; NOTAMigo, NOTAM Explainer, rotatepilot.com, and
notamdecoder.com are free/hobby alternatives). This project does not try to
out-build them. Its value is the documented, evaluated ML engineering
process behind interpreting NOTAMs — rule-vs-ML tradeoffs, measured
accuracy/latency/faithfulness, and honest failure analysis — none of which
those tools publish. Read this README with that framing: the evaluation is
the deliverable, not the demo.

## At a glance

**What this is**: a pipeline that takes raw NOTAM text and, for a given
flight route, returns each NOTAM tagged with a category and severity, a
plain-language summary, a faithfulness check on that summary, and a
relevance-ranked order — plus a full evaluation of how well each of those
four steps actually works, with real numbers and documented failure modes.

**What it does, concretely** (`python scripts/run_pipeline.py KJFK,KBOS`):
1. Parses raw ICAO-format NOTAM text into structured fields (rule-based).
2. Classifies each NOTAM's `category` (runway/taxiway/navaid/airspace/
   lighting/admin/other) and `severity` (critical/advisory/informational)
   using a trained model.
3. Generates a short summary built only from the parsed fields, then checks
   that summary against the source fields for missing facts or invented
   claims.
4. Ranks all NOTAMs by relevance to the airports on your route, so the
   handful that actually matter surface above the routine chart-amendment
   noise.

**Status**: core pipeline + evaluation complete (see Section 5). Beyond the
original scope this also includes DistilBERT and zero-shot-LLM comparisons, a
FIR-level corpus expansion, 5-fold cross-validated metrics, and a zero-ML
Q-code rule baseline that the ML classifier is measured against head-to-head.
What's genuinely still open, and how to pick it up, is in Section 8,
"What's left."

## 1. Problem

NOTAMs (Notices to Air Missions) are official flight-safety notices written
in dense ICAO-standard shorthand. A single busy airport can have 50+ active
NOTAMs at once, and the genuinely critical one is often buried among dozens
of trivial ones (chart-amendment notes, faded taxiway paint, an obstruction
light out miles from the airport). This is a documented aviation
human-factors problem, not an invented one — this pipeline parses, classifies,
summarizes, and ranks real NOTAMs for a given flight route to surface what
actually matters first.

## 2. Why NOTAMs are hard

The Q-line and A/B/C/D fields follow a fixed ICAO standard — no model is
needed to parse those, ever. The E-field (free text) is where this project's
difficulty lives, because different authors/airports/countries write it with
different abbreviations, clause ordering, and ambiguous scope. There is no
fixed dictionary that covers it. Four real examples from this exact corpus
show why a keyword-matching rule can't do this job:

- `RWY 15L/33R CLSD EXC TAX 30MIN PPR 617-561-1919` (`KBOS-07-223`) — closed
  for takeoff/landing, but still open for taxiing under a time-limited PPR
  condition. A naive "CLSD -> critical" rule gets this wrong.
- `TWY SS BTN TWY A AND TWY J CLSDTWY JRSTRCTN>0? TO| ACFT WINGSPAN MORE
  THAN...` (`KORD-05-543`) — relevance depends on data not even present in
  the NOTAM (the aircraft's wingspan).
- `TWY KG BTN TWY A AND TWY B CLSD` (`KJFK-11-346`) — only a specific segment
  is closed, defined relative to two other named taxiways, not a fixed
  location.
- `RWY 15 TWY DIRECTION SIGN AT TWY A FOR TWY A LEFT SIDE NOT STD`
  (`KMVY-07-023`) — the primary subject (a taxiway sign) is genuinely
  ambiguous relative to the runway it's located near; even the hand-labeling
  rules in `docs/labeling_guide.md` couldn't fully resolve this one (see
  `failure_analysis.md`, Classifier Example 3).

## 3. Architecture

```
raw NOTAM text
     │
     ▼
src/parser.py        rule-based, ICAO-standard Q/A/B/C/D/E field extraction
     │
     ▼
TF-IDF + LogisticRegression   category + severity classification (2 models)
     │
     ▼
src/summarizer.py    template summary built ONLY from parsed fields + E-text
     │
     ▼
src/faithfulness.py  rule-based checker: does the summary actually reflect
                      the source fields, with no hallucinated terms?
     │
     ▼
src/ranker.py         rule-based relevance scoring for a given flight route
```

Rules where the input has a fixed, known vocabulary (Q-line, A/B/C/D fields,
labeling heuristics, faithfulness checks, ranking weights). ML only where a
fixed vocabulary can't cover the input (E-field category/severity
classification). See `docs/labeling_guide.md` for how that design principle
was applied to hand-labeling too.

## 4. Dataset

346 real NOTAMs pulled from the FAA public NOTAM Search backend
(`notams.aim.faa.gov/notamSearch/search` — the API behind the public FAA
NOTAM Search tool; no API key required). `aviationweather.gov`'s
`/api/data` endpoint, suggested as a starting point, turned out not to expose
a NOTAM product (only METAR/TAF/PIREP/SIGMET/etc. — confirmed by inspecting
its OpenAPI spec), so this FAA backend was used instead. Sourced from 18 US
airports, mixing busy (KJFK, KATL, KORD, KBOS, KBNA) and small/GA-heavy
(KASE, KOSH, KACK, KMVY, KEGE, KTEB, KBED, KHPN, KFRG, KISP, KPWK, KVNY,
KAPA) fields — the latter group was expanded from the original 10 airports
specifically to reduce the corpus's category/severity imbalance — plus 6
ARTCC/FIR designators (`ZNY`, `ZBW`, `ZAU`, `ZTL`, `ZME`, `ZDV`), added after
discovering (by testing, not assuming) that the same search endpoint also
accepts a bare FIR code and returns genuinely different, area-wide airspace/
route NOTAMs that a per-airport search structurally cannot surface — see
`scripts/fetch_notams.py`.

**Labeling taxonomy** (`docs/labeling_guide.md`):
- category: `runway | taxiway | navaid | airspace | lighting | admin | other`
- severity: `critical | advisory | informational`

Labels were applied via a deterministic rule set over the E-field text
(`scripts/apply_labels.py`) — the same "use rules for fixed vocabulary"
principle applied to the hand-labeling process itself, then spot-checked by
hand. Resulting distribution (real, not balanced by design):

| category | count | | severity | count |
|---|---|---|---|---|
| navaid | 120 | | advisory | 242 |
| airspace | 67 | | critical | 98 |
| lighting | 50 | | informational | 6 |
| taxiway | 49 | | | |
| runway | 48 | | | |
| other | 10 | | | |
| admin | 2 | | | |

`airspace` went from 1 example (after expanding to 18 airports alone) to 67
once the FIR-level search above was added — see `docs/labeling_guide.md`,
"FIR-level search fixed the `airspace` gap," which also documents that an
earlier version of this doc wrongly claimed that gap was unfixable without a
different data source. `admin` remains genuinely thin (2 examples) even
after that fix. These counts also reflect a second labeling-rule refinement
(D-field schedule conditions, sign-illumination-vs-fixture-closure, and a
windcone-categorization bug) described in `docs/labeling_guide.md`, "Second
labeling pass" — it moved 14 records from `critical` to `advisory` and 2
from `runway` to `lighting`.

## 5. Evaluation

Faithfulness runs over all 346 records; the ranker over all 18 hand-judged
scenarios. For the classifiers, the headline numbers are **5-fold
cross-validated** (`scripts/cross_validate.py`) rather than single-split,
because the corpus is small enough that one 20% split is noisy — severity
macro-F1 swings from 0.62 to 0.92 across folds, so a single split can
easily over- or under-state any result. Single-split runs
(`scripts/train_classifier.py`) are still used where a specific mechanism is
easier to show on one fold (the class-weighting ablation below), and flagged
as such.

### Classifier — cross-validated, and measured against a zero-ML rule baseline

The README's premise is "rule vs ML tradeoffs, measured," so the primary
comparison is the TF-IDF classifier against an actual rule: a transparent
majority-vote lookup on the ICAO Q-code (`src/qcode_baseline.py` — the
Q-code is a fixed-vocabulary field the parser already extracts, so decoding
it is a rule, not a model). 5-fold CV, macro-F1 (mean ± std):

| Task | Q-code lookup (zero ML) | TF-IDF + LR | TF-IDF + LR + `guard_severity` |
|---|---|---|---|
| category | 0.852 ± 0.036 | **0.945 ± 0.015** | — |
| severity | 0.437 ± 0.037 | 0.842 ± 0.116 | **0.902 ± 0.133** |

This is the honest, thesis-sharpening result the project is really about:
- **Category is mostly a lookup.** A zero-ML Q-code decode already scores
  0.85; ML adds only ~0.09. By the project's own "use a rule where the
  vocabulary is fixed" principle, category classification is arguably
  *over-engineered* as an ML task — the ML gain is real but small, and comes
  mostly from the genuinely ambiguous `MM` movement-area code.
- **Severity is where ML earns its place.** The Q-code condition sub-code
  can't see the `EXC`/`PPR`/schedule conditionality that separates
  `critical` from `advisory`, so the rule collapses to 0.44; the free-text
  classifier more than doubles it. This is the concrete evidence for "why
  NOTAMs are hard" that the project previously only asserted.
- **Severity is high-variance** (±0.11–0.13). The single-split deltas the
  ablation below reports are real mechanisms but sit inside that fold-to-fold
  noise band — read them as "how the fix works," not "the fix is worth
  exactly N points."

I also confirmed the classifier isn't just reading the Q-code out of the
full text: training on the **E-field alone** scores essentially the same
(category 0.92, severity 0.88), so the "ML on free text" claim holds up.

**Latency and transformer comparison** (single-split, same 276/70 split for
a fair head-to-head; all inference times measured on Apple Silicon MPS):

| Model | Task | Accuracy | Macro F1 | Avg inference time |
|---|---|---|---|---|
| TF-IDF + LR (+`guard_severity`) | severity | 0.93 | 0.94 | 0.23 ms |
| TF-IDF + LR | category | 0.96 | 0.93 | 0.23 ms |
| DistilBERT (fine-tuned, 6 epochs) | severity | 0.99 | 0.99 | 3.21 ms |
| DistilBERT (fine-tuned, 6 epochs) | category | 0.96 | 0.96 | 3.07 ms |
| Zero-shot LLM† | severity | 0.76 | 0.59 | not comparable‡ |
| Zero-shot LLM† | category | 0.73 | 0.73 | not comparable‡ |

### Classifier — how the severity fix works (single-split ablation)

The class-weighting fix is the highest-leverage change in this whole
pipeline, but it's a two-part story once you account for a second, later
labeling-rule refinement (D-field schedule conditions, sign-illumination-
vs-fixture-closure — see Section 4 and `docs/labeling_guide.md`). Single
split, so read it for the mechanism, not the exact points (the CV table
above is the honest headline):

| Stage | Severity accuracy | `critical` recall | `critical` precision |
|---|---|---|---|
| Unweighted LR, original labels | 0.81 | 0.38 (6/16) | 1.00 |
| + `class_weight="balanced"`, original labels | 0.90 | 0.96 (22/23) | 0.79 |
| + refined labels, no guard | 0.83 | 0.75 (15/20) | 0.68 |
| + `guard_severity` post-hoc rule | **0.93** | 0.75 (15/20) | **1.00** |

Making the ground truth more accurate (distinguishing "a sign's light is
out" from "the fixture is closed") made the raw classification task
*harder* for TF-IDF — that's a real, honest tradeoff, not a wrong turn.
`guard_severity` (`src/severity_guard.py`), a small rule layered on top of
the model's prediction at inference time, recovers precision back to 1.00,
but can't fix recall — 5 genuinely-critical NOTAMs the model doesn't call
critical in the first place are the biggest concrete gap left in this
project (see Section 8). See `failure_analysis.md` for the exact false
positives/negatives at each stage.

**DistilBERT vs TF-IDF (`scripts/train_distilbert.py`, same 276/70
train/test split as the baseline, rerun on the refined labels for a fair
comparison)**: DistilBERT still edges out TF-IDF (even guard-corrected) on
both tasks, at **~13x the inference latency** (3.0-3.2ms vs 0.23ms on Apple
Silicon MPS) — and notably, its severity recall barely moved after the
labels got harder (23/23 -> 19/20), unlike TF-IDF's sharp drop, consistent
with the refined distinctions being subtler than a linear bag-of-words
model can represent but well within a transformer's capacity. Given the
latency cost, TF-IDF+guard remains the practical choice for a corpus this
size.

**† Zero-shot LLM comparison predates the FIR-level corpus expansion below**
(it ran on an earlier 295-NOTAM/59-item-test-set snapshot, not the current
346-NOTAM/70-item one) — see `failure_analysis.md` for why it wasn't rerun.
A fresh, context-isolated agent was given only the taxonomy definitions and
quick rules (no ground truth) and asked to classify the held-out test set by
reading each NOTAM directly. *Result: it underperforms both baselines
(0.73-0.76 accuracy) — but not because LLMs are worse at language
understanding.* It lost mainly by faithfully applying the taxonomy table as
written, while the actual ground truth was produced by a more specific rule
implementation with an undocumented override (since fixed — see
`docs/labeling_guide.md`'s note on the `runway`/`navaid` boundary) — the
zero-shot model was scored against labels generated by a slightly different
rule set than the one it was shown. **‡** latency is marked "not comparable":
this used an agentic tool-calling loop over a 105-item batch (real
wall-clock time ~3.7s/item average), not a single raw LLM API call per item.
See `failure_analysis.md` for the specific misclassification examples.

### Summarizer / faithfulness

**346/346 (100%) faithful** across the full corpus, zero issues raised.
This is not a solved-summarization signal — the current summarizer is a
pure field-echo template, which makes the faithfulness checker pass
trivially. See `failure_analysis.md` for why 100% here is a checker-coverage
limitation, not evidence that summarization is easy.

### Ranker (precision@k)

Measured across the full history of fixes (same 18 scenarios throughout,
though stage 5's scenario ground truth itself changed — see below):

| Stage | Mean precision@5 | Mean precision@10 | Mean R-precision |
|---|---|---|---|
| 1. Unweighted classifier + original ranker weights | 0.544 | 0.489 | -- |
| 2. + `class_weight="balanced"` severity classifier | 0.833 | 0.706 | -- |
| 3. + ranker tie-breaker & route-match-bonus fixes | 0.878 | 0.728 | -- |
| 4. + R-precision metric (no ranker/classifier change) | 0.878 | 0.728 | 0.824 |
| 5. + severity guard + refined labels + relabeled scenarios | **0.733** | **0.644** | **0.748** |

Stage 1 -> 2 (fixing only the classifier, `src/ranker.py` untouched)
recovered almost the entire gap to the ground-truth-label ceiling — direct
proof that most of the ranker's real-world error was inherited from the
classifier, not the ranking formula. Stage 2 -> 3 (two targeted fixes to
`src/ranker.py`: a keyword-based tie-breaker and a much larger route-match
bonus) closed most of what remained; at that point 15/18 scenarios were at
their mathematical precision@5 ceiling. Stage 4 added R-precision
(precision@R where R = number of relevant items) as a better metric for
small scenarios — it sidesteps the fixed-k ceiling entirely, at the cost of
not being directly comparable to precision@5/@10 numbers.

**Stage 5's drop is a real, honest finding, not a regression.** The second
labeling pass correctly downgraded 5 NOTAMs from `critical` to `advisory`
(all `SIGN ... LGT U/S` cases — a sign's light being out, not the fixture
itself being closed). Per this project's own documented "relevant =
critical + genuine closures" methodology, those 5 were removed from the
affected scenarios' `relevant_ids`. The ranker's top-5 output didn't change
at all — it still ranks those sign-light NOTAMs competitively (their
category+route-match+keyword score is close to genuine closures') — but
they're no longer counted as hits, which is why precision dropped. This
surfaced a real, previously invisible question this project hadn't
confronted: **is ranking relevance supposed to track severity, or is it a
genuinely separate axis?** A pilot arguably still wants to know a taxiway
sign is dark at night even though it's correctly `advisory` now. See
`failure_analysis.md`, Ranker Example 5.

## 6. Failure analysis

Full writeup with real examples (not constructed cases) for the classifier,
summarizer, and ranker is in [`failure_analysis.md`](failure_analysis.md).
Headline findings:
- **Category classification is mostly a fixed-vocabulary lookup, not really
  an ML problem.** A zero-ML Q-code decode scores 0.85 macro-F1 (5-fold CV);
  the TF-IDF classifier adds only ~0.09. Severity is the opposite — the
  Q-code rule collapses to 0.44 because it can't see conditionality, and the
  free-text classifier more than doubles it. That contrast *is* the
  project's "why NOTAMs are hard" thesis, now measured rather than asserted.
- The corpus is small enough that single-split metrics are noisy (severity
  macro-F1 ranges 0.62–0.92 across folds), so headline classifier numbers
  are 5-fold cross-validated. Several deltas the earlier write-ups treated as
  precise findings sit inside that noise band and are now framed as
  mechanisms, not point estimates.
- Severity classifier's `critical` recall was the single biggest lever on
  end-to-end quality: `class_weight="balanced"` took it from 0.38 to 0.96
  recall. A later, more accurate second labeling pass (distinguishing a
  sign's light being out from the fixture itself being closed) made the raw
  classification task harder and dropped recall back to 0.75 — a real
  tradeoff, not a wrong turn. `guard_severity`, a rule layered on top of the
  model's prediction, recovers precision to 1.00 but can't fix recall; the
  5 remaining false negatives are this project's most concrete open item.
- Two small, targeted ranker fixes (a keyword tie-breaker, a larger
  route-match bonus) closed most of the ranker's gap to its theoretical
  ceiling — but a later fix (the same severity relabeling above) then
  dropped ranker precision@5 from 0.878 to 0.733, honestly: it surfaced a
  real tension between "is this severity=critical" and "does a pilot want
  to know about this" that the ranker's severity-weighted formula doesn't
  fully capture (see Ranker Example 5).
- An earlier version of this project claimed the `airspace` category's
  class-imbalance gap (1 example) was structurally unfixable without a
  different data source. That claim was wrong and has been corrected: the
  same FAA search endpoint also accepts FIR/ARTCC designators, and pulling
  from 6 of them took `airspace` from 1 example to 67 (see
  `docs/labeling_guide.md`). `admin` (2 examples) remains genuinely thin.
- Testing the parser against the FIR-sourced records found a real bug:
  parenthetical abbreviations ending in a field letter (`THEDFORD (TDD)`)
  were mistaken for field markers, corrupting 12 of 17 `D)` field
  extractions. Fixed by requiring a whitespace boundary before every field
  marker.
- The faithfulness checker is only as strong as the summarizer it's checking
  — it can't yet catch anything beyond a 4-word hallucination trigger list,
  which is fine for today's template summarizer but would need to grow before
  it could check an LLM-based one.
- The ranker's scoring formula has a real design tension between severity
  weight and route-match bonus (an off-route critical NOTAM can outrank an
  on-route advisory one) that's a policy choice, not a bug — see Ranker
  Example 4.

## 7. Getting started / Demo

Setup (once):
```
cd notam-assistant
pip install -r requirements.txt --break-system-packages   # torch/transformers
                                                            # only needed for
                                                            # train_distilbert.py
```

Reproduce everything from scratch, in order:
```
pytest tests/                           # 55 unit tests -- see note below
python scripts/fetch_notams.py          # pull NOTAMs from FAA public feed
python scripts/make_labels.py           # regenerate label template from raw data
python scripts/apply_labels.py          # apply the rule-based label set
python scripts/train_classifier.py      # train category/severity models (single split)
python scripts/cross_validate.py        # 5-fold CV + zero-ML Q-code rule baseline
python scripts/train_distilbert.py      # (optional, slow) DistilBERT comparison
python scripts/evaluate_faithfulness.py # faithfulness rate across the corpus
python scripts/evaluate_ranker.py       # precision@k / R-precision across 18 scenarios
python scripts/run_pipeline.py KJFK,KBOS   # ranked, summarized NOTAMs for a route
python scripts/notam_qa.py "TWY CLSD" KOSH # RAG retrieval + live weather (see Section 8)
streamlit run app.py                    # interactive UI (upload notams.json)
```

`pytest tests/` now covers 55 tests (parser, faithfulness, ranker, severity
guard, RAG retrieval, the Q-code rule baseline, and the HTML-corruption /
D-field / multi-part parser fixes).

Note: `fetch_notams.py` and `train_classifier.py` must run before
`run_pipeline.py`/`app.py` will work — they produce `data/raw/notams.json`
and `model_category.joblib`/`model_severity.joblib` respectively, none of
which are committed to git (see `.gitignore`; they're regenerated outputs,
not source).

`run_pipeline.py` prints each NOTAM for the given route in ranked order with
its predicted category/severity, generated summary, and faithfulness flag —
see the Evaluation section above for what those numbers mean in aggregate.

## 8. What's left

Everything in the original project brief's core scope, all four stretch
items from a first follow-up pass (DistilBERT comparison, zero-shot LLM
comparison, corpus expansion, pytest suite), and — from a second pass —
the FIR-level NOTAM search, every concrete code fix identified at that
point (severity guard, faithfulness datetime/coverage fixes, R-precision
metric, second severity labeling pass), and the RAG retrieval + live
weather groundwork are done. The FIR-level search item in particular used
to be listed here as "needs new data, not new code, and possibly not
fixable" — that assumption was tested rather than taken on faith, turned
out to be wrong, and is now fixed (see Section 4 and
`docs/labeling_guide.md`). What's below is what's actually still open.

**Built, with one structural gap that isn't fixable in this environment:**
- **RAG Q&A over NOTAMs + live weather** (`scripts/notam_qa.py`,
  `src/rag.py`, `src/weather.py`). Live METAR/TAF fetching and NOTAM
  retrieval (TF-IDF query similarity blended with the existing route-
  relevance ranker) are real, tested code — try
  `python scripts/notam_qa.py "TWY CLSD" KOSH`. Testing it surfaced a real
  finding: natural-language questions ("any taxiway closures") have almost
  zero lexical overlap with terse ICAO shorthand ("TWY C3 CLSD"), so
  retrieval quality depends heavily on the query using NOTAM-like
  vocabulary — see `failure_analysis.md` for the measured before/after and
  a worked demo where the generation step was completed by hand. The
  actual generation call needs an LLM API key this environment doesn't
  have; the script detects `ANTHROPIC_API_KEY` and uses it if present,
  otherwise it prints the assembled context and stops there honestly
  rather than faking a response.

**Concrete code fixes still open:**
- The 5 remaining `critical` recall misses (0.75 recall even after the
  severity guard — see Section 5) are the most concrete unresolved item.
  `guard_severity` can only suppress false positives, not create true
  positives a model never predicted; fixing this needs either more critical
  training examples for whatever pattern those 5 share, or a rule-based
  override that catches them independently of the model.
- A small rule-based pre-filter distinguishing ILS-component abbreviations
  (`OM`/`MM`/`IM`/`LOC`/`GS`/`GP`) from runway-lighting abbreviations
  (`PAPI`/`REIL`/`RAI`/`RTHL`) would resolve the recurring `navaid`/`runway`
  confusion deterministically (Classifier Example 2).
- The severity/relevance conflation the second labeling pass surfaced
  (Ranker Example 5) isn't fixed and probably shouldn't be fixed reflexively
  — it's a scope question (should ranking relevance track severity, or is
  it a separate axis?) worth a deliberate decision, not a quick patch.

**Needs new data, not new code:**
- **An independently hand-labeled test set.** The ground-truth labels are
  rule-derived (`scripts/apply_labels.py`), so the classifier metrics
  measure rule-reproduction accuracy more than real-world accuracy (see
  `docs/labeling_guide.md`, "Honest caveat"). A few hundred human-labeled
  NOTAMs with no rules involved would let the eval measure the thing it
  really wants to. This is the single highest-value data investment left.
- `admin` is still stuck at 2 examples even after the FIR-level search
  fixed `airspace`. It's survived two attempted fixes now (more airports,
  then FIR-level search), which is better evidence of a genuine feed
  limitation than the original untested claim about `airspace` was — though
  a targeted search specifically for frequency-change NOTAMs hasn't been
  tried yet either.
- The zero-shot LLM comparison predates both the FIR-level corpus expansion
  and the second severity labeling pass (see the † note in Section 5);
  rerunning it would cost another full agent pass and hasn't been
  prioritized since the specific finding it produced (a documentation bug)
  doesn't depend on corpus size or label refinement.

**Not started, larger scope:**
- **Benchmark against Notamify's public demo.** Their tool is presumably an
  interactive web app (route input, possibly file upload); a text-based web
  tool can fetch static pages but can't drive that kind of interaction, so
  this needs either a human running it manually or a browser-automation
  tool this environment doesn't have.
- **Streamlit UI polish.** `app.py` works (built per the original spec,
  smoke-tested) but is intentionally the least-invested part of the
  project, consistent with the brief's own priority order ("cut UI before
  cutting anything from evaluation").
