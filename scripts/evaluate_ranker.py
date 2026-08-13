import json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import load_models, load_raw_notams, process_notams
from src.ranker import rank_notams

SCENARIOS_PATH  = "data/labeled/ranking_scenarios.json"
K_VALUES = [5, 10]


def precision_at_k(ranked_ids, relevant_ids, k):
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def r_precision(ranked_ids, relevant_ids):
    """
    Precision@R where R = |relevant_ids| -- a standard IR metric (see e.g.
    Manning/Raghavan/Schutze) that sidesteps the fixed-k ceiling problem
    documented in failure_analysis.md (Ranker Example 2): precision@5 can
    never exceed n_relevant/5 for a scenario with fewer than 5 relevant
    items, which penalizes small-airport scenarios for the metric's choice
    of k rather than the ranker's actual quality. R-precision instead asks
    "of the top R candidates, how many are relevant?", which has a ceiling
    of 1.0 for every scenario regardless of how many relevant items exist.
    """
    r = len(relevant_ids)
    if r == 0:
        return None
    return precision_at_k(ranked_ids, relevant_ids, r)


def main():
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    cat_model, sev_model = load_models()
    base_processed = process_notams(load_raw_notams(), cat_model, sev_model)

    print(f"{'Scenario':6} {'Route':25} {'#Relevant':10} {'P@5':6} {'P@10':6} {'R-Prec':6}")
    print("-" * 68)

    totals = {k: [] for k in K_VALUES}
    r_prec_totals = []
    per_scenario_results = []
    for scenario in scenarios:
        route = scenario["route_airports"]
        relevant = set(scenario["relevant_ids"])

        processed = [dict(n) for n in base_processed]
        ranked = rank_notams(processed, route)
        ranked_ids = [n["id"] for n in ranked]

        scores = {}
        for k in K_VALUES:
            p = precision_at_k(ranked_ids, relevant, k)
            scores[f"p@{k}"] = round(p, 3)
            totals[k].append(p)

        rp = r_precision(ranked_ids, relevant)
        scores["r_precision"] = round(rp, 3) if rp is not None else None
        if rp is not None:
            r_prec_totals.append(rp)

        per_scenario_results.append({
            "scenario_id": scenario["scenario_id"],
            "route_airports": route,
            "n_relevant": len(relevant),
            **scores,
            "top_5_ids": ranked_ids[:5],
        })

        route_str = ",".join(route)
        rp_str = f"{scores['r_precision']:.3f}" if scores["r_precision"] is not None else "n/a"
        print(f"{scenario['scenario_id']:6} {route_str:25} {len(relevant):10} "
              f"{scores['p@5']:<6} {scores['p@10']:<6} {rp_str:6}")

    print("-" * 68)
    for k in K_VALUES:
        avg = sum(totals[k]) / len(totals[k])
        print(f"Mean precision@{k} across {len(scenarios)} scenarios: {avg:.3f}")
    if r_prec_totals:
        avg_rp = sum(r_prec_totals) / len(r_prec_totals)
        print(f"Mean R-precision across {len(scenarios)} scenarios: {avg_rp:.3f}  "
              f"(precision@R, R=n_relevant -- avoids the fixed-k ceiling for small scenarios)")

    with open("ranker_eval_results.json", "w") as f:
        json.dump(per_scenario_results, f, indent=2)
    print("\nDetailed per-scenario results -> ranker_eval_results.json")


if __name__ == "__main__":
    main()
