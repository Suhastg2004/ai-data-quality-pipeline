# dq_given_data/comparator.py

"""
Loads results from all four models and prints a unified comparison.

Reads:
    output/lr_results.json   (Logistic Regression)
    output/rf_results.json   (Random Forest)
    output/xgb_results.json  (XGBoost)
    output/if_results.json   (Isolation Forest)

No training happens here — just comparison of saved results.
"""

import json
import os
import pandas as pd


def load_results(path: str, model_name: str) -> dict:
    """Loads a results JSON. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — run {model_name} first.")
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison(results: list) -> None:
    """
    Prints three comparison tables:
        1. Overall metrics (Precision, Recall, F1, TP, FP, FN)
        2. Per anomaly type F1 for each model side by side
        3. Winner per anomaly type
    """

    # Table 1: Overall metrics
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON — OVERALL METRICS")
    print("=" * 70)
    print(f"  {'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 70)

    for r in results:
        marker = " ← best F1" if r["f1"] == max(x["f1"] for x in results) else ""
        print(f"  {r['model']:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r['tp']:>8,} {r['fp']:>8,} {r['fn']:>8,}{marker}")

    print("=" * 70)

    # Table 2: Per anomaly type F1 side by side
    print("\n" + "=" * 70)
    print("  PER ANOMALY TYPE — F1 COMPARISON")
    print("=" * 70)
    print("  (Higher is better | — = type not in results)")
    print()

    # Collect all anomaly types across all models
    all_types = set()
    for r in results:
        for row in r.get("per_type", []):
            all_types.add(row["anomaly_type"])
    all_types = sorted(all_types)

    # Build lookup: model → type → f1
    lookup = {}
    for r in results:
        lookup[r["model"]] = {
            row["anomaly_type"]: row["f1"]
            for row in r.get("per_type", [])
        }

    model_names = [r["model"] for r in results]

    # Header
    header = f"  {'Anomaly Type':<28}"
    for name in model_names:
        short = name.split()[0][:10]
        header += f" {short:>10}"
    print(header)
    print("-" * 70)

    for atype in all_types:
        row_str = f"  {atype:<28}"
        f1_values = []
        for name in model_names:
            val = lookup.get(name, {}).get(atype)
            f1_values.append(val)

        best_val = max((v for v in f1_values if v is not None), default=None)

        for val in f1_values:
            if val is None:
                row_str += f"{'—':>10}"
            elif val == best_val and best_val is not None:
                row_str += f"{val:>9.3f}*"   # * marks the best for this type
            else:
                row_str += f"{val:>10.3f}"
        print(row_str)

    print("-" * 70)
    print("  * = best F1 for that anomaly type")

    # Table 3: Summary — who wins where
    print("\n" + "=" * 70)
    print("  WINNER PER ANOMALY TYPE")
    print("=" * 70)

    for atype in all_types:
        best_model = None
        best_f1    = -1
        for name in model_names:
            val = lookup.get(name, {}).get(atype)
            if val is not None and val > best_f1:
                best_f1    = val
                best_model = name
        print(f"  {atype:<28} → {best_model} (F1={best_f1:.3f})")

    # Overall winner
    overall_winner = max(results, key=lambda r: r["f1"])
    print("\n" + "=" * 70)
    print(f"  OVERALL WINNER : {overall_winner['model']}")
    print(f"  F1 Score       : {overall_winner['f1']:.4f}")
    print(f"  Precision      : {overall_winner['precision']:.4f}")
    print(f"  Recall         : {overall_winner['recall']:.4f}")
    print("=" * 70)

    # Key insight
    supervised   = [r for r in results if r["model"] != "Isolation Forest"]
    unsupervised = [r for r in results if r["model"] == "Isolation Forest"]

    best_sup = max(supervised, key=lambda r: r["f1"])
    if_result = unsupervised[0] if unsupervised else None

    print("\n  KEY INSIGHT:")
    if if_result:
        gap = round(best_sup["f1"] - if_result["f1"], 4)
        print(f"  Best supervised ({best_sup['model']}) F1 = {best_sup['f1']:.4f}")
        print(f"  Isolation Forest (no labels)       F1 = {if_result['f1']:.4f}")
        print(f"  Gap = {gap:.4f} — this is the measurable value of having labels.")
        print()
        print("  Isolation Forest is the only method that works with zero labels.")
        print("  Use it on new data sources before any anomalies are confirmed.")
        print("  Once labels accumulate, switch to a supervised model.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AI/ML DQ PIPELINE — MODEL COMPARISON REPORT")
    print("=" * 70)

    # Load all four result files
    files = [
        ("output/lr_results.json",  "Logistic Regression"),
        ("output/rf_results.json",  "Random Forest"),
        ("output/xgb_results.json", "XGBoost"),
        ("output/if_results.json",  "Isolation Forest"),
    ]

    results = []
    for path, name in files:
        r = load_results(path, name)
        if r is not None:
            results.append(r)

    if len(results) == 0:
        print("  No results found. Run each model file first.")
    else:
        print(f"\n  Models loaded: {len(results)}/4")
        print_comparison(results)