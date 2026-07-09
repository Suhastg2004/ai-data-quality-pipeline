# dq_given_data/comparator.py

import json
import os
import pandas as pd


def load_results(path, model_name):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — run {model_name} first.")
        return None
    with open(path) as f:
        return json.load(f)


def print_comparison(results):
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON — OVERALL METRICS")
    print("=" * 70)
    print(f"  {'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 70)

    best_f1 = max(r["f1"] for r in results)
    for r in results:
        marker = " ← best" if r["f1"] == best_f1 else ""
        print(f"  {r['model']:<25} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} {r['tp']:>8,} {r['fp']:>8,} {r['fn']:>8,}{marker}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("  PER ANOMALY TYPE — F1 COMPARISON")
    print("=" * 70)

    all_types = set()
    for r in results:
        for row in r.get("per_type", []):
            all_types.add(row["anomaly_type"])
    all_types = sorted(all_types)

    lookup = {}
    for r in results:
        lookup[r["model"]] = {
            row["anomaly_type"]: row["f1"]
            for row in r.get("per_type", [])
        }

    model_names = [r["model"] for r in results]
    header = f"  {'Anomaly Type':<28}"
    for name in model_names:
        header += f" {name.split()[0][:10]:>10}"
    print(header)
    print("-" * 70)

    for atype in all_types:
        f1_values = [lookup.get(name, {}).get(atype) for name in model_names]
        best_val  = max((v for v in f1_values if v is not None), default=None)
        row_str   = f"  {atype:<28}"
        for val in f1_values:
            if val is None:
                row_str += f"{'—':>10}"
            elif val == best_val:
                row_str += f"{val:>9.3f}*"
            else:
                row_str += f"{val:>10.3f}"
        print(row_str)

    print("-" * 70)
    print("  * = best F1 for that type")

    overall_winner = max(results, key=lambda r: r["f1"])
    supervised   = [r for r in results if r["model"] != "Isolation Forest"]
    unsupervised = next((r for r in results if r["model"] == "Isolation Forest"), None)

    print("\n" + "=" * 70)
    print(f"  OVERALL WINNER : {overall_winner['model']}")
    print(f"  F1 Score       : {overall_winner['f1']:.4f}")
    print("=" * 70)

    if unsupervised and supervised:
        best_sup = max(supervised, key=lambda r: r["f1"])
        gap = round(best_sup["f1"] - unsupervised["f1"], 4)
        print(f"\n  KEY INSIGHT:")
        print(f"  Best supervised ({best_sup['model']}) F1 = {best_sup['f1']:.4f}")
        print(f"  Isolation Forest (no labels)       F1 = {unsupervised['f1']:.4f}")
        print(f"  Gap = {gap:.4f} — the measurable value of having labels.")
        print("\n  Isolation Forest is the only method that works with zero labels.")
        print("  Once labels accumulate, switch to a supervised model.")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AI/ML DQ PIPELINE — MODEL COMPARISON REPORT")
    print("=" * 70)

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

    print(f"\n  Models loaded: {len(results)}/4")
    if results:
        print_comparison(results)