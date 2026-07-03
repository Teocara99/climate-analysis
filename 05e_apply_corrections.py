"""
Deterministic post-processing pass — fixes the systematic GVC-vs-CVC
confusion found during QA (05d_qa_classification.py) without spending more
Ollama time, for both investor_type and green_focus.

Rule: if the model's own reasoning text affirms exactly ONE specific
value different from the JSON field it produced, trust the reasoning and
override the field. (Reasoning text correctly says e.g. "a venture arm of
ABB, a multinational corporation" but the field said GVC — the affirmed
type in prose is the more reliable signal in these cases.)
Also applies the brand-name override for corporate venture arms (e.g.
Equinor Ventures, bp Ventures) the QA script flagged independently.

Safe to re-run anytime — works on partial (checkpointed) results, and can be
re-run again on the final completed file.

Outputs:
  - all_investors_classified_corrected.json
  - corrections_log.csv
"""
import json
from pathlib import Path
from importlib import import_module
import pandas as pd

OUT = Path(__file__).parent / "output"
qa = import_module("05d_qa_classification")


def main():
    results = json.loads((OUT / "all_investors_classified.json").read_text())
    corrections = []

    for r in results:
        if r["investor_type"] == "PARSE_ERROR":
            continue

        # ── investor_type ────────────────────────────────────────────────
        applied = None
        conflict = qa.reasoning_mentions_other_type(r["investor_type"], r.get("reasoning", ""))
        if conflict and "," not in conflict:  # exactly one affirmed alternative
            applied = conflict
            reason = "self_consistency"
        else:
            name_conflict = qa.name_pattern_conflict(r["investor_name"], r["investor_type"])
            if name_conflict and "corporate venture arm" in name_conflict:
                applied = "CVC"
                reason = "name_pattern_corporate_brand"

        if applied and applied != r["investor_type"]:
            corrections.append({
                "investor_id": r["investor_id"], "investor_name": r["investor_name"],
                "field": "investor_type", "old_value": r["investor_type"], "new_value": applied,
                "reason": reason,
            })
            r["investor_type"] = applied
            r["confidence_type"] = "medium"  # downgrade since this was a rule-based override
            r["reasoning"] = r.get("reasoning", "") + " [auto-corrected: type overridden from reasoning text]"

        # ── green_focus ──────────────────────────────────────────────────
        green_conflict = qa.reasoning_mentions_other_green(r["green_focus"], r.get("reasoning", ""))
        if green_conflict and "," not in green_conflict and green_conflict != r["green_focus"]:
            corrections.append({
                "investor_id": r["investor_id"], "investor_name": r["investor_name"],
                "field": "green_focus", "old_value": r["green_focus"], "new_value": green_conflict,
                "reason": "self_consistency",
            })
            r["green_focus"] = green_conflict
            r["confidence_green"] = "medium"
            r["reasoning"] = r.get("reasoning", "") + " [auto-corrected: green_focus overridden from reasoning text]"

    (OUT / "all_investors_classified_corrected.json").write_text(json.dumps(results, indent=1))
    corrections_df = pd.DataFrame(corrections)
    corrections_df.to_csv(OUT / "corrections_log.csv", index=False)

    print(f"Total rows: {len(results):,}")
    print(f"Corrections applied: {len(corrections):,} ({len(corrections) / len(results):.2%})")
    if corrections:
        for field in ["investor_type", "green_focus"]:
            field_df = corrections_df[corrections_df["field"] == field]
            if field_df.empty:
                continue
            print(f"\n--- {field} correction breakdown (old -> new) ---")
            print(field_df.groupby(["old_value", "new_value"]).size()
                  .sort_values(ascending=False).to_string())
        print("\n--- Sample corrections ---")
        for c in corrections[:15]:
            print(f"{c['investor_name'][:35]:35s} | {c['field']:14s} | "
                  f"{c['old_value']:12s} -> {c['new_value']:12s} | {c['reason']}")

    print(f"\nSaved → {OUT / 'all_investors_classified_corrected.json'}")
    print(f"Saved → {OUT / 'corrections_log.csv'}")


if __name__ == "__main__":
    main()
