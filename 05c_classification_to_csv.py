"""
Phase 3 — Convert Ollama classification results (JSON) into the final CSV,
plus summary statistics by investor_type and green_focus.

Works on partial results too (the JSON checkpoint updates every 1,000
investors while 05b is still running), so this can be re-run anytime to see
progress so far.

Outputs:
  - investors_classified_output.csv
"""
import json
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "output"


def main():
    results_file = OUT / "all_investors_classified.json"
    if not results_file.exists():
        print(f"No results yet at {results_file}")
        return

    results = json.loads(results_file.read_text())
    df = pd.DataFrame(results)
    df = df[["investor_id", "investor_name", "investor_type", "green_focus",
              "confidence_type", "confidence_green", "reasoning"]]

    df.to_csv(OUT / "investors_classified_output.csv", index=False)
    print(f"Classified so far: {len(df):,} investors")
    print(f"Saved → {OUT / 'investors_classified_output.csv'}")

    print("\n--- Investor Type distribution ---")
    print(df["investor_type"].value_counts().to_string())

    print("\n--- Green Focus distribution ---")
    print(df["green_focus"].value_counts().to_string())

    print("\n--- Cross-tab: Investor Type x Green Focus ---")
    print(pd.crosstab(df["investor_type"], df["green_focus"]).to_string())

    n_errors = (df["investor_type"] == "PARSE_ERROR").sum()
    if n_errors:
        print(f"\nPARSE_ERROR count: {n_errors:,} ({n_errors / len(df):.1%})")


if __name__ == "__main__":
    main()
