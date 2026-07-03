"""
QA pass on classification results so far — checks quality without requiring
a human to read every row.

1. Distribution sanity check (type x green_focus counts).
2. Self-consistency check: does the model's own reasoning text mention a
   DIFFERENT valid investor_type than the one it put in the JSON field?
   (We saw this happen during Phase 1 validation, e.g. "Google Accelerator"
   reasoning said CVC but the field said GVC.)
3. Name-pattern override check: strong GVC/CVC name signals (e.g. "Department
   of", "Ministry", known corporate brands + "Ventures") that the assigned
   type contradicts.

Run anytime — works on partial (checkpointed) results.
"""
import json
import re
from pathlib import Path
import pandas as pd

OUT = Path(__file__).parent / "output"

VALID_TYPES = ["GVC", "CVC", "IVC", "Impact_VC", "Bank_VC", "Angel_Network", "University_VC", "OTHER"]

GVC_NAME_PATTERNS = [
    r"\bDepartment of\b", r"\bMinistry\b", r"\bNational Science Foundation\b",
    r"\bEuropean (Union|Commission)\b", r"\bEuropean Innovation Council\b",
    r"\bHorizon 2020\b", r"\bInnovate UK\b", r"\b(State|Federal) Government\b",
    r"\bSovereign Wealth\b", r"\bScottish Enterprise\b", r"\bEnterprise Ireland\b",
]
CORPORATE_BRANDS = [
    "Google", "Shell", "Aramco", "Intel", "Microsoft", "IBM", "Equinor", "BP ",
    "Chevron", "ExxonMobil", "TotalEnergies", "Samsung", "Amazon", "Salesforce",
]


NEGATION_CUE = re.compile(
    r"\b(not|n't|none of|doesn't fit|does not fit|cannot be classified|isn't|"
    r"without (additional|more) (information|context)|cannot be determined|"
    r"no description (provided|is provided)|not possible to definitively|"
    r"unclear|uncertain|could be (either|any)|may be (an?|a)\b.*\bor another|"
    r"difficult to determine)\b",
    re.IGNORECASE,
)


GREEN_VC_COLLOQUIAL = re.compile(r"Green\s+Venture\s+Capital\s*\(?\s*GVC\s*\)?", re.IGNORECASE)


def reasoning_mentions_other_type(assigned_type: str, reasoning: str) -> str | None:
    """Return a conflicting type if reasoning text AFFIRMS a different type than
    assigned. Checks per-sentence: if a sentence contains a negation cue, every
    type mention in that whole sentence is ignored (it's an exclusion list, not
    an affirmation), even if the cue appears after the mention.

    Special case: the model sometimes writes 'Green Venture Capital (GVC)' as a
    colloquial abbreviation while reasoning about green_focus — that is NOT an
    affirmation of the investor_type schema's GVC (=government-backed). Those
    occurrences are stripped before matching."""
    mentioned = set()
    for sentence in re.split(r"(?<=[.;])\s+", reasoning):
        sentence = GREEN_VC_COLLOQUIAL.sub("", sentence)
        if NEGATION_CUE.search(sentence):
            continue
        for t in VALID_TYPES:
            if re.search(rf"(?<![A-Za-z_]){re.escape(t)}(?![A-Za-z_])", sentence):
                mentioned.add(t)
    mentioned.discard(assigned_type)
    if mentioned:
        return ", ".join(sorted(mentioned))
    return None


VALID_GREEN = ["GREEN_VC", "ESG_ALIGNED", "TRADITIONAL"]
GREEN_TOKEN_PATTERNS = {
    "GREEN_VC": re.compile(r"\bGREEN_VC\b|Green\s+Venture\s+Capital", re.IGNORECASE),
    "ESG_ALIGNED": re.compile(r"\bESG_ALIGNED\b|ESG[\s-]?aligned", re.IGNORECASE),
    "TRADITIONAL": re.compile(r"\bTRADITIONAL\b", re.IGNORECASE),
}


def reasoning_mentions_other_green(assigned_green: str, reasoning: str) -> str | None:
    """Same idea as reasoning_mentions_other_type, but for the green_focus axis.
    Unlike the type check, 'Green Venture Capital (GVC)' here IS a genuine
    affirmation of GREEN_VC — this axis is exactly what that phrase describes."""
    mentioned = set()
    for sentence in re.split(r"(?<=[.;])\s+", reasoning):
        if NEGATION_CUE.search(sentence):
            continue
        for g, pat in GREEN_TOKEN_PATTERNS.items():
            if pat.search(sentence):
                mentioned.add(g)
    mentioned.discard(assigned_green)
    if mentioned:
        return ", ".join(sorted(mentioned))
    return None


def name_pattern_conflict(name: str, assigned_type: str) -> str | None:
    for pat in GVC_NAME_PATTERNS:
        if re.search(pat, name, re.IGNORECASE) and assigned_type != "GVC":
            return f"name matches GVC pattern '{pat}' but assigned {assigned_type}"
    name_lower = name.lower()
    for brand in CORPORATE_BRANDS:
        brand_word = brand.strip().lower()
        if re.search(rf"\b{re.escape(brand_word)}\b", name_lower) and \
           re.search(r"\b(ventures|capital)\b", name_lower):
            if assigned_type != "CVC":
                return f"name suggests corporate venture arm ({brand.strip()}) but assigned {assigned_type}"
    return None


def main():
    results = json.loads((OUT / "all_investors_classified.json").read_text())
    df = pd.DataFrame(results)
    print(f"Total classified so far: {len(df):,}\n")

    print("--- investor_type distribution ---")
    print(df["investor_type"].value_counts().to_string())
    print("\n--- green_focus distribution ---")
    print(df["green_focus"].value_counts().to_string())

    n_parse_err = (df["investor_type"] == "PARSE_ERROR").sum()
    print(f"\nPARSE_ERROR rows: {n_parse_err:,} ({n_parse_err / len(df):.2%})")

    low_conf = df[(df["confidence_type"] == "low") | (df["confidence_green"] == "low")]
    print(f"Low-confidence rows (either field): {len(low_conf):,} ({len(low_conf) / len(df):.2%})")

    # ── Self-consistency check ──────────────────────────────────────────────
    inconsistent = []
    for r in results:
        conflict = reasoning_mentions_other_type(r["investor_type"], r.get("reasoning", ""))
        if conflict:
            inconsistent.append({**r, "conflict_with": conflict})
    print(f"\nSelf-consistency conflicts (reasoning mentions a different type than assigned): "
          f"{len(inconsistent):,} ({len(inconsistent) / len(df):.2%})")

    # ── Name-pattern override check ─────────────────────────────────────────
    name_conflicts = []
    for r in results:
        conflict = name_pattern_conflict(r["investor_name"], r["investor_type"])
        if conflict:
            name_conflicts.append({**r, "conflict_with": conflict})
    print(f"Name-pattern conflicts (strong GVC/CVC name signal contradicts assigned type): "
          f"{len(name_conflicts):,} ({len(name_conflicts) / len(df):.2%})")

    # ── Green-focus self-consistency check ──────────────────────────────────
    green_inconsistent = []
    for r in results:
        conflict = reasoning_mentions_other_green(r["green_focus"], r.get("reasoning", ""))
        if conflict:
            green_inconsistent.append({**r, "conflict_with": conflict})
    print(f"Green-focus self-consistency conflicts: "
          f"{len(green_inconsistent):,} ({len(green_inconsistent) / len(df):.2%})")

    estimated_error_rate = (len(set(r["investor_id"] for r in inconsistent + name_conflicts)) +
                             n_parse_err) / len(df)
    print(f"\nEstimated investor_type flagged-error rate (union of type checks): {estimated_error_rate:.2%}")
    estimated_green_error_rate = (len(set(r["investor_id"] for r in green_inconsistent)) +
                                   n_parse_err) / len(df)
    print(f"Estimated green_focus flagged-error rate: {estimated_green_error_rate:.2%}")

    print("\n--- Sample of flagged self-consistency conflicts ---")
    for r in inconsistent[:15]:
        print(f"{r['investor_name'][:35]:35s} | assigned={r['investor_type']:12s} | "
              f"conflict_with={r['conflict_with']:20s} | {r['reasoning'][:100]}")

    print("\n--- Sample of flagged name-pattern conflicts ---")
    for r in name_conflicts[:15]:
        print(f"{r['investor_name'][:35]:35s} | assigned={r['investor_type']:12s} | {r['conflict_with']}")

    print("\n--- Sample of flagged green-focus conflicts ---")
    for r in green_inconsistent[:15]:
        print(f"{r['investor_name'][:35]:35s} | assigned={r['green_focus']:12s} | "
              f"conflict_with={r['conflict_with']:20s} | {r['reasoning'][:100]}")

    # Save flagged rows for review
    flagged_ids = set(r["investor_id"] for r in inconsistent + name_conflicts + green_inconsistent)
    flagged_df = df[df["investor_id"].isin(flagged_ids) | (df["investor_type"] == "PARSE_ERROR")]
    flagged_df.to_csv(OUT / "qa_flagged_for_review.csv", index=False)
    print(f"\nSaved {len(flagged_df):,} flagged rows → {OUT / 'qa_flagged_for_review.csv'}")


if __name__ == "__main__":
    main()
