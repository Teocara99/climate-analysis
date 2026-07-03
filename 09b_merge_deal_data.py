"""
09b — Merge deal-level data with FRED controls, policy shock dummies,
and investor green-mix classification.

Outputs:
  data/deals_with_external_data.csv
"""
import pandas as pd
import re
from pathlib import Path
from load_data import load_deals

DATA = Path(__file__).parent / "data"
OUT  = Path(__file__).parent / "output"


def load_investor_mix() -> dict:
    """Load classified investor table → {name: (investor_type, green_focus)}."""
    clf = pd.read_csv(OUT / "investors_classified_output.csv")
    clf = clf[~clf["investor_type"].isin(["PARSE_ERROR"])]
    return clf.set_index("investor_name")[["investor_type", "green_focus"]].to_dict("index")


def clean_name(raw: str) -> str:
    return re.sub(r"\([^)]*\)", "", str(raw)).strip()


def compute_syndicate(investor_str: str, clf_map: dict) -> dict:
    """Return green_share, syndicate_type, and type counts for one deal."""
    if pd.isna(investor_str):
        return {}
    names = [clean_name(n) for n in re.split(r",\s*", str(investor_str).replace("\n", ", "))]
    names = [n for n in names if n]

    types  = [clf_map[n]["investor_type"]  for n in names if n in clf_map]
    greens = [clf_map[n]["green_focus"]    for n in names if n in clf_map]
    n = len(types)
    if n == 0:
        return {}

    pct_green = greens.count("GREEN_VC") / n * 100
    return {
        "n_investors_classified": n,
        "pct_green":    pct_green,
        "pct_gvc":      types.count("GVC")       / n * 100,
        "pct_ivc":      types.count("IVC")        / n * 100,
        "pct_cvc":      types.count("CVC")        / n * 100,
        "pct_impact":   types.count("Impact_VC")  / n * 100,
        "syndicate_type": (
            "Pure Green"     if pct_green >= 75 else
            "Pure Non-Green" if pct_green <= 25 else
            "Mixed"
        ),
    }


def add_policy_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add post-event dummies, respecting geographic scope."""
    ps = pd.read_csv(DATA / "policy_shocks.csv", parse_dates=["date"])
    country = df["Company Country/Territory/Region"].fillna("")

    def is_us(c): return c in ("United States", "USA", "US")
    def is_eu(c): return c in (
        "Germany", "France", "Netherlands", "Sweden", "Denmark",
        "Finland", "Norway", "Spain", "Italy", "Belgium", "Austria",
        "United Kingdom", "Switzerland", "European Union"
    )

    for _, row in ps.iterrows():
        col   = "post_" + re.sub(r"[^a-z0-9]", "_", row["event_name"].lower()).strip("_")
        scope = row["geographic_scope"]
        after = df["Deal Date"] > row["date"]
        if scope == "US":
            df[col] = (after & df["Company Country/Territory/Region"].apply(is_us)).astype(int)
        elif scope == "EU":
            df[col] = (after & df["Company Country/Territory/Region"].apply(is_eu)).astype(int)
        else:  # global
            df[col] = after.astype(int)

    return df


def main():
    print("Loading deals...")
    deals = load_deals()
    deals = deals.dropna(subset=["Year", "Deal Date"]).query("2012 <= Year <= 2026")

    # ── Investor mix ───────────────────────────────────────────────────────
    print("Computing investor mix per deal...")
    clf_map = load_investor_mix()
    mix_rows = deals["Investors"].apply(lambda x: compute_syndicate(x, clf_map))
    mix_df = pd.DataFrame(mix_rows.tolist(), index=deals.index)
    deals = pd.concat([deals, mix_df], axis=1)
    deals = deals[deals["n_investors_classified"] >= 1]
    print(f"  Deals with ≥1 classified investor: {len(deals):,}")

    # ── FRED controls ─────────────────────────────────────────────────────
    fred_path = DATA / "fred_economic_controls.csv"
    if fred_path.exists():
        print("Merging FRED controls...")
        fred = pd.read_csv(fred_path)
        fred["year_month"] = pd.PeriodIndex(fred["year_month"], freq="M")
        deals["year_month"] = deals["Deal Date"].dt.to_period("M")
        deals = deals.merge(fred[["year_month","oil_price","interest_rate",
                                   "vix","gdp","jobless_claims"]],
                             on="year_month", how="left")
        print(f"  FRED coverage: {deals['vix'].notna().sum():,}/{len(deals):,} deals")
    else:
        print("  FRED data not found — run 09a first with your API key.")
        deals["year_month"] = deals["Deal Date"].dt.to_period("M")
        for col in ["oil_price","interest_rate","vix","gdp","jobless_claims"]:
            deals[col] = None

    # ── Policy shock dummies ───────────────────────────────────────────────
    print("Adding policy shock dummies...")
    deals = add_policy_dummies(deals)

    # ── Company age at deal ────────────────────────────────────────────────
    deals["company_age"] = deals["Year"] - deals["Year Founded"].apply(
        pd.to_numeric, errors="coerce")

    # ── Save ───────────────────────────────────────────────────────────────
    keep = [
        "Deal ID", "Companies", "Deal Date", "Year", "Quarter",
        "Deal Size (USD M)", "Deal Type", "VC Round",
        "Company Country/Territory/Region", "HQ Global Region",
        "company_age", "Year Founded",
        "n_investors_classified", "pct_green", "pct_gvc", "pct_ivc",
        "pct_cvc", "pct_impact", "syndicate_type",
        "oil_price", "interest_rate", "vix", "gdp", "jobless_claims",
    ] + [c for c in deals.columns if c.startswith("post_")]

    keep = [c for c in keep if c in deals.columns]
    deals[keep].to_csv(DATA / "deals_with_external_data.csv", index=False)
    print(f"\nSaved {len(deals):,} deals → data/deals_with_external_data.csv")
    print(f"Columns: {len(keep)}")
    print("\nSyndicate type distribution:")
    print(deals["syndicate_type"].value_counts().to_string())
    post_cols = [c for c in deals.columns if c.startswith("post_")]
    print("\nPolicy dummy means:")
    print(deals[post_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
