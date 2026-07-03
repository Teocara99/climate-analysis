"""
Analysis 4 — Full list of unique investors present in the dataset.

Outputs:
  - unique_investors.csv (one row per investor, with participation/deal counts)
"""
from load_data import load_deals
from importlib import import_module
from pathlib import Path
import re

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

explode_investors = import_module("02_investor_analysis").explode_investors


def extract_investor_websites(df) -> dict:
    """Parse the 'Investors Websites' column ('Name (site), Name (site), ...')
    into a {investor name: website} lookup. Each entry always ends in '(site)',
    so splitting right after a closing paren+comma reliably separates entries
    even when names themselves contain parentheses."""
    websites = {}
    for text in df["Investors Websites"].dropna():
        text = text.replace("\n", ", ")
        for entry in re.split(r"(?<=\)),\s*", text):
            m = re.match(r"^(.*)\s\(([^()]+)\)$", entry)
            if not m:
                continue
            name, site = m.group(1).strip(), m.group(2).strip()
            websites.setdefault(name, site)
    return websites


def main():
    df = load_deals()
    df = df.dropna(subset=["Year"]).query("2012 <= Year <= 2026")

    inv = explode_investors(df)
    websites = extract_investor_websites(df)

    unique_investors = (
        inv.groupby("Investor")
        .agg(
            Participations=("Deal ID", "count"),
            Unique_Deals=("Deal ID", "nunique"),
            Total_Capital_USD_M=("Deal Size (USD M)", "sum"),
        )
        .sort_values("Participations", ascending=False)
        .reset_index()
    )
    unique_investors["Website"] = unique_investors["Investor"].map(websites)

    unique_investors.to_csv(OUT / "unique_investors.csv", index=False)
    n_with_site = unique_investors["Website"].notna().sum()
    print(f"Total unique investors: {len(unique_investors):,}")
    print(f"Investors with a known website: {n_with_site:,} "
          f"({n_with_site / len(unique_investors):.1%})")
    print(unique_investors.head(20).to_string(index=False))
    print(f"\nSaved → {OUT / 'unique_investors.csv'}")


if __name__ == "__main__":
    main()
