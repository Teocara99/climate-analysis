"""
Shared data loader — import this in every analysis script.
"""
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "Climate_Tech_2012-2026.xlsx"
COMPANY_DATA_PATH = Path(__file__).parent.parent / "PitchBook_Search_Result_Columns_2026_07_03_11_20_06.xlsx"

# Known green / climate-focused investors (add more as needed)
GREEN_INVESTOR_KEYWORDS = [
    "green", "climate", "clean", "sustainable", "sustainability",
    "energy transition", "impact", "esg", "carbon", "renewabl",
    "earth", "eco", "environmental", "nature", "solar", "wind",
    "breakthrough energy", "congruent", "lowercarbon", "energize",
    "energy capital", "world fund", "gef capital", "ecosystem",
    "blue horizon", "azolla", "cqp", "carbon13",
]


def load_deals() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="Data", header=7)

    df = df.copy()  # de-fragment before adding derived columns

    # Parse dates
    df["Deal Date"] = pd.to_datetime(df["Deal Date"], errors="coerce")
    df["Year"] = df["Deal Date"].dt.year
    df["Quarter"] = df["Deal Date"].dt.to_period("Q")

    # Deal Size is in USD millions (PitchBook default)
    df["Deal Size (USD M)"] = pd.to_numeric(df["Deal Size"], errors="coerce")

    return df


def load_companies() -> pd.DataFrame:
    df = pd.read_excel(COMPANY_DATA_PATH, sheet_name="Data", header=7)
    df = df.copy()

    # Numeric fields
    for col in ["Total Raised", "Success Probability", "IPO Probability",
                "M&A Probability", "No Exit Probability", "Valuation Estimate",
                "Last Known Valuation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    for col in ["First Financing Date", "Last Financing Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Combine active + former investors into one field for lookup
    def _combine_investors(row):
        parts = []
        for col in ["Active Investors", "Former Investors"]:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                parts.append(str(v).strip())
        return ", ".join(parts) if parts else None

    df["All Investors"] = df.apply(_combine_investors, axis=1)

    return df


def classify_investor(investor_str: str) -> str:
    """Return 'Green' if the investor string matches known green keywords, else 'General'."""
    if pd.isna(investor_str):
        return "Unknown"
    low = investor_str.lower()
    for kw in GREEN_INVESTOR_KEYWORDS:
        if kw in low:
            return "Green"
    return "General"
