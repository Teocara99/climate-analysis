"""
09a — Download FRED economic controls and validate policy shocks.

Usage:
  python3 09a_download_external_data.py <FRED_API_KEY>

FRED series downloaded (monthly, 2010-2026):
  DCOILWTICO : WTI crude oil price
  FEDFUNDS   : Federal funds rate
  VIXCLS     : VIX volatility index
  GDP        : US GDP (quarterly → interpolated monthly)
  ICSA       : Initial jobless claims

Outputs:
  data/fred_economic_controls.csv
  data/policy_shocks.csv  (already created — just validated here)
"""
import sys
import pandas as pd
from fredapi import Fred
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

FRED_SERIES = {
    "DCOILWTICO": "oil_price",
    "FEDFUNDS":   "interest_rate",
    "VIXCLS":     "vix",
    "GDP":        "gdp",
    "ICSA":       "jobless_claims",
}
START, END = "2010-01-01", "2026-06-30"


def download_fred(api_key: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    frames = {}
    for series_id, col in FRED_SERIES.items():
        print(f"  Downloading {series_id} ({col})...")
        s = fred.get_series(series_id, observation_start=START, observation_end=END)
        s.index = pd.to_datetime(s.index)
        # Resample everything to month-end
        s = s.resample("ME").last()
        if series_id == "GDP":
            s = s.resample("ME").interpolate(method="time")  # quarterly → monthly
        frames[col] = s

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df = df.reset_index()
    df["year_month"] = df["date"].dt.to_period("M")
    df.to_csv(DATA / "fred_economic_controls.csv", index=False)
    print(f"  Saved {len(df)} rows → data/fred_economic_controls.csv")
    return df


def validate_policy_shocks():
    ps = pd.read_csv(DATA / "policy_shocks.csv", parse_dates=["date"])
    print(f"  Policy shocks loaded: {len(ps)} events")
    print(ps[["date", "event_name", "direction"]].to_string(index=False))
    return ps


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 09a_download_external_data.py <FRED_API_KEY>")
        sys.exit(1)
    api_key = sys.argv[1]
    print("Downloading FRED economic controls...")
    df = download_fred(api_key)
    print(df.tail(3).to_string())
    print("\nValidating policy shocks...")
    validate_policy_shocks()
    print("\nDone. Run 09b_merge_deal_data.py next.")
