"""
Analysis 1 — Investment volume & capital deployed over time.

Outputs:
  - deals_by_year.csv
  - capital_by_year.csv
  - plots: deals_over_time.png, capital_over_time.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from load_data import load_deals
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def main():
    df = load_deals()

    # --- filter to complete years with a deal date ---
    df = df.dropna(subset=["Year", "Deal Date"])
    df = df[df["Year"].between(2012, 2026)]

    # ── 1. Deal count by year ──────────────────────────────────────────────
    deals_by_year = (
        df.groupby("Year")
        .size()
        .reset_index(name="Deal Count")
    )
    deals_by_year.to_csv(OUT / "deals_by_year.csv", index=False)
    print(deals_by_year.to_string(index=False))

    # ── 2. Capital deployed by year (USD M) ────────────────────────────────
    capital_by_year = (
        df.groupby("Year")["Deal Size (USD M)"]
        .agg(["sum", "mean", "median", "count"])
        .rename(columns={"sum": "Total (USD M)", "mean": "Avg (USD M)",
                         "median": "Median (USD M)", "count": "Deals with Size"})
        .reset_index()
    )
    capital_by_year.to_csv(OUT / "capital_by_year.csv", index=False)
    print("\n", capital_by_year.to_string(index=False))

    # ── 3. Plot deals per year ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(deals_by_year["Year"], deals_by_year["Deal Count"], color="#2ecc71", edgecolor="white")
    ax.set_title("Climate Tech Deal Count by Year (2012–2026)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Deals")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "deals_over_time.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {OUT / 'deals_over_time.png'}")

    # ── 4. Plot capital by year ────────────────────────────────────────────
    cap = capital_by_year.dropna(subset=["Total (USD M)"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(cap["Year"], cap["Total (USD M)"] / 1_000, color="#3498db", edgecolor="white")
    ax.set_title("Climate Tech Total Capital Deployed by Year (2012–2026)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Capital (USD B)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "capital_over_time.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {OUT / 'capital_over_time.png'}")


if __name__ == "__main__":
    main()
