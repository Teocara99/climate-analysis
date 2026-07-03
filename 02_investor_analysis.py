"""
Analysis 2 — Green vs General investor behaviour.

Strategy:
  - Explode the "Investors" column (semi-colon separated) so each row = one investor participation.
  - Classify each investor as Green or General based on name keywords.
  - Compare deal count, capital deployed, deal stage mix, and geography.

Outputs (in output/):
  - investor_type_summary.csv
  - top_green_investors.csv
  - top_general_investors.csv
  - plots: investor_type_deals.png, investor_type_capital.png,
           green_vs_general_by_year.png, investor_stage_mix.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from load_data import load_deals, classify_investor
from pathlib import Path
import re

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def _clean_investor_name(raw: str) -> str:
    """Strip parenthetical role notes, e.g. 'Sequoia Capital(Lead)' → 'Sequoia Capital'."""
    return re.sub(r"\([^)]*\)", "", raw).strip()


def explode_investors(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form df: one row per (deal, investor)."""
    inv = df[["Deal ID", "Year", "Quarter", "Deal Size (USD M)",
              "VC Round", "Deal Type", "HQ Global Region",
              "Company Country/Territory/Region", "Investors"]].copy()
    inv = inv.dropna(subset=["Investors"])
    # PitchBook separates investors with ", " inside a single cell; also newlines
    inv["Investors"] = inv["Investors"].str.replace("\n", ", ", regex=False)
    inv = inv.assign(Investor=inv["Investors"].str.split(r",\s*")).explode("Investor")
    inv["Investor"] = inv["Investor"].apply(_clean_investor_name)
    inv = inv[inv["Investor"] != ""]
    inv["Investor Type"] = inv["Investor"].apply(classify_investor)
    return inv


def main():
    df = load_deals()
    df = df.dropna(subset=["Year"]).query("2012 <= Year <= 2026")

    inv = explode_investors(df)

    # ── 1. High-level summary ──────────────────────────────────────────────
    summary = (
        inv.groupby("Investor Type")
        .agg(
            Participations=("Investor", "count"),
            Unique_Investors=("Investor", "nunique"),
            Unique_Deals=("Deal ID", "nunique"),
            Total_Capital_USD_M=("Deal Size (USD M)", "sum"),
            Avg_Deal_Size_USD_M=("Deal Size (USD M)", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "investor_type_summary.csv", index=False)

    # ── 2. Top investors by type ───────────────────────────────────────────
    for itype in ["Green", "General"]:
        top = (
            inv[inv["Investor Type"] == itype]
            .groupby("Investor")
            .agg(Participations=("Deal ID", "count"),
                 Capital_USD_M=("Deal Size (USD M)", "sum"))
            .sort_values("Participations", ascending=False)
            .head(30)
            .reset_index()
        )
        fname = OUT / f"top_{itype.lower()}_investors.csv"
        top.to_csv(fname, index=False)
        print(f"\nTop 10 {itype} investors:\n", top.head(10).to_string(index=False))

    # ── 3. Green vs General deal count by year ─────────────────────────────
    by_year_type = (
        inv.groupby(["Year", "Investor Type"])["Deal ID"]
        .nunique()
        .reset_index(name="Unique Deals")
    )
    pivot = by_year_type.pivot(index="Year", columns="Investor Type", values="Unique Deals").fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = pivot.index
    width = 0.38
    offsets = {"Green": -width / 2, "General": width / 2}
    colors = {"Green": "#27ae60", "General": "#2980b9", "Unknown": "#bdc3c7"}
    for col in pivot.columns:
        offset = offsets.get(col, 0)
        ax.bar(x + offset, pivot[col], width=width, label=col, color=colors.get(col, "grey"), edgecolor="white")
    ax.set_title("Climate Tech Deals per Year — Green vs General Investors", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique Deals")
    ax.legend(title="Investor Type")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "green_vs_general_by_year.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {OUT / 'green_vs_general_by_year.png'}")

    # ── 4. Stage mix comparison (VC Round) ────────────────────────────────
    stage_mix = (
        inv[inv["Investor Type"].isin(["Green", "General"])]
        .dropna(subset=["VC Round"])
        .groupby(["Investor Type", "VC Round"])["Deal ID"]
        .nunique()
        .reset_index(name="Deals")
    )
    # Normalise to % within each investor type
    totals = stage_mix.groupby("Investor Type")["Deals"].transform("sum")
    stage_mix["Share (%)"] = (stage_mix["Deals"] / totals * 100).round(1)
    # Keep top 8 rounds by total volume
    top_rounds = (
        stage_mix.groupby("VC Round")["Deals"].sum()
        .nlargest(8).index.tolist()
    )
    stage_mix = stage_mix[stage_mix["VC Round"].isin(top_rounds)]
    pivot_stage = stage_mix.pivot(index="VC Round", columns="Investor Type", values="Share (%)").fillna(0)
    pivot_stage = pivot_stage.sort_values("Green", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = range(len(pivot_stage))
    h = 0.35
    for i, col in enumerate(pivot_stage.columns):
        ax.barh([yi + (i - 0.5) * h for yi in y], pivot_stage[col],
                height=h, label=col, color=colors.get(col, "grey"), edgecolor="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot_stage.index)
    ax.set_xlabel("Share of Deals (%)")
    ax.set_title("Deal Stage Mix — Green vs General Investors", fontsize=13)
    ax.legend(title="Investor Type")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "investor_stage_mix.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {OUT / 'investor_stage_mix.png'}")

    # ── 5. Geographic focus ───────────────────────────────────────────────
    geo = (
        inv[inv["Investor Type"].isin(["Green", "General"])]
        .dropna(subset=["HQ Global Region"])
        .groupby(["Investor Type", "HQ Global Region"])["Deal ID"]
        .nunique()
        .reset_index(name="Deals")
    )
    totals = geo.groupby("Investor Type")["Deals"].transform("sum")
    geo["Share (%)"] = (geo["Deals"] / totals * 100).round(1)
    geo_out = geo.sort_values(["Investor Type", "Share (%)"], ascending=[True, False])
    geo_out.to_csv(OUT / "investor_geo_mix.csv", index=False)
    print(f"\nGeographic mix saved → {OUT / 'investor_geo_mix.csv'}")


if __name__ == "__main__":
    main()
