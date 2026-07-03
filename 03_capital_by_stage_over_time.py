"""
Analysis 3 — Capital deployed over time, split by investment stage
(Seed, Early Stage, Late Stage, Angel), with deal count overlaid as a line.

Outputs:
  - capital_by_stage_year.csv
  - plots: capital_by_stage_over_time.png
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

STAGE_MAP = {
    "Seed Round": "Seed",
    "Early Stage VC": "Early Stage",
    "Later Stage VC": "Late Stage",
    "Angel (individual)": "Angel",
}
STAGE_ORDER = ["Angel", "Seed", "Early Stage", "Late Stage"]
STAGE_COLORS = {
    "Angel": "#9b59b6",
    "Seed": "#2ecc71",
    "Early Stage": "#3498db",
    "Late Stage": "#e67e22",
}


def main():
    df = load_deals()

    df = df.dropna(subset=["Year", "Deal Date"])
    df = df[df["Year"].between(2012, 2026)]
    df = df[df["Deal Type"].isin(STAGE_MAP)]
    df["Stage"] = df["Deal Type"].map(STAGE_MAP)

    # ── Capital by year x stage (USD M) ────────────────────────────────────
    capital = (
        df.groupby(["Year", "Stage"])["Deal Size (USD M)"]
        .sum()
        .unstack("Stage")
        .reindex(columns=STAGE_ORDER, fill_value=0)
        .fillna(0)
    )

    # ── Deal count by year (across the 4 stages) ───────────────────────────
    deal_count = df.groupby("Year").size().reindex(capital.index, fill_value=0)

    out_df = capital.copy()
    out_df["Deal Count"] = deal_count
    out_df.to_csv(OUT / "capital_by_stage_year.csv")
    print(out_df.to_string())

    # ── Plot: stacked bars (capital, USD B) + line (deal count) ────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    bottom = pd.Series(0.0, index=capital.index)
    for stage in STAGE_ORDER:
        values = capital[stage] / 1_000  # USD B
        ax.bar(capital.index, values, bottom=bottom, label=stage,
               color=STAGE_COLORS[stage], edgecolor="white")
        bottom += values

    ax.set_title("Climate Tech Capital Deployed by Stage, with Deal Count (2012–2026)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Capital Deployed (USD B)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax.twinx()
    ax2.plot(deal_count.index, deal_count.values, color="black", marker="o",
              linewidth=2, label="Deal Count")
    ax2.set_ylabel("Number of Deals")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Combined legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    fig.savefig(OUT / "capital_by_stage_over_time.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {OUT / 'capital_by_stage_over_time.png'}")


if __name__ == "__main__":
    main()
