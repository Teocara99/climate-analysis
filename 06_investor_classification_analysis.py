"""
Analysis 6 — Insights from Ollama-classified investor data (partial results).

Charts produced:
  1. investor_type_distribution.png   — share of each investor type
  2. green_focus_by_type.png          — green focus breakdown within each type
  3. capital_by_investor_type.png     — capital deployed (USD B) by investor type
  4. deals_by_investor_type_year.png  — deal count over time by investor type
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import re
from pathlib import Path
from load_data import load_deals

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

TYPE_COLORS = {
    "GVC": "#3498db", "IVC": "#2ecc71", "OTHER": "#95a5a6",
    "CVC": "#e67e22", "Impact_VC": "#9b59b6", "University_VC": "#1abc9c",
    "Angel_Network": "#f39c12", "Bank_VC": "#e74c3c",
}
GREEN_COLORS = {"GREEN_VC": "#27ae60", "ESG_ALIGNED": "#f39c12", "TRADITIONAL": "#bdc3c7"}


def load_classified() -> pd.DataFrame:
    df = pd.read_csv(OUT / "investors_classified_output.csv")
    df = df[df["investor_type"] != "PARSE_ERROR"]
    return df


def clean_investor_name(raw: str) -> str:
    return re.sub(r"\([^)]*\)", "", str(raw)).strip()


def main():
    clf = load_classified()
    print(f"Classified investors loaded: {len(clf):,}")

    # ── 1. Investor type distribution ─────────────────────────────────────
    type_counts = clf["investor_type"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [TYPE_COLORS.get(t, "#7f8c8d") for t in type_counts.index]
    bars = ax.bar(type_counts.index, type_counts.values, color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:,}\n({v/len(clf):.1%})" for v in type_counts.values],
                 padding=4, fontsize=9)
    ax.set_title("Climate Tech Investor Type Distribution (classified investors)", fontsize=13)
    ax.set_ylabel("Number of Investors")
    ax.set_ylim(0, type_counts.max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "investor_type_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved → investor_type_distribution.png")

    # ── 2. Green focus breakdown by investor type ──────────────────────────
    green_by_type = (
        clf.groupby(["investor_type", "green_focus"]).size()
        .unstack("green_focus", fill_value=0)
    )
    green_by_type = green_by_type.div(green_by_type.sum(axis=1), axis=0) * 100
    green_by_type = green_by_type.reindex(
        type_counts.index,
        columns=["GREEN_VC", "ESG_ALIGNED", "TRADITIONAL"]
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = pd.Series(0.0, index=green_by_type.index)
    for col in ["GREEN_VC", "ESG_ALIGNED", "TRADITIONAL"]:
        if col in green_by_type.columns:
            ax.bar(green_by_type.index, green_by_type[col], bottom=bottom,
                   label=col, color=GREEN_COLORS[col], edgecolor="white")
            bottom += green_by_type[col]
    ax.set_title("Green Focus Breakdown by Investor Type (%)", fontsize=13)
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "green_focus_by_type.png", dpi=150)
    plt.close(fig)
    print(f"Saved → green_focus_by_type.png")

    # ── 3 & 4. Join classification → deals ────────────────────────────────
    deals = load_deals()
    deals = deals.dropna(subset=["Year"]).query("2012 <= Year <= 2026")
    deals = deals.dropna(subset=["Investors"])

    clf_map = clf.set_index("investor_name")[["investor_type", "green_focus"]].to_dict("index")

    rows = []
    for _, row in deals.iterrows():
        investors_str = str(row["Investors"]).replace("\n", ", ")
        for raw in re.split(r",\s*", investors_str):
            name = clean_investor_name(raw)
            if name and name in clf_map:
                rows.append({
                    "Year": row["Year"],
                    "Deal Size (USD M)": row["Deal Size (USD M)"],
                    "investor_type": clf_map[name]["investor_type"],
                    "green_focus": clf_map[name]["green_focus"],
                })

    inv_deals = pd.DataFrame(rows)
    print(f"Deal-investor pairs matched: {len(inv_deals):,}")

    # ── 3. Capital by investor type ────────────────────────────────────────
    cap_by_type = (
        inv_deals.groupby("investor_type")["Deal Size (USD M)"]
        .sum().sort_values(ascending=False) / 1_000
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [TYPE_COLORS.get(t, "#7f8c8d") for t in cap_by_type.index]
    bars = ax.bar(cap_by_type.index, cap_by_type.values, color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"${v:.0f}B" for v in cap_by_type.values], padding=4, fontsize=9)
    ax.set_title("Capital Deployed by Investor Type (USD B, deals with known size)", fontsize=13)
    ax.set_ylabel("Capital (USD B)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}B"))
    ax.set_ylim(0, cap_by_type.max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "capital_by_investor_type.png", dpi=150)
    plt.close(fig)
    print(f"Saved → capital_by_investor_type.png")

    # ── 4. Deal count over time by investor type ───────────────────────────
    top_types = ["GVC", "IVC", "OTHER", "CVC", "Impact_VC"]
    deal_ct = (
        inv_deals[inv_deals["investor_type"].isin(top_types)]
        .groupby(["Year", "investor_type"]).size()
        .unstack("investor_type", fill_value=0)
        .reindex(columns=top_types, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for t in top_types:
        if t in deal_ct.columns:
            ax.plot(deal_ct.index, deal_ct[t], marker="o", label=t,
                    color=TYPE_COLORS.get(t, "grey"), linewidth=2)
    ax.set_title("Deal Participations Over Time by Investor Type (2012–2026)", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Deal Participations")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(title="Investor Type")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "deals_by_investor_type_year.png", dpi=150)
    plt.close(fig)
    print(f"Saved → deals_by_investor_type_year.png")

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n--- Capital deployed by investor type (USD B) ---")
    print(cap_by_type.to_string())
    print("\n--- Green focus % by type ---")
    print(green_by_type[["GREEN_VC","ESG_ALIGNED","TRADITIONAL"]].round(1).to_string())


if __name__ == "__main__":
    main()
