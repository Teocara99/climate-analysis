"""
Analysis 7 — Company success outcomes vs investor green/type mix.

For each company: look up every investor in our Ollama-classified table,
compute the share of GREEN_VC, GVC, IVC, Impact_VC, etc. investors, then
correlate with success signals from the company dataset.

Outputs:
  - company_investor_mix.csv
  - success_by_green_quartile.png
  - success_prob_by_green_share.png
  - business_status_by_green_quartile.png
  - investor_type_mix_by_outcome.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import re
from pathlib import Path
from load_data import load_companies

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def clean_name(raw: str) -> str:
    return re.sub(r"\([^)]*\)", "", str(raw)).strip()


def build_company_investor_mix(companies: pd.DataFrame,
                                clf: pd.DataFrame) -> pd.DataFrame:
    """For each company, compute:
    - n_investors_matched: how many investors we have a classification for
    - pct_green:     share classified GREEN_VC
    - pct_esg:       share classified ESG_ALIGNED
    - pct_traditional: share classified TRADITIONAL
    - pct_gvc, pct_ivc, pct_cvc, pct_impact: share by investor type
    - green_majority: True if pct_green > 50%
    """
    clf_type = clf.set_index("investor_name")["investor_type"].to_dict()
    clf_green = clf.set_index("investor_name")["green_focus"].to_dict()

    rows = []
    for _, row in companies.iterrows():
        raw_inv = row["All Investors"]
        if pd.isna(raw_inv):
            rows.append({"Company ID": row["Company ID"], "n_investors_matched": 0})
            continue

        names = [clean_name(n) for n in re.split(r",\s*", str(raw_inv).replace("\n", ", "))]
        names = [n for n in names if n]

        types, greens = [], []
        for n in names:
            if n in clf_type:
                types.append(clf_type[n])
                greens.append(clf_green[n])

        n = len(types)
        if n == 0:
            rows.append({"Company ID": row["Company ID"], "n_investors_matched": 0})
            continue

        rows.append({
            "Company ID": row["Company ID"],
            "n_investors_matched": n,
            "pct_green":       greens.count("GREEN_VC") / n * 100,
            "pct_esg":         greens.count("ESG_ALIGNED") / n * 100,
            "pct_traditional": greens.count("TRADITIONAL") / n * 100,
            "pct_gvc":    types.count("GVC") / n * 100,
            "pct_ivc":    types.count("IVC") / n * 100,
            "pct_cvc":    types.count("CVC") / n * 100,
            "pct_impact": types.count("Impact_VC") / n * 100,
        })

    return pd.DataFrame(rows)


def outcome_label(row) -> str:
    """Map ownership/business status to a clean 4-way outcome."""
    own = str(row.get("Ownership Status", ""))
    biz = str(row.get("Business Status", ""))
    if "Publicly Held" in own or "IPO" in own:
        return "IPO / Public"
    if "Acquired" in own:
        return "Acquired"
    if "Out of Business" in own or "Liquidation" in biz or "Out of Business" in biz:
        return "Failed"
    return "Active / Private"


def main():
    companies = load_companies()
    clf = pd.read_csv(OUT / "investors_classified_output.csv")
    clf = clf[~clf["investor_type"].isin(["PARSE_ERROR"])]

    print(f"Companies: {len(companies):,} | Classified investors: {len(clf):,}")

    # ── Build mix ──────────────────────────────────────────────────────────
    mix = build_company_investor_mix(companies, clf)
    df = companies.merge(mix, on="Company ID", how="left")
    df = df[df["n_investors_matched"] >= 2]  # need at least 2 classified investors
    df["outcome"] = df.apply(outcome_label, axis=1)
    df["green_quartile"] = pd.qcut(df["pct_green"], q=4,
                                   labels=["Q1\n(0-25%)", "Q2\n(25-50%)",
                                           "Q3\n(50-75%)", "Q4\n(75-100%)"])
    print(f"Companies with ≥2 matched investors: {len(df):,}")
    print(df[["outcome"]].value_counts().to_string())

    df.to_csv(OUT / "company_investor_mix.csv", index=False,
              columns=["Company ID", "Companies", "Business Status", "Ownership Status",
                       "Success Class", "Success Probability", "IPO Probability",
                       "M&A Probability", "Total Raised", "outcome",
                       "n_investors_matched", "pct_green", "pct_esg",
                       "pct_traditional", "pct_gvc", "pct_ivc", "pct_cvc", "pct_impact",
                       "green_quartile"])

    # ── 1. Success probability by green investor quartile ──────────────────
    sp = df.dropna(subset=["Success Probability", "green_quartile"])
    fig, ax = plt.subplots(figsize=(9, 5))
    q_means = sp.groupby("green_quartile", observed=True)["Success Probability"].mean()
    q_counts = sp.groupby("green_quartile", observed=True).size()
    colors = ["#bdc3c7", "#85c1e9", "#27ae60", "#1a5276"]
    bars = ax.bar(q_means.index, q_means.values, color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:.1f}\n(n={q_counts[i]:,})"
                                for i, v in zip(q_means.index, q_means.values)],
                 padding=4, fontsize=9)
    ax.set_title("Avg Success Probability by Green Investor Share Quartile", fontsize=13)
    ax.set_xlabel("Green Investor Share Quartile")
    ax.set_ylabel("Avg Success Probability (%)")
    ax.set_ylim(0, 100)
    ax.axhline(sp["Success Probability"].mean(), color="red", linestyle="--",
               alpha=0.6, label=f"Overall avg ({sp['Success Probability'].mean():.1f}%)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "success_prob_by_green_quartile.png", dpi=150)
    plt.close(fig)
    print(f"Saved → success_prob_by_green_quartile.png")

    # ── 2. Business outcome distribution by green quartile ─────────────────
    outcome_order = ["IPO / Public", "Acquired", "Active / Private", "Failed"]
    outcome_colors = {"IPO / Public": "#2980b9", "Acquired": "#27ae60",
                      "Active / Private": "#95a5a6", "Failed": "#e74c3c"}
    pivot = (df.groupby(["green_quartile", "outcome"], observed=True).size()
             .unstack("outcome", fill_value=0)
             .reindex(columns=outcome_order, fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = pd.Series(0.0, index=pivot_pct.index)
    for outcome in outcome_order:
        if outcome in pivot_pct.columns:
            ax.bar(pivot_pct.index, pivot_pct[outcome], bottom=bottom,
                   label=outcome, color=outcome_colors[outcome], edgecolor="white")
            bottom += pivot_pct[outcome]
    ax.set_title("Business Outcome by Green Investor Share Quartile", fontsize=13)
    ax.set_ylabel("Share of Companies (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "business_status_by_green_quartile.png", dpi=150)
    plt.close(fig)
    print(f"Saved → business_status_by_green_quartile.png")

    # ── 3. Investor type mix by outcome ────────────────────────────────────
    type_cols = {"GVC": "#3498db", "IVC": "#2ecc71", "CVC": "#e67e22", "Impact_VC": "#9b59b6"}
    mix_by_outcome = df.groupby("outcome")[["pct_gvc", "pct_ivc", "pct_cvc", "pct_impact"]].mean()
    mix_by_outcome.columns = ["GVC", "IVC", "CVC", "Impact_VC"]
    mix_by_outcome = mix_by_outcome.reindex(outcome_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(mix_by_outcome))
    width = 0.2
    for i, (col, color) in enumerate(type_cols.items()):
        if col in mix_by_outcome.columns:
            ax.bar(x + i * width, mix_by_outcome[col], width=width,
                   label=col, color=color, edgecolor="white")
    ax.set_title("Average Investor Type Mix by Company Outcome", fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(mix_by_outcome.index)
    ax.set_ylabel("Avg Share of Investors (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.legend(title="Investor Type")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(OUT / "investor_type_mix_by_outcome.png", dpi=150)
    plt.close(fig)
    print(f"Saved → investor_type_mix_by_outcome.png")

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n--- Avg green share & success prob by outcome ---")
    print(df.groupby("outcome")[["pct_green", "pct_traditional", "Success Probability",
                                  "IPO Probability"]].mean().round(1).to_string())

    print("\n--- Success probability by green quartile ---")
    print(sp.groupby("green_quartile", observed=True)["Success Probability"]
          .agg(["mean", "median", "count"]).round(1).to_string())


if __name__ == "__main__":
    main()
