"""
09f — US-only interaction model: syndicate type × political era

Tests whether Mixed syndicates had significantly higher exit rates during
Trump 1 compared to Pure Green — the "fire-sale exit" hypothesis:
  Mixed = green deal flow + M&A networks → best positioned for fast exits
  Pure Green = mission-driven, patient capital → resisted selling
  Pure Non-Green = fewer climate deals to exit → lower volume

Reference category: Mixed syndicate, Pre-Trump era.

Interaction terms to watch:
  Pure_Green:post_trump_2016   → negative = Pure Green underperformed Mixed during Trump 1
  Pure_Non_Green:post_trump_2016 → tests whether traditional VCs outperformed during Trump 1

Outputs:
  output/us_interaction_model.png
  output/us_interaction_table.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from itertools import product
from pathlib import Path
from load_data import load_companies

OUT  = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}
ERA_COLORS = {
    "Pre-Trump": "#95a5a6",
    "Trump 1":   "#e74c3c",
    "Biden":     "#3498db",
    "Trump 2":   "#c0392b",
}
ERA_ORDER = ["Pre-Trump", "Trump 1", "Biden", "Trump 2"]
ERA_BOUNDS = [
    ("Pre-Trump", pd.Timestamp("2012-01-01"), pd.Timestamp("2016-11-07")),
    ("Trump 1",   pd.Timestamp("2016-11-08"), pd.Timestamp("2021-01-19")),
    ("Biden",     pd.Timestamp("2021-01-20"), pd.Timestamp("2024-11-04")),
    ("Trump 2",   pd.Timestamp("2024-11-05"), pd.Timestamp("2026-12-31")),
]


def load_data() -> pd.DataFrame:
    deals = pd.read_csv(DATA / "deals_with_external_data.csv", parse_dates=["Deal Date"])

    # US only
    deals = deals[deals["Company Country/Territory/Region"] == "United States"].copy()

    companies = load_companies()
    def outcome(row):
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired" in own: return "acquired"
        if "Out of Business" in own or "Liquidation" in biz or \
           "Out of Business" in biz: return "failed"
        return "operating"
    companies["outcome"] = companies.apply(outcome, axis=1)
    comp_map = companies.set_index("Companies")["outcome"].to_dict()

    deals["outcome"]  = deals["Companies"].map(comp_map).fillna("operating")
    deals["exited"]   = deals["outcome"].isin(["ipo","acquired"]).astype(int)
    deals["failed"]   = (deals["outcome"] == "failed").astype(int)

    def assign_era(d):
        for name, start, end in ERA_BOUNDS:
            if start <= d <= end: return name
        return "Unknown"
    deals["era"] = deals["Deal Date"].apply(assign_era)
    deals = deals[deals["era"] != "Unknown"]

    # Dummies — Mixed is the reference category
    deals["Pure_Green"]     = (deals["syndicate_type"] == "Pure Green").astype(int)
    deals["Pure_Non_Green"] = (deals["syndicate_type"] == "Pure Non-Green").astype(int)
    deals["inv_year"] = deals["Year"].astype(float).astype(int).astype(str)
    deals["company_age_z"] = (
        (deals["company_age"] - deals["company_age"].mean()) /
        deals["company_age"].std()
    )
    for col in ["oil_price","interest_rate","vix"]:
        if col in deals.columns and deals[col].notna().any():
            m, s = deals[col].mean(), deals[col].std()
            if s > 0: deals[f"{col}_z"] = (deals[col] - m) / s
    fred_z = " + oil_price_z + interest_rate_z + vix_z" \
             if "oil_price_z" in deals.columns and deals["oil_price_z"].notna().any() else ""

    deals = deals.dropna(subset=["syndicate_type","exited","Year","company_age"])
    return deals, fred_z


def run_interaction_model(df, fred_z):
    shocks = [c for c in ["post_trump_election_2016","post_biden_inauguration",
                           "post_trump_election_2024"] if c in df.columns]
    shock_str = " + ".join(shocks)
    inter_str = " + ".join(
        f"{s}:Pure_Green + {s}:Pure_Non_Green" for s in shocks
    )
    formula = (
        f"exited ~ Pure_Green + Pure_Non_Green"
        f" + {shock_str}"
        f" + {inter_str}"
        f" + company_age_z{fred_z}"
    )
    print(f"\nFormula:\n  {formula}\n")
    try:
        model = smf.logit(formula, data=df).fit(disp=False, maxiter=300)
        return model, shocks
    except Exception as e:
        print(f"ERROR: {e}")
        return None, shocks


def compute_predicted_exit_rates(model, df, shocks):
    """Predicted probability for each syndicate × political era combination."""
    records = []
    for era, pg, png in product(
        ERA_ORDER,
        [(0,"Mixed"),(1,"Pure Green")],
        [(0,None)]
    ):
        pg_val, pg_name = pg
        # Skip if pg_name doesn't match pure/non-pure logic
        pass

    rows = []
    for grp, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
        for era in ERA_ORDER:
            row = {"Pure_Green": pg, "Pure_Non_Green": png, "company_age_z": 0}
            # Set all shock dummies based on era
            era_start = {"Pre-Trump": 0, "Trump 1": 1, "Biden": 1, "Trump 2": 1}
            row["post_trump_election_2016"] = 1 if era in ["Trump 1","Biden","Trump 2"] else 0
            row["post_biden_inauguration"]  = 1 if era in ["Biden","Trump 2"] else 0
            row["post_trump_election_2024"] = 1 if era == "Trump 2" else 0
            for col in ["oil_price_z","interest_rate_z","vix_z"]:
                if col in df.columns: row[col] = 0
            rows.append({**row, "group": grp, "era": era})

    pred_df = pd.DataFrame(rows)
    try:
        pred_df["pred_exit"] = model.predict(pred_df) * 100
    except Exception as e:
        print(f"Prediction error: {e}")
        pred_df["pred_exit"] = np.nan
    return pred_df


def main():
    df, fred_z = load_data()
    print(f"US deals: {len(df):,}")
    print(f"Syndicate distribution:\n{df['syndicate_type'].value_counts()}")
    print(f"\nRaw exit rates by era × syndicate:")
    print(df.groupby(["era","syndicate_type"])["exited"].agg(["mean","count"])
          .round(3).to_string())

    model, shocks = run_interaction_model(df, fred_z)
    if model is None:
        return

    # Save table
    tbl = model.summary2().tables[1].copy()
    tbl.to_csv(OUT / "us_interaction_table.csv")

    # Key coefficients
    params = model.params
    conf   = model.conf_int()
    pvals  = model.pvalues

    print("\n=== KEY INTERACTION RESULTS ===")
    key_vars = [v for v in params.index if any(
        x in v for x in ["Pure_Green","Pure_Non_Green","post_trump","post_biden"]
    )]
    for v in key_vars:
        stars = "***" if pvals[v]<0.001 else "**" if pvals[v]<0.01 else "*" if pvals[v]<0.05 else ""
        print(f"  {v:55s}: {params[v]:+.4f}  p={pvals[v]:.4f}  {stars}")

    pred_df = compute_predicted_exit_rates(model, df, shocks)

    # ── 4-panel chart ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "US Deals Only — Interaction Model: Syndicate Type × Political Era\n"
        "Key question: Did Mixed syndicates have higher exits during Trump 1 (fire-sale hypothesis)?",
        fontsize=12, fontweight="bold", y=1.01
    )

    # ── 1. Predicted exit probabilities (model output) ────────────────────
    ax = axes[0, 0]
    x = np.arange(len(ERA_ORDER))
    width = 0.25
    for j, grp in enumerate(["Pure Green","Mixed","Pure Non-Green"]):
        sub = pred_df[pred_df["group"]==grp].set_index("era").reindex(ERA_ORDER)
        bars = ax.bar(x + j*width, sub["pred_exit"].fillna(0), width,
                     label=grp, color=PALETTE[grp], edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, sub["pred_exit"].fillna(0)):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ERA_ORDER)
    ax.set_ylabel("Predicted Exit Probability (%)")
    ax.set_title("1. Predicted Exit Rates (model-adjusted)\ncontrols held at mean; reference = Mixed, Pre-Trump",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── 2. Interaction coefficients forest plot ───────────────────────────
    ax = axes[0, 1]
    interaction_map = {
        "post_trump_election_2016:Pure_Green":
            "Pure Green × Trump 2016\n(vs Mixed × Trump 2016)",
        "post_trump_election_2016:Pure_Non_Green":
            "Pure Non-Green × Trump 2016\n(vs Mixed × Trump 2016)",
        "post_biden_inauguration:Pure_Green":
            "Pure Green × Biden\n(vs Mixed × Biden)",
        "post_biden_inauguration:Pure_Non_Green":
            "Pure Non-Green × Biden\n(vs Mixed × Biden)",
        "post_trump_election_2024:Pure_Green":
            "Pure Green × Trump 2024\n(vs Mixed × Trump 2024)",
        "post_trump_election_2024:Pure_Non_Green":
            "Pure Non-Green × Trump 2024\n(vs Mixed × Trump 2024)",
    }
    # Also check reversed order (statsmodels can store either way)
    reversed_map = {v.replace("post_","").replace(":Pure","_Pure").replace("election_",""):k
                    for k,v in interaction_map.items()}

    ys = list(range(len(interaction_map)))
    plotted = []
    for i, (var, label) in enumerate(interaction_map.items()):
        # Try both orderings
        actual_var = var if var in params.index else \
                     ":".join(var.split(":")[::-1]) if ":".join(var.split(":")[::-1]) in params.index else None
        if actual_var is None:
            plotted.append((label, None, None, None, None))
            continue
        c   = params[actual_var]
        lo, hi = conf.loc[actual_var]
        p   = pvals[actual_var]
        plotted.append((label, c, lo, hi, p))

    for i, (label, c, lo, hi, p) in enumerate(plotted):
        if c is None:
            ax.barh(i, 0, color="#eeeeee", height=0.4)
            ax.text(0.01, i, "not estimable", va="center", fontsize=8, color="grey")
            continue
        color = "#e74c3c" if p < 0.05 else ("#3498db" if p < 0.15 else "#bdc3c7")
        ax.barh(i, c, color=color, alpha=0.8, edgecolor="white", height=0.5)
        ax.errorbar(c, i, xerr=[[c-lo],[hi-c]],
                   fmt="none", color="black", capsize=4, linewidth=1.5)
        stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else \
                "(†)" if p<0.15 else "ns"
        side = hi+0.02 if c >= 0 else lo-0.02
        ax.text(side, i, f"{c:+.3f} {stars}", va="center", fontsize=8.5,
               ha="left" if c >= 0 else "right",
               color="#e74c3c" if p<0.05 else ("darkorange" if p<0.15 else "grey"))

    ax.set_yticks(ys)
    ax.set_yticklabels([l for l,*_ in plotted], fontsize=8.5)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Interaction Coefficient\n(relative to Mixed syndicate in same era)")
    ax.set_title("2. Interaction Effects: Syndicate Type × Political Era\nred=p<0.05, orange=p<0.15",
                fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(-1.5, 1.5)

    # ── 3. Gap between Mixed and Pure Green: Trump1 vs Biden vs Trump2 ────
    ax = axes[1, 0]
    raw = df.groupby(["era","syndicate_type"])["exited"].mean().unstack()*100
    raw = raw.reindex(ERA_ORDER)
    gap_pg  = (raw.get("Mixed",pd.Series()) - raw.get("Pure Green",pd.Series())).fillna(0)
    gap_png = (raw.get("Mixed",pd.Series()) - raw.get("Pure Non-Green",pd.Series())).fillna(0)

    x = np.arange(len(ERA_ORDER))
    w = 0.35
    b1 = ax.bar(x-w/2, gap_pg.reindex(ERA_ORDER).fillna(0), w,
               label="Mixed minus Pure Green", color="#27ae60", alpha=0.8, edgecolor="white")
    b2 = ax.bar(x+w/2, gap_png.reindex(ERA_ORDER).fillna(0), w,
               label="Mixed minus Pure Non-Green", color="#7f8c8d", alpha=0.8, edgecolor="white")
    ax.bar_label(b1, fmt="{:.1f}pp", padding=3, fontsize=8.5)
    ax.bar_label(b2, fmt="{:.1f}pp", padding=3, fontsize=8.5)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ERA_ORDER)
    ax.set_ylabel("Exit Rate Gap (percentage points)")
    ax.set_title("3. Mixed Portfolio Advantage by Political Era\n(positive = Mixed outperforms; raw rates)",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}pp"))

    # ── 4. Annotated timeline (US only) ──────────────────────────────────
    ax = axes[1, 1]
    annual = df.groupby(["Year","syndicate_type"])["exited"].mean().unstack()*100

    for grp in ["Pure Green","Mixed","Pure Non-Green"]:
        if grp in annual.columns:
            ax.plot(annual.index, annual[grp], marker="o", markersize=5,
                   color=PALETTE[grp], linewidth=2.5, label=grp, alpha=0.9)

    # Political era shading
    era_spans = [
        ("Pre-Trump", 2012, 2016.9),
        ("Trump 1",   2016.9, 2021.05),
        ("Biden",     2021.05, 2024.85),
        ("Trump 2",   2024.85, 2026.5),
    ]
    for name, ys, ye in era_spans:
        ax.axvspan(ys, ye, alpha=0.07, color=ERA_COLORS[name])
        ax.text((ys+ye)/2, 18, name, ha="center", fontsize=8,
               color=ERA_COLORS[name], fontweight="bold", va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("Exit Rate (%, US deals only)")
    ax.set_title("4. Annual Exit Rate Timeline — US Deals\n(political eras shaded)",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(2011.5, 2026.5)

    plt.tight_layout()
    out_path = OUT / "us_interaction_model.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
