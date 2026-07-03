"""
09e — Political cycle analysis: Trump 1 → Biden → Trump 2

Defines four political eras and tests whether climate tech exits and
investor composition follow US political cycles.

Charts:
  1. Exit rate by political era × syndicate type (bar chart)
  2. Trump 2016 vs Trump 2024 coefficient comparison (forest plot)
  3. Syndicate type composition shift across political eras (stacked bar)
  4. Full timeline: exit rate & syndicate mix over time with era shading
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from load_data import load_companies

OUT  = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

ERA_BOUNDS = [
    ("Pre-Trump",  pd.Timestamp("2012-01-01"), pd.Timestamp("2016-11-07"), "#95a5a6"),
    ("Trump 1",    pd.Timestamp("2016-11-08"), pd.Timestamp("2021-01-19"), "#e74c3c"),
    ("Biden",      pd.Timestamp("2021-01-20"), pd.Timestamp("2024-11-04"), "#3498db"),
    ("Trump 2",    pd.Timestamp("2024-11-05"), pd.Timestamp("2026-12-31"), "#c0392b"),
]
ERA_COLORS = {e[0]: e[3] for e in ERA_BOUNDS}

PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}


def load_data() -> pd.DataFrame:
    deals = pd.read_csv(DATA / "deals_with_external_data.csv", parse_dates=["Deal Date"])
    companies = load_companies()

    def outcome(row):
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired" in own: return "acquired"
        if "Out of Business" in own or "Liquidation" in biz or "Out of Business" in biz:
            return "failed"
        return "operating"

    companies["outcome"] = companies.apply(outcome, axis=1)
    comp_map = companies.set_index("Companies")["outcome"].to_dict()
    deals["outcome"] = deals["Companies"].map(comp_map).fillna("operating")
    deals["exited"]  = deals["outcome"].isin(["ipo", "acquired"]).astype(int)
    deals["failed"]  = (deals["outcome"] == "failed").astype(int)
    deals["survived"]= deals["outcome"].isin(["ipo","acquired","operating"]).astype(int)

    # Assign political era
    def assign_era(d):
        for name, start, end, _ in ERA_BOUNDS:
            if start <= d <= end:
                return name
        return "Unknown"
    deals["era"] = deals["Deal Date"].apply(assign_era)

    deals["Pure_Green"]     = (deals["syndicate_type"] == "Pure Green").astype(int)
    deals["Pure_Non_Green"] = (deals["syndicate_type"] == "Pure Non-Green").astype(int)
    deals["inv_year"]  = deals["Year"].astype(float).astype(int).astype(str)
    deals["region"]    = deals["HQ Global Region"].fillna("Unknown")
    deals["company_age_z"] = (
        (deals["company_age"] - deals["company_age"].mean()) /
        deals["company_age"].std()
    )
    for col in ["oil_price","interest_rate","vix"]:
        if col in deals.columns and deals[col].notna().any():
            m, s = deals[col].mean(), deals[col].std()
            if s > 0: deals[f"{col}_z"] = (deals[col] - m) / s

    deals = deals.dropna(subset=["syndicate_type","exited","era","Year"])
    deals = deals[deals["era"] != "Unknown"]
    return deals


def run_model(formula, df):
    try:
        m = smf.logit(formula, data=df.dropna(subset=["company_age"])).fit(disp=False, maxiter=200)
        return m
    except Exception as e:
        print(f"  Model failed: {e}")
        return None


def main():
    df = load_data()
    print(f"Deals: {len(df):,}")
    print(df.groupby("era")[["exited","syndicate_type"]].agg(
        {"exited": ["count","mean"], "syndicate_type": lambda x: (x=="Pure Green").mean()}
    ).round(3))

    fred_available = "oil_price_z" in df.columns and df["oil_price_z"].notna().any()
    fred_z = " + oil_price_z + interest_rate_z + vix_z" if fred_available else ""
    base   = f"company_age_z + C(region){fred_z}"

    # Models for Trump comparison — both dummies in same model
    trump_dummies = [c for c in ["post_trump_election_2016","post_biden_inauguration",
                                  "post_trump_election_2024"] if c in df.columns]
    trump_str = " + ".join(trump_dummies)
    f_trump = f"exited ~ Pure_Green + Pure_Non_Green + {trump_str} + {base}"
    m_trump = run_model(f_trump, df)

    # Era-level model (era fixed effects)
    f_era = f"exited ~ Pure_Green + Pure_Non_Green + C(era) + {base}"
    m_era = run_model(f_era, df)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Political Cycle Analysis: Climate Tech Exits & Investor Composition\n"
                 "Trump 1 (2016) → Biden (2021) → Trump 2 (2024)",
                 fontsize=13, fontweight="bold", y=1.01)

    era_order = ["Pre-Trump", "Trump 1", "Biden", "Trump 2"]

    # ── 1. Exit rate by era × syndicate type ─────────────────────────────
    ax = axes[0, 0]
    era_syn = df.groupby(["era","syndicate_type"])["exited"].agg(["mean","count"]).reset_index()
    era_syn.columns = ["era","syndicate_type","exit_rate","n"]
    era_syn["exit_rate"] *= 100
    era_syn = era_syn[era_syn["era"].isin(era_order)]

    x = np.arange(len(era_order))
    width = 0.25
    for j, grp in enumerate(["Pure Green","Mixed","Pure Non-Green"]):
        sub = era_syn[era_syn["syndicate_type"]==grp].set_index("era").reindex(era_order)
        bars = ax.bar(x + j*width, sub["exit_rate"].fillna(0), width,
                     label=grp, color=PALETTE[grp], edgecolor="white", alpha=0.9)
        for bar, (_, row) in zip(bars, sub.iterrows()):
            if pd.notna(row["exit_rate"]) and row["n"] > 10:
                ax.text(bar.get_x()+bar.get_width()/2,
                       bar.get_height()+0.1, f"{row['exit_rate']:.1f}%",
                       ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(era_order, fontsize=10)
    ax.set_ylabel("Exit Rate (%)")
    ax.set_title("1. Exit Rate by Political Era × Syndicate Type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Era background shading
    for i, era in enumerate(era_order):
        ax.axvspan(i-0.4, i+0.8, alpha=0.04,
                  color=ERA_COLORS.get(era,"grey"))

    # ── 2. Trump 2016 vs Trump 2024 coefficient comparison ────────────────
    ax = axes[0, 1]
    trump_vars = {
        "post_trump_election_2016": "Trump Election\n2016 (n=9,537)",
        "post_biden_inauguration":  "Biden\nInauguration 2021 (n=6,398)",
        "post_trump_election_2024": "Trump Election\n2024 (n=1,537)",
    }
    if m_trump is not None:
        params = m_trump.params
        conf   = m_trump.conf_int()
        pvals  = m_trump.pvalues
        ys = list(range(len(trump_vars)))
        for i, (var, label) in enumerate(trump_vars.items()):
            if var not in params: continue
            c   = params[var]
            lo, hi = conf.loc[var]
            p   = pvals[var]
            color = "#e74c3c" if p < 0.05 else "#bdc3c7"
            ax.barh(i, c, color=color, alpha=0.8, edgecolor="white", height=0.5)
            ax.errorbar(c, i, xerr=[[c-lo],[hi-c]],
                       fmt="none", color="black", capsize=5, linewidth=1.5)
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            ax.text(max(hi,abs(c))*np.sign(c) + 0.02*(1 if c>=0 else -1),
                   i, f" {c:+.3f} ({stars})", va="center", fontsize=9,
                   color="#e74c3c" if p<0.05 else "grey")

        ax.set_yticks(ys)
        ax.set_yticklabels(list(trump_vars.values()), fontsize=9)
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Logit Coefficient (effect on exit probability)\nred = significant p<0.05")
        ax.set_title("2. Trump 2016 vs Trump 2024 Election Effect\n(all in same model, US deals only)",
                    fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_xlim(-0.4, 0.8)
        n_trump2 = (df["post_trump_election_2024"] == 1).sum() if "post_trump_election_2024" in df.columns else 0
        ax.text(0.02, 0.04, f"⚠ Trump 2024: only {n_trump2:,} deals post-election (limited signal)",
               transform=ax.transAxes, fontsize=8, color="orange",
               bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", ec="orange", alpha=0.8))

    # ── 3. Syndicate composition shift across eras ────────────────────────
    ax = axes[1, 0]
    comp = df.groupby("era")["syndicate_type"].value_counts(normalize=True).unstack(fill_value=0)*100
    comp = comp.reindex(era_order).reindex(columns=["Pure Green","Mixed","Pure Non-Green"], fill_value=0)
    n_per_era = df.groupby("era").size().reindex(era_order)

    bottom = np.zeros(len(era_order))
    for grp in ["Pure Green","Mixed","Pure Non-Green"]:
        vals = comp[grp].values
        bars = ax.bar(era_order, vals, bottom=bottom, label=grp,
                     color=PALETTE[grp], edgecolor="white", alpha=0.9)
        for bar, val, era in zip(bars, vals, era_order):
            if val > 5:
                ax.text(bar.get_x()+bar.get_width()/2,
                       bottom[era_order.index(era)] + val/2,
                       f"{val:.0f}%", ha="center", va="center",
                       fontsize=9, color="white", fontweight="bold")
        bottom += vals

    for i, (era, n) in enumerate(n_per_era.items()):
        ax.text(i, 102, f"n={n:,}", ha="center", fontsize=8, color="grey")

    ax.set_ylabel("Share of Deals (%)")
    ax.set_title("3. Syndicate Type Composition by Political Era\n(shift in investor mix over political cycles)",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── 4. Timeline: annual exit rate + era shading ───────────────────────
    ax  = axes[1, 1]
    ax2 = ax.twinx()

    annual = df.groupby(["Year","syndicate_type"])["exited"].mean().unstack(fill_value=np.nan)*100
    annual_n = df.groupby("Year").size()

    for grp in ["Pure Green","Mixed","Pure Non-Green"]:
        if grp in annual.columns:
            ax.plot(annual.index, annual[grp], marker="o", markersize=5,
                   color=PALETTE[grp], linewidth=2, label=grp, alpha=0.9)

    ax2.bar(annual_n.index, annual_n.values, alpha=0.12, color="steelblue", width=0.8)
    ax2.set_ylabel("Deal Count", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue")

    # Era shading
    for name, start, end, color in ERA_BOUNDS:
        y_s = max(start.year, int(df["Year"].min()))
        y_e = min(end.year, int(df["Year"].max()))
        if y_s <= y_e:
            ax.axvspan(y_s-0.5, y_e+0.5, alpha=0.08, color=color, label=f"_{name}")
            mid = (y_s + y_e) / 2
            ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 15,
                   name, ha="center", fontsize=8, color=color, fontweight="bold", va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("Exit Rate (%)")
    ax.set_title("4. Annual Exit Rate by Syndicate Type + Political Eras",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linestyle="--", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    out_path = OUT / "political_cycle_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # ── Print comparison table ────────────────────────────────────────────
    print("\n=== POLITICAL ERA SUMMARY ===")
    summary = df.groupby("era").agg(
        n=("exited","count"),
        exit_rate=("exited", lambda x: f"{x.mean()*100:.1f}%"),
        failure_rate=("failed", lambda x: f"{x.mean()*100:.1f}%"),
        pct_pure_green=("Pure_Green", lambda x: f"{x.mean()*100:.1f}%"),
        pct_pure_nong=("Pure_Non_Green", lambda x: f"{x.mean()*100:.1f}%"),
    ).reindex(era_order)
    print(summary.to_string())

    if m_trump is not None:
        print("\n=== TRUMP 2016 vs TRUMP 2024 REGRESSION ===")
        for var in trump_vars:
            if var in m_trump.params:
                c = m_trump.params[var]
                p = m_trump.pvalues[var]
                ci = m_trump.conf_int().loc[var]
                print(f"  {var:40s}: coef={c:+.4f}  p={p:.4f}  95%CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")


if __name__ == "__main__":
    main()
