"""
11b_novelty_analysis.py — Novelty × Investor Mix × Outcomes
============================================================
Phase 2 of 2: merge novelty scores with the deals/investor/company dataset
and run the substantive analyses. Requires 11a_novelty_compute.py to have
been run first (reads output/novelty/novelty_scores.csv).

ANALYSES
--------
1. Novelty map          — UMAP 2-D scatter (cluster × novelty × sector)
2. Novelty distribution — by investor type (pure green / mixed / non-green)
3. Novelty × success    — logit: does novelty predict exit? + investor interaction
4. Novelty × funding    — does novelty predict total capital raised?
5. Novelty by era       — did political shocks change which novelty tier got funded?
6. Cluster profiles     — avg novelty, exit rate, investor mix per HDBSCAN cluster

OUTPUTS (output/novelty/)
--------------------------
novelty_map.png
novelty_by_investor_type.png
novelty_regression_table.csv
novelty_by_era.png
novelty_cluster_profiles.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path
from load_data import load_companies

OUT = Path(__file__).parent / "output" / "novelty"
DATA = Path(__file__).parent / "data"

PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}
ERA_COLORS = {
    "Pre-Trump": "#95a5a6", "Trump 1": "#e74c3c",
    "Biden": "#3498db",     "Trump 2": "#c0392b",
}
ERA_BOUNDS = [
    ("Pre-Trump", pd.Timestamp("2012-01-01"), pd.Timestamp("2016-11-07")),
    ("Trump 1",   pd.Timestamp("2016-11-08"), pd.Timestamp("2021-01-19")),
    ("Biden",     pd.Timestamp("2021-01-20"), pd.Timestamp("2024-11-04")),
    ("Trump 2",   pd.Timestamp("2024-11-05"), pd.Timestamp("2026-12-31")),
]


def load_data() -> pd.DataFrame:
    nov = pd.read_csv(OUT / "novelty_scores.csv")

    # detect primary model from model_comparison.txt
    comp_txt = (OUT / "model_comparison.txt").read_text()
    primary = "bge" if "Primary model:               BGE" in comp_txt else "specter"
    nov["novelty"]  = nov[f"{primary}_novelty"]
    nov["density"]  = nov[f"{primary}_density"]
    nov["cluster"]  = nov[f"{primary}_cluster"]
    nov["keywords"] = nov[f"{primary}_keywords"]
    nov["umap_x"]   = nov[f"umap_{primary}_x"]
    nov["umap_y"]   = nov[f"umap_{primary}_y"]
    print(f"Using primary model: {primary.upper()}")

    mix = pd.read_csv(OUT.parent / "company_investor_mix.csv")

    # join by company name (both come from PitchBook, names should match)
    df = nov.merge(mix, left_on="name", right_on="Companies", how="inner")
    print(f"Companies matched to investor mix: {len(df):,}")

    companies = load_companies()
    def comp_outcome(row):
        own = str(row.get("Ownership Status",""))
        biz = str(row.get("Business Status",""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired" in own: return "acquired"
        if "Out of Business" in own or "Liquidation" in biz: return "failed"
        return "operating"
    companies["outcome"] = companies.apply(comp_outcome, axis=1)
    comp_map = companies.set_index("Companies")[["outcome","Total Raised"]].to_dict("index")

    df["outcome"]      = df["name"].map(lambda n: comp_map.get(n,{}).get("outcome","operating"))
    df["total_raised"] = df["name"].map(lambda n: comp_map.get(n,{}).get("Total Raised",np.nan))
    df["total_raised"] = pd.to_numeric(df["total_raised"], errors="coerce")
    df["exited"]       = df["outcome"].isin(["ipo","acquired"]).astype(int)
    df["failed"]       = (df["outcome"]=="failed").astype(int)

    # novelty quartile
    df["novelty_q"] = pd.qcut(df["novelty"], q=4,
                               labels=["Q1\n(least novel)","Q2","Q3","Q4\n(most novel)"])

    # load deals for political era
    deals = pd.read_csv(DATA / "deals_with_external_data.csv", parse_dates=["Deal Date"])
    era_map = {}
    for _, row in deals.iterrows():
        for name, start, end in ERA_BOUNDS:
            if start <= row["Deal Date"] <= end:
                era_map[row["Companies"]] = name
                break
    df["era"] = df["name"].map(era_map).fillna("Unknown")

    return df, primary


def plot_novelty_map(df: pd.DataFrame, primary: str):
    print("Plotting novelty map...")
    fig, ax = plt.subplots(figsize=(14, 10))

    clusters = sorted(df["cluster"].unique())
    palette  = plt.cm.tab20.colors
    noise    = df[df["cluster"] == -1]
    ax.scatter(noise["umap_x"], noise["umap_y"], s=8, color="#dddddd",
               alpha=0.4, label="noise", zorder=1)

    for i, c in enumerate([cl for cl in clusters if cl != -1]):
        sub = df[df["cluster"] == c]
        sizes = 20 + 200 * sub["novelty"]
        label = sub["keywords"].iloc[0][:35]
        ax.scatter(sub["umap_x"], sub["umap_y"], s=sizes,
                  color=palette[i % len(palette)], alpha=0.7,
                  edgecolors="white", linewidths=0.4,
                  label=f"{c}: {label}", zorder=2)

    ax.set_title(f"Climate Tech Novelty Map — {primary.upper()} + UMAP + HDBSCAN\n"
                 f"(dot size = novelty score; colour = cluster; n={len(df):,})",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP dimension 1"); ax.set_ylabel("UMAP dimension 2")
    ax.legend(fontsize=6, loc="best", ncol=2, markerscale=0.8)
    plt.tight_layout()
    fig.savefig(OUT / "novelty_map.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → novelty_map.png")


def plot_novelty_by_investor_type(df: pd.DataFrame):
    print("Plotting novelty by investor type...")
    groups = ["Pure Green", "Mixed", "Pure Non-Green"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Novelty Score by Investor Mix", fontsize=13, fontweight="bold")

    # 1. violin
    ax = axes[0]
    parts = ax.violinplot(
        [df[df["syndicate_type"]==g]["novelty"].dropna() for g in groups],
        positions=range(len(groups)), showmedians=True
    )
    for i, (pc, g) in enumerate(zip(parts["bodies"], groups)):
        pc.set_facecolor(PALETTE[g]); pc.set_alpha(0.7)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Novelty Score"); ax.set_title("1. Distribution (violin)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 2. mean novelty per quartile × syndicate
    ax = axes[1]
    x = np.arange(4); w = 0.25
    q_labels = ["Q1\n(least novel)","Q2","Q3","Q4\n(most novel)"]
    for j, grp in enumerate(groups):
        sub = df[df["syndicate_type"]==grp]
        means = [sub[sub["novelty_q"]==q]["exited"].mean()*100
                 for q in q_labels]
        ax.bar(x + j*w, means, w, label=grp, color=PALETTE[grp],
               edgecolor="white", alpha=0.9)
    ax.set_xticks(x+w); ax.set_xticklabels(q_labels, fontsize=8)
    ax.set_ylabel("Exit Rate (%)"); ax.set_title("2. Exit Rate by Novelty Quartile")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    # 3. avg novelty by syndicate type
    ax = axes[2]
    means = [df[df["syndicate_type"]==g]["novelty"].mean() for g in groups]
    sems  = [df[df["syndicate_type"]==g]["novelty"].sem()  for g in groups]
    colors = [PALETTE[g] for g in groups]
    bars = ax.bar(groups, means, color=colors, edgecolor="white", alpha=0.9)
    ax.errorbar(range(len(groups)), means, yerr=sems, fmt="none",
               color="black", capsize=5, linewidth=1.5)
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=3, fontsize=9)
    ax.set_ylabel("Avg Novelty Score"); ax.set_title("3. Avg Novelty by Investor Mix")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT / "novelty_by_investor_type.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → novelty_by_investor_type.png")


def run_novelty_regression(df: pd.DataFrame):
    print("Running novelty regressions...")
    df = df.copy()
    df["novelty_z"]   = (df["novelty"] - df["novelty"].mean()) / df["novelty"].std()
    df["Pure_Green"]  = (df["syndicate_type"]=="Pure Green").astype(int)
    df["Pure_NG"]     = (df["syndicate_type"]=="Pure Non-Green").astype(int)
    df["log_raised"]  = np.log1p(df["total_raised"].fillna(0))

    results = []
    formulas = {
        "Exit ~ Novelty":
            "exited ~ novelty_z + Pure_Green + Pure_NG",
        "Exit ~ Novelty + Interaction":
            "exited ~ novelty_z * Pure_Green + novelty_z * Pure_NG",
        "Log(Raised) ~ Novelty":
            "log_raised ~ novelty_z + Pure_Green + Pure_NG",
    }
    for label, formula in formulas.items():
        try:
            out_col = formula.split("~")[0].strip().split()[0]
            if out_col in ("exited","failed"):
                m = smf.logit(formula, data=df.dropna(subset=["novelty_z"])).fit(disp=False)
            else:
                m = smf.ols(formula, data=df.dropna(subset=["novelty_z"])).fit()
            tbl = m.summary2().tables[1][["Coef.","Std.Err.","P>|z|" if hasattr(m,"prsquared") else "P>|t|"]].copy()
            tbl.columns = ["coef","se","p"]
            tbl["model"] = label
            tbl["n"] = int(m.nobs)
            results.append(tbl)
            print(f"  [{label}] n={int(m.nobs):,}")
            key = [v for v in tbl.index if "novelty" in v.lower()]
            for v in key:
                stars = "***" if tbl.loc[v,"p"]<0.001 else "**" if tbl.loc[v,"p"]<0.01 else "*" if tbl.loc[v,"p"]<0.05 else ""
                print(f"    {v}: coef={tbl.loc[v,'coef']:+.4f}  p={tbl.loc[v,'p']:.4f} {stars}")
        except Exception as e:
            print(f"  [{label}] failed: {e}")

    if results:
        pd.concat(results).to_csv(OUT / "novelty_regression_table.csv")
        print(f"  Saved → novelty_regression_table.csv")


def plot_novelty_by_era(df: pd.DataFrame):
    print("Plotting novelty by political era...")
    era_order = ["Pre-Trump","Trump 1","Biden","Trump 2"]
    df_era = df[df["era"].isin(era_order)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Novelty Score by Political Era", fontsize=12, fontweight="bold")

    # avg novelty per era
    ax = axes[0]
    means = [df_era[df_era["era"]==e]["novelty"].mean() for e in era_order]
    sems  = [df_era[df_era["era"]==e]["novelty"].sem()  for e in era_order]
    colors = [ERA_COLORS[e] for e in era_order]
    bars = ax.bar(era_order, means, color=colors, edgecolor="white", alpha=0.9)
    ax.errorbar(range(len(era_order)), means, yerr=sems,
               fmt="none", color="black", capsize=5)
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=3, fontsize=9)
    ax.set_ylabel("Avg Novelty of Funded Companies")
    ax.set_title("1. Avg Novelty by Political Era\n(companies funded in each era)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # novelty quartile funding shift
    ax = axes[1]
    q_labels = ["Q1\n(least novel)","Q2","Q3","Q4\n(most novel)"]
    x = np.arange(len(era_order)); w = 0.22
    for j, q in enumerate(q_labels):
        share = [df_era[df_era["era"]==e]["novelty_q"].value_counts(normalize=True).get(q,0)*100
                 for e in era_order]
        ax.bar(x + j*w, share, w, label=q.replace("\n"," "),
               edgecolor="white", alpha=0.85)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(era_order)
    ax.set_ylabel("Share of funded companies (%)")
    ax.set_title("2. Novelty Quartile Mix by Political Era")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    plt.tight_layout()
    fig.savefig(OUT / "novelty_by_era.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → novelty_by_era.png")


def plot_cluster_profiles(df: pd.DataFrame):
    print("Plotting cluster profiles...")
    clusters = df[df["cluster"] != -1].groupby("cluster").agg(
        n=("novelty","count"),
        avg_novelty=("novelty","mean"),
        exit_rate=("exited","mean"),
        pct_green=("Pure_Green","mean"),
        keywords=("keywords","first"),
    ).reset_index()
    clusters = clusters[clusters["n"] >= 10].sort_values("avg_novelty", ascending=False)
    clusters["exit_rate"] *= 100
    clusters["pct_green"] *= 100
    clusters["label"] = clusters["keywords"].str[:30]

    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(clusters)*0.35+2)), sharey=True)
    fig.suptitle("HDBSCAN Cluster Profiles\n(sorted by avg novelty, top clusters)",
                 fontsize=12, fontweight="bold")

    for ax, col, title, color in [
        (axes[0], "avg_novelty",  "Avg Novelty Score", "#9b59b6"),
        (axes[1], "exit_rate",    "Exit Rate (%)",      "#27ae60"),
        (axes[2], "pct_green",    "% Pure Green Investors", "#3498db"),
    ]:
        ax.barh(clusters["label"], clusters[col], color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(OUT / "novelty_cluster_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → novelty_cluster_profiles.png")


def main():
    df, primary = load_data()
    print(f"\nDataset: {len(df):,} companies matched")
    print(f"Novelty range: {df['novelty'].min():.3f} – {df['novelty'].max():.3f}  "
          f"mean={df['novelty'].mean():.3f}")

    plot_novelty_map(df, primary)
    plot_novelty_by_investor_type(df)
    run_novelty_regression(df)
    plot_novelty_by_era(df)
    plot_cluster_profiles(df)
    print("\nAll done. Outputs in output/novelty/")


if __name__ == "__main__":
    main()
