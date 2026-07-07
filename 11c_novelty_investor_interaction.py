"""
11c — Novelty × Investor Type interaction deep-dive.

Hypotheses tested:
  H1. Pure Non-Green × High Novelty → more exits (fire-sale of novel assets)
  H2. Pure Non-Green × High Novelty → more failures too (risk amplification)
  H3. Pure Green × High Novelty → same exit rate regardless (patient capital,
      novelty doesn't accelerate exit pressure)
  H4. The novel companies that Non-Green investors exit are in specific sectors
      (hard tech, deep tech niches) vs the ones they fail with

Charts:
  1. Outcome decomposition: exit / fail / operating by novelty quartile × syndicate
  2. Dual regression forest plot: novelty × syndicate → exit AND → failure
  3. Cluster heatmap: exit & failure rates per novelty cluster × syndicate
  4. Novelty–outcome scatter: which sectors sit in the high-novelty / non-green zone?
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
from load_data import load_companies

OUT  = Path(__file__).parent / "output" / "novelty"
DATA = Path(__file__).parent / "data"

PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}
OUTCOME_COLORS = {
    "Exit (IPO/M&A)":   "#2980b9",
    "Active / Private": "#95a5a6",
    "Failed":           "#e74c3c",
}


def load_data() -> pd.DataFrame:
    nov = pd.read_csv(OUT / "novelty_scores.csv")
    mix = pd.read_csv(OUT.parent / "company_investor_mix.csv")
    df  = nov.merge(mix, left_on="name", right_on="Companies", how="inner")

    df["syndicate_type"] = pd.cut(
        df["pct_green"], bins=[-1, 25, 75, 101],
        labels=["Pure Non-Green", "Mixed", "Pure Green"]
    ).astype(str)

    companies = load_companies()
    def outcome(row):
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own: return "Exit (IPO/M&A)"
        if "Acquired" in own:                       return "Exit (IPO/M&A)"
        if "Out of Business" in own or "Liquidation" in biz: return "Failed"
        return "Active / Private"
    companies["outcome_label"] = companies.apply(outcome, axis=1)
    comp_map = companies.drop_duplicates("Companies").set_index("Companies")["outcome_label"].to_dict()
    df["outcome_label"] = df["name"].map(comp_map).fillna("Active / Private")
    df["exited"]  = (df["outcome_label"] == "Exit (IPO/M&A)").astype(int)
    df["failed"]  = (df["outcome_label"] == "Failed").astype(int)

    df["novelty_q"] = pd.qcut(df["specter_novelty"], q=4,
                               labels=["Q1\n(least novel)", "Q2", "Q3", "Q4\n(most novel)"])
    df["novelty_z"] = (df["specter_novelty"] - df["specter_novelty"].mean()) / df["specter_novelty"].std()
    df["Pure_Green"] = (df["syndicate_type"] == "Pure Green").astype(int)
    df["Pure_NG"]    = (df["syndicate_type"] == "Pure Non-Green").astype(int)

    return df


def main():
    df = load_data()
    groups   = ["Pure Green", "Mixed", "Pure Non-Green"]
    q_labels = ["Q1\n(least novel)", "Q2", "Q3", "Q4\n(most novel)"]
    print(f"Dataset: {len(df):,} companies")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Novelty × Investor Type Interaction\n"
        "H1: Non-Green + High Novelty → exits (fire-sale)  |  "
        "H2: Non-Green + High Novelty → failures too  |  "
        "H3: Green = patient capital regardless of novelty",
        fontsize=12, fontweight="bold", y=1.01
    )

    # ── 1. Outcome decomposition: exit/fail/active by quartile × syndicate ──
    ax = axes[0, 0]
    x = np.arange(len(q_labels))
    w = 0.25
    for j, grp in enumerate(groups):
        exit_rates = []
        fail_rates = []
        for q in q_labels:
            sub = df[(df["syndicate_type"] == grp) & (df["novelty_q"] == q)]
            exit_rates.append(sub["exited"].mean() * 100 if len(sub) > 5 else np.nan)
            fail_rates.append(sub["failed"].mean() * 100 if len(sub) > 5 else np.nan)

        ax.plot(x + j*w, exit_rates, marker="o", color=PALETTE[grp],
               linewidth=2.5, markersize=8, label=f"{grp} — Exit", linestyle="-")
        ax.plot(x + j*w, fail_rates, marker="s", color=PALETTE[grp],
               linewidth=2.5, markersize=8, label=f"{grp} — Fail", linestyle="--", alpha=0.6)

        for xi, (er, fr) in enumerate(zip(exit_rates, fail_rates)):
            if not np.isnan(er):
                ax.text(xi + j*w, er + 0.3, f"{er:.1f}%", ha="center", fontsize=7, color=PALETTE[grp])

    ax.set_xticks(x + w); ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_title("1. Exit (solid) & Failure (dashed) by Novelty Quartile\n× Syndicate Type",
                fontsize=10, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    # clean legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=PALETTE[g], lw=2, marker="o", label=g) for g in groups]
    handles += [Line2D([0],[0], color="black", lw=2, ls="-", label="Exit"),
                Line2D([0],[0], color="black", lw=2, ls="--", label="Failure", alpha=0.6)]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    # ── 2. Forest plot: novelty × syndicate → exit AND → failure ────────────
    ax = axes[0, 1]
    formulas = {
        "exit →":    "exited ~ novelty_z * Pure_Green + novelty_z * Pure_NG",
        "failure →": "failed  ~ novelty_z * Pure_Green + novelty_z * Pure_NG",
    }
    interaction_vars = {
        "novelty_z:Pure_Green": "Novelty × Pure Green\n(vs Mixed)",
        "novelty_z:Pure_NG":    "Novelty × Pure Non-Green\n(vs Mixed)",
        "novelty_z":            "Novelty main effect\n(Mixed baseline)",
    }
    model_colors = {"exit →": "#2980b9", "failure →": "#e74c3c"}
    offsets = {"exit →": -0.15, "failure →": 0.15}
    ys = {v: i for i, v in enumerate(interaction_vars)}

    for outcome_label, formula in formulas.items():
        try:
            m = smf.logit(formula, data=df).fit(disp=False, maxiter=300)
            params = m.params; conf = m.conf_int(); pvals = m.pvalues
            for var, label in interaction_vars.items():
                if var not in params: continue
                c = params[var]; lo, hi = conf.loc[var]; p = pvals[var]
                y = ys[var] + offsets[outcome_label]
                ax.errorbar(c, y, xerr=[[c-lo],[hi-c]], fmt="o",
                           color=model_colors[outcome_label], markersize=8,
                           capsize=5, linewidth=2, label=outcome_label if var==list(interaction_vars)[0] else "")
                stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "(†)" if p<0.15 else ""
                if stars:
                    ax.text(hi+0.05, y, stars, va="center", fontsize=10,
                           color=model_colors[outcome_label], fontweight="bold")
                print(f"  [{outcome_label}] {var}: coef={c:+.3f}  p={p:.4f} {stars}")
        except Exception as e:
            print(f"  [{outcome_label}] failed: {e}")

    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels(list(interaction_vars.values()), fontsize=9)
    ax.set_xlabel("Logit Coefficient")
    ax.set_title("2. Novelty × Syndicate Interaction Coefficients\nblue=exit model, red=failure model",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.text(0.02, 0.04, "† p<0.15  * p<0.05  ** p<0.01  *** p<0.001",
           transform=ax.transAxes, fontsize=8, color="grey")

    # ── 3. Cluster-level: exit & failure rate by cluster top keywords ────────
    ax = axes[1, 0]
    # for the most interesting clusters: top-10 by novelty AND with ≥20 non-green companies
    df["keywords_short"] = df["specter_keywords"].str.split(",").str[:3].str.join(", ")
    cluster_stats = df.groupby("keywords_short").apply(lambda g: pd.Series({
        "n_total":    len(g),
        "n_png":      (g["syndicate_type"]=="Pure Non-Green").sum(),
        "n_green":    (g["syndicate_type"]=="Pure Green").sum(),
        "avg_novelty": g["specter_novelty"].mean(),
        "exit_png":   g[g["syndicate_type"]=="Pure Non-Green"]["exited"].mean()*100 if (g["syndicate_type"]=="Pure Non-Green").sum()>5 else np.nan,
        "exit_green": g[g["syndicate_type"]=="Pure Green"]["exited"].mean()*100 if (g["syndicate_type"]=="Pure Green").sum()>5 else np.nan,
        "fail_png":   g[g["syndicate_type"]=="Pure Non-Green"]["failed"].mean()*100 if (g["syndicate_type"]=="Pure Non-Green").sum()>5 else np.nan,
        "fail_green": g[g["syndicate_type"]=="Pure Green"]["failed"].mean()*100 if (g["syndicate_type"]=="Pure Green").sum()>5 else np.nan,
    }), include_groups=False).reset_index()
    cluster_stats = cluster_stats.dropna(subset=["exit_png","exit_green"]).nlargest(15,"avg_novelty")
    cluster_stats = cluster_stats.sort_values("exit_png", ascending=True)

    y = np.arange(len(cluster_stats))
    w2 = 0.35
    ax.barh(y - w2/2, cluster_stats["exit_png"],  w2, label="Exit — Pure Non-Green",
           color="#7f8c8d", edgecolor="white", alpha=0.9)
    ax.barh(y + w2/2, cluster_stats["exit_green"], w2, label="Exit — Pure Green",
           color="#27ae60", edgecolor="white", alpha=0.9)
    ax.errorbar(cluster_stats["fail_png"],  y - w2/2, fmt="x", color="#e74c3c",
               markersize=8, markeredgewidth=2, label="Failure rate (×)")
    ax.errorbar(cluster_stats["fail_green"], y + w2/2, fmt="x", color="#c0392b",
               markersize=8, markeredgewidth=2)
    ax.set_yticks(y)
    ax.set_yticklabels(cluster_stats["keywords_short"], fontsize=8)
    ax.set_xlabel("Rate (%)")
    ax.set_title("3. Exit vs Failure Rate by Technology Cluster\n(top-15 most novel clusters with ≥5 non-green companies)",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── 4. Novelty vs pct_green, size=exit rate, color=failure rate ─────────
    ax = axes[1, 1]
    # bin novelty into 10 buckets × syndicate
    df["novelty_bin"] = pd.cut(df["specter_novelty"], bins=10)
    scatter_df = df.groupby(["syndicate_type","novelty_bin"]).agg(
        novelty=("specter_novelty","mean"),
        pct_green=("pct_green","mean"),
        exit_rate=("exited","mean"),
        fail_rate=("failed","mean"),
        n=("exited","count"),
    ).reset_index()
    scatter_df = scatter_df[scatter_df["n"] >= 10]

    colors = [PALETTE[g] for g in scatter_df["syndicate_type"]]
    sc = ax.scatter(scatter_df["novelty"], scatter_df["exit_rate"]*100,
                   s=scatter_df["fail_rate"]*1000 + 30,
                   c=scatter_df["fail_rate"], cmap="RdYlGn_r",
                   vmin=0, vmax=0.20, alpha=0.8, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Failure rate (colour)")

    for grp in groups:
        sub = scatter_df[scatter_df["syndicate_type"]==grp]
        ax.plot(sub["novelty"], sub["exit_rate"]*100, color=PALETTE[grp],
               linewidth=2, alpha=0.7, label=grp)

    ax.set_xlabel("Avg Novelty Score (SPECTER2)")
    ax.set_ylabel("Exit Rate (%)")
    ax.set_title("4. Novelty vs Exit Rate by Syndicate\n(dot size = failure rate; colour = failure intensity)",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    out_path = OUT / "novelty_investor_interaction.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")

    # ── Print full 3-way table ───────────────────────────────────────────────
    print("\n=== FULL OUTCOME TABLE: syndicate × novelty quartile ===")
    tbl = df.groupby(["syndicate_type","novelty_q"], observed=True).agg(
        n=("exited","count"),
        exit_pct=("exited",lambda x: f"{x.mean()*100:.1f}%"),
        fail_pct=("failed",lambda x: f"{x.mean()*100:.1f}%"),
    ).reset_index()
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
