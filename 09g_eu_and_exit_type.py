"""
09g — Two analyses in one script:

A) EU deals only: syndicate_type × post_eu_green_deal
   Does the composition effect differ after the EU Green Deal?

B) Exit type split: P(M&A) and P(IPO) × syndicate_type × post_trump_2016
   Tests the fire-sale hypothesis:
   If Trump 1 exits were fire-sale M&A → effect should be M&A-driven, not IPO.

Outputs:
  output/eu_green_deal_interaction.png
  output/exit_type_split.png
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

OUT  = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

EU_COUNTRIES = {
    "Germany","France","Netherlands","Sweden","Denmark","Finland","Norway",
    "Spain","Italy","Belgium","Austria","United Kingdom","Switzerland",
    "Ireland","Portugal","Poland","Czech Republic","Hungary","Romania",
    "Greece","Croatia","Slovenia","Slovakia","Estonia","Latvia","Lithuania",
    "Luxembourg","Malta","Cyprus","Bulgaria",
}
PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}
ERA_COLORS = {"Pre-Green Deal": "#95a5a6", "Post-Green Deal": "#2ecc71"}


def load_base() -> pd.DataFrame:
    deals = pd.read_csv(DATA / "deals_with_external_data.csv", parse_dates=["Deal Date"])
    companies = load_companies()

    def outcome(row):
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own:  return "ipo"
        if "Acquired" in own:                         return "acquired"
        if "Out of Business" in own or "Liquidation" in biz or \
           "Out of Business" in biz:                  return "failed"
        return "operating"

    companies["outcome"] = companies.apply(outcome, axis=1)
    comp_map = companies.set_index("Companies")["outcome"].to_dict()
    deals["outcome"]    = deals["Companies"].map(comp_map).fillna("operating")
    deals["exited"]     = deals["outcome"].isin(["ipo","acquired"]).astype(int)
    deals["is_ipo"]     = (deals["outcome"] == "ipo").astype(int)
    deals["is_ma"]      = (deals["outcome"] == "acquired").astype(int)
    deals["failed"]     = (deals["outcome"] == "failed").astype(int)

    deals["Pure_Green"]     = (deals["syndicate_type"] == "Pure Green").astype(int)
    deals["Pure_Non_Green"] = (deals["syndicate_type"] == "Pure Non-Green").astype(int)
    deals["inv_year"]       = deals["Year"].astype(float).astype(int).astype(str)
    deals["company_age_z"]  = (
        (deals["company_age"] - deals["company_age"].mean()) /
        deals["company_age"].std()
    )
    for col in ["oil_price","interest_rate","vix"]:
        if col in deals.columns and deals[col].notna().any():
            m, s = deals[col].mean(), deals[col].std()
            if s > 0: deals[f"{col}_z"] = (deals[col] - m) / s
    return deals.dropna(subset=["syndicate_type","exited","Year","company_age"])


def run_logit(formula, df, label=""):
    try:
        m = smf.logit(formula, data=df).fit(disp=False, maxiter=300)
        print(f"  [{label}] n={int(m.nobs):,}  pseudo-R2={m.prsquared:.3f}")
        return m
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return None


def forest_panel(ax, models_dict, vars_of_interest, title, xlim=(-2,2)):
    """Generic forest/bar plot for multiple models side-by-side."""
    palette = ["#e74c3c","#3498db","#27ae60","#f39c12"]
    y_labels = list(vars_of_interest.values())
    y_pos    = np.arange(len(y_labels))
    offsets  = np.linspace(-0.2*(len(models_dict)-1)/2,
                            0.2*(len(models_dict)-1)/2, len(models_dict))

    for (mname, model), offset, color in zip(models_dict.items(), offsets, palette):
        if model is None: continue
        params = model.params; conf = model.conf_int(); pvals = model.pvalues
        for i, (var, _) in enumerate(vars_of_interest.items()):
            actual = var if var in params.index else \
                     ":".join(var.split(":")[::-1]) if ":".join(var.split(":")[::-1]) in params.index else None
            if actual is None: continue
            c = params[actual]; lo, hi = conf.loc[actual]; p = pvals[actual]
            y = y_pos[i] + offset
            ax.errorbar(c, y, xerr=[[c-lo],[hi-c]], fmt="o", color=color,
                       markersize=7, capsize=4, linewidth=1.8, label=mname if i==0 else "")
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else \
                    "(†)" if p<0.15 else ""
            if stars:
                ax.text(hi+0.05, y, stars, va="center", fontsize=9, color=color, fontweight="bold")

    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlim(xlim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.02,0.02,"* p<0.05  ** p<0.01  *** p<0.001  (†) p<0.15",
           transform=ax.transAxes, fontsize=7.5, color="grey")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS A: EU Green Deal interaction
# ══════════════════════════════════════════════════════════════════════════════
def run_eu_analysis(df):
    eu = df[df["Company Country/Territory/Region"].isin(EU_COUNTRIES)].copy()
    eu["era"] = np.where(eu.get("post_eu_green_deal", pd.Series(0, index=eu.index))==1,
                         "Post-Green Deal", "Pre-Green Deal")
    print(f"\n[EU] n={len(eu):,}")
    print(eu.groupby(["era","syndicate_type"])["exited"].agg(["mean","count"]).round(3))

    fred_z = " + oil_price_z + interest_rate_z + vix_z" \
             if "oil_price_z" in eu.columns and eu["oil_price_z"].notna().any() else ""

    base_formula = f"exited ~ Pure_Green + Pure_Non_Green + company_age_z + C(inv_year){fred_z}"
    int_formula  = (f"exited ~ Pure_Green + Pure_Non_Green"
                    f" + post_eu_green_deal"
                    f" + post_eu_green_deal:Pure_Green"
                    f" + post_eu_green_deal:Pure_Non_Green"
                    f" + company_age_z + C(inv_year){fred_z}")

    mA = run_logit(base_formula,  eu, "EU Model A (no interaction)")
    mB = run_logit(int_formula,   eu, "EU Model B (with interaction)")

    print("\n[EU] Interaction results:")
    if mB:
        for v in ["post_eu_green_deal","post_eu_green_deal:Pure_Green",
                  "post_eu_green_deal:Pure_Non_Green","Pure_Green","Pure_Non_Green"]:
            alt = ":".join(v.split(":")[::-1])
            actual = v if v in mB.params else (alt if alt in mB.params else None)
            if actual:
                c,p = mB.params[actual], mB.pvalues[actual]
                stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
                print(f"  {actual:50s}: {c:+.4f}  p={p:.4f} {stars}")

    return eu, mA, mB


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS B: Exit type split (M&A vs IPO)
# ══════════════════════════════════════════════════════════════════════════════
def run_exit_type_analysis(df):
    us = df[df["Company Country/Territory/Region"] == "United States"].copy()
    print(f"\n[EXIT TYPE] US deals: {len(us):,}")
    print(f"  M&A count: {us['is_ma'].sum():,} ({us['is_ma'].mean()*100:.1f}%)")
    print(f"  IPO count: {us['is_ipo'].sum():,} ({us['is_ipo'].mean()*100:.1f}%)")
    print(f"  By era:")
    print(us.groupby("post_trump_election_2016")[["is_ma","is_ipo"]].mean().round(3)*100)

    fred_z = " + oil_price_z + interest_rate_z + vix_z" \
             if "oil_price_z" in us.columns and us["oil_price_z"].notna().any() else ""

    def interaction_formula(outcome):
        return (f"{outcome} ~ Pure_Green + Pure_Non_Green"
                f" + post_trump_election_2016"
                f" + post_trump_election_2016:Pure_Green"
                f" + post_trump_election_2016:Pure_Non_Green"
                f" + company_age_z{fred_z}")

    m_ma  = run_logit(interaction_formula("is_ma"),  us, "M&A model")
    m_ipo = run_logit(interaction_formula("is_ipo"), us, "IPO model")

    print("\n[EXIT TYPE] M&A vs IPO coefficients:")
    for label, model in [("M&A", m_ma), ("IPO", m_ipo)]:
        if model is None: continue
        for v in ["post_trump_election_2016",
                  "post_trump_election_2016:Pure_Green",
                  "post_trump_election_2016:Pure_Non_Green"]:
            alt = ":".join(v.split(":")[::-1])
            actual = v if v in model.params else (alt if alt in model.params else None)
            if actual:
                c,p = model.params[actual], model.pvalues[actual]
                stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "(†)" if p<0.15 else ""
                print(f"  [{label}] {actual:50s}: {c:+.4f}  p={p:.4f} {stars}")

    return us, m_ma, m_ipo


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def make_eu_chart(eu, mA, mB):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("EU Deals — Syndicate Type × EU Green Deal (2019)\n"
                 "Does investor composition effect change after the Green Deal?",
                 fontsize=12, fontweight="bold")

    era_order = ["Pre-Green Deal","Post-Green Deal"]

    # 1. Raw exit rates pre/post Green Deal by syndicate
    ax = axes[0]
    groups = ["Pure Green","Mixed","Pure Non-Green"]
    x = np.arange(2)
    w = 0.25
    for j, grp in enumerate(groups):
        sub = eu.groupby(["era","syndicate_type"])["exited"].mean()*100
        vals = [sub.get((era, grp), np.nan) for era in era_order]
        ns   = [eu[(eu["era"]==era)&(eu["syndicate_type"]==grp)].shape[0] for era in era_order]
        bars = ax.bar(x + j*w, vals, w, label=grp, color=PALETTE[grp],
                     edgecolor="white", alpha=0.9)
        for bar, v, n in zip(bars, vals, ns):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                       f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x+w)
    ax.set_xticklabels(era_order)
    ax.set_ylabel("Exit Rate (%)"); ax.legend(fontsize=8)
    ax.set_title("1. Raw Exit Rates\nPre vs Post EU Green Deal", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # 2. Change in exit rate (post - pre)
    ax = axes[1]
    changes = {}
    for grp in groups:
        pre  = eu[(eu["era"]=="Pre-Green Deal")  &(eu["syndicate_type"]==grp)]["exited"].mean()*100
        post = eu[(eu["era"]=="Post-Green Deal") &(eu["syndicate_type"]==grp)]["exited"].mean()*100
        changes[grp] = post - pre
    colors = [PALETTE[g] for g in groups]
    bars = ax.bar(groups, [changes[g] for g in groups], color=colors, edgecolor="white", alpha=0.9)
    ax.bar_label(bars, labels=[f"{changes[g]:+.2f}pp" for g in groups], padding=3, fontsize=10)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Change in Exit Rate (pp)")
    ax.set_title("2. Change in Exit Rate\nPost EU Green Deal vs Pre (raw)",
                fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}pp"))

    # 3. Interaction coefficients
    ax = axes[2]
    vars_int = {
        "post_eu_green_deal":             "Post EU Green Deal\n(Mixed baseline)",
        "post_eu_green_deal:Pure_Green":  "Pure Green × Post Green Deal\n(vs Mixed)",
        "post_eu_green_deal:Pure_Non_Green": "Pure Non-Green × Post Green Deal\n(vs Mixed)",
    }
    forest_panel(ax, {"Model B (interaction)": mB}, vars_int,
                "3. Interaction Coefficients\n(Model B, EU deals only)", xlim=(-1.5,1.5))
    ax.set_xlabel("Logit Coefficient")

    plt.tight_layout()
    fig.savefig(OUT / "eu_green_deal_interaction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT / 'eu_green_deal_interaction.png'}")


def make_exit_type_chart(us, m_ma, m_ipo):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Fire-Sale Test: M&A vs IPO Split — US Deals × Trump 2016\n"
                 "If Trump exits were fire-sales → M&A should dominate, not IPO",
                 fontsize=12, fontweight="bold")

    groups = ["Pure Green","Mixed","Pure Non-Green"]
    era_labels = ["Pre-Trump 2016","Post-Trump 2016"]
    trump_col  = "post_trump_election_2016"

    # 1. M&A and IPO rates by era × syndicate
    ax = axes[0]
    for outcome, marker, ls in [("is_ma","o","-"),("is_ipo","s","--")]:
        label_base = "M&A" if outcome=="is_ma" else "IPO"
        for grp in groups:
            vals = []
            for era_val in [0,1]:
                sub = us[(us[trump_col]==era_val)&(us["syndicate_type"]==grp)]
                vals.append(sub[outcome].mean()*100 if len(sub)>10 else np.nan)
            ax.plot([0,1], vals, marker=marker, linestyle=ls, color=PALETTE[grp],
                   linewidth=2, markersize=7,
                   label=f"{label_base} — {grp}")
    ax.set_xticks([0,1]); ax.set_xticklabels(era_labels)
    ax.set_ylabel("Exit Rate (%)")
    ax.set_title("1. M&A (solid) vs IPO (dashed)\nby Syndicate × Trump 2016",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # 2. Trump coefficient comparison: M&A vs IPO
    ax = axes[1]
    vars_trump = {
        "post_trump_election_2016":              "Trump 2016 main effect\n(Mixed baseline)",
        "post_trump_election_2016:Pure_Green":   "Pure Green × Trump 2016\n(vs Mixed)",
        "post_trump_election_2016:Pure_Non_Green":"Pure Non-Green × Trump 2016\n(vs Mixed)",
    }
    forest_panel(ax, {"M&A outcome": m_ma, "IPO outcome": m_ipo},
                vars_trump,
                "2. Trump 2016 Coefficient\nM&A vs IPO model (US only)",
                xlim=(-2.5, 3.5))
    ax.set_xlabel("Logit Coefficient")

    # 3. Decomposition: where did Trump 1 exits go?
    ax = axes[2]
    decomp = {}
    for grp in groups:
        pre_ma  = us[(us[trump_col]==0)&(us["syndicate_type"]==grp)]["is_ma"].mean()*100
        post_ma = us[(us[trump_col]==1)&(us["syndicate_type"]==grp)]["is_ma"].mean()*100
        pre_ipo = us[(us[trump_col]==0)&(us["syndicate_type"]==grp)]["is_ipo"].mean()*100
        post_ipo= us[(us[trump_col]==1)&(us["syndicate_type"]==grp)]["is_ipo"].mean()*100
        decomp[grp] = {"Δ M&A": post_ma-pre_ma, "Δ IPO": post_ipo-pre_ipo}

    x = np.arange(len(groups))
    w = 0.35
    ma_delta  = [decomp[g]["Δ M&A"]  for g in groups]
    ipo_delta = [decomp[g]["Δ IPO"]  for g in groups]
    b1 = ax.bar(x-w/2, ma_delta,  w, label="Δ M&A rate",  color="#e67e22", edgecolor="white", alpha=0.9)
    b2 = ax.bar(x+w/2, ipo_delta, w, label="Δ IPO rate",  color="#3498db", edgecolor="white", alpha=0.9)
    ax.bar_label(b1, labels=[f"{v:+.2f}pp" for v in ma_delta],  padding=3, fontsize=9)
    ax.bar_label(b2, labels=[f"{v:+.2f}pp" for v in ipo_delta], padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Change in exit rate (pp)\nPost Trump 2016 vs Pre")
    ax.set_title("3. Exit Decomposition: M&A vs IPO\nPost-Trump 2016 change by syndicate",
                fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.1f}pp"))

    plt.tight_layout()
    fig.savefig(OUT / "exit_type_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT / 'exit_type_split.png'}")


def main():
    df = load_base()
    df["post_trump_election_2016"] = df.get("post_trump_election_2016",
                                             (df["Deal Date"] > "2016-11-08").astype(int))

    print("=" * 60)
    print("ANALYSIS A: EU Green Deal")
    print("=" * 60)
    eu, mA_eu, mB_eu = run_eu_analysis(df)
    make_eu_chart(eu, mA_eu, mB_eu)

    print("\n" + "=" * 60)
    print("ANALYSIS B: Exit Type Split (M&A vs IPO)")
    print("=" * 60)
    us, m_ma, m_ipo = run_exit_type_analysis(df)
    make_exit_type_chart(us, m_ma, m_ipo)


if __name__ == "__main__":
    main()
