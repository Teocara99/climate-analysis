"""
09d — Clean visualisations of the regression results from 09c.

4 charts:
  1. Forest plot: Pure Green & Pure Non-Green effect across all models
  2. Policy shock coefficients with 95% CIs (Model 2B)
  3. Observed exit rates by syndicate type (raw + adjusted)
  4. Pre vs post key policy shocks by syndicate type (interaction)
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
CENSORING_DATE = pd.Timestamp("2026-07-03")

PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
    "sig":    "#e74c3c",
    "insig":  "#bdc3c7",
    "accent": "#2980b9",
}


def load_data():
    deals = pd.read_csv(DATA / "deals_with_external_data.csv", parse_dates=["Deal Date"])
    companies = load_companies()

    def outcome(row):
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired" in own:                       return "acquired"
        if "Out of Business" in own or "Liquidation" in biz or "Out of Business" in biz:
            return "failed"
        return "operating"

    companies["outcome"] = companies.apply(outcome, axis=1)
    comp_map = companies.set_index("Companies")["outcome"].to_dict()
    deals["outcome"]   = deals["Companies"].map(comp_map).fillna("operating")
    deals["exited"]    = deals["outcome"].isin(["ipo", "acquired"]).astype(int)
    deals["survived"]  = deals["outcome"].isin(["ipo","acquired","operating"]).astype(int)
    deals["failed"]    = (deals["outcome"] == "failed").astype(int)
    deals["Pure_Green"]     = (deals["syndicate_type"] == "Pure Green").astype(int)
    deals["Pure_Non_Green"] = (deals["syndicate_type"] == "Pure Non-Green").astype(int)
    deals["inv_year"]  = deals["Year"].astype(float).astype(int).astype(str)
    deals["region"]    = deals["HQ Global Region"].fillna("Unknown")
    for col in ["oil_price","interest_rate","vix"]:
        if col in deals.columns and deals[col].notna().any():
            m, s = deals[col].mean(), deals[col].std()
            if s > 0: deals[f"{col}_z"] = (deals[col] - m) / s
    deals["company_age_z"] = (deals["company_age"] - deals["company_age"].mean()) / deals["company_age"].std()
    deals = deals.dropna(subset=["syndicate_type","exited","Year"])
    return deals


def run_model(formula, df):
    try:
        return smf.logit(formula, data=df.dropna(subset=["company_age"])).fit(disp=False, maxiter=200)
    except Exception:
        return None


def main():
    df = load_data()
    shocks = [c for c in ["post_trump_election","post_eu_green_deal",
                           "post_covid_pandemic","post_inflation_reduction_act"]
              if c in df.columns and df[c].std() > 0]
    fred_available = "oil_price_z" in df.columns and df["oil_price_z"].notna().any()

    base = "company_age_z + C(inv_year) + C(region)"
    fred_z = " + oil_price_z + interest_rate_z + vix_z" if fred_available else ""
    shock_str = " + ".join(shocks)

    fA = f"exited ~ Pure_Green + Pure_Non_Green + {base}{fred_z}"
    fB = f"exited ~ Pure_Green + Pure_Non_Green + {shock_str} + {base}{fred_z}"

    mA = run_model(fA, df)
    mB = run_model(fB, df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Regression Results: Investor Green Mix & Policy Shocks\nvs Climate Tech Exit Success (n=32,503 deals)",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── 1. Forest plot: investor composition effect ───────────────────────
    ax = axes[0, 0]
    models = {"Model A\n(controls only)": mA, "Model B\n(+ policy shocks)": mB}
    y_positions = {"Pure_Green": 1, "Pure_Non_Green": 0}
    label_map   = {"Pure_Green": "Pure Green (≥75%)", "Pure_Non_Green": "Pure Non-Green (≤25%)"}
    colors_fp   = {"Pure_Green": PALETTE["Pure Green"], "Pure_Non_Green": PALETTE["Pure Non-Green"]}
    offsets = {"Model A\n(controls only)": -0.15, "Model B\n(+ policy shocks)": 0.15}
    markers = {"Model A\n(controls only)": "o", "Model B\n(+ policy shocks)": "s"}

    for mname, model in models.items():
        if model is None: continue
        params = model.params
        conf   = model.conf_int()
        for var in ["Pure_Green", "Pure_Non_Green"]:
            if var not in params: continue
            y = y_positions[var] + offsets[mname]
            coef = params[var]
            lo, hi = conf.loc[var]
            color = colors_fp[var]
            ax.errorbar(coef, y, xerr=[[coef-lo],[hi-coef]],
                       fmt=markers[mname], color=color, markersize=9,
                       capsize=5, linewidth=2,
                       label=mname if var == "Pure_Green" else "")
            pval = model.pvalues[var]
            stars = "***" if pval<0.001 else "**" if pval<0.01 else "*" if pval<0.05 else ""
            ax.text(hi+0.01, y, f"{coef:+.3f}{stars}", va="center", fontsize=8)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([label_map[v] for v in y_positions], fontsize=10)
    ax.set_xlabel("Logit Coefficient (vs Mixed portfolio)")
    ax.set_title("1. Investor Composition Effect on Exit Success\n(controlling for year, region, company age, macro)", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(-0.55, 0.45)
    ax.text(0.02, 0.04, "* p<0.05  ** p<0.01  *** p<0.001",
            transform=ax.transAxes, fontsize=8, color="grey")
    ax.fill_betweenx([-0.5, 1.5], 0, 0, alpha=0)  # invisible, just to set range
    ax.set_ylim(-0.5, 1.5)

    # ── 2. Policy shock coefficients (Model B) ────────────────────────────
    ax = axes[0, 1]
    if mB is not None:
        shock_labels = {
            "post_trump_election":          "Trump\nElection\n(2016)",
            "post_eu_green_deal":           "EU Green\nDeal\n(2019)",
            "post_covid_pandemic":          "COVID\nPandemic\n(2020)",
            "post_inflation_reduction_act": "Inflation\nReduction Act\n(2022)",
        }
        params = mB.params
        conf   = mB.conf_int()
        pvals  = mB.pvalues
        xs, ys, errs, clrs, lbls = [], [], [], [], []
        for i, (var, label) in enumerate(shock_labels.items()):
            if var not in params: continue
            c   = params[var]
            lo, hi = conf.loc[var]
            p   = pvals[var]
            xs.append(c); ys.append(i)
            errs.append([[c-lo],[hi-c]])
            clrs.append(PALETTE["sig"] if p < 0.05 else PALETTE["insig"])
            stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            lbls.append((label, c, hi, p, stars))

        for i, (x, y, err, color) in enumerate(zip(xs, ys, errs, clrs)):
            ax.barh(y, x, color=color, alpha=0.8, edgecolor="white", height=0.5)
            lo_e = err[0][0]; hi_e = err[1][0]
            ax.errorbar(x, y, xerr=[[lo_e],[hi_e]], fmt="none",
                       color="black", capsize=5, linewidth=1.5)
        for label, c, hi, p, stars in lbls:
            ax.text(max(hi, abs(c))*1.05 + 0.02, lbls.index((label,c,hi,p,stars)),
                   f"{c:+.3f} ({stars})", va="center", fontsize=9,
                   color=PALETTE["sig"] if p<0.05 else "grey")
        ax.set_yticks(range(len(lbls)))
        ax.set_yticklabels([l[0] for l in lbls], fontsize=9)
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Logit Coefficient (effect on exit probability)")
        ax.set_title("2. Policy Shock Effects on Exit Success (Model B)\nred = significant (p<0.05)", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_xlim(-0.3, 0.7)

    # ── 3. Observed exit rates by syndicate type ──────────────────────────
    ax = axes[1, 0]
    groups = ["Pure Green", "Mixed", "Pure Non-Green"]
    raw_rates  = [df[df["syndicate_type"]==g]["exited"].mean()*100 for g in groups]
    ns         = [df[df["syndicate_type"]==g].shape[0] for g in groups]
    colors_bar = [PALETTE[g] for g in groups]
    bars = ax.bar(groups, raw_rates, color=colors_bar, edgecolor="white", width=0.55)
    ax.bar_label(bars, labels=[f"{r:.1f}%\n(n={n:,})" for r,n in zip(raw_rates, ns)],
                 padding=4, fontsize=9)

    # Add adjusted rates (predicted probability from Model A at mean controls)
    if mA is not None:
        base_row = pd.DataFrame({
            "company_age_z": [0], "inv_year": ["2020"],
            "region": ["Europe"],
            "Pure_Green": [0], "Pure_Non_Green": [0],
            **({col: [0] for col in ["oil_price_z","interest_rate_z","vix_z"]} if fred_available else {})
        })
        adj_rates = {}
        for g, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
            row = base_row.copy()
            row["Pure_Green"] = pg; row["Pure_Non_Green"] = png
            adj_rates[g] = float(mA.predict(row).iloc[0]) * 100
        ax.plot(groups, [adj_rates[g] for g in groups], "D--",
               color="black", markersize=8, linewidth=2, label="Adjusted (Model A)")
        ax.legend(fontsize=9)

    ax.set_ylabel("Exit Rate (%)")
    ax.set_title("3. Observed Exit Rate by Syndicate Type\n(IPO or M&A as success)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(raw_rates)*1.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # ── 4. Pre vs Post IRA & EU Green Deal by syndicate ───────────────────
    ax = axes[1, 1]
    key_shocks = [(c, l) for c, l in [
        ("post_inflation_reduction_act", "Post-IRA\n(2022)"),
        ("post_eu_green_deal",           "Post-EU Green Deal\n(2019)"),
        ("post_trump_election",          "Post-Trump\nElection (2016)"),
    ] if c in df.columns]

    n_shocks = len(key_shocks)
    x = np.arange(n_shocks)
    width = 0.25
    for j, grp in enumerate(["Pure Green", "Mixed", "Pure Non-Green"]):
        vals = []
        for shock_col, _ in key_shocks:
            post  = df[(df["syndicate_type"]==grp) & (df[shock_col]==1)]["exited"].mean()*100
            pre   = df[(df["syndicate_type"]==grp) & (df[shock_col]==0)]["exited"].mean()*100
            vals.append(post - pre)
        bars = ax.bar(x + j*width, vals, width, label=grp,
                     color=PALETTE[grp], edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1*(1 if val>=0 else -1),
                   f"{val:+.1f}pp", ha="center", va="bottom" if val>=0 else "top", fontsize=8)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x + width)
    ax.set_xticklabels([l for _, l in key_shocks], fontsize=9)
    ax.set_ylabel("Change in Exit Rate (percentage points)")
    ax.set_title("4. Change in Exit Rate Pre→Post Policy Shock\nby Syndicate Type", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}pp"))

    plt.tight_layout()
    path = OUT / "regression_results_4panel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
