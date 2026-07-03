"""
09c — Regression models A, B, C × 4 success definitions.

Model A: investor composition → outcome (no policy shocks)
Model B: composition + policy shocks → outcome
Model C: composition + policy shocks + INTERACTIONS → outcome

4 success definitions:
  1. Survival      (operating + exit = 1)
  2. Exit Success  (IPO/M&A = 1)
  3. Failure       (closed/bankrupt = 1)
  4. Cox Survival  (time-to-failure, censored for still-operating)

Syndicate type encoding: Mixed = reference; dummies for Pure_Green, Pure_Non_Green.

Outputs:
  output/regression_tables.csv   — all coefficients/p-values
  output/regression_summary.txt  — readable summary
  output/policy_interaction_chart.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from pathlib import Path
from load_data import load_companies

DATA = Path(__file__).parent / "data"
OUT  = Path(__file__).parent / "output"

POLICY_SHOCKS = [
    "post_paris_agreement",
    "post_trump_election",
    "post_eu_green_deal",
    "post_covid_pandemic",
    "post_inflation_reduction_act",
]
FRED_CONTROLS = ["oil_price", "interest_rate", "vix"]
CENSORING_DATE = pd.Timestamp("2026-07-03")

SYNDICATE_COLORS = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}


def load_data() -> pd.DataFrame:
    deals = pd.read_csv(DATA / "deals_with_external_data.csv", parse_dates=["Deal Date"])
    companies = load_companies()

    def outcome(row) -> str:
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired" in own:                       return "acquired"
        if "Out of Business" in own or "Liquidation" in biz or \
           "Out of Business" in biz:                return "failed"
        return "operating"

    companies["outcome"] = companies.apply(outcome, axis=1)
    comp_map = companies.set_index("Companies")["outcome"].to_dict()

    deals["outcome"] = deals["Companies"].map(comp_map).fillna("operating")
    deals["survived"] = (deals["outcome"].isin(["ipo","acquired","operating"])).astype(int)
    deals["exited"]   = (deals["outcome"].isin(["ipo","acquired"])).astype(int)
    deals["failed"]   = (deals["outcome"] == "failed").astype(int)

    # Survival time (company level, proxied from deal date to censoring)
    deals["t_years"] = (CENSORING_DATE - deals["Deal Date"]).dt.days / 365.25
    deals["km_event"]    = deals["failed"]
    deals["km_duration"] = deals["t_years"].clip(lower=0.01)

    # Encode syndicate type (Mixed = reference)
    deals["Pure_Green"]     = (deals["syndicate_type"] == "Pure Green").astype(int)
    deals["Pure_Non_Green"] = (deals["syndicate_type"] == "Pure Non-Green").astype(int)

    # Region dummies
    deals["region"] = deals["HQ Global Region"].fillna("Unknown")
    deals["inv_year"] = deals["Year"].astype(float).astype("Int64").astype(str)

    # Fill missing controls with median (for deals without FRED data)
    for col in FRED_CONTROLS:
        if col in deals.columns and deals[col].notna().any():
            deals[col] = deals[col].fillna(deals[col].median())

    # Standardise continuous controls
    for col in FRED_CONTROLS + ["company_age", "pct_green"]:
        if col in deals.columns and deals[col].notna().any():
            m, s = deals[col].mean(), deals[col].std()
            if s > 0:
                deals[col + "_z"] = (deals[col] - m) / s
            else:
                deals[col + "_z"] = 0

    deals = deals.dropna(subset=["syndicate_type", "survived", "Year"])
    return deals


def safe_available_shocks(df: pd.DataFrame) -> list:
    avail = [c for c in POLICY_SHOCKS if c in df.columns and df[c].std() > 0]
    return avail


def run_logit(df: pd.DataFrame, formula: str, label: str) -> pd.DataFrame:
    try:
        model = smf.logit(formula, data=df.dropna(subset=["company_age"])).fit(
            disp=False, maxiter=200)
        res = model.summary2().tables[1].copy()
        res["model"] = label
        res["n"] = int(model.nobs)
        res["pseudo_r2"] = round(model.prsquared, 4)
        return res
    except Exception as e:
        print(f"    WARNING [{label}]: {e}")
        return pd.DataFrame()


def run_cox(df: pd.DataFrame, covariates: list, label: str) -> pd.DataFrame:
    cols = ["km_duration", "km_event"] + [c for c in covariates if c in df.columns]
    sub  = df[cols].dropna()
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(sub, duration_col="km_duration", event_col="km_event")
        res = cph.summary.copy()
        res["model"] = label
        res["n"] = int(cph._n_examples)
        return res
    except Exception as e:
        print(f"    WARNING [{label}]: {e}")
        return pd.DataFrame()


def build_formulas(shocks: list, outcome: str, df: pd.DataFrame) -> dict:
    base_controls = "company_age_z + C(inv_year) + C(region)"
    # Only include FRED controls that are actually in the dataframe with real values
    available_fred = [c for c in FRED_CONTROLS
                      if f"{c}_z" in df.columns and df[f"{c}_z"].notna().any()]
    fred_z = " + ".join(f"{c}_z" for c in available_fred)

    syndicates = "Pure_Green + Pure_Non_Green"
    shock_str  = " + ".join(shocks) if shocks else "0"
    inter_str  = " + ".join(
        f"{s}:Pure_Green + {s}:Pure_Non_Green" for s in shocks
    ) if shocks else "0"

    fa = f"{outcome} ~ {syndicates} + {base_controls}"
    fb = f"{outcome} ~ {syndicates} + {shock_str} + {base_controls}"
    fc = f"{outcome} ~ {syndicates} + {shock_str} + {inter_str} + {base_controls}"
    if fred_z.strip():
        fa += f" + {fred_z}"
        fb += f" + {fred_z}"
        fc += f" + {fred_z}"
    return {"A": fa, "B": fb, "C": fc}


def main():
    print("Loading merged deal data...")
    df = load_data()
    print(f"  Deals: {len(df):,} | Syndicate types: {df['syndicate_type'].value_counts().to_dict()}")

    shocks = safe_available_shocks(df)
    print(f"  Available policy shock dummies: {shocks}")

    fred_available = all(f"{c}_z" in df.columns and df[f"{c}_z"].notna().any()
                         for c in FRED_CONTROLS)
    if not fred_available:
        print("  NOTE: FRED controls not available — running without them.")

    outcomes = {
        "survived": "Model 1 — Survival",
        "exited":   "Model 2 — Exit Success",
        "failed":   "Model 3 — Failure",
    }

    all_results = []

    for out_col, out_label in outcomes.items():
        print(f"\n{'='*60}")
        print(f"  {out_label}")
        formulas = build_formulas(shocks, out_col, df)
        for mname, formula in formulas.items():
            label = f"{out_label} | Model {mname}"
            print(f"    Running logit Model {mname}...")
            res = run_logit(df, formula, label)
            if not res.empty:
                all_results.append(res)

    # ── Model 4: Cox ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Model 4 — Cox Survival")
    cox_vars_base = ["Pure_Green", "Pure_Non_Green", "company_age_z"]
    for c in FRED_CONTROLS:
        if f"{c}_z" in df.columns:
            cox_vars_base.append(f"{c}_z")
    cox_vars_B = cox_vars_base + shocks
    cox_vars_C = cox_vars_B + [f"{s}_Pure_Green" for s in shocks] + \
                              [f"{s}_Pure_Non_Green" for s in shocks]
    for s in shocks:
        df[f"{s}_Pure_Green"]     = df[s] * df["Pure_Green"]
        df[f"{s}_Pure_Non_Green"] = df[s] * df["Pure_Non_Green"]

    for mname, cox_vars in [("A", cox_vars_base), ("B", cox_vars_B), ("C", cox_vars_C)]:
        print(f"    Running Cox Model {mname}...")
        res = run_cox(df, cox_vars, f"Model 4 — Cox | Model {mname}")
        if not res.empty:
            all_results.append(res)

    # ── Save tables ────────────────────────────────────────────────────────
    if all_results:
        combined = pd.concat([r for r in all_results if not r.empty], sort=False)
        combined.to_csv(OUT / "regression_tables.csv")
        print(f"\nSaved → output/regression_tables.csv ({len(combined)} coefficient rows)")

    # ── Interaction chart ──────────────────────────────────────────────────
    _plot_interaction_chart(df, shocks)

    # ── Print key Model C results ─────────────────────────────────────────
    _print_summary(all_results)


def _plot_interaction_chart(df: pd.DataFrame, shocks: list):
    if not shocks:
        print("No policy shocks available for interaction chart.")
        return

    # For each shock and syndicate type: survival rate pre vs post
    key_shocks = [s for s in ["post_paris_agreement","post_inflation_reduction_act",
                               "post_eu_green_deal","post_covid_pandemic"] if s in shocks]
    if not key_shocks:
        key_shocks = shocks[:4]

    fig, axes = plt.subplots(1, len(key_shocks), figsize=(5*len(key_shocks), 5), sharey=True)
    if len(key_shocks) == 1:
        axes = [axes]

    fig.suptitle("Survival Rate by Syndicate Type: Pre vs Post Policy Shock",
                 fontsize=13, fontweight="bold")

    for ax, shock in zip(axes, key_shocks):
        label = shock.replace("post_","").replace("_"," ").title()
        x = np.arange(3)
        width = 0.35
        groups = ["Pure Green", "Mixed", "Pure Non-Green"]
        pre_rates, post_rates = [], []
        for grp in groups:
            sub = df[df["syndicate_type"] == grp]
            pre_rates.append(sub[sub[shock]==0]["survived"].mean()*100)
            post_rates.append(sub[sub[shock]==1]["survived"].mean()*100)
        colors = [SYNDICATE_COLORS[g] for g in groups]
        bars1 = ax.bar(x - width/2, pre_rates,  width, label="Pre",
                       color=colors, alpha=0.5, edgecolor="white")
        bars2 = ax.bar(x + width/2, post_rates, width, label="Post",
                       color=colors, alpha=1.0, edgecolor="white")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Pure\nGreen","Mixed","Pure\nNon-Green"], fontsize=8)
        ax.set_ylabel("Survival Rate (%)") if ax == axes[0] else None
        ax.set_ylim(70, 100)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(OUT / "policy_interaction_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → output/policy_interaction_chart.png")


def _print_summary(all_results: list):
    print("\n" + "="*60)
    print("KEY FINDINGS — Pure_Green & Pure_Non_Green coefficients")
    print("="*60)
    for res in all_results:
        if res.empty:
            continue
        key_rows = res[res.index.str.startswith(("Pure_Green","Pure_Non_Green","T.Pure"))]
        if key_rows.empty:
            continue
        model_name = res["model"].iloc[0] if "model" in res.columns else "?"
        n = res["n"].iloc[0] if "n" in res.columns else "?"
        print(f"\n{model_name}  (n={n})")
        cols = [c for c in ["Coef.","Std.Err.","P>|z|","[0.025","0.975]",
                              "coef","p","exp(coef)"] if c in key_rows.columns]
        print(key_rows[cols].to_string())


if __name__ == "__main__":
    main()
