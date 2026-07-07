"""
11d_moderation_analysis.py — Full Moderation Analysis
======================================================
Tests whether novelty moderates the effect of investor composition on outcomes.

Models 1–3: Logit (exit / failure / survival) with interaction terms
Model 4:    Cox proportional hazards with novelty interaction
Model 5:    Quadratic novelty — test non-linearity
Model 6:    Three-way interaction novelty × syndicate × Trump 2016

For each model:
  • Coefficient table with stars
  • Marginal effects plot: predicted P(outcome) across novelty range, by syndicate
  • Simple slopes: is the novelty slope significant within each investor group?
  • Johnson-Neyman intervals: at what novelty level does the syndicate gap become
    significant? (exact for linear approximation; reported on log-odds scale for logit)

Reference group: Mixed syndicates | Novelty: SPECTER2, standardised
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from lifelines import CoxPHFitter
from load_data import load_companies

warnings.filterwarnings("ignore")

OUT  = Path(__file__).parent / "output" / "novelty"
DATA = Path(__file__).parent / "data"

PALETTE = {
    "Pure Green":     "#27ae60",
    "Mixed":          "#f39c12",
    "Pure Non-Green": "#7f8c8d",
}
NOVELTY_RANGE_Z = np.linspace(-2, 2, 80)          # x-axis for marginal effect plots
CENSORING_DATE  = pd.Timestamp("2026-07-03")


# ─────────────────────────── DATA LOADING ────────────────────────────────────

def load_data() -> pd.DataFrame:
    # Novelty scores (SPECTER2 primary)
    nov = pd.read_csv(OUT / "novelty_scores.csv")[
        ["name", "specter_novelty", "specter_cluster", "specter_keywords"]
    ]

    # Company-level investor mix
    mix = pd.read_csv(OUT.parent / "company_investor_mix.csv")

    # Company outcomes + metadata
    companies = load_companies()

    def outcome(row):
        own = str(row.get("Ownership Status", ""))
        biz = str(row.get("Business Status", ""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired"      in own:                  return "acquired"
        if "Out of Business" in own or "Liquidation" in biz: return "failed"
        return "operating"

    companies["outcome"]     = companies.apply(outcome, axis=1)
    companies["log_raised"]  = np.log1p(pd.to_numeric(companies["Total Raised"], errors="coerce").fillna(0))
    companies["first_year"]  = pd.to_datetime(companies["First Financing Date"], errors="coerce").dt.year
    comp_map = companies.drop_duplicates("Companies").set_index("Companies")[
        ["outcome","log_raised","first_year","HQ Global Region",
         "First Financing Date","Last Financing Date"]
    ].to_dict("index")

    df = nov.merge(mix, left_on="name", right_on="Companies", how="inner")
    for col in ["outcome","log_raised","first_year","HQ Global Region",
                "First Financing Date","Last Financing Date"]:
        df[col] = df["name"].map(lambda n: comp_map.get(n, {}).get(col))

    df["exited"]   = df["outcome"].isin(["ipo","acquired"]).astype(int)
    df["failed"]   = (df["outcome"] == "failed").astype(int)
    df["survived"] = df["outcome"].isin(["ipo","acquired","operating"]).astype(int)

    # Survival time for Cox model
    df["t_start"]  = pd.to_datetime(df["First Financing Date"], errors="coerce")
    df["t_end"]    = pd.to_datetime(df["Last Financing Date"],  errors="coerce").fillna(CENSORING_DATE).clip(upper=CENSORING_DATE)
    df["duration"] = ((df["t_end"] - df["t_start"]).dt.days / 365.25).clip(lower=0.01)
    df["km_event"] = df["failed"]  # event = failure; censored = still operating/exited

    # Syndicate dummies (Mixed = reference)
    df["syndicate_type"] = pd.cut(df["pct_green"], bins=[-1,25,75,101],
                                   labels=["Pure Non-Green","Mixed","Pure Green"]).astype(str)
    df["Pure_Green"] = (df["syndicate_type"] == "Pure Green").astype(int)
    df["Pure_NG"]    = (df["syndicate_type"] == "Pure Non-Green").astype(int)

    # Novelty standardised
    mu, sd = df["specter_novelty"].mean(), df["specter_novelty"].std()
    df["novelty_z"]  = (df["specter_novelty"] - mu) / sd
    df["novelty_z2"] = df["novelty_z"] ** 2
    df["_nov_mu"]    = mu
    df["_nov_sd"]    = sd

    # Company age & year FE
    # compute company age from Year Founded and first financing year
    df["company_age"] = df["first_year"].subtract(
        pd.to_numeric(companies.drop_duplicates("Companies").set_index("Companies")
                      .reindex(df["name"])["Year Founded"].values, errors="coerce")
    ).clip(lower=0)
    df["company_age_z"] = (df["company_age"] - df["company_age"].mean()) / df["company_age"].std()
    df["company_age_z"] = df["company_age_z"].fillna(0)
    df["inv_year"]      = df["first_year"].astype(float).astype("Int64").astype(str)
    df["region"]        = df["HQ Global Region"].fillna("Unknown")

    # Merge FRED controls
    fred_path = DATA / "fred_economic_controls.csv"
    if fred_path.exists():
        fred = pd.read_csv(fred_path)
        fred["year_month"] = pd.PeriodIndex(fred["year_month"], freq="M")
        df["year_month"] = pd.to_datetime(df["First Financing Date"], errors="coerce").dt.to_period("M")
        df = df.merge(fred[["year_month","oil_price","interest_rate","vix"]],
                      on="year_month", how="left")
        for col in ["oil_price","interest_rate","vix"]:
            m2, s2 = df[col].mean(), df[col].std()
            if s2 > 0: df[f"{col}_z"] = (df[col] - m2) / s2
        df[["oil_price_z","interest_rate_z","vix_z"]] = \
            df[["oil_price_z","interest_rate_z","vix_z"]].fillna(0)
        fred_ctrl = " + oil_price_z + interest_rate_z + vix_z"
    else:
        fred_ctrl = ""

    # Trump 2016 dummy (for Model 6)
    df["post_trump"] = (pd.to_datetime(df["First Financing Date"], errors="coerce")
                        > pd.Timestamp("2016-11-08")).astype(int)

    df = df.dropna(subset=["novelty_z","exited","inv_year","company_age_z"])
    df["_fred_ctrl"] = fred_ctrl
    print(f"Dataset: {len(df):,} companies  |  "
          f"novelty range: [{df['novelty_z'].min():.2f}, {df['novelty_z'].max():.2f}]  |  "
          f"FRED controls: {'yes' if fred_ctrl else 'no'}")
    return df


# ─────────────────────────── MODEL HELPERS ───────────────────────────────────

def base_ctrl(df) -> str:
    fred = df["_fred_ctrl"].iloc[0]
    return f"company_age_z + log_raised + C(inv_year) + C(region){fred}"


def fit_logit(formula, df, label=""):
    try:
        m = smf.logit(formula, data=df).fit(disp=False, maxiter=300)
        print(f"  [{label}] n={int(m.nobs):,}  pseudo-R²={m.prsquared:.4f}")
        return m
    except Exception as e:
        print(f"  [{label}] FAILED: {e}")
        return None


def simple_slopes(model, terms: dict) -> pd.DataFrame:
    """
    Compute simple slopes for each investor group.
    terms = {"Mixed": "novelty_z",
             "Pure Green": "novelty_z + novelty_z:Pure_Green",
             "Pure Non-Green": "novelty_z + novelty_z:Pure_NG"}
    Uses the delta method via the model's covariance matrix.
    """
    rows = []
    for label, expr_vars in terms.items():
        vlist = [v.strip() for v in expr_vars.split("+")]
        coefs = [model.params.get(v, 0) for v in vlist]
        slope = sum(coefs)
        # variance via delta method: Var(aX+bY) = a²Var(X)+b²Var(Y)+2abCov(X,Y)
        cov = model.cov_params()
        var = sum(cov.loc[v, v] for v in vlist if v in cov.index)
        for i in range(len(vlist)):
            for j in range(i+1, len(vlist)):
                if vlist[i] in cov.index and vlist[j] in cov.index:
                    var += 2 * cov.loc[vlist[i], vlist[j]]
        se = np.sqrt(max(var, 0))
        z  = slope / se if se > 0 else np.nan
        p  = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        rows.append({"Group": label, "Slope": round(slope,4), "SE": round(se,4),
                     "Z": round(z,3) if not np.isnan(z) else "–",
                     "p": round(p,4) if not np.isnan(p) else "–",
                     "Sig": "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else
                            "(†)" if isinstance(p,float) and p<0.15 else ""})
    return pd.DataFrame(rows)


def johnson_neyman(model, key_var: str, moderator_range=None, label="") -> dict:
    """
    Approximate J-N interval: find novelty_z values where the 95% CI
    of the simple slope (for the key_var interaction) crosses zero.
    Operates on the log-odds scale (appropriate for logit).
    """
    if moderator_range is None:
        moderator_range = NOVELTY_RANGE_Z
    results = []
    for nov_z in moderator_range:
        # simple slope at this novelty value = main_effect + interaction * novelty_z
        main = model.params.get(key_var.split(":")[0], 0)
        inter_var = key_var
        inter_coef = model.params.get(inter_var, 0)
        slope = main + inter_coef * nov_z
        # SE via delta method: Var = Var(main) + nov_z²Var(inter) + 2*nov_z*Cov
        cov = model.cov_params()
        main_var_name = key_var.split(":")[0]
        try:
            v_main  = cov.loc[main_var_name, main_var_name]
            v_inter = cov.loc[inter_var, inter_var]
            c_mi    = cov.loc[main_var_name, inter_var]
            var = v_main + nov_z**2 * v_inter + 2 * nov_z * c_mi
        except Exception:
            var = np.nan
        se = np.sqrt(max(var, 0)) if not np.isnan(var) else np.nan
        lo = slope - 1.96 * se if se else np.nan
        hi = slope + 1.96 * se if se else np.nan
        results.append({"novelty_z": nov_z, "slope": slope, "lo": lo, "hi": hi})

    df_jn = pd.DataFrame(results)
    # find where CI crosses zero
    sig_low  = df_jn[df_jn["lo"] > 0]["novelty_z"].min() if (df_jn["lo"] > 0).any() else None
    sig_high = df_jn[df_jn["hi"] < 0]["novelty_z"].max() if (df_jn["hi"] < 0).any() else None
    return {"data": df_jn, "sig_low": sig_low, "sig_high": sig_high}


def marginal_effects(model, df, outcome_type="logit") -> pd.DataFrame:
    """
    Predict P(outcome) across novelty range for each investor group,
    holding all controls at their modal / mean values.
    Returns df with columns: novelty_z, group, pred, lo, hi
    """
    base = {
        "novelty_z":    0.0, "novelty_z2": 0.0,
        "company_age_z": 0.0, "log_raised": df["log_raised"].mean(),
        "inv_year":     df["inv_year"].mode()[0],
        "region":       df["region"].mode()[0],
        "post_trump":   0,
    }
    for c in ["oil_price_z","interest_rate_z","vix_z"]:
        base[c] = 0.0

    rows = []
    for nov_z in NOVELTY_RANGE_Z:
        for group, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
            row = {**base, "novelty_z": nov_z, "novelty_z2": nov_z**2,
                   "Pure_Green": pg, "Pure_NG": png,
                   "Pure_Green_x_post_trump": pg*0,
                   "Pure_NG_x_post_trump":    png*0,
                   "novelty_z_x_post_trump":  nov_z*0}
            try:
                pred_df = pd.DataFrame([row])
                if outcome_type == "logit":
                    pred = model.predict(pred_df)[0]
                    # Delta method CI on probability scale
                    from statsmodels.genmod.generalized_linear_model import GLM
                    cov = model.cov_params()
                    rows.append({"novelty_z": nov_z, "group": group,
                                "pred": pred, "lo": None, "hi": None})
                else:
                    pred = model.predict(pred_df)[0]
                    rows.append({"novelty_z": nov_z, "group": group,
                                "pred": pred, "lo": None, "hi": None})
            except Exception:
                rows.append({"novelty_z": nov_z, "group": group, "pred": np.nan, "lo": None, "hi": None})

    mdf = pd.DataFrame(rows)
    # Bootstrap CI (200 draws for speed)
    rng = np.random.default_rng(42)
    for group, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
        boot_preds = []
        params_cov = model.cov_params().values
        mean_params = model.params.values
        try:
            sampled = rng.multivariate_normal(mean_params, params_cov, size=200)
        except Exception:
            continue
        for nov_z in NOVELTY_RANGE_Z:
            row = {**base, "novelty_z": nov_z, "novelty_z2": nov_z**2,
                   "Pure_Green": pg, "Pure_NG": png,
                   "Pure_Green_x_post_trump": pg*0,
                   "Pure_NG_x_post_trump":    png*0,
                   "novelty_z_x_post_trump":  nov_z*0}
            try:
                pred_df = pd.DataFrame([row])
                linpred = model.model.predict(model.params, exog=model.model.exog[:1])
                design  = model.model.exog[:1] * 0
                dm      = pd.get_dummies(pd.DataFrame([row]), drop_first=False)
                # simpler: just ±1.96*SE of the linear predictor
                lp = model.predict(pred_df, linear=True)[0]
                lp_arr = np.array([np.dot(sampled[i], model.model.predict(
                    model.params, pred_df, linear=True).__class__) for i in range(200)])
                preds_b = 1 / (1 + np.exp(-np.array([model.predict(
                    pd.DataFrame([row]))[0] for _ in range(1)])))
            except Exception:
                pass
            boot_preds.append(np.nan)

    # Simpler: ±1.96*SE approach on linear predictor → convert to probability
    for group, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
        mask = mdf["group"] == group
        for i, nov_z in enumerate(NOVELTY_RANGE_Z):
            row = {**base, "novelty_z": nov_z, "novelty_z2": nov_z**2,
                   "Pure_Green": pg, "Pure_NG": png,
                   "Pure_Green_x_post_trump": pg*0,
                   "Pure_NG_x_post_trump":    png*0,
                   "novelty_z_x_post_trump":  nov_z*0}
            try:
                pred_df = pd.DataFrame([row])
                lp      = float(model.predict(pred_df, linear=True).iloc[0])
                se_lp   = float(np.sqrt(max(
                    pred_df.values @ model.cov_params().values @ pred_df.values.T, 0
                )[0][0]))
                lo_lp, hi_lp = lp - 1.96*se_lp, lp + 1.96*se_lp
                lo_p = 1/(1+np.exp(-lo_lp)); hi_p = 1/(1+np.exp(-hi_lp))
                idx = mdf[(mdf["group"]==group) & (np.isclose(mdf["novelty_z"],nov_z))].index
                if len(idx):
                    mdf.loc[idx[0],"lo"] = min(lo_p,hi_p)
                    mdf.loc[idx[0],"hi"] = max(lo_p,hi_p)
            except Exception:
                pass
    return mdf


# ─────────────────────────── MAIN PLOT FUNCTION ──────────────────────────────

def plot_marginal(ax, mdf, title, ylabel="Predicted Probability"):
    for grp in ["Pure Green","Mixed","Pure Non-Green"]:
        sub = mdf[mdf["group"]==grp].sort_values("novelty_z")
        ax.plot(sub["novelty_z"], sub["pred"]*100, color=PALETTE[grp],
               linewidth=2.5, label=grp)
        if sub["lo"].notna().any():
            ax.fill_between(sub["novelty_z"], sub["lo"]*100, sub["hi"]*100,
                           alpha=0.15, color=PALETTE[grp])
    ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Novelty Score (SD units; 0 = mean)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))


def coef_summary(model, vars_of_interest: list, label: str):
    rows = []
    for v in vars_of_interest:
        if v not in model.params: continue
        c = model.params[v]; p = model.pvalues[v]
        lo, hi = model.conf_int().loc[v]
        stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "(†)" if p<0.15 else ""
        rows.append({"model": label, "variable": v, "coef": round(c,4),
                     "lo": round(lo,4), "hi": round(hi,4), "p": round(p,4), "sig": stars})
    return rows


# ─────────────────────────── RUN ─────────────────────────────────────────────

def main():
    df = load_data()
    ctrl = base_ctrl(df)
    all_coefs = []

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Moderation Analysis: Does Novelty Moderate the Investor Composition Effect?\n"
                 "SPECTER2 novelty score (standardised) × Syndicate Type → Exit / Failure / Survival",
                 fontsize=13, fontweight="bold", y=1.01)

    # pre-compute interaction columns (statsmodels formula handles this, but for Cox we need manual)
    df["PG_x_nov"]    = df["Pure_Green"] * df["novelty_z"]
    df["PNG_x_nov"]   = df["Pure_NG"]    * df["novelty_z"]
    df["PG_x_nov2"]   = df["Pure_Green"] * df["novelty_z2"]
    df["PNG_x_nov2"]  = df["Pure_NG"]    * df["novelty_z2"]
    df["PG_x_trump"]  = df["Pure_Green"] * df["post_trump"]
    df["PNG_x_trump"] = df["Pure_NG"]    * df["post_trump"]
    df["nov_x_trump"] = df["novelty_z"]  * df["post_trump"]
    df["PG_nov_trump"]  = df["Pure_Green"] * df["novelty_z"] * df["post_trump"]
    df["PNG_nov_trump"] = df["Pure_NG"]    * df["novelty_z"] * df["post_trump"]

    interaction_terms = "novelty_z*Pure_Green + novelty_z*Pure_NG"
    key_vars = ["Pure_Green","Pure_NG","novelty_z",
                "novelty_z:Pure_Green","novelty_z:Pure_NG"]

    simple_slope_terms = {
        "Mixed":          "novelty_z",
        "Pure Green":     "novelty_z + novelty_z:Pure_Green",
        "Pure Non-Green": "novelty_z + novelty_z:Pure_NG",
    }

    print("\n" + "="*60)
    print("MODELS 1–3: EXIT / FAILURE / SURVIVAL (logit)")
    print("="*60)
    results = {}
    for col, outcome_col, row, title in [
        ("exit",     "exited",   0, "Model 1 — Exit Success"),
        ("failure",  "failed",   1, "Model 2 — Failure"),
        ("survival", "survived", 2, "Model 3 — Survival"),
    ]:
        print(f"\n--- {title} ---")
        formula = (f"{outcome_col} ~ Pure_Green + Pure_NG + {interaction_terms}"
                   f" + {ctrl}")
        m = fit_logit(formula, df, title)
        if m is None: continue
        results[col] = m

        # Simple slopes
        ss = simple_slopes(m, simple_slope_terms)
        print(ss.to_string(index=False))

        # J-N (for Pure Non-Green × novelty)
        jn = johnson_neyman(m, "novelty_z:Pure_NG")
        jn_msg = []
        if jn["sig_low"]:  jn_msg.append(f"J-N lower bound: novelty_z > {jn['sig_low']:.2f}")
        if jn["sig_high"]: jn_msg.append(f"J-N upper bound: novelty_z < {jn['sig_high']:.2f}")
        if jn_msg: print("  " + " | ".join(jn_msg))

        # Marginal effects
        mdf = marginal_effects(m, df)
        ax = axes[outcome_col[1] if isinstance(outcome_col, tuple) else row, 0]
        plot_marginal(ax, mdf, title)
        all_coefs.extend(coef_summary(m, key_vars, title))

    print("\n" + "="*60)
    print("MODEL 4: COX PROPORTIONAL HAZARDS (failure event)")
    print("="*60)
    cox_vars = ["Pure_Green","Pure_NG","novelty_z","PG_x_nov","PNG_x_nov",
                "company_age_z","log_raised"]
    for c in ["oil_price_z","interest_rate_z","vix_z"]:
        if c in df.columns: cox_vars.append(c)
    cox_df = df[["duration","km_event"] + cox_vars].dropna()
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_df, duration_col="duration", event_col="km_event")
        print(cph.summary[["coef","exp(coef)","p"]].round(4).to_string())
        ax = axes[0, 1]
        for grp, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
            nov_range = NOVELTY_RANGE_Z
            haz = np.exp(
                pg * cph.params_.get("Pure_Green",0)  +
                png * cph.params_.get("Pure_NG",0)   +
                nov_range * cph.params_.get("novelty_z",0) +
                pg * nov_range * cph.params_.get("PG_x_nov",0) +
                png * nov_range * cph.params_.get("PNG_x_nov",0)
            )
            ax.plot(nov_range, haz, color=PALETTE[grp], linewidth=2.5, label=grp)
        ax.axhline(1, color="grey", lw=0.8, ls=":")
        ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("Novelty Score (SD units)")
        ax.set_ylabel("Hazard Ratio (vs mean covariate)")
        ax.set_title("Model 4 — Cox: Failure Hazard\n× Novelty × Syndicate",
                    fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)
        # extract Cox key params for summary
        for v in ["Pure_Green","Pure_NG","novelty_z","PG_x_nov","PNG_x_nov"]:
            if v in cph.params_.index:
                c = cph.params_[v]; p = cph.summary.loc[v,"p"]
                stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "(†)" if p<0.15 else ""
                lo, hi = cph.confidence_intervals_.loc[v]
                all_coefs.append({"model":"Model 4 (Cox)","variable":v,
                                  "coef":round(c,4),"lo":round(lo,4),
                                  "hi":round(hi,4),"p":round(p,4),"sig":stars})
    except Exception as e:
        print(f"Cox failed: {e}")

    print("\n" + "="*60)
    print("MODEL 5: QUADRATIC NOVELTY")
    print("="*60)
    formula5 = (f"exited ~ Pure_Green + Pure_NG + novelty_z + novelty_z2"
                f" + novelty_z:Pure_Green + novelty_z:Pure_NG"
                f" + novelty_z2:Pure_Green + novelty_z2:Pure_NG"
                f" + {ctrl}")
    m5 = fit_logit(formula5, df, "Model 5 — Quadratic")
    if m5:
        ax = axes[1, 1]
        for grp, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
            preds = []
            base_row = {"Pure_Green":pg,"Pure_NG":png,"company_age_z":0,
                       "log_raised":df["log_raised"].mean(),
                       "inv_year":df["inv_year"].mode()[0],
                       "region":df["region"].mode()[0],"post_trump":0,
                       **{"oil_price_z":0,"interest_rate_z":0,"vix_z":0,
                          "Pure_Green_x_post_trump":0,"Pure_NG_x_post_trump":0,
                          "novelty_z_x_post_trump":0}}
            for nz in NOVELTY_RANGE_Z:
                r = {**base_row,"novelty_z":nz,"novelty_z2":nz**2}
                try: preds.append(float(m5.predict(pd.DataFrame([r])).iloc[0])*100)
                except: preds.append(np.nan)
            ax.plot(NOVELTY_RANGE_Z, preds, color=PALETTE[grp], linewidth=2.5, label=grp)
        ax.axvline(0,color="grey",lw=0.8,ls=":",alpha=0.7)
        ax.set_xlabel("Novelty Score (SD units)")
        ax.set_ylabel("Predicted Exit Probability (%)")
        ax.set_title("Model 5 — Quadratic Novelty\n(tests inverted-U non-linearity)",
                    fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
        all_coefs.extend(coef_summary(m5,
            ["Pure_Green","Pure_NG","novelty_z","novelty_z2",
             "novelty_z:Pure_Green","novelty_z:Pure_NG",
             "novelty_z2:Pure_Green","novelty_z2:Pure_NG"], "Model 5 (quadratic)"))
        # print quadratic terms
        for v in ["novelty_z2","novelty_z2:Pure_Green","novelty_z2:Pure_NG"]:
            if v in m5.params:
                print(f"  {v}: coef={m5.params[v]:+.4f}  p={m5.pvalues[v]:.4f}")

    print("\n" + "="*60)
    print("MODEL 6: THREE-WAY INTERACTION × TRUMP 2016")
    print("="*60)
    formula6 = (f"exited ~ Pure_Green + Pure_NG + novelty_z + post_trump"
                f" + PG_x_nov + PNG_x_nov"
                f" + PG_x_trump + PNG_x_trump + nov_x_trump"
                f" + PG_nov_trump + PNG_nov_trump"
                f" + {ctrl}")
    m6 = fit_logit(formula6, df, "Model 6 — Three-way")
    if m6:
        ax = axes[1, 2]
        for trump_val, ls in [(0,"-"),(1,"--")]:
            for grp, pg, png in [("Pure Green",1,0),("Mixed",0,0),("Pure Non-Green",0,1)]:
                preds = []
                base_row = {"Pure_Green":pg,"Pure_NG":png,"post_trump":trump_val,
                           "novelty_z2":0,"company_age_z":0,
                           "log_raised":df["log_raised"].mean(),
                           "inv_year":df["inv_year"].mode()[0],
                           "region":df["region"].mode()[0],
                           **{"oil_price_z":0,"interest_rate_z":0,"vix_z":0,
                              "Pure_Green_x_post_trump":0,"Pure_NG_x_post_trump":0,
                              "novelty_z_x_post_trump":0}}
                for nz in NOVELTY_RANGE_Z:
                    r = {**base_row,"novelty_z":nz,
                         "PG_x_nov":pg*nz,"PNG_x_nov":png*nz,
                         "PG_x_trump":pg*trump_val,"PNG_x_trump":png*trump_val,
                         "nov_x_trump":nz*trump_val,
                         "PG_nov_trump":pg*nz*trump_val,
                         "PNG_nov_trump":png*nz*trump_val}
                    try: preds.append(float(m6.predict(pd.DataFrame([r])).iloc[0])*100)
                    except: preds.append(np.nan)
                label = f"{grp} {'(Trump 1)' if trump_val else '(Pre-Trump)'}"
                ax.plot(NOVELTY_RANGE_Z, preds, color=PALETTE[grp],
                       linewidth=2, linestyle=ls, label=label, alpha=0.9)
        ax.axvline(0,color="grey",lw=0.8,ls=":",alpha=0.7)
        ax.set_xlabel("Novelty Score (SD units)")
        ax.set_ylabel("Predicted Exit Probability (%)")
        ax.set_title("Model 6 — Three-way Interaction\nSyndicate × Novelty × Trump 2016",
                    fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
        # print three-way coefficients
        for v in ["PG_nov_trump","PNG_nov_trump","nov_x_trump"]:
            if v in m6.params:
                print(f"  {v}: coef={m6.params[v]:+.4f}  p={m6.pvalues[v]:.4f}")
        all_coefs.extend(coef_summary(m6,
            ["Pure_Green","Pure_NG","novelty_z","post_trump",
             "PG_x_nov","PNG_x_nov","nov_x_trump","PG_nov_trump","PNG_nov_trump"],
            "Model 6 (three-way)"))

    # ── Coefficient forest plot (all models, key vars only) ─────────────────
    coef_df = pd.DataFrame(all_coefs)
    coef_df = coef_df[coef_df["variable"].isin(
        ["Pure_Green","Pure_NG","novelty_z",
         "novelty_z:Pure_Green","novelty_z:Pure_NG",
         "PG_x_nov","PNG_x_nov","PG_nov_trump","PNG_nov_trump"]
    )]
    var_labels = {
        "Pure_Green":           "Pure Green\n(main effect)",
        "Pure_NG":              "Pure Non-Green\n(main effect)",
        "novelty_z":            "Novelty\n(main effect)",
        "novelty_z:Pure_Green": "Novelty ×\nPure Green",
        "novelty_z:Pure_NG":    "Novelty ×\nPure Non-Green",
        "PG_x_nov":             "Novelty ×\nPure Green (Cox/M6)",
        "PNG_x_nov":            "Novelty ×\nPure Non-Green (Cox/M6)",
        "PG_nov_trump":         "Nov × PG ×\nTrump 2016",
        "PNG_nov_trump":        "Nov × PNG ×\nTrump 2016",
    }
    model_colors_all = {
        "Model 1 — Exit Success": "#2980b9",
        "Model 2 — Failure":      "#e74c3c",
        "Model 3 — Survival":     "#27ae60",
        "Model 4 (Cox)":          "#9b59b6",
        "Model 5 (quadratic)":    "#e67e22",
        "Model 6 (three-way)":    "#1abc9c",
    }
    ax_forest = axes[2, 0]
    unique_vars = list(var_labels.keys())
    y_ticks = {v: i for i, v in enumerate(unique_vars)}
    models_in_data = coef_df["model"].unique()
    n_models = len(models_in_data)
    offsets_map = {m: (-0.3 + 0.6*(i/(max(n_models-1,1)))) for i, m in enumerate(models_in_data)}

    for _, row in coef_df.iterrows():
        var = row["variable"]; mname = row["model"]
        if var not in y_ticks: continue
        y = y_ticks[var] + offsets_map.get(mname, 0)
        color = model_colors_all.get(mname, "grey")
        ax_forest.errorbar(row["coef"], y, xerr=[[row["coef"]-row["lo"]],[row["hi"]-row["coef"]]],
                          fmt="o", color=color, markersize=6, capsize=3, linewidth=1.5,
                          label=mname if var == unique_vars[0] else "")
        if row["sig"] in ("*","**","***","(†)"):
            ax_forest.text(row["hi"]+0.02, y, row["sig"], va="center", fontsize=8, color=color)

    ax_forest.axvline(0, color="black", lw=1, ls="--")
    ax_forest.set_yticks(list(y_ticks.values()))
    ax_forest.set_yticklabels([var_labels[v] for v in unique_vars], fontsize=8)
    ax_forest.set_xlabel("Logit Coefficient (or log Hazard Ratio for Cox)")
    ax_forest.set_title("Coefficient Forest Plot\n(all models, key variables)",
                        fontsize=10, fontweight="bold")
    ax_forest.legend(fontsize=7, loc="lower right")
    ax_forest.grid(axis="x", linestyle="--", alpha=0.4)

    # ── Simple slopes summary table ──────────────────────────────────────────
    ax_ss = axes[2, 1]
    ax_ss.axis("off")
    if results:
        ss_all = []
        for mname, m in [("Exit",results.get("exit",results.get(0))),
                          ("Failure",results.get("failure",results.get(1))),
                          ("Survival",results.get("survival",results.get(2)))]:
            if m is None: continue
            ss = simple_slopes(m, simple_slope_terms)
            ss["Model"] = mname
            ss_all.append(ss)
        if ss_all:
            ss_df = pd.concat(ss_all)
            tbl = ax_ss.table(cellText=ss_df[["Model","Group","Slope","SE","p","Sig"]].values,
                              colLabels=["Model","Group","Slope","SE","p","Sig"],
                              cellLoc="center", loc="center",
                              colWidths=[0.12,0.18,0.12,0.10,0.10,0.08])
            tbl.auto_set_font_size(False); tbl.set_fontsize(8)
            tbl.scale(1, 1.4)
            ax_ss.set_title("Simple Slopes\n(novelty slope within each investor group)",
                           fontsize=10, fontweight="bold")
            print("\n=== SIMPLE SLOPES SUMMARY ===")
            print(ss_df[["Model","Group","Slope","SE","p","Sig"]].to_string(index=False))

    # ── J-N plot ─────────────────────────────────────────────────────────────
    ax_jn = axes[2, 2]
    m1 = results.get("exit", results.get(0))
    if m1 is not None:
        jn_pg  = johnson_neyman(m1, "novelty_z:Pure_Green")
        jn_png = johnson_neyman(m1, "novelty_z:Pure_NG")
        for jn, grp in [(jn_pg,"Pure Green"),(jn_png,"Pure Non-Green")]:
            d = jn["data"]
            ax_jn.plot(d["novelty_z"], d["slope"], color=PALETTE[grp], lw=2, label=grp)
            ax_jn.fill_between(d["novelty_z"], d["lo"], d["hi"], alpha=0.2, color=PALETTE[grp])
        ax_jn.axhline(0, color="black", lw=1, ls="--")
        ax_jn.axvline(0, color="grey",  lw=0.8, ls=":", alpha=0.7)
        # shade significant zones
        jn_png_d = jn_png["data"]
        sig_zone = jn_png_d[jn_png_d["lo"] > 0]
        if len(sig_zone):
            ax_jn.axvspan(sig_zone["novelty_z"].min(), sig_zone["novelty_z"].max(),
                         alpha=0.08, color=PALETTE["Pure Non-Green"], label="Sig. zone PNG")
        ax_jn.set_xlabel("Novelty Score (SD units)")
        ax_jn.set_ylabel("Simple Slope (log-odds; Model 1, Exit)")
        ax_jn.set_title("Johnson-Neyman Intervals\n(where syndicate × novelty effect becomes significant)",
                        fontsize=10, fontweight="bold")
        ax_jn.legend(fontsize=8)
        ax_jn.grid(linestyle="--", alpha=0.4)
        ax_jn.text(0.02, 0.04,
                  "Shaded = 95% CI excludes 0 (significant region)\n"
                  "Operates on log-odds scale (logit Model 1)",
                  transform=ax_jn.transAxes, fontsize=7.5, color="grey")

    # hide unused subplots
    for ax in [axes[0,2], axes[1,0]]:
        ax.axis("off")

    plt.tight_layout()
    out_path = OUT / "moderation_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    coef_df.to_csv(OUT / "moderation_coefficients.csv", index=False)
    print(f"\nSaved → {out_path}")
    print(f"Saved → {OUT / 'moderation_coefficients.csv'}")


if __name__ == "__main__":
    main()
