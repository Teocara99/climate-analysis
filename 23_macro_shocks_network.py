"""
23 — Macro shocks × cross-community investor network fragmentation.

Tests whether VIX, oil, interest rates, GPR, and CPU each fragment
the cross-community syndication network over time, and which matters most.

Steps:
  1. Build quarterly cross-community share time series
  2. Step 1: Each shock → cross-community share (VAR-style lags, quarterly)
  3. Step 2: Mediation analysis (Bootstrap Sobel) for significant disruptors
  4. Step 3: Horse race — all shocks simultaneously
  5. Step 4: Mediation comparison table

Outputs → output/macro_shocks/
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import re, sys, json
from pathlib import Path
from collections import defaultdict
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT  = ROOT / "output" / "macro_shocks"
OUT.mkdir(parents=True, exist_ok=True)

CENSORING = pd.Timestamp("2026-07-01")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: BUILD QUARTERLY CROSS-COMMUNITY SHARE TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_quarterly_panel():
    print("Building quarterly cross-community panel …")

    deals_raw = load_deals()
    deals_raw["Deal Date"] = pd.to_datetime(deals_raw["Announced Date"], errors="coerce")
    deals_raw = deals_raw.dropna(subset=["Deal Date"])
    deals_raw["YQ"] = deals_raw["Deal Date"].dt.to_period("Q")

    # Investor community map: investor name → community ID
    metrics = pd.read_csv(ROOT / "output/network/network_metrics.csv")
    def clean_name(n):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", str(n).lower())).strip()
    metrics["inv_clean"] = metrics["investor"].apply(clean_name)
    inv_comm = dict(zip(metrics["inv_clean"], metrics["community"]))
    clf_set  = set(inv_comm.keys())

    # For each deal, determine how many distinct investor communities
    def n_communities_for_deal(investors_str):
        if pd.isna(investors_str): return 0
        names = [clean_name(n) for n in re.split(r",\s*", str(investors_str).replace("\n", ", "))]
        comms = {inv_comm[n] for n in names if n in clf_set}
        return len(comms)

    print("  Computing deal-level cross-community flag …")
    deals_raw["n_comms"] = deals_raw["Investors"].apply(n_communities_for_deal)
    deals_raw["is_cross"] = (deals_raw["n_comms"] >= 2).astype(int)
    deals_raw["has_classified"] = (deals_raw["n_comms"] >= 1).astype(int)

    # Quarterly aggregate: fraction of classified deals that are cross-community
    qdf = deals_raw[deals_raw["has_classified"] == 1].groupby("YQ").agg(
        n_deals     = ("Deal ID", "count"),
        n_cross     = ("is_cross", "sum"),
    ).reset_index()
    qdf["pct_cross"] = qdf["n_cross"] / qdf["n_deals"]
    qdf["date"]      = qdf["YQ"].dt.to_timestamp()
    qdf = qdf[qdf["n_deals"] >= 10].copy()   # drop tiny quarters

    print(f"  Quarterly obs: {len(qdf)}  ({qdf['date'].min().year}–{qdf['date'].max().year})")
    print(f"  Mean pct_cross: {qdf['pct_cross'].mean()*100:.1f}%  "
          f"SD: {qdf['pct_cross'].std()*100:.1f}%")
    return qdf


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0b: BUILD QUARTERLY MACRO PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_macro_panel():
    print("Building quarterly macro panel …")

    # FRED (monthly → quarterly mean)
    fred = pd.read_csv(DATA / "fred_economic_controls.csv")
    fred["date"] = pd.to_datetime(fred["date"])
    fred["YQ"]   = fred["date"].dt.to_period("Q")
    fred_q = fred.groupby("YQ")[["oil_price","interest_rate","vix"]].mean().reset_index()

    # CPU (monthly → quarterly mean, use CPU_US / Gavriilidis)
    cpu_m = pd.read_csv(DATA / "cpu_multicountry.csv")
    cpu_m["date"] = pd.to_datetime(cpu_m["year_month"] + "-01")
    cpu_m["YQ"]   = cpu_m["date"].dt.to_period("Q")
    cpu_g = pd.read_csv(DATA / "cpu_gavriilidis_us.csv")
    # Try to get CPU column
    cpu_col = [c for c in cpu_g.columns if "cpu" in c.lower() or "CPU" in c]
    print(f"  Gavriilidis CPU cols: {list(cpu_g.columns)}")
    cpu_g["date"] = pd.to_datetime(cpu_g.iloc[:, 0].astype(str) + "-01", errors="coerce")
    cpu_g["YQ"]   = cpu_g["date"].dt.to_period("Q")
    # Use cpu_narrow from multicountry (CPU_US)
    cpu_q = cpu_m.groupby("YQ")["CPU_US"].mean().reset_index().rename(columns={"CPU_US": "cpu"})

    # GPR (daily → quarterly mean)
    gpr_daily = pd.read_excel("/tmp/gpr_data.xls", engine="xlrd")
    gpr_daily["date_parsed"] = pd.to_datetime(gpr_daily["date"].astype(str), errors="coerce")
    gpr_daily = gpr_daily.dropna(subset=["date_parsed", "GPRD"])
    gpr_daily = gpr_daily[gpr_daily["GPRD"] > 0]
    gpr_daily["YQ"] = gpr_daily["date_parsed"].dt.to_period("Q")
    gpr_q = gpr_daily.groupby("YQ")["GPRD"].mean().reset_index().rename(columns={"GPRD": "gpr"})

    # Merge all
    macro = fred_q.merge(cpu_q, on="YQ", how="left")
    macro = macro.merge(gpr_q, on="YQ", how="left")
    macro["date"] = macro["YQ"].dt.to_timestamp()

    # Standardise
    for col in ["oil_price","interest_rate","vix","cpu","gpr"]:
        macro[f"{col}_z"] = (macro[col] - macro[col].mean()) / macro[col].std()

    print(f"  Macro panel: {len(macro)} quarters, "
          f"oil/rate/vix/cpu/gpr all present: "
          f"{macro[['oil_price','interest_rate','vix','cpu','gpr']].notna().all(axis=1).sum()}")
    return macro


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: EACH SHOCK → CROSS-COMMUNITY SHARE (with 1-2 lags)
# ═══════════════════════════════════════════════════════════════════════════════

def run_step1(panel, lines):
    print("\nStep 1: shock → cross-community …")
    shocks = {
        "VIX":           "vix_z",
        "Oil Price":     "oil_price_z",
        "Interest Rate": "interest_rate_z",
        "GPR":           "gpr_z",
        "CPU":           "cpu_z",
    }

    # Add lags of each shock and of pct_cross itself
    p = panel.copy().sort_values("date").reset_index(drop=True)
    for col in [s for s in shocks.values()] + ["pct_cross"]:
        p[f"{col}_l1"] = p[col].shift(1)
        p[f"{col}_l2"] = p[col].shift(2)
    p["quarter_trend"] = np.arange(len(p))
    p["q2"] = (p["date"].dt.quarter == 2).astype(int)
    p["q3"] = (p["date"].dt.quarter == 3).astype(int)
    p["q4"] = (p["date"].dt.quarter == 4).astype(int)

    controls = "pct_cross_l1 + quarter_trend + q2 + q3 + q4"

    def sig(p_):
        return "***" if p_ < 0.001 else ("**" if p_ < 0.01 else ("*" if p_ < 0.05 else ("†" if p_ < 0.10 else "n.s.")))

    lines += ["\n" + "="*72,
              "STEP 1: SHOCK → QUARTERLY CROSS-COMMUNITY SHARE",
              "  DV: pct_cross(t); Controls: pct_cross(t-1), trend, quarter dummies",
              "="*72]

    step1_results = {}
    for name, col in shocks.items():
        sub = p.dropna(subset=["pct_cross", f"{col}_l1", f"{col}_l2", "pct_cross_l1"])
        formula = f"pct_cross ~ {col}_l1 + {col}_l2 + {controls}"
        try:
            m = smf.ols(formula, data=sub).fit(cov_type="HC3")
            b1 = m.params[f"{col}_l1"]; p1 = m.pvalues[f"{col}_l1"]
            b2 = m.params[f"{col}_l2"]; p2 = m.pvalues[f"{col}_l2"]
            n  = int(m.nobs)
            r2 = m.rsquared_adj
            # Combined F-test for both lags
            from statsmodels.stats.anova import anova_lm
            m0 = smf.ols(f"pct_cross ~ {controls}", data=sub).fit(cov_type="HC3")
            f_stat = ((m0.ssr - m.ssr) / 2) / (m.ssr / (n - m.df_model - 1))
            f_p    = 1 - stats.f.cdf(f_stat, 2, n - m.df_model - 1)
            step1_results[name] = {"col": col, "b1": b1, "p1": p1, "b2": b2, "p2": p2,
                                   "f_p": f_p, "n": n, "r2": r2, "model": m}
            sig1 = sig(p1); sig2 = sig(p2); sigf = sig(f_p)
            lines.append(f"\n  [{name}]  N={n} quarters  adj-R²={r2:.3f}  F-test p={f_p:.3f} {sigf}")
            lines.append(f"    {col}_l1: β={b1:+.4f}  p={p1:.3f} {sig1}")
            lines.append(f"    {col}_l2: β={b2:+.4f}  p={p2:.3f} {sig2}")
        except Exception as e:
            lines.append(f"\n  [{name}] ERROR: {e}")

    return step1_results, p


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: MEDIATION — BOOTSTRAP SOBEL
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_sobel(company_df, shock_col, med_col, outcome_col, n_boot=1000, seed=42):
    """
    Path A: shock → mediator (OLS, company-level)
    Path B: mediator → outcome (Logit, with shock as control)
    Path C': shock → outcome direct (Logit)
    Indirect effect = a × b  (delta method + bootstrap CI)
    """
    rng = np.random.default_rng(seed)
    df  = company_df[[shock_col, med_col, outcome_col,
                       "hq_region","log_capital","deal_year","n_rounds"]].dropna().copy()
    controls = "C(hq_region) + log_capital + deal_year + n_rounds"

    def fit_paths(d):
        # Path A: shock → mediator
        mA = smf.ols(f"{med_col} ~ {shock_col} + {controls}", data=d).fit()
        a  = mA.params[shock_col]; se_a = mA.bse[shock_col]

        # Path B + C': mediator + shock → outcome
        mBC = smf.logit(f"{outcome_col} ~ {med_col} + {shock_col} + {controls}", data=d).fit(disp=0, maxiter=400)
        b   = mBC.params.get(med_col,  np.nan); se_b = mBC.bse.get(med_col,   np.nan)
        c_p = mBC.params.get(shock_col, np.nan)

        # Path C (total): shock → outcome
        mC  = smf.logit(f"{outcome_col} ~ {shock_col} + {controls}", data=d).fit(disp=0, maxiter=400)
        c   = mC.params.get(shock_col, np.nan); p_c = mC.pvalues.get(shock_col, np.nan)
        p_a = mA.pvalues[shock_col]; p_b = mBC.pvalues.get(med_col, np.nan)

        indirect = a * b
        return dict(a=a, se_a=se_a, b=b, se_b=se_b, c=c, c_p=c_p,
                    c_prime=c_p, indirect=indirect, p_a=p_a, p_b=p_b, p_c=p_c,
                    n=len(d))

    point = fit_paths(df)
    if np.isnan(point["indirect"]):
        return point, np.nan, np.nan, np.nan

    # Bootstrap
    boot_indirect = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        try:
            res = fit_paths(df.iloc[idx])
            boot_indirect.append(res["indirect"])
        except Exception:
            pass

    boot_arr = np.array([x for x in boot_indirect if np.isfinite(x)])
    ci_lo = np.percentile(boot_arr, 2.5)
    ci_hi = np.percentile(boot_arr, 97.5)
    p_boot = 2 * min(np.mean(boot_arr >= 0), np.mean(boot_arr <= 0))

    # Proportion mediated
    total = point["indirect"] + point["c_prime"]
    prop_med = abs(point["indirect"] / total) if abs(total) > 1e-9 else np.nan

    return point, ci_lo, ci_hi, prop_med


def run_step2(step1_results, company_df, lines):
    print("\nStep 2: Mediation analysis …")

    # Identify significant disruptors from Step 1 (F-test p < 0.10)
    sig_shocks = {k: v for k, v in step1_results.items() if v.get("f_p", 1) < 0.10}
    # Always include CPU for comparison
    all_test = dict(sig_shocks)
    if "CPU" not in all_test and "CPU" in step1_results:
        all_test["CPU"] = step1_results["CPU"]

    lines += ["\n" + "="*72,
              "STEP 2: BOOTSTRAP MEDIATION ANALYSIS (1,000 replications)",
              "  Path A: shock(company-level) → cross_community",
              "  Path B: cross_community → failure (logit)",
              "  Path C': shock → failure (direct, logit)",
              "  Indirect effect = A × B  [95% bias-corrected bootstrap CI]",
              "="*72]

    if not all_test:
        lines.append("  No shocks significant at p<0.10 in Step 1 — skipping mediation.")
        return {}

    # Use company-level shock values (cross-sectional: each company's first-deal quarter)
    shock_map = {
        "VIX":           "vix_z",
        "Oil Price":     "oil_price_z",
        "Interest Rate": "interest_rate_z",
        "GPR":           "gpr_z",
        "CPU":           "cpu_narrow_z",
    }
    mediation_results = {}
    for name, info in all_test.items():
        scol = shock_map.get(name, info["col"])
        if scol not in company_df.columns:
            lines.append(f"\n  [{name}] shock column '{scol}' not in company dataset — skip")
            continue
        df_m = company_df.rename(columns={"n_communities": "n_comms_raw"}).copy()
        df_m["log_capital"] = np.log1p(df_m["Total Raised"].fillna(0))
        df_m["hq_region"]   = df_m["hq_region"].fillna("Unknown")
        df_m["deal_year"]   = df_m["deal_year"].fillna(2018)
        df_m["n_rounds"]    = df_m["n_rounds"].fillna(1)

        for outcome, olabel in [("failed","FAILURE"), ("exited","EXIT")]:
            if outcome not in df_m.columns: continue
            print(f"    Bootstrapping {name} → cross_community → {outcome} …")
            pt, ci_lo, ci_hi, prop = bootstrap_sobel(
                df_m, scol, "cross_community", outcome)

            def sig(p): return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))
            lines += [
                f"\n  [{name} → cross_community → {olabel}]  N={pt['n']:,}",
                f"    Path A (shock→cross_comm):  β={pt['a']:+.4f}  p={pt['p_a']:.3f} {sig(pt['p_a'])}",
                f"    Path B (cross_comm→{outcome}): β={pt['b']:+.4f}  p={pt['p_b']:.3f} {sig(pt['p_b'])}",
                f"    Path C (shock→{outcome} direct): β={pt['c']:+.4f}  p={pt['p_c']:.3f} {sig(pt['p_c'])}",
                f"    Indirect (A×B):  {pt['indirect']:+.6f}  95%CI [{ci_lo:+.6f}, {ci_hi:+.6f}]  p_boot={prop:.3f}" if not np.isnan(ci_lo) else f"    Indirect (A×B):  {pt['indirect']:+.6f}  CI: n/a",
            ]
            if not np.isnan(prop) and not np.isnan(ci_lo):
                includes_zero = ci_lo < 0 < ci_hi
                verdict = "NOT significant (CI includes 0)" if includes_zero else "SIGNIFICANT mediation"
                lines.append(f"    Verdict: {verdict}  |  Prop. mediated ≈ {prop*100:.1f}%")
            mediation_results[f"{name}_{outcome}"] = dict(
                name=name, outcome=outcome, **pt,
                ci_lo=ci_lo, ci_hi=ci_hi, prop_med=prop)

    return mediation_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: HORSE RACE — ALL SHOCKS SIMULTANEOUSLY
# ═══════════════════════════════════════════════════════════════════════════════

def run_step3(p, lines):
    print("\nStep 3: Horse race regression …")
    sub = p.dropna(subset=["pct_cross","vix_z_l1","oil_price_z_l1",
                            "interest_rate_z_l1","gpr_z_l1","cpu_z_l1",
                            "pct_cross_l1"]).copy()

    formula = ("pct_cross ~ vix_z_l1 + oil_price_z_l1 + interest_rate_z_l1 "
               "+ gpr_z_l1 + cpu_z_l1 + pct_cross_l1 + quarter_trend + q2 + q3 + q4")
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC3")
        n  = int(m.nobs); r2 = m.rsquared_adj

        def sig(p): return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

        lines += ["\n" + "="*72,
                  "STEP 3: HORSE RACE — ALL SHOCKS SIMULTANEOUSLY (lag=1Q)",
                  f"  DV: pct_cross(t)  N={n} quarters  adj-R²={r2:.3f}  HC3 SE",
                  "="*72]

        shock_labels = {
            "vix_z_l1":           "VIX(t-1)",
            "oil_price_z_l1":     "Oil(t-1)",
            "interest_rate_z_l1": "Interest Rate(t-1)",
            "gpr_z_l1":           "GPR(t-1)",
            "cpu_z_l1":           "CPU(t-1)",
        }
        for col, label in shock_labels.items():
            b = m.params[col]; p_ = m.pvalues[col]
            lines.append(f"  {label:<25}  β={b:+.5f}  p={p_:.3f} {sig(p_)}")

        lines.append(f"\n  AR(1) pct_cross_l1:      β={m.params['pct_cross_l1']:+.4f}  p={m.pvalues['pct_cross_l1']:.3f}")
        lines.append(f"  Overall model F p-value: {m.f_pvalue:.4f}")
        return m
    except Exception as e:
        lines.append(f"  ERROR: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPARE MEDIATION EFFECT SIZES
# ═══════════════════════════════════════════════════════════════════════════════

def run_step4(mediation_results, step1_results, lines):
    print("\nStep 4: Mediation comparison …")
    lines += ["\n" + "="*72,
              "STEP 4: MEDIATION COMPARISON TABLE",
              "  Sorted by absolute indirect effect size (failure outcome)",
              "="*72]

    rows = []
    for key, r in mediation_results.items():
        if r["outcome"] != "failed": continue
        name = r["name"]
        f_p  = step1_results.get(name, {}).get("f_p", np.nan)
        rows.append({
            "Shock":           name,
            "Step1 F-p":       f_p,
            "Path A (β)":      r["a"],
            "Path A p":        r["p_a"],
            "Path B (β)":      r["b"],
            "Path B p":        r["p_b"],
            "Indirect":        r["indirect"],
            "CI_lo":           r.get("ci_lo", np.nan),
            "CI_hi":           r.get("ci_hi", np.nan),
            "Prop.mediated":   r.get("prop_med", np.nan),
        })

    tbl = pd.DataFrame(rows).sort_values("Indirect", key=abs, ascending=False)
    for _, row in tbl.iterrows():
        def sp(p): return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))
        ci_str = (f"[{row['CI_lo']:+.5f}, {row['CI_hi']:+.5f}]"
                  if not np.isnan(row.get("CI_lo",np.nan)) else "n/a")
        prop_str = f"{row['Prop.mediated']*100:.1f}%" if not np.isnan(row.get("Prop.mediated",np.nan)) else "n/a"
        lines += [
            f"\n  {row['Shock']}",
            f"    Step1 F-p = {row['Step1 F-p']:.3f}  |  Path A β={row['Path A (β)']:+.4f} {sp(row['Path A p'])}  "
            f"Path B β={row['Path B (β)']:+.4f} {sp(row['Path B p'])}",
            f"    Indirect = {row['Indirect']:+.6f}  95%CI {ci_str}  Prop.med = {prop_str}",
        ]
    tbl.to_csv(OUT / "mediation_comparison.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: Time series of cross-community share with shock overlays
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timeseries(panel, step1_results):
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=130, facecolor="white")
    axes = axes.flatten()
    shocks = ["VIX", "Oil Price", "Interest Rate", "GPR", "CPU"]
    labels = {"VIX": "VIX", "Oil Price": "Oil Price (USD)",
              "Interest Rate": "Fed Funds Rate (%)",
              "GPR": "Geopolitical Risk (GPR)", "CPU": "Climate Policy Uncertainty"}
    cols   = {"VIX": "vix_z", "Oil Price": "oil_price_z",
              "Interest Rate": "interest_rate_z", "GPR": "gpr_z", "CPU": "cpu_z"}

    for i, name in enumerate(shocks):
        ax = axes[i]
        col = cols[name]
        sub = panel.dropna(subset=["pct_cross", col])
        ax2 = ax.twinx()
        ax.plot(sub["date"], sub["pct_cross"] * 100, color="#2980b9", lw=2, label="% Cross-community")
        ax.fill_between(sub["date"], sub["pct_cross"] * 100, alpha=0.15, color="#2980b9")
        ax2.plot(sub["date"], sub[col], color="#e74c3c", lw=1.3, ls="--", alpha=0.8, label=labels[name])
        f_p = step1_results.get(name, {}).get("f_p", 1)
        title_sig = "***" if f_p<0.001 else ("**" if f_p<0.01 else ("*" if f_p<0.05 else ("†" if f_p<0.1 else "")))
        ax.set_title(f"{labels[name]} {title_sig}", fontsize=9, fontweight="bold")
        ax.set_ylabel("% Cross-community", color="#2980b9", fontsize=8)
        ax2.set_ylabel(f"{name} (z-score)", color="#e74c3c", fontsize=8)
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.spines["top"].set_visible(False)
        for label in ax.get_xticklabels(): label.set_rotation(30)

    # Last panel: horse race bar
    ax = axes[5]
    if step1_results:
        names = list(step1_results.keys())
        f_ps  = [step1_results[n]["f_p"] for n in names]
        colors = ["#27ae60" if p < 0.05 else ("#f39c12" if p < 0.10 else "#bdc3c7") for p in f_ps]
        bars = ax.bar(names, [-np.log10(max(p, 1e-6)) for p in f_ps], color=colors)
        ax.axhline(-np.log10(0.05), ls="--", color="gray", lw=1)
        ax.axhline(-np.log10(0.10), ls=":", color="gray", lw=1)
        ax.set_ylabel("−log₁₀(F-test p)", fontsize=9)
        ax.set_title("Step 1: Shock significance\n(green=p<0.05, orange=p<0.10)", fontsize=9)
        ax.tick_params(labelsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Macro Shocks × Cross-Community Network Fragmentation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig1_shocks_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_shocks_timeseries.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = ["=" * 72,
             "MACRO SHOCKS × CROSS-COMMUNITY NETWORK FRAGMENTATION",
             "Caldara & Iacoviello GPR | FRED VIX/Oil/Rate | CPU Gavriilidis",
             "=" * 72]

    # 0. Build panels
    qdf   = build_quarterly_panel()
    macro = build_macro_panel()
    panel = qdf.merge(macro, on=["YQ","date"], how="inner")
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["pct_cross_z"] = (panel["pct_cross"] - panel["pct_cross"].mean()) / panel["pct_cross"].std()
    print(f"\n  Merged panel: {len(panel)} quarters")
    panel.to_csv(OUT / "quarterly_panel.csv", index=False)

    # Load company-level dataset for mediation
    company_df = pd.read_csv(ROOT / "output/cpu/cpu_company_dataset.csv")

    # Merge GPR to company_df (by first-deal year)
    # Use gpr quarterly values averaged to annual for company-level merge
    gpr_ann = macro.groupby(macro["date"].dt.year)["gpr_z"].mean().reset_index()
    gpr_ann.columns = ["deal_year", "gpr_z"]
    company_df = company_df.merge(gpr_ann, on="deal_year", how="left")

    # 1. Step 1
    step1_results, panel_with_lags = run_step1(panel, lines)

    # 2. Step 2 — mediation
    mediation_results = run_step2(step1_results, company_df, lines)

    # 3. Step 3 — horse race
    run_step3(panel_with_lags, lines)

    # 4. Step 4 — compare
    if mediation_results:
        run_step4(mediation_results, step1_results, lines)

    # 5. Figures
    plot_timeseries(panel, step1_results)

    # 6. Summary
    lines += ["\n" + "="*72, "SUMMARY", "="*72]
    sig_s1 = [k for k, v in step1_results.items() if v.get("f_p", 1) < 0.10]
    sig_s1_05 = [k for k, v in step1_results.items() if v.get("f_p", 1) < 0.05]
    lines.append(f"\n  Shocks that fragment cross-community network (p<0.05): {sig_s1_05 or 'none'}")
    lines.append(f"  Shocks at p<0.10 (marginal): {[s for s in sig_s1 if s not in sig_s1_05] or 'none'}")
    if mediation_results:
        sig_med = [k for k, v in mediation_results.items()
                   if not np.isnan(v.get("ci_lo", np.nan))
                   and not (v["ci_lo"] < 0 < v["ci_hi"])]
        lines.append(f"  Significant mediation paths (CI excludes 0): {sig_med or 'none'}")

    output = "\n".join(lines)
    print("\n" + output)
    (OUT / "results.txt").write_text(output)
    print(f"\nAll outputs saved to {OUT}/")


if __name__ == "__main__":
    main()
