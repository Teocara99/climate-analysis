"""
Script 60: Monthly frequency + Event study + Mediation analysis
================================================================
Deepens the ecosystem amplification finding from Scripts 57-59.

PART A: Monthly siloing × CPU panel (N≈130 months)
PART B: Event-study around 7 discrete policy shocks
PART C: Mediation — does ecosystem siloing work through syndicate homogeneity?
"""

import warnings; warnings.filterwarnings("ignore")
import re, sys
from itertools import combinations
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.stats.sandwich_covariance import cov_hac

sys.path.insert(0, ".")
from load_data import load_deals

OUT_A = Path("output/monthly_siloing");   OUT_A.mkdir(parents=True, exist_ok=True)
OUT_B = Path("output/event_study");       OUT_B.mkdir(parents=True, exist_ok=True)
OUT_C = Path("output/mediation_analysis");OUT_C.mkdir(parents=True, exist_ok=True)

def sig(p):
    if p is None or np.isnan(p): return ""
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

def parse_investors(s):
    if pd.isna(s): return []
    parts = [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]
    return [re.sub(r"\s*\([^)]*\)\s*$", "", p).strip() for p in parts if p]

def nw_se(model, nlags=4):
    """Return Newey-West HAC covariance-based results."""
    try:
        hac = cov_hac(model, nlags=nlags)
        return model.get_robustcov_results(cov_type="HAC",
                                           cov_kwds={"maxlags": nlags})
    except Exception:
        return model

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
deals_raw  = load_deals()
deals_raw["date"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce")
deals_raw["ym"]   = deals_raw["date"].dt.to_period("M").astype(str)
deals_raw["company_key"] = deals_raw["Companies"].str.lower().str.strip()

nm  = pd.read_csv("output/network/network_metrics.csv")
inv_to_comm = dict(zip(nm["investor"].str.lower().str.strip(),
                       nm["community"].astype(int)))

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["company_key"] = cdf["Companies"].str.lower().str.strip()

cpu = pd.read_csv("data/cpu_gavriilidis_us.csv")
cpu.columns = ["ym", "cpu_narrow", "cpu_broad", "cpu_sentiment"]

# Company first/last deal month and failure proxy
deals_dated = deals_raw.dropna(subset=["date"])
company_dates = (deals_dated.groupby("company_key")["date"]
                             .agg(["min","max"])
                             .rename(columns={"min":"first_deal","max":"last_deal"}))
company_dates["deals_first_ym"] = company_dates["first_deal"].dt.to_period("M").astype(str)
company_dates["last_ym"]        = company_dates["last_deal"].dt.to_period("M").astype(str)
cdf = cdf.merge(company_dates.reset_index()[["company_key","deals_first_ym","last_ym"]],
                on="company_key", how="left")
# Prefer deals-derived first_ym when available; fallback to cdf's own first_ym
cdf["first_ym"] = cdf["deals_first_ym"].fillna(cdf["first_ym"])
# Failure month proxy: last deal month for failed companies
cdf["failure_ym"] = np.where(cdf["failed"]==1, cdf["last_ym"], pd.NA)
print(f"  Companies: {len(cdf):,}  |  Failed: {cdf['failed'].sum():,}")
print(f"  Deal date range: {deals_dated['date'].min().date()} – {deals_dated['date'].max().date()}")


# ══════════════════════════════════════════════════════════════════════════════
# PART A: MONTHLY FREQUENCY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART A: MONTHLY FREQUENCY ANALYSIS")
print("="*72)

# ── A1: Monthly siloing ───────────────────────────────────────────────────────
print("\n  A1: Computing monthly siloing (investor-pair within/cross community) …")

deals_inv = deals_dated[deals_dated["Investors"].notna()].copy()
all_months = sorted(deals_inv["ym"].dropna().unique())

monthly_rows = []
for ym in all_months:
    month_deals = deals_inv[deals_inv["ym"] == ym]
    n_deals     = len(month_deals)

    cross_pairs = total_pairs = 0
    total_funding = month_deals["Deal Size (USD M)"].fillna(0).sum()
    n_companies   = month_deals["company_key"].nunique()

    for inv_str in month_deals["Investors"]:
        inv_list = parse_investors(inv_str)
        comms = [inv_to_comm.get(i.lower()) for i in inv_list]
        comms = [c for c in comms if c is not None]
        for i, (ca, cb) in enumerate(combinations(comms, 2)):
            total_pairs += 1
            if ca != cb:
                cross_pairs += 1

    siloing_m = (1 - cross_pairs / total_pairs) if total_pairs > 0 else np.nan
    monthly_rows.append({
        "ym": ym, "siloing_m": siloing_m,
        "n_deals_m": n_deals, "n_companies_m": n_companies,
        "total_funding_m": total_funding,
        "n_cross_pairs": cross_pairs, "n_total_pairs": total_pairs,
        "sparse": int(n_deals < 5),
    })

monthly_panel = pd.DataFrame(monthly_rows)

# A2: Merge with CPU
print("  A2: Merging with CPU index …")
monthly_panel = monthly_panel.merge(cpu, on="ym", how="left")

# A3: Monthly failure rate
print("  A3: Computing monthly failure rate …")
fail_ym_counts = (cdf[cdf["failed"] == 1]["failure_ym"]
                   .dropna().value_counts().rename("n_failed_m"))

# n_active_m: companies with first_deal ≤ month m (cumulative ever-started)
# Using 3-year rolling active company count as denominator
ym_sorted = sorted(monthly_panel["ym"])
ym_first_counts = cdf["first_ym"].dropna().value_counts().sort_index()

n_active_by_ym = {}
cumulative = 0
all_sorted_ym = sorted(ym_first_counts.index)
for ym in ym_sorted:
    # Cumulative companies started by this month
    for y in list(all_sorted_ym):
        if y <= ym and y not in [k for k in n_active_by_ym]:
            cumulative += ym_first_counts.get(y, 0)
    n_active_by_ym[ym] = cumulative

monthly_panel["n_active_m"]  = monthly_panel["ym"].map(n_active_by_ym)
monthly_panel["n_failed_m"]  = monthly_panel["ym"].map(fail_ym_counts).fillna(0)
monthly_panel["failure_rate_m"] = monthly_panel["n_failed_m"] / monthly_panel["n_active_m"].clip(lower=1)

# Filter to 2013-01 through 2025-12 (analysis window with data)
monthly_panel = monthly_panel[monthly_panel["ym"].between("2013-01", "2025-12")].copy()
monthly_panel = monthly_panel.sort_values("ym").reset_index(drop=True)
monthly_panel["t"]  = np.arange(len(monthly_panel))
monthly_panel["t2"] = monthly_panel["t"] ** 2

cpu_ok  = monthly_panel["cpu_narrow"].notna()
sil_ok  = monthly_panel["siloing_m"].notna()
panel_ok = monthly_panel[cpu_ok & sil_ok].copy()

print(f"\n  Monthly panel: {len(monthly_panel)} months total")
print(f"  Months with both siloing & CPU: {len(panel_ok)}")
print(f"  Date range: {panel_ok['ym'].min()} – {panel_ok['ym'].max()}")
print(f"  Siloing: mean={panel_ok['siloing_m'].mean():.4f}  "
      f"std={panel_ok['siloing_m'].std():.4f}")
print(f"  CPU narrow: mean={panel_ok['cpu_narrow'].mean():.2f}  "
      f"std={panel_ok['cpu_narrow'].std():.2f}")
print(f"  Monthly failure rate: mean={panel_ok['failure_rate_m'].mean():.5f}  "
      f"max={panel_ok['failure_rate_m'].max():.5f}")

monthly_panel.to_csv(OUT_A / "monthly_panel.csv", index=False)

# ── A4: Monthly regressions ───────────────────────────────────────────────────
print("\n  A4: Monthly regressions (Newey-West HAC SEs, nlags=4) …\n")

p = panel_ok.copy()
# Standardize for comparability
for col in ["siloing_m", "cpu_narrow", "cpu_broad", "n_deals_m",
            "total_funding_m", "failure_rate_m"]:
    p[f"{col}_z"] = (p[col] - p[col].mean()) / p[col].std()

p["sil_x_cpu"]     = p["siloing_m_z"] * p["cpu_narrow_z"]
p["log_funding"]   = np.log1p(p["total_funding_m"])
p["log_n_deals"]   = np.log1p(p["n_deals_m"])
p["Q2"] = p["ym"].str[-2:].isin(["04","05","06"]).astype(int)
p["Q3"] = p["ym"].str[-2:].isin(["07","08","09"]).astype(int)
p["Q4"] = p["ym"].str[-2:].isin(["10","11","12"]).astype(int)

controls = "log_n_deals + log_funding + t + t2 + Q2 + Q3 + Q4"

model_specs = [
    ("M1 (levels, no interaction)",
     f"failure_rate_m ~ siloing_m_z + cpu_narrow_z + {controls}"),
    ("M2 (+ sil×cpu interaction)",
     f"failure_rate_m ~ siloing_m_z + cpu_narrow_z + sil_x_cpu + {controls}"),
]

print(f"  {'Model':<35} {'N':>4}  {'β_siloing':>10}  {'β_cpu':>10}  "
      f"{'β_sil×cpu':>10}  {'R²':>6}")
print(f"  {'─'*80}")

fitted_models = {}
for label, formula in model_specs:
    try:
        m = smf.ols(formula, data=p).fit()
        m_hac = nw_se(m, nlags=4)
        bs  = m_hac.params.get("siloing_m_z", np.nan)
        bc  = m_hac.params.get("cpu_narrow_z", np.nan)
        bi  = m_hac.params.get("sil_x_cpu", np.nan)
        ps  = m_hac.pvalues.get("siloing_m_z", np.nan)
        pc  = m_hac.pvalues.get("cpu_narrow_z", np.nan)
        pi  = m_hac.pvalues.get("sil_x_cpu", np.nan)
        r2  = m.rsquared
        print(f"  {label:<35} {int(m.nobs):>4}  "
              f"{bs:>+8.4f}{sig(ps):<3}  {bc:>+8.4f}{sig(pc):<3}  "
              f"{bi:>+8.4f}{sig(pi):<3}  {r2:>6.3f}")
        fitted_models[label] = (m, m_hac)
    except Exception as e:
        print(f"  {label}: error — {e}")

# M3/M4: Detrended versions
print()
for method, label in [("hp", "M3 (HP-filter λ=129600, detrended)"),
                       ("fd", "M4 (first-difference)")]:
    try:
        p2 = p.copy()
        if method == "hp":
            _, trend_sil = hpfilter(p2["siloing_m"].dropna(), lamb=129600)
            _, trend_cpu = hpfilter(p2["cpu_narrow"].dropna(), lamb=129600)
            p2["sil_dt"] = (p2["siloing_m"] - p2["siloing_m"].index.map(
                lambda i: trend_sil.iloc[i] if i < len(trend_sil) else np.nan))
            _, trend_sil = hpfilter(p2["siloing_m"].values, lamb=129600)
            _, trend_cpu = hpfilter(p2["cpu_narrow"].values, lamb=129600)
            p2["sil_dt"] = p2["siloing_m"].values - trend_sil
            p2["cpu_dt"] = p2["cpu_narrow"].values - trend_cpu
        else:
            p2["sil_dt"] = p2["siloing_m"].diff()
            p2["cpu_dt"] = p2["cpu_narrow"].diff()
            p2["fail_dt"] = p2["failure_rate_m"].diff()

        for col in ["sil_dt", "cpu_dt"]:
            p2[f"{col}_z"] = (p2[col] - p2[col].mean()) / p2[col].std()
        p2["sil_x_cpu_dt"] = p2["sil_dt_z"] * p2["cpu_dt_z"]

        dep = "fail_dt" if method == "fd" else "failure_rate_m"
        formula_dt = (f"{dep} ~ sil_dt_z + cpu_dt_z + sil_x_cpu_dt "
                      f"+ log_n_deals + Q2 + Q3 + Q4")
        sub = p2.dropna(subset=[dep, "sil_dt_z", "cpu_dt_z"])
        m_dt  = smf.ols(formula_dt, data=sub).fit()
        m_hac = nw_se(m_dt, nlags=4)
        bs = m_hac.params.get("sil_dt_z", np.nan)
        bc = m_hac.params.get("cpu_dt_z", np.nan)
        bi = m_hac.params.get("sil_x_cpu_dt", np.nan)
        ps = m_hac.pvalues.get("sil_dt_z", np.nan)
        pc = m_hac.pvalues.get("cpu_dt_z", np.nan)
        pi = m_hac.pvalues.get("sil_x_cpu_dt", np.nan)
        r2 = m_dt.rsquared
        print(f"  {label:<35} {int(m_dt.nobs):>4}  "
              f"{bs:>+8.4f}{sig(ps):<3}  {bc:>+8.4f}{sig(pc):<3}  "
              f"{bi:>+8.4f}{sig(pi):<3}  {r2:>6.3f}")
        fitted_models[label] = (m_dt, m_hac)
    except Exception as e:
        print(f"  {label}: error — {e}")

print(f"\n  Interpretation reference: Script 58 quarterly β_sil×cpu = +0.196† (M9c)")
print(f"  Sign consistency check: monthly interaction β > 0 → CONSISTENT; < 0 → INCONSISTENT")

# ── A5: Signed policy decomposition ──────────────────────────────────────────
print("\n  A5: SIGNED POLICY DECOMPOSITION (monthly)")

# Define policy regimes
POLICY_EVENTS = [
    ("2015-12", +1, "Paris Agreement"),
    ("2017-06", -1, "Trump Paris withdrawal"),
    ("2018-01", -1, "Trump tax reform / CEI uncertainty"),
    ("2020-11", +1, "Biden election"),
    ("2021-01", +1, "Biden inauguration"),
    ("2022-08", +1, "IRA signed"),
    ("2025-01", -1, "Trump 2.0 inauguration"),
]
EVENTS_DF = pd.DataFrame(POLICY_EVENTS, columns=["event_ym","direction","label"])

def assign_regime(ym_str):
    """Return +1 (positive) or -1 (negative) based on most recent event."""
    past = EVENTS_DF[EVENTS_DF["event_ym"] <= ym_str]
    if len(past) == 0:
        return 0  # pre-2015: undefined
    return past.iloc[-1]["direction"]

p["regime"] = p["ym"].apply(assign_regime)
p["signed_cpu"] = p["cpu_narrow"] * p["regime"]
p["signed_cpu_z"] = (p["signed_cpu"] - p["signed_cpu"].mean()) / p["signed_cpu"].std()
p["sil_x_signed_cpu"] = p["siloing_m_z"] * p["signed_cpu_z"]

print(f"  Regime dist: {p['regime'].value_counts().to_dict()}")

for label, formula in [
    ("M5 (signed_cpu)", "failure_rate_m ~ siloing_m_z + signed_cpu_z + sil_x_signed_cpu + log_n_deals + t + t2 + Q2 + Q3 + Q4"),
]:
    try:
        m5 = smf.ols(formula, data=p[p["regime"] != 0]).fit()
        m5_hac = nw_se(m5, nlags=4)
        print(f"\n  {label}:")
        for v in ["siloing_m_z","signed_cpu_z","sil_x_signed_cpu"]:
            b = m5_hac.params.get(v, np.nan)
            pv = m5_hac.pvalues.get(v, np.nan)
            print(f"    {v:<30} β={b:>+8.4f}  {sig(pv)}")
        print(f"    N={int(m5.nobs)}  R²={m5.rsquared:.3f}")
    except Exception as e:
        print(f"  {label}: error — {e}")

# M6: Split-sample by regime
print("\n  M6: Split-sample by regime (positive vs negative months):")
for regime_val, regime_lbl in [(+1, "Positive regime months"), (-1, "Negative regime months")]:
    sub = p[p["regime"] == regime_val].copy()
    if len(sub) < 10:
        print(f"  {regime_lbl}: too few observations (N={len(sub)})")
        continue
    try:
        m6 = smf.ols("failure_rate_m ~ siloing_m_z + cpu_narrow_z + sil_x_cpu + log_n_deals + Q2 + Q3 + Q4",
                     data=sub).fit()
        m6_hac = nw_se(m6, nlags=4)
        bi = m6_hac.params.get("sil_x_cpu", np.nan)
        pi = m6_hac.pvalues.get("sil_x_cpu", np.nan)
        print(f"  {regime_lbl:<30} N={int(m6.nobs):>3}  β_sil×cpu={bi:>+8.4f}  {sig(pi)}")
    except Exception as e:
        print(f"  {regime_lbl}: error — {e}")

print("\n  Interpretation: if β_sil×cpu is POSITIVE in negative regimes and NEGATIVE in positive")
print("  regimes, siloing AMPLIFIES shocks in both directions (echo chamber = bidirectional).")


# ══════════════════════════════════════════════════════════════════════════════
# PART B: EVENT-STUDY DESIGN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART B: EVENT-STUDY DESIGN")
print("="*72)

# ── B1-B3: Aggregate event study ─────────────────────────────────────────────
print("\n  B1-B3: Computing pre/post windows for each of 7 policy events …\n")

WINDOW = 6  # months before and after

event_rows = []
for ev_ym, ev_dir, ev_lbl in POLICY_EVENTS:
    ev_period = pd.Period(ev_ym, freq="M")
    pre_months  = [str(ev_period - i) for i in range(1, WINDOW+1)]
    post_months = [str(ev_period + i) for i in range(0, WINDOW)]

    pre_data  = panel_ok[panel_ok["ym"].isin(pre_months)]
    post_data = panel_ok[panel_ok["ym"].isin(post_months)]

    def safe_mean(df, col):
        v = df[col].dropna()
        return v.mean() if len(v) > 0 else np.nan

    pre_sil  = safe_mean(pre_data, "siloing_m")
    post_sil = safe_mean(post_data, "siloing_m")
    pre_cpu  = safe_mean(pre_data, "cpu_narrow")
    post_cpu = safe_mean(post_data, "cpu_narrow")
    pre_fail = safe_mean(pre_data, "failure_rate_m")
    post_fail= safe_mean(post_data, "failure_rate_m")

    row = {
        "event_ym": ev_ym, "direction": ev_dir, "label": ev_lbl,
        "pre_sil": pre_sil, "post_sil": post_sil,
        "pre_cpu": pre_cpu, "post_cpu": post_cpu,
        "pre_fail": pre_fail, "post_fail": post_fail,
        "delta_sil": post_sil - pre_sil,
        "delta_cpu": post_cpu - pre_cpu,
        "delta_fail": post_fail - pre_fail,
        "n_pre_months": len(pre_data), "n_post_months": len(post_data),
    }
    event_rows.append(row)

event_df = pd.DataFrame(event_rows)

print(f"  {'Event':<35} {'Dir':>5}  {'Δsil':>8}  {'ΔCPU':>8}  {'Δfail':>10}  N pre/post")
print(f"  {'─'*80}")
for _, r in event_df.iterrows():
    d_str = "POS" if r["direction"] > 0 else "NEG"
    dir_arrow = "↑" if r["direction"] > 0 else "↓"
    print(f"  {r['label']:<35} {dir_arrow:>5}  "
          f"{r['delta_sil']:>+8.4f}  {r['delta_cpu']:>+8.3f}  "
          f"{r['delta_fail']:>+10.6f}  "
          f"{r['n_pre_months']:.0f} / {r['n_post_months']:.0f}")

event_df.to_csv(OUT_B / "event_study_aggregate.csv", index=False)

# B3: Cross-event regression (N=7, descriptive)
print("\n  B3: Cross-event regression (N=7, descriptive):")
if len(event_df) >= 5:
    ev = event_df.dropna(subset=["delta_sil","delta_cpu","delta_fail"]).copy()
    ev["sil_x_dir"] = ev["delta_sil"] * ev["direction"]
    try:
        m_ev = smf.ols("delta_fail ~ delta_sil + delta_cpu + sil_x_dir", data=ev).fit()
        for v in ["delta_sil","delta_cpu","sil_x_dir"]:
            b = m_ev.params.get(v, np.nan)
            pv = m_ev.pvalues.get(v, np.nan)
            print(f"    {v:<25} β={b:>+10.6f}  {sig(pv)}")
        print(f"    N={int(m_ev.nobs)}  R²={m_ev.rsquared:.3f}  (N too small for inference)")
        print(f"    Note: this regression is purely descriptive — 7 events, no inferential weight.")
    except Exception as e:
        print(f"    Error: {e}")
else:
    print("  Insufficient events for regression.")

# ── B4: Company-level event study (DiD) ──────────────────────────────────────
print("\n  B4: COMPANY-LEVEL EVENT STUDY (DiD across all 7 events)")
print("  Specification: failed_i ~ sil_i × post_e + company controls + event FE")

# Build syndicate homogeneity for each company
print("  Computing company-level syndicate homogeneity …")
company_sil_rows = []
for company_key, inv_series in (deals_raw[deals_raw["Investors"].notna()]
                                  .groupby("company_key")["Investors"]):
    all_comms = []
    for inv_str in inv_series:
        inv_list = parse_investors(inv_str)
        comms = [inv_to_comm.get(i.lower()) for i in inv_list if i]
        all_comms.extend([c for c in comms if c is not None])

    if len(all_comms) < 2:
        homogeneity = 1.0 if len(all_comms) == 1 else np.nan
    else:
        cross = within = 0
        for ca, cb in combinations(all_comms, 2):
            if ca == cb: within += 1
            else: cross += 1
        total = within + cross
        homogeneity = within / total if total > 0 else np.nan

    company_sil_rows.append({
        "company_key": company_key,
        "syndicate_homogeneity": homogeneity,
    })

company_sil_df = pd.DataFrame(company_sil_rows)
cdf = cdf.merge(company_sil_df, on="company_key", how="left")
print(f"  Syndicate homogeneity: N={cdf['syndicate_homogeneity'].notna().sum():,}  "
      f"mean={cdf['syndicate_homogeneity'].mean():.3f}")

# For each event: identify companies funded in the 12-month window
# Post = funded in [event, event+6], Pre = funded in [event-6, event-1]
event_company_rows = []
for ev_ym, ev_dir, ev_lbl in POLICY_EVENTS:
    ev_period = pd.Period(ev_ym, freq="M")
    pre_yms   = [str(ev_period - i) for i in range(1, WINDOW+1)]
    post_yms  = [str(ev_period + i) for i in range(0, WINDOW)]

    # Companies first funded in the 12-month window
    pre_cos  = cdf[cdf["first_ym"].isin(pre_yms)].copy()
    post_cos = cdf[cdf["first_ym"].isin(post_yms)].copy()

    pre_cos["post"]  = 0
    post_cos["post"] = 1
    pre_cos["event_ym"]  = ev_ym
    post_cos["event_ym"] = ev_ym
    pre_cos["direction"]  = ev_dir
    post_cos["direction"] = ev_dir

    event_company_rows.append(pre_cos)
    event_company_rows.append(post_cos)

event_cos = pd.concat(event_company_rows, ignore_index=True)

# Merge ecosystem siloing at company's first deal month
# Monthly siloing already in monthly_panel
ym_to_sil = dict(zip(monthly_panel["ym"], monthly_panel["siloing_m"]))
event_cos["eco_siloing"] = event_cos["first_ym"].map(ym_to_sil)
event_cos["sil_post"]    = event_cos["syndicate_homogeneity"] * event_cos["post"]
event_cos["is_positive"] = (event_cos["direction"] == 1).astype(int)
event_cos["log_cap"]     = np.log1p(event_cos["Total Raised"].fillna(0))

required = ["failed","syndicate_homogeneity","post","log_cap","hq_region","event_ym"]
event_cos_ok = event_cos.dropna(subset=required).copy()

print(f"\n  DiD dataset: {len(event_cos_ok):,} company-event observations "
      f"({event_cos_ok['company_key'].nunique():,} unique companies × events)")
print(f"  Events represented: {event_cos_ok['event_ym'].nunique()}")
print(f"  Failure rate in dataset: {event_cos_ok['failed'].mean():.3f}")

# Pool across all events with event fixed effects
try:
    did_m = smf.ols(
        "failed ~ syndicate_homogeneity + post + sil_post "
        "+ log_cap + C(hq_region) + C(event_ym)",
        data=event_cos_ok
    ).fit(cov_type="HC1")
    print(f"\n  Pooled DiD (all 7 events, event FE):")
    for v in ["syndicate_homogeneity","post","sil_post"]:
        b = did_m.params.get(v, np.nan)
        pv = did_m.pvalues.get(v, np.nan)
        print(f"    {v:<30} β={b:>+8.4f}  {sig(pv)}")
    print(f"    N={int(did_m.nobs):,}  R²={did_m.rsquared:.3f}")
except Exception as e:
    print(f"  DiD pooled error: {e}")

# Triple interaction: sil × post × positive_direction
try:
    event_cos_ok["sil_post_pos"] = (event_cos_ok["syndicate_homogeneity"]
                                     * event_cos_ok["post"]
                                     * event_cos_ok["is_positive"])
    did_triple = smf.ols(
        "failed ~ syndicate_homogeneity + post + is_positive "
        "+ sil_post + sil_post_pos "
        "+ log_cap + C(hq_region) + C(event_ym)",
        data=event_cos_ok
    ).fit(cov_type="HC1")
    print(f"\n  Triple interaction (sil × post × positive_direction):")
    for v in ["sil_post","sil_post_pos"]:
        b = did_triple.params.get(v, np.nan)
        pv = did_triple.pvalues.get(v, np.nan)
        print(f"    {v:<30} β={b:>+8.4f}  {sig(pv)}")
    print(f"    N={int(did_triple.nobs):,}  R²={did_triple.rsquared:.3f}")
    print(f"  Interpretation: if sil_post_pos < 0, high-homogeneity companies fare BETTER "
          f"after positive policy shocks.\n"
          f"  If sil_post > 0 and sil_post_pos < 0, positive events partially offset "
          f"the penalty of syndicate homogeneity.")
except Exception as e:
    print(f"  Triple DiD error: {e}")

# Event-by-event estimates
print(f"\n  Event-specific DiD estimates (sil × post coefficient):")
print(f"  {'Event':<35} {'N':>5}  {'β_sil×post':>12}  Sig")
print(f"  {'─'*60}")
for ev_ym, ev_dir, ev_lbl in POLICY_EVENTS:
    sub = event_cos_ok[event_cos_ok["event_ym"] == ev_ym].copy()
    if len(sub) < 20:
        print(f"  {ev_lbl:<35} {len(sub):>5}  {'—':>12}  (too small)")
        continue
    try:
        m_ev = smf.ols("failed ~ syndicate_homogeneity + post + sil_post + log_cap",
                       data=sub).fit(cov_type="HC1")
        b  = m_ev.params.get("sil_post", np.nan)
        pv = m_ev.pvalues.get("sil_post", np.nan)
        dir_str = "↑POS" if ev_dir > 0 else "↓NEG"
        print(f"  {ev_lbl:<35} {int(m_ev.nobs):>5}  {b:>+12.4f}  {sig(pv)}  [{dir_str}]")
    except Exception as e:
        print(f"  {ev_lbl:<35}: error — {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART C: MEDIATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PART C: MEDIATION — DOES ECOSYSTEM SILOING WORK THROUGH SYNDICATE HOMOGENEITY?")
print("="*72)

# ── C1-C2: Build company-level mediation dataset ──────────────────────────────
print("\n  C1-C2: Building company-level mediation dataset …")

# Ecosystem siloing at company's first deal month
cdf["eco_siloing_m"] = cdf["first_ym"].map(ym_to_sil)

required_med = ["failed","syndicate_homogeneity","eco_siloing_m","log_capital","hq_region","deal_year"]
cdf_med = cdf.dropna(subset=required_med).copy()
cdf_med["eco_sil_z"] = ((cdf_med["eco_siloing_m"] - cdf_med["eco_siloing_m"].mean())
                         / cdf_med["eco_siloing_m"].std())
cdf_med["syn_hom_z"] = ((cdf_med["syndicate_homogeneity"] - cdf_med["syndicate_homogeneity"].mean())
                         / cdf_med["syndicate_homogeneity"].std())
cdf_med["log_cap"]   = np.log1p(cdf_med["Total Raised"].fillna(0))

print(f"  Mediation dataset: N={len(cdf_med):,} companies")
corr_val = cdf_med["eco_siloing_m"].corr(cdf_med["syndicate_homogeneity"])
print(f"  corr(ecosystem_siloing, syndicate_homogeneity) = {corr_val:+.4f}")
if corr_val > 0.05:
    print(f"  → POSITIVE correlation: when the ecosystem is siloed, individual")
    print(f"    syndicates are more homogeneous — mediation story is plausible.")
elif corr_val < -0.05:
    print(f"  → NEGATIVE correlation: ecosystem siloing does NOT drive within-syndicate")
    print(f"    homogeneity — mediation story is not supported by this path.")
else:
    print(f"  → Near-zero correlation: weak association between ecosystem and syndicate level.")

# ── C3: Baron & Kenny mediation ───────────────────────────────────────────────
print("\n  C3: BARON & KENNY MEDIATION (OLS pathways)")

controls_bk = "log_cap + C(hq_region)"

# Path c (total effect): ecosystem_siloing → failed
try:
    path_c = smf.ols(f"failed ~ eco_sil_z + {controls_bk}", data=cdf_med).fit(cov_type="HC1")
    b_c  = path_c.params["eco_sil_z"]
    p_c  = path_c.pvalues["eco_sil_z"]
    se_c = path_c.bse["eco_sil_z"]
    print(f"  Path c  (total):  eco_sil → failed              β={b_c:>+8.4f}  {sig(p_c)}")
except Exception as e:
    b_c = se_c = p_c = np.nan; print(f"  Path c error: {e}")

# Path a: ecosystem_siloing → syndicate_homogeneity
try:
    path_a = smf.ols(f"syn_hom_z ~ eco_sil_z + {controls_bk}", data=cdf_med).fit(cov_type="HC1")
    b_a  = path_a.params["eco_sil_z"]
    p_a  = path_a.pvalues["eco_sil_z"]
    se_a = path_a.bse["eco_sil_z"]
    print(f"  Path a  (eco→med): eco_sil → syn_hom             β={b_a:>+8.4f}  {sig(p_a)}")
except Exception as e:
    b_a = se_a = p_a = np.nan; print(f"  Path a error: {e}")

# Path b: syndicate_homogeneity → failed | ecosystem_siloing
try:
    path_b = smf.ols(f"failed ~ syn_hom_z + eco_sil_z + {controls_bk}",
                     data=cdf_med).fit(cov_type="HC1")
    b_b   = path_b.params["syn_hom_z"]
    p_b   = path_b.pvalues["syn_hom_z"]
    se_b  = path_b.bse["syn_hom_z"]
    b_cprime = path_b.params.get("eco_sil_z", np.nan)
    p_cprime = path_b.pvalues.get("eco_sil_z", np.nan)
    print(f"  Path b  (med→Y):   syn_hom → failed | eco_sil    β={b_b:>+8.4f}  {sig(p_b)}")
    print(f"  Path c' (direct):  eco_sil → failed | syn_hom    β={b_cprime:>+8.4f}  {sig(p_cprime)}")
except Exception as e:
    b_b = se_b = p_b = np.nan; b_cprime = p_cprime = np.nan; print(f"  Path b error: {e}")

# Indirect effect a×b + Sobel test
if not any(np.isnan([b_a, b_b, se_a, se_b])):
    indirect = b_a * b_b
    sobel_se = np.sqrt(b_b**2 * se_a**2 + b_a**2 * se_b**2)
    sobel_z  = indirect / sobel_se
    sobel_p  = 2 * (1 - stats.norm.cdf(abs(sobel_z))) if not np.isnan(sobel_z) else np.nan
    print(f"\n  Indirect effect (a×b):  {indirect:>+8.4f}")
    print(f"  Sobel SE:               {sobel_se:>8.4f}")
    print(f"  Sobel z:                {sobel_z:>8.4f}")
    print(f"  Sobel p:                {sobel_p:>8.4f}  {sig(sobel_p)}")
    mediation_fraction = indirect / b_c if (b_c != 0 and not np.isnan(b_c)) else np.nan
    print(f"  Mediation fraction (a×b / c): {mediation_fraction:>+8.4f}"
          f"  ({100*abs(mediation_fraction):.1f}% of total effect)" if not np.isnan(mediation_fraction) else "")
else:
    indirect = sobel_z = sobel_p = np.nan
    print(f"\n  Cannot compute Sobel test — missing coefficients.")

# Bootstrap CI for indirect effect
print(f"\n  Bootstrap CI for indirect effect (1000 iterations) …")
rng = np.random.default_rng(42)
boot_indirect = []
n_med = len(cdf_med)
X_a = sm.add_constant(cdf_med[["eco_sil_z","log_cap"]].astype(float))
X_b = sm.add_constant(cdf_med[["syn_hom_z","eco_sil_z","log_cap"]].astype(float))
y_m = cdf_med["syn_hom_z"].values
y_f = cdf_med["failed"].values.astype(float)

for _ in range(1000):
    idx = rng.choice(n_med, size=n_med, replace=True)
    try:
        Xa_b = X_a.iloc[idx].values
        ba = np.linalg.lstsq(Xa_b, y_m[idx], rcond=None)[0][1]
        Xb_b = X_b.iloc[idx].values
        bb = np.linalg.lstsq(Xb_b, y_f[idx], rcond=None)[0][1]
        boot_indirect.append(ba * bb)
    except Exception:
        continue

boot_indirect = np.array(boot_indirect)
ci_lo, ci_hi = np.percentile(boot_indirect, [2.5, 97.5])
print(f"  Bootstrap indirect effect: mean={np.mean(boot_indirect):>+8.4f}  "
      f"SE={np.std(boot_indirect,ddof=1):>8.4f}")
print(f"  95% CI: [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}]")
if ci_lo > 0 or ci_hi < 0:
    print(f"  → CI excludes zero — indirect effect is statistically significant.")
    print(f"  → Ecosystem siloing works (partially) through syndicate homogeneity.")
else:
    print(f"  → CI includes zero — indirect effect is not precisely estimated.")
    print(f"  → Mediation via syndicate homogeneity is DIRECTIONAL but not precisely confirmed.")

# ── C4: Conditional mediation (pre/post-2020 split) ──────────────────────────
print(f"\n  C4: CONDITIONAL MEDIATION (pre/post-2020)")

for period_lbl, ym_filter in [("Pre-2020", "< '2020-01'"), ("Post-2020", ">= '2020-01'")]:
    sub = cdf_med[cdf_med["first_ym"].fillna("").str[:7].apply(
        lambda y: (y < "2020-01") if "Pre" in period_lbl else (y >= "2020-01")
    )].copy()
    if len(sub) < 50:
        print(f"  {period_lbl}: too few companies (N={len(sub)})")
        continue

    try:
        pa = smf.ols(f"syn_hom_z ~ eco_sil_z + log_cap", data=sub).fit()
        pb = smf.ols(f"failed    ~ syn_hom_z + eco_sil_z + log_cap", data=sub).fit()
        ba = pa.params.get("eco_sil_z", np.nan)
        bb = pb.params.get("syn_hom_z", np.nan)
        ind = ba * bb if not (np.isnan(ba) or np.isnan(bb)) else np.nan
        print(f"  {period_lbl:<12} N={len(sub):,}  "
              f"a={ba:>+7.4f}  b={bb:>+7.4f}  "
              f"indirect a×b={ind:>+8.4f}")
    except Exception as e:
        print(f"  {period_lbl}: error — {e}")

print(f"\n  If indirect effect is LARGER post-2020, the echo chamber mechanism became more")
print(f"  consequential during the IRA/policy-uncertainty regime shift.")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("SUMMARY ACROSS PARTS A–C")
print("="*72)

print("""
  PART A — Monthly frequency (N≈130 months):
  • Baseline finding from Script 58 (quarterly β_sil×cpu = +0.196†, N=28)
    tested at monthly frequency for robustness.
  • Key question: does the sil×cpu interaction sign hold at monthly granularity
    after controlling for trends? Report the M2 and M3 (HP-filtered) estimates.

  PART B — Event study (7 discrete policy shocks):
  • Aggregate level (N=7 events): documents whether Δsiloing and ΔfailureRate
    move in theoretically expected directions around policy events.
  • Company level DiD: tests whether more-siloed syndicates respond more
    strongly to policy shocks (triple interaction sil × post × direction).

  PART C — Mediation:
  • Tests whether the MECHANISM linking ecosystem siloing to failure outcomes
    is the within-company echo chamber (syndicate_homogeneity).
  • If a×b > 0 and CI excludes zero → echo chamber is the mediating channel.
  • Conditional mediation post-2020 tests if the mechanism intensified
    during the high-uncertainty regime of the IRA era.
""")

print(f"Outputs saved to: {OUT_A}/  {OUT_B}/  {OUT_C}/")
print("DONE")
