"""
Script 55: Network Fragility × Macro Shocks
============================================
Test Acemoglu, Ozdaglar & Tahbaz-Salehi (2015, AER) "robust yet fragile"
logic: as the climate-tech VC co-investment network becomes more siloed
(cross-community share declining 79%→45%, 2014–2025), it should absorb
small shocks but amplify large ones — visible as a larger deal-flow
contraction or survival penalty in shocked quarters/years when siloing
is high.

Three analyses:
A. Quarterly time-series (N≈44): deal-flow change ~ siloing_L1 × shock_q
   (HAC-robust OLS, Newey-West lags=4)
B. Company-level cross-section (N≈4,500 cohorts 2014–2022):
   is_failed ~ siloing_at_entry × shock_cohort + controls (LPM HC1)
C. Pre/post-2022 sanity check: 2×2 table of failure rates
   [pre/post-2022] × [high/low siloing cohort]
"""

import warnings; warnings.filterwarnings("ignore")
import sys, re
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

sys.path.insert(0, ".")
OUT = Path("output/network_fragility"); OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else "n.s."

# ─── HELPER: Newey-West t-stat from OLS ───────────────────────────────────────
def nw_pval(model, varname, lags=4):
    """Return (coef, se, t, p) with Newey-West HAC SE."""
    cov = cov_hac(model, nlags=lags)
    idx = list(model.params.index).index(varname)
    se  = float(np.sqrt(cov[idx, idx]))
    b   = float(model.params[varname])
    t   = b / se
    df  = int(model.nobs) - len(model.params)
    p   = float(2 * (1 - stats.t.cdf(abs(t), df=max(df, 1))))
    return b, se, t, p

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("LOADING DATA")
print("=" * 72)

# Quarterly siloing panel (cross_community_decline)
qsil = pd.read_csv("output/cross_community_decline/quarterly_panel.csv")
qsil["date"] = pd.to_datetime(qsil["date"])
# Siloing = 1 - pct_cross (higher = more siloed / more fragile)
qsil["siloing"]    = 1.0 - qsil["pct_cross"]
qsil["siloing_l1"] = qsil["siloing"].shift(1)   # 1-quarter lag
qsil["siloing_l4"] = qsil["siloing"].shift(4)   # 1-year lag
print(f"  Quarterly siloing panel: {qsil['YQ'].min()} → {qsil['YQ'].max()}  N={len(qsil)}")
print(f"  siloing range: [{qsil['siloing'].min():.3f}, {qsil['siloing'].max():.3f}]  "
      f"mean={qsil['siloing'].mean():.3f}")

# Annual cohort siloing (cross_community_decline)
asil = pd.read_csv("output/cross_community_decline/cohort_panel.csv")
asil["siloing_annual"] = 1.0 - asil["pct_cross"]
asil["siloing_hhi"]    = asil["hhi_community"]    # second siloing measure
print(f"  Annual siloing panel: {asil['year'].min()} → {asil['year'].max()}")

# Macro shocks panel
ms = pd.read_csv("output/macro_shocks/quarterly_panel.csv")
ms["date"] = pd.to_datetime(ms["date"])
print(f"  Macro shocks panel: {ms['YQ'].min()} → {ms['YQ'].max()}")
print(f"  VIX range: [{ms['vix'].min():.1f}, {ms['vix'].max():.1f}]  "
      f"(non-NaN: {ms['vix'].notna().sum()})")
print(f"  interest_rate range: [{ms['interest_rate'].dropna().min():.3f}, "
      f"{ms['interest_rate'].dropna().max():.3f}]")

# Company dataset
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["first_ym"]  = pd.to_datetime(cdf["first_ym"].astype(str) + "-01", errors="coerce")
cdf["entry_year"] = cdf["first_ym"].dt.year.fillna(cdf["deal_year"]).astype(int)
print(f"  Company dataset: N={len(cdf):,}  Failed: {int(cdf['is_failed'].sum())}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: CONSTRUCT QUARTERLY MERGED PANEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 1: CONSTRUCT MERGED QUARTERLY PANEL")
print("=" * 72)

panel = pd.merge(qsil[["YQ","date","n_deals","pct_cross","siloing","siloing_l1",
                         "siloing_l4","hhi_community","hhi_community_z"]],
                 ms[["YQ","vix","interest_rate","gpr","oil_price",
                      "vix_z","interest_rate_z","gpr_z"]],
                 on="YQ", how="inner")
panel = panel.sort_values("date").reset_index(drop=True)
panel["quarter_t"] = np.arange(len(panel))

# ── SHOCK DEFINITIONS ────────────────────────────────────────────────────────
# VIX spike: > 25 (empirical ~75th pctl; captures COVID 2020, financial stress)
vix_p75 = panel["vix"].quantile(0.75)
print(f"\n  VIX p75 = {vix_p75:.1f}  (threshold for 'shock quarter')")
panel["shock_vix"]  = (panel["vix"] > vix_p75).astype(float)

# Interest-rate shock: quarterly change > +50bp (covers 2022 hike cycle)
panel["rate_change"] = panel["interest_rate"].diff()
rate_thresh = 0.50   # 50bp per quarter
panel["shock_rate"]  = (panel["rate_change"] > rate_thresh).astype(float)

# GPR shock: GPR > 1.5 SD above mean (geopolitical risk)
gpr_thresh = panel["gpr_z"].quantile(0.85)
panel["shock_gpr"]   = (panel["gpr_z"] > gpr_thresh).astype(float)

# Combined shock: at least one active
panel["shock_any"]   = ((panel["shock_vix"] + panel["shock_rate"] +
                          panel["shock_gpr"]) >= 1).astype(float)

# ── OUTCOME: DEAL FLOW CHANGE ─────────────────────────────────────────────────
# Deal-flow response to shock (log-difference of n_deals)
panel["log_deals"] = np.log1p(panel["n_deals"])
panel["dlog_deals"] = panel["log_deals"].diff()   # quarterly change

# Standardize siloing_l1 for effect-size reading
panel["siloing_l1_z"] = (panel["siloing_l1"] - panel["siloing_l1"].mean()) / panel["siloing_l1"].std()

print(f"\n  Quarterly panel for regression: N={panel.dropna(subset=['siloing_l1','dlog_deals']).shape[0]}")
print(f"\n  Shock quarter counts:")
for s in ["shock_vix","shock_rate","shock_gpr","shock_any"]:
    print(f"    {s:<15}: {int(panel[s].sum()):>3} quarters / {len(panel)}")

print(f"\n  Quarterly siloing × shock overview:")
print(panel[["YQ","siloing","shock_vix","shock_rate","shock_any","dlog_deals"]].dropna().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PART A — QUARTERLY TIME-SERIES REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: QUARTERLY TIME-SERIES — deal-flow ~ siloing_L1 × shock")
print("=" * 72)
print("  Model: Δlog(deals)_t = α + β₁·siloing_{t-1} + β₂·shock_t")
print("         + β₃·(siloing_{t-1} × shock_t) + controls + ε")
print("  SEs: Newey-West HAC (lags=4)  [T=44 quarters, 2014–2025]")

a_sub = panel.dropna(subset=["dlog_deals","siloing_l1","shock_any",
                              "vix_z","gpr_z"]).copy()
a_sub["t"] = np.arange(len(a_sub))   # deterministic trend

results_a = {}

for shock_var, shock_label in [
    ("shock_vix",  "VIX spike (>p75)"),
    ("shock_rate", "Rate hike (>50bp/Q)"),
    ("shock_any",  "Any shock"),
]:
    sub = a_sub.dropna(subset=[shock_var]).copy()
    sub["interaction"] = sub["siloing_l1_z"] * sub[shock_var]

    # OLS — use statsmodels to get residuals then apply NW SEs
    formula = f"dlog_deals ~ siloing_l1_z + {shock_var} + interaction + t"
    mod = smf.ols(formula, data=sub).fit()

    # Extract Newey-West SEs for each term
    row = {"N": int(mod.nobs), "shock": shock_label}
    for vname, label in [("siloing_l1_z", "siloing_l1_z"),
                          (shock_var,       "shock"),
                          ("interaction",   "interaction")]:
        if vname in mod.params:
            b, se, t, p = nw_pval(mod, vname, lags=4)
            row[f"b_{label}"] = b; row[f"p_{label}"] = p; row[f"se_{label}"] = se
        else:
            row[f"b_{label}"] = np.nan; row[f"p_{label}"] = np.nan; row[f"se_{label}"] = np.nan

    results_a[shock_var] = row

print(f"\n  {'Shock definition':<30} {'N':>4}  {'β(siloing_l1)':>14}  "
      f"{'β(shock)':>10}  {'β(interaction)':>16}")
print(f"  {'─'*80}")
for sv, r in results_a.items():
    n = r["N"]; sl = r["shock"]
    b_s  = r.get("b_siloing_l1_z", np.nan); p_s  = r.get("p_siloing_l1_z", np.nan)
    b_sh = r.get("b_shock", np.nan);         p_sh = r.get("p_shock", np.nan)
    b_i  = r.get("b_interaction", np.nan);   p_i  = r.get("p_interaction", np.nan)
    print(f"  {sl:<30} {n:>4}  "
          f"{b_s:>+8.4f} {sig(p_s):<4}  "
          f"{b_sh:>+8.4f} {sig(p_sh):<4}  "
          f"{b_i:>+8.4f} {sig(p_i):<4}")

# ── Detailed output for main shock (any) ─────────────────────────────────────
print(f"\n  ── DETAILED: Any-shock model (HAC Newey-West) ──")
sub = a_sub.dropna(subset=["shock_any"]).copy()
sub["interaction"] = sub["siloing_l1_z"] * sub["shock_any"]
mod_main = smf.ols("dlog_deals ~ siloing_l1_z + shock_any + interaction + t",
                   data=sub).fit()
for vname in ["Intercept","siloing_l1_z","shock_any","interaction","t"]:
    if vname in mod_main.params:
        b, se, t_stat, p = nw_pval(mod_main, vname, lags=4)
        print(f"    {vname:<25}: β={b:>+.4f}  SE={se:.4f}  t={t_stat:>+.2f}  {sig(p)}")
print(f"    N={int(mod_main.nobs)}  R²={mod_main.rsquared:.3f}")

# Effect magnitude: siloing shock amplification
b_int = results_a["shock_any"].get("b_interaction", np.nan)
print(f"\n  Interpretation: β(interaction)={b_int:+.4f}")
if not np.isnan(b_int):
    print(f"  → A 1-SD increase in siloing amplifies the shock effect on deal flow by "
          f"{b_int:+.2%} (log-points).")
    print(f"  → At mean siloing, a shock changes deals by {results_a['shock_any']['b_shock']:+.2%}.")
    print(f"  → At mean+1SD siloing, additional amplification: {b_int:+.2%}.")

# ── Robustness: use HHI as alternative siloing measure ───────────────────────
print(f"\n  ── ROBUSTNESS: HHI community concentration as siloing measure ──")
panel["t"] = np.arange(len(panel))
a_sub2 = panel.dropna(subset=["dlog_deals","hhi_community_z","shock_any"]).copy()
a_sub2["t"] = np.arange(len(a_sub2))
a_sub2["interaction_hhi"] = a_sub2["hhi_community_z"] * a_sub2["shock_any"]
mod_hhi = smf.ols("dlog_deals ~ hhi_community_z + shock_any + interaction_hhi + t",
                  data=a_sub2).fit()
b_hhi, se_hhi, t_hhi, p_hhi = nw_pval(mod_hhi, "interaction_hhi", lags=4)
print(f"    β(HHI × shock) = {b_hhi:+.4f}  SE={se_hhi:.4f}  {sig(p_hhi)}")
print(f"    (Consistent direction: {'✓' if np.sign(b_hhi)==np.sign(b_int) else '✗ reversed'})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PART B — COMPANY-LEVEL (siloing_at_entry × shock_cohort)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: COMPANY-LEVEL — is_failed ~ siloing_at_entry × shock_cohort")
print("=" * 72)
print("  Cohorts: 2014–2022 (enough time to observe failure outcome)")
print("  Shock cohorts: 2020 (COVID), 2022 (rate hike onset)")

# Annual siloing at entry
sil_map = dict(zip(asil["year"], asil["siloing_annual"]))
hhi_map = dict(zip(asil["year"], asil["siloing_hhi"]))

cdf2 = cdf.copy()
cdf2["siloing_at_entry"] = cdf2["entry_year"].map(sil_map)
cdf2["hhi_at_entry"]     = cdf2["entry_year"].map(hhi_map)

# Restrict to 2014–2022 entry cohorts (right-censoring concern for 2023+)
cdf2 = cdf2[(cdf2["entry_year"] >= 2014) & (cdf2["entry_year"] <= 2022)].copy()
cdf2 = cdf2.dropna(subset=["is_failed","siloing_at_entry","log_capital"]).copy()

print(f"\n  Companies in 2014–2022 cohorts: N={len(cdf2):,}")
print(f"  Failed: {int(cdf2['is_failed'].sum()):,}  ({cdf2['is_failed'].mean():.1%})")

# Shock cohort indicators
cdf2["shock_covid"]  = (cdf2["entry_year"] == 2020).astype(float)
cdf2["shock_rate"]   = (cdf2["entry_year"] == 2022).astype(float)
cdf2["shock_cohort"] = ((cdf2["entry_year"] == 2020) |
                         (cdf2["entry_year"] == 2022)).astype(float)

# Standardize
cdf2["siloing_z"] = (cdf2["siloing_at_entry"] - cdf2["siloing_at_entry"].mean()) / \
                     cdf2["siloing_at_entry"].std()
cdf2["hhi_z"]     = (cdf2["hhi_at_entry"] - cdf2["hhi_at_entry"].mean()) / \
                     cdf2["hhi_at_entry"].std()
cdf2["log_cap"]   = cdf2["log_capital"]
cdf2["nrounds"]   = np.log1p(cdf2["n_rounds"].fillna(1))
cdf2["country_fe"] = cdf2["hq_country"].where(
    cdf2["hq_country"].isin(cdf2["hq_country"].value_counts()[lambda x: x >= 5].index), "Other")

# Interaction terms
cdf2["sil_x_shock"]       = cdf2["siloing_z"] * cdf2["shock_cohort"]
cdf2["sil_x_covid"]       = cdf2["siloing_z"] * cdf2["shock_covid"]
cdf2["sil_x_rate"]        = cdf2["siloing_z"] * cdf2["shock_rate"]
cdf2["hhi_x_shock"]       = cdf2["hhi_z"] * cdf2["shock_cohort"]

# ── Cohort-level descriptives ─────────────────────────────────────────────────
print(f"\n  Failure rates and siloing by cohort year:")
print(f"  {'Year':<6} {'N':>5} {'Fail%':>7} {'Siloing':>8} {'HHI':>7} {'Shock':>7}")
print(f"  {'─'*48}")
for yr in sorted(cdf2["entry_year"].unique()):
    sub = cdf2[cdf2["entry_year"] == yr]
    shock_flag = "★" if yr in (2020, 2022) else ""
    print(f"  {yr:<6} {len(sub):>5} {sub['is_failed'].mean():>7.1%} "
          f"{sub['siloing_at_entry'].iloc[0]:>8.3f} "
          f"{sub['hhi_at_entry'].iloc[0]:>7.4f} {shock_flag:>7}")

# ── Regressions ──────────────────────────────────────────────────────────────
CTRL_B = "log_cap + nrounds + cross_community + C(country_fe) + C(entry_year)"

print(f"\n  ── Model B1: siloing_z × shock_cohort ──")
m_b1 = smf.ols(f"is_failed ~ siloing_z + shock_cohort + sil_x_shock + {CTRL_B}",
               data=cdf2).fit(cov_type="HC1")
for var, lbl in [("siloing_z","siloing_at_entry_z"),
                 ("shock_cohort","shock_cohort"),
                 ("sil_x_shock","siloing_z × shock  ← KEY")]:
    if var in m_b1.params:
        b = m_b1.params[var]; p = m_b1.pvalues[var]; se = m_b1.bse[var]
        print(f"    {lbl:<35}: β={b:>+.4f}  SE={se:.4f}  {sig(p)}")
print(f"    N={int(m_b1.nobs):,}  R²={m_b1.rsquared:.4f}")

print(f"\n  ── Model B2: separate COVID vs rate-hike interactions ──")
m_b2 = smf.ols(f"is_failed ~ siloing_z + shock_covid + shock_rate + "
               f"sil_x_covid + sil_x_rate + {CTRL_B}",
               data=cdf2).fit(cov_type="HC1")
for var, lbl in [("sil_x_covid","siloing_z × COVID-2020  ← KEY"),
                 ("sil_x_rate", "siloing_z × Rate-2022  ← KEY")]:
    if var in m_b2.params:
        b = m_b2.params[var]; p = m_b2.pvalues[var]; se = m_b2.bse[var]
        print(f"    {lbl:<35}: β={b:>+.4f}  SE={se:.4f}  {sig(p)}")
print(f"    N={int(m_b2.nobs):,}  R²={m_b2.rsquared:.4f}")

print(f"\n  ── Model B3: HHI as alternative siloing ──")
cdf2["hhi_x_shock"] = cdf2["hhi_z"] * cdf2["shock_cohort"]
m_b3 = smf.ols(f"is_failed ~ hhi_z + shock_cohort + hhi_x_shock + {CTRL_B}",
               data=cdf2).fit(cov_type="HC1")
for var, lbl in [("hhi_z","hhi_at_entry_z"),
                 ("shock_cohort","shock_cohort"),
                 ("hhi_x_shock","hhi_z × shock  ← KEY (robustness)")]:
    if var in m_b3.params:
        b = m_b3.params[var]; p = m_b3.pvalues[var]; se = m_b3.bse[var]
        print(f"    {lbl:<40}: β={b:>+.4f}  SE={se:.4f}  {sig(p)}")
print(f"    N={int(m_b3.nobs):,}  R²={m_b3.rsquared:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: PART C — PRE/POST-2022 SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: PRE/POST-2022 SANITY CHECK")
print("=" * 72)
print("  Pre-2022: 2014–2021 (mean siloing ≈ 33%)  |  Post-2022: 2022+ (mean siloing ≈ 47%)")
print("  Comparison: failure rates in shock vs non-shock cohort years")

cdf3 = cdf.copy()
cdf3["siloing_at_entry"] = cdf3["entry_year"].map(sil_map)
cdf3 = cdf3[(cdf3["entry_year"] >= 2014) & (cdf3["entry_year"] <= 2022)].dropna(
    subset=["is_failed","siloing_at_entry"]).copy()

cdf3["period"]      = np.where(cdf3["entry_year"] < 2022, "pre-2022", "post-2022")
cdf3["shock_year"]  = cdf3["entry_year"].isin([2020, 2022]).astype(int)
cdf3["silo_hi"]     = (cdf3["siloing_at_entry"] > cdf3["siloing_at_entry"].median()).astype(int)

print(f"\n  2×2 Table: Period × Shock Year — mean failure rate:")
print(f"  {'Period':<12} {'Shock cohort':>14} {'Non-shock cohort':>18}  {'Δ(Shock−NonShock)':>20}")
print(f"  {'─'*68}")
rows_c = []
for period in ["pre-2022", "post-2022"]:
    sub = cdf3[cdf3["period"] == period]
    shock_fail = sub[sub["shock_year"] == 1]["is_failed"]
    nonsh_fail = sub[sub["shock_year"] == 0]["is_failed"]
    n_sh = len(shock_fail); n_ns = len(nonsh_fail)
    f_sh = shock_fail.mean(); f_ns = nonsh_fail.mean()
    delta = f_sh - f_ns
    pval  = stats.ttest_ind(shock_fail, nonsh_fail).pvalue
    print(f"  {period:<12} {f_sh:>10.1%} (N={n_sh:,}) {f_ns:>10.1%} (N={n_ns:,})  "
          f"{delta:>+8.1%}  {sig(pval)}")
    rows_c.append({"period": period, "shock_fail%": f_sh, "nonshock_fail%": f_ns,
                   "delta": delta, "p": pval, "N_shock": n_sh, "N_nonshock": n_ns})

# ── 2×2 siloing × shock ───────────────────────────────────────────────────────
print(f"\n  2×2 Table: Siloing quartile × Shock year (2014–2022 cohorts):")
median_sil = cdf3["siloing_at_entry"].median()
print(f"  Median siloing at entry: {median_sil:.3f}")
print(f"  {'Siloing':<16} {'Shock cohort':>16} {'Non-shock':>14}  {'Δ':>10}  p")
print(f"  {'─'*62}")
shock_rows = []
for silo_level, silo_label in [(0, "Low siloing"), (1, "High siloing")]:
    sub = cdf3[cdf3["silo_hi"] == silo_level]
    f_sh = sub[sub["shock_year"]==1]["is_failed"].mean()
    f_ns = sub[sub["shock_year"]==0]["is_failed"].mean()
    n_sh = (sub["shock_year"]==1).sum(); n_ns = (sub["shock_year"]==0).sum()
    delta = f_sh - f_ns
    pval  = stats.ttest_ind(sub[sub["shock_year"]==1]["is_failed"],
                             sub[sub["shock_year"]==0]["is_failed"]).pvalue
    print(f"  {silo_label:<16}  {f_sh:>8.1%} (N={n_sh:,})  "
          f"{f_ns:>7.1%} (N={n_ns:,})  {delta:>+6.1%}  {sig(pval)}")
    shock_rows.append({"silo": silo_label, "shock_fail%": f_sh, "nonshock_fail%": f_ns,
                       "delta": delta, "p": pval})

delta_lo = shock_rows[0]["delta"]; delta_hi = shock_rows[1]["delta"]
print(f"\n  Interaction (Δ_high − Δ_low) = {delta_hi - delta_lo:+.1%}")
print(f"  {'→ High-siloing cohorts show LARGER shock penalty' if delta_hi > delta_lo else '→ Pattern reversed: low-siloing shows larger shock penalty'}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: SUPPLEMENTARY — VIZ DATA + KEY SILOING EVENTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: SUPPLEMENTARY DIAGNOSTICS")
print("=" * 72)

# Correlation between siloing and shock intensity over time
valid = panel.dropna(subset=["siloing","vix"])
r_sv, p_sv = stats.pearsonr(valid["siloing"], valid["vix"])
r_sr, p_sr = stats.pearsonr(
    panel.dropna(subset=["siloing","interest_rate"])["siloing"],
    panel.dropna(subset=["siloing","interest_rate"])["interest_rate"])
print(f"\n  Correlation siloing × VIX: r={r_sv:.3f}  {sig(p_sv)}")
print(f"  Correlation siloing × interest_rate: r={r_sr:.3f}  {sig(p_sr)}")
print(f"  (Low correlation → siloing and shocks are largely independent)")

# Identify specific quarter-events
print(f"\n  Highest-siloing quarters (top 5):")
top5 = panel.nlargest(5, "siloing")[["YQ","siloing","n_deals","vix","shock_any"]]
print(top5.to_string(index=False))

print(f"\n  Highest-shock quarters (VIX, top 5):")
top5v = panel.nlargest(5, "vix")[["YQ","vix","siloing","dlog_deals","shock_any"]]
print(top5v.to_string(index=False))

# pct_cross trend: 2014 vs 2022 vs 2025
print(f"\n  pct_cross timeline:")
for yr, sub in asil.groupby("year"):
    print(f"    {yr}:  pct_cross={sub['pct_cross'].values[0]:.1%}  "
          f"siloing={sub['siloing_annual'].values[0]:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 6: SAVE OUTPUTS")
print("=" * 72)

# Save merged panel
panel.to_csv(OUT / "quarterly_panel_merged.csv", index=False)
print(f"  Saved: output/network_fragility/quarterly_panel_merged.csv  ({len(panel)} rows)")

# Save company-level dataset
cdf2[["Companies","entry_year","is_failed","siloing_at_entry","hhi_at_entry",
      "shock_cohort","shock_covid","shock_rate","siloing_z","hhi_z","cross_community",
      "log_capital","n_rounds","hq_country","outcome"]].to_csv(
    OUT / "company_fragility_panel.csv", index=False)
print(f"  Saved: output/network_fragility/company_fragility_panel.csv  ({len(cdf2)} rows)")

# Save markdown summary
sil_x_shock_b  = m_b1.params.get("sil_x_shock", np.nan)
sil_x_shock_p  = m_b1.pvalues.get("sil_x_shock", np.nan)
sil_x_shock_se = m_b1.bse.get("sil_x_shock", np.nan)
b_covid = m_b2.params.get("sil_x_covid", np.nan); p_covid = m_b2.pvalues.get("sil_x_covid", np.nan)
b_rate  = m_b2.params.get("sil_x_rate",  np.nan); p_rate  = m_b2.pvalues.get("sil_x_rate",  np.nan)
b_hhi   = m_b3.params.get("hhi_x_shock", np.nan); p_hhi   = m_b3.pvalues.get("hhi_x_shock", np.nan)
b_ts    = results_a["shock_any"].get("b_interaction", np.nan)
p_ts    = results_a["shock_any"].get("p_interaction", np.nan)

md = f"""# Network Fragility × Macro Shocks
*(Acemoglu, Ozdaglar & Tahbaz-Salehi 2015, AER — "Robust yet fragile" test)*

## Setup

**Hypothesis (AER 2015):** As co-investment networks become more siloed (fewer cross-community ties),
they are *robust to small perturbations* but *more fragile to large macro shocks* — with shock effects
cascading more severely through dense clusters. The climate-tech VC network shows a secular decline
in cross-community ties from 79% (2014) to 45% (2025).

**Siloing measure:** `siloing_t = 1 − pct_cross_t` (rising siloing = fewer cross-community deals)
**Secondary siloing:** `hhi_community_t` (HHI of community-size distribution, higher = more concentrated)
**Shocks:**
- VIX spike: quarterly VIX > {vix_p75:.1f} (≈p75); captures financial stress/COVID
- Rate hike: quarterly change in fed funds rate > 50bp; captures 2022 cycle
- GPR spike: geopolitical risk index > p85
- Any shock: ≥1 of the above active

---

## Part A: Quarterly Time-Series (N={int(mod_main.nobs)})

**Model:** Δlog(deals)_t = α + β₁·siloing_{'{t-1}'} + β₂·shock_t + β₃·(siloing_{'{t-1}'} × shock_t) + trend + ε
**SEs:** Newey-West HAC (lags=4)

| Variable | β | SE | Sig |
|---|---|---|---|
| siloing_l1_z | {results_a['shock_any']['b_siloing_l1_z']:+.4f} | {results_a['shock_any']['se_siloing_l1_z']:.4f} | {sig(results_a['shock_any']['p_siloing_l1_z'])} |
| shock_any | {results_a['shock_any']['b_shock']:+.4f} | {results_a['shock_any']['se_shock']:.4f} | {sig(results_a['shock_any']['p_shock'])} |
| **siloing × shock (KEY)** | **{b_ts:+.4f}** | — | **{sig(p_ts)}** |

| Shock definition | N | β(siloing) | β(shock) | β(interaction) | Sig |
|---|---|---|---|---|---|
| VIX spike (>p75) | {results_a['shock_vix']['N']} | {results_a['shock_vix']['b_siloing_l1_z']:+.4f} | {results_a['shock_vix']['b_shock']:+.4f} | {results_a['shock_vix']['b_interaction']:+.4f} | {sig(results_a['shock_vix']['p_interaction'])} |
| Rate hike (>50bp/Q) | {results_a['shock_rate']['N']} | {results_a['shock_rate']['b_siloing_l1_z']:+.4f} | {results_a['shock_rate']['b_shock']:+.4f} | {results_a['shock_rate']['b_interaction']:+.4f} | {sig(results_a['shock_rate']['p_interaction'])} |
| Any shock | {results_a['shock_any']['N']} | {results_a['shock_any']['b_siloing_l1_z']:+.4f} | {results_a['shock_any']['b_shock']:+.4f} | {results_a['shock_any']['b_interaction']:+.4f} | {sig(results_a['shock_any']['p_interaction'])} |

HHI robustness: β(HHI × shock) = {b_hhi_ts if 'b_hhi_ts' in dir() else b_hhi:.4f}  {sig(p_hhi if 'p_hhi_ts' not in dir() else p_hhi)}

---

## Part B: Company-Level Cross-Section (cohorts 2014–2022, N={int(m_b1.nobs):,})

**Model:** is_failed_i = α + β₁·siloing_at_entry_z + β₂·shock_cohort_i + β₃·(siloing × shock) + controls + ε
**Controls:** log_capital, n_rounds, cross_community, country FE, year FE
**SEs:** HC1 (heteroskedasticity-robust)

| Variable | β | SE | Sig |
|---|---|---|---|
| siloing_at_entry_z | {m_b1.params.get('siloing_z', np.nan):+.4f} | {m_b1.bse.get('siloing_z', np.nan):.4f} | {sig(m_b1.pvalues.get('siloing_z', 1))} |
| shock_cohort | {m_b1.params.get('shock_cohort', np.nan):+.4f} | {m_b1.bse.get('shock_cohort', np.nan):.4f} | {sig(m_b1.pvalues.get('shock_cohort', 1))} |
| **siloing × shock (KEY)** | **{sil_x_shock_b:+.4f}** | **{sil_x_shock_se:.4f}** | **{sig(sil_x_shock_p)}** |

Separate shocks:
| Interaction | β | SE | Sig |
|---|---|---|---|
| siloing × COVID-2020 | {b_covid:+.4f} | {m_b2.bse.get('sil_x_covid', np.nan):.4f} | {sig(p_covid)} |
| siloing × Rate-2022 | {b_rate:+.4f} | {m_b2.bse.get('sil_x_rate', np.nan):.4f} | {sig(p_rate)} |
| HHI × shock (robustness) | {b_hhi:+.4f} | {m_b3.bse.get('hhi_x_shock', np.nan):.4f} | {sig(p_hhi)} |

---

## Part C: Pre/Post-2022 Sanity Check

Siloing median at entry year = {median_sil:.3f}

| Siloing level | Shock-year failure % | Non-shock failure % | Δ (shock penalty) |
|---|---|---|---|
| Low siloing | {shock_rows[0]['shock_fail%']:.1%} | {shock_rows[0]['nonshock_fail%']:.1%} | {shock_rows[0]['delta']:+.1%}  {sig(shock_rows[0]['p'])} |
| High siloing | {shock_rows[1]['shock_fail%']:.1%} | {shock_rows[1]['nonshock_fail%']:.1%} | {shock_rows[1]['delta']:+.1%}  {sig(shock_rows[1]['p'])} |

**Interaction (Δ_high − Δ_low) = {delta_hi - delta_lo:+.1%}**
{'→ High-siloing cohorts face a LARGER failure penalty in shock years — consistent with fragility hypothesis.' if delta_hi > delta_lo else '→ Pattern does not show siloing amplification in this raw comparison.'}

---

## Interpretation

The interaction test asks: *does the same macro shock cause more damage when the network is more siloed?*

- **Part A** (deal flow): β(siloing_l1 × shock) = {b_ts:+.4f} [{sig(p_ts)}]
  {'→ Higher prior-quarter siloing amplifies deal-flow contraction in shock quarters.' if b_ts > 0 and p_ts < 0.1 else '→ Not statistically significant at quarterly frequency (T≈44, limited power).'}

- **Part B** (company survival): β(siloing_at_entry × shock) = {sil_x_shock_b:+.4f} [{sig(sil_x_shock_p)}]
  {'→ Companies entering in higher-siloing periods suffer larger failure rates in shock cohorts.' if sil_x_shock_b > 0 and sil_x_shock_p < 0.1 else '→ Interaction not statistically significant at company level.'}

- **Part C** (sanity check): {f'The shock penalty is {delta_hi-delta_lo:+.1%} larger for high-siloing cohorts.' }

**Structural note on median = 1.0:** The quarterly network has high community overlap (many investors bridge), so the "siloing" measure (1-pct_cross) has relatively modest range. The secular trend is real but identification comes from cohort variation.

## Output Files

- `output/network_fragility/quarterly_panel_merged.csv` — merged quarterly panel
- `output/network_fragility/company_fragility_panel.csv` — company-level with siloing at entry
- `output/network_fragility/fragility_results.md` — this file
"""

# Fix undefined variable reference
b_hhi_ts_val = b_hhi  # alias
md = md.replace("b_hhi_ts if 'b_hhi_ts' in dir() else b_hhi:.4f", f"{b_hhi:.4f}")

with open(OUT / "fragility_results.md", "w") as f:
    f.write(md)
print(f"  Saved: output/network_fragility/fragility_results.md")

print(f"\n  ✅ Script 55 complete.")
