"""
Script 56: Policy-Shock Fragility Extension (Script 55 → sector-specific shocks)
===================================================================================
Tests whether network siloing moderates the ecosystem's deal-flow response to
POLICY shocks (not generic financial shocks), motivated by the echo-chamber
mechanism: policy signals require interpretation, so silo-locked investors
react less to positive shocks and over-cascade to negative ones.

Data:
  - Quarterly siloing panel (cross_community_decline, 2014Q1–2025Q4)
  - Macro shocks panel (macro_shocks, VIX/rate/GPR, same dates)
  - Noailly et al. (2024) EnvP index [downloaded, Jan 1981–Mar 2019]
  - Noailly et al. (2022) EnvPU index [downloaded, Jan 1990–Mar 2019]
  - 7 hand-coded discrete policy events (binary dummies)

Models (all: Δlog(deals) ~ siloing_l1_z × shock + controls, HAC NW lags=4):
  M1: Any policy shock (7 events pooled)
  M2: Negative policy shocks only (4 events)
  M3: Positive policy shocks only (3 events)
  M4: Rate hike (Script 55 replication)
  M5: VIX spike (Script 55 replication)
  M6: Continuous EnvP level (quarterly mean, 2014–2019 subsample)
  M7: Continuous EnvPU level (quarterly mean, 2014–2019 subsample)
  M_comp: Policy + VIX interactions jointly, Wald test for difference
"""

import warnings; warnings.filterwarnings("ignore")
import sys, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

sys.path.insert(0, ".")
OUT = Path("output/policy_shock_fragility"); OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 \
           else "†" if p < 0.10 else "n.s."

def nw_pval(model, varname, lags=4):
    """Newey-West HAC p-value for a single coefficient."""
    cov = cov_hac(model, nlags=lags)
    idx = list(model.params.index).index(varname)
    se  = float(np.sqrt(cov[idx, idx]))
    b   = float(model.params[varname])
    t   = b / se
    df  = max(int(model.nobs) - len(model.params), 1)
    p   = float(2 * (1 - stats.t.cdf(abs(t), df=df)))
    return b, se, t, p

def run_model(sub, shock_var, label, lags=4, dep="dlog_deals"):
    """OLS + HAC SEs: dep ~ siloing_l1_z + shock + interaction + trend + Q-dummies."""
    sub = sub.dropna(subset=[dep, "siloing_l1_z", shock_var]).copy()
    sub["interaction"] = sub["siloing_l1_z"] * sub[shock_var]
    # Quarter dummies (drop Q1 as reference)
    for q in ["q2","q3","q4"]:
        if q not in sub.columns:
            sub[q] = 0.0
    formula = f"{dep} ~ siloing_l1_z + {shock_var} + interaction + t + q2 + q3 + q4"
    try:
        mod = smf.ols(formula, data=sub).fit()
    except Exception as e:
        return {"label": label, "N": len(sub), "N_shock": int(sub[shock_var].sum()),
                "b_sil": np.nan, "b_shock": np.nan, "b_int": np.nan,
                "se_int": np.nan, "p_int": np.nan, "corr_sil_shock": np.nan,
                "error": str(e)}
    N_shock = int(sub[shock_var].sum())
    try:
        b_int, se_int, t_int, p_int = nw_pval(mod, "interaction", lags=lags)
        b_sil, _, _, _ = nw_pval(mod, "siloing_l1_z", lags=lags)
        b_sh,  _, _, _ = nw_pval(mod, shock_var, lags=lags)
    except Exception:
        b_int = se_int = p_int = b_sil = b_sh = np.nan
    corr = sub["siloing_l1_z"].corr(sub[shock_var])
    return {"label": label, "N": int(mod.nobs), "N_shock": N_shock,
            "b_sil": b_sil, "b_shock": b_sh,
            "b_int": b_int, "se_int": se_int, "p_int": p_int,
            "corr_sil_shock": corr}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & REBUILD QUARTERLY PANEL WITH PROPER DATE-BASED LAGS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1: LOAD DATA")
print("=" * 72)

# Full siloing quarterly panel (47 quarters, including 2014Q2/Q3 gaps)
qsil = pd.read_csv("output/cross_community_decline/quarterly_panel.csv")
qsil["date"] = pd.to_datetime(qsil["date"])
qsil["YQ"]   = qsil["YQ"].str.strip()
qsil["siloing"] = 1.0 - qsil["pct_cross"]

# Build proper date-based lag: siloing for the previous CALENDAR quarter
yq_to_sil = dict(zip(qsil["YQ"], qsil["siloing"]))
def prev_quarter_yq(yq):
    y, q = int(yq[:4]), int(yq[-1])
    pq = q - 1 if q > 1 else 4
    py = y if q > 1 else y - 1
    return f"{py}Q{pq}"

qsil["siloing_l1_proper"] = qsil["YQ"].map(lambda yq: yq_to_sil.get(prev_quarter_yq(yq)))

# Macro shocks panel
ms = pd.read_csv("output/macro_shocks/quarterly_panel.csv")
ms["date"] = pd.to_datetime(ms["date"])
ms["YQ"]   = ms["YQ"].str.strip()

# Compute quarter dummies and deal-flow change
ms["log_deals"]  = np.log1p(ms["n_deals"])
ms["dlog_deals"] = ms["log_deals"].diff()
ms["q2"] = (ms["date"].dt.month == 4).astype(float)
ms["q3"] = (ms["date"].dt.month == 7).astype(float)
ms["q4"] = (ms["date"].dt.month == 10).astype(float)

# Merge
panel = pd.merge(
    qsil[["YQ","date","n_deals","pct_cross","siloing","siloing_l1_proper",
          "hhi_community","hhi_community_z"]].rename(
              columns={"n_deals":"n_deals_sil", "date":"date_sil"}),
    ms[["YQ","date","n_deals","vix","interest_rate","gpr","oil_price",
        "vix_z","interest_rate_z","gpr_z","log_deals","dlog_deals","q2","q3","q4"]],
    on="YQ", how="inner"
).sort_values("date").reset_index(drop=True)

panel["t"]          = np.arange(len(panel))
panel["siloing_l1"] = panel["siloing_l1_proper"]
panel["siloing_l1_z"] = (
    (panel["siloing_l1"] - panel["siloing_l1"].mean()) /
     panel["siloing_l1"].std()
)
panel["hhi_l1"]   = panel["hhi_community"].shift(1)
panel["hhi_l1_z"] = (panel["hhi_l1"] - panel["hhi_l1"].mean()) / panel["hhi_l1"].std()

# VIX and rate shocks (Script 55 replication)
vix_p75 = panel["vix"].quantile(0.75)
panel["shock_vix"]  = (panel["vix"] > vix_p75).astype(float)
panel["rate_change"] = panel["interest_rate"].diff()
panel["shock_rate"]  = (panel["rate_change"] > 0.50).astype(float)

print(f"  Merged panel: {panel['YQ'].iloc[0]} → {panel['YQ'].iloc[-1]}  N={len(panel)}")
print(f"  dlog_deals valid: {panel['dlog_deals'].notna().sum()}")
print(f"  siloing_l1_z valid: {panel['siloing_l1_z'].notna().sum()}")
print(f"  VIX p75 threshold: {vix_p75:.1f}  shock_vix quarters: {int(panel['shock_vix'].sum())}")
print(f"  shock_rate quarters (>50bp): {int(panel['shock_rate'].sum())}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: LOAD EnvP / EnvPU & AGGREGATE TO QUARTERLY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: LOAD EnvP/EnvPU & AGGREGATE TO QUARTERLY")
print("=" * 72)

envp_raw  = pd.read_excel("/tmp/EnvP_data.xls")
envpu_raw = pd.read_excel("/tmp/EnvPU_data.xlsx")

def monthly_to_quarterly(df, val_col):
    df = df.copy()
    df["date"] = pd.to_datetime({"year": df["Year"], "month": df["Month"], "day": 1})
    df["quarter"] = df["date"].dt.to_period("Q")
    q = df.groupby("quarter")[val_col].mean().reset_index()
    q["YQ"] = q["quarter"].dt.strftime("%YQ") + q["quarter"].dt.quarter.astype(str)
    return q[["YQ", val_col]].rename(columns={val_col: val_col.replace(" ", "_")})

envp_q  = monthly_to_quarterly(envp_raw,  "EnvP Index")
envpu_q = monthly_to_quarterly(envpu_raw, "EnvPU Index")

# Standardise (within full index, not just our panel window)
envp_q["EnvP_z"]  = (envp_q["EnvP_Index"]  - envp_q["EnvP_Index"].mean()) / envp_q["EnvP_Index"].std()
envpu_q["EnvPU_z"] = (envpu_q["EnvPU_Index"] - envpu_q["EnvPU_Index"].mean()) / envpu_q["EnvPU_Index"].std()

# How many panel quarters have EnvP coverage?
panel = panel.merge(envp_q[["YQ","EnvP_Index","EnvP_z"]], on="YQ", how="left")
panel = panel.merge(envpu_q[["YQ","EnvPU_Index","EnvPU_z"]], on="YQ", how="left")

n_envp  = panel["EnvP_Index"].notna().sum()
n_envpu = panel["EnvPU_Index"].notna().sum()
print(f"  EnvP quarters available in panel:  {n_envp} / {len(panel)}")
print(f"  EnvPU quarters available in panel: {n_envpu} / {len(panel)}")
print(f"  EnvP range in panel: [{panel['EnvP_Index'].min():.1f}, {panel['EnvP_Index'].max():.1f}]  "
      f"mean={panel['EnvP_Index'].mean():.1f}")
print(f"  EnvPU range in panel: [{panel['EnvPU_Index'].min():.1f}, {panel['EnvPU_Index'].max():.1f}]  "
      f"mean={panel['EnvPU_Index'].mean():.1f}")

# ── EnvP/EnvPU aligned with policy events ────────────────────────────────────
print(f"\n  EnvP values at key policy quarters:")
for yq in ["2015Q4","2016Q4","2017Q2","2019Q4"]:
    row = panel[panel["YQ"]==yq]
    if not row.empty:
        print(f"    {yq}: EnvP={row['EnvP_Index'].values[0]:.1f}  "
              f"EnvPU={row['EnvPU_Index'].values[0]:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CODE POLICY SHOCK DUMMIES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: POLICY EVENT DUMMIES")
print("=" * 72)

POSITIVE_EVENTS = {
    "2015Q4": "Paris Agreement adopted (Dec 2015)",
    "2021Q1": "Biden inauguration + Paris re-entry (Jan 2021)",
    "2022Q3": "Inflation Reduction Act signed (Aug 2022)",
}
NEGATIVE_EVENTS = {
    "2016Q4": "Trump election (Nov 2016)",
    "2017Q2": "Trump announces Paris withdrawal (Jun 2017)",
    "2019Q4": "Formal US Paris withdrawal notification (Nov 2019)",
    "2025Q1": "Trump 2.0 inauguration + IRA repeal threats (Jan 2025)",
}

panel["shock_policy_pos"]  = panel["YQ"].isin(POSITIVE_EVENTS).astype(float)
panel["shock_policy_neg"]  = panel["YQ"].isin(NEGATIVE_EVENTS).astype(float)
panel["shock_policy_any"]  = ((panel["shock_policy_pos"] + panel["shock_policy_neg"]) > 0).astype(float)

print(f"\n  {'Quarter':<10} {'YQ':<8} {'siloing':>8} {'dlog_deals':>12} {'shock':>8} {'event'}")
print(f"  {'─'*80}")
all_events = {**{k: (v, "POS") for k, v in POSITIVE_EVENTS.items()},
              **{k: (v, "NEG") for k, v in NEGATIVE_EVENTS.items()}}
for yq in sorted(all_events.keys()):
    desc, direction = all_events[yq]
    row = panel[panel["YQ"] == yq]
    if row.empty:
        print(f"  ⚠ {yq} NOT IN PANEL")
        continue
    sil = row["siloing"].values[0]
    dl  = row["dlog_deals"].values[0] if pd.notna(row["dlog_deals"].values[0]) else float("nan")
    env = row["EnvP_Index"].values[0] if pd.notna(row["EnvP_Index"].values[0]) else float("nan")
    env_str = f"EnvP={env:.0f}" if not np.isnan(env) else "EnvP=N/A"
    print(f"  {direction:<10} {yq:<8} {sil:>8.3f} {dl:>+12.3f} {'★'*1:>8}  "
          f"{desc[:45]}  [{env_str}]")

# Confound check: correlation of each shock with siloing_l1_z
print(f"\n  Confound check: correlation of shock variable with siloing_l1_z")
print(f"  {'Shock':<25} {'r':>7}  {'p':>8}  comment")
print(f"  {'─'*55}")
for sv, slabel in [("shock_policy_any","policy_any"), ("shock_policy_pos","policy_pos"),
                   ("shock_policy_neg","policy_neg"), ("shock_rate","rate_hike"),
                   ("shock_vix","vix_spike")]:
    tmp = panel.dropna(subset=["siloing_l1_z", sv])
    if len(tmp) < 5: continue
    r, p = stats.pearsonr(tmp["siloing_l1_z"], tmp[sv])
    flag = "⚠ CONFOUND" if abs(r) > 0.3 else "ok"
    print(f"  {slabel:<25} {r:>+7.3f}  {sig(p):>8}  {flag}")

# Also: in Script 55, we found r(siloing, interest_rate)=−0.453** — check
r_rate, p_rate = stats.pearsonr(panel.dropna(subset=["siloing_l1_z","shock_rate"])["siloing_l1_z"],
                                 panel.dropna(subset=["siloing_l1_z","shock_rate"])["shock_rate"])
print(f"\n  (Reminder: r(siloing_l1, shock_rate) = {r_rate:+.3f}  {sig(p_rate)})")
print(f"  → Policy shocks span entire sample; rate shocks cluster in 2022–23 when siloing is also high")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: RUN ALL MODELS M1–M7
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: REGRESSIONS M1–M7 (Δlog(deals) ~ siloing_l1_z × shock + controls)")
print("=" * 72)
print("  Controls: time trend (t), quarter dummies (Q2/Q3/Q4)  |  SE: NW HAC lags=4")

results = []

# ── M1–M5: binary shocks ─────────────────────────────────────────────────────
for shock_var, label in [
    ("shock_policy_any", "M1: Any policy shock (7 events)"),
    ("shock_policy_neg", "M2: Negative policy shocks (4 events)"),
    ("shock_policy_pos", "M3: Positive policy shocks (3 events)"),
    ("shock_rate",       "M4: Rate hike >50bp/Q (Script 55 repl.)"),
    ("shock_vix",        "M5: VIX spike >p75 (Script 55 repl.)"),
]:
    r = run_model(panel, shock_var, label)
    results.append(r)

# ── M6: Continuous EnvP ──────────────────────────────────────────────────────
sub6 = panel.dropna(subset=["EnvP_z","siloing_l1_z","dlog_deals"]).copy()
sub6["t"] = np.arange(len(sub6))
r6 = run_model(sub6, "EnvP_z", "M6: EnvP continuous (2014–2019)")
results.append(r6)

# ── M7: Continuous EnvPU ─────────────────────────────────────────────────────
sub7 = panel.dropna(subset=["EnvPU_z","siloing_l1_z","dlog_deals"]).copy()
sub7["t"] = np.arange(len(sub7))
r7 = run_model(sub7, "EnvPU_z", "M7: EnvPU continuous (2014–2019)")
results.append(r7)

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n  {'Model':<40} {'N':>4} {'N_shock':>8} {'β(sil×shock)':>14} {'SE':>8} {'p':>10} {'r(sil,shock)':>14}")
print(f"  {'─'*105}")
for r in results:
    n_s = r.get("N_shock", 0)
    b   = r.get("b_int", np.nan)
    se  = r.get("se_int", np.nan)
    p   = r.get("p_int", np.nan)
    cr  = r.get("corr_sil_shock", np.nan)
    flag = " ⚠" if abs(cr) > 0.3 else ""
    print(f"  {r['label']:<40} {r['N']:>4} {n_s:>8} "
          f"{b:>+12.4f}  {se:>8.4f}  {sig(p):<10}  {cr:>+8.3f}{flag}")

# ── Detailed output for M1 (main policy model) ───────────────────────────────
print(f"\n  ── DETAILED: M1 (Any policy shock) ──")
sub1 = panel.dropna(subset=["dlog_deals","siloing_l1_z","shock_policy_any"]).copy()
sub1["interaction"] = sub1["siloing_l1_z"] * sub1["shock_policy_any"]
mod1 = smf.ols("dlog_deals ~ siloing_l1_z + shock_policy_any + interaction + t + q2 + q3 + q4",
               data=sub1).fit()
for vname in ["Intercept","siloing_l1_z","shock_policy_any","interaction","t"]:
    if vname in mod1.params:
        b, se, t_stat, p = nw_pval(mod1, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  t={t_stat:>+.2f}  {sig(p)}")
print(f"    N={int(mod1.nobs)}  R²={mod1.rsquared:.3f}")

# ── Detailed output for M2 (negative policy) ─────────────────────────────────
print(f"\n  ── DETAILED: M2 (Negative policy shocks) ──")
sub2 = panel.dropna(subset=["dlog_deals","siloing_l1_z","shock_policy_neg"]).copy()
sub2["interaction"] = sub2["siloing_l1_z"] * sub2["shock_policy_neg"]
mod2 = smf.ols("dlog_deals ~ siloing_l1_z + shock_policy_neg + interaction + t + q2 + q3 + q4",
               data=sub2).fit()
for vname in ["siloing_l1_z","shock_policy_neg","interaction","t"]:
    if vname in mod2.params:
        b, se, t_stat, p = nw_pval(mod2, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  t={t_stat:>+.2f}  {sig(p)}")
print(f"    N={int(mod2.nobs)}  R²={mod2.rsquared:.3f}")

# ── Detailed output for M3 (positive policy) ─────────────────────────────────
print(f"\n  ── DETAILED: M3 (Positive policy shocks) ──")
sub3 = panel.dropna(subset=["dlog_deals","siloing_l1_z","shock_policy_pos"]).copy()
sub3["interaction"] = sub3["siloing_l1_z"] * sub3["shock_policy_pos"]
mod3 = smf.ols("dlog_deals ~ siloing_l1_z + shock_policy_pos + interaction + t + q2 + q3 + q4",
               data=sub3).fit()
for vname in ["siloing_l1_z","shock_policy_pos","interaction","t"]:
    if vname in mod3.params:
        b, se, t_stat, p = nw_pval(mod3, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  t={t_stat:>+.2f}  {sig(p)}")
print(f"    N={int(mod3.nobs)}  R²={mod3.rsquared:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: M_comp — POLICY vs FINANCIAL INTERACTION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: COMPARATIVE TEST — policy vs financial interaction")
print("=" * 72)
print("  Single model: dlog_deals ~ siloing × policy_any + siloing × vix_spike + controls")
print("  Wald test: H₀: β(sil×policy) = β(sil×vix)")

sub_c = panel.dropna(subset=["dlog_deals","siloing_l1_z",
                              "shock_policy_any","shock_vix"]).copy()
sub_c["int_policy"] = sub_c["siloing_l1_z"] * sub_c["shock_policy_any"]
sub_c["int_vix"]    = sub_c["siloing_l1_z"] * sub_c["shock_vix"]
mod_c = smf.ols("dlog_deals ~ siloing_l1_z + shock_policy_any + shock_vix "
                "+ int_policy + int_vix + t + q2 + q3 + q4",
                data=sub_c).fit()

print(f"\n  Joint model (N={int(mod_c.nobs)}, R²={mod_c.rsquared:.3f}):")
for vname in ["siloing_l1_z","shock_policy_any","shock_vix","int_policy","int_vix"]:
    if vname in mod_c.params:
        b, se, t_stat, p = nw_pval(mod_c, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  {sig(p)}")

# Wald test using statsmodels (OLS chi2 version, then convert)
try:
    wald = mod_c.f_test("int_policy = int_vix")
    f_wald  = float(wald.statistic)
    p_wald  = float(wald.pvalue)
    delta   = mod_c.params.get("int_policy", 0) - mod_c.params.get("int_vix", 0)
    print(f"\n  Wald test (int_policy = int_vix): F={f_wald:.3f}  p={p_wald:.4f}  {sig(p_wald)}")
    print(f"  Δ(policy − vix interaction) = {delta:+.4f}")
    if p_wald < 0.05:
        print(f"  ✅ Policy interaction is SIGNIFICANTLY DIFFERENT from financial shock interaction")
    else:
        print(f"  ❌ Cannot distinguish policy vs financial interaction at p<0.05")
except Exception as e:
    print(f"  Wald test failed: {e}")

# Also run with rate hike instead of VIX
print(f"\n  Joint model with rate hike (stronger Script 55 channel):")
sub_cr = panel.dropna(subset=["dlog_deals","siloing_l1_z",
                               "shock_policy_any","shock_rate"]).copy()
sub_cr["int_policy"] = sub_cr["siloing_l1_z"] * sub_cr["shock_policy_any"]
sub_cr["int_rate"]   = sub_cr["siloing_l1_z"] * sub_cr["shock_rate"]
mod_cr = smf.ols("dlog_deals ~ siloing_l1_z + shock_policy_any + shock_rate "
                 "+ int_policy + int_rate + t + q2 + q3 + q4",
                 data=sub_cr).fit()
for vname in ["int_policy","int_rate"]:
    if vname in mod_cr.params:
        b, se, t_stat, p = nw_pval(mod_cr, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  {sig(p)}")
try:
    wald2 = mod_cr.f_test("int_policy = int_rate")
    print(f"  Wald (int_policy = int_rate): F={float(wald2.statistic):.3f}  p={float(wald2.pvalue):.4f}  {sig(float(wald2.pvalue))}")
except Exception as e:
    print(f"  Wald test failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: DIAGNOSTICS — correlations, shock timeline
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 6: DIAGNOSTICS")
print("=" * 72)

# Siloing trend alongside policy events
print(f"\n  Siloing × deal-flow at each policy quarter:")
print(f"  {'Event':<10} {'YQ':<8} {'siloing_l1':>11} {'dlog_deals':>12} {'EnvP':>8} {'EnvPU':>7}")
print(f"  {'─'*65}")
for yq in sorted(all_events.keys()):
    desc, direction = all_events[yq]
    row = panel[panel["YQ"] == yq]
    if row.empty: continue
    sil1 = row["siloing_l1"].values[0]
    dl   = row["dlog_deals"].values[0] if pd.notna(row["dlog_deals"].values[0]) else np.nan
    ep   = row["EnvP_Index"].values[0] if pd.notna(row["EnvP_Index"].values[0]) else np.nan
    epu  = row["EnvPU_Index"].values[0] if pd.notna(row["EnvPU_Index"].values[0]) else np.nan
    ep_s  = f"{ep:.0f}" if not np.isnan(ep) else "N/A"
    epu_s = f"{epu:.0f}" if not np.isnan(epu) else "N/A"
    print(f"  {direction:<10} {yq:<8} {sil1:>11.3f} {dl:>+12.3f} {ep_s:>8} {epu_s:>7}")

# Distribution of siloing on shock vs non-shock quarters
print(f"\n  Mean siloing_l1 on shock vs non-shock quarters:")
for sv, slabel in [("shock_policy_any","policy_any"),
                   ("shock_policy_neg","policy_neg"),
                   ("shock_policy_pos","policy_pos"),
                   ("shock_rate","rate_hike"),
                   ("shock_vix","vix_spike")]:
    tmp = panel.dropna(subset=["siloing_l1", sv])
    m1 = tmp[tmp[sv]==1]["siloing_l1"].mean()
    m0 = tmp[tmp[sv]==0]["siloing_l1"].mean()
    print(f"    {slabel:<20}: shock={m1:.3f}  non-shock={m0:.3f}  Δ={m1-m0:+.3f}")

# EnvP/EnvPU distribution by policy period
print(f"\n  EnvP index during key sub-periods (quarterly means):")
for label, mask in [("2014–2015 (early)", panel["date"].dt.year.isin([2014,2015])),
                    ("2016–2017 (Trump 1)", panel["date"].dt.year.isin([2016,2017])),
                    ("2018–2019 (late EnvP)", panel["date"].dt.year.isin([2018,2019]))]:
    sub = panel[mask & panel["EnvP_Index"].notna()]
    if len(sub) > 0:
        print(f"    {label}: mean EnvP={sub['EnvP_Index'].mean():.1f}  "
              f"mean EnvPU={sub['EnvPU_Index'].mean():.1f}  (N={len(sub)} quarters)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 7: SAVE OUTPUTS")
print("=" * 72)

# Augmented panel
panel.to_csv(OUT / "fragility_panel_with_policy_shocks.csv", index=False)
print(f"  Saved: output/policy_shock_fragility/fragility_panel_with_policy_shocks.csv  ({len(panel)} rows)")

# Build summary results table for markdown
res_df = pd.DataFrame(results)

# ── Collect M_comp coefficients ───────────────────────────────────────────────
b_comp_pol, se_comp_pol, _, p_comp_pol = nw_pval(mod_c, "int_policy", lags=4)
b_comp_vix, se_comp_vix, _, p_comp_vix = nw_pval(mod_c, "int_vix", lags=4)
b_comp_rat, se_comp_rat, _, p_comp_rat = nw_pval(mod_cr, "int_rate", lags=4)
b_comp_p2,  se_comp_p2,  _, p_comp_p2  = nw_pval(mod_cr, "int_policy", lags=4)

# Wald values
try:
    wald_f  = float(wald.statistic);  wald_p  = float(wald.pvalue)
    wald2_f = float(wald2.statistic); wald2_p = float(wald2.pvalue)
except:
    wald_f = wald_p = wald2_f = wald2_p = np.nan

md = f"""# Policy-Shock Fragility Extension (Script 56)
*(Extends Script 55 — sector-specific policy shocks instead of generic financial uncertainty)*

## Data Sources

- **Quarterly siloing panel**: `output/cross_community_decline/quarterly_panel.csv`
  47 quarters (2014Q1–2025Q4); siloing = 1 − pct_cross; range [0.21, 1.00]
- **Macro shocks panel**: `output/macro_shocks/quarterly_panel.csv`
  VIX, interest rate, GPR — same date range
- **Noailly et al. (2024) EnvP index** [downloaded from Harvard Dataverse doi:10.7910/DVN/TCX06Z]
  Monthly, Jan 1981–Mar 2019 → covers {n_envp} of {len(panel)} panel quarters
- **Noailly et al. (2022) EnvPU index** [downloaded from doi:10.7910/DVN/8TKRSJ]
  Monthly, Jan 1990–Mar 2019 → covers {n_envpu} of {len(panel)} panel quarters

**EnvP limitation:** Both indices end March 2019, so they cannot cover the three most consequential
post-2020 policy shocks (Biden 2021, IRA 2022, Trump 2.0 2025). M6/M7 are restricted to the
2014–2019 subsample.

---

## Policy Event Dummies

### Positive shocks (pro-climate)
| Quarter | Event | siloing_l1 | Δlog(deals) | EnvP |
|---|---|---|---|---|
| 2015Q4 | Paris Agreement adopted (Dec 2015) | {panel[panel.YQ=="2015Q4"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2015Q4"]["dlog_deals"].values[0]:+.3f} | {panel[panel.YQ=="2015Q4"]["EnvP_Index"].values[0]:.0f} |
| 2021Q1 | Biden inauguration + Paris re-entry | {panel[panel.YQ=="2021Q1"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2021Q1"]["dlog_deals"].values[0]:+.3f} | N/A |
| 2022Q3 | Inflation Reduction Act signed | {panel[panel.YQ=="2022Q3"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2022Q3"]["dlog_deals"].values[0]:+.3f} | N/A |

### Negative shocks (anti-climate / uncertainty)
| Quarter | Event | siloing_l1 | Δlog(deals) | EnvP |
|---|---|---|---|---|
| 2016Q4 | Trump election (Nov 2016) | {panel[panel.YQ=="2016Q4"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2016Q4"]["dlog_deals"].values[0]:+.3f} | {panel[panel.YQ=="2016Q4"]["EnvP_Index"].values[0]:.0f} |
| 2017Q2 | Trump Paris withdrawal announcement | {panel[panel.YQ=="2017Q2"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2017Q2"]["dlog_deals"].values[0]:+.3f} | {panel[panel.YQ=="2017Q2"]["EnvP_Index"].values[0]:.0f} |
| 2019Q4 | Formal US Paris withdrawal notification | {panel[panel.YQ=="2019Q4"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2019Q4"]["dlog_deals"].values[0]:+.3f} | {panel[panel.YQ=="2019Q4"]["EnvP_Index"].values[0]:.0f} |
| 2025Q1 | Trump 2.0 inauguration + IRA repeal | {panel[panel.YQ=="2025Q1"]["siloing_l1"].values[0]:.3f} | {panel[panel.YQ=="2025Q1"]["dlog_deals"].values[0]:+.3f} | N/A |

---

## Regression Results

**Model:** Δlog(deals)_t = α + β₁·siloing_{{t−1}} + β₂·shock_t + **β₃·(siloing_{{t−1}} × shock_t)** + trend + Q-dummies + ε
**SEs:** Newey-West HAC (lags=4) | **KEY COEFFICIENT:** β₃

| Model | N | N_shock | β(sil×shock) | SE | Sig | r(sil,shock) |
|---|---|---|---|---|---|---|
"""
for r in results:
    confound_flag = " ⚠" if abs(r.get("corr_sil_shock", 0)) > 0.3 else ""
    md += (f"| {r['label']} | {r['N']} | {r.get('N_shock',0)} | "
           f"{r.get('b_int', np.nan):+.4f} | {r.get('se_int', np.nan):.4f} | "
           f"{sig(r.get('p_int', 1.0))} | {r.get('corr_sil_shock', np.nan):+.3f}{confound_flag} |\n")

md += f"""
---

## Comparative Test: Policy vs Financial Interaction (M_comp)

**Single model** with both interactions jointly estimated, Wald test for H₀: β(sil×policy) = β(sil×vix)

| Term | β | SE | Sig |
|---|---|---|---|
| siloing × policy_any (M_comp) | {b_comp_pol:+.4f} | {se_comp_pol:.4f} | {sig(p_comp_pol)} |
| siloing × vix_spike (M_comp) | {b_comp_vix:+.4f} | {se_comp_vix:.4f} | {sig(p_comp_vix)} |

**Wald test (policy = vix):** F = {wald_f:.3f}  p = {wald_p:.4f}  {sig(wald_p)}

| Term | β | SE | Sig |
|---|---|---|---|
| siloing × policy_any (vs rate) | {b_comp_p2:+.4f} | {se_comp_p2:.4f} | {sig(p_comp_p2)} |
| siloing × rate_hike (vs policy) | {b_comp_rat:+.4f} | {se_comp_rat:.4f} | {sig(p_comp_rat)} |

**Wald test (policy = rate):** F = {wald2_f:.3f}  p = {wald2_p:.4f}  {sig(wald2_p)}

---

## Confound Check: Siloing × Shock Correlations

| Shock variable | r(siloing_l1, shock) | Comment |
|---|---|---|
"""
for sv, slabel in [("shock_policy_any","policy_any"),("shock_policy_neg","policy_neg"),
                   ("shock_policy_pos","policy_pos"),("shock_rate","rate_hike"),
                   ("shock_vix","vix_spike")]:
    tmp = panel.dropna(subset=["siloing_l1_z", sv])
    if len(tmp) < 5: continue
    r_c, p_c = stats.pearsonr(tmp["siloing_l1_z"], tmp[sv])
    flag = "⚠ trending confound" if abs(r_c) > 0.3 else "OK"
    md += f"| {slabel} | {r_c:+.3f} {sig(p_c)} | {flag} |\n"

md += f"""
**Script 55 finding:** r(siloing_l1, shock_rate) = {r_rate:+.3f} {sig(p_rate)}
→ Rate-hike shock is correlated with siloing (both trend post-2021), which threatened
the Script 55 *** finding. Policy shocks span 2015–2025 and thus have lower correlation
with the siloing trend — making them stronger identification.

---

## Interpretation

### Signal strength comparison

The rate-hike result from Script 55 (β = −0.303***) rested on 4 quarters all in 2022,
a period when siloing also happened to be high — making the *** suspicious of common-trend
confound. Policy shocks provide {int(panel['shock_policy_any'].sum())} shock-quarter observations
spread across 2015–2025, avoiding this concern.

### Policy-specific channel (echo-chamber mechanism)

The hypothesis is that policy signals require INTERPRETATION, and siloed networks cannot
disaggregate expert information across communities:
- Positive policy shocks: siloed networks respond less (echo chambers confirm their existing
  views; limited outward signals to mobilize capital from other communities)
- Negative policy shocks: cascading within communities but insufficient counter-signal from
  outside → amplified contraction

Key results:
- M1 (any policy, 7 events): β = {results[0]['b_int']:+.4f} {sig(results[0]['p_int'])}
- M2 (negative only, 4 events): β = {results[1]['b_int']:+.4f} {sig(results[1]['p_int'])}
- M3 (positive only, 3 events): β = {results[2]['b_int']:+.4f} {sig(results[2]['p_int'])}
- M6 (EnvP continuous, N={results[5]['N']}): β = {results[5]['b_int']:+.4f} {sig(results[5]['p_int'])}
- M7 (EnvPU continuous, N={results[6]['N']}): β = {results[6]['b_int']:+.4f} {sig(results[6]['p_int'])}

### Power constraints

With T=46 quarters and 7 policy events, detecting moderate interaction effects (β~0.05-0.10)
requires very high signal-to-noise. The NW HAC correction further reduces effective sample size.
M6/M7 are restricted to ≤{n_envp} quarters. These are the binding constraints.

## Output Files

- `output/policy_shock_fragility/fragility_panel_with_policy_shocks.csv` — augmented panel
- `output/policy_shock_fragility/policy_shocks_fragility_results.md` — this file
"""

with open(OUT / "policy_shocks_fragility_results.md", "w") as f:
    f.write(md)
print(f"  Saved: output/policy_shock_fragility/policy_shocks_fragility_results.md")
print(f"\n  ✅ Script 56 complete.")
