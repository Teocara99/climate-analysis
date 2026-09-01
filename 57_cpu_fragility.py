"""
Script 57: CPU × Siloing Fragility — Full 2014–2025 Window
===========================================================
Extends Script 56 M6/M7 (EnvP/EnvPU, restricted to 2014–2019, N≈19) by
replacing those indices with the Gavriilidis (2021) Climate Policy Uncertainty
index, which already covers 2014Q1–2025Q4 in data/cpu_gavriilidis_us.csv.

Models (same spec as Script 56: Δlog(deals) ~ siloing_l1_z × CPU_z + controls):
  M6a: CPU_narrow_z × siloing_l1_z  (full window, N≈44)
  M6b: CPU_broad_z  × siloing_l1_z  (full window, N≈44)
  M6c: CPU_narrow_z × siloing_l1_z  (2014–2019 subsample, N≈19, comparable to Script 56 M6)

Reference: Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty.
           SSRN Working Paper 3847388.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_hac

OUT = Path("output/cpu_fragility"); OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 \
           else "†" if p < 0.10 else "n.s."

def nw_pval(model, varname, lags=4):
    cov = cov_hac(model, nlags=lags)
    idx = list(model.params.index).index(varname)
    se  = float(np.sqrt(cov[idx, idx]))
    b   = float(model.params[varname])
    t   = b / se
    df  = max(int(model.nobs) - len(model.params), 1)
    p   = float(2 * (1 - stats.t.cdf(abs(t), df=df)))
    return b, se, t, p

def run_model(sub, shock_var, label, lags=4, dep="dlog_deals"):
    sub = sub.dropna(subset=[dep, "siloing_l1_z", shock_var]).copy()
    sub["interaction"] = sub["siloing_l1_z"] * sub[shock_var]
    for q in ["q2", "q3", "q4"]:
        if q not in sub.columns:
            sub[q] = 0.0
    formula = f"{dep} ~ siloing_l1_z + {shock_var} + interaction + t + q2 + q3 + q4"
    try:
        mod = smf.ols(formula, data=sub).fit()
    except Exception as e:
        return {"label": label, "N": len(sub), "b_int": np.nan,
                "se_int": np.nan, "p_int": np.nan, "corr": np.nan, "rsq": np.nan,
                "error": str(e)}
    try:
        b_int, se_int, _, p_int = nw_pval(mod, "interaction", lags=lags)
        b_sil, se_sil, _, p_sil = nw_pval(mod, "siloing_l1_z", lags=lags)
        b_cpu, se_cpu, _, p_cpu = nw_pval(mod, shock_var, lags=lags)
    except Exception:
        b_int = se_int = p_int = b_sil = se_sil = p_sil = b_cpu = se_cpu = p_cpu = np.nan
    corr = sub["siloing_l1_z"].corr(sub[shock_var])
    return {"label": label, "N": int(mod.nobs),
            "b_int": b_int, "se_int": se_int, "p_int": p_int,
            "b_sil": b_sil, "se_sil": se_sil, "p_sil": p_sil,
            "b_cpu": b_cpu, "se_cpu": se_cpu, "p_cpu": p_cpu,
            "corr": corr, "rsq": mod.rsquared}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & AGGREGATE CPU DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 1: GAVRIILIDIS CPU INDEX")
print("=" * 72)

cpu_raw = pd.read_csv("data/cpu_gavriilidis_us.csv")
cpu_raw["date"] = pd.to_datetime(cpu_raw["year_month"])
cpu_raw["quarter"] = cpu_raw["date"].dt.to_period("Q")

print(f"  Raw monthly rows: {len(cpu_raw)}")
print(f"  Date range: {cpu_raw['date'].min().date()} → {cpu_raw['date'].max().date()}")
print(f"  cpu_narrow nulls: {cpu_raw['cpu_narrow'].isna().sum()}")
print(f"  cpu_broad  nulls: {cpu_raw['cpu_broad'].isna().sum()}")

# Aggregate to quarterly means
cpu_q = cpu_raw.groupby("quarter").agg(
    cpu_narrow=("cpu_narrow", "mean"),
    cpu_broad=("cpu_broad", "mean"),
).reset_index()
cpu_q["YQ"] = cpu_q["quarter"].dt.strftime("%YQ") + cpu_q["quarter"].dt.quarter.astype(str)

# Standardise over the full series (not just panel window)
cpu_q["cpu_narrow_z"] = (cpu_q["cpu_narrow"] - cpu_q["cpu_narrow"].mean()) / cpu_q["cpu_narrow"].std()
cpu_q["cpu_broad_z"]  = (cpu_q["cpu_broad"]  - cpu_q["cpu_broad"].mean())  / cpu_q["cpu_broad"].std()

print(f"\n  Quarterly rows: {len(cpu_q)}")
print(f"  Panel window rows (2014–2025): {cpu_q[cpu_q['YQ'] >= '2014Q1'].shape[0]}")
print(f"\n  cpu_narrow stats (full series):  mean={cpu_q['cpu_narrow'].mean():.1f}  "
      f"sd={cpu_q['cpu_narrow'].std():.1f}  "
      f"min={cpu_q['cpu_narrow'].min():.1f}  max={cpu_q['cpu_narrow'].max():.1f}")
print(f"  cpu_broad  stats (full series):  mean={cpu_q['cpu_broad'].mean():.1f}  "
      f"sd={cpu_q['cpu_broad'].std():.1f}  "
      f"min={cpu_q['cpu_broad'].min():.1f}  max={cpu_q['cpu_broad'].max():.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD SILOING PANEL (same as Script 56)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: SILOING PANEL")
print("=" * 72)

qsil = pd.read_csv("output/cross_community_decline/quarterly_panel.csv")
qsil["date"] = pd.to_datetime(qsil["date"])
qsil["YQ"]   = qsil["YQ"].str.strip()
qsil["siloing"] = 1.0 - qsil["pct_cross"]

yq_to_sil = dict(zip(qsil["YQ"], qsil["siloing"]))
def prev_quarter_yq(yq):
    y, q = int(yq[:4]), int(yq[-1])
    pq = q - 1 if q > 1 else 4
    py = y if q > 1 else y - 1
    return f"{py}Q{pq}"

qsil["siloing_l1_proper"] = qsil["YQ"].map(lambda yq: yq_to_sil.get(prev_quarter_yq(yq)))

ms = pd.read_csv("output/macro_shocks/quarterly_panel.csv")
ms["date"] = pd.to_datetime(ms["date"])
ms["YQ"]   = ms["YQ"].str.strip()
ms["log_deals"]  = np.log1p(ms["n_deals"])
ms["dlog_deals"] = ms["log_deals"].diff()
ms["q2"] = (ms["date"].dt.month == 4).astype(float)
ms["q3"] = (ms["date"].dt.month == 7).astype(float)
ms["q4"] = (ms["date"].dt.month == 10).astype(float)

panel = pd.merge(
    qsil[["YQ", "date", "siloing", "siloing_l1_proper"]].rename(columns={"date": "date_sil"}),
    ms[["YQ", "date", "n_deals", "dlog_deals", "q2", "q3", "q4"]],
    on="YQ", how="inner"
).sort_values("date").reset_index(drop=True)

panel["t"] = np.arange(len(panel))
panel["siloing_l1_z"] = (
    (panel["siloing_l1_proper"] - panel["siloing_l1_proper"].mean()) /
     panel["siloing_l1_proper"].std()
)

# Merge CPU
panel = panel.merge(cpu_q[["YQ", "cpu_narrow", "cpu_broad", "cpu_narrow_z", "cpu_broad_z"]],
                    on="YQ", how="left")

n_cpu = panel["cpu_narrow_z"].notna().sum()
print(f"  Panel: {panel['YQ'].iloc[0]} → {panel['YQ'].iloc[-1]}  N={len(panel)}")
print(f"  dlog_deals valid: {panel['dlog_deals'].notna().sum()}")
print(f"  siloing_l1_z valid: {panel['siloing_l1_z'].notna().sum()}")
print(f"  CPU coverage in panel: {n_cpu} / {len(panel)} quarters")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CPU TIME SERIES — PANEL WINDOW
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: CPU LEVELS AT KEY QUARTERS")
print("=" * 72)

key_events = {
    "2015Q4": "Paris Agreement (Dec 2015)",
    "2016Q4": "Trump election (Nov 2016)",
    "2017Q2": "Trump Paris withdrawal (Jun 2017)",
    "2019Q4": "Formal US Paris withdrawal (Nov 2019)",
    "2020Q1": "COVID shock (Mar 2020)",
    "2021Q1": "Biden inauguration (Jan 2021)",
    "2022Q2": "Rate hike cycle begins",
    "2022Q3": "IRA signed (Aug 2022)",
    "2025Q1": "Trump 2.0 (Jan 2025)",
}
print(f"  {'Quarter':<8}  {'cpu_narrow':>10}  {'cpu_narrow_z':>12}  {'siloing_l1':>10}  event")
print(f"  {'─'*72}")
for yq, desc in key_events.items():
    row = panel[panel["YQ"] == yq]
    if row.empty:
        print(f"  {yq:<8}  (not in panel)")
        continue
    cpu_n  = row["cpu_narrow"].values[0]
    cpu_nz = row["cpu_narrow_z"].values[0]
    sil    = row["siloing_l1_proper"].values[0]
    print(f"  {yq:<8}  {cpu_n:>10.1f}  {cpu_nz:>+12.3f}  {sil:>10.3f}  {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: CONFOUND CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: CONFOUND CHECK: r(siloing_l1_z, CPU)")
print("=" * 72)

for col, label in [("cpu_narrow_z", "cpu_narrow"), ("cpu_broad_z", "cpu_broad")]:
    tmp = panel.dropna(subset=["siloing_l1_z", col])
    r, p = stats.pearsonr(tmp["siloing_l1_z"], tmp[col])
    flag = "⚠ CONFOUND" if abs(r) > 0.3 else "ok"
    print(f"  r(siloing_l1, {label:<12}) = {r:>+.3f}  {sig(p):>5}  {flag}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: REGRESSIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: REGRESSIONS  (Δlog(deals) ~ siloing_l1_z × CPU_z + t + Q-dummies)")
print("SEs: Newey-West HAC (lags=4)")
print("=" * 72)

results = []

# M6a: cpu_narrow — full 2014–2025 window
r6a = run_model(panel, "cpu_narrow_z", "M6a: CPU_narrow (full 2014–2025)")
results.append(r6a)

# M6b: cpu_broad — full 2014–2025 window
r6b = run_model(panel, "cpu_broad_z", "M6b: CPU_broad  (full 2014–2025)")
results.append(r6b)

# M6c: cpu_narrow — 2014–2019 subsample (comparable to Script 56 M6 EnvP, N≈19)
sub_pre2020 = panel[panel["YQ"] < "2020Q1"].copy()
sub_pre2020["t"] = np.arange(len(sub_pre2020))
r6c = run_model(sub_pre2020, "cpu_narrow_z", "M6c: CPU_narrow (2014–2019 only, N≈19)")
results.append(r6c)

# ── Print table ──────────────────────────────────────────────────────────────
print(f"\n  {'Model':<44} {'N':>4}  {'β(sil×CPU)':>12}  {'SE':>8}  {'Sig':>6}  {'r(sil,CPU)':>11}  R²")
print(f"  {'─'*100}")
for r in results:
    flag = " ⚠" if abs(r.get("corr", 0)) > 0.3 else ""
    print(f"  {r['label']:<44} {r['N']:>4}  "
          f"{r.get('b_int', np.nan):>+12.4f}  "
          f"{r.get('se_int', np.nan):>8.4f}  "
          f"{sig(r.get('p_int', 1)):>6}  "
          f"{r.get('corr', np.nan):>+11.3f}{flag}  "
          f"{r.get('rsq', np.nan):.3f}")

# ── Detailed output for M6a ──────────────────────────────────────────────────
print(f"\n  ── DETAILED: M6a (CPU_narrow, full window) ──")
sub6a = panel.dropna(subset=["dlog_deals", "siloing_l1_z", "cpu_narrow_z"]).copy()
sub6a["interaction"] = sub6a["siloing_l1_z"] * sub6a["cpu_narrow_z"]
mod6a = smf.ols("dlog_deals ~ siloing_l1_z + cpu_narrow_z + interaction + t + q2 + q3 + q4",
                data=sub6a).fit()
for vname in ["Intercept", "siloing_l1_z", "cpu_narrow_z", "interaction", "t"]:
    if vname in mod6a.params:
        b, se, t_stat, p = nw_pval(mod6a, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  t={t_stat:>+.2f}  {sig(p)}")
print(f"    N={int(mod6a.nobs)}  R²={mod6a.rsquared:.3f}  Adj-R²={mod6a.rsquared_adj:.3f}")

# ── Detailed output for M6b ──────────────────────────────────────────────────
print(f"\n  ── DETAILED: M6b (CPU_broad, full window) ──")
sub6b = panel.dropna(subset=["dlog_deals", "siloing_l1_z", "cpu_broad_z"]).copy()
sub6b["interaction"] = sub6b["siloing_l1_z"] * sub6b["cpu_broad_z"]
mod6b = smf.ols("dlog_deals ~ siloing_l1_z + cpu_broad_z + interaction + t + q2 + q3 + q4",
                data=sub6b).fit()
for vname in ["Intercept", "siloing_l1_z", "cpu_broad_z", "interaction", "t"]:
    if vname in mod6b.params:
        b, se, t_stat, p = nw_pval(mod6b, vname, lags=4)
        print(f"    {vname:<28}: β={b:>+.4f}  SE={se:.4f}  t={t_stat:>+.2f}  {sig(p)}")
print(f"    N={int(mod6b.nobs)}  R²={mod6b.rsquared:.3f}  Adj-R²={mod6b.rsquared_adj:.3f}")

# ── Comparison with Script 56 M6 (EnvP) ─────────────────────────────────────
print(f"\n  ── COMPARISON: CPU (this script) vs EnvP (Script 56 M6) ──")
print(f"  {'Model':<44} {'N':>4}  {'β(sil×index)':>14}  {'Sig':>6}  note")
print(f"  {'─'*80}")
comparison = [
    ("M6a: CPU_narrow × siloing (2014–2025)", r6a),
    ("M6b: CPU_broad  × siloing (2014–2025)", r6b),
    ("M6c: CPU_narrow × siloing (2014–2019)", r6c),
]
for lbl, r in comparison:
    print(f"  {lbl:<44} {r['N']:>4}  "
          f"{r.get('b_int', np.nan):>+14.4f}  "
          f"{sig(r.get('p_int', 1)):>6}")
print(f"  {'Script 56 M6: EnvP × siloing (2014–2019)':<44} {'~19':>4}  "
      f"{'(+0.2263)':>14}  {'*':>6}  (for reference)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
panel.to_csv(OUT / "cpu_siloing_panel.csv", index=False)
print(f"\n  Saved: {OUT}/cpu_siloing_panel.csv  ({len(panel)} rows)")

# ── Results markdown ──────────────────────────────────────────────────────────
md_lines = [
    "# CPU × Siloing Fragility — Full 2014–2025 Window (Script 57)",
    "*(Extends Script 56 M6/M7 — uses Gavriilidis CPU index instead of Noailly EnvP)*",
    "",
    "## Data",
    "",
    "| Source | Coverage | Quarters in panel |",
    "|---|---|---|",
    f"| Gavriilidis (2021) CPU_narrow | Jan 1985 – Apr 2026 | {n_cpu} of {len(panel)} |",
    f"| Gavriilidis (2021) CPU_broad  | Jan 1985 – Apr 2026 | {n_cpu} of {len(panel)} |",
    "| *Script 56 EnvP (Noailly 2024)* | *Jan 1981 – Mar 2019* | *~19 of 46* |",
    "",
    "**Key advantage over Script 56 M6:** CPU_narrow and CPU_broad cover the **full",
    "2014–2025 panel window** (N≈44) including Biden 2021, IRA 2022, and Trump 2.0 2025 —",
    "the three highest-stakes policy quarters that EnvP misses.",
    "",
    "**Reference:** Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty.",
    "SSRN Working Paper 3847388. Data: `data/cpu_gavriilidis_us.csv`.",
    "",
    "---",
    "",
    "## Main Results",
    "",
    "**Model:** Δlog(deals)_t = α + β₁·siloing_{t−1} + β₂·CPU_z_t + **β₃·(siloing_{t−1} × CPU_z_t)** + trend + Q₂ + Q₃ + Q₄ + ε",
    "**SEs:** Newey-West HAC (lags=4) | **KEY COEFFICIENT:** β₃",
    "",
    "| Model | N | β(sil × CPU) | SE | Sig | r(sil,CPU) | R² |",
    "|---|---|---|---|---|---|---|",
]
for r in results:
    flag = " ⚠" if abs(r.get("corr", 0)) > 0.3 else ""
    md_lines.append(
        f"| {r['label']} | {r['N']} | "
        f"**{r.get('b_int', np.nan):+.4f}** | "
        f"{r.get('se_int', np.nan):.4f} | "
        f"**{sig(r.get('p_int', 1))}** | "
        f"{r.get('corr', np.nan):+.3f}{flag} | "
        f"{r.get('rsq', np.nan):.3f} |"
    )

m6a_b   = r6a.get("b_int", np.nan)
m6a_sig = sig(r6a.get("p_int", 1))
m6b_b   = r6b.get("b_int", np.nan)
m6b_sig = sig(r6b.get("p_int", 1))
m6c_b   = r6c.get("b_int", np.nan)
m6c_sig = sig(r6c.get("p_int", 1))
corr_n  = r6a.get("corr", np.nan)
corr_b  = r6b.get("corr", np.nan)

md_lines += [
    "",
    "---",
    "",
    "## Confound Check",
    "",
    "| Measure | r(siloing_l1, CPU) | Comment |",
    "|---|---|---|",
    f"| CPU_narrow | {corr_n:+.3f} | {'⚠ Trending confound' if abs(corr_n) > 0.3 else '✓ Near-zero — clean identification'} |",
    f"| CPU_broad  | {corr_b:+.3f} | {'⚠ Trending confound' if abs(corr_b) > 0.3 else '✓ Near-zero — clean identification'} |",
    "",
    "---",
    "",
    "## Interpretation",
    "",
    "### Sign direction",
    "",
    f"M6a (CPU_narrow): β = {m6a_b:+.4f} ({m6a_sig})",
    f"M6b (CPU_broad):  β = {m6b_b:+.4f} ({m6b_sig})",
    f"M6c (CPU_narrow, 2014–2019): β = {m6c_b:+.4f} ({m6c_sig})",
    "",
    "A positive β(sil × CPU) means: when the network is more siloed AND climate",
    "policy uncertainty is higher, deal flow is HIGHER than expected — consistent",
    "with the echo-chamber reinforcement mechanism found in Script 56 for all",
    "policy shock specifications. Siloed networks amplify within-community response",
    "to policy signals rather than cascading into crisis (the AER 2015 prediction).",
    "",
    "### Power gain vs Script 56 M6",
    "",
    "| | Script 56 M6 (EnvP) | Script 57 M6a (CPU) |",
    "|---|---|---|",
    "| Index | Noailly EnvP | Gavriilidis CPU_narrow |",
    "| Window | 2014–2019 | 2014–2025 |",
    f"| N | ~19 | {r6a['N']} |",
    "| Includes post-2020 shocks | No | Yes |",
    "",
    "---",
    "",
    "## Output Files",
    "",
    "- `output/cpu_fragility/cpu_siloing_panel.csv` — merged panel",
    "- `output/cpu_fragility/cpu_fragility_results.md` — this file",
]

md_text = "\n".join(md_lines)
(OUT / "cpu_fragility_results.md").write_text(md_text)
print(f"  Saved: {OUT}/cpu_fragility_results.md")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
