"""
Script 84: H4 — IRA convergence.

PRE-SPECIFIED (report regardless of outcome):
  H4: post-IRA US IRA_tech startups are LESS novel (lower overall_sparsity_t)
  relative to all three comparison cells.
  Predicted sign on US × IRA_tech × post_IRA: NEGATIVE.

Secondary DV: niche_cohesion_t.

VALIDITY FIRST: event study (US × IRA_tech by year, ref = 2021).
  Pre-2022 coefficients must be flat. If they trend, report plainly: design fails.

MODEL: full triple DiD
  overall_sparsity_t ~ US + IRA_tech + US×IRA_tech + US×post_IRA
                     + IRA_tech×post_IRA + US×IRA_tech×post_IRA [H4]
                     + log_capital + year-quarter FE + tech-cluster FE + country FE
  SEs clustered by tech cluster (specter_cluster).

ROBUSTNESS:
  R1: post_IRA = founding year >= 2023
  R2: alt IRA date 2022-07-27 (Senate passage)
  R3: drop top 1% overall_sparsity_t
  R4-a: cluster 4 fully included (micromobility as IRA_tech)
  R4-b: cluster 4 fully excluded

Do NOT run H5–H10.
"""

import os, sys, re, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUT_DIR  = os.path.join(os.path.dirname(__file__), "output", "h4_ira")
FIG_DIR  = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

IRA_DATE  = pd.Timestamp("2022-08-16")
ALT_DATE  = pd.Timestamp("2022-07-27")

# ── IRA cluster definitions (locked in preregistration) ──────────────────
IRA_CLUSTERS_FULL = [0, 2, 3, 4, 5, 13, 17, 40, 42, 54, 55, 56, 57]  # cluster 4 full
MFG_CLUSTERS      = [55, 42, 2, 13]   # 45X + 45V + 45Q (manufacturing-intensive)

MICRO_KEYWORDS = re.compile(
    r"e-bike|ebike|\bbike\b|scooter|micromobility|micro-mobility|bicycle|cargo bike",
    re.IGNORECASE,
)

print("=" * 70)
print("SCRIPT 84 — H4: IRA CONVERGENCE")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading data ──")

nov_t = pd.read_csv("output/novelty_t/novelty_time_aware.csv",
                    parse_dates=["first_date"])

ns = pd.read_csv("output/novelty/novelty_scores.csv")[
    ["name", "specter_cluster", "Description"]
].rename(columns={"name": "Companies"})

cpu = pd.read_csv("output/cpu/cpu_company_dataset.csv")[
    ["Companies", "is_us", "hq_country", "log_capital"]
]

# Merge
df = nov_t.merge(ns, on="Companies", how="left")
df = df.merge(cpu, on="Companies", how="left")

# Country: fill is_us from hq_country if missing
df["is_US"] = df["is_us"].fillna(
    (df["hq_country"] == "United States").astype(float)
).fillna(0).astype(int)

# ── Micromobility flag (cluster 4 exclusion) ─────────────────────────────
df["is_micro"] = (
    (df["specter_cluster"] == 4) &
    df["Description"].fillna("").apply(lambda x: bool(MICRO_KEYWORDS.search(x)))
).astype(int)
n_micro = df["is_micro"].sum()
print(f"  Micromobility startups (cluster 4, will be non-IRA): {n_micro}")

# IRA_tech: base = full clusters minus micromobility
df["IRA_tech"] = (
    df["specter_cluster"].isin(IRA_CLUSTERS_FULL) & (df["is_micro"] == 0)
).astype(int)

df["mfg_credit"] = df["specter_cluster"].isin(MFG_CLUSTERS).astype(int)

# Post-IRA flags
df["post_IRA"]  = (df["first_date"] >= IRA_DATE).astype(int)
df["post_IRA_alt"] = (df["first_date"] >= ALT_DATE).astype(int)
df["post_2023"] = (df["first_year"] >= 2023).astype(int)

# Year-quarter FE
df["yq"] = (df["first_date"].dt.year.astype(str) + "Q" +
            df["first_date"].dt.quarter.astype(str))

# Country FE (truncate small countries)
top_countries = df["hq_country"].value_counts().head(30).index.tolist()
df["hq_ctry_fe"] = df["hq_country"].where(
    df["hq_country"].isin(top_countries), other="Other"
)

# Cluster FE: use string for formula compatibility
df["cluster_fe"] = df["specter_cluster"].fillna(-999).astype(int).astype(str)

# Standardise log_capital (don't have it for all rows; fill median)
df["log_capital"] = df["log_capital"].fillna(df["log_capital"].median())

# Drop rows missing the DV
df = df.dropna(subset=["overall_sparsity_t", "niche_cohesion_t"])

print(f"  Working sample: N={len(df):,}")
print(f"  IRA_tech: {df['IRA_tech'].sum():,}  |  US: {df['is_US'].sum():,}")
print(f"  US × IRA_tech × post-IRA: "
      f"{(df['is_US'] & df['IRA_tech'].astype(bool) & df['post_IRA'].astype(bool)).sum()}")

# Interaction terms
df["US_x_IRA"]     = df["is_US"]  * df["IRA_tech"]
df["US_x_post"]    = df["is_US"]  * df["post_IRA"]
df["IRA_x_post"]   = df["IRA_tech"] * df["post_IRA"]
df["TRIPLE"]       = df["is_US"]  * df["IRA_tech"] * df["post_IRA"]

# ═══════════════════════════════════════════════════════════════════════════
# VALIDITY: EVENT STUDY  (2017-2025, reference 2021)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── VALIDITY: Event study (US × IRA_tech by year, ref=2021) ──")

ES_YEARS = list(range(2017, 2026))
REF_YEAR = 2021

es_df = df[df["first_year"].isin(ES_YEARS)].copy()

# Create year dummies interacted with US × IRA_tech
for yr in ES_YEARS:
    if yr == REF_YEAR:
        continue
    es_df[f"yr{yr}_x_US_x_IRA"] = (
        (es_df["first_year"] == yr) & (es_df["is_US"] == 1) &
        (es_df["IRA_tech"] == 1)
    ).astype(float)

# Also need all year dummies (absorbed by yq FE except the interactions)
# Build formula
yr_terms = [f"yr{yr}_x_US_x_IRA" for yr in ES_YEARS if yr != REF_YEAR]

# Separate out the IRA_tech main effect + US × IRA_tech interaction
# so the year × US × IRA interaction is identified relative to 2021 × US × IRA
es_formula = (
    "overall_sparsity_t ~ is_US + IRA_tech + US_x_IRA + US_x_post + IRA_x_post + "
    + " + ".join(yr_terms) +
    " + log_capital + C(yq) + C(cluster_fe) + C(hq_ctry_fe) - 1"
)

grps = es_df["cluster_fe"].values
try:
    es_fit = smf.ols(es_formula, data=es_df).fit(
        cov_type="cluster", cov_kwds={"groups": grps}
    )
    es_coefs = {}
    for yr in ES_YEARS:
        if yr == REF_YEAR:
            es_coefs[yr] = (0.0, 0.0)
        else:
            key = f"yr{yr}_x_US_x_IRA"
            if key in es_fit.params:
                es_coefs[yr] = (es_fit.params[key], es_fit.bse[key])
            else:
                es_coefs[yr] = (np.nan, np.nan)

    print(f"\n  Year   β(US×IRA_tech)    SE       95% CI")
    print("  " + "─" * 50)
    pre_betas = []
    post_betas = []
    for yr in ES_YEARS:
        b, se = es_coefs[yr]
        if np.isnan(b):
            print(f"  {yr}   {'—':>12}   {'—':>8}")
            continue
        lo, hi = b - 1.96 * se, b + 1.96 * se
        marker = " ← POST-IRA" if yr >= 2022 else ("  [ref]" if yr == REF_YEAR else "")
        print(f"  {yr}   {b:+10.4f}    {se:8.4f}    [{lo:+.3f}, {hi:+.3f}]{marker}")
        if yr < REF_YEAR:
            pre_betas.append(b)
        elif yr >= 2022:
            post_betas.append(b)

    # Pre-trend test
    pre_nonzero = [b for b in pre_betas if not np.isnan(b)]
    if pre_nonzero:
        avg_pre = np.mean(np.abs(pre_nonzero))
        avg_post = np.mean(np.abs(post_betas)) if post_betas else 0.0
        print(f"\n  Avg |β| pre-2021: {avg_pre:.4f}   Avg |β| 2022+: {avg_post:.4f}")
        if avg_pre > 0.05 and avg_pre > 0.5 * avg_post:
            print("  *** PRE-TREND WARNING: pre-2022 coefficients are not flat.")
            print("  *** Parallel-trends assumption is suspect.")
        else:
            print("  Pre-trend: coefficients are roughly flat. Parallel-trends plausible.")

    # Plot event study
    yrs_plot = [yr for yr in ES_YEARS]
    betas = [es_coefs[yr][0] for yr in yrs_plot]
    ses   = [es_coefs[yr][1] for yr in yrs_plot]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(2022 - 0.5, color="#cc3333", lw=1.2, ls=":", label="IRA signed (2022-08-16)")
    for yr, b, se in zip(yrs_plot, betas, ses):
        if np.isnan(b):
            continue
        color = "#2166ac" if yr < 2022 else "#d6604d"
        ax.errorbar(yr, b, yerr=1.96 * se, fmt="o", color=color,
                    capsize=4, capthick=1.2, ms=6)
    ax.plot([yr for yr, b in zip(yrs_plot, betas) if not np.isnan(b)],
            [b for b in betas if not np.isnan(b)],
            "o-", color="#555555", lw=1, ms=0, zorder=0)
    ax.set_xlabel("Year of first funding round")
    ax.set_ylabel("β (US × IRA_tech year effect) relative to 2021")
    ax.set_title("Event study: US × IRA_tech novelty effect by year\n"
                 "(overall_sparsity_t; ref = 2021; 95% CIs; SEs clustered by tech cluster)")
    ax.legend()
    ax.set_xticks(ES_YEARS)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "event_study.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Event study figure → {fig_path}")

except Exception as e:
    print(f"  Event study failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN MODEL: TRIPLE DiD
# ═══════════════════════════════════════════════════════════════════════════
print("\n── MAIN MODEL: Triple DiD ──")

BASE_FORMULA = (
    "{dv} ~ is_US + IRA_tech + US_x_IRA + US_x_post + IRA_x_post + TRIPLE"
    " + log_capital + C(yq) + C(cluster_fe) + C(hq_ctry_fe) - 1"
)

def run_model(data, dv, label, grp_col="cluster_fe"):
    formula = BASE_FORMULA.format(dv=dv)
    grps = data[grp_col].values
    fit = smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": grps}
    )
    b   = fit.params.get("TRIPLE", np.nan)
    se  = fit.bse.get("TRIPLE", np.nan)
    pv  = fit.pvalues.get("TRIPLE", np.nan)
    n   = int(fit.nobs)
    r2  = fit.rsquared
    return {"label": label, "dv": dv, "N": n, "β": b, "SE": se,
            "p": pv, "R2": r2, "fit": fit}

results = []

# Primary: overall_sparsity_t
print("\n  [A] Primary DV: overall_sparsity_t")
m_main = run_model(df, "overall_sparsity_t", "Main (base spec)")
results.append(m_main)

# Secondary: niche_cohesion_t
print("  [B] Secondary DV: niche_cohesion_t")
m_nc = run_model(df, "niche_cohesion_t", "Main — NC_t")
results.append(m_nc)

def fmt_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""

print()
print(f"  {'Label':<30}  {'DV':<25}  N      β(TRIPLE)   SE      p       ")
print("  " + "─" * 80)
for r in results:
    st = fmt_stars(r["p"])
    print(f"  {r['label']:<30}  {r['dv']:<25}  {r['N']:>5}  "
          f"{r['β']:+8.4f}{st:<3}  {r['SE']:6.4f}  {r['p']:.4f}  R²={r['R2']:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── ROBUSTNESS ──")

rob = []

# R1: treatment = post-2023
df_r1 = df.copy()
df_r1["post_IRA"]  = df_r1["post_2023"]
df_r1["US_x_post"] = df_r1["is_US"]    * df_r1["post_IRA"]
df_r1["IRA_x_post"]= df_r1["IRA_tech"] * df_r1["post_IRA"]
df_r1["TRIPLE"]    = df_r1["is_US"]    * df_r1["IRA_tech"] * df_r1["post_IRA"]
rob.append(run_model(df_r1, "overall_sparsity_t", "R1: founding yr ≥ 2023"))

# R2: alternative IRA date 2022-07-27
df_r2 = df.copy()
df_r2["post_IRA"]  = df_r2["post_IRA_alt"]
df_r2["US_x_post"] = df_r2["is_US"]    * df_r2["post_IRA"]
df_r2["IRA_x_post"]= df_r2["IRA_tech"] * df_r2["post_IRA"]
df_r2["TRIPLE"]    = df_r2["is_US"]    * df_r2["IRA_tech"] * df_r2["post_IRA"]
rob.append(run_model(df_r2, "overall_sparsity_t", "R2: IRA date 2022-07-27"))

# R3: drop top 1% overall_sparsity_t
p99 = df["overall_sparsity_t"].quantile(0.99)
df_r3 = df[df["overall_sparsity_t"] < p99].copy()
rob.append(run_model(df_r3, "overall_sparsity_t", f"R3: drop top 1% (>{p99:.2f})"))

# R4-a: cluster 4 fully included
df_r4a = df.copy()
df_r4a["IRA_tech"]  = df_r4a["specter_cluster"].isin(IRA_CLUSTERS_FULL).astype(int)
df_r4a["US_x_IRA"]  = df_r4a["is_US"]    * df_r4a["IRA_tech"]
df_r4a["IRA_x_post"]= df_r4a["IRA_tech"] * df_r4a["post_IRA"]
df_r4a["TRIPLE"]    = df_r4a["is_US"]    * df_r4a["IRA_tech"] * df_r4a["post_IRA"]
rob.append(run_model(df_r4a, "overall_sparsity_t", "R4-a: cluster 4 fully in"))

# R4-b: cluster 4 fully excluded
ira_no4 = [c for c in IRA_CLUSTERS_FULL if c != 4]
df_r4b = df.copy()
df_r4b["IRA_tech"]  = df_r4b["specter_cluster"].isin(ira_no4).astype(int)
df_r4b["US_x_IRA"]  = df_r4b["is_US"]    * df_r4b["IRA_tech"]
df_r4b["IRA_x_post"]= df_r4b["IRA_tech"] * df_r4b["post_IRA"]
df_r4b["TRIPLE"]    = df_r4b["is_US"]    * df_r4b["IRA_tech"] * df_r4b["post_IRA"]
rob.append(run_model(df_r4b, "overall_sparsity_t", "R4-b: cluster 4 fully out"))

print()
print(f"  {'Label':<35}  {'N':>5}  β(TRIPLE)   SE      p")
print("  " + "─" * 68)
for r in rob:
    st = fmt_stars(r["p"])
    print(f"  {r['label']:<35}  {r['N']:>5}  {r['β']:+8.4f}{st:<3}  {r['SE']:6.4f}  {r['p']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# MDE
# ═══════════════════════════════════════════════════════════════════════════
print("\n── MDE (80% power, α=0.05) ──")
se_main = m_main["SE"]
mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se_main
print(f"  SE of triple interaction (main): {se_main:.4f}")
print(f"  MDE at 80% power: {mde:.4f}  "
      f"[{mde / df['overall_sparsity_t'].std():.3f} SD of OS_t]")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE COEFFICIENT TABLE
# ═══════════════════════════════════════════════════════════════════════════
all_rows = []
for r in results + rob:
    all_rows.append({
        "spec": r["label"],
        "dv": r["dv"],
        "N": r["N"],
        "beta_TRIPLE": r["β"],
        "SE": r["SE"],
        "p": r["p"],
        "stars": fmt_stars(r["p"]),
        "R2": r.get("R2", np.nan),
    })
coef_df = pd.DataFrame(all_rows)
coef_path = os.path.join(OUT_DIR, "h4_coefficients.csv")
coef_df.to_csv(coef_path, index=False)
print(f"\n  Coefficient table → {coef_path}")

# ── Full main-model summary for reference ─────────────────────────────────
print("\n── Key coefficients from main specification (overall_sparsity_t) ──")
fit = m_main["fit"]
key_vars = ["is_US", "IRA_tech", "US_x_IRA", "US_x_post", "IRA_x_post", "TRIPLE", "log_capital"]
print()
print(f"  {'Variable':<20}  {'β':>9}  {'SE':>7}  {'p':>7}  {'':>4}")
print("  " + "─" * 52)
for v in key_vars:
    if v in fit.params:
        b  = fit.params[v]
        se = fit.bse[v]
        p  = fit.pvalues[v]
        print(f"  {v:<20}  {b:+9.4f}  {se:7.4f}  {p:7.4f}  {fmt_stars(p)}")

# ═══════════════════════════════════════════════════════════════════════════
# PLAIN VERDICT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PLAIN VERDICT — H4: IRA CONVERGENCE")
print("=" * 70)

b   = m_main["β"]
se  = m_main["SE"]
p   = m_main["p"]
mde_val = mde
dv_sd = df["overall_sparsity_t"].std()

sign_ok  = b < 0
sig_ok   = p < 0.05
st       = fmt_stars(p)

if sig_ok and sign_ok:
    verdict = "SUPPORTED"
elif sign_ok and p < 0.10:
    verdict = "DIRECTIONALLY SUPPORTED (marginal)"
elif sign_ok and not sig_ok:
    verdict = "DIRECTIONALLY CORRECT BUT NOT SIGNIFICANT"
elif not sign_ok and sig_ok:
    verdict = "WRONG SIGN — SIGNIFICANT"
else:
    verdict = "NULL — WRONG SIGN"

# Count robust replications
rob_sign_ok = sum(1 for r in rob if r["β"] < 0)
rob_sig_ok  = sum(1 for r in rob if r["β"] < 0 and r["p"] < 0.05)

print(f"\n  Pre-specified sign: NEGATIVE")
print(f"  Main estimate:  β={b:+.4f}{st}  SE={se:.4f}  p={p:.4f}  "
      f"[{b/dv_sd:.3f} SD]")
print(f"  MDE (80% pwr):  {mde_val:.4f}  [{mde_val/dv_sd:.3f} SD]")
print(f"  Event study pre-trend: see figure output/h4_ira/event_study.png")
print()
print(f"  Robustness: {rob_sign_ok}/5 specs have correct (negative) sign; "
      f"{rob_sig_ok}/5 are significant.")
print()
print(f"  ** H4 VERDICT: {verdict} **")

if not sig_ok and mde_val / dv_sd > 0.05:
    print(f"\n  Note: MDE = {mde_val/dv_sd:.3f} SD is large relative to plausible effect sizes.")
    print("  The study is underpowered to detect small convergence effects in this sample.")

print("\n" + "=" * 70)
print("SCRIPT 84 COMPLETE")
print("=" * 70)
