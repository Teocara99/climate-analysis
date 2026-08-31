"""
Script 42: Bridge Community Investor Type × Outcomes
=====================================================

Tests whether the TYPE of investor in the bridge community predicts
which specific outcomes improve from cross-community investment.

6 analyses:
  MODEL 1 — Bridge type → failure
  MODEL 2 — Bridge type → exit
  MODEL 3 — Bridge type → follow-on rounds
  MODEL 4 — Bridge type → employees (growth)
  MODEL 5 — Bridge type → patents (innovation)
  MODEL 6 — Descriptive table by bridge type
  + Green vs Traditional bridge comparison
"""

import warnings; warnings.filterwarnings("ignore")
import sys, itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, ".")
from load_data import load_deals, load_companies

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data…")
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["is_exited"]  = cdf["exited"].astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["got_followon"] = (cdf["n_rounds"].fillna(0) > 1).astype(float)
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"
CTRL_NOROUNDS = "log_cap + specter_novelty + C(country_fe) + C(deal_year)"

net = pd.read_csv("output/network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm  = net.set_index("investor")["community"].to_dict()

# Community-level dominant type and green composition
comm_type = (net.groupby("community")["investor_type"]
             .apply(lambda x: x.value_counts().index[0])
             .to_dict())
comm_green_pct = (net.groupby("community")["green_focus"]
                  .apply(lambda x: (x == "GREEN_VC").mean())
                  .to_dict())
comm_size = net.groupby("community").size().to_dict()

deals = load_deals()
deals["co"]   = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")

companies = load_companies()
companies["co"] = companies["Companies"].str.strip()

# Outcome variables from companies file
co_employees = (companies.dropna(subset=["Employees"])
                .set_index("co")["Employees"].to_dict())
co_patents   = (companies.dropna(subset=["Total Patent Documents"])
                .set_index("co")["Total Patent Documents"].to_dict())
cdf["log_employees"] = np.log1p(cdf["co"].map(co_employees).fillna(np.nan))
cdf["has_patent"]    = (cdf["co"].map(co_patents).fillna(0) > 0).astype(float)

# ── Build company → community timing ────────────────────────────────────────
print("Building community timing structure…")
d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))
d_exp["community"] = d_exp["inv"].map(inv_to_comm)
d_exp_comm = d_exp.dropna(subset=["community", "Year"])

co_timing = {}
for co, grp in d_exp_comm.groupby("co"):
    co_timing[co] = grp.groupby("Year")["community"].apply(set).to_dict()

# For each cross-community company: home comm and bridge comm
cross_cos = set(cdf[cdf["cross_community"] == 1]["co"])
timing_records = []
for co in cross_cos:
    yt = co_timing.get(co, {})
    if not yt:
        continue
    sorted_years = sorted(yt.keys())
    comm_first = {}
    for yr in sorted_years:
        for c in yt[yr]:
            if c not in comm_first:
                comm_first[c] = yr
    if len(comm_first) < 2:
        continue
    sorted_entries = sorted(comm_first.items(), key=lambda x: x[1])
    home_comm   = sorted_entries[0][0]
    bridge_comm = sorted_entries[1][0]   # first new community to enter
    timing_records.append({
        "co":           co,
        "home_comm":    home_comm,
        "bridge_comm":  bridge_comm,
        "bridge_type":  comm_type.get(bridge_comm, "OTHER"),
        "bridge_green_pct": comm_green_pct.get(bridge_comm, np.nan),
        "home_green_pct":   comm_green_pct.get(home_comm, np.nan),
        "bridge_size":  comm_size.get(bridge_comm, 0),
        "home_size":    comm_size.get(home_comm, 0),
    })

timing_df = pd.DataFrame(timing_records)
print(f"  Cross-community companies with bridge identified: {len(timing_df):,}")
print(f"  Bridge type distribution:")
print(timing_df["bridge_type"].value_counts().to_string())

# Create bridge type dummies
for t in ["IVC", "GVC", "CVC", "IMPACT", "OTHER"]:
    timing_df[f"bridge_{t.lower()}"] = (timing_df["bridge_type"] == t).astype(float)

# Green/trad bridge classification
timing_df["bridge_is_green"] = (timing_df["bridge_green_pct"] >= 0.5).astype(float)
timing_df["bridge_is_trad"]  = (timing_df["bridge_green_pct"] <  0.5).astype(float)

# Merge into cdf
df = cdf.merge(timing_df[["co","bridge_type","bridge_ivc","bridge_gvc","bridge_cvc",
                           "bridge_impact","bridge_other",
                           "bridge_is_green","bridge_is_trad",
                           "bridge_green_pct","home_green_pct"]],
               on="co", how="left")

# Within-community companies: all bridge dummies = 0
for col in ["bridge_ivc","bridge_gvc","bridge_cvc","bridge_impact",
            "bridge_other","bridge_is_green","bridge_is_trad"]:
    df[col] = df[col].fillna(0)

df_cross = df[df["cross_community"] == 1].copy()
df_model = df.dropna(subset=["log_cap","deal_year"]).copy()

# ── MODEL 6: Descriptive table ───────────────────────────────────────────────
print()
print("="*72)
print("MODEL 6: DESCRIPTIVE TABLE BY BRIDGE TYPE")
print("="*72)

wc = cdf[cdf["cross_community"]==0]
wc_stats = {
    "type": "Within-community (baseline)",
    "n": len(wc),
    "fail": wc["is_failed"].mean(),
    "exit": wc["is_exited"].mean(),
    "followon": wc["got_followon"].mean(),
    "log_cap": wc["log_cap"].mean(),
    "emp": np.expm1(wc["log_employees"].dropna()).mean(),
    "patent": wc["has_patent"].mean(),
}

rows = [wc_stats]
for bt in ["IVC","GVC","CVC","IMPACT","OTHER"]:
    sub = df_cross[df_cross["bridge_type"] == bt]
    if len(sub) < 5:
        continue
    rows.append({
        "type": f"Bridge = {bt}",
        "n": len(sub),
        "fail": sub["is_failed"].mean(),
        "exit": sub["is_exited"].mean(),
        "followon": sub["got_followon"].mean(),
        "log_cap": sub["log_cap"].mean(),
        "emp": np.expm1(sub["log_employees"].dropna()).mean(),
        "patent": sub["has_patent"].mean(),
    })

desc = pd.DataFrame(rows)
print(f"\n  {'Type':<35}  {'N':>5}  {'Fail%':>7}  {'Exit%':>7}  {'Followon%':>9}  {'Employees':>10}  {'Patent%':>8}")
print(f"  {'-'*82}")
for _, r in desc.iterrows():
    print(f"  {r['type']:<35}  {int(r['n']):>5}  {r['fail']:>7.1%}  {r['exit']:>7.1%}  "
          f"{r['followon']:>9.1%}  {r['emp']:>10.0f}  {r['patent']:>8.1%}")

# ── MODELS 1–5: Regressions ──────────────────────────────────────────────────
outcomes = [
    ("is_failed",      "failure",    CTRL,          "MODEL 1: FAILURE"),
    ("is_exited",      "exit",       CTRL,          "MODEL 2: EXIT"),
    ("got_followon",   "follow-on",  CTRL_NOROUNDS, "MODEL 3: FOLLOW-ON"),
    ("log_employees",  "employees",  CTRL,          "MODEL 4: GROWTH (log employees)"),
    ("has_patent",     "patents",    CTRL,          "MODEL 5: PATENTS"),
]

bridge_vars = "bridge_ivc + bridge_gvc + bridge_cvc + bridge_impact + bridge_other"
# Reference = within-community (all bridge dummies = 0)

for outcome, label, ctrl, title in outcomes:
    print()
    print("="*72)
    print(title)
    print("="*72)

    sub = df_model.dropna(subset=[outcome]).copy()
    n_obs = len(sub)

    m = smf.ols(f"{outcome} ~ {bridge_vars} + {ctrl}", data=sub).fit(cov_type="HC1")

    wc_mean = sub[sub["cross_community"]==0][outcome].mean()
    print(f"\n  Reference (within-community): mean {label} = {wc_mean:.3f}  N={int((sub['cross_community']==0).sum()):,}")
    print(f"\n  {'Bridge type':<20}  {'β':>8}  {'SE':>7}  {'Sig':>5}  {'Effect':>12}  N")
    print(f"  {'-'*65}")

    for var, bname in [("bridge_ivc","IVC"),("bridge_gvc","GVC"),
                       ("bridge_cvc","CVC"),("bridge_impact","IMPACT"),
                       ("bridge_other","OTHER")]:
        b  = m.params.get(var, np.nan)
        se = m.bse.get(var, np.nan)
        p  = m.pvalues.get(var, 1.0)
        n  = int(df_cross[df_cross["bridge_type"]==bname].shape[0])
        if outcome in ("is_failed","is_exited","got_followon","has_patent"):
            effect_str = f"{b*100:>+.1f}pp"
        else:
            effect_str = f"exp({b:.2f})={np.exp(b):.2f}×"
        print(f"  {f'Bridge={bname}':<20}  {b:>+8.4f}  {se:>7.4f}  {sig(p):>5}  {effect_str:>12}  {n}")

    print(f"\n  R²={m.rsquared:.3f}   N={int(m.nobs):,}")

    # Best and worst bridge type
    coefs = {var: m.params.get(var, np.nan)
             for var in ["bridge_ivc","bridge_gvc","bridge_cvc","bridge_impact","bridge_other"]}
    coefs = {k: v for k, v in coefs.items() if not np.isnan(v)}
    if coefs:
        if outcome in ("is_failed",):
            best = min(coefs, key=coefs.get)
            worst = max(coefs, key=coefs.get)
        else:
            best = max(coefs, key=coefs.get)
            worst = min(coefs, key=coefs.get)
        best_name  = best.replace("bridge_", "").upper()
        worst_name = worst.replace("bridge_", "").upper()
        print(f"  → Best bridge for {label}: {best_name} (β={coefs[best]:>+.4f})")
        print(f"  → Worst bridge for {label}: {worst_name} (β={coefs[worst]:>+.4f})")

# ── GREEN vs TRADITIONAL BRIDGE ───────────────────────────────────────────────
print()
print("="*72)
print("GREEN vs TRADITIONAL BRIDGE")
print("="*72)

sub_gt = df_model.dropna(subset=["bridge_green_pct"]).copy()
# For within-community companies bridge_green_pct=NaN → excluded
# We compare cross-community only: green bridge vs trad bridge vs within-comm
sub_gt_full = df_model.copy()

print(f"\n  Bridge = GREEN (≥50% green investors): {df_cross['bridge_is_green'].sum():.0f} companies")
print(f"  Bridge = TRAD  (<50% green investors): {df_cross['bridge_is_trad'].sum():.0f} companies")

for outcome, label in [("is_failed","failure"), ("is_exited","exit"),
                       ("got_followon","follow-on"), ("has_patent","patents")]:
    sub = df_model.dropna(subset=[outcome]).copy()
    m_gt = smf.ols(f"{outcome} ~ bridge_is_green + bridge_is_trad + {CTRL}", data=sub).fit(cov_type="HC1")
    b_g = m_gt.params.get("bridge_is_green", np.nan)
    p_g = m_gt.pvalues.get("bridge_is_green", 1.0)
    b_t = m_gt.params.get("bridge_is_trad",  np.nan)
    p_t = m_gt.pvalues.get("bridge_is_trad",  1.0)

    # Wald test: green = trad?
    cov_m = m_gt.cov_params()
    if "bridge_is_green" in cov_m.index and "bridge_is_trad" in cov_m.index:
        se_diff = np.sqrt(float(cov_m.loc["bridge_is_green","bridge_is_green"]) +
                          float(cov_m.loc["bridge_is_trad","bridge_is_trad"]) -
                          2*float(cov_m.loc["bridge_is_green","bridge_is_trad"]))
        t_wald  = (b_g - b_t) / se_diff if se_diff > 0 else np.nan
        p_wald  = 2 * stats.t.sf(abs(t_wald), df=m_gt.df_resid) if not np.isnan(t_wald) else np.nan
        wald_str = f"Wald(green=trad): t={t_wald:.2f} {sig(p_wald)}"
    else:
        wald_str = ""

    print(f"\n  {label.upper():<12}  β_green={b_g:>+.4f} {sig(p_g)}   β_trad={b_t:>+.4f} {sig(p_t)}   {wald_str}")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print()
print("="*72)
print("SUMMARY: DOES BRIDGE TYPE MATTER?")
print("="*72)
print("""
  KEY QUESTION: Is the cross-community effect uniform across bridge types,
  or does the SPECIFIC TYPE of bridge investor predict SPECIFIC outcomes?

  Hypotheses tested:
  → IVC bridge:    better exits (M&A networks, commercial connections)
  → GVC bridge:    more follow-on (government signals safety)
  → CVC bridge:    more employees (market access, scale)
  → Green bridge:  better survival (mission alignment, patient capital)
  → Trad bridge:   better exits (commercial validation)
""")

# Quick summary across models
print("  Bridge type × outcome matrix (sign of β, significance):")
print(f"  {'Type':<12} {'Failure':>10} {'Exit':>10} {'Follow-on':>10} {'Employees':>10} {'Patents':>10}")
print(f"  {'-'*62}")

for bt in ["IVC","GVC","CVC","IMPACT","OTHER"]:
    var = f"bridge_{bt.lower()}"
    row = f"  {bt:<12}"
    for outcome, _, ctrl, _ in outcomes:
        sub = df_model.dropna(subset=[outcome]).copy()
        try:
            m = smf.ols(f"{outcome} ~ {bridge_vars} + {ctrl}", data=sub).fit(cov_type="HC1")
            b = m.params.get(var, np.nan)
            p = m.pvalues.get(var, 1.0)
            if np.isnan(b):
                cell = "     n/a"
            else:
                sign = "+" if b > 0 else "-"
                cell = f" {sign}{sig(p):>5}"
        except:
            cell = "     err"
        row += f"{cell:>10}"
    print(row)

print(f"\n  Note: For failure, '-' is GOOD (reduces failure). For others, '+' is GOOD.")
