"""
Script 43: Failure Variable Fix + Robustness Tests
===================================================

FIX 1 — Recode 16 Bankruptcy: Admin/Reorg companies from Active → Failed
FIX 2 — Three failure definitions (narrow / broad / very_broad)
FIX 3 — Re-run main cross-community models under all three definitions
FIX 4 — Identify distress acquisitions (M&A at a loss or within 12 months of last round)
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals, load_companies

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

ROOT = Path(".")
OUT  = ROOT / "output"

CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ── Load base data ────────────────────────────────────────────────────────────
print("Loading data…")
cdf  = pd.read_csv(OUT / "cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["log_cap"]  = cdf["log_capital"]
cdf["nrounds"]  = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

companies = load_companies()
companies["co"] = companies["Companies"].str.strip()

deals = load_deals()
deals["co"]   = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")

# Last deal date per company
last_deal = deals.groupby("co")["Deal Date"].max().rename("last_deal")

# Merge PitchBook status fields into cdf
cdf = cdf.merge(
    companies[["co","Business Status","Ownership Status"]],
    on="co", how="left"
).merge(last_deal, on="co", how="left")
# Total Raised already in cdf from cpu_company_dataset; merge in separately if missing
if "Total Raised" not in cdf.columns:
    cdf = cdf.merge(companies[["co","Total Raised"]], on="co", how="left")

CUTOFF = pd.Timestamp("2026-07-03")
cdf["years_since_deal"] = (CUTOFF - cdf["last_deal"]).dt.days / 365.25

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: RECODE 16 BANKRUPTCY: ADMIN/REORG CASES
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("FIX 1: RECODE BANKRUPTCY: ADMIN/REORG → FAILED")
print("="*72)

reorg_mask = (
    (cdf["outcome"] == "Active / Private") &
    (cdf["Business Status"] == "Bankruptcy: Admin/Reorg")
)
reorg_cos = cdf[reorg_mask][["co","Business Status","Ownership Status"]].copy()

print(f"\n  Companies recoded from 'Active / Private' → 'Failed': {len(reorg_cos)}")
print(f"\n  {'Company':<55} {'Business Status':<28} {'Ownership Status'}")
print(f"  {'-'*110}")
for _, r in reorg_cos.iterrows():
    print(f"  {r['co']:<55} {str(r['Business Status']):<28} {r['Ownership Status']}")

cdf["outcome_fixed"] = cdf["outcome"].copy()
cdf.loc[reorg_mask, "outcome_fixed"] = "Failed"

print(f"\n  Original Failed:  {(cdf['outcome']=='Failed').sum():,}")
print(f"  Fixed Failed:     {(cdf['outcome_fixed']=='Failed').sum():,}")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: THREE FAILURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("FIX 2: THREE FAILURE DEFINITIONS")
print("="*72)

# narrow: fixed outcome (reorg recoded)
cdf["failed_narrow"] = (cdf["outcome_fixed"] == "Failed").astype(float)

# broad: narrow + dormant 5+ years (no deal since before July 2021)
dormant_5yr = (
    (cdf["outcome_fixed"] == "Active / Private") &
    (cdf["years_since_deal"] >= 5)
)
cdf["failed_broad"] = ((cdf["failed_narrow"] == 1) | dormant_5yr).astype(float)

# very_broad: narrow + dormant 3+ years (no deal since before July 2023)
dormant_3yr = (
    (cdf["outcome_fixed"] == "Active / Private") &
    (cdf["years_since_deal"] >= 3)
)
cdf["failed_very_broad"] = ((cdf["failed_narrow"] == 1) | dormant_3yr).astype(float)

n_narrow     = int(cdf["failed_narrow"].sum())
n_broad      = int(cdf["failed_broad"].sum())
n_very_broad = int(cdf["failed_very_broad"].sum())
N            = len(cdf)

print(f"\n  {'Definition':<20}  {'N Failed':>9}  {'Rate':>7}  {'Components'}")
print(f"  {'-'*70}")
print(f"  {'failed_narrow':<20}  {n_narrow:>9,}  {n_narrow/N:>7.1%}  "
      f"Out-of-Business + Bankruptcy (Liq + Reorg)")
print(f"  {'failed_broad':<20}  {n_broad:>9,}  {n_broad/N:>7.1%}  "
      f"narrow + dormant 5+ years (last deal ≤ 2021)")
print(f"  {'failed_very_broad':<20}  {n_very_broad:>9,}  {n_very_broad/N:>7.1%}  "
      f"narrow + dormant 3+ years (last deal ≤ 2023)")

# Dormant companies added
dormant_5_added = cdf[dormant_5yr][["co","years_since_deal","last_deal"]].copy()
dormant_3_added = cdf[dormant_3yr][["co","years_since_deal","last_deal"]].copy()
print(f"\n  Dormant 5yr additions: {len(dormant_5_added):,}  "
      f"(last deal range: {dormant_5_added['last_deal'].min().date()} – "
      f"{dormant_5_added['last_deal'].max().date()})")
print(f"  Dormant 3yr additions: {len(dormant_3_added):,}  "
      f"(last deal range: {dormant_3_added['last_deal'].min().date()} – "
      f"{dormant_3_added['last_deal'].max().date()})")

# Cross-community rate among each failure definition's additions
for name, mask, added in [("dormant 5yr", dormant_5yr, dormant_5_added),
                           ("dormant 3yr", dormant_3yr, dormant_3_added)]:
    cross_rate = cdf[mask]["cross_community"].mean()
    narrow_rate = cdf[cdf["failed_narrow"]==1]["cross_community"].mean()
    print(f"\n  {name} cross-community rate: {cross_rate:.1%}  "
          f"(vs narrow failed: {narrow_rate:.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: RE-RUN MAIN MODELS ACROSS ALL THREE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("FIX 3: MAIN MODELS ACROSS THREE FAILURE DEFINITIONS")
print("="*72)

df = cdf.dropna(subset=["log_cap","deal_year","specter_novelty"]).copy()

failure_defs = [
    ("failed_narrow",     "Narrow (578)"),
    ("failed_broad",      "Broad (5yr dormant, ~914)"),
    ("failed_very_broad", "Very Broad (3yr dormant, ~1,956)"),
]

models = [
    ("A. Cross-community OLS",
     lambda y, d: smf.ols(f"{y} ~ cross_community + {CTRL}", data=d).fit(cov_type="HC1"),
     "cross_community"),
    ("B. Green → failure OLS",
     lambda y, d: smf.ols(f"{y} ~ pct_green + {CTRL}", data=d).fit(cov_type="HC1"),
     "pct_green"),
    ("C. GVC → failure OLS",
     lambda y, d: smf.ols(f"{y} ~ pct_gvc + {CTRL}", data=d).fit(cov_type="HC1"),
     "pct_gvc"),
]

# Print header
print()
col_w = 32
header = f"  {'Model':<{col_w}}"
for _, label in failure_defs:
    header += f"  {label:>28}"
print(header)
print(f"  {'-'*100}")

for model_name, fit_fn, var in models:
    print(f"\n  {model_name}")
    for col_label, row_label in [("β",""), ("p","")]:
        pass

    row_b = f"    {'β':>10}"
    row_p = f"    {'p':>10}"
    row_n = f"    {'N':>10}"
    for yvar, label in failure_defs:
        sub = df.dropna(subset=[yvar]).copy()
        try:
            m = fit_fn(yvar, sub)
            b = m.params.get(var, np.nan)
            p = m.pvalues.get(var, 1.0)
            row_b += f"  {b:>+10.4f}  {sig(p):>5}"
            row_p += f"  {'p='+f'{p:.3f}':>16}"
            row_n += f"  {'N='+str(int(m.nobs)):>16}"
        except Exception as e:
            row_b += f"  {'ERROR':>16}"
            row_p += f"  {'':>16}"
            row_n += f"  {'':>16}"
    print(row_b)
    print(row_p)
    print(row_n)

# 2SLS (community size IV) — simplified within-FE version
print()
print("  D. 2SLS (community size IV) — cross-community → failure")
print(f"  {'':>12}  {'':>28}", end="")

net = pd.read_csv(OUT / "network/network_metrics.csv")
comm_size = net.groupby("community").size().to_dict()
inv_to_comm = net.set_index("investor")["community"].to_dict()

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))
d_exp["community"] = d_exp["inv"].map(inv_to_comm)
co_comm_sizes = (d_exp.dropna(subset=["community"])
                 .groupby("co")["community"]
                 .apply(lambda x: min(comm_size.get(c, 999) for c in x.unique()))
                 .rename("min_community_size"))
df2 = df.merge(co_comm_sizes, on="co", how="left")
df2 = df2.dropna(subset=["min_community_size"])
df2["log_min_cs"] = np.log(df2["min_community_size"])

# Within-transform FE
df2["g"] = df2["country_fe"].astype(str) + "_" + df2["deal_year"].astype(str)

rows_iv = {"β_2sls": {}, "p_2sls": {}, "kp_f": {}}
for yvar, label in failure_defs:
    sub = df2.dropna(subset=[yvar]).copy()
    cols_needed = [yvar, "cross_community", "log_cap", "nrounds", "specter_novelty", "log_min_cs"]
    sub = sub.dropna(subset=cols_needed)
    try:
        for col in cols_needed:
            gm = sub.groupby("g")[col].transform("mean")
            sub[col+"_dm"] = sub[col] - gm
        # First stage
        fs = smf.ols(f"cross_community_dm ~ log_min_cs_dm + log_cap_dm + nrounds_dm + specter_novelty_dm",
                     data=sub).fit()
        kp_f = float(fs.fvalue)
        sub["cross_hat"] = fs.fittedvalues
        # Second stage
        ss = smf.ols(f"{yvar}_dm ~ cross_hat + log_cap_dm + nrounds_dm + specter_novelty_dm",
                     data=sub).fit(cov_type="HC1")
        b = ss.params.get("cross_hat", np.nan)
        p = ss.pvalues.get("cross_hat", 1.0)
        rows_iv["β_2sls"][label] = f"{b:>+.4f} {sig(p)}"
        rows_iv["p_2sls"][label] = f"p={p:.3f}"
        rows_iv["kp_f"][label]   = f"KP-F={kp_f:.1f}"
    except Exception as e:
        rows_iv["β_2sls"][label] = f"ERROR"
        rows_iv["p_2sls"][label] = ""
        rows_iv["kp_f"][label]   = ""

for metric, vals in rows_iv.items():
    row = f"    {metric:<10}"
    for _, label in failure_defs:
        row += f"  {vals.get(label,''):>28}"
    print(row)

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4: DISTRESS ACQUISITIONS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("FIX 4: DISTRESS ACQUISITION IDENTIFICATION")
print("="*72)

# Get acquired companies
acquired = cdf[cdf["outcome_fixed"] == "Acquired"].copy()
print(f"\n  Acquired companies: {len(acquired):,}")

# Get M&A deal records from deals file
deals_raw = pd.read_excel("../Climate_Tech_2012-2026.xlsx", sheet_name="Data", header=7)
deals_raw["co"] = deals_raw["Companies"].str.strip()
deals_raw["Deal Date"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce")
deals_raw["Deal Size"] = pd.to_numeric(deals_raw["Deal Size"], errors="coerce")

ma_deals = deals_raw[deals_raw["Deal Type"] == "Merger/Acquisition"].copy()
print(f"  M&A deal records: {len(ma_deals):,}")
print(f"  M&A with deal size: {ma_deals['Deal Size'].notna().sum():,} ({ma_deals['Deal Size'].notna().mean():.1%})")

# For each acquired company: get M&A deal size and total raised
acq_with_deals = acquired.merge(
    ma_deals[["co","Deal Date","Deal Size"]].rename(columns={"Deal Date":"acq_date","Deal Size":"acq_size"}),
    on="co", how="left"
)
acq_with_deals["Total Raised"] = pd.to_numeric(acq_with_deals["Total Raised"], errors="coerce")

# Check coverage
has_both = acq_with_deals.dropna(subset=["acq_size","Total Raised"])
print(f"  Acquired with BOTH M&A price + Total Raised: {len(has_both):,}")
print(f"  → Distress signal (price < raised) computable for these {len(has_both):,} companies")

# Distress signal 1: acquisition price < total raised
has_both["price_to_raised"] = has_both["acq_size"] / has_both["Total Raised"]
distress_price = has_both[has_both["price_to_raised"] < 1.0]
print(f"\n  DISTRESS SIGNAL 1 — Acq price < Total Raised:")
print(f"  N = {len(distress_price):,} / {len(has_both):,} evaluable  ({len(distress_price)/len(has_both):.1%})")
if len(distress_price) > 0:
    print(f"\n  {'Company':<45} {'Acq $M':>8}  {'Raised $M':>10}  {'Ratio':>6}")
    print(f"  {'-'*75}")
    for _, r in distress_price.sort_values("price_to_raised").head(15).iterrows():
        print(f"  {r['co']:<45} {r['acq_size']:>8.1f}  {r['Total Raised']:>10.1f}  {r['price_to_raised']:>6.2f}×")

# Distress signal 2: acquired within 12 months of last VC round
last_vc = (deals_raw[deals_raw["Deal Type"].isin(
    ["Early Stage VC","Later Stage VC","Seed Round","Angel (individual)"])]
    .groupby("co")["Deal Date"].max().rename("last_vc_date"))

acq_timing = acq_with_deals.dropna(subset=["acq_date"]).merge(last_vc, on="co", how="left")
acq_timing["acq_date"] = pd.to_datetime(acq_timing["acq_date"])
acq_timing["months_since_vc"] = (acq_timing["acq_date"] - acq_timing["last_vc_date"]).dt.days / 30.4
fire_sale = acq_timing[(acq_timing["months_since_vc"] >= 0) & (acq_timing["months_since_vc"] <= 12)]
print(f"\n  DISTRESS SIGNAL 2 — Acquired within 12 months of last VC round:")
print(f"  N = {len(fire_sale):,} companies  ({len(fire_sale)/len(acquired):.1%} of acquired)")

# Bankruptcy deal records for acquired companies (strong signal)
bk_deals = deals_raw[deals_raw["Deal Type"].str.contains("Bankruptcy", na=False)]
acq_with_bk = set(acquired["co"]) & set(bk_deals["co"])
print(f"\n  DISTRESS SIGNAL 3 — Acquired company with prior Bankruptcy deal record:")
print(f"  N = {len(acq_with_bk):,}")
for co in sorted(acq_with_bk):
    bk_types = bk_deals[bk_deals["co"]==co]["Deal Type"].unique()
    print(f"    {co}  →  {', '.join(bk_types)}")

# Composite distress flag: any signal
distress_cos  = (set(distress_price["co"]) | set(fire_sale["co"]) | acq_with_bk)
print(f"\n  COMPOSITE DISTRESS (any signal): {len(distress_cos):,} / {len(acquired):,} acquired companies")

# Mark distress in cdf
cdf["distress_acq"] = cdf["co"].isin(distress_cos).astype(float)
cdf["failure_adjusted"] = ((cdf["failed_narrow"] == 1) | (cdf["distress_acq"] == 1)).astype(float)

n_adj = int(cdf["failure_adjusted"].sum())
print(f"\n  failure_adjusted (narrow + distress acq): {n_adj:,}  ({n_adj/N:.1%})")

# Re-run cross-community model with failure_adjusted
print()
# Re-attach distress columns to df (df was sliced from cdf before these were created)
df = df.merge(cdf[["co","distress_acq","failure_adjusted"]], on="co", how="left")
print("  Cross-community → failure_adjusted vs failed_narrow:")
print(f"  {'Outcome':<25}  {'β_cross':>10}  {'SE':>8}  {'Sig':>5}  {'Rate':>7}  N")
print(f"  {'-'*68}")
for yvar in ["failed_narrow","failure_adjusted"]:
    sub = df.dropna(subset=[yvar]).copy()
    m = smf.ols(f"{yvar} ~ cross_community + {CTRL}", data=sub).fit(cov_type="HC1")
    b = m.params["cross_community"]
    se = m.bse["cross_community"]
    p = m.pvalues["cross_community"]
    rate = sub[yvar].mean()
    print(f"  {yvar:<25}  {b:>+10.4f}  {se:>8.4f}  {sig(p):>5}  {rate:>7.1%}  {int(m.nobs):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("SUMMARY: DO MAIN RESULTS HOLD ACROSS FAILURE DEFINITIONS?")
print("="*72)

all_outcomes = [
    ("failed_narrow",     "Narrow (N=578, 8.6%)"),
    ("failed_broad",      "Broad  (5yr, ~914)  "),
    ("failed_very_broad", "V.Broad(3yr, ~1,956)"),
    ("failure_adjusted",  "Adj    (+distress)  "),
]
for yv, _ in all_outcomes:
    if yv not in df.columns:
        df = df.merge(cdf[["co", yv]], on="co", how="left")

print(f"\n  Cross-community → failure (OLS, HC1 SEs, full controls)")
print(f"\n  {'Definition':<30}  {'β':>8}  {'SE':>7}  {'Sig':>5}  {'N Failed':>9}  {'Rate':>7}")
print(f"  {'-'*75}")
for yvar, label in all_outcomes:
    sub = df.dropna(subset=[yvar]).copy()
    m = smf.ols(f"{yvar} ~ cross_community + {CTRL}", data=sub).fit(cov_type="HC1")
    b  = m.params["cross_community"]
    se = m.bse["cross_community"]
    p  = m.pvalues["cross_community"]
    nf = int(sub[yvar].sum())
    rate = sub[yvar].mean()
    print(f"  {label:<30}  {b:>+8.4f}  {se:>7.4f}  {sig(p):>5}  {nf:>9,}  {rate:>7.1%}")

print(f"""
  INTERPRETATION:
  → If β stays negative and significant across all definitions:
    the cross-community survival advantage is robust to how we measure failure.
  → If β attenuates for failed_broad/very_broad:
    the effect is concentrated among formally-documented failures;
    dormant companies may be censored, not truly failed.
  → Note: dormant companies are right-censored observations, not confirmed failures.
    The very-broad definition treats survivorship censoring as failure —
    use it as a robustness bound, not the preferred specification.
""")
