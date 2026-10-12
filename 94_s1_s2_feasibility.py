"""
Script 94: S1 and S2 feasibility. NO outcome regressions.

1. Type coverage for the first-round+novelty sample.
2. S1 cells: investor type × OS tertile, × NC tertile (tie level).
3. S2 feasibility: startups with ≥1 GVC and ≥1 non-GVC, both dated.
4. Confound check: GVC presence vs country, founding year, log_capital.
5. Pre-registration draft for S1 and S2.
"""

import os, re, warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

OUT_DIR = "output/s1_s2"
os.makedirs(OUT_DIR, exist_ok=True)

def sep(title=""):
    print(f"\n{'─'*68}")
    if title:
        print(f"  {title}")
        print(f"{'─'*68}")

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

print("=" * 68)
print("SCRIPT 94 — S1 / S2 FEASIBILITY")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
sep("Loading data")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl["year"] = dl["Deal Date"].dt.year
dl["co"]   = dl["Companies"].str.strip()
deals_v = dl.dropna(subset=["year","Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

# Novelty
nat = pd.read_csv("output/novelty_t/novelty_time_aware.csv")
nov = pd.read_csv("output/novelty/novelty_scores.csv")
co_os  = dict(zip(nat["Companies"], nat["overall_sparsity_t"]))
co_nc  = dict(zip(nat["Companies"], nat["niche_cohesion_t"]))
valid_cos = set(nat["Companies"]) & set(nov["name"])

# First rounds (for S1, S2)
first_round = deals_v.sort_values("Deal Date").groupby("co").first().reset_index()
fr = first_round[first_round["co"].isin(valid_cos)].copy()
fr["os"] = fr["co"].map(co_os)
fr["nc"] = fr["co"].map(co_nc)

# Deals data (capital)
capital_col = next((c for c in dl.columns if "Total Raised" in str(c)), None)
if capital_col:
    fr["capital_raw"] = pd.to_numeric(dl.groupby("co")[capital_col].max().reindex(fr["co"]).values,
                                       errors="coerce")
    fr["log_capital"]  = np.log1p(fr["capital_raw"].fillna(0))
else:
    fr["log_capital"] = np.nan

# Founding year from deals date
fr["founding_year"] = fr["Deal Date"].dt.year

# Country
fr["country"] = fr["Company Country/Territory/Region"].fillna("").str.strip()

print(f"  First-round sample (first_round ∩ nat ∩ nov): {len(fr):,}")

# ─────────────────────────────────────────────────────────────────────
# Investor type mapping
# ─────────────────────────────────────────────────────────────────────
sep("Investor type mapping")

inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors","Primary Investor Type",
                                 "Year Founded","HQ Country/Territory/Region"])
inv_raw["k"]    = inv_raw["Investors"].str.strip().str.lower()
inv_raw["ptype"] = inv_raw["Primary Investor Type"].fillna("").str.strip()
type_map   = dict(zip(inv_raw["k"], inv_raw["ptype"]))
inv_yr_map = dict(zip(inv_raw["k"], pd.to_numeric(inv_raw["Year Founded"], errors="coerce")))

# Canonical type buckets: GVC / CVC / IVC / Impact / Other
GVC_LABELS   = {"Government", "Sovereign Wealth Fund", "Not-For-Profit Venture Capital"}
CVC_LABELS   = {"Corporate Venture Capital"}
IVC_LABELS   = {"Venture Capital"}
IMP_LABELS   = {"Impact Investing"}

def canonical_type(ptype):
    if ptype in GVC_LABELS:  return "GVC"
    if ptype in CVC_LABELS:  return "CVC"
    if ptype in IVC_LABELS:  return "IVC"
    if ptype in IMP_LABELS:  return "Impact"
    if ptype == "":          return None          # unknown
    return "Other"

# Universe coverage
all_inv_keys = set()
for _, row in deals_v[deals_v["co"].isin(valid_cos)].iterrows():
    for i in comma_parse(str(row["Investors"])):
        all_inv_keys.add(i.lower())

n_inv_total  = len(all_inv_keys)
n_inv_typed  = sum(1 for k in all_inv_keys if canonical_type(type_map.get(k,"")) is not None)
type_dist    = Counter(canonical_type(type_map.get(k,"")) for k in all_inv_keys
                       if canonical_type(type_map.get(k,"")) is not None)

print(f"  Unique investor names in sample: {n_inv_total:,}")
print(f"  Investors with type:             {n_inv_typed:,}  ({100*n_inv_typed/n_inv_total:.1f}%)")
print(f"  Type distribution (typed investors):")
for t in ["IVC","CVC","GVC","Impact","Other"]:
    print(f"    {t:<8} {type_dist.get(t,0):>5}")

# ─────────────────────────────────────────────────────────────────────
# 1. Type coverage by startup
# ─────────────────────────────────────────────────────────────────────
sep("1. Type coverage by startup")

cov_rows = []
for _, row in fr.iterrows():
    co   = row["co"]
    invs = comma_parse(str(row["Investors"]))
    n_inv = len(invs)
    types = [canonical_type(type_map.get(i.lower(),"")) for i in invs]
    typed = [t for t in types if t is not None]
    pct   = len(typed)/n_inv if n_inv > 0 else 0.0
    type_flags = {t: int(t in typed) for t in ["GVC","CVC","IVC","Impact","Other"]}
    cov_rows.append({"co": co, "n_inv": n_inv, "n_typed": len(typed),
                     "pct_typed": pct, **type_flags,
                     "os": row.get("os"), "nc": row.get("nc"),
                     "country": row["country"], "founding_year": row["founding_year"],
                     "log_capital": row["log_capital"]})

cov = pd.DataFrame(cov_rows)

# Coverage tiers
t100 = (cov["pct_typed"] == 1.0).sum()
t75  = (cov["pct_typed"] >= 0.75).sum()
t50  = (cov["pct_typed"] >= 0.50).sum()
t1   = (cov["pct_typed"] > 0).sum()
t0   = (cov["pct_typed"] == 0).sum()

print(f"  Coverage tiers (N={len(cov):,}):")
print(f"    100% typed:  {t100:>5}  ({100*t100/len(cov):.1f}%)")
print(f"    ≥ 75% typed: {t75:>5}  ({100*t75/len(cov):.1f}%)  ← recommended analysis set")
print(f"    ≥ 50% typed: {t50:>5}  ({100*t50/len(cov):.1f}%)")
print(f"    ≥ 1  typed:  {t1:>5}  ({100*t1/len(cov):.1f}%)")
print(f"    0   typed:   {t0:>5}  ({100*t0/len(cov):.1f}%)")

# Type presence (any investor of type X in first-round syndicate)
print(f"\n  Startups with ≥1 investor of each type (full sample):")
for t in ["GVC","CVC","IVC","Impact","Other"]:
    n = cov[t].sum()
    print(f"    {t:<8} {n:>5}  ({100*n/len(cov):.1f}%)")

print(f"\n  Startups with ≥1 investor of each type (≥75% typed subset, n={t75:,}):")
cov75 = cov[cov["pct_typed"] >= 0.75]
for t in ["GVC","CVC","IVC","Impact","Other"]:
    n = cov75[t].sum()
    print(f"    {t:<8} {n:>5}  ({100*n/len(cov75):.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# Build tie-level dataset (investor × startup, first-round only)
# ─────────────────────────────────────────────────────────────────────
sep("Building tie-level dataset")

tie_rows = []
for _, row in fr.iterrows():
    co   = row["co"]
    os_v = co_os.get(co); nc_v = co_nc.get(co)
    if os_v is None or nc_v is None: continue
    invs = comma_parse(str(row["Investors"]))
    for inv in invs:
        inv_k = inv.lower()
        ctype = canonical_type(type_map.get(inv_k,""))
        if ctype is None: continue  # skip untyped
        tie_rows.append({"co": co, "investor": inv_k, "type": ctype,
                         "os": os_v, "nc": nc_v,
                         "country": row["country"],
                         "founding_year": int(row["founding_year"])
                         if pd.notna(row["founding_year"]) else None,
                         "log_capital": row["log_capital"]})

ties = pd.DataFrame(tie_rows)
print(f"  Tie-level rows (typed investors only): {len(ties):,}")
print(f"  Unique startups: {ties['co'].nunique():,}")
print(f"  Type distribution in ties:")
print(ties["type"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────
# 2. S1 cells: type × OS tertile and × NC tertile
# ─────────────────────────────────────────────────────────────────────
sep("2. S1 cells: type × novelty tertile")

ties["os_t"] = pd.qcut(ties["os"], q=3, labels=["Low","Mid","High"])
ties["nc_t"] = pd.qcut(ties["nc"], q=3, labels=["Low","Mid","High"])

TYPES = ["GVC","CVC","IVC","Impact","Other"]

# Cross-tab: type × OS tertile
print("\n  Tie counts: investor type × OS tertile  (rows=type, cols=tertile)")
ct_os = pd.crosstab(ties["type"], ties["os_t"])
ct_os = ct_os.reindex(index=TYPES, fill_value=0)
print(ct_os.to_string())

# Flag cells < 200
flags_os = [(t, lv) for t in TYPES for lv in ["Low","Mid","High"]
            if ct_os.loc[t, lv] < 200] if "Low" in ct_os.columns else []
if flags_os:
    print(f"\n  ⚠ Cells < 200 (OS):  {flags_os}")
else:
    print("\n  All OS cells ≥ 200")

# Cross-tab: type × NC tertile
print("\n  Tie counts: investor type × NC tertile  (rows=type, cols=tertile)")
ct_nc = pd.crosstab(ties["type"], ties["nc_t"])
ct_nc = ct_nc.reindex(index=TYPES, fill_value=0)
print(ct_nc.to_string())

flags_nc = [(t, lv) for t in TYPES for lv in ["Low","Mid","High"]
            if ct_nc.loc[t, lv] < 200] if "Low" in ct_nc.columns else []
if flags_nc:
    print(f"\n  ⚠ Cells < 200 (NC):  {flags_nc}")
else:
    print("\n  All NC cells ≥ 200")

# Mean OS and NC per type
print("\n  Mean OS and NC by investor type (at tie level):")
agg = ties.groupby("type").agg(
    n=("co","count"),
    os_mean=("os","mean"), os_sd=("os","std"),
    nc_mean=("nc","mean"), nc_sd=("nc","std")
).reindex(TYPES)
print(agg.round(3).to_string())

# ─────────────────────────────────────────────────────────────────────
# Type counts by country (top 10 countries)
# ─────────────────────────────────────────────────────────────────────
sep("Type counts by country (startup HQ)")
top_ctry = ties["country"].value_counts().head(8).index.tolist()
ct_ctry  = pd.crosstab(ties["country"].where(ties["country"].isin(top_ctry), other="Other"),
                        ties["type"])
ct_ctry  = ct_ctry.reindex(columns=TYPES, fill_value=0)
ct_ctry["Total"] = ct_ctry.sum(axis=1)
ct_ctry = ct_ctry.sort_values("Total", ascending=False).head(10)
print(ct_ctry.to_string())

# ─────────────────────────────────────────────────────────────────────
# Type counts by round number
# ─────────────────────────────────────────────────────────────────────
sep("Type counts by round number")

# Need all rounds, not just first round
round_type_rows = []
for _, row in deals_v[deals_v["co"].isin(valid_cos)].iterrows():
    co = row["co"]
    rn = row.get("Round Number", row.get("Deal Type", None))
    if pd.isna(rn): rn = "Unknown"
    for inv in comma_parse(str(row["Investors"])):
        ctype = canonical_type(type_map.get(inv.lower(),""))
        if ctype:
            round_type_rows.append({"co": co, "round": str(rn), "type": ctype})

all_rounds = pd.DataFrame(round_type_rows)
if len(all_rounds) > 0:
    # Normalise round labels
    round_col = "round"
    ct_round = pd.crosstab(all_rounds[round_col], all_rounds["type"])
    ct_round = ct_round.reindex(columns=TYPES, fill_value=0)
    ct_round["Total"] = ct_round.sum(axis=1)
    ct_round = ct_round.sort_values("Total", ascending=False).head(15)
    print(ct_round.to_string())
else:
    print("  No round data available.")

# ─────────────────────────────────────────────────────────────────────
# 3. S2 feasibility: startups with ≥1 GVC and ≥1 non-GVC investor
# ─────────────────────────────────────────────────────────────────────
sep("3. S2 feasibility: GVC co-investors")

# For S2 we need all rounds per startup, with dates
# Find first date of GVC investment and first date of non-GVC investment per startup
gvc_dates  = defaultdict(list)
priv_dates = defaultdict(list)

for _, row in deals_v[deals_v["co"].isin(valid_cos)].iterrows():
    co   = row["co"]
    dt   = row["Deal Date"]
    if pd.isna(dt): continue
    for inv in comma_parse(str(row["Investors"])):
        ctype = canonical_type(type_map.get(inv.lower(),""))
        if ctype == "GVC":
            gvc_dates[co].append(dt)
        elif ctype in ("CVC","IVC","Impact","Other"):
            priv_dates[co].append(dt)

# Startups with both
both = [co for co in valid_cos if gvc_dates.get(co) and priv_dates.get(co)]
print(f"  Startups with ≥1 GVC investment:         {len(gvc_dates):>5}")
print(f"  Startups with ≥1 non-GVC investment:     {len(priv_dates):>5}")
print(f"  Both (S2 eligible):                      {len(both):>5}")

# Distribution of Δ days: first GVC date − first non-GVC date
delta_days = []
for co in both:
    d_gvc  = min(gvc_dates[co])
    d_priv = min(priv_dates[co])
    delta_days.append((d_gvc - d_priv).days)

delta_s = pd.Series(delta_days)
print(f"\n  Distribution of (first GVC date) − (first private date), days:")
print(f"    N:         {len(delta_s):>6}")
print(f"    Mean:      {delta_s.mean():>+8.1f}")
print(f"    Median:    {delta_s.median():>+8.1f}")
print(f"    SD:        {delta_s.std():>8.1f}")
print(f"    P10:       {delta_s.quantile(.10):>+8.1f}")
print(f"    P25:       {delta_s.quantile(.25):>+8.1f}")
print(f"    P75:       {delta_s.quantile(.75):>+8.1f}")
print(f"    P90:       {delta_s.quantile(.90):>+8.1f}")
print(f"    GVC first (Δ<0): {(delta_s<0).sum():>4}  ({100*(delta_s<0).mean():.1f}%)")
print(f"    Same day  (Δ=0): {(delta_s==0).sum():>4}  ({100*(delta_s==0).mean():.1f}%)")
print(f"    GVC later (Δ>0): {(delta_s>0).sum():>4}  ({100*(delta_s>0).mean():.1f}%)")

# In years
delta_yr = delta_s / 365.25
print(f"\n  In years — Mean: {delta_yr.mean():+.2f}  Median: {delta_yr.median():+.2f}  "
      f"SD: {delta_yr.std():.2f}")

# ─────────────────────────────────────────────────────────────────────
# 4. Confound check: GVC presence vs country, founding year, log_capital
# ─────────────────────────────────────────────────────────────────────
sep("4. Confound check")

# Startup-level flag: any GVC in first round syndicate
cov["any_gvc"] = cov["GVC"]
cov["log_cap_nonan"] = cov["log_capital"].fillna(0)
cov["fy_nonan"]      = cov["founding_year"].fillna(cov["founding_year"].median())
cov["is_US"]         = (cov["country"] == "United States").astype(float)
cov["is_EU"]         = cov["country"].isin(
    ["United Kingdom","Germany","France","Sweden","Netherlands",
     "Denmark","Finland","Norway","Switzerland","Spain","Italy"]).astype(float)
cov["is_China"]      = (cov["country"] == "China").astype(float)

gvc_flag = cov["any_gvc"].values.astype(float)

print("\n  Point-biserial correlations with GVC presence (first-round syndicate):")
print(f"  {'Variable':<25}  {'r':>7}  {'p':>8}")
for col, lbl in [("log_cap_nonan","log_capital"),
                 ("fy_nonan","founding_year"),
                 ("is_US","is_US"),
                 ("is_EU","is_EU"),
                 ("is_China","is_China")]:
    r, p = pearsonr(cov[col].values, gvc_flag)
    stars = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
    print(f"  {lbl:<25}  {r:>+7.4f}  {p:>8.4f}  {stars}")

# GVC rate by top countries
print("\n  GVC rate by country (top 10 by N):")
gvc_by_ctry = (cov.groupby("country")
               .agg(n=("co","count"), gvc_rate=("any_gvc","mean"))
               .query("n >= 20")
               .sort_values("n", ascending=False)
               .head(10))
gvc_by_ctry["gvc_rate_pct"] = gvc_by_ctry["gvc_rate"] * 100
print(gvc_by_ctry[["n","gvc_rate_pct"]].round(1).to_string())

# GVC rate by founding year quinquennium
cov["year_bin"] = pd.cut(cov["founding_year"], bins=[1990,1999,2004,2009,2014,2019,2030],
                          labels=["≤1999","2000-04","2005-09","2010-14","2015-19","2020+"])
print("\n  GVC rate by founding year bin:")
print(cov.groupby("year_bin").agg(n=("co","count"),
                                   gvc_rate=("any_gvc","mean")).assign(
    gvc_pct=lambda x: (x.gvc_rate*100).round(1)).drop(columns="gvc_rate").to_string())

# ─────────────────────────────────────────────────────────────────────
# Save tables
# ─────────────────────────────────────────────────────────────────────
ct_os.to_csv(f"{OUT_DIR}/s1_cells_os.csv")
ct_nc.to_csv(f"{OUT_DIR}/s1_cells_nc.csv")
agg.round(3).to_csv(f"{OUT_DIR}/s1_type_novelty.csv")
pd.DataFrame({"delta_days": delta_days}).to_csv(f"{OUT_DIR}/s2_delta_days.csv", index=False)
cov.to_csv(f"{OUT_DIR}/s1_coverage.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 5. Pre-registration draft
# ─────────────────────────────────────────────────────────────────────
sep("5. Pre-registration draft — Families G (S1) and H (S2)")

print("""
  DRAFT — DO NOT LOCK UNTIL REVIEWED
  ════════════════════════════════════════════════════════════

  Family G — Investor type and startup novelty (S1)

  Unit of analysis: investor–startup tie (first-round syndicates).
  Sample: startups with ≥75% of investors classified by type,
    intersected with the novelty-time-aware dataset.
  Outcome: overall_sparsity_t (primary); niche_cohesion_t (secondary).
  Model: OLS with startup cluster-robust SEs (investor clustered too?
    → decision needed). Include controls: founding_year, log_capital,
    country FE, year FE.

  H-G1 (S1 primary): GVC investors fund more novel startups than IVC,
    which fund more than CVC.
    Pre-specified ordering: E[OS | GVC] > E[OS | IVC] > E[OS | CVC].
    Test: pairwise contrasts from the OLS, Holm-corrected.
    Sign: β(GVC) > β(IVC) > β(CVC) relative to Other baseline.

  H-G2 (S1 secondary): same ordering on niche_cohesion_t.

  Holm across G1–G2 (two outcomes, same ordering hypothesis).

  ────────────────────────────────────────────────────────────

  Family H — GVC timing (S2)

  Unit of analysis: startup (with ≥1 GVC and ≥1 non-GVC investment,
    both rounds dated).
  Outcome: Δ_days = (first GVC investment date) − (first private
    investment date). Negative = GVC came first.
  Test: one-sample t-test of H0: E[Δ_days] = 0 vs H1: E[Δ_days] < 0.
    (GVC enters earlier than private capital, on average.)

  H-H1 (S2): mean Δ_days < 0 — public/government capital enters
    before private VC, consistent with de-risking role of public VC.

  Controls to add: country FE, founding year, log_capital (as
    robustness — primary test is unconditional on Δ_days).

  Note on confounds (from section 4):
    GVC presence correlates with country (GVC rate varies by country),
    founding year, and capital. A conditional analysis is needed to
    separate GVC timing from selection on these covariates.
    Robustness: repeat S2 within US-only and EU-only subsamples.

  ════════════════════════════════════════════════════════════
  STATUS: DRAFT — review cells, delta distribution, confounds before
          locking. Particularly:
          (a) Flag whether GVC cells are adequate for all type pairs.
          (b) Confirm GVC definition (Government + SWF + NFP-VC).
          (c) Decide on tie-level vs startup-level model for S1.
""")

print("=" * 68)
print("SCRIPT 94 COMPLETE")
print("=" * 68)
