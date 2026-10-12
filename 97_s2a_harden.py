"""
Script 97: Harden S2a.

1. Stage labels (Seed / Series A / B / C+) instead of round numbers
2. Country and year FE (as S2b required)
3. Three groups (earlier / same / later) with startup characteristics
4. Right-censoring: GVC-equity only vs private only
5. Pre/post-2020 split
6. Impact investors separately from GVC
"""

import os, re, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import ttest_1samp, mannwhitneyu, chi2_contingency
from scipy.stats import t as t_dist

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

def parse_vc_round(r):
    if pd.isna(r): return None
    m = re.match(r"(\d+)", str(r))
    return int(m.group(1)) if m else None

def stage_label(deal_type, vc_round_num):
    """Map deal type + VC round number to stage label and ordinal."""
    dt = str(deal_type) if not pd.isna(deal_type) else ""
    if "Grant" in dt:
        return "Grant", -1
    if any(x in dt for x in ["Accelerator","Angel","Pre-Seed","Pre-seed"]):
        return "Pre-seed", 0
    if "Seed" in dt:
        return "Seed", 1
    if vc_round_num is not None:
        if vc_round_num == 1: return "Series A", 2
        if vc_round_num == 2: return "Series B", 3
        if vc_round_num >= 3: return "Series C+", 4
    if "Early Stage VC" in dt:
        return "Series A/B", 2   # ordinal 2 for early
    if "Later Stage VC" in dt:
        return "Series C+", 4
    return "Unknown", np.nan

def ols_with_fe_ttest(delta, country_arr, year_arr, side="less"):
    """OLS intercept = conditional mean Δ after country+year FE."""
    df = pd.DataFrame({"d": delta, "country": country_arr, "year": year_arr})
    df["c_fe"] = df["country"].where(
        df["country"].map(df["country"].value_counts() >= 10), other="Other")
    Xd = pd.get_dummies(df[["c_fe","year"]], dtype=float, drop_first=True)
    X  = np.hstack([np.ones((len(df),1)), Xd.values])
    y  = df["d"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res   = y - X @ b
    n, p  = X.shape
    se    = np.sqrt(np.linalg.inv(X.T @ X)[0,0] * np.mean(res**2) * n / (n - p))
    tv    = b[0] / se
    if side == "less":
        pv = float(t_dist.cdf(tv, df=n-p))
    elif side == "greater":
        pv = float(t_dist.sf(tv, df=n-p))
    else:
        pv = 2 * float(t_dist.sf(abs(tv), df=n-p))
    return b[0], se, tv, pv

def describe_group(df, label):
    n = len(df)
    if n == 0:
        print(f"    {label}: N=0")
        return
    os_m  = df["os"].mean()  if "os"  in df else np.nan
    lc_m  = df["log_capital"].mean() if "log_capital" in df else np.nan
    fy_m  = df["founding_year"].mean() if "founding_year" in df else np.nan
    n_inv = df["n_inv"].mean() if "n_inv" in df else np.nan
    pct_us = (df["country"] == "United States").mean() if "country" in df else np.nan
    print(f"    {label} (N={n:,}):  OS={os_m:+.3f}  log_cap={lc_m:.3f}  "
          f"FY={fy_m:.1f}  inv/deal={n_inv:.1f}  %US={100*pct_us:.0f}%")

print("=" * 68)
print("SCRIPT 97 — S2a HARDENING")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
sep("Loading data")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl["year"]      = dl["Deal Date"].dt.year
dl["co"]        = dl["Companies"].str.strip()
dl["deal_size"] = pd.to_numeric(dl["Deal Size (USD M)"], errors="coerce")

deals_v = dl.dropna(subset=["year","Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

# Novelty
nat = pd.read_csv("output/novelty_t/novelty_time_aware.csv")
nov = pd.read_csv("output/novelty/novelty_scores.csv")
co_os = dict(zip(nat["Companies"], nat["overall_sparsity_t"]))
valid_cos = set(nat["Companies"]) & set(nov["name"])

# Investor type map
inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors","Primary Investor Type"])
inv_raw["k"]     = inv_raw["Investors"].str.strip().str.lower()
inv_raw["ptype"] = inv_raw["Primary Investor Type"].fillna("").str.strip()
type_map = dict(zip(inv_raw["k"], inv_raw["ptype"]))

GVC_LABELS = {"Government","Sovereign Wealth Fund","Not-For-Profit Venture Capital"}
def canonical_type(ptype):
    if ptype in GVC_LABELS:           return "GVC"
    if ptype == "Corporate Venture Capital": return "CVC"
    if ptype == "Venture Capital":    return "IVC"
    if ptype == "Impact Investing":   return "Impact"
    if ptype == "":                   return None
    return "Other"

# Startup metadata
co_country = deals_v.sort_values("Deal Date").groupby("co")["Company Country/Territory/Region"].first()
co_founded = deals_v.sort_values("Deal Date").groupby("co")["Deal Date"].first().dt.year
co_logcap  = deals_v.groupby("co")["deal_size"].max().apply(np.log1p)
co_ninv    = deals_v.groupby("co")["Investors"].apply(
    lambda s: np.mean([len(comma_parse(str(x))) for x in s]))

print(f"  Valid companies: {len(valid_cos):,}")

# ─────────────────────────────────────────────────────────────────────
# Build full deal-level dataset with stage labels
# ─────────────────────────────────────────────────────────────────────
sep("Building deal-level dataset with stage labels")

deal_rows = []
for _, row in deals_v[deals_v["co"].isin(valid_cos)].iterrows():
    co  = row["co"]
    dt  = row["Deal Date"]
    if pd.isna(dt): continue
    deal_type = row.get("Deal Type","")
    vc_rn     = parse_vc_round(row.get("VC Round", None))
    lbl, ord_ = stage_label(deal_type, vc_rn)
    yr        = int(row["year"])
    country   = str(row.get("Company Country/Territory/Region","")).strip()
    invs = comma_parse(str(row["Investors"]))
    for inv in invs:
        ctype = canonical_type(type_map.get(inv.lower(),""))
        if ctype is None: continue
        is_grant_deal = "Grant" in str(deal_type)
        is_gvc_equity = (ctype=="GVC")  and (not is_grant_deal)
        is_impact     = (ctype=="Impact")
        is_private    = (ctype in ("IVC","CVC","Other")) and (not is_grant_deal)
        deal_rows.append({
            "co":          co,
            "date":        dt,
            "year":        yr,
            "country":     country,
            "stage_lbl":   lbl,
            "stage_ord":   ord_,
            "vc_rn":       vc_rn,
            "is_gvc_eq":   int(is_gvc_equity),
            "is_impact":   int(is_impact),
            "is_private":  int(is_private),
        })

dall = pd.DataFrame(deal_rows)
print(f"  Tie rows (typed): {len(dall):,}  |  cos: {dall['co'].nunique():,}")
print(f"  Stage distribution:")
print(dall.groupby("stage_lbl").size().sort_values(ascending=False).head(10).to_string())

# ─────────────────────────────────────────────────────────────────────
# 4. Right-censoring check
# ─────────────────────────────────────────────────────────────────────
sep("4. Right-censoring check")

cos_gvc_eq  = set(dall[dall["is_gvc_eq"]==1]["co"])
cos_impact  = set(dall[dall["is_impact"]==1]["co"])
cos_private = set(dall[dall["is_private"]==1]["co"])
cos_both_a  = cos_gvc_eq & cos_private
cos_both_imp = cos_impact & cos_private

print(f"  Startups with GVC-equity investor (any round):  {len(cos_gvc_eq):>5}")
print(f"  Startups with private investor    (any round):  {len(cos_private):>5}")
print(f"  Both (S2a eligible):                            {len(cos_both_a):>5}")
print(f"  GVC-equity ONLY (no private yet):               {len(cos_gvc_eq - cos_private):>5}  "
      f"({100*len(cos_gvc_eq - cos_private)/len(cos_gvc_eq):.1f}% of GVC-eq startups)")
print(f"  Private ONLY (no GVC-equity yet):               {len(cos_private - cos_gvc_eq):>5}  "
      f"({100*len(cos_private - cos_gvc_eq)/len(cos_private):.1f}% of private startups)")
print(f"\n  Impact:")
print(f"  Startups with Impact investor (any round):      {len(cos_impact):>5}")
print(f"  Both Impact + private (S2-Impact eligible):     {len(cos_both_imp):>5}")
print(f"  Impact ONLY (no private yet):                   {len(cos_impact - cos_private):>5}  "
      f"({100*len(cos_impact - cos_private)/max(len(cos_impact),1):.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# S2a base: first date per investor-type per startup
# ─────────────────────────────────────────────────────────────────────
def build_s2_dataset(dall, gvc_col, label):
    gvc_first_dt = (dall[dall[gvc_col]==1]
                    .groupby("co")[["date","stage_ord","year","country"]]
                    .agg(first_dt=("date","min"),
                         first_stage=("stage_ord","min"))
                    .reset_index())
    priv_first_dt = (dall[dall["is_private"]==1]
                     .groupby("co")[["date","stage_ord"]]
                     .agg(priv_dt=("date","min"),
                          priv_stage=("stage_ord","min"))
                     .reset_index())
    merged = gvc_first_dt.merge(priv_first_dt, on="co", how="inner")
    merged["delta_days"]  = (merged["first_dt"] - merged["priv_dt"]).dt.days
    merged["delta_stage"] = merged["first_stage"] - merged["priv_stage"]
    # attach metadata
    merged["os"]           = merged["co"].map(co_os)
    merged["log_capital"]  = merged["co"].map(co_logcap).apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)
    merged["founding_year"] = merged["co"].map(co_founded)
    merged["country"]      = merged["co"].map(co_country).fillna("Unknown")
    merged["n_inv"]        = merged["co"].map(co_ninv)
    return merged

s2a = build_s2_dataset(dall, "is_gvc_eq",  "GVC-equity")
s2i = build_s2_dataset(dall, "is_impact",  "Impact")
print(f"\n  S2a sample (GVC-equity + private): {len(s2a):,}")
print(f"  S2-Impact sample:                  {len(s2i):,}")

# ─────────────────────────────────────────────────────────────────────
# 1. Stage-label analysis (Seed / Series A / B / C+)
# ─────────────────────────────────────────────────────────────────────
sep("1. Stage-label analysis (GVC-equity vs private)")

# Only rows with valid stage_ord for both
s2a_st = s2a.dropna(subset=["delta_stage"]).copy()
# Map delta_stage back to readable label
def delta_to_label(d):
    if d < 0: return "GVC earlier"
    if d == 0: return "Same stage"
    return "GVC later"

s2a_st["direction"] = s2a_st["delta_stage"].apply(delta_to_label)

print(f"\n  N with valid stage for both: {len(s2a_st):,}")
vc = s2a_st["direction"].value_counts()
for k in ["GVC earlier","Same stage","GVC later"]:
    n = vc.get(k,0)
    print(f"    {k:<16}  {n:>4}  ({100*n/len(s2a_st):.1f}%)")

d_st = s2a_st["delta_stage"]
stat_st, p_st_raw = ttest_1samp(d_st.values, popmean=0, alternative="less")
print(f"\n  One-sample t-test on Δ_stage (H1: μ<0): "
      f"mean={d_st.mean():+.3f}  t={stat_st:.3f}  p={p_st_raw:.4f}")

# Stage cross-tab: what stage was GVC first investment?
print(f"\n  Stage of first GVC-equity investment:")
print(s2a_st["first_stage"].value_counts().sort_index().to_string())
print(f"\n  Stage of first private investment:")
print(s2a_st["priv_stage"].value_counts().sort_index().to_string())

# Readable stage crosstab
stage_map = {0:"Pre-seed", 1:"Seed", 2:"Ser.A/Early", 3:"Ser.B", 4:"Ser.C+"}
s2a_st["gvc_stage_lbl"]  = s2a_st["first_stage"].map(stage_map).fillna("Unknown")
s2a_st["priv_stage_lbl"] = s2a_st["priv_stage"].map(stage_map).fillna("Unknown")
ct = pd.crosstab(s2a_st["gvc_stage_lbl"], s2a_st["priv_stage_lbl"])
print(f"\n  Crosstab: GVC-equity stage (rows) × private stage (cols)")
print(ct.to_string())

# ─────────────────────────────────────────────────────────────────────
# 2. With country and year FE (days)
# ─────────────────────────────────────────────────────────────────────
sep("2. S2a with country and year FE")

s2a_cln = s2a.dropna(subset=["delta_days","country"]).copy()
s2a_cln["yr_str"] = s2a_cln["first_dt"].dt.year.astype(str)

# Unconditional
stat_u, p_u = ttest_1samp(s2a_cln["delta_days"].values, popmean=0, alternative="less")

# Conditional
b_fe, se_fe, t_fe, p_fe = ols_with_fe_ttest(
    s2a_cln["delta_days"].values,
    s2a_cln["country"].values,
    s2a_cln["yr_str"].values,
    side="less")

print(f"\n  N = {len(s2a_cln):,}")
print(f"  {'Test':<35}  {'mean/intercept':>14}  {'SE':>8}  {'t':>7}  {'p (H1<0)':>10}")
print(f"  {'─'*35}  {'─'*14}  {'─'*8}  {'─'*7}  {'─'*10}")
print(f"  {'Unconditional t-test':<35}  {s2a_cln['delta_days'].mean():>+14.1f}  "
      f"  {'—':>6}  {stat_u:>+7.3f}  {p_u:>10.4f}")
print(f"  {'OLS + country + year FE':<35}  {b_fe:>+14.1f}  {se_fe:>8.1f}  "
      f"{t_fe:>+7.3f}  {p_fe:>10.4f}")

# Also for stage
s2a_st_cln = s2a_st.dropna(subset=["delta_stage","country"]).copy()
s2a_st_cln["yr_str"] = s2a_st_cln["first_dt"].dt.year.astype(str)
b_fe_st, se_fe_st, t_fe_st, p_fe_st = ols_with_fe_ttest(
    s2a_st_cln["delta_stage"].values,
    s2a_st_cln["country"].values,
    s2a_st_cln["yr_str"].values,
    side="less")

stat_u_st, p_u_st = ttest_1samp(s2a_st_cln["delta_stage"].values, popmean=0, alternative="less")
print(f"\n  Stage analysis:")
print(f"  {'Unconditional t-test':<35}  {s2a_st_cln['delta_stage'].mean():>+14.3f}  "
      f"  {'—':>6}  {stat_u_st:>+7.3f}  {p_u_st:>10.4f}")
print(f"  {'OLS + country + year FE':<35}  {b_fe_st:>+14.3f}  {se_fe_st:>8.3f}  "
      f"{t_fe_st:>+7.3f}  {p_fe_st:>10.4f}")

# ─────────────────────────────────────────────────────────────────────
# 3. Characteristics by direction group
# ─────────────────────────────────────────────────────────────────────
sep("3. Startup characteristics by direction group (GVC-equity)")

# Direction based on days
s2a_cln["direction"] = pd.cut(s2a_cln["delta_days"],
                               bins=[-np.inf, -1, 0, np.inf],
                               labels=["GVC earlier","Same day","GVC later"])

print(f"\n  {'Group':<16}  {'N':>5}  {'OS':>8}  {'log_cap':>8}  "
      f"{'FY':>7}  {'inv/deal':>9}  {'%US':>6}")
print(f"  {'─'*16}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*6}")

groups = {}
for g in ["GVC earlier","Same day","GVC later"]:
    sub = s2a_cln[s2a_cln["direction"]==g].copy()
    groups[g] = sub
    os_m  = sub["os"].mean()  if len(sub)>0 else np.nan
    lc_m  = sub["log_capital"].mean() if len(sub)>0 else np.nan
    fy_m  = sub["founding_year"].mean() if len(sub)>0 else np.nan
    ni_m  = sub["n_inv"].mean() if len(sub)>0 else np.nan
    pus   = (sub["country"]=="United States").mean() if len(sub)>0 else np.nan
    print(f"  {g:<16}  {len(sub):>5}  {os_m:>+8.3f}  {lc_m:>8.3f}  "
          f"{fy_m:>7.1f}  {ni_m:>9.2f}  {100*pus:>5.1f}%")

# Test OS difference: GVC-earlier vs GVC-later
if len(groups["GVC earlier"]) > 0 and len(groups["GVC later"]) > 0:
    stat_mw, p_mw = mannwhitneyu(
        groups["GVC earlier"]["os"].dropna(),
        groups["GVC later"]["os"].dropna(),
        alternative="two-sided")
    print(f"\n  OS: GVC-earlier vs GVC-later (Mann-Whitney): "
          f"U={stat_mw:.0f}  p={p_mw:.4f}")

# Country composition by group
print(f"\n  Country breakdown by direction group:")
ct_ctry = pd.crosstab(s2a_cln["direction"],
                       s2a_cln["country"].where(
    s2a_cln["country"].map(s2a_cln["country"].value_counts() >= 20),
    other="Other"))
print(ct_ctry.to_string())

# ─────────────────────────────────────────────────────────────────────
# 5. Pre / post-2020 split
# ─────────────────────────────────────────────────────────────────────
sep("5. Pre/post-2020 split (GVC-equity)")

s2a_cln["cohort"] = s2a_cln["founding_year"].apply(
    lambda y: "Pre-2020" if pd.notna(y) and y < 2020 else "2020+")

print(f"\n  {'Cohort':<10}  {'N':>5}  {'mean Δ_days':>12}  {'t':>7}  {'p (H1<0)':>10}  "
      f"{'%GVC-earlier':>13}")
print(f"  {'─'*10}  {'─'*5}  {'─'*12}  {'─'*7}  {'─'*10}  {'─'*13}")

for cohort in ["Pre-2020","2020+"]:
    sub = s2a_cln[s2a_cln["cohort"]==cohort]["delta_days"].dropna()
    if len(sub) < 10:
        print(f"  {cohort:<10}  {len(sub):>5}  (too few)")
        continue
    stat_c, p_c = ttest_1samp(sub.values, popmean=0, alternative="less")
    pct_earlier = 100*(sub < 0).mean()
    print(f"  {cohort:<10}  {len(sub):>5}  {sub.mean():>+12.1f}  "
          f"{stat_c:>+7.3f}  {p_c:>10.4f}  {pct_earlier:>12.1f}%")

# Stage-level pre/post-2020
print(f"\n  Stage-based split:")
for cohort in ["Pre-2020","2020+"]:
    sub = s2a_st_cln[s2a_st_cln["founding_year"].apply(
        lambda y: "Pre-2020" if pd.notna(y) and y < 2020 else "2020+") == cohort]["delta_stage"].dropna()
    if len(sub) < 10: continue
    stat_c, p_c = ttest_1samp(sub.values, popmean=0, alternative="less")
    print(f"  {cohort:<10}  N={len(sub):>4}  mean Δ_stage={sub.mean():+.3f}  "
          f"t={stat_c:+.3f}  p={p_c:.4f}")

# ─────────────────────────────────────────────────────────────────────
# 6. Impact investors separately
# ─────────────────────────────────────────────────────────────────────
sep("6. Impact investors separately")

# Build stage labels for Impact
s2i_cln = s2i.dropna(subset=["delta_days","country"]).copy()
s2i_cln["yr_str"] = s2i_cln["first_dt"].dt.year.astype(str)

# S2i stage
s2i_st = s2i.dropna(subset=["delta_stage"]).copy()
s2i_st["yr_str"] = s2i_st["first_dt"].dt.year.astype(str)
s2i_st["direction"] = s2i_st["delta_stage"].apply(delta_to_label)

print(f"\n  Impact + private sample: {len(s2i_cln):,}")
print(f"\n  Direction (stage-based, N={len(s2i_st):,}):")
vc_i = s2i_st["direction"].value_counts()
for k in ["GVC earlier","Same stage","GVC later"]:
    n = vc_i.get(k,0)
    print(f"    {k:<16}  {n:>4}  ({100*n/max(len(s2i_st),1):.1f}%)")

# Days
stat_iu, p_iu = ttest_1samp(s2i_cln["delta_days"].values, popmean=0, alternative="less")

if len(s2i_cln) > 20:
    b_i_fe, se_i_fe, t_i_fe, p_i_fe = ols_with_fe_ttest(
        s2i_cln["delta_days"].values,
        s2i_cln["country"].values,
        s2i_cln["yr_str"].values,
        side="less")
else:
    b_i_fe = se_i_fe = t_i_fe = p_i_fe = np.nan

print(f"\n  {'Test':<35}  {'mean/intercept':>14}  {'SE':>8}  {'t':>7}  {'p (H1<0)':>10}")
print(f"  {'─'*35}  {'─'*14}  {'─'*8}  {'─'*7}  {'─'*10}")
print(f"  {'Unconditional':<35}  {s2i_cln['delta_days'].mean():>+14.1f}  "
      f"  {'—':>6}  {stat_iu:>+7.3f}  {p_iu:>10.4f}")
print(f"  {'OLS + country + year FE':<35}  {b_i_fe:>+14.1f}  {se_i_fe:>8.1f}  "
      f"{t_i_fe:>+7.3f}  {p_i_fe:>10.4f}")

# Stage
if len(s2i_st) > 20:
    stat_ist, p_ist = ttest_1samp(s2i_st["delta_stage"].dropna().values, popmean=0, alternative="less")
    b_ist_fe, se_ist_fe, t_ist_fe, p_ist_fe = ols_with_fe_ttest(
        s2i_st["delta_stage"].values,
        s2i_st["country"].values,
        s2i_st["yr_str"].values,
        side="less")
    print(f"\n  Stage analysis (Impact, N={len(s2i_st):,}):")
    print(f"  {'Unconditional':<35}  {s2i_st['delta_stage'].mean():>+14.3f}  "
          f"  {'—':>6}  {stat_ist:>+7.3f}  {p_ist:>10.4f}")
    print(f"  {'OLS + country + year FE':<35}  {b_ist_fe:>+14.3f}  {se_ist_fe:>8.3f}  "
          f"{t_ist_fe:>+7.3f}  {p_ist_fe:>10.4f}")

# Impact characteristics by group
s2i_cln["direction"] = pd.cut(s2i_cln["delta_days"],
                                bins=[-np.inf,-1,0,np.inf],
                                labels=["Impact earlier","Same day","Impact later"])
print(f"\n  Impact startup characteristics by direction group:")
print(f"  {'Group':<17}  {'N':>5}  {'OS':>8}  {'log_cap':>8}  {'FY':>7}  {'%US':>6}")
print(f"  {'─'*17}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*6}")
for g in ["Impact earlier","Same day","Impact later"]:
    sub = s2i_cln[s2i_cln["direction"]==g]
    if len(sub)==0: continue
    print(f"  {g:<17}  {len(sub):>5}  "
          f"{sub['os'].mean():>+8.3f}  {sub['log_capital'].mean():>8.3f}  "
          f"{sub['founding_year'].mean():>7.1f}  "
          f"{100*(sub['country']=='United States').mean():>5.1f}%")

# ─────────────────────────────────────────────────────────────────────
# Summary comparison table
# ─────────────────────────────────────────────────────────────────────
sep("Summary comparison")

print(f"""
  Investor type  N (both)  mean Δ_days  %earlier  p uncond  p cond(FE)
  ─────────────  ────────  ───────────  ────────  ────────  ──────────
  GVC-equity     {len(s2a_cln):>8,}  {s2a_cln['delta_days'].mean():>+11.1f}  {100*(s2a_cln['delta_days']<0).mean():>7.1f}%  {p_u:.4f}    {p_fe:.4f}
  Impact         {len(s2i_cln):>8,}  {s2i_cln['delta_days'].mean():>+11.1f}  {100*(s2i_cln['delta_days']<0).mean():>7.1f}%  {p_iu:.4f}    {p_i_fe:.4f}

  Stage-based (Δ_stage < 0 = public enters earlier)
  ─────────────  ────────  ────────────  ───────  ─────────  ──────────
  GVC-equity     {len(s2a_st_cln):>8,}  {s2a_st_cln['delta_stage'].mean():>+12.3f}  {100*(s2a_st_cln['delta_stage']<0).mean():>6.1f}%  {p_u_st:.4f}     {p_fe_st:.4f}
  Impact         {len(s2i_st):>8,}  (see above)
""")

# Save
s2a_cln.to_csv(f"{OUT_DIR}/s2a_full_hardened.csv", index=False)
s2i_cln.to_csv(f"{OUT_DIR}/s2_impact.csv", index=False)

print("=" * 68)
print("SCRIPT 97 COMPLETE")
print("=" * 68)
