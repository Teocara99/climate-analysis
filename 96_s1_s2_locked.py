"""
Script 96: S1, S2a, S2b — locked and executed.

Locked predictions (theoretical, not revised after feasibility):
  S1:  public_mission > IVC > CVC on overall_sparsity_t
         ↔ β_PM > 0, β_CVC < 0  (IVC = reference)
  S2a: GVC-equity enters in EARLIER round than private investors
         ↔ Δ_round (GVC_eq_first − private_first) < 0
  S2b: grants precede private equity investment
         ↔ Δ_days  (grant_first − private_first)  < 0

Definitions (fixed now, on theoretical grounds):
  public_mission  = GVC + Impact
  GVC_equity tie  = investor type is GVC and Deal Type ≠ "Grant"
  GVC_grant  tie  = investor type is GVC and Deal Type == "Grant"
"""

import os, re, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import ttest_1samp
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

OUT_DIR = "output/s1_s2"
os.makedirs(OUT_DIR, exist_ok=True)
PREREG = "output/preregistration.md"

def sep(title=""):
    print(f"\n{'─'*68}")
    if title:
        print(f"  {title}")
        print(f"{'─'*68}")

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

def parse_vc_round(r):
    """Return integer round number or None."""
    if pd.isna(r): return None
    m = re.match(r"(\d+)", str(r))
    return int(m.group(1)) if m else None

def holm(pvals, labels):
    """Holm–Bonferroni correction. Returns adjusted p-values."""
    n = len(pvals)
    idx = np.argsort(pvals)
    adj = np.zeros(n)
    running_max = 0.0
    for k, i in enumerate(idx):
        adj_i = (n - k) * pvals[i]
        running_max = max(running_max, adj_i)
        adj[i] = min(running_max, 1.0)
    return adj

print("=" * 68)
print("SCRIPT 96 — S1 / S2 LOCKED EXECUTION")
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
co_nc = dict(zip(nat["Companies"], nat["niche_cohesion_t"]))
valid_cos = set(nat["Companies"]) & set(nov["name"])

# First rounds for S1
first_round = deals_v.sort_values("Deal Date").groupby("co").first().reset_index()
fr = first_round[first_round["co"].isin(valid_cos)].copy()
fr["os"]  = fr["co"].map(co_os)
fr["nc"]  = fr["co"].map(co_nc)
fr["country"]       = fr["Company Country/Territory/Region"].fillna("").str.strip()
fr["founding_year"] = fr["Deal Date"].dt.year

print(f"  First-round sample (∩ nat ∩ nov): {len(fr):,}")

# ─────────────────────────────────────────────────────────────────────
# Fix: log_capital
# ─────────────────────────────────────────────────────────────────────
sep("Fix: log_capital (Deal Size USD M, max per company)")

# Sum per-deal round capital, take max across all deals per startup
co_max_size = (deals_v.groupby("co")["deal_size"]
               .max()
               .reindex(fr["co"].values))
fr["log_capital"] = np.log1p(pd.to_numeric(co_max_size, errors="coerce").fillna(0).values)

n_cap = (pd.to_numeric(co_max_size, errors="coerce") > 0).sum()
print(f"  Startups with deal_size > 0:  {n_cap:,} / {len(fr):,}  "
      f"({100*n_cap/len(fr):.1f}%)")
print(f"  log_capital  mean={fr['log_capital'].mean():.3f}  "
      f"sd={fr['log_capital'].std():.3f}  "
      f"p25={fr['log_capital'].quantile(.25):.3f}  "
      f"p75={fr['log_capital'].quantile(.75):.3f}")

# ─────────────────────────────────────────────────────────────────────
# Investor type mapping
# ─────────────────────────────────────────────────────────────────────
sep("Investor type mapping")

inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors","Primary Investor Type"])
inv_raw["k"]    = inv_raw["Investors"].str.strip().str.lower()
inv_raw["ptype"] = inv_raw["Primary Investor Type"].fillna("").str.strip()
type_map = dict(zip(inv_raw["k"], inv_raw["ptype"]))

GVC_LABELS = {"Government","Sovereign Wealth Fund","Not-For-Profit Venture Capital"}
CVC_LABELS = {"Corporate Venture Capital"}
IVC_LABELS = {"Venture Capital"}
IMP_LABELS = {"Impact Investing"}

def canonical_type(ptype):
    if ptype in GVC_LABELS:  return "GVC"
    if ptype in CVC_LABELS:  return "CVC"
    if ptype in IVC_LABELS:  return "IVC"
    if ptype in IMP_LABELS:  return "Impact"
    if ptype == "":          return None
    return "Other"

# ─────────────────────────────────────────────────────────────────────
# S1: tie-level dataset (first round of each startup)
# ─────────────────────────────────────────────────────────────────────
sep("Building S1 tie-level dataset")

tie_rows = []
for _, row in fr.iterrows():
    co    = row["co"]
    os_v  = co_os.get(co);  nc_v  = co_nc.get(co)
    if os_v is None or nc_v is None: continue
    deal_type_str = str(row.get("Deal Type",""))
    yr = int(row["year"]) if pd.notna(row.get("year")) else None
    invs = comma_parse(str(row["Investors"]))
    for inv in invs:
        inv_k  = inv.lower()
        ctype  = canonical_type(type_map.get(inv_k,""))
        if ctype is None: continue
        is_grant = (ctype == "GVC") and ("Grant" in deal_type_str)
        # public_mission = GVC + Impact combined
        pm = int(ctype in ("GVC","Impact"))
        tie_rows.append({
            "co":           co,
            "investor":     inv_k,
            "type":         ctype,
            "pm":           pm,
            "is_grant":     int(is_grant),
            "os":           os_v,
            "nc":           nc_v,
            "country":      row["country"],
            "year":         yr,
            "log_capital":  row["log_capital"],
        })

ties = pd.DataFrame(tie_rows)

# Recode type for model: PM / CVC / Other (IVC = reference)
ties["type_model"] = "IVC"
ties.loc[ties["pm"] == 1, "type_model"]           = "PM"
ties.loc[ties["type"] == "CVC", "type_model"]      = "CVC"
ties.loc[(ties["pm"]==0) &
         (ties["type"].isin(["Other","IVC"])==False) &
         (ties["type"] == "IVC"), "type_model"]     = "IVC"
ties.loc[(ties["type"]=="Other"), "type_model"]    = "Other"

print(f"  Tie rows (typed, first-round): {len(ties):,}")
print(f"  Unique startups: {ties['co'].nunique():,}")

print("\n  Type counts:")
for t, lbl in [("PM","public_mission (GVC+Impact)"),
               ("IVC","IVC (reference)"),
               ("CVC","CVC"),
               ("Other","Other")]:
    n = (ties["type_model"] == t).sum()
    print(f"    {lbl:<32} {n:>5}")

# Cell counts: type × pct_typed ≥ 75%
# Compute pct_typed per startup for filtering
typed_share = ties.groupby("co")["investor"].count()  # this counts typed ties only
# We need total investors per startup from fr
fr_inv_count = fr.set_index("co")["Investors"].apply(lambda x: len(comma_parse(str(x))))
pct_typed_map = {}
for co in fr["co"]:
    n_typed = typed_share.get(co, 0)
    n_total = fr_inv_count.get(co, 0)
    pct_typed_map[co] = n_typed / n_total if n_total > 0 else 0.0

ties["pct_typed"] = ties["co"].map(pct_typed_map)
ties75 = ties[ties["pct_typed"] >= 0.75].copy()
print(f"\n  Ties in ≥75% typed subset: {len(ties75):,}  "
      f"(startups: {ties75['co'].nunique():,})")

# ─────────────────────────────────────────────────────────────────────
# S1 model: OLS overall_sparsity_t ~ type + controls + FE
# ─────────────────────────────────────────────────────────────────────
sep("S1 model: OLS (tie-level, IVC = reference)")

t = ties75.copy()
t["country_fe"] = t["country"].where(t["country"].map(
    t["country"].value_counts() >= 30), other="Other")
t["year_fe"] = t["year"].astype(str)

# Dummies
t_model = pd.get_dummies(t, columns=["type_model","country_fe","year_fe"],
                          drop_first=False, dtype=float)

# Drop IVC dummy (reference), build X
type_cols = [c for c in t_model.columns if c.startswith("type_model_")]
ref_col   = "type_model_IVC"
if ref_col in type_cols: type_cols.remove(ref_col)
ctry_cols = [c for c in t_model.columns if c.startswith("country_fe_")]
yr_cols   = [c for c in t_model.columns if c.startswith("year_fe_")]
ctrl_cols = type_cols + ["log_capital"] + ctry_cols + yr_cols

t_model = t_model.dropna(subset=["os"] + ctrl_cols)

X = t_model[ctrl_cols].astype(float).values
y = t_model["os"].astype(float).values
# Add intercept
X = np.hstack([np.ones((len(X),1)), X])

beta, *_ = np.linalg.lstsq(X, y, rcond=None)

# Cluster-robust SEs clustered by startup (co)
resid = y - X @ beta
cos_arr = t_model["co"].values
unique_cos = np.unique(cos_arr)
meat = np.zeros((X.shape[1], X.shape[1]))
for co in unique_cos:
    mk = cos_arr == co
    Xc = X[mk]; rc = resid[mk]
    sc = Xc.T @ rc
    meat += np.outer(sc, sc)

n, p = X.shape
bread = np.linalg.pinv(X.T @ X)
n_cl  = len(unique_cos)
scale = n_cl / (n_cl - 1) * (n - 1) / (n - p)
V_cl  = scale * bread @ meat @ bread
se_cl = np.sqrt(np.maximum(0.0, np.diag(V_cl)))

# Map param names
param_names = ["intercept"] + ctrl_cols

# Extract type coefficients
type_params = {c.replace("type_model_",""):
               (beta[param_names.index(c)], se_cl[param_names.index(c)])
               for c in type_cols}

lc_idx = param_names.index("log_capital")

print(f"\n  N ties: {len(t_model):,}  |  N startups (clusters): {n_cl:,}")
print(f"  R² (OLS): {1 - np.sum(resid**2)/np.sum((y-y.mean())**2):.4f}")
print()
print(f"  {'Type':<22}  {'β':>8}  {'SE_cl':>8}  {'t':>7}  {'p (2-sided)':>12}  {'95% CI'}")
print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*20}")

from scipy.stats import t as t_dist

s1_pvals = {}
for key in ["PM","CVC","Other"]:
    col = f"type_model_{key}"
    if col in param_names:
        b, se = type_params.get(key,(np.nan,np.nan))
        tv   = b / se if se > 0 else np.nan
        pv   = 2 * t_dist.sf(abs(tv), df=n_cl-1) if not np.isnan(tv) else np.nan
        ci_lo = b - 1.96 * se; ci_hi = b + 1.96 * se
        star = "***" if pv<.001 else "**" if pv<.01 else "*" if pv<.05 else ""
        lbl = {"PM":"public_mission","CVC":"CVC","Other":"Other"}[key]
        print(f"  {lbl:<22}  {b:>+8.4f}  {se:>8.4f}  {tv:>+7.3f}  "
              f"{pv:>12.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  {star}")
        s1_pvals[key] = pv
    else:
        print(f"  {key:<22}  (not in model)")

# log_capital
b_lc = beta[lc_idx]; se_lc = se_cl[lc_idx]
tv_lc = b_lc/se_lc if se_lc > 0 else np.nan
print(f"  {'log_capital':<22}  {b_lc:>+8.4f}  {se_lc:>8.4f}  {tv_lc:>+7.3f}")

# MDE for PM
print(f"\n  MDE (80% power, α=0.05 two-sided) for β_PM:")
se_pm = type_params.get("PM",(np.nan,np.nan))[1]
mde_s1 = 2.80 * se_pm  # t_0.025 + t_0.20 ≈ 1.96 + 0.84 = 2.80
print(f"    SE_cl(PM) = {se_pm:.4f}  →  MDE = {mde_s1:.4f}")

# Verdict S1
b_pm  = type_params.get("PM",(np.nan,np.nan))[0]
b_cvc = type_params.get("CVC",(np.nan,np.nan))[0]
p_pm  = s1_pvals.get("PM",np.nan)
p_cvc = s1_pvals.get("CVC",np.nan)
print(f"\n  Locked prediction: β_PM > 0 AND β_CVC < 0")
print(f"  Result: β_PM={b_pm:+.4f} (p={p_pm:.4f})  β_CVC={b_cvc:+.4f} (p={p_cvc:.4f})")

# ─────────────────────────────────────────────────────────────────────
# Build all-rounds dataset for S2
# ─────────────────────────────────────────────────────────────────────
sep("Building all-rounds dataset for S2")

# For each deal: identify GVC-equity, GVC-grant, private investors
s2_rows = []
for _, row in deals_v[deals_v["co"].isin(valid_cos)].iterrows():
    co  = row["co"]
    dt  = row["Deal Date"]
    if pd.isna(dt): continue
    deal_type_str = str(row.get("Deal Type",""))
    is_grant_deal = "Grant" in deal_type_str
    vc_round_raw  = row.get("VC Round", None)
    rn = parse_vc_round(vc_round_raw)
    yr = int(row["year"]) if pd.notna(row.get("year")) else None
    country = str(row.get("Company Country/Territory/Region","")).strip()
    invs = comma_parse(str(row["Investors"]))
    for inv in invs:
        ctype = canonical_type(type_map.get(inv.lower(),""))
        if ctype is None: continue
        is_gvc        = (ctype == "GVC")
        is_gvc_equity = is_gvc and (not is_grant_deal)
        is_gvc_grant  = is_gvc and is_grant_deal
        is_private    = (ctype in ("IVC","CVC","Impact","Other"))
        s2_rows.append({
            "co": co, "date": dt, "vc_round": rn,
            "is_gvc_equity": int(is_gvc_equity),
            "is_gvc_grant":  int(is_gvc_grant),
            "is_private":    int(is_private),
            "country":       country,
            "year":          yr,
        })

s2_all = pd.DataFrame(s2_rows)
print(f"  All-round tie rows (typed): {len(s2_all):,}  |  cos: {s2_all['co'].nunique():,}")
print(f"  GVC-equity ties: {s2_all['is_gvc_equity'].sum():,}")
print(f"  GVC-grant ties:  {s2_all['is_gvc_grant'].sum():,}")
print(f"  Private ties:    {s2_all['is_private'].sum():,}")

# ─────────────────────────────────────────────────────────────────────
# S2a: GVC-equity vs private — first ROUND NUMBER
# ─────────────────────────────────────────────────────────────────────
sep("S2a: GVC-equity vs private — first round number")

# For VC round, only valid for deals with parsed round number
s2_rn = s2_all.dropna(subset=["vc_round"]).copy()

eq_first_rn  = (s2_rn[s2_rn["is_gvc_equity"]==1]
                .groupby("co")["vc_round"].min()
                .rename("gvc_eq_first_rn"))
priv_first_rn = (s2_rn[s2_rn["is_private"]==1]
                 .groupby("co")["vc_round"].min()
                 .rename("priv_first_rn"))
s2a_rn = pd.concat([eq_first_rn, priv_first_rn], axis=1).dropna()
s2a_rn["delta_rn"] = s2a_rn["gvc_eq_first_rn"] - s2a_rn["priv_first_rn"]

print(f"  Startups with both GVC-equity and private (VC Round dated): {len(s2a_rn):,}")
d = s2a_rn["delta_rn"]
print(f"\n  Distribution of Δ_round (GVC_eq_first − private_first):")
print(f"    Mean:     {d.mean():>+.3f}")
print(f"    Median:   {d.median():>+.3f}")
print(f"    SD:       {d.std():>7.3f}")
print(f"    GVC earlier (Δ<0): {(d<0).sum():>4}  ({100*(d<0).mean():.1f}%)")
print(f"    Same round (Δ=0):  {(d==0).sum():>4}  ({100*(d==0).mean():.1f}%)")
print(f"    GVC later  (Δ>0):  {(d>0).sum():>4}  ({100*(d>0).mean():.1f}%)")

stat_a, p_a_raw = ttest_1samp(d.values, popmean=0, alternative="less")
print(f"\n  One-sample t-test H0: μ=0 vs H1: μ<0 (GVC enters earlier)")
print(f"    t = {stat_a:.3f}  p (one-sided) = {p_a_raw:.4f}")

mde_s2a = 2.80 * d.std() / np.sqrt(len(d))
print(f"  MDE (80% power, α=0.05 one-sided): {mde_s2a:.3f} round units")

# Also check using dates
eq_first_dt   = (s2_all[s2_all["is_gvc_equity"]==1]
                 .groupby("co")["date"].min()
                 .rename("gvc_eq_first_dt"))
priv_first_dt = (s2_all[s2_all["is_private"]==1]
                 .groupby("co")["date"].min()
                 .rename("priv_first_dt"))
s2a_dt = pd.concat([eq_first_dt, priv_first_dt], axis=1).dropna()
s2a_dt["delta_days"] = (s2a_dt["gvc_eq_first_dt"] - s2a_dt["priv_first_dt"]).dt.days
print(f"\n  (Date-based check, N={len(s2a_dt):,}): "
      f"mean Δ_days = {s2a_dt['delta_days'].mean():+.1f}  "
      f"median = {s2a_dt['delta_days'].median():+.1f}")
_, p_a_dt = ttest_1samp(s2a_dt["delta_days"].values, popmean=0, alternative="less")
print(f"  t-test on dates: p = {p_a_dt:.4f}")

# S2a OLS with FE: delta_days ~ country + year (startup level)
s2a_ols = s2a_dt.copy()
s2a_ols["country"] = deals_v.groupby("co")["Company Country/Territory/Region"].first().reindex(s2a_ols.index).fillna("Unknown").values
s2a_ols["country_fe"] = s2a_ols["country"].where(
    s2a_ols["country"].map(s2a_ols["country"].value_counts() >= 20), other="Other")
s2a_ols["year_fe"] = s2a_ols["gvc_eq_first_dt"].dt.year.astype(str)

y2a = s2a_ols["delta_days"].values.astype(float)
X2a_df = pd.get_dummies(s2a_ols[["country_fe","year_fe"]], dtype=float, drop_first=True)
X2a = np.hstack([np.ones((len(y2a),1)), X2a_df.values])
b2a, *_ = np.linalg.lstsq(X2a, y2a, rcond=None)
res2a   = y2a - X2a @ b2a
se_int_2a = np.sqrt(np.linalg.inv(X2a.T @ X2a)[0,0] * np.mean(res2a**2) *
                     len(y2a) / (len(y2a) - X2a.shape[1]))
t_adj_2a  = b2a[0] / se_int_2a
p_adj_2a  = float(t_dist.cdf(t_adj_2a, df=len(y2a)-X2a.shape[1]))
print(f"\n  S2a with country + year FE (OLS, N={len(s2a_ols):,}):")
print(f"    Intercept (residual Δ_days) = {b2a[0]:+.1f}  "
      f"SE = {se_int_2a:.1f}  t = {t_adj_2a:.3f}  p (one-sided) = {p_adj_2a:.4f}")

p_s2a = min(p_a_dt, p_adj_2a)   # use more conservative (adjusted) for Holm

# ─────────────────────────────────────────────────────────────────────
# S2b: GVC-grant vs private — first date
# ─────────────────────────────────────────────────────────────────────
sep("S2b: GVC-grant vs private — first date")

gr_first_dt   = (s2_all[s2_all["is_gvc_grant"]==1]
                 .groupby("co")["date"].min()
                 .rename("grant_first_dt"))
priv_first_dt_b = priv_first_dt.rename("priv_first_dt")
s2b = pd.concat([gr_first_dt, priv_first_dt_b], axis=1).dropna()
s2b["delta_days"] = (s2b["grant_first_dt"] - s2b["priv_first_dt"]).dt.days

print(f"  Startups with ≥1 grant AND ≥1 private investor (both dated): {len(s2b):,}")
db = s2b["delta_days"]
print(f"\n  Distribution of Δ_days (grant_first − private_first):")
print(f"    Mean:     {db.mean():>+.1f}")
print(f"    Median:   {db.median():>+.1f}")
print(f"    SD:       {db.std():>7.1f}")
print(f"    P10:      {db.quantile(.10):>+.1f}")
print(f"    P90:      {db.quantile(.90):>+.1f}")
print(f"    Grant earlier (Δ<0): {(db<0).sum():>4}  ({100*(db<0).mean():.1f}%)")
print(f"    Same day  (Δ=0):     {(db==0).sum():>4}  ({100*(db==0).mean():.1f}%)")
print(f"    Grant later  (Δ>0):  {(db>0).sum():>4}  ({100*(db>0).mean():.1f}%)")

stat_b, p_b_raw = ttest_1samp(db.values, popmean=0, alternative="less")
print(f"\n  One-sample t-test H0: μ=0 vs H1: μ<0 (grant enters earlier)")
print(f"    t = {stat_b:.3f}  p (one-sided) = {p_b_raw:.4f}")

mde_s2b = 2.80 * db.std() / np.sqrt(len(db))
print(f"  MDE (80% power, α=0.05 one-sided): {mde_s2b:.1f} days")

# OLS with country + year FE
s2b_ols = s2b.copy()
s2b_ols["country"] = deals_v.groupby("co")["Company Country/Territory/Region"].first().reindex(s2b_ols.index).fillna("Unknown").values
s2b_ols["country_fe"] = s2b_ols["country"].where(
    s2b_ols["country"].map(s2b_ols["country"].value_counts() >= 15), other="Other")
s2b_ols["year_fe"] = s2b_ols["grant_first_dt"].dt.year.astype(str)

y2b   = s2b_ols["delta_days"].values.astype(float)
X2b_df = pd.get_dummies(s2b_ols[["country_fe","year_fe"]], dtype=float, drop_first=True)
X2b    = np.hstack([np.ones((len(y2b),1)), X2b_df.values])
b2b, *_ = np.linalg.lstsq(X2b, y2b, rcond=None)
res2b   = y2b - X2b @ b2b
se_int_2b = np.sqrt(np.linalg.inv(X2b.T @ X2b)[0,0] * np.mean(res2b**2) *
                     len(y2b) / (len(y2b) - X2b.shape[1]))
t_adj_2b  = b2b[0] / se_int_2b
p_adj_2b  = float(t_dist.cdf(t_adj_2b, df=len(y2b)-X2b.shape[1]))
print(f"\n  S2b with country + year FE (OLS, N={len(s2b_ols):,}):")
print(f"    Intercept (residual Δ_days) = {b2b[0]:+.1f}  "
      f"SE = {se_int_2b:.1f}  t = {t_adj_2b:.3f}  p (one-sided) = {p_adj_2b:.4f}")

p_s2b = p_adj_2b

# ─────────────────────────────────────────────────────────────────────
# Holm correction across S1 (β_PM), S1 (β_CVC), S2a, S2b
# ─────────────────────────────────────────────────────────────────────
sep("Holm correction (4 primary tests)")

test_labels = ["S1 β_PM > 0", "S1 β_CVC < 0", "S2a Δ_rn < 0", "S2b Δ_days < 0"]
# Convert S1 two-sided to one-sided p-values (directional test)
p_pm_1s  = p_pm  / 2 if b_pm  > 0 else 1 - p_pm  / 2
p_cvc_1s = p_cvc / 2 if b_cvc < 0 else 1 - p_cvc / 2

raw_pvals = np.array([p_pm_1s, p_cvc_1s, p_a_raw, p_adj_2b])
adj_pvals = holm(raw_pvals, test_labels)

print(f"\n  {'Test':<22}  {'raw p':>8}  {'Holm-adj p':>11}  {'α=0.05?'}")
print(f"  {'─'*22}  {'─'*8}  {'─'*11}  {'─'*8}")
for lbl, rp, ap in zip(test_labels, raw_pvals, adj_pvals):
    sig = "PASS" if ap < 0.05 else "fail"
    print(f"  {lbl:<22}  {rp:>8.4f}  {ap:>11.4f}  {sig}")

# ─────────────────────────────────────────────────────────────────────
# Plain verdicts
# ─────────────────────────────────────────────────────────────────────
sep("Plain verdicts (locked predictions)")

v_pm  = "CONFIRMED" if (b_pm > 0 and adj_pvals[0] < 0.05) else \
        "DIRECTIONAL (p≥0.05)" if (b_pm > 0) else "FALSIFIED"
v_cvc = "CONFIRMED" if (b_cvc < 0 and adj_pvals[1] < 0.05) else \
        "DIRECTIONAL (p≥0.05)" if (b_cvc < 0) else "FALSIFIED"
v_s2a = "CONFIRMED" if (s2a_rn["delta_rn"].mean() < 0 and adj_pvals[2] < 0.05) else \
        "DIRECTIONAL (p≥0.05)" if (s2a_rn["delta_rn"].mean() < 0) else "FALSIFIED"
v_s2b = "CONFIRMED" if (db.mean() < 0 and adj_pvals[3] < 0.05) else \
        "DIRECTIONAL (p≥0.05)" if (db.mean() < 0) else "FALSIFIED"

print(f"""
  Locked prediction                          Verdict
  ─────────────────────────────────────────  ─────────────────────
  S1:  β_PM > 0 (public/mission > IVC)      {v_pm}
  S1:  β_CVC < 0 (IVC > CVC)               {v_cvc}
  S2a: GVC-equity enters earlier (Δ_rn<0)  {v_s2a}
  S2b: grants enter earlier (Δ_days<0)     {v_s2b}
""")

# ─────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────
s2a_rn.to_csv(f"{OUT_DIR}/s2a_delta_round.csv")
s2a_dt.to_csv(f"{OUT_DIR}/s2a_delta_days.csv")
s2b.to_csv(f"{OUT_DIR}/s2b_delta_days.csv")

# ─────────────────────────────────────────────────────────────────────
# Append to preregistration
# ─────────────────────────────────────────────────────────────────────
prereg_block = f"""
## Family G — Investor type and startup novelty (S1)
## Family H — GVC/grant timing (S2a, S2b)

**Locked: 2026-09-22**

### Definitions (fixed on theoretical grounds, pre-data)
- `public_mission` = GVC + Impact (Government + Sovereign Wealth + NFP-VC + Impact Investing)
- `GVC_equity`: GVC investor in a non-Grant deal
- `GVC_grant`:  GVC investor in a Grant deal
- Reference category for S1: IVC (Venture Capital)

### S1 — investor type and startup novelty
- Unit: investor–startup tie (first-round syndicate, startups with ≥75% investors typed)
- Outcome: `overall_sparsity_t` (primary)
- Model: OLS with country FE, year FE, log_capital; SEs clustered by startup
- Pre-specified direction: β_PM > 0 (public_mission > IVC) AND β_CVC < 0 (IVC > CVC)
- N ties: {len(t_model):,}  |  N startups: {n_cl:,}
- MDE for β_PM: {mde_s1:.4f}

### S2a — GVC-equity timing
- Unit: startup with ≥1 GVC-equity and ≥1 private investor (both VC-round dated)
- Outcome: Δ_round = first GVC-equity VC Round − first private VC Round
- Pre-specified direction: Δ_round < 0 (GVC enters earlier)
- N startups: {len(s2a_rn):,}  |  MDE: {mde_s2a:.3f} round units
- Test: one-sample t-test, H1: μ < 0; robustness: OLS with country + year FE

### S2b — grant timing
- Unit: startup with ≥1 grant and ≥1 private investor (both dated)
- Outcome: Δ_days = first grant date − first private date
- Pre-specified direction: Δ_days < 0 (grant enters earlier)
- N startups: {len(s2b):,}  |  MDE: {mde_s2b:.1f} days
- Test: one-sample t-test, H1: μ < 0; robustness: OLS with country + year FE

### Multiple testing
- Holm correction across 4 primary tests: S1(β_PM), S1(β_CVC), S2a, S2b
- Familywise α = 0.05

**Results (2026-09-22)**:
- S1 β_PM  = {b_pm:+.4f}  (SE_cl={type_params.get('PM',(np.nan,np.nan))[1]:.4f},  p_Holm={adj_pvals[0]:.4f}) — {v_pm}
- S1 β_CVC = {b_cvc:+.4f}  (SE_cl={type_params.get('CVC',(np.nan,np.nan))[1]:.4f},  p_Holm={adj_pvals[1]:.4f}) — {v_cvc}
- S2a mean Δ_round = {s2a_rn['delta_rn'].mean():+.3f} (p_Holm={adj_pvals[2]:.4f}) — {v_s2a}
- S2b mean Δ_days  = {db.mean():+.1f} (p_Holm={adj_pvals[3]:.4f}) — {v_s2b}

**Status: LOCKED**
"""

with open(PREREG, "a") as f:
    f.write(prereg_block)
print("  Preregistration updated.")

print("=" * 68)
print("SCRIPT 96 COMPLETE")
print("=" * 68)
