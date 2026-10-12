"""
Script 71: Dyadic Tie-Formation Model
======================================
Does investor-startup semantic distance predict who funds whom?

PRE-SPECIFIED (report regardless of outcome):
  P1: semantic_distance NEGATIVELY predicts tie formation
      (closer investor ↔ startup = more likely investment)
  P2: geo_same_country POSITIVELY predicts tie formation
  P3: semantic_distance remains significant after controlling
      for geographic distance and community membership
  FAILURE P1: null or positive coefficient on semantic_distance
  FAILURE P3: semantic_distance becomes null once geo/community added
      → would mean semantic distance is a proxy for geography

UNIT: investor-startup dyad. No startup outcome variables.
"""
import re, warnings, time, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
t0 = time.time()

def banner(s):
    print(f"\n{'='*72}\n{s}\n{'='*72}")

def elapsed():
    return f"{time.time()-t0:.1f}s"

def comma_parse(s):
    if pd.isna(s) or not str(s).strip(): return []
    return [re.sub(r"\s*\(.*?\)\s*$","",p).strip()
            for p in re.split(r",\s*|\n", str(s)) if p.strip()]

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return np.nan
    return float(np.dot(a, b) / (na * nb))

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
banner("LOADING DATA")

# Investor SPECTER2 embeddings (14409 × 768)
emb_matrix = np.load(BASE/"data"/"investor_embeddings.npy")        # float32
inv_names   = pd.read_csv(BASE/"data"/"investor_names.csv")["investor"].str.strip()
inv_emb_map = {n.lower(): emb_matrix[i] for i, n in enumerate(inv_names)}
print(f"  Investor embeddings: {emb_matrix.shape}  ({len(inv_emb_map):,} names)")

# Startup SPECTER2 embeddings (9749 × 768)
with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns = pickle.load(fh)
startup_emb_raw   = np.array(ns["embs_raw"])   # (9749, 768)
startup_emb_names = [n.strip() for n in ns["names"]]
startup_emb_map   = {n: startup_emb_raw[i] for i, n in enumerate(startup_emb_names)}
print(f"  Startup SPECTER2 embeddings: {startup_emb_raw.shape}  ({len(startup_emb_map):,} names)")

# BGE embeddings for robustness (same index)
with open(BASE/"output"/"novelty"/"novelty_bge.pkl","rb") as fh:
    nb = pickle.load(fh)
startup_bge_raw  = np.array(nb["embs_raw"])    # (9749, 1024)
startup_bge_map  = {n: startup_bge_raw[i] for i, n in enumerate(startup_emb_names)}
print(f"  Startup BGE embeddings: {startup_bge_raw.shape}")

# Deals
import sys; sys.path.insert(0, str(BASE))
from load_data import load_deals
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()
deals["Deal Date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals["year"] = deals["Deal Date"].dt.year.fillna(deals["Year"]).astype("Int64")

# CPU company dataset (specter_novelty, log_capital, countries)
cdf = pd.read_csv(BASE/"output"/"cpu"/"cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
co_novelty   = dict(zip(cdf["co"], cdf["specter_novelty"]))
co_country   = dict(zip(cdf["co"], cdf["hq_country"]))
co_region    = dict(zip(cdf["co"], cdf["hq_region"]))
co_log_cap   = dict(zip(cdf["co"], cdf["log_capital"]))

# Investor data (HQ country, total investments)
inv_raw = pd.read_excel(BASE/"data"/"Investors_data.xlsx", skiprows=7)
inv_raw["inv_key"] = inv_raw["Investors"].str.strip().str.lower()
inv_country_map = dict(zip(inv_raw["inv_key"],
    inv_raw["HQ Country/Territory/Region"].fillna("")))
inv_region_map  = dict(zip(inv_raw["inv_key"],
    inv_raw["HQ Global Region"].fillna("")))
inv_total_deals  = dict(zip(inv_raw["inv_key"],
    pd.to_numeric(inv_raw["Total Investments"], errors="coerce").fillna(0)))
print(f"  Investor HQ data: {len(inv_raw):,} rows")

# Network metrics (community)
nm = pd.read_csv(BASE/"output"/"network"/"network_metrics.csv")
nm["inv_key"] = nm["investor"].str.strip().str.lower()
inv_degree_map = dict(zip(nm["inv_key"], nm["degree"]))
inv_comm_map   = dict(zip(nm["inv_key"], nm["community"]))
print(f"  Network metrics: {len(nm):,} investors, {nm['community'].nunique()} communities")

# ──────────────────────────────────────────────────────────────
# STEP 1: BUILD THE RISK SET
# ──────────────────────────────────────────────────────────────
banner("STEP 1: RISK SET CONSTRUCTION")
print("  5 counterfactuals per realised tie, matched on year")
print("  Counterfactual must: be active in year t, not invest in j, have embedding")

N_COUNTER = 5
rng = np.random.default_rng(42)

# Identify first-round deals (earliest deal per company)
deals_sorted = deals.dropna(subset=["year","Investors"]).copy()
deals_sorted["year"] = deals_sorted["year"].astype(int)

first_round = (deals_sorted.sort_values("Deal Date")
                            .groupby("co")
                            .first()
                            .reset_index()[["co","year","Investors",
                                            "Company Country/Territory/Region",
                                            "HQ Global Region"]])
print(f"\n  Companies in deals (first round): {len(first_round):,}")

# Map of which investors were active in each year (at least 1 deal)
year_active = {}
for _, row in deals_sorted.iterrows():
    y = row["year"]
    if y not in year_active: year_active[y] = set()
    for inv in comma_parse(row["Investors"]):
        year_active[y].add(inv.lower())

# Build dyads
rows_real   = []
rows_fake   = []
drop_no_startup_emb = 0
drop_no_inv_emb     = 0
drop_no_active      = 0

for _, fr in first_round.iterrows():
    co  = fr["co"]
    yr  = fr["year"]
    inv_str = fr["Investors"]
    co_ctry = str(fr.get("Company Country/Territory/Region","")).strip()
    co_rgn  = str(fr.get("HQ Global Region","")).strip()

    # startup embedding check
    if co not in startup_emb_map:
        drop_no_startup_emb += 1
        continue

    investors = comma_parse(inv_str)
    if not investors: continue

    # pool of active investors this year with embeddings
    active_this_yr = year_active.get(yr, set())
    actual_set     = {i.lower() for i in investors}
    pool = [i for i in active_this_yr
            if i not in actual_set
            and i in inv_emb_map]

    if len(pool) < N_COUNTER:
        drop_no_active += 1
        continue

    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map:
            drop_no_inv_emb += 1
            continue
        rows_real.append({
            "co": co, "investor": inv_k,
            "co_country": co_ctry, "co_region": co_rgn,
            "year": yr, "tied": 1
        })
        # sample 5 counterfactuals
        chosen = rng.choice(pool, size=N_COUNTER, replace=False)
        for fake_inv in chosen:
            rows_fake.append({
                "co": co, "investor": fake_inv,
                "co_country": co_ctry, "co_region": co_rgn,
                "year": yr, "tied": 0
            })

dyads = pd.DataFrame(rows_real + rows_fake).reset_index(drop=True)
N_real  = dyads["tied"].sum()
N_total = len(dyads)

print(f"\n  Realised ties (with investor embedding):  {N_real:,}")
print(f"  Total dyads (real + {N_COUNTER}× counterfactual):  {N_total:,}")
print(f"  Strata (unique startups):                 {dyads['co'].nunique():,}")
print(f"\n  Ties dropped:")
print(f"    No startup embedding (SPECTER2):   {drop_no_startup_emb:,}")
print(f"    No investor embedding (SPECTER2):  {drop_no_inv_emb:,}")
print(f"    Pool < {N_COUNTER} active investors:         {drop_no_active:,}")

# ──────────────────────────────────────────────────────────────
# STEP 2: DYAD VARIABLES
# ──────────────────────────────────────────────────────────────
banner("STEP 2: COMPUTE DYAD VARIABLES")

print("  Computing semantic distances (SPECTER2)...")
sem_dists = []
for _, row in dyads.iterrows():
    sv = startup_emb_map[row["co"]]
    iv = inv_emb_map[row["investor"]]
    cs = cosine_sim(sv, iv)
    sem_dists.append(1.0 - cs if not np.isnan(cs) else np.nan)
dyads["semantic_distance"] = sem_dists

print("  Computing BGE semantic distances for robustness...")
sem_bge = []
for _, row in dyads.iterrows():
    sv = startup_bge_map.get(row["co"])
    iv = inv_emb_map.get(row["investor"])  # note: investor embs are 768-dim, BGE is 1024
    if sv is None or iv is None:
        sem_bge.append(np.nan)
    else:
        # investor embeddings are SPECTER2 768-dim; startup BGE is 1024-dim — INCOMPATIBLE
        # flag this; report in robustness that cross-model comparison is not possible
        sem_bge.append(np.nan)
dyads["semantic_distance_bge"] = sem_bge
# NOTE: investor_embeddings.npy is SPECTER2 (768-dim); startup BGE is 1024-dim.
# Cross-model cosine is undefined. Step 5 robustness uses startup SPECTER2 only.
print("  NOTE: investor_embeddings.npy = SPECTER2 (768-dim). BGE startup = 1024-dim.")
print("        Cross-model cosine undefined. Robustness Step 5 uses SPECTER2 only.")

# Geographic variables
dyads["inv_country"] = dyads["investor"].map(inv_country_map).fillna("")
dyads["inv_region"]  = dyads["investor"].map(inv_region_map).fillna("")
dyads["geo_same_country"] = (
    (dyads["co_country"] != "") & (dyads["inv_country"] != "") &
    (dyads["co_country"] == dyads["inv_country"])).astype(float)
dyads["geo_same_region"] = (
    (dyads["co_region"] != "") & (dyads["inv_region"] != "") &
    (dyads["co_region"] == dyads["inv_region"])).astype(float)

# Investor-level variables
dyads["investor_degree"]      = dyads["investor"].map(inv_degree_map).fillna(0)
dyads["log_investor_degree"]  = np.log1p(dyads["investor_degree"])
dyads["investor_total_deals"] = dyads["investor"].map(inv_total_deals).fillna(0)
dyads["log_investor_deals"]   = np.log1p(dyads["investor_total_deals"])

# Startup community (lead investor community) for same_community_as_lead
# Use all investors in the first round; their modal community = "startup community"
co_lead_comm = {}
for _, fr in first_round.iterrows():
    co = fr["co"]
    investors = comma_parse(fr["Investors"])
    comms = [inv_comm_map.get(i.lower()) for i in investors if inv_comm_map.get(i.lower()) is not None]
    if comms:
        from collections import Counter
        modal_comm = Counter(comms).most_common(1)[0][0]
        co_lead_comm[co] = modal_comm

dyads["co_community"]  = dyads["co"].map(co_lead_comm)
dyads["inv_community"] = dyads["investor"].map(inv_comm_map)
dyads["same_community_as_lead"] = (
    dyads["co_community"].notna() & dyads["inv_community"].notna() &
    (dyads["co_community"] == dyads["inv_community"])).astype(float)

# Startup variables
dyads["startup_novelty"] = dyads["co"].map(co_novelty)
dyads["startup_log_cap"] = dyads["co"].map(co_log_cap)

print(f"\n  Variable distributions (full dyad set, N={len(dyads):,}):")
for col in ["semantic_distance","geo_same_country","geo_same_region",
            "log_investor_degree","log_investor_deals",
            "same_community_as_lead","startup_novelty","startup_log_cap"]:
    s = dyads[col].dropna()
    print(f"    {col:<30} N={len(s):>6,}  mean={s.mean():+.4f}  sd={s.std():.4f}  "
          f"min={s.min():+.4f}  max={s.max():+.4f}")

# SD check on semantic_distance
sd_sem = dyads["semantic_distance"].std()
print(f"\n  *** SD CHECK: semantic_distance SD = {sd_sem:.4f} ***")
if sd_sem < 0.02:
    print("  !!! WARNING: SD < 0.02 — distribution collapsed; model is likely untestable !!!")
else:
    print("  OK: SD sufficient for modelling")

# ──────────────────────────────────────────────────────────────
# STEP 3: CONDITIONAL LOGIT MODELS
# ──────────────────────────────────────────────────────────────
banner("STEP 3: CONDITIONAL LOGIT MODELS (stratified by startup)")

# Drop strata with 0 variation in outcome (all 0 or all 1)
strata_ok = dyads.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
dyads_m = dyads[strata_ok & dyads["semantic_distance"].notna()].copy()
print(f"\n  Modelling sample: {len(dyads_m):,} dyads, {dyads_m['co'].nunique():,} strata")

# Assign integer stratum IDs
co_ids = {co: i for i, co in enumerate(dyads_m["co"].unique())}
dyads_m["stratum"] = dyads_m["co"].map(co_ids)

# Standardise continuous predictors for coefficient comparison
def std_col(df, col):
    s = df[col].dropna()
    mu, sd = s.mean(), s.std()
    if sd < 1e-10: return df[col].copy()
    return (df[col] - mu) / sd

dyads_m["sem_dist_z"]      = std_col(dyads_m, "semantic_distance")
dyads_m["log_deg_z"]       = std_col(dyads_m, "log_investor_degree")
dyads_m["log_deals_z"]     = std_col(dyads_m, "log_investor_deals")
dyads_m["novelty_z"]       = std_col(dyads_m, "startup_novelty")
dyads_m["log_cap_z"]       = std_col(dyads_m, "startup_log_cap")

def run_clogit(df, xvars, label):
    """Fit conditional logit; return result summary row."""
    sub = df[["tied","stratum"] + xvars].dropna().copy()
    n_obs    = len(sub)
    n_strata = sub["stratum"].nunique()
    if n_obs < 100 or n_strata < 10:
        print(f"  {label}: too few obs ({n_obs}) or strata ({n_strata}), skip")
        return None
    y = sub["tied"].values
    X = sub[xvars].values
    groups = sub["stratum"].values
    try:
        model = ConditionalLogit(y, X, groups=groups)
        res   = model.fit(method="bfgs", maxiter=500, disp=False)
        return res, xvars, n_obs, n_strata
    except Exception as e:
        print(f"  {label}: ERROR — {e}")
        return None

def print_model(label, res_tup):
    if res_tup is None: return
    res, xvars, n_obs, n_strata = res_tup
    print(f"\n  {label}  (N dyads={n_obs:,}  N strata={n_strata:,})")
    print(f"  {'Variable':<30} {'β':>8} {'SE':>8} {'p':>8} {'sig':>4} {'stdβ':>8}")
    print(f"  {'-'*72}")
    # standardised beta = beta * sd_x (already standardised for _z vars)
    for i, var in enumerate(xvars):
        b   = res.params[i]
        se  = res.bse[i]
        p   = res.pvalues[i]
        # stdβ: if var ends in _z already standardised; else note
        stdb = b  # already standardised
        print(f"  {var:<30} {b:>+8.4f} {se:>8.4f} {p:>8.4f} {sig(p):>4} {stdb:>+8.4f}")
    ll = getattr(res, 'llf', np.nan)
    print(f"  Log-likelihood: {ll:.2f}" if not np.isnan(ll) else "")

models = {}

# M1: semantic_distance only
m1_vars = ["sem_dist_z"]
models["M1"] = run_clogit(dyads_m, m1_vars, "M1")
print_model("M1: semantic distance only", models["M1"])

# M2: + geographic
m2_vars = ["sem_dist_z","geo_same_country","geo_same_region"]
models["M2"] = run_clogit(dyads_m, m2_vars, "M2")
print_model("M2: + geographic distance", models["M2"])

# M3: + investor activity
m3_vars = ["sem_dist_z","geo_same_country","geo_same_region",
           "log_deg_z","log_deals_z"]
models["M3"] = run_clogit(dyads_m, m3_vars, "M3")
print_model("M3: + investor degree + total deals", models["M3"])

# M4: + community membership
m4_vars = ["sem_dist_z","geo_same_country","geo_same_region",
           "log_deg_z","log_deals_z","same_community_as_lead"]
models["M4"] = run_clogit(dyads_m, m4_vars, "M4")
print_model("M4: + same_community_as_lead (FULL)", models["M4"])

# ──────────────────────────────────────────────────────────────
# STEP 4: DEGREE DEPENDENCE CHECK
# ──────────────────────────────────────────────────────────────
banner("STEP 4: DEGREE DEPENDENCE CHECK")

for col, label in [("log_investor_degree","log(degree)"),
                   ("log_investor_deals","log(total deals)")]:
    r = dyads_m["semantic_distance"].corr(dyads_m[col])
    print(f"  corr(semantic_distance, {label}): r = {r:+.4f}")

print("\n  M1 within investor-activity strata (by lifetime deals):")
strata_def = [(1,5,"1-5 deals"),(6,20,"6-20 deals"),
              (21,100,"21-100 deals"),(101,99999,"101+ deals")]
for lo, hi, s_label in strata_def:
    mask = (dyads_m["investor_total_deals"]>=lo) & (dyads_m["investor_total_deals"]<=hi)
    sub = dyads_m[mask].copy()
    # re-number strata within subset
    ok = sub.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
    sub = sub[ok].copy()
    sub["stratum"] = pd.factorize(sub["co"])[0]
    n_s = len(sub); n_st = sub["stratum"].nunique()
    if n_s < 50 or n_st < 5:
        print(f"    {s_label}: N={n_s} — too small"); continue
    try:
        res_s = ConditionalLogit(sub["tied"].values,
                                 sub["sem_dist_z"].values,
                                 groups=sub["stratum"].values).fit(
            method="bfgs", maxiter=500, disp=False)
        b = res_s.params[0]; p = res_s.pvalues[0]
        print(f"    {s_label:<15} N dyads={n_s:>6,}  N strata={n_st:>5,}  "
              f"β={b:+.4f}  p={p:.4f} {sig(p)}")
    except Exception as e:
        print(f"    {s_label}: ERROR — {e}")

# ──────────────────────────────────────────────────────────────
# STEP 5: ROBUSTNESS
# ──────────────────────────────────────────────────────────────
banner("STEP 5: ROBUSTNESS")

# 5a: BGE robustness — cross-model cosine is undefined (768 vs 1024 dims)
print("\n  5a. BGE/MiniLM robustness:")
print("      investor_embeddings.npy is SPECTER2 (768-dim).")
print("      Startup BGE embeddings are 1024-dim.")
print("      Cross-model cosine undefined → BGE robustness NOT feasible.")
print("      RECOMMENDATION: Re-embed investors with BGE to enable this check.")

# 5b: 10 counterfactuals per tie
print("\n  5b. Robustness with N_COUNTER=10:")
N_COUNTER2 = 10
rows_r2 = []
rows_f2 = []
for _, fr in first_round.iterrows():
    co  = fr["co"]
    yr  = fr["year"]
    inv_str = fr["Investors"]
    co_ctry = str(fr.get("Company Country/Territory/Region","")).strip()
    co_rgn  = str(fr.get("HQ Global Region","")).strip()
    if co not in startup_emb_map: continue
    investors = comma_parse(inv_str)
    if not investors: continue
    active_this_yr = year_active.get(yr, set())
    actual_set     = {i.lower() for i in investors}
    pool = [i for i in active_this_yr if i not in actual_set and i in inv_emb_map]
    if len(pool) < N_COUNTER2: continue
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map: continue
        rows_r2.append({"co":co,"investor":inv_k,"co_country":co_ctry,
                        "co_region":co_rgn,"year":yr,"tied":1})
        chosen = rng.choice(pool, size=N_COUNTER2, replace=False)
        for fi in chosen:
            rows_f2.append({"co":co,"investor":fi,"co_country":co_ctry,
                            "co_region":co_rgn,"year":yr,"tied":0})

dyads2 = pd.DataFrame(rows_r2+rows_f2).reset_index(drop=True)
dyads2["inv_country"] = dyads2["investor"].map(inv_country_map).fillna("")
dyads2["inv_region"]  = dyads2["investor"].map(inv_region_map).fillna("")
dyads2["geo_same_country"] = (
    (dyads2["co_country"]!="") & (dyads2["inv_country"]!="") &
    (dyads2["co_country"]==dyads2["inv_country"])).astype(float)
dyads2["geo_same_region"] = (
    (dyads2["co_region"]!="") & (dyads2["inv_region"]!="") &
    (dyads2["co_region"]==dyads2["inv_region"])).astype(float)

sem2 = []
for _, row in dyads2.iterrows():
    sv = startup_emb_map.get(row["co"]); iv = inv_emb_map.get(row["investor"])
    if sv is None or iv is None: sem2.append(np.nan); continue
    cs = cosine_sim(sv, iv)
    sem2.append(1.0 - cs if not np.isnan(cs) else np.nan)
dyads2["semantic_distance"]    = sem2
dyads2["log_investor_degree"]  = np.log1p(dyads2["investor"].map(inv_degree_map).fillna(0))
dyads2["log_investor_deals"]   = np.log1p(dyads2["investor"].map(inv_total_deals).fillna(0))
dyads2["sem_dist_z"]  = std_col(dyads2, "semantic_distance")
dyads2["log_deg_z"]   = std_col(dyads2, "log_investor_degree")
dyads2["log_deals_z"] = std_col(dyads2, "log_investor_deals")

ok2 = dyads2.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
dyads2_m = dyads2[ok2 & dyads2["semantic_distance"].notna()].copy()
dyads2_m["stratum"] = pd.factorize(dyads2_m["co"])[0]
print(f"      10-counterfactual sample: {len(dyads2_m):,} dyads, {dyads2_m['co'].nunique():,} strata")

res_rob = run_clogit(dyads2_m, ["sem_dist_z","geo_same_country","geo_same_region",
                                  "log_deg_z","log_deals_z"], "M3 (10 counter)")
print_model("M3 replicated with 10 counterfactuals", res_rob)

if models["M3"] is not None and res_rob is not None:
    b5 = models["M3"][0].params[0]; b10 = res_rob[0].params[0]
    print(f"\n      Comparison M3 sem_dist_z: 5-counter β={b5:+.4f}  10-counter β={b10:+.4f}  "
          f"Δ={b10-b5:+.4f}")

# 5c: MDE for semantic coefficient
print("\n  5c. MDE for semantic_distance coefficient:")
N_dy = len(dyads_m) if dyads_m is not None else 0
if N_dy > 0:
    se_approx = models["M3"][0].bse[0] if models["M3"] is not None else np.nan
    mde_95 = 1.96 * se_approx if not np.isnan(se_approx) else np.nan
    print(f"      N dyads in M3 sample:  {N_dy:,}")
    print(f"      SE(sem_dist_z) in M3:  {se_approx:.4f}")
    print(f"      MDE (90% power, α=0.05): β ≥ {mde_95*1.3:.4f}  "
          f"(≈ {mde_95*1.3/dyads_m['semantic_distance'].std():.4f} raw SD units)")

# ──────────────────────────────────────────────────────────────
# STEP 6: DESCRIPTIVE
# ──────────────────────────────────────────────────────────────
banner("STEP 6: DESCRIPTIVE — REALISED vs COUNTERFACTUAL SEMANTIC DISTANCE")

real_sd  = dyads_m[dyads_m["tied"]==1]["semantic_distance"].dropna()
fake_sd  = dyads_m[dyads_m["tied"]==0]["semantic_distance"].dropna()
from scipy import stats
t_stat, t_p = stats.ttest_ind(real_sd, fake_sd, equal_var=False)
diff = real_sd.mean() - fake_sd.mean()
se_diff = np.sqrt(real_sd.var()/len(real_sd) + fake_sd.var()/len(fake_sd))

print(f"\n  Realised ties:       N={len(real_sd):>6,}  mean={real_sd.mean():.4f}  sd={real_sd.std():.4f}")
print(f"  Counterfactuals:     N={len(fake_sd):>6,}  mean={fake_sd.mean():.4f}  sd={fake_sd.std():.4f}")
print(f"  Difference (real-counter): {diff:+.4f}  SE={se_diff:.4f}  "
      f"t={t_stat:.3f}  p={t_p:.4f} {sig(t_p)}")
print(f"\n  95% CI on gap: [{diff-1.96*se_diff:+.4f}, {diff+1.96*se_diff:+.4f}]")

print(f"\n  Distribution (quantiles):")
for q in [5,10,25,50,75,90,95]:
    r_q = np.percentile(real_sd, q); f_q = np.percentile(fake_sd, q)
    print(f"    {q}th pct:  realised={r_q:.4f}  counter={f_q:.4f}  gap={r_q-f_q:+.4f}")

print(f"\n  ASCII density (semantic_distance, bins of 0.025):")
bins_d = np.arange(0, 1.025, 0.025)
rc, _ = np.histogram(real_sd, bins=bins_d, density=True)
fc, _ = np.histogram(fake_sd, bins=bins_d, density=True)
print(f"  {'bin':>8}  {'realised':>30}  {'counter':>30}")
for i in range(len(bins_d)-1):
    if rc[i]==0 and fc[i]==0: continue
    rb = "█" * int(rc[i]*15); fb = "█" * int(fc[i]*15)
    print(f"  {bins_d[i]:.3f}    {rb:<30}  {fb:<30}")

# ──────────────────────────────────────────────────────────────
# VERDICTS
# ──────────────────────────────────────────────────────────────
banner("VERDICTS")

def extract_sem_coef(res_tup):
    if res_tup is None: return np.nan, np.nan
    res, xvars, n_obs, n_strata = res_tup
    idx = xvars.index("sem_dist_z")
    return res.params[idx], res.pvalues[idx]

b_m1, p_m1 = extract_sem_coef(models["M1"])
b_m3, p_m3 = extract_sem_coef(models["M3"])
b_m4, p_m4 = extract_sem_coef(models["M4"])

print(f"\n  M1 semantic_distance_z: β={b_m1:+.4f}  p={p_m1:.4f} {sig(p_m1)}")
print(f"  M3 semantic_distance_z: β={b_m3:+.4f}  p={p_m3:.4f} {sig(p_m3)}")
print(f"  M4 semantic_distance_z: β={b_m4:+.4f}  p={p_m4:.4f} {sig(p_m4)}")

print(f"\n  P1 (semantic distance → NEGATIVE tie probability):")
if b_m1 < 0 and p_m1 < 0.05:
    print(f"    SUPPORTED  β={b_m1:+.4f}  p={p_m1:.4f}")
elif p_m1 >= 0.05:
    print(f"    FAILS (null): β={b_m1:+.4f}  p={p_m1:.4f}")
else:
    print(f"    FAILS (wrong sign): β={b_m1:+.4f}  p={p_m1:.4f}")

print(f"\n  P2 (geo_same_country → POSITIVE):")
def extract_geo(res_tup):
    if res_tup is None: return np.nan, np.nan
    res, xvars, *_ = res_tup
    if "geo_same_country" not in xvars: return np.nan, np.nan
    idx = xvars.index("geo_same_country")
    return res.params[idx], res.pvalues[idx]
bg, pg = extract_geo(models["M2"])
if bg > 0 and pg < 0.05:
    print(f"    SUPPORTED  β={bg:+.4f}  p={pg:.4f}")
elif pg >= 0.05:
    print(f"    FAILS (null): β={bg:+.4f}  p={pg:.4f}")
else:
    print(f"    FAILS (wrong sign): β={bg:+.4f}  p={pg:.4f}")

print(f"\n  P3 (semantic_distance survives geo + community controls):")
if b_m4 < 0 and p_m4 < 0.05:
    print(f"    SUPPORTED  β M4={b_m4:+.4f}  p={p_m4:.4f}")
    print(f"    Semantic distance is NOT merely a proxy for geography/community")
elif p_m4 >= 0.05:
    attn = b_m1 - b_m4 if not np.isnan(b_m1) else np.nan
    print(f"    FAILS: β attenuates to {b_m4:+.4f} (p={p_m4:.4f})")
    print(f"    Attenuation from M1: {attn:+.4f}")
    print(f"    Semantic distance may be a proxy for geography or community structure")

print(f"\n  Raw descriptive gap: realised ties have semantic_distance "
      f"{'lower' if diff < 0 else 'higher'} by {abs(diff):.4f} (p={t_p:.4f} {sig(t_p)})")
print(f"\n  [{elapsed()}]")
