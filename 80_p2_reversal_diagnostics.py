"""
Script 80: Settle the P2 Reversal
===================================
Three diagnostics on the sd×OS sign reversal from Script 79r:
  1. Add sem_dist_z² to the Step 4 model — does sd×OS survive?
  2. Add has_prior_network and its interaction with sd×OS —
     is the reversal driven by new entrants?
  3. Report corr(log_deg_z, log_deals_z).
All reported regardless of outcome.
"""
import re, warnings, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.stats import spearmanr, pearsonr
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
t0   = time.time()

def elapsed(): return f"{time.time()-t0:.1f}s"
def banner(s): print(f"\n{'='*72}\n{s}\n{'='*72}")
def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."
def comma_parse(s):
    if pd.isna(s) or not str(s).strip(): return []
    return [re.sub(r"\s*\(.*?\)\s*$","",p).strip()
            for p in re.split(r",\s*|\n", str(s)) if p.strip()]
def std_col(df, col):
    s = df[col].dropna(); mu, sd = s.mean(), s.std()
    return (df[col] - mu) / sd if sd > 1e-10 else df[col].copy()
def holm(ps):
    ps = np.array(ps, dtype=float); m = len(ps); o = np.argsort(ps); adj = np.empty(m)
    adj[o] = [min(1.0, ps[o[i]]*(m-i)) for i in range(m)]
    for i in range(1, m): adj[o[i]] = max(adj[o[i]], adj[o[i-1]])
    return adj

# ──────────────────────────────────────────────────────────────
# LOAD (lean — no diagnostics printing)
# ──────────────────────────────────────────────────────────────
banner("LOADING DATA")
import sys; sys.path.insert(0, str(BASE))
from load_data import load_deals

deals = load_deals()
deals["co"]        = deals["Companies"].str.strip()
deals["Deal Date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals["year"]      = deals["Deal Date"].dt.year.fillna(deals["Year"]).astype("Int64")
deals_v = deals.dropna(subset=["year","Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

first_round = (deals_v.sort_values("Deal Date")
               .groupby("co").first()
               .reset_index()
               [["co","year","Investors",
                 "Company Country/Territory/Region","HQ Global Region"]])

emb_matrix    = np.load(BASE/"data"/"investor_embeddings.npy")
inv_names_ser = pd.read_csv(BASE/"data"/"investor_names.csv")["investor"].str.strip()
inv_emb_map   = {n.lower(): emb_matrix[i] for i, n in enumerate(inv_names_ser)}

with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns_pkl = pickle.load(fh)
startup_names_all = [n.strip() for n in ns_pkl["names"]]
embs_raw = np.array(ns_pkl["embs_raw"], dtype=np.float32)
norms    = np.linalg.norm(embs_raw, axis=1, keepdims=True)
embs_l2  = embs_raw / np.where(norms < 1e-10, 1.0, norms)
startup_emb_map = {n: embs_l2[i] for i, n in enumerate(startup_names_all)}

inv_raw = pd.read_excel(BASE/"data"/"Investors_data.xlsx", skiprows=7)
inv_raw["inv_key"] = inv_raw["Investors"].str.strip().str.lower()
inv_country_map = dict(zip(inv_raw["inv_key"], inv_raw["HQ Country/Territory/Region"].fillna("")))
inv_region_map  = dict(zip(inv_raw["inv_key"], inv_raw["HQ Global Region"].fillna("")))

years_needed = sorted(first_round["year"].dropna().unique().astype(int))
min_yr, max_yr = int(min(years_needed)), int(max(years_needed))
print(f"  First rounds: {len(first_round):,}  [{elapsed()}]")

# ──────────────────────────────────────────────────────────────
# TIME-AWARE NETWORK (lean)
# ──────────────────────────────────────────────────────────────
banner("BUILDING TIME-AWARE NETWORK")

edges_by_year = defaultdict(lambda: defaultdict(float))
for _, row in deals_v.iterrows():
    yr   = int(row["year"])
    invs = [x.lower() for x in comma_parse(row["Investors"])]
    for a, b in combinations(invs, 2):
        u, v = (a,b) if a<b else (b,a)
        edges_by_year[yr][(u,v)] += 1

inv_deals_before = defaultdict(lambda: defaultdict(int))
for _, row in deals_v.iterrows():
    yr = int(row["year"])
    for inv in comma_parse(row["Investors"]):
        inv_k = inv.lower()
        for fy in range(yr+1, max_yr+3):
            inv_deals_before[fy][inv_k] += 1

lagged_comm   = {}
lagged_degree = {}
cumulative_edges = defaultdict(float)

for yr in range(min_yr-1, max_yr+2):
    if yr in edges_by_year:
        for (u,v), w in edges_by_year[yr].items():
            cumulative_edges[(u,v)] += w
    G = nx.Graph()
    for (u,v), w in cumulative_edges.items(): G.add_edge(u, v, weight=w)
    if G.number_of_edges() < 10:
        lagged_comm[yr+1] = {}; lagged_degree[yr+1] = {}; continue
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    lagged_comm[yr+1]   = partition
    lagged_degree[yr+1] = dict(G.degree())

print(f"  Network built  [{elapsed()}]")

# ──────────────────────────────────────────────────────────────
# TIME-AWARE NOVELTY (lean)
# ──────────────────────────────────────────────────────────────
banner("TIME-AWARE NOVELTY")

from sklearn.neighbors import NearestNeighbors

co_to_year     = dict(zip(first_round["co"], first_round["year"].astype(int)))
embeddable_cos = [co for co in startup_names_all if co in co_to_year]
embeddable_yrs = np.array([co_to_year[co] for co in embeddable_cos], dtype=int)
embeddable_emb = np.array([startup_emb_map[co] for co in embeddable_cos], dtype=np.float32)
N_emb = len(embeddable_cos)

sort_idx   = np.argsort(embeddable_yrs, kind="stable")
sorted_cos = [embeddable_cos[i] for i in sort_idx]
sorted_yrs = embeddable_yrs[sort_idx]
sorted_emb = embeddable_emb[sort_idx]

MIN_PRIOR = 100
rows_tanov = []
prior_cos, prior_embs = [], []

for yr in sorted(set(sorted_yrs)):
    yr_mask = sorted_yrs == yr
    yr_cos  = [sorted_cos[i] for i in range(N_emb) if yr_mask[i]]
    yr_embs = sorted_emb[yr_mask]
    n_prior = len(prior_cos)

    if n_prior >= MIN_PRIOR:
        prior_arr = np.array(prior_embs, dtype=np.float32)
        k_max = min(51, n_prior)
        nn = NearestNeighbors(n_neighbors=k_max, metric="cosine",
                              algorithm="brute", n_jobs=-1).fit(prior_arr)
        dists, _ = nn.kneighbors(yr_embs, n_neighbors=k_max)
        sims = 1.0 - dists
        k5  = min(5,  n_prior)
        k50 = min(50, n_prior)
        for j, co in enumerate(yr_cos):
            rows_tanov.append({"co": co, "year": yr,
                               "loc5_t":  float(sims[j,:k5].mean()),
                               "reg50_t": float(sims[j,:k50].mean()),
                               "ok": True})
    else:
        for co in yr_cos:
            rows_tanov.append({"co": co, "year": yr,
                               "loc5_t": np.nan, "reg50_t": np.nan, "ok": False})
    prior_cos.extend(yr_cos); prior_embs.extend(yr_embs.tolist())

tanov_df = pd.DataFrame(rows_tanov)
tanov_ok = tanov_df[tanov_df["ok"]].copy()

# Within-year standardise
for col_z, col_raw in [("z_loc","loc5_t"),("z_reg","reg50_t")]:
    tanov_ok[col_z] = np.nan
    for yr, grp in tanov_ok.groupby("year"):
        v = grp[col_raw].dropna()
        if len(v) < 3: continue
        mu, sd = v.mean(), v.std()
        if sd < 1e-12: continue
        tanov_ok.loc[grp.index, col_z] = (grp[col_raw] - mu) / sd

tanov_ok["overall_sparsity_t"] = -(tanov_ok["z_loc"] + tanov_ok["z_reg"]) / 2
tanov_ok["niche_cohesion_t"]   =   tanov_ok["z_loc"] - tanov_ok["z_reg"]
co_os_t = dict(zip(tanov_ok["co"], tanov_ok["overall_sparsity_t"]))
co_nc_t = dict(zip(tanov_ok["co"], tanov_ok["niche_cohesion_t"]))
print(f"  Time-aware novelty: {len(tanov_ok):,} startups  [{elapsed()}]")

# ──────────────────────────────────────────────────────────────
# BUILD DYADS
# ──────────────────────────────────────────────────────────────
banner("BUILDING DYADS")

RNG = np.random.default_rng(42)
year_active = defaultdict(set)
for _, row in deals_v.iterrows():
    for inv in comma_parse(row["Investors"]):
        year_active[int(row["year"])].add(inv.lower())

rows_real, rows_fake = [], []
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None or co not in startup_emb_map or co not in co_os_t: continue
    investors = comma_parse(str(fr["Investors"]))
    if not investors: continue
    actual_set = {i.lower() for i in investors}
    pool = [i for i in year_active.get(yr,set())
            if i not in actual_set and i in inv_emb_map]
    if len(pool) < 5: continue
    co_ctry = str(fr.get("Company Country/Territory/Region","")).strip()
    co_rgn  = str(fr.get("HQ Global Region","")).strip()
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map: continue
        meta = {"co":co,"investor":inv_k,"co_country":co_ctry,
                "co_region":co_rgn,"year":yr,"tied":1}
        rows_real.append(meta)
        for fi in RNG.choice(pool, 5, replace=False):
            rows_fake.append({**meta,"investor":fi,"tied":0})

dyads = pd.DataFrame(rows_real + rows_fake).reset_index(drop=True)
print(f"  Dyads: {len(dyads):,}  real: {dyads['tied'].sum():,}  "
      f"strata: {dyads['co'].nunique():,}  [{elapsed()}]")

# ── Compute variables ─────────────────────────────────────────
sem = []
for _, row in dyads.iterrows():
    sv = startup_emb_map.get(row["co"]); iv = inv_emb_map.get(row["investor"])
    if sv is None or iv is None: sem.append(np.nan); continue
    na = np.linalg.norm(sv); nb = np.linalg.norm(iv)
    cs = float(np.dot(sv,iv)/(na*nb)) if na>1e-10 and nb>1e-10 else np.nan
    sem.append(1.0-cs if not np.isnan(cs) else np.nan)
dyads["semantic_distance"] = sem
dyads["inv_country"] = dyads["investor"].map(inv_country_map).fillna("")
dyads["inv_region"]  = dyads["investor"].map(inv_region_map).fillna("")
dyads["geo_same_country"] = ((dyads["co_country"]!="")&(dyads["inv_country"]!="")&
    (dyads["co_country"]==dyads["inv_country"])).astype(float)
dyads["geo_same_region"]  = ((dyads["co_region"]!="")&(dyads["inv_region"]!="")&
    (dyads["co_region"]==dyads["inv_region"])).astype(float)

co_lead_comm_t = {}
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    cm = lagged_comm.get(yr,{})
    comms = [cm.get(i.lower()) for i in comma_parse(str(fr["Investors"])) if cm.get(i.lower()) is not None]
    co_lead_comm_t[co] = Counter(comms).most_common(1)[0][0] if comms else None

sc_t=[]; log_deg_t=[]; log_deals_t=[]
for _, row in dyads.iterrows():
    yr=int(row["year"]); inv_k=row["investor"]; co=row["co"]
    cm  = lagged_comm.get(yr,{}); dm_ = lagged_degree.get(yr,{})
    lc  = co_lead_comm_t.get(co); ic = cm.get(inv_k)
    sc_t.append(1.0 if lc is not None and ic is not None and ic==lc else 0.0)
    log_deg_t.append(np.log1p(dm_.get(inv_k,0)))
    log_deals_t.append(np.log1p(inv_deals_before[yr].get(inv_k,0)))

dyads["same_community_t"]   = sc_t
dyads["log_deg_t"]          = log_deg_t
dyads["log_deals_t"]        = log_deals_t
dyads["overall_sparsity_t"] = dyads["co"].map(co_os_t)
dyads["niche_cohesion_t"]   = dyads["co"].map(co_nc_t)
# has_prior_network: 1 if investor had ANY deal before t
dyads["has_prior_network"]  = (np.array(log_deals_t) > 0).astype(float)

dm = dyads.dropna(subset=["semantic_distance"]).copy()
dm["sem_dist_z"]  = std_col(dm, "semantic_distance")
dm["log_deg_z"]   = std_col(dm, "log_deg_t")
dm["log_deals_z"] = std_col(dm, "log_deals_t")
dm["nc_z"]        = std_col(dm, "niche_cohesion_t")
dm["os_z"]        = std_col(dm, "overall_sparsity_t")
dm["sem_sq_z"]    = dm["sem_dist_z"]**2
dm["sd_x_nc"]     = dm["sem_dist_z"] * dm["nc_z"]
dm["sd_x_os"]     = dm["sem_dist_z"] * dm["os_z"]
dm["geo_x_os"]    = dm["geo_same_country"] * dm["os_z"]
dm["sc_x_os"]     = dm["same_community_t"] * dm["os_z"]
dm["hpn_x_sd_x_os"] = dm["has_prior_network"] * dm["sd_x_os"]   # three-way: prior×sd×os
dm["hpn_x_os"]    = dm["has_prior_network"] * dm["os_z"]         # prior×os main

print(f"  Variables computed  [{elapsed()}]")

# ──────────────────────────────────────────────────────────────
# CHECK 3: corr(log_deg_z, log_deals_z) — fast, do first
# ──────────────────────────────────────────────────────────────
banner("CHECK 3: COLLINEARITY — corr(log_deg_z, log_deals_z)")

valid = dm[["log_deg_z","log_deals_z"]].dropna()
rp, _ = pearsonr(valid["log_deg_z"], valid["log_deals_z"])
rs, _ = spearmanr(valid["log_deg_z"], valid["log_deals_z"])
print(f"  Pearson  r  = {rp:+.4f}")
print(f"  Spearman ρ  = {rs:+.4f}")
if abs(rp) > 0.80:
    print(f"  WARNING: high collinearity. VIF ≈ {1/(1-rp**2):.1f}.")
    print(f"  The opposing signs (β_deg<0, β_deals>0) likely reflect this.")
else:
    print(f"  Collinearity moderate — VIF ≈ {1/(1-rp**2):.1f}.")

# Also report means by has_prior_network
for lab, mask in [("new entrant (0 prior deals)", dm["has_prior_network"]==0),
                   ("experienced (>0 prior deals)",dm["has_prior_network"]==1)]:
    n = mask.sum()
    d = dm.loc[mask,"log_deg_t"].mean()
    td = dm.loc[mask,"log_deals_t"].mean()
    os_ = dm.loc[mask,"overall_sparsity_t"].mean()
    sd_ = dm.loc[mask,"semantic_distance"].mean()
    print(f"\n  {lab} (N={n:,}):")
    print(f"    log_deg_t mean={d:.3f}  log_deals_t mean={td:.3f}  "
          f"OS_t mean={os_:.3f}  sem_dist mean={sd_:.4f}")

def run_clogit(df_sub, xvars, label=""):
    sub = df_sub[["tied","co"]+xvars].dropna().copy()
    ok  = sub.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
    sub = sub[ok].copy()
    if len(sub) < 200 or sub["co"].nunique() < 20:
        print(f"  {label}: too few ({len(sub)})"); return None
    sub["stratum"] = pd.factorize(sub["co"])[0]
    try:
        res = ConditionalLogit(sub["tied"].values,
                               sub[xvars].values,
                               groups=sub["stratum"].values).fit(
            method="bfgs", maxiter=800, disp=False)
        return res, xvars, len(sub), sub["co"].nunique()
    except Exception as e:
        print(f"  {label}: {e}"); return None

# ── Reference model from 79r (Step 4) ────────────────────────
b_vars_ref = ["sem_dist_z","sd_x_nc","sd_x_os",
              "geo_same_country","geo_same_region","geo_x_os",
              "same_community_t","sc_x_os",
              "log_deg_z","log_deals_z"]
r_ref = run_clogit(dm, b_vars_ref, "Reference (79r Step 4)")

# ──────────────────────────────────────────────────────────────
# CHECK 1: ADD sem_dist_z²
# ──────────────────────────────────────────────────────────────
banner("CHECK 1: ADD sem_dist_z² TO STEP 4 MODEL")

b_vars_sq = ["sem_dist_z","sem_sq_z",          # linear + quadratic
             "sd_x_nc","sd_x_os",
             "geo_same_country","geo_same_region","geo_x_os",
             "same_community_t","sc_x_os",
             "log_deg_z","log_deals_z"]
r_sq = run_clogit(dm, b_vars_sq, "Check 1")

if r_sq and r_ref:
    res_sq, xv_sq, n_sq, st_sq = r_sq
    res_ref, xv_ref, n_ref, st_ref = r_ref
    print(f"\n  Model with sem_dist_z² (N={n_sq:,} dyads, {st_sq:,} strata):")
    print(f"  {'Variable':<30} {'β (ref)':>10} {'β (+sq)':>10} {'SE (+sq)':>9} {'p (+sq)':>9} sig")
    print(f"  {'-'*72}")
    for v in b_vars_sq:
        b_new = res_sq.params[xv_sq.index(v)]
        se_new= res_sq.bse[xv_sq.index(v)]
        p_new = res_sq.pvalues[xv_sq.index(v)]
        b_old = (res_ref.params[xv_ref.index(v)]
                 if v in xv_ref else np.nan)
        b_old_str = f"{b_old:>+10.4f}" if not np.isnan(b_old) else f"{'—':>10}"
        print(f"  {v:<30} {b_old_str} {b_new:>+10.4f} {se_new:>9.4f} {p_new:>9.4f} {sig(p_new)}")
    # Key question
    b_os_sq  = res_sq.params[xv_sq.index("sd_x_os")]
    p_os_sq  = res_sq.pvalues[xv_sq.index("sd_x_os")]
    b_sq_    = res_sq.params[xv_sq.index("sem_sq_z")]
    p_sq_    = res_sq.pvalues[xv_sq.index("sem_sq_z")]
    print(f"\n  KEY: sem_dist_z²  β={b_sq_:+.4f}  p={p_sq_:.4f} {sig(p_sq_)}")
    print(f"  KEY: sd×OS after adding sq term: β={b_os_sq:+.4f}  p={p_os_sq:.4f} {sig(p_os_sq)}")
    if p_os_sq < 0.05 and b_os_sq < 0:
        print(f"  → sd×OS reversal PERSISTS after controlling for curvature.")
    elif p_os_sq >= 0.10:
        print(f"  → sd×OS becomes null after controlling for curvature (curvature mediation).")
    else:
        print(f"  → sd×OS weakens (p={p_os_sq:.3f}) — partial curvature mediation.")

# ──────────────────────────────────────────────────────────────
# CHECK 2: has_prior_network × sd×OS
# ──────────────────────────────────────────────────────────────
banner("CHECK 2: HAS_PRIOR_NETWORK × sd×OS")

# Model A: add has_prior_network as control
# Model B: add hpn_x_os and hpn_x_sd_x_os (three-way interaction)
# The question is: does sd×OS vanish / reverse in sign for experienced investors?

# Model A: add prior_network as main effect
b_vars_hpn_a = b_vars_ref + ["has_prior_network"]
r_hpn_a = run_clogit(dm, b_vars_hpn_a, "Check 2a")

# Model B: add three-way interaction
b_vars_hpn_b = b_vars_ref + ["has_prior_network","hpn_x_os","hpn_x_sd_x_os"]
r_hpn_b = run_clogit(dm, b_vars_hpn_b, "Check 2b")

# Model C: separate regressions by new-entrant vs experienced
print(f"\n  Sample split by has_prior_network:")
for lab, mask in [("New entrants only (has_prior=0)", dm["has_prior_network"]==0),
                   ("Experienced only (has_prior=1)",  dm["has_prior_network"]==1)]:
    sub_ = dm[mask].copy()
    r_ = run_clogit(sub_, b_vars_ref, lab)
    if r_ is None: continue
    res_, xv_, n_, st_ = r_
    b_sd  = res_.params[xv_.index("sem_dist_z")]
    p_sd  = res_.pvalues[xv_.index("sem_dist_z")]
    b_os  = res_.params[xv_.index("sd_x_os")]
    p_os  = res_.pvalues[xv_.index("sd_x_os")]
    b_nc  = res_.params[xv_.index("sd_x_nc")]
    p_nc  = res_.pvalues[xv_.index("sd_x_nc")]
    b_sc  = res_.params[xv_.index("sc_x_os")]
    p_sc  = res_.pvalues[xv_.index("sc_x_os")]
    print(f"\n  {lab} (N={n_:,} dyads, {st_:,} strata):")
    print(f"    sem_dist_z β={b_sd:+.4f} {sig(p_sd)}")
    print(f"    sd×OS      β={b_os:+.4f} {sig(p_os)}  "
          + ("REVERSAL HOLDS" if p_os<0.10 and b_os<0 else
             "null" if p_os>=0.10 else "CORRECT SIGN"))
    print(f"    sd×NC      β={b_nc:+.4f} {sig(p_nc)}")
    print(f"    sc×OS      β={b_sc:+.4f} {sig(p_sc)}")

# Summary of Model B (three-way)
if r_hpn_b:
    res_b, xv_b, n_b, st_b = r_hpn_b
    print(f"\n  Model B: three-way interaction (N={n_b:,}, {st_b:,} strata):")
    for v in ["has_prior_network","hpn_x_os","hpn_x_sd_x_os","sd_x_os"]:
        if v not in xv_b: continue
        b = res_b.params[xv_b.index(v)]
        p = res_b.pvalues[xv_b.index(v)]
        print(f"    {v:<28} β={b:+.4f}  p={p:.4f} {sig(p)}")
    b_os_new = res_b.params[xv_b.index("sd_x_os")]
    b_3way   = res_b.params[xv_b.index("hpn_x_sd_x_os")]
    p_3way   = res_b.pvalues[xv_b.index("hpn_x_sd_x_os")]
    print(f"\n  Implied sd×OS slope for new entrants:  β={b_os_new:+.4f}")
    print(f"  Implied sd×OS slope for experienced:   β={b_os_new+b_3way:+.4f}  "
          f"(three-way p={p_3way:.4f} {sig(p_3way)})")
    if p_3way < 0.10:
        print(f"  → Reversal IS moderated by prior network: effect differs by entrant status.")
    else:
        print(f"  → Three-way null: reversal is NOT specific to new entrants.")

# ──────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────
banner("FINAL SUMMARY")
print(f"  Reference: sd×OS β=−0.0545**  Holm p=0.021 (Script 79r Step 4)")
if r_sq:
    b_os_sq = r_sq[0].params[b_vars_sq.index("sd_x_os")]
    p_os_sq = r_sq[0].pvalues[b_vars_sq.index("sd_x_os")]
    b_sq2   = r_sq[0].params[b_vars_sq.index("sem_sq_z")]
    p_sq2   = r_sq[0].pvalues[b_vars_sq.index("sem_sq_z")]
    print(f"\n  1. Add sem_dist_z²:")
    print(f"     sem_dist_z²  β={b_sq2:+.4f}  p={p_sq2:.4f} {sig(p_sq2)}")
    print(f"     sd×OS after: β={b_os_sq:+.4f}  p={p_os_sq:.4f} {sig(p_os_sq)}")
    verdict1 = ("reversal PERSISTS (not driven by curvature)" if p_os_sq<0.10 and b_os_sq<0
                else "reversal DISSOLVES (was curvature artefact)" if p_os_sq>=0.10
                else f"reversal WEAKENS (p={p_os_sq:.3f})")
    print(f"     → {verdict1}")

if r_hpn_b:
    p_3way = r_hpn_b[0].pvalues[r_hpn_b[1].index("hpn_x_sd_x_os")]
    b_os_base = r_hpn_b[0].params[r_hpn_b[1].index("sd_x_os")]
    b_3way    = r_hpn_b[0].params[r_hpn_b[1].index("hpn_x_sd_x_os")]
    print(f"\n  2. New-entrant moderation (has_prior_network × sd×OS):")
    print(f"     Three-way p={p_3way:.4f} {sig(p_3way)}")
    print(f"     sd×OS new entrants: β={b_os_base:+.4f}")
    print(f"     sd×OS experienced:  β={b_os_base+b_3way:+.4f}")
    verdict2 = ("reversal IS entrant-specific" if p_3way<0.10
                else "reversal is NOT specific to new entrants (uniform across experience)")
    print(f"     → {verdict2}")

print(f"\n  3. corr(log_deg_z, log_deals_z): Pearson r={rp:+.4f}  Spearman ρ={rs:+.4f}")
verdict3 = ("HIGH collinearity" if abs(rp)>0.8 else
            "moderate collinearity" if abs(rp)>0.5 else "low collinearity")
print(f"     → {verdict3} — opposing signs (deg<0, deals>0) "
      + ("likely artefact of VIF" if abs(rp)>0.8 else "may reflect genuine separation"))

print(f"\n  [{elapsed()}]")
