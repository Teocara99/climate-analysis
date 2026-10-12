"""
Script 81: Novelty and Cross-Community Investment
===================================================
PRE-SPECIFIED (report regardless of outcome):
  P1: overall_sparsity_t  → MORE excess community spread
  P2: niche_cohesion_t    → LESS excess community spread

Unit: startup (not dyad). No failure variable.
Outcome: excess_spread = size-normalised community diversity
         among a startup's investors relative to a random draw
         from the year-t pool.
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
import statsmodels.formula.api as smf

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
def holm(ps):
    ps = np.array(ps, dtype=float); m = len(ps); o = np.argsort(ps); adj = np.empty(m)
    adj[o] = [min(1.0, ps[o[i]]*(m-i)) for i in range(m)]
    for i in range(1, m): adj[o[i]] = max(adj[o[i]], adj[o[i-1]])
    return adj

# ──────────────────────────────────────────────────────────────
# LOAD
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

years_needed = sorted(first_round["year"].dropna().unique().astype(int))
min_yr, max_yr = int(min(years_needed)), int(max(years_needed))

# Novelty axes (time-aware, from 79r computation)
from sklearn.neighbors import NearestNeighbors
with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns_pkl = pickle.load(fh)
startup_names_all = [n.strip() for n in ns_pkl["names"]]
embs_raw = np.array(ns_pkl["embs_raw"], dtype=np.float32)
norms    = np.linalg.norm(embs_raw, axis=1, keepdims=True)
embs_l2  = embs_raw / np.where(norms < 1e-10, 1.0, norms)
startup_emb_map = {n: embs_l2[i] for i, n in enumerate(startup_names_all)}

# CPU dataset for controls
cdf = pd.read_csv(BASE/"output"/"cpu"/"cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
co_log_cap = dict(zip(cdf["co"], cdf["log_capital"]))
co_country = dict(zip(cdf["co"], cdf["hq_country"]))
print(f"  Loaded  [{elapsed()}]")

# ──────────────────────────────────────────────────────────────
# BUILD TIME-LAGGED COMMUNITIES (lean)
# ──────────────────────────────────────────────────────────────
banner("TIME-LAGGED LOUVAIN COMMUNITIES")

edges_by_year = defaultdict(lambda: defaultdict(float))
for _, row in deals_v.iterrows():
    yr = int(row["year"])
    invs = [x.lower() for x in comma_parse(row["Investors"])]
    for a, b in combinations(invs, 2):
        u, v = (a,b) if a<b else (b,a)
        edges_by_year[yr][(u,v)] += 1

lagged_comm  = {}   # lagged_comm[yr] = {inv: comm_id}
static_comm  = {}   # static (built on all deals) for robustness
cumulative_edges = defaultdict(float)
ALL_EDGES = defaultdict(float)

for yr in range(min_yr-1, max_yr+2):
    if yr in edges_by_year:
        for (u,v), w in edges_by_year[yr].items():
            cumulative_edges[(u,v)] += w
            ALL_EDGES[(u,v)] += w
    G = nx.Graph()
    for (u,v), w in cumulative_edges.items(): G.add_edge(u, v, weight=w)
    if G.number_of_edges() < 10:
        lagged_comm[yr+1] = {}; continue
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    lagged_comm[yr+1] = partition

# Static community (full network)
G_full = nx.Graph()
for (u,v), w in ALL_EDGES.items(): G_full.add_edge(u, v, weight=w)
static_partition = community_louvain.best_partition(G_full, weight="weight", random_state=42)
print(f"  Lagged maps: {len(lagged_comm)} year keys  "
      f"(largest: {max(len(v) for v in lagged_comm.values()):,} investors)")
print(f"  Static partition: {len(static_partition):,} investors, "
      f"{len(set(static_partition.values()))} communities  [{elapsed()}]")

# ──────────────────────────────────────────────────────────────
# TIME-AWARE NOVELTY AXES (lean rebuild)
# ──────────────────────────────────────────────────────────────
banner("TIME-AWARE NOVELTY AXES")

co_to_year     = dict(zip(first_round["co"], first_round["year"].astype(int)))
embeddable_cos = [co for co in startup_names_all if co in co_to_year]
embeddable_yrs = np.array([co_to_year[co] for co in embeddable_cos], dtype=int)
embeddable_emb = np.array([startup_emb_map[co] for co in embeddable_cos], dtype=np.float32)
sort_idx       = np.argsort(embeddable_yrs, kind="stable")
sorted_cos     = [embeddable_cos[i] for i in sort_idx]
sorted_yrs     = embeddable_yrs[sort_idx]
sorted_emb     = embeddable_emb[sort_idx]

MIN_PRIOR = 100
tanov_rows = []
prior_cos, prior_embs = [], []

for yr in sorted(set(sorted_yrs)):
    yr_mask = sorted_yrs == yr
    yr_cos  = [sorted_cos[i] for i in range(len(sorted_cos)) if yr_mask[i]]
    yr_embs = sorted_emb[yr_mask]
    n_prior = len(prior_cos)
    if n_prior >= MIN_PRIOR:
        prior_arr = np.array(prior_embs, dtype=np.float32)
        k_max = min(51, n_prior)
        nn = NearestNeighbors(n_neighbors=k_max, metric="cosine",
                              algorithm="brute", n_jobs=-1).fit(prior_arr)
        dists, _ = nn.kneighbors(yr_embs, n_neighbors=k_max)
        sims = 1.0 - dists
        k5  = min(5,  n_prior); k50 = min(50, n_prior)
        for j, co in enumerate(yr_cos):
            tanov_rows.append({"co": co, "year": yr,
                               "loc5_t":  float(sims[j,:k5].mean()),
                               "reg50_t": float(sims[j,:k50].mean()),
                               "ok": True})
    else:
        for co in yr_cos:
            tanov_rows.append({"co": co, "year": yr, "loc5_t": np.nan,
                               "reg50_t": np.nan, "ok": False})
    prior_cos.extend(yr_cos); prior_embs.extend(yr_embs.tolist())

tanov_df = pd.DataFrame(tanov_rows)
tanov_ok = tanov_df[tanov_df["ok"]].copy()
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
# STEP 1: COMMUNITY ASSIGNMENT AND SAMPLE SELECTION
# ──────────────────────────────────────────────────────────────
banner("STEP 1: COMMUNITY ASSIGNMENT")

COVERAGE_THRESH = 0.75
MIN_INVESTORS   = 2

records = []
for _, fr in first_round.iterrows():
    co  = fr["co"]
    yr  = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    if co not in co_os_t: continue          # need time-aware novelty
    inv_list = comma_parse(str(fr["Investors"]))
    k_total  = len(inv_list)
    if k_total < MIN_INVESTORS: continue

    comm_map = lagged_comm.get(yr, {})
    assigned_comms = [comm_map.get(i.lower()) for i in inv_list]
    k_with_comm = sum(1 for c in assigned_comms if c is not None)
    coverage    = k_with_comm / k_total

    static_comms = [static_partition.get(i.lower()) for i in inv_list]
    k_static     = sum(1 for c in static_comms if c is not None)

    records.append({
        "co": co, "year": yr, "k_total": k_total,
        "k_with_comm": k_with_comm, "coverage": coverage,
        "lagged_comms": [c for c in assigned_comms if c is not None],
        "static_comms": [c for c in static_comms if c is not None],
        "k_static": k_static,
        "overall_sparsity_t": co_os_t[co],
        "niche_cohesion_t":   co_nc_t[co],
        "log_capital": co_log_cap.get(co, np.nan),
        "hq_country":  co_country.get(co, ""),
        "co_country":  str(fr.get("Company Country/Territory/Region","")).strip(),
    })

all_df = pd.DataFrame(records)
print(f"  All startups with ≥2 investors + novelty: {len(all_df):,}")

n_cov  = (all_df["coverage"] >= COVERAGE_THRESH).sum()
main_df = all_df[all_df["coverage"] >= COVERAGE_THRESH].copy()
print(f"  Retained (coverage ≥{COVERAGE_THRESH:.0%}): {n_cov:,}")
print(f"  Dropped (low coverage): {len(all_df)-n_cov:,}")
print(f"\n  Investor count distribution (main sample):")
for lo, hi, lab in [(2,2,"exactly 2"),(3,4,"3–4"),(5,9,"5–9"),(10,999,"10+")]:
    n = ((main_df["k_total"]>=lo)&(main_df["k_total"]<=hi)).sum()
    pct = n/len(main_df)
    print(f"    {lab:<12}: {n:,} ({pct:.1%})")
print(f"\n  Coverage stats: mean={main_df['coverage'].mean():.3f}  "
      f"median={main_df['coverage'].median():.3f}")

# ──────────────────────────────────────────────────────────────
# STEP 2: EXCESS SPREAD (simulation)
# ──────────────────────────────────────────────────────────────
banner("STEP 2: EXCESS SPREAD (500 Monte Carlo draws)")

RNG = np.random.default_rng(42)
N_SIM = 500

# Year-t pools: investors active in year t AND have lagged community
year_pool = {}
for yr in years_needed:
    active_yr = set()
    for _, row in deals_v[deals_v["year"]==yr].iterrows():
        for inv in comma_parse(row["Investors"]):
            active_yr.add(inv.lower())
    comm_map = lagged_comm.get(yr, {})
    pool = [(inv, comm_map[inv]) for inv in active_yr if inv in comm_map]
    year_pool[yr] = pool

print(f"  Pool sizes by year:")
for yr in sorted(year_pool.keys()):
    print(f"    {yr}: {len(year_pool[yr]):,} investors with lagged community")

# Simulate
excess_spreads = []; obs_distincts = []; exp_spans = []; sd_spans = []
k_eff_list     = []
n_low_sd       = 0

for _, row in main_df.iterrows():
    yr     = int(row["year"])
    obs_c  = row["lagged_comms"]     # communities of investors with lagged assignment
    k_eff  = len(obs_c)
    pool   = year_pool.get(yr, [])

    # observed distinct communities
    obs_distinct = len(set(obs_c))

    if len(pool) < k_eff:
        # Pool too small — draw with replacement
        pool_comms = np.array([c for _, c in pool], dtype=int)
        draws = RNG.choice(pool_comms, size=(N_SIM, max(k_eff, 1)), replace=True)
    else:
        pool_comms = np.array([c for _, c in pool], dtype=int)
        draws = np.vstack([
            RNG.choice(pool_comms, size=k_eff, replace=False)
            for _ in range(N_SIM)
        ])

    sim_counts = np.array([len(np.unique(draws[s])) for s in range(N_SIM)],
                          dtype=float)
    exp_span = sim_counts.mean()
    sd_span  = sim_counts.std()

    if sd_span < 0.01:
        n_low_sd += 1
        sd_span = 0.01       # floor to avoid div-by-zero

    es = (obs_distinct - exp_span) / sd_span
    excess_spreads.append(es)
    obs_distincts.append(obs_distinct)
    exp_spans.append(exp_span)
    sd_spans.append(sd_span)
    k_eff_list.append(k_eff)

main_df["excess_spread"]   = excess_spreads
main_df["obs_distinct"]    = obs_distincts
main_df["exp_span"]        = exp_spans
main_df["sd_span"]         = sd_spans
main_df["k_eff"]           = k_eff_list

print(f"\n  [{elapsed()}]")
print(f"  N = {len(main_df):,} startups computed")
print(f"  Cases with sd_span < 0.01 (floored): {n_low_sd:,}")
print(f"\n  Excess spread distribution:")
es = main_df["excess_spread"]
for pct in [1,5,25,50,75,95,99]:
    print(f"    p{pct:>2}: {np.percentile(es, pct):>+7.3f}")
print(f"  Mean={es.mean():+.3f}  SD={es.std():.3f}  "
      f"min={es.min():+.3f}  max={es.max():+.3f}")
print(f"\n  Observed vs expected distinct communities:")
print(f"  Mean observed: {main_df['obs_distinct'].mean():.2f}  "
      f"mean expected: {main_df['exp_span'].mean():.2f}  "
      f"mean SD(sim): {main_df['sd_span'].mean():.3f}")

# Correlation with novelty axes
for col in ["overall_sparsity_t","niche_cohesion_t"]:
    rs, ps = spearmanr(main_df[col].dropna(), main_df.loc[main_df[col].notna(),"excess_spread"])
    print(f"  Spearman ρ(excess_spread, {col}) = {rs:+.3f}  {sig(ps)}")

# ──────────────────────────────────────────────────────────────
# OLS HELPER WITH YEAR-CLUSTERED SEs
# ──────────────────────────────────────────────────────────────
def run_ols_clustered(df_sub, formula, cluster_col="year"):
    sub = df_sub.dropna(subset=["excess_spread","overall_sparsity_t",
                                 "niche_cohesion_t","log_capital","hq_country","year"]).copy()
    if len(sub) < 50: return None
    # Need country dummies (top countries only to keep matrix stable)
    top_countries = sub["hq_country"].value_counts().nlargest(20).index
    sub["hq_ctry_trunc"] = sub["hq_country"].where(
        sub["hq_country"].isin(top_countries), other="Other")
    try:
        model = smf.ols(
            "excess_spread ~ overall_sparsity_t + niche_cohesion_t + "
            "log_capital + C(year) + C(hq_ctry_trunc)",
            data=sub)
        res = model.fit(cov_type="cluster",
                        cov_kwds={"groups": sub[cluster_col].values})
        return res, sub
    except Exception as e:
        print(f"  OLS error: {e}"); return None

# ──────────────────────────────────────────────────────────────
# STEP 3: MAIN MODEL
# ──────────────────────────────────────────────────────────────
banner("STEP 3: MAIN MODEL")

r_main = run_ols_clustered(main_df, "", cluster_col="year")
if r_main:
    res, sub = r_main
    n_used = len(sub)
    n_yrs  = sub["year"].nunique()
    print(f"  N={n_used:,} startups, {n_yrs} year clusters")
    print(f"  NOTE: {n_yrs} clusters — clustered SEs have limited asymptotic justification (<20).")
    print(f"  R² = {res.rsquared:.4f}  adj-R² = {res.rsquared_adj:.4f}")

    print(f"\n  {'Variable':<30} {'β':>9} {'SE':>9} {'p':>9} sig")
    print(f"  {'-'*60}")
    focal = ["overall_sparsity_t","niche_cohesion_t","log_capital"]
    for v in focal:
        if v not in res.params.index:
            print(f"  {v:<30} — (not in model)"); continue
        b, se, p = res.params[v], res.bse[v], res.pvalues[v]
        print(f"  {v:<30} {b:>+9.4f} {se:>9.4f} {p:>9.4f} {sig(p)}")

    ps2  = [res.pvalues.get("overall_sparsity_t", 1.0),
            res.pvalues.get("niche_cohesion_t",   1.0)]
    bs2  = [res.params.get("overall_sparsity_t",  0.0),
            res.params.get("niche_cohesion_t",    0.0)]
    h2   = holm(ps2)
    print(f"\n  Holm on 2 pre-specified terms:")
    print(f"  {'Term':<24} {'β':>9} {'p_raw':>9} {'Holm':>9}")
    for lab, b, p, hp in zip(["overall_sparsity_t","niche_cohesion_t"],
                               bs2, ps2, h2):
        print(f"  {lab:<24} {b:>+9.4f} {p:>9.4f} {hp:>9.4f} {sig(hp)}")

    b_os, p_os, hp_os = bs2[0], ps2[0], h2[0]
    b_nc, p_nc, hp_nc = bs2[1], ps2[1], h2[1]
    print(f"\n  P1 (OS → MORE spread, β>0): β={b_os:+.4f}  Holm={hp_os:.3f}  "
          + ("SUPPORTED" if b_os>0 and hp_os<0.10 else
             "WRONG SIGN" if hp_os<0.10 else "NULL"))
    print(f"  P2 (NC → LESS spread, β<0): β={b_nc:+.4f}  Holm={hp_nc:.3f}  "
          + ("SUPPORTED" if b_nc<0 and hp_nc<0.10 else
             "WRONG SIGN" if hp_nc<0.10 else "NULL"))

# ──────────────────────────────────────────────────────────────
# STEP 4: STRATIFIED BY INVESTOR COUNT
# ──────────────────────────────────────────────────────────────
banner("STEP 4: STRATIFIED BY INVESTOR COUNT")
print(f"  {'Stratum':<10} {'N':>6}  {'OS_t β':>9} {'p':>7}  {'NC_t β':>9} {'p':>7}  verdict")
print(f"  {'-'*70}")
for lo, hi, lab in [(2,2,"k=2"),(3,4,"k=3-4"),(5,9,"k=5-9"),(10,999,"k=10+")]:
    sub_k = main_df[(main_df["k_total"]>=lo)&(main_df["k_total"]<=hi)]
    if len(sub_k) < 30:
        print(f"  {lab:<10} {'too few':>6}"); continue
    r_k = run_ols_clustered(sub_k, "", cluster_col="year")
    if r_k is None:
        print(f"  {lab:<10} error"); continue
    res_k, sub_k2 = r_k
    b_os = res_k.params.get("overall_sparsity_t", np.nan)
    p_os = res_k.pvalues.get("overall_sparsity_t", 1.0)
    b_nc = res_k.params.get("niche_cohesion_t",   np.nan)
    p_nc = res_k.pvalues.get("niche_cohesion_t",  1.0)
    v_os = ("✓" if b_os>0 and p_os<0.10 else "✗" if p_os<0.10 else "·")
    v_nc = ("✓" if b_nc<0 and p_nc<0.10 else "✗" if p_nc<0.10 else "·")
    print(f"  {lab:<10} {len(sub_k2):>6,}  {b_os:>+9.4f} {p_os:>7.4f}  "
          f"{b_nc:>+9.4f} {p_nc:>7.4f}  OS:{v_os} NC:{v_nc}")

# ──────────────────────────────────────────────────────────────
# STEP 5: ROBUSTNESS
# ──────────────────────────────────────────────────────────────
banner("STEP 5: ROBUSTNESS")

# 5a: static communities
print(f"  5a. Static communities (full-network Louvain):")
static_rows = []
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None or co not in co_os_t: continue
    inv_list = comma_parse(str(fr["Investors"]))
    if len(inv_list) < MIN_INVESTORS: continue
    s_comms = [static_partition.get(i.lower()) for i in inv_list]
    k_s = sum(1 for c in s_comms if c is not None)
    cov_s = k_s / len(inv_list)
    if cov_s < COVERAGE_THRESH: continue
    static_rows.append({"co": co, "year": yr,
                        "overall_sparsity_t": co_os_t[co],
                        "niche_cohesion_t":   co_nc_t[co],
                        "log_capital": co_log_cap.get(co, np.nan),
                        "hq_country": co_country.get(co, ""),
                        "static_comms_list": [c for c in s_comms if c is not None],
                        "k_total": len(inv_list), "k_static": k_s})

static_df = pd.DataFrame(static_rows)

# Build year-t static pool
year_pool_static = {}
for yr in years_needed:
    active_yr = set()
    for _, row in deals_v[deals_v["year"]==yr].iterrows():
        for inv in comma_parse(row["Investors"]): active_yr.add(inv.lower())
    pool = [(inv, static_partition[inv]) for inv in active_yr if inv in static_partition]
    year_pool_static[yr] = pool

es_static = []
for _, row in static_df.iterrows():
    yr = int(row["year"]); obs_c = row["static_comms_list"]; k_eff = len(obs_c)
    pool = year_pool_static.get(yr, [])
    if len(pool) < k_eff: continue
    pool_comms = np.array([c for _, c in pool], dtype=int)
    draws = np.vstack([RNG.choice(pool_comms, k_eff, replace=False)
                       for _ in range(N_SIM)])
    sim_counts = np.array([len(np.unique(draws[s])) for s in range(N_SIM)], dtype=float)
    exp_s = sim_counts.mean(); sd_s = max(sim_counts.std(), 0.01)
    es_static.append((obs_distinct - exp_s) / sd_s if False else  # use row's observed
                     (len(set(obs_c)) - exp_s) / sd_s)

static_df = static_df.iloc[:len(es_static)].copy()
static_df["excess_spread"] = es_static

r_static = run_ols_clustered(static_df, "", cluster_col="year")
if r_static:
    res_s, sub_s = r_static
    b_os_s = res_s.params.get("overall_sparsity_t", np.nan)
    p_os_s = res_s.pvalues.get("overall_sparsity_t", 1.0)
    b_nc_s = res_s.params.get("niche_cohesion_t",   np.nan)
    p_nc_s = res_s.pvalues.get("niche_cohesion_t",  1.0)
    print(f"    N={len(sub_s):,}  OS_t β={b_os_s:+.4f} {sig(p_os_s)}  "
          f"NC_t β={b_nc_s:+.4f} {sig(p_nc_s)}")

# 5b: exclude top 1% overall_sparsity_t
p99_os = np.percentile(main_df["overall_sparsity_t"].dropna(), 99)
main_trim = main_df[main_df["overall_sparsity_t"] <= p99_os].copy()
print(f"\n  5b. Exclude OS_t > p99 ({p99_os:.3f}), drop {len(main_df)-len(main_trim):,}:")
r_trim = run_ols_clustered(main_trim, "", cluster_col="year")
if r_trim:
    res_t, sub_t = r_trim
    b_os_t = res_t.params.get("overall_sparsity_t", np.nan)
    p_os_t = res_t.pvalues.get("overall_sparsity_t", 1.0)
    b_nc_t = res_t.params.get("niche_cohesion_t",   np.nan)
    p_nc_t = res_t.pvalues.get("niche_cohesion_t",  1.0)
    print(f"    N={len(sub_t):,}  OS_t β={b_os_t:+.4f} {sig(p_os_t)}  "
          f"NC_t β={b_nc_t:+.4f} {sig(p_nc_t)}")
    if r_main:
        for lab, b_full, b_new, p_new in [
            ("OS_t", bs2[0], b_os_t, p_os_t),
            ("NC_t", bs2[1], b_nc_t, p_nc_t)
        ]:
            flag = " ← SIGN CHANGE" if np.sign(b_full)!=np.sign(b_new) and not np.isnan(b_new) else ""
            print(f"    {lab}: {b_full:+.4f} → {b_new:+.4f}{flag}")

# 5c: MDE
if r_main:
    se_os = r_main[0].bse.get("overall_sparsity_t", np.nan)
    se_nc = r_main[0].bse.get("niche_cohesion_t",   np.nan)
    print(f"\n  5c. MDE at 80% power (N={len(sub):,}):")
    print(f"  {'Term':<24} {'SE':>8} {'MDE':>10} {'β actual':>10} {'β/MDE':>8}")
    for lab, se, b_act in [("overall_sparsity_t", se_os, bs2[0]),
                            ("niche_cohesion_t",   se_nc, bs2[1])]:
        if np.isnan(se): continue
        mde = (1.96+0.842)*se
        ratio = b_act/mde if abs(mde)>1e-10 else np.nan
        print(f"  {lab:<24} {se:>8.4f} {mde:>10.4f} {b_act:>+10.4f} "
              + (f"{ratio:>8.2f}" if not np.isnan(ratio) else f"{'n/a':>8}"))

# ──────────────────────────────────────────────────────────────
# FINAL VERDICTS
# ──────────────────────────────────────────────────────────────
banner("FINAL VERDICTS")
if r_main:
    b_os, p_os, hp_os = bs2[0], ps2[0], h2[0]
    b_nc, p_nc, hp_nc = bs2[1], ps2[1], h2[1]
    print(f"  P1 (OS_t → MORE excess spread, β>0):")
    print(f"    β={b_os:+.4f}  SE={r_main[0].bse.get('overall_sparsity_t',np.nan):.4f}  "
          f"p_raw={p_os:.4f}  Holm={hp_os:.4f}  {sig(hp_os)}")
    if b_os > 0 and hp_os < 0.10:   v1 = "SUPPORTED"
    elif hp_os < 0.10:               v1 = "WRONG SIGN"
    else:                            v1 = "NULL"
    print(f"    → {v1}")
    print(f"\n  P2 (NC_t → LESS excess spread, β<0):")
    print(f"    β={b_nc:+.4f}  SE={r_main[0].bse.get('niche_cohesion_t',np.nan):.4f}  "
          f"p_raw={p_nc:.4f}  Holm={hp_nc:.4f}  {sig(hp_nc)}")
    if b_nc < 0 and hp_nc < 0.10:   v2 = "SUPPORTED"
    elif hp_nc < 0.10:               v2 = "WRONG SIGN"
    else:                            v2 = "NULL"
    print(f"    → {v2}")
print(f"\n  [{elapsed()}]")
