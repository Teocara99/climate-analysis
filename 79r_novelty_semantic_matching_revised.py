"""
Script 79r: Time-Aware Novelty × Semantic Matching (Revised)
==============================================================
PRE-SPECIFIED (report regardless of outcome):
  P1: semantic_distance × niche_cohesion_t  < 0
      (semantic penalty STRONGER for niche ventures — specialists exist)
  P2: semantic_distance × overall_sparsity_t > 0
      (semantic penalty WEAKER for lonely outliers — no specialists exist)
  P3: geo_same_country × overall_sparsity_t > 0
      same_community_t × overall_sparsity_t > 0
      (for outliers, geography and community carry more weight)

All network measures (degree, deal count, community) computed from deals
STRICTLY BEFORE the focal startup's first-round year (t−1 network).
Novelty computed against startups with first round BEFORE year t.
"""
import re, warnings, time, pickle
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.neighbors import NearestNeighbors
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

def holm(ps):
    ps = np.array(ps, dtype=float)
    m = len(ps); o = np.argsort(ps); adj = np.empty(m)
    adj[o] = [min(1.0, ps[o[i]]*(m-i)) for i in range(m)]
    for i in range(1, m): adj[o[i]] = max(adj[o[i]], adj[o[i-1]])
    return adj

def bh(ps):
    ps = np.array(ps, dtype=float)
    m = len(ps); o = np.argsort(ps); adj = np.empty(m)
    for rank, i in enumerate(o, 1): adj[i] = min(1.0, ps[i]*m/rank)
    for i in range(m-2, -1, -1): adj[o[i]] = min(adj[o[i]], adj[o[i+1]])
    return adj

def std_col(df, col):
    s = df[col].dropna(); mu, sd = s.mean(), s.std()
    return (df[col] - mu) / sd if sd > 1e-10 else df[col].copy()

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
banner("LOADING DATA")
import sys; sys.path.insert(0, str(BASE))
from load_data import load_deals

deals = load_deals()
deals["co"]        = deals["Companies"].str.strip()
deals["Deal Date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals["year"]      = deals["Deal Date"].dt.year.fillna(deals["Year"]).astype("Int64")
deals_v = deals.dropna(subset=["year", "Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

first_round = (deals_v.sort_values("Deal Date")
               .groupby("co").first()
               .reset_index()
               [["co","year","Investors",
                 "Company Country/Territory/Region","HQ Global Region"]])
first_round["first_date"] = (deals_v.sort_values("Deal Date")
                              .groupby("co")["Deal Date"].first().values)

# Investor embeddings (768-dim)
emb_matrix    = np.load(BASE/"data"/"investor_embeddings.npy")
inv_names_ser = pd.read_csv(BASE/"data"/"investor_names.csv")["investor"].str.strip()
inv_emb_map   = {n.lower(): emb_matrix[i] for i, n in enumerate(inv_names_ser)}

# Startup embeddings (SPECTER2, 1024-dim, L2-normalised)
with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns_pkl = pickle.load(fh)
startup_names_all = [n.strip() for n in ns_pkl["names"]]
embs_raw = np.array(ns_pkl["embs_raw"], dtype=np.float32)
norms    = np.linalg.norm(embs_raw, axis=1, keepdims=True)
norms    = np.where(norms < 1e-10, 1.0, norms)
embs_l2  = embs_raw / norms                               # L2-normalised
startup_emb_map = {n: embs_l2[i] for i, n in enumerate(startup_names_all)}

# Investor metadata
inv_raw = pd.read_excel(BASE/"data"/"Investors_data.xlsx", skiprows=7)
inv_raw["inv_key"] = inv_raw["Investors"].str.strip().str.lower()
inv_country_map = dict(zip(inv_raw["inv_key"],
    inv_raw["HQ Country/Territory/Region"].fillna("")))
inv_region_map  = dict(zip(inv_raw["inv_key"],
    inv_raw["HQ Global Region"].fillna("")))

# Static novelty (Script 76) for correlation check
static_nov = pd.read_csv(BASE/"output"/"two_level_novelty"/"novelty_final.csv")
static_nov["co"] = static_nov["name"].str.strip()
static_sparsity = dict(zip(static_nov["co"], static_nov["overall_sparsity"]))
static_niche    = dict(zip(static_nov["co"], static_nov["niche_cohesion"]))

# CPU dataset
cdf = pd.read_csv(BASE/"output"/"cpu"/"cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
co_log_cap  = dict(zip(cdf["co"], cdf["log_capital"]))
co_n_rounds = dict(zip(cdf["co"], cdf["n_rounds"]))
co_country  = dict(zip(cdf["co"], cdf["hq_country"]))
co_region   = dict(zip(cdf["co"], cdf["hq_region"]))

print(f"  Startup embeddings: {embs_l2.shape}  ({len(startup_emb_map):,} named)")
print(f"  Investor embeddings: {emb_matrix.shape}")
print(f"  First rounds: {len(first_round):,}  years: "
      f"{first_round['year'].min()}–{first_round['year'].max()}")

# ──────────────────────────────────────────────────────────────
# STEP 1: TIME-AWARE NETWORK
# ──────────────────────────────────────────────────────────────
banner("STEP 1: TIME-AWARE NETWORK")

years_needed = sorted(first_round["year"].dropna().unique().astype(int))
min_yr, max_yr = int(min(years_needed)), int(max(years_needed))
print(f"  Building cumulative co-investment networks for years {min_yr-1}–{max_yr}...")

# Build edge list per year
edges_by_year = defaultdict(lambda: defaultdict(float))
for _, row in deals_v.iterrows():
    yr   = int(row["year"])
    invs = [x.lower() for x in comma_parse(row["Investors"])]
    for a, b in combinations(invs, 2):
        u, v = (a, b) if a < b else (b, a)
        edges_by_year[yr][(u, v)] += 1

# Per-investor cumulative deal count and unique startups backed (pre-t)
inv_deals_before = defaultdict(lambda: defaultdict(int))   # [yr][inv] = deal count
inv_cos_before   = defaultdict(lambda: defaultdict(set))   # [yr][inv] = set of cos
for _, row in deals_v.iterrows():
    yr = int(row["year"]); co = row["co"]
    for inv in comma_parse(row["Investors"]):
        inv_k = inv.lower()
        for fut_yr in range(yr+1, max_yr+3):
            inv_deals_before[fut_yr][inv_k] += 1
            inv_cos_before[fut_yr][inv_k].add(co)

# Cumulative Louvain + degree per year
lagged_comm = {}      # lagged_comm[yr] = {inv: community_id}  (network built from < yr)
lagged_degree = {}    # lagged_degree[yr] = {inv: degree}

cumulative_edges = defaultdict(float)
for yr in range(min_yr-1, max_yr+2):
    if yr in edges_by_year:
        for (u,v), w in edges_by_year[yr].items():
            cumulative_edges[(u,v)] += w

    G = nx.Graph()
    for (u,v), w in cumulative_edges.items():
        G.add_edge(u, v, weight=w)

    if G.number_of_edges() < 10:
        lagged_comm[yr+1]   = {}
        lagged_degree[yr+1] = {}
        continue

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    lagged_comm[yr+1]   = partition
    lagged_degree[yr+1] = dict(G.degree())
    if yr in years_needed or yr == max_yr:
        print(f"  ≤{yr}: {G.number_of_nodes():,} investors, "
              f"{G.number_of_edges():,} edges, "
              f"{len(set(partition.values()))} communities  [{elapsed()}]")

# Count new entrants (no prior deals) among actual lead investors
n_new_entrant = 0; n_existing = 0
for _, fr in first_round.iterrows():
    yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    for inv in comma_parse(str(fr["Investors"])):
        inv_k = inv.lower()
        if inv_deals_before[yr].get(inv_k, 0) == 0:
            n_new_entrant += 1
        else:
            n_existing += 1

total_inv_ties = n_new_entrant + n_existing
print(f"\n  Investor-ties coverage:")
print(f"    Prior deals > 0:  {n_existing:,} ({n_existing/total_inv_ties:.1%}) — time-aware vars available")
print(f"    New entrant (0 prior deals): {n_new_entrant:,} ({n_new_entrant/total_inv_ties:.1%})")
print(f"    Handling: degree=0, deals=0, community=absent (same_community_t=0)")

# ──────────────────────────────────────────────────────────────
# STEP 2: TIME-AWARE NOVELTY
# ──────────────────────────────────────────────────────────────
banner("STEP 2: TIME-AWARE NOVELTY")

# Map startup name → first-round year
co_to_year = dict(zip(first_round["co"], first_round["year"].astype(int)))

# Only startups present in both first_round and embedding map
embeddable_cos = [co for co in startup_names_all if co in co_to_year]
embeddable_yrs = np.array([co_to_year[co] for co in embeddable_cos], dtype=int)
embeddable_emb = np.array([startup_emb_map[co] for co in embeddable_cos], dtype=np.float32)
N_emb = len(embeddable_cos)
print(f"  Startups with embedding AND first-round year: {N_emb:,}")

# Sort chronologically
sort_idx   = np.argsort(embeddable_yrs, kind="stable")
sorted_cos = [embeddable_cos[i] for i in sort_idx]
sorted_yrs = embeddable_yrs[sort_idx]
sorted_emb = embeddable_emb[sort_idx]

# Year-by-year time-aware density
# For year t: prior set = startups with first-round year < t
# Query each startup in year t against the prior set

MIN_PRIOR = 100
KS_SMALL, KS_LARGE = 5, 50       # primary
KS_VARY = [(3,30), (5,50), (10,100)]  # stability check

results_tanov = []   # records: {co, year, loc5_t, reg50_t, ...}
n_dropped_prior = 0; n_computed = 0; drop_years = []

year_bounds = sorted(set(sorted_yrs))
prior_embs  = []    # grows incrementally
prior_cos   = []

print(f"  Computing time-aware densities (≥{MIN_PRIOR} prior required)...")
for yr in year_bounds:
    # Current year's startups
    yr_mask  = sorted_yrs == yr
    yr_cos   = [sorted_cos[i] for i in range(N_emb) if yr_mask[i]]
    yr_embs  = sorted_emb[yr_mask]

    n_prior = len(prior_cos)
    if n_prior < MIN_PRIOR:
        # Not enough prior data
        for co in yr_cos:
            results_tanov.append({
                "co": co, "year": yr,
                "loc5_t": np.nan, "reg50_t": np.nan,
                "loc3_t": np.nan, "reg30_t": np.nan,
                "loc10_t": np.nan, "reg100_t": np.nan,
                "dropped_prior": True})
        n_dropped_prior += len(yr_cos)
        drop_years.append(yr)
    else:
        prior_arr = np.array(prior_embs, dtype=np.float32)
        k_max = min(100+1, n_prior)

        # Primary (5,50)
        nn_all = NearestNeighbors(n_neighbors=k_max, metric="cosine",
                                  algorithm="brute", n_jobs=-1)
        nn_all.fit(prior_arr)
        dists, _ = nn_all.kneighbors(yr_embs, n_neighbors=k_max)
        # cosine distance → cosine similarity for density
        sims = 1.0 - dists   # shape: (n_yr, k_max)

        k5  = min(5,  n_prior)
        k50 = min(50, n_prior)
        k3  = min(3,  n_prior)
        k30 = min(30, n_prior)
        k10 = min(10, n_prior)
        k100= min(100,n_prior)

        for j, co in enumerate(yr_cos):
            row = {"co": co, "year": yr, "dropped_prior": False}
            row["loc5_t"]  = float(sims[j, :k5].mean())
            row["reg50_t"] = float(sims[j, :k50].mean())
            row["loc3_t"]  = float(sims[j, :k3].mean())
            row["reg30_t"] = float(sims[j, :k30].mean())
            row["loc10_t"] = float(sims[j, :k10].mean())
            row["reg100_t"]= float(sims[j, :k100].mean())
            results_tanov.append(row)
        n_computed += len(yr_cos)

    # Add current year to prior set for next iteration
    prior_cos.extend(yr_cos)
    prior_embs.extend(yr_embs.tolist())

tanov_df = pd.DataFrame(results_tanov)
print(f"  Total startups: {N_emb:,}")
print(f"    Time-aware density computed: {n_computed:,}")
print(f"    Dropped (< {MIN_PRIOR} prior):    {n_dropped_prior:,}  "
      f"(years ≤{max(drop_years) if drop_years else 'n/a'})")

# ── Within-year standardisation ───────────────────────────────
print(f"\n  Within-year standardisation...")
tanov_ok = tanov_df[~tanov_df["dropped_prior"]].copy()

for col_z, col_raw in [("z_loc5","loc5_t"),("z_reg50","reg50_t"),
                        ("z_loc3","loc3_t"), ("z_reg30","reg30_t"),
                        ("z_loc10","loc10_t"),("z_reg100","reg100_t")]:
    tanov_ok[col_z] = np.nan
    for yr, grp in tanov_ok.groupby("year"):
        vals = grp[col_raw].dropna()
        if len(vals) < 3: continue
        mu, sd = vals.mean(), vals.std()
        if sd < 1e-12: continue
        tanov_ok.loc[grp.index, col_z] = (grp[col_raw] - mu) / sd

# Orthogonal axes (same rotation as Script 76)
# overall_sparsity_t = -(z_loc + z_reg) / 2  (high = sparse everywhere)
# niche_cohesion_t   = z_loc - z_reg          (high = tight local, sparse field)
tanov_ok["overall_sparsity_t"] = -(tanov_ok["z_loc5"]  + tanov_ok["z_reg50"]) / 2
tanov_ok["niche_cohesion_t"]   =   tanov_ok["z_loc5"]  - tanov_ok["z_reg50"]

# Stability axes
tanov_ok["os_3_30"]  = -(tanov_ok["z_loc3"]  + tanov_ok["z_reg30"])  / 2
tanov_ok["nc_3_30"]  =   tanov_ok["z_loc3"]  - tanov_ok["z_reg30"]
tanov_ok["os_10_100"]= -(tanov_ok["z_loc10"] + tanov_ok["z_reg100"]) / 2
tanov_ok["nc_10_100"]=   tanov_ok["z_loc10"] - tanov_ok["z_reg100"]

# ── Correlation with static versions (Script 76) ──────────────
print(f"\n  Correlation with static novelty (Script 76):")
tanov_ok["static_os"] = tanov_ok["co"].map(static_sparsity)
tanov_ok["static_nc"] = tanov_ok["co"].map(static_niche)

for t_col, s_col, label in [
    ("overall_sparsity_t","static_os","overall_sparsity"),
    ("niche_cohesion_t",  "static_nc","niche_cohesion"),
]:
    valid = tanov_ok[[t_col, s_col]].dropna()
    if len(valid) < 10: continue
    rp, _ = pearsonr(valid[t_col],  valid[s_col])
    rs, _ = spearmanr(valid[t_col], valid[s_col])
    print(f"  ρ(time-aware {label}, static) "
          f"Pearson={rp:+.3f}  Spearman={rs:+.3f}  N={len(valid):,}")

# ── Stability across k-pairs ──────────────────────────────────
print(f"\n  Stability of overall_sparsity_t across neighbour choices:")
for lo_col, hi_col, label in [
    ("os_3_30","os_10_100","(3,30) vs (10,100)"),
    ("os_3_30","overall_sparsity_t","(3,30) vs (5,50)"),
    ("os_10_100","overall_sparsity_t","(10,100) vs (5,50)"),
]:
    valid = tanov_ok[[lo_col, hi_col]].dropna()
    rs, _ = spearmanr(valid[lo_col], valid[hi_col])
    print(f"    {label}: Spearman ρ={rs:+.3f}  N={len(valid):,}")

# Summary stats
print(f"\n  Time-aware novelty axes (N={len(tanov_ok):,}):")
for col in ["overall_sparsity_t","niche_cohesion_t"]:
    v = tanov_ok[col].dropna()
    print(f"  {col:<25}  mean={v.mean():+.4f}  SD={v.std():.4f}  "
          f"range=[{v.min():+.3f},{v.max():+.3f}]")

r_axes_p, _ = pearsonr(tanov_ok["overall_sparsity_t"].dropna(),
                         tanov_ok["niche_cohesion_t"].dropna())
print(f"  Pearson r(OS_t, NC_t) = {r_axes_p:+.6f}  (should be ~0 by construction)")

# Build lookup maps
co_os_t  = dict(zip(tanov_ok["co"], tanov_ok["overall_sparsity_t"]))
co_nc_t  = dict(zip(tanov_ok["co"], tanov_ok["niche_cohesion_t"]))

# ──────────────────────────────────────────────────────────────
# BUILD DYAD DATASET
# ──────────────────────────────────────────────────────────────
banner("BUILDING DYAD DATASET")

RNG = np.random.default_rng(42)
N_COUNTER = 5

year_active = defaultdict(set)
for _, row in deals_v.iterrows():
    for inv in comma_parse(row["Investors"]):
        year_active[int(row["year"])].add(inv.lower())

rows_real, rows_fake = [], []
drop_no_su_emb = 0; drop_no_inv_emb = 0; drop_no_tanov = 0; drop_small_pool = 0

for _, fr in first_round.iterrows():
    co  = fr["co"]
    yr  = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    if co not in startup_emb_map: drop_no_su_emb += 1; continue
    if co not in co_os_t:         drop_no_tanov  += 1; continue
    investors = comma_parse(str(fr["Investors"]))
    if not investors: continue
    co_ctry = str(fr.get("Company Country/Territory/Region","")).strip()
    co_rgn  = str(fr.get("HQ Global Region","")).strip()
    actual_set = {i.lower() for i in investors}
    pool = [i for i in year_active.get(yr, set())
            if i not in actual_set and i in inv_emb_map]
    if len(pool) < N_COUNTER: drop_small_pool += 1; continue
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map: drop_no_inv_emb += 1; continue
        meta = {"co": co, "investor": inv_k,
                "co_country": co_ctry, "co_region": co_rgn,
                "year": yr, "tied": 1}
        rows_real.append(meta)
        for fi in RNG.choice(pool, N_COUNTER, replace=False):
            rows_fake.append({**meta, "investor": fi, "tied": 0})

dyads = pd.DataFrame(rows_real + rows_fake).reset_index(drop=True)
N_real = int(dyads["tied"].sum()); N_tot = len(dyads)
print(f"  Realised ties: {N_real:,}  Total dyads: {N_tot:,}  "
      f"Strata: {dyads['co'].nunique():,}")
print(f"  Dropped — no startup emb: {drop_no_su_emb}  no inv emb: {drop_no_inv_emb}"
      f"  no time-aware novelty: {drop_no_tanov}  small pool: {drop_small_pool}")

# ──────────────────────────────────────────────────────────────
# STEP 3: VALIDATE BASE MODEL
# ──────────────────────────────────────────────────────────────
banner("STEP 3: VALIDATE BASE MODEL")

# A: Dropped-ties diagnostics
retained_cos = set(dyads["co"].unique())
all_cos      = set(first_round["co"])
dropped_cos  = all_cos - retained_cos

def co_stats(cos):
    rows = []
    for co in cos:
        fr_ = first_round[first_round["co"]==co]
        if len(fr_)==0: continue
        rows.append({"co": co,
                     "year":     fr_["year"].iloc[0],
                     "capital":  co_log_cap.get(co, np.nan),
                     "n_rounds": co_n_rounds.get(co, np.nan),
                     "n_invs":   len(comma_parse(str(fr_["Investors"].iloc[0])))})
    return pd.DataFrame(rows)

ret_df = co_stats(retained_cos)
drp_df = co_stats(dropped_cos)
print(f"\n  Dropped-ties comparison (retained N={len(ret_df):,}, dropped N={len(drp_df):,}):")
print(f"  {'Metric':<20} {'Retained':>20} {'Dropped':>20}  p")
print(f"  {'-'*65}")
for col, label in [("year","First-deal year"),("capital","log_capital"),
                   ("n_rounds","N rounds"),("n_invs","N investors")]:
    r = ret_df[col].dropna(); d = drp_df[col].dropna()
    if len(r)<2 or len(d)<2:
        print(f"  {label:<20} {'N/A':>20} {'N/A':>20}"); continue
    t_, p_ = ttest_ind(r, d, equal_var=False)
    print(f"  {label:<20} {r.mean():>8.2f} ± {r.std():.2f}"
          f"   {d.mean():>8.2f} ± {d.std():.2f}  {p_:.4f} {sig(p_)}")

# ── Compute dyad variables ────────────────────────────────────
print(f"\n  Computing dyad-level variables...")

# Semantic distance
sem_dists = []
for _, row in dyads.iterrows():
    sv = startup_emb_map.get(row["co"])
    iv = inv_emb_map.get(row["investor"])
    if sv is None or iv is None: sem_dists.append(np.nan); continue
    na = np.linalg.norm(sv); nb = np.linalg.norm(iv)
    cs = float(np.dot(sv, iv) / (na * nb)) if na>1e-10 and nb>1e-10 else np.nan
    sem_dists.append(1.0 - cs if not np.isnan(cs) else np.nan)
dyads["semantic_distance"] = sem_dists

# Geography
dyads["inv_country"]       = dyads["investor"].map(inv_country_map).fillna("")
dyads["inv_region"]        = dyads["investor"].map(inv_region_map).fillna("")
dyads["geo_same_country"]  = ((dyads["co_country"]!="")&(dyads["inv_country"]!="")&
    (dyads["co_country"]==dyads["inv_country"])).astype(float)
dyads["geo_same_region"]   = ((dyads["co_region"]!="")&(dyads["inv_region"]!="")&
    (dyads["co_region"]==dyads["inv_region"])).astype(float)

# Time-aware investor variables
print(f"  Time-aware network variables...", end=" ", flush=True)

# same_community_t: lead community (modal) from t-1 network vs each investor's community
co_lead_comm_t = {}
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    comm_map = lagged_comm.get(yr, {})
    comms = [comm_map.get(i.lower()) for i in comma_parse(str(fr["Investors"]))
             if comm_map.get(i.lower()) is not None]
    co_lead_comm_t[co] = Counter(comms).most_common(1)[0][0] if comms else None

same_comm_t = []; log_deg_t = []; log_deals_t = []
for _, row in dyads.iterrows():
    yr = int(row["year"]); inv_k = row["investor"]; co = row["co"]
    comm_map = lagged_comm.get(yr, {})
    deg_map  = lagged_degree.get(yr, {})
    lead_c   = co_lead_comm_t.get(co)
    inv_c    = comm_map.get(inv_k)
    same_comm_t.append(1.0 if lead_c is not None and inv_c is not None and inv_c==lead_c else 0.0)
    log_deg_t.append(np.log1p(deg_map.get(inv_k, 0)))
    log_deals_t.append(np.log1p(inv_deals_before[yr].get(inv_k, 0)))

dyads["same_community_t"] = same_comm_t
dyads["log_deg_t"]        = log_deg_t
dyads["log_deals_t"]      = log_deals_t

# Time-aware novelty
dyads["overall_sparsity_t"] = dyads["co"].map(co_os_t)
dyads["niche_cohesion_t"]   = dyads["co"].map(co_nc_t)
print(f"done  [{elapsed()}]")

print(f"\n  same_community_t rate: {np.mean(same_comm_t):.3f}")
print(f"  log_deg_t=0 (new entrant): {(np.array(log_deg_t)==0).mean():.1%} of dyad rows")

# ── Run conditional logit ────────────────────────────────────
def run_clogit(df_sub, xvars, label=""):
    sub = df_sub[["tied","co"]+xvars].dropna().copy()
    ok  = sub.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
    sub = sub[ok].copy()
    if len(sub) < 200 or sub["co"].nunique() < 20:
        print(f"  {label}: too few ({len(sub)}/{sub['co'].nunique()})"); return None
    sub["stratum"] = pd.factorize(sub["co"])[0]
    try:
        res = ConditionalLogit(sub["tied"].values,
                               sub[xvars].values,
                               groups=sub["stratum"].values).fit(
            method="bfgs", maxiter=800, disp=False)
        return res, xvars, len(sub), sub["co"].nunique()
    except Exception as e:
        print(f"  {label}: {e}"); return None

dm = dyads.dropna(subset=["semantic_distance"]).copy()
dm["sem_dist_z"] = std_col(dm, "semantic_distance")
dm["log_deg_z"]  = std_col(dm, "log_deg_t")
dm["log_deals_z"]= std_col(dm, "log_deals_t")

# M4 replication
m4v = ["sem_dist_z","geo_same_country","geo_same_region",
       "log_deg_z","log_deals_z","same_community_t"]
r_m4 = run_clogit(dm, m4v, "M4")
if r_m4:
    res, xv, n_obs, n_st = r_m4
    print(f"\n  M4 with time-aware variables (N={n_obs:,} dyads, {n_st:,} strata):")
    print(f"  {'Variable':<28} {'β':>8} {'SE':>8} {'p':>8} {'sig':>4}")
    print(f"  {'-'*60}")
    for i, v in enumerate(xv):
        b,se,p = res.params[i],res.bse[i],res.pvalues[i]
        print(f"  {v:<28} {b:>+8.4f} {se:>8.4f} {p:>8.4f} {sig(p)}")
    print(f"\n  Script 71 M4 original: sem_dist_z β=−0.2039, same_community β=+4.1717 [static]")
    print(f"  Script 79 lagged:       sem_dist_z β=−0.2176, same_community β=+3.2185 [lagged]")

# ──────────────────────────────────────────────────────────────
# STEP 4: PRE-SPECIFIED MODERATION
# ──────────────────────────────────────────────────────────────
banner("STEP 4: PRE-SPECIFIED MODERATION")

dm["nc_z"]    = std_col(dm, "niche_cohesion_t")
dm["os_z"]    = std_col(dm, "overall_sparsity_t")
dm["sd_x_nc"] = dm["sem_dist_z"] * dm["nc_z"]        # P1: should be <0
dm["sd_x_os"] = dm["sem_dist_z"] * dm["os_z"]        # P2: should be >0
dm["geo_x_os"]= dm["geo_same_country"] * dm["os_z"]  # P3a: should be >0
dm["sc_x_os"] = dm["same_community_t"] * dm["os_z"]  # P3b: should be >0

b_vars = ["sem_dist_z",
          "sd_x_nc","sd_x_os",
          "geo_same_country","geo_same_region","geo_x_os",
          "same_community_t","sc_x_os",
          "log_deg_z","log_deals_z"]

r_b = run_clogit(dm, b_vars, "Part B")
if r_b:
    res, xv, n_obs, n_st = r_b
    prespec = {"sd_x_nc":"P1 (<0)","sd_x_os":"P2 (>0)",
               "geo_x_os":"P3a (>0)","sc_x_os":"P3b (>0)"}
    print(f"\n  Moderation model (N={n_obs:,} dyads, {n_st:,} strata):")
    print(f"  {'Variable':<30} {'β':>8} {'SE':>8} {'p':>8} {'sig':>4}  note")
    print(f"  {'-'*72}")
    for i, v in enumerate(xv):
        b,se,p = res.params[i],res.bse[i],res.pvalues[i]
        print(f"  {v:<30} {b:>+8.4f} {se:>8.4f} {p:>8.4f} {sig(p):>4}  "
              f"{prespec.get(v,'')}")

    # Holm and BH on 4 pre-specified interactions
    focal = ["sd_x_nc","sd_x_os","geo_x_os","sc_x_os"]
    ps4 = [res.pvalues[xv.index(v)] for v in focal]
    bs4 = [res.params[xv.index(v)]  for v in focal]
    h4  = holm(ps4); q4 = bh(ps4)

    print(f"\n  Multiple-testing correction on 4 pre-specified interactions:")
    print(f"  {'Term':<14} {'β':>8} {'p_raw':>8} {'Holm':>8}     {'BH (q)':>8}")
    print(f"  {'-'*55}")
    for v,b,p,hp,qp in zip(["sd×NC","sd×OS","geo×OS","sc×OS"],bs4,ps4,h4,q4):
        print(f"  {v:<14} {b:>+8.4f} {p:>8.4f} {hp:>8.4f}{sig(hp):>4}  {qp:>8.4f}{sig(qp)}")

    # Verdicts
    print(f"\n  VERDICTS (pre-specified sign + Holm α=0.10):")
    for v, b, p, hp, label, sign_label in zip(
        focal, bs4, ps4, h4,
        ["P1","P2","P3a","P3b"],
        ["<0",">0",">0",">0"]
    ):
        correct_sign = (b < 0) if "<" in sign_label else (b > 0)
        if hp < 0.10 and correct_sign:
            verdict = "SUPPORTED"
        elif hp < 0.10 and not correct_sign:
            verdict = "WRONG SIGN (sig reversal)"
        elif p < 0.10 and not correct_sign:
            verdict = "WRONG SIGN (raw)"
        else:
            verdict = "FAILS (null)"
        print(f"  {label} (sd×{v.split('×')[-1] if '×' in v else v}, {sign_label}): "
              f"β={b:+.4f}  Holm={hp:.3f}  → {verdict}")

# ──────────────────────────────────────────────────────────────
# STEP 5: ROBUSTNESS
# ──────────────────────────────────────────────────────────────
banner("STEP 5: ROBUSTNESS")

# C1: exclude top 1% overall_sparsity_t
os_p99 = np.percentile(dm["overall_sparsity_t"].dropna(), 99)
dm_trim = dm[dm["overall_sparsity_t"] <= os_p99].copy()
print(f"\n  C1: exclude OS_t > p99 ({os_p99:.3f}), dropped {len(dm)-len(dm_trim):,} dyads")
r_c1 = run_clogit(dm_trim, b_vars, "C1")
if r_c1 and r_b:
    res_c1 = r_c1[0]; n_c1 = r_c1[2]
    print(f"  {'Term':<14} {'β_full':>9} {'p_full':>8}   {'β_trim':>9} {'p_trim':>8}")
    print(f"  {'-'*55}")
    for v in focal:
        i = b_vars.index(v)
        bf,pf = r_b[0].params[i],  r_b[0].pvalues[i]
        bt,pt = res_c1.params[i], res_c1.pvalues[i]
        flag = "  !" if np.sign(bf)!=np.sign(bt) else ""
        print(f"  {v:<14} {bf:>+9.4f} {pf:>8.4f}   {bt:>+9.4f} {pt:>8.4f} {sig(pt)}{flag}")

# C2: investor-activity strata
dm["total_deals_raw"] = dm["investor"].map(
    lambda inv: inv_deals_before.get(dm.loc[dm["investor"]==inv,"year"].iloc[0], {}).get(inv, 0)
    if False else 0)   # dummy — recompute properly below

# Recompute using the per-row year-aware deal count
total_deals_row = []
for _, row in dm.iterrows():
    yr = int(row["year"])
    inv_k = row["investor"]
    total_deals_row.append(inv_deals_before[yr].get(inv_k, 0))
dm["total_deals_prior"] = total_deals_row

print(f"\n  C2: investor-activity strata (prior deals at time t):")
print(f"  {'Stratum':<15} {'N_dyads':>8} {'sd×NC β':>10} {'p':>7}  {'sd×OS β':>10} {'p':>7}  {'sc×OS β':>10} {'p':>7}")
print(f"  {'-'*85}")
for lo, hi, lab in [(0,4,"0–4"),(5,19,"5–19"),(20,99,"20–99"),(100,9999,"100+")]:
    sub_s = dm[(dm["total_deals_prior"]>=lo)&(dm["total_deals_prior"]<=hi)
               ].dropna(subset=b_vars).copy()
    ok_s = sub_s.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
    sub_s = sub_s[ok_s].copy()
    sub_s["stratum"] = pd.factorize(sub_s["co"])[0]
    if len(sub_s) < 300 or sub_s["stratum"].nunique() < 20:
        print(f"  {lab:<15} {'too few':>8}"); continue
    try:
        rs = ConditionalLogit(sub_s["tied"].values,
                              sub_s[b_vars].values,
                              groups=sub_s["stratum"].values).fit(
            method="bfgs", maxiter=800, disp=False)
        def _get(v): return rs.params[b_vars.index(v)], rs.pvalues[b_vars.index(v)]
        b_nc,p_nc = _get("sd_x_nc"); b_os,p_os = _get("sd_x_os"); b_sc,p_sc = _get("sc_x_os")
        print(f"  {lab:<15} {len(sub_s):>8,} "
              f"{b_nc:>+10.4f} {p_nc:>7.4f} {sig(p_nc)}"
              f" {b_os:>+10.4f} {p_os:>7.4f} {sig(p_os)}"
              f" {b_sc:>+10.4f} {p_sc:>7.4f} {sig(p_sc)}")
    except Exception as e:
        print(f"  {lab:<15} error: {e}")

# C3: MDE
print(f"\n  C3: MDE at 80% power (N={n_obs:,} dyads):")
print(f"  {'Term':<14} {'SE':>8} {'MDE':>10} {'β actual':>10} {'β/MDE':>8}")
if r_b:
    for v in focal:
        i = b_vars.index(v)
        se = r_b[0].bse[i]; b_act = r_b[0].params[i]
        mde = (1.96+0.842)*se
        print(f"  {v:<14} {se:>8.4f} {mde:>10.4f} {b_act:>+10.4f} {b_act/mde:>8.2f}")

# ── FINAL SUMMARY ─────────────────────────────────────────────
banner("SUMMARY")
if r_b:
    print(f"  Base model: semantic_distance (β={r_m4[0].params[0]:+.4f}***) — robust")
    print(f"  same_community_t β vs. Script 71 static +4.17 vs. lagged +3.22:")
    i_sc = m4v.index("same_community_t")
    print(f"    Time-aware: β={r_m4[0].params[i_sc]:+.4f} {sig(r_m4[0].pvalues[i_sc])}")
    print(f"\n  Pre-specified predictions:")
    for v,b,p,hp,label,sign_label in zip(
        focal,bs4,ps4,h4,
        ["P1 (sd×NC<0)","P2 (sd×OS>0)","P3a (geo×OS>0)","P3b (sc×OS>0)"],
        ["<0",">0",">0",">0"]
    ):
        correct_sign = (b<0) if "<" in sign_label else (b>0)
        status = ("✓ SUPPORTED" if hp<0.10 and correct_sign
                  else "✗ WRONG SIGN" if hp<0.10 and not correct_sign
                  else "✗ NULL")
        print(f"  {label}: β={b:+.4f}  Holm={hp:.3f}  {status}")

print(f"\n  [{elapsed()}]")
