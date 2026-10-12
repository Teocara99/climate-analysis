"""
Script 79: Novelty × Semantic Matching in the Investor Network
================================================================
PRE-SPECIFIED (report regardless of outcome):
  P1: semantic_distance × niche_cohesion  < 0
      (semantic penalty STRONGER for niche ventures — specialists exist)
  P2: semantic_distance × overall_sparsity > 0
      (semantic penalty WEAKER for lonely outliers — no specialists exist)
  P3: geo_same_country × overall_sparsity > 0
      same_community_lagged × overall_sparsity > 0
      (for outliers, geography and relationships carry more weight)
  Failure: opposite signs OR both P1 and P2 null.

PART A: Validate base model (lagged community fix + dropped-tie diagnostics)
PART B: Pre-specified moderation (conditional logit, lagged community)
PART C: Robustness (trim outliers, investor-activity strata, MDE)
"""
import re, warnings, time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.stats import ttest_ind
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
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
    m = len(ps); o = np.argsort(ps)
    adj = np.empty(m)
    adj[o] = [min(1.0, ps[o[i]]*(m-i)) for i in range(m)]
    for i in range(1, m): adj[o[i]] = max(adj[o[i]], adj[o[i-1]])
    return adj

def bh(ps):
    ps = np.array(ps, dtype=float)
    m = len(ps); o = np.argsort(ps)
    adj = np.empty(m)
    for rank, i in enumerate(o, 1): adj[i] = min(1.0, ps[i]*m/rank)
    for i in range(m-2, -1, -1): adj[o[i]] = min(adj[o[i]], adj[o[i+1]])
    return adj

# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────
banner("LOADING DATA")

import sys; sys.path.insert(0, str(BASE))
from load_data import load_deals
deals = load_deals()
deals["co"]       = deals["Companies"].str.strip()
deals["Deal Date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals["year"]     = deals["Deal Date"].dt.year.fillna(deals["Year"]).astype("Int64")
deals_v = deals.dropna(subset=["year","Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

# First rounds
first_round = (deals_v.sort_values("Deal Date")
               .groupby("co").first()
               .reset_index()
               [["co","year","Investors",
                 "Company Country/Territory/Region","HQ Global Region"]])

# Investor embeddings
import pickle
emb_matrix = np.load(BASE/"data"/"investor_embeddings.npy")
inv_names_ser = pd.read_csv(BASE/"data"/"investor_names.csv")["investor"].str.strip()
inv_emb_map = {n.lower(): emb_matrix[i] for i, n in enumerate(inv_names_ser)}

# Startup embeddings
with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns_pkl = pickle.load(fh)
startup_emb_map = {n.strip(): np.array(ns_pkl["embs_raw"][i], dtype=np.float32)
                   for i, n in enumerate(ns_pkl["names"])}
# L2-normalise startup embeddings for cosine
for k in startup_emb_map:
    v = startup_emb_map[k]
    n = np.linalg.norm(v)
    if n > 1e-10: startup_emb_map[k] = v / n

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return np.nan
    return float(np.dot(a,b)/(na*nb))

# Investor metadata
inv_raw = pd.read_excel(BASE/"data"/"Investors_data.xlsx", skiprows=7)
inv_raw["inv_key"]  = inv_raw["Investors"].str.strip().str.lower()
inv_country_map = dict(zip(inv_raw["inv_key"], inv_raw["HQ Country/Territory/Region"].fillna("")))
inv_region_map  = dict(zip(inv_raw["inv_key"], inv_raw["HQ Global Region"].fillna("")))
inv_total_deals  = dict(zip(inv_raw["inv_key"],
    pd.to_numeric(inv_raw["Total Investments"], errors="coerce").fillna(0)))

# Network degree (static, for log_investor_degree)
nm = pd.read_csv(BASE/"output"/"network"/"network_metrics.csv")
nm["inv_key"] = nm["investor"].str.strip().str.lower()
inv_degree_map = dict(zip(nm["inv_key"], nm["degree"]))

# Novelty axes
nf = pd.read_csv(BASE/"output"/"two_level_novelty"/"novelty_final.csv")
nf["co"] = nf["name"].str.strip()
co_niche    = dict(zip(nf["co"], nf["niche_cohesion"]))
co_sparsity = dict(zip(nf["co"], nf["overall_sparsity"]))

# CPU data for capital
cdf = pd.read_csv(BASE/"output"/"cpu"/"cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
co_log_cap = dict(zip(cdf["co"], cdf["log_capital"]))
co_country = dict(zip(cdf["co"], cdf["hq_country"]))
co_region  = dict(zip(cdf["co"], cdf["hq_region"]))
co_capital_all = dict(zip(cdf["co"], cdf["Total Raised"]))
co_n_rounds = dict(zip(cdf["co"], cdf["n_rounds"]))

print(f"  Investor embeddings: {emb_matrix.shape}  ({len(inv_emb_map):,} names)")
print(f"  Startup embeddings: {len(startup_emb_map):,}")
print(f"  First rounds: {len(first_round):,}")

# ──────────────────────────────────────────────────────────────
# A2: TIME-LAGGED LOUVAIN COMMUNITIES
# ──────────────────────────────────────────────────────────────
banner("A2: TIME-LAGGED LOUVAIN COMMUNITIES")
print("  Building cumulative co-investment network for each year cutoff...")

years_needed = sorted(first_round["year"].dropna().unique())
min_yr, max_yr = int(min(years_needed)), int(max(years_needed))
print(f"  Year range: {min_yr}–{max_yr}  ({len(years_needed)} cutoffs)")

# Build edge list per year incrementally
edges_by_year = defaultdict(lambda: defaultdict(float))   # year → {(i,j): weight}
for _, row in deals_v.iterrows():
    yr = row["year"]
    invs = [x.lower() for x in comma_parse(row["Investors"])]
    for a, b in combinations(invs, 2):
        u, v = (a,b) if a<b else (b,a)
        edges_by_year[yr][(u,v)] += 1

# Cumulative community assignments: lagged_comm[yr][inv_key] = community_id
lagged_comm = {}   # yr → {inv_key: comm}
cumulative_edges = defaultdict(float)

all_years = sorted(set(deals_v["year"].unique()) | set(years_needed))
for yr in range(min_yr-1, max_yr+2):
    # Add this year's edges to cumulative graph
    if yr in edges_by_year:
        for (u,v), w in edges_by_year[yr].items():
            cumulative_edges[(u,v)] += w

    # Build graph and detect communities
    G = nx.Graph()
    for (u,v), w in cumulative_edges.items():
        G.add_edge(u, v, weight=w)

    if G.number_of_edges() < 10:
        lagged_comm[yr+1] = {}   # next year uses communities from ≤ yr
        continue

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    lagged_comm[yr+1] = partition   # communities available AT START of year yr+1
    if yr in years_needed or yr == max_yr:
        n_comm = len(set(partition.values()))
        print(f"  Cutoff ≤{yr}: {G.number_of_nodes():,} investors, "
              f"{G.number_of_edges():,} edges, {n_comm} communities  [{elapsed()}]")

print(f"  Lagged community maps built for years: {sorted(lagged_comm.keys())[:5]}...{sorted(lagged_comm.keys())[-3:]}")

# Count coverage
first_years = first_round["year"].dropna().astype(int)
n_no_comm = 0; n_with_comm = 0
for _, fr in first_round.iterrows():
    yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    comm_map = lagged_comm.get(yr, {})
    for inv in comma_parse(fr["Investors"]):
        if inv.lower() in comm_map: n_with_comm += 1
        else: n_no_comm += 1
total_ties = n_with_comm + n_no_comm
print(f"\n  Actual-investor community coverage:")
print(f"    With lagged community:    {n_with_comm:,} ({n_with_comm/total_ties:.1%})")
print(f"    Without (new entrant):    {n_no_comm:,}  → same_community_lagged = 0 (treated as no-community)")

# ──────────────────────────────────────────────────────────────
# BUILD DYAD DATASET (same structure as Script 71)
# ──────────────────────────────────────────────────────────────
banner("BUILDING DYAD DATASET (5 counterfactuals, year-matched)")

RNG = np.random.default_rng(42)
N_COUNTER = 5

year_active = {}
for _, row in deals_v.iterrows():
    y = row["year"]
    if y not in year_active: year_active[y] = set()
    for inv in comma_parse(row["Investors"]):
        year_active[y].add(inv.lower())

rows_real, rows_fake = [], []
drop_no_startup_emb = 0; drop_no_inv_emb = 0; drop_small_pool = 0

for _, fr in first_round.iterrows():
    co  = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    inv_str = fr["Investors"]
    co_ctry = str(fr.get("Company Country/Territory/Region","")).strip()
    co_rgn  = str(fr.get("HQ Global Region","")).strip()
    if co not in startup_emb_map: drop_no_startup_emb += 1; continue
    investors = comma_parse(inv_str)
    if not investors: continue
    actual_set = {i.lower() for i in investors}
    pool = [i for i in year_active.get(yr,set()) if i not in actual_set and i in inv_emb_map]
    if len(pool) < N_COUNTER: drop_small_pool += 1; continue
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map: drop_no_inv_emb += 1; continue
        rows_real.append({"co":co,"investor":inv_k,"co_country":co_ctry,
                          "co_region":co_rgn,"year":yr,"tied":1})
        for fi in RNG.choice(pool, N_COUNTER, replace=False):
            rows_fake.append({"co":co,"investor":fi,"co_country":co_ctry,
                              "co_region":co_rgn,"year":yr,"tied":0})

dyads = pd.DataFrame(rows_real + rows_fake).reset_index(drop=True)
N_real = dyads["tied"].sum(); N_total = len(dyads)
print(f"\n  Realised ties: {N_real:,}  Total dyads: {N_total:,}  Strata: {dyads['co'].nunique():,}")
print(f"  Dropped — no startup emb: {drop_no_startup_emb}  no inv emb: {drop_no_inv_emb}  small pool: {drop_small_pool}")

# ──────────────────────────────────────────────────────────────
# A1: DROPPED-TIE DIAGNOSTICS
# ──────────────────────────────────────────────────────────────
banner("A1: DROPPED-TIE DIAGNOSTICS")

retained_cos = set(dyads["co"].unique())
all_cos = set(first_round["co"])
dropped_cos = all_cos - retained_cos

def co_stats(cos):
    rows = []
    for co in cos:
        fr_ = first_round[first_round["co"]==co]
        if len(fr_)==0: continue
        rows.append({
            "co": co,
            "year": fr_["year"].iloc[0],
            "capital": co_log_cap.get(co, np.nan),
            "n_rounds": co_n_rounds.get(co, np.nan),
            "n_invs": len(comma_parse(str(fr_["Investors"].iloc[0]))),
        })
    return pd.DataFrame(rows)

retained_df = co_stats(retained_cos)
dropped_df  = co_stats(dropped_cos)
print(f"\n  {'Metric':<22} {'Retained (N={:,})'.format(len(retained_df)):>22} "
      f"{'Dropped (N={:,})'.format(len(dropped_df)):>22}  {'t-test p':>10}")
for col, label in [("year","First-deal year"),("capital","log_capital"),
                   ("n_rounds","N rounds"),("n_invs","N investors")]:
    r = retained_df[col].dropna(); d = dropped_df[col].dropna()
    if len(r)<2 or len(d)<2:
        print(f"  {label:<22} {'N/A':>22} {'N/A':>22}")
        continue
    from scipy.stats import ttest_ind as ttind
    t_, p_ = ttind(r, d, equal_var=False)
    print(f"  {label:<22} {r.mean():>10.2f} ± {r.std():.2f}  "
          f"{d.mean():>10.2f} ± {d.std():.2f}  {p_:>10.4f} {sig(p_)}")

# Investor-level: dropped ties = investors in real deals who had no embedding
# (identified above as drop_no_inv_emb); compare with retained
print(f"\n  Dropped investor-ties: {drop_no_inv_emb:,} (no SPECTER2 embedding)")
print(f"  These are likely newer/smaller investors not yet embedded at data cut.")
retained_inv = set(dyads["investor"].unique())
all_inv_in_deals = set()
for _, fr in first_round.iterrows():
    for inv in comma_parse(str(fr["Investors"])): all_inv_in_deals.add(inv.lower())
dropped_inv = all_inv_in_deals - retained_inv - set(inv_emb_map.keys())
print(f"  Unique dropped investors: {len(dropped_inv):,}")
r_deals = [inv_total_deals.get(i,0) for i in retained_inv if inv_total_deals.get(i,0)>0]
d_deals = [inv_total_deals.get(i,0) for i in dropped_inv  if inv_total_deals.get(i,0)>0]
if r_deals and d_deals:
    t_, p_ = ttest_ind(r_deals, d_deals, equal_var=False)
    print(f"  Total deals — retained: {np.mean(r_deals):.1f}  dropped: {np.mean(d_deals):.1f}  p={p_:.4f} {sig(p_)}")

# ── COMPUTE DYAD VARIABLES ───────────────────────────────────
banner("COMPUTING DYAD VARIABLES")

print("  Semantic distances...", end=" ", flush=True)
sem = []
for _, row in dyads.iterrows():
    sv = startup_emb_map.get(row["co"]); iv = inv_emb_map.get(row["investor"])
    if sv is None or iv is None: sem.append(np.nan); continue
    cs = cosine_sim(sv, iv)
    sem.append(1.0-cs if not np.isnan(cs) else np.nan)
dyads["semantic_distance"] = sem
print("done")

# Geographic
dyads["inv_country"] = dyads["investor"].map(inv_country_map).fillna("")
dyads["inv_region"]  = dyads["investor"].map(inv_region_map).fillna("")
dyads["geo_same_country"] = ((dyads["co_country"]!="")&(dyads["inv_country"]!="")&
    (dyads["co_country"]==dyads["inv_country"])).astype(float)
dyads["geo_same_region"] = ((dyads["co_region"]!="")&(dyads["inv_region"]!="")&
    (dyads["co_region"]==dyads["inv_region"])).astype(float)

# Investor activity
dyads["log_investor_degree"] = np.log1p(dyads["investor"].map(inv_degree_map).fillna(0))
dyads["log_investor_deals"]  = np.log1p(dyads["investor"].map(inv_total_deals).fillna(0))

# Novelty measures (startup-level, from novelty_final.csv)
dyads["niche_cohesion"]   = dyads["co"].map(co_niche)
dyads["overall_sparsity"] = dyads["co"].map(co_sparsity)

# Lagged community
print("  Lagged community...", end=" ", flush=True)
co_lead_comm_lagged = {}
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    comm_map = lagged_comm.get(yr, {})
    investors = comma_parse(str(fr["Investors"]))
    comms = [comm_map.get(i.lower()) for i in investors if comm_map.get(i.lower()) is not None]
    if comms:
        from collections import Counter
        co_lead_comm_lagged[co] = Counter(comms).most_common(1)[0][0]
    else:
        co_lead_comm_lagged[co] = None

same_comm_lag = []
for _, row in dyads.iterrows():
    yr     = row["year"]
    inv_k  = row["investor"]
    co     = row["co"]
    lead_c = co_lead_comm_lagged.get(co)
    if lead_c is None:
        same_comm_lag.append(0.0)
        continue
    comm_map = lagged_comm.get(int(yr), {})
    inv_c    = comm_map.get(inv_k)
    same_comm_lag.append(1.0 if inv_c is not None and inv_c == lead_c else 0.0)
dyads["same_community_lagged"] = same_comm_lag
print(f"done  rate={np.mean(same_comm_lag):.3f} (cf. static: from Script 71 ~0.203)")

# ── A3: REPLICATE M4 WITH LAGGED COMMUNITY ──────────────────
banner("A3: REPLICATE SCRIPT 71 M4 WITH LAGGED COMMUNITY")

def run_clogit(df_sub, xvars, label):
    sub = df_sub[["tied","co"]+xvars].dropna().copy()
    ok  = sub.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
    sub = sub[ok].copy()
    sub["stratum"] = pd.factorize(sub["co"])[0]
    n_obs = len(sub); n_st = sub["stratum"].nunique()
    if n_obs < 100 or n_st < 10:
        print(f"  {label}: too few ({n_obs}/{n_st}), skip"); return None
    try:
        res = ConditionalLogit(sub["tied"].values,
                               sub[xvars].values,
                               groups=sub["stratum"].values).fit(
            method="bfgs", maxiter=600, disp=False)
        return res, xvars, n_obs, n_st
    except Exception as e:
        print(f"  {label}: {e}"); return None

def std_col(df_sub, col):
    s = df_sub[col].dropna(); mu, sd = s.mean(), s.std()
    return (df_sub[col]-mu)/sd if sd>1e-10 else df_sub[col]

dm = dyads.dropna(subset=["semantic_distance"]).copy()
dm["sem_dist_z"]   = std_col(dm, "semantic_distance")
dm["log_deg_z"]    = std_col(dm, "log_investor_degree")
dm["log_deals_z"]  = std_col(dm, "log_investor_deals")

m4_vars = ["sem_dist_z","geo_same_country","geo_same_region",
           "log_deg_z","log_deals_z","same_community_lagged"]
r_a3 = run_clogit(dm, m4_vars, "M4-lagged")
if r_a3:
    res, xv, n_obs, n_st = r_a3
    print(f"\n  M4 with lagged community (N={n_obs:,} dyads, {n_st:,} strata):")
    print(f"  {'Variable':<28} {'β':>8} {'SE':>8} {'p':>8} {'sig':>4}")
    print(f"  {'-'*58}")
    for i, v in enumerate(xv):
        b,se,p = res.params[i],res.bse[i],res.pvalues[i]
        print(f"  {v:<28} {b:>+8.4f} {se:>8.4f} {p:>8.4f} {sig(p)}")
    print(f"\n  Compare Script 71 M4: sem_dist_z β=−0.2039, same_community β=+4.1717")

# ──────────────────────────────────────────────────────────────
# PART B: PRE-SPECIFIED MODERATION
# ──────────────────────────────────────────────────────────────
banner("PART B: PRE-SPECIFIED MODERATION (lagged community)")

dm["nc_z"]    = std_col(dm, "niche_cohesion")
dm["os_z"]    = std_col(dm, "overall_sparsity")
dm["sd_x_nc"] = dm["sem_dist_z"] * dm["nc_z"]    # P1
dm["sd_x_os"] = dm["sem_dist_z"] * dm["os_z"]    # P2
dm["geo_x_os"]= dm["geo_same_country"] * dm["os_z"]  # P3a
dm["sc_x_os"] = dm["same_community_lagged"] * dm["os_z"]  # P3b

b_vars = ["sem_dist_z",
          "sd_x_nc","sd_x_os",                          # P1, P2
          "geo_same_country","geo_same_region",
          "geo_x_os",                                   # P3a
          "same_community_lagged",
          "sc_x_os",                                    # P3b
          "log_deg_z","log_deals_z"]

r_b = run_clogit(dm, b_vars, "Part B")
if r_b:
    res, xv, n_obs, n_st = r_b
    print(f"\n  Part B model (N={n_obs:,} dyads, {n_st:,} strata):")
    print(f"  {'Variable':<30} {'β':>8} {'SE':>8} {'p':>8} {'sig':>4}  note")
    print(f"  {'-'*70}")
    prespec = {"sd_x_nc":"P1 (<0)","sd_x_os":"P2 (>0)","geo_x_os":"P3a (>0)","sc_x_os":"P3b (>0)"}
    for i, v in enumerate(xv):
        b,se,p = res.params[i],res.bse[i],res.pvalues[i]
        note = prespec.get(v,"")
        print(f"  {v:<30} {b:>+8.4f} {se:>8.4f} {p:>8.4f} {sig(p):>4}  {note}")

    # Holm + BH on the 4 pre-specified interactions
    ps4 = [res.pvalues[xv.index(v)] for v in ["sd_x_nc","sd_x_os","geo_x_os","sc_x_os"]]
    bs4 = [res.params[xv.index(v)]  for v in ["sd_x_nc","sd_x_os","geo_x_os","sc_x_os"]]
    holm4 = holm(ps4); bh4 = bh(ps4)
    print(f"\n  Holm/BH on 4 pre-specified interactions:")
    print(f"  {'Term':<12} {'β':>8} {'p_raw':>8} {'Holm':>8} {'BH':>8}")
    for v,b,p,hp,bhp in zip(["sd×NC","sd×OS","geo×OS","sc×OS"],bs4,ps4,holm4,bh4):
        print(f"  {v:<12} {b:>+8.4f} {p:>8.4f} {hp:>8.4f} {sig(hp)}  {bhp:>8.4f} {sig(bhp)}")

    # Verdicts
    print(f"\n  P1 (sd×NC < 0): β={bs4[0]:+.4f}  p={ps4[0]:.4f} {sig(ps4[0])}  "
          + ("SUPPORTED" if bs4[0]<0 and ps4[0]<0.10 else "FAILS" if ps4[0]>=0.10 else "WRONG SIGN"))
    print(f"  P2 (sd×OS > 0): β={bs4[1]:+.4f}  p={ps4[1]:.4f} {sig(ps4[1])}  "
          + ("SUPPORTED" if bs4[1]>0 and ps4[1]<0.10 else "FAILS" if ps4[1]>=0.10 else "WRONG SIGN"))
    print(f"  P3a(geo×OS> 0): β={bs4[2]:+.4f}  p={ps4[2]:.4f} {sig(ps4[2])}  "
          + ("SUPPORTED" if bs4[2]>0 and ps4[2]<0.10 else "FAILS" if ps4[2]>=0.10 else "WRONG SIGN"))
    print(f"  P3b(sc×OS > 0): β={bs4[3]:+.4f}  p={ps4[3]:.4f} {sig(ps4[3])}  "
          + ("SUPPORTED" if bs4[3]>0 and ps4[3]<0.10 else "FAILS" if ps4[3]>=0.10 else "WRONG SIGN"))

# ──────────────────────────────────────────────────────────────
# PART C: ROBUSTNESS
# ──────────────────────────────────────────────────────────────
banner("PART C: ROBUSTNESS")

# C1: exclude top 1% overall_sparsity
os_p99 = np.percentile(dm["overall_sparsity"].dropna(), 99)
dm_trim = dm[dm["overall_sparsity"] <= os_p99].copy()
print(f"\n  C1: exclude OS > p99 ({os_p99:.3f}), N trimmed={len(dm)-len(dm_trim)}")
r_c1 = run_clogit(dm_trim, b_vars, "C1-trim")
if r_c1 and r_b:
    res_c1, xv_c1, n_c1, st_c1 = r_c1
    print(f"  Compare Part B vs C1 on 4 pre-specified interactions (N={n_c1:,}):")
    print(f"  {'Term':<12} {'β_full':>9} {'p_full':>8}   {'β_trim':>9} {'p_trim':>8}")
    for v in ["sd_x_nc","sd_x_os","geo_x_os","sc_x_os"]:
        i_b  = b_vars.index(v)
        b_f, p_f = r_b[0].params[i_b],  r_b[0].pvalues[i_b]
        b_t, p_t = r_c1[0].params[i_b], r_c1[0].pvalues[i_b]
        chg = "↑" if abs(b_t)>abs(b_f) else "↓"
        print(f"  {v:<12} {b_f:>+9.4f} {p_f:>8.4f}   {b_t:>+9.4f} {p_t:>8.4f} {sig(p_t)} {chg}")

# C2: investor-activity strata
print(f"\n  C2: investor-activity strata (lifetime deals):")
print(f"  {'Stratum':<15} {'N_dyads':>8} {'sd_x_nc β':>11} {'p':>8}   {'sd_x_os β':>11} {'p':>8}")
print(f"  {'-'*70}")
dm["total_deals_raw"] = dm["investor"].map(inv_total_deals).fillna(0)
for lo, hi, lab in [(1,5,"1-5"),(6,20,"6-20"),(21,100,"21-100"),(101,9999,"101+")]:
    sub_s = dm[(dm["total_deals_raw"]>=lo)&(dm["total_deals_raw"]<=hi)].copy()
    sub_s = sub_s.dropna(subset=b_vars).copy()
    ok_s  = sub_s.groupby("co")["tied"].transform(lambda x: x.nunique()==2)
    sub_s = sub_s[ok_s].copy()
    sub_s["stratum"] = pd.factorize(sub_s["co"])[0]
    if len(sub_s)<200 or sub_s["stratum"].nunique()<20:
        print(f"  {lab:<15} {'too few':>8}"); continue
    try:
        res_s = ConditionalLogit(sub_s["tied"].values,
                                 sub_s[b_vars].values,
                                 groups=sub_s["stratum"].values).fit(
            method="bfgs", maxiter=600, disp=False)
        b_nc = res_s.params[b_vars.index("sd_x_nc")]
        p_nc = res_s.pvalues[b_vars.index("sd_x_nc")]
        b_os = res_s.params[b_vars.index("sd_x_os")]
        p_os = res_s.pvalues[b_vars.index("sd_x_os")]
        print(f"  {lab:<15} {len(sub_s):>8,} {b_nc:>+11.4f} {p_nc:>8.4f} {sig(p_nc)}"
              f"   {b_os:>+11.4f} {p_os:>8.4f} {sig(p_os)}")
    except Exception as e:
        print(f"  {lab:<15} error: {e}")

# C3: MDE at 80% power
print(f"\n  C3: MDE at 80% power for each interaction (N dyads={n_obs:,}):")
print(f"  {'Term':<12} {'SE':>8} {'MDE(α=.05,80%)':>16} {'β actual':>10} {'β/MDE':>8}")
if r_b:
    for v in ["sd_x_nc","sd_x_os","geo_x_os","sc_x_os"]:
        i_v = b_vars.index(v)
        se  = r_b[0].bse[i_v]; b_act = r_b[0].params[i_v]
        mde = (1.96+0.842)*se   # two-sided α=.05, 80% power
        print(f"  {v:<12} {se:>8.4f} {mde:>16.4f} {b_act:>+10.4f} {b_act/mde:>8.2f}")

print(f"\n  [{elapsed()}]")
