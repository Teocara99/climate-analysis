"""
Script 88: Core re-check (Part A) + H1 structural folds (Part B).

PART A: Re-run Script 79r base model M4 with tuned Louvain communities
  (from Script 87). Report sem_dist and same_community coefficients
  alongside the original 79r values.

PART B — H1 (PRE-SPECIFIED, report regardless):
  Fold investors are MORE likely to fund niche startups.
  Primary:   fold × niche_cohesion_t  > 0
  Secondary: fold × overall_sparsity_t

  1. Fold classification from deals < 2020 only. Test on 2021+ first rounds.
  2. Feasibility gate: >= 500 dyads with fold investors, MDE.
  3. Conditional logit: tie ~ fold + fold×NC_t + fold×OS_t
                             + sem_dist + geo + same_comm_tuned
                             + log_prior_deals + has_prior_network
  4. Robustness: prior-activity quartiles; full-network fold classification.
"""

import os, sys, re, warnings, pickle
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from scipy.stats import norm, pearsonr
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "h1_folds")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
N_COUNTER = 5

# ── reference values from 79r (for comparison table in Part A) ──
REF_79R = {
    "sem_dist_z":      {"beta": -0.2117, "p": 0.0000},
    "same_community_t":{"beta": +3.4420, "p": 0.0000},
    "geo_same_country":{"beta": +2.6810, "p": 0.0000},
    "log_deg_z":       {"beta": -0.5870, "p": 0.0000},
    "log_deals_z":     {"beta": +0.3900, "p": 0.0000},
}

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return "n.s."

print("=" * 70)
print("SCRIPT 88 — PART A: BASE RE-CHECK  |  PART B: H1 FOLDS")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# SHARED: Load data
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading shared data ──")

deals_df = load_deals()
deals_df["Deal Date"] = pd.to_datetime(deals_df["Deal Date"], errors="coerce")
deals_df["year"] = deals_df["Deal Date"].dt.year
deals_df["co"]   = deals_df["Companies"].str.strip()
deals_v  = deals_df.dropna(subset=["year", "Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

first_round = (
    deals_v.sort_values("Deal Date")
    .groupby("co").first()
    .reset_index()[["co", "year", "Investors",
                    "Company Country/Territory/Region", "HQ Global Region"]]
)

# Embeddings
inv_names_raw = pd.read_csv("data/investor_names.csv", header=None, names=["investor"])
inv_emb_raw2  = np.load("data/investor_embeddings.npy").astype(np.float32)
inv_norms2    = np.linalg.norm(inv_emb_raw2, axis=1, keepdims=True)
inv_norms2[inv_norms2 == 0] = 1.0
inv_emb_raw2  = inv_emb_raw2 / inv_norms2
# Drop header/nan rows then re-index to match embedding rows
inv_names_ser = [str(n).strip() for n in inv_names_raw["investor"]
                 if str(n).strip() not in ("investor", "nan", "")]
inv_emb_arr   = inv_emb_raw2[:len(inv_names_ser)]
inv_emb_map   = {n.lower(): inv_emb_arr[i] for i, n in enumerate(inv_names_ser)}

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    ns_pkl = pickle.load(f)
su_emb_names = [n.strip() for n in ns_pkl["names"]]
su_emb_raw   = np.array(ns_pkl["embs_raw"], dtype=np.float32)
su_norms     = np.linalg.norm(su_emb_raw, axis=1, keepdims=True)
su_norms[su_norms == 0] = 1.0
su_emb_arr   = su_emb_raw / su_norms
startup_emb_map = {n: su_emb_arr[i] for i, n in enumerate(su_emb_names)}

inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors", "HQ Country/Territory/Region",
                                 "HQ Global Region"])
inv_raw["inv_key"] = inv_raw["Investors"].str.strip().str.lower()
inv_country_map = dict(zip(inv_raw["inv_key"],
                           inv_raw["HQ Country/Territory/Region"].fillna("")))
inv_region_map  = dict(zip(inv_raw["inv_key"],
                           inv_raw["HQ Global Region"].fillna("")))

# Time-aware novelty (saved)
nov_t = pd.read_csv("output/novelty_t/novelty_time_aware.csv")
co_os_t = dict(zip(nov_t["Companies"], nov_t["overall_sparsity_t"]))
co_nc_t = dict(zip(nov_t["Companies"], nov_t["niche_cohesion_t"]))

# Tuned communities (Script 87)
tuned_raw = pd.read_csv("output/h3_redesign/lagged_comm_tuned.csv")
tuned_lc  = {}  # (investor, year) → community
for _, r in tuned_raw.iterrows():
    tuned_lc[(r["investor"], int(r["year"]))] = int(r["community_tuned"])

# ── Build per-year network variables ────────────────────────────────────
print("  Building cumulative network variables (degree, deal counts)…")

edges_by_year = defaultdict(list)
for _, row in deals_v.iterrows():
    invs = [x.lower() for x in comma_parse(row["Investors"])]
    yr   = int(row["year"])
    for a, b in combinations(invs, 2):
        edges_by_year[yr].append((a, b))

inv_deals_before = defaultdict(lambda: defaultdict(int))
for _, row in deals_v.iterrows():
    yr = int(row["year"])
    for inv in comma_parse(row["Investors"]):
        inv_k = inv.lower()
        for fut_yr in range(yr + 1, 2028):
            inv_deals_before[fut_yr][inv_k] += 1

# Cumulative degree per year
lagged_degree = {}
cumulative_edges = defaultdict(float)
all_years = sorted(edges_by_year.keys())
for yr in range(min(all_years) - 1, max(all_years) + 2):
    for a, b in edges_by_year.get(yr, []):
        cumulative_edges[(a, b)] = cumulative_edges.get((a, b), 0) + 1
    if cumulative_edges:
        # degree from cumulative edges
        deg = defaultdict(int)
        for (a, b) in cumulative_edges:
            deg[a] += 1; deg[b] += 1
        lagged_degree[yr + 1] = dict(deg)
    else:
        lagged_degree[yr + 1] = {}

# Active investors per year
year_active = defaultdict(set)
for _, row in deals_v.iterrows():
    for inv in comma_parse(row["Investors"]):
        year_active[int(row["year"])].add(inv.lower())

print(f"  First-round startups: {len(first_round):,}")

# ═══════════════════════════════════════════════════════════════════════════
# BUILD DYAD DATASET  (shared between Part A and B)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Building dyad dataset ──")

rows_real, rows_fake = [], []
for _, fr in first_round.iterrows():
    co  = fr["co"]
    yr  = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None or co not in startup_emb_map or co not in co_os_t:
        continue
    investors = comma_parse(str(fr["Investors"]))
    if not investors:
        continue
    co_ctry = str(fr.get("Company Country/Territory/Region", "")).strip()
    co_rgn  = str(fr.get("HQ Global Region", "")).strip()
    actual_set = {i.lower() for i in investors}
    pool = [i for i in year_active.get(yr, set())
            if i not in actual_set and i in inv_emb_map]
    if len(pool) < N_COUNTER:
        continue
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map:
            continue
        meta = {"co": co, "investor": inv_k, "co_country": co_ctry,
                "co_region": co_rgn, "year": yr, "tied": 1}
        rows_real.append(meta)
        for fi in RNG.choice(pool, N_COUNTER, replace=False):
            rows_fake.append({**meta, "investor": fi, "tied": 0})

dyads = pd.DataFrame(rows_real + rows_fake).reset_index(drop=True)
print(f"  Real ties: {dyads['tied'].sum():,}  Total dyads: {len(dyads):,}  "
      f"Strata: {dyads['co'].nunique():,}")

# Semantic distance
sem_d = []
for _, row in dyads.iterrows():
    sv = startup_emb_map.get(row["co"])
    iv = inv_emb_map.get(row["investor"])
    if sv is None or iv is None:
        sem_d.append(np.nan)
    else:
        sem_d.append(float(1.0 - sv @ iv))
dyads["semantic_distance"] = sem_d

# Geo
dyads["inv_country"]      = dyads["investor"].map(inv_country_map).fillna("")
dyads["inv_region"]       = dyads["investor"].map(inv_region_map).fillna("")
dyads["geo_same_country"] = ((dyads["co_country"] != "") &
                              (dyads["inv_country"] != "") &
                              (dyads["co_country"] == dyads["inv_country"])).astype(float)
dyads["geo_same_region"]  = ((dyads["co_region"] != "") &
                              (dyads["inv_region"] != "") &
                              (dyads["co_region"] == dyads["inv_region"])).astype(float)

# Prior activity
dyads["log_prior_deals"] = [
    float(np.log1p(inv_deals_before[int(r["year"])].get(r["investor"], 0)))
    for _, r in dyads.iterrows()
]
dyads["has_prior_network"] = (dyads["log_prior_deals"] > 0).astype(float)
dyads["log_deg_t"]  = [
    float(np.log1p(lagged_degree.get(int(r["year"]), {}).get(r["investor"], 0)))
    for _, r in dyads.iterrows()
]
dyads["log_deals_t"] = dyads["log_prior_deals"]  # same concept

# Tuned same_community_t
co_tuned_lead_comm = {}
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    comms = [tuned_lc.get((i.lower(), yr))
             for i in comma_parse(str(fr["Investors"]))
             if tuned_lc.get((i.lower(), yr)) is not None]
    co_tuned_lead_comm[co] = Counter(comms).most_common(1)[0][0] if comms else None

dyads["same_community_tuned"] = [
    1.0 if (co_tuned_lead_comm.get(r["co"]) is not None and
            tuned_lc.get((r["investor"], int(r["year"]))) == co_tuned_lead_comm.get(r["co"]))
    else 0.0
    for _, r in dyads.iterrows()
]

# Novelty
dyads["overall_sparsity_t"] = dyads["co"].map(co_os_t)
dyads["niche_cohesion_t"]   = dyads["co"].map(co_nc_t)

# Standardise
dm = dyads.dropna(subset=["semantic_distance"]).copy()
for col, col_z in [("semantic_distance","sem_dist_z"), ("log_deg_t","log_deg_z"),
                   ("log_deals_t","log_deals_z"), ("log_prior_deals","log_prior_deals_z"),
                   ("overall_sparsity_t","os_z"), ("niche_cohesion_t","nc_z")]:
    mu = dm[col].mean(); sd = dm[col].std()
    dm[col_z] = (dm[col] - mu) / sd if sd > 0 else 0.0

def has_both(grp): return grp["tied"].nunique() == 2

def run_clogit(data, b_vars, label=""):
    sub = data[["tied", "co"] + b_vars].dropna().copy()
    ok  = sub.groupby("co")["tied"].transform(lambda x: x.nunique() == 2)
    sub = sub[ok].copy()
    if len(sub) < 200 or sub["co"].nunique() < 20:
        print(f"  {label}: too few rows ({len(sub)})"); return None
    grps = sub["co"].values
    exog = sub[b_vars].values.astype(float)
    endog = sub["tied"].values.astype(float)
    mask = np.isfinite(exog).all(axis=1)
    for method in ["bfgs", "newton", "lbfgs"]:
        try:
            res = ConditionalLogit(endog[mask], exog[mask],
                                   groups=grps[mask]).fit(
                method=method, maxiter=600, disp=False)
            if np.isfinite(res.bse).all():
                return res, b_vars, int(mask.sum()), int(sub["co"].nunique()), sub.loc[mask]
        except Exception:
            continue
    print(f"  {label}: all optimisers failed"); return None

# ═══════════════════════════════════════════════════════════════════════════
# PART A: Base model with tuned communities
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART A — Base model (tuned vs original communities)")
print("=" * 70)

M4_VARS = ["sem_dist_z", "geo_same_country", "geo_same_region",
           "log_deg_z", "log_deals_z", "same_community_tuned"]

r_a = run_clogit(dm, M4_VARS, "Part A M4 tuned")
if r_a:
    res_a, xv_a, n_a, ns_a, _ = r_a
    print(f"\n  N dyads: {n_a:,}  |  N strata: {ns_a:,}")
    print()
    print(f"  {'Variable':<26}  {'79r β':>8}  {'88 β':>8}  {'SE':>7}  {'p':>7}  {'sig':>4}")
    print("  " + "─" * 66)
    for i, v in enumerate(xv_a):
        b79 = REF_79R.get(v.replace("_tuned","_t"), {}).get("beta", np.nan)
        b88 = res_a.params[i]; se = res_a.bse[i]; p = res_a.pvalues[i]
        b79_s = f"{b79:+8.4f}" if not np.isnan(b79) else "       —"
        print(f"  {v:<26}  {b79_s}  {b88:+8.4f}  {se:7.4f}  {p:7.4f}  {sig(p)}")
    print()
    print(f"  Note: '79r β' for same_community uses original Louvain (default res=1.0).")
    print(f"  '88 β' uses tuned resolution targeting 40-60 communities per year.")

# ═══════════════════════════════════════════════════════════════════════════
# PART B — H1 STRUCTURAL FOLDS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B — H1: Structural folds × niche novelty")
print("=" * 70)

# ── Step 1: Fold classification from deals < 2020 ─────────────────────────
print("\n── Step 1: Fold classification (deals < 2020) ──")

G_pre20 = nx.Graph()
for _, row in deals_v[deals_v["year"] < 2020].iterrows():
    invs = [x.lower() for x in comma_parse(row["Investors"])]
    for a, b in combinations(invs, 2):
        if G_pre20.has_edge(a, b):
            G_pre20[a][b]["weight"] += 1
        else:
            G_pre20.add_edge(a, b, weight=1)

# LCC only
lcc_nodes = max(nx.connected_components(G_pre20), key=len)
G_lcc20   = G_pre20.subgraph(lcc_nodes).copy()

print(f"  Pre-2020 network: {G_pre20.number_of_nodes():,} nodes, "
      f"{G_pre20.number_of_edges():,} edges")
print(f"  LCC: {G_lcc20.number_of_nodes():,} nodes "
      f"({100*G_lcc20.number_of_nodes()/G_pre20.number_of_nodes():.1f}%)")

partition_20 = community_louvain.best_partition(
    G_lcc20, weight="weight", random_state=42
)
n_comm_20 = len(set(partition_20.values()))
print(f"  Louvain communities (default res): {n_comm_20}")

# Tune to 40-60 if outside range
if not (40 <= n_comm_20 <= 60):
    lo_r, hi_r, best_part20 = 0.5, 4.0, partition_20
    for _ in range(20):
        mid_r = (lo_r + hi_r) / 2
        part  = community_louvain.best_partition(
            G_lcc20, weight="weight", resolution=mid_r, random_state=42)
        nc = len(set(part.values()))
        if nc < 40:   lo_r = mid_r
        elif nc > 60: hi_r = mid_r
        else: best_part20 = part; break
        if abs(nc - 50) < abs(len(set(best_part20.values())) - 50):
            best_part20 = part
    partition_20 = best_part20
    print(f"  After tuning: {len(set(partition_20.values()))} communities")

# Community-community adjacency (full, not removing focal investor)
comm_adj = set()
for u, v in G_lcc20.edges():
    cu = partition_20.get(u); cv = partition_20.get(v)
    if cu is not None and cv is not None and cu != cv:
        comm_adj.add((min(cu, cv), max(cu, cv)))

# Classify each investor in the LCC
fold_classification = {}
for inv in G_lcc20.nodes():
    own_c = partition_20.get(inv)
    nbr_comms = {partition_20.get(n) for n in G_lcc20.neighbors(inv)
                 if partition_20.get(n) is not None}
    bridged = nbr_comms - ({own_c} if own_c is not None else set())
    if len(bridged) < 2:
        fold_classification[inv] = "EMBEDDED"
        continue
    is_fold = any(
        (min(c_a, c_b), max(c_a, c_b)) in comm_adj
        for c_a, c_b in combinations(bridged, 2)
    )
    fold_classification[inv] = "FOLD" if is_fold else "HOLE"

fold_counts = Counter(fold_classification.values())
print(f"  Classification: {dict(fold_counts)}")
print(f"  FOLD investors: {fold_counts['FOLD']:,}")

# Save fold classification
fold_df = pd.DataFrame([
    {"investor": k, "classification_pre20": v}
    for k, v in fold_classification.items()
])
fold_path = os.path.join(OUT_DIR, "fold_classification_pre2020.csv")
fold_df.to_csv(fold_path, index=False)
print(f"  Saved → {fold_path}")

# ── Step 2: Restrict dyads to 2021+ first rounds ──────────────────────────
print("\n── Step 2: Restrict to 2021+ first-round ties ──")

dm_21 = dm[dm["year"] >= 2021].copy()
dm_21["fold"] = dm_21["investor"].map(
    lambda inv: 1.0 if fold_classification.get(inv) == "FOLD" else 0.0
)
dm_21["fold_x_nc"] = dm_21["fold"] * dm_21["nc_z"]
dm_21["fold_x_os"] = dm_21["fold"] * dm_21["os_z"]

n_fold_dyads = (dm_21["fold"] == 1).sum()
n_strata_21  = dm_21["co"].nunique()

print(f"  Dyads (2021+): {len(dm_21):,}  |  Strata: {n_strata_21:,}")
print(f"  Dyads involving fold investors: {int(n_fold_dyads):,}")
print(f"  Fold investor fraction: {dm_21['fold'].mean():.3f}")

# Correlation fold status with prior_activity
valid_fold = dm_21.dropna(subset=["log_prior_deals"])
r_corr, p_corr = pearsonr(valid_fold["fold"], valid_fold["log_prior_deals"])
print(f"  r(fold, log_prior_deals): {r_corr:.4f}  p={p_corr:.4f}  {sig(p_corr)}")

# MDE
b_vars_h1 = ["fold", "fold_x_nc", "fold_x_os",
              "sem_dist_z", "geo_same_country", "geo_same_region",
              "log_prior_deals_z", "has_prior_network", "same_community_tuned"]

FEASIBILITY_GATE = 500
proceed = n_fold_dyads >= FEASIBILITY_GATE
print(f"\n  Feasibility gate (≥{FEASIBILITY_GATE} fold dyads): "
      f"{'PASS' if proceed else 'FAIL'}  (n_fold={int(n_fold_dyads):,})")

if proceed:
    # Quick MDE from univariate run on fold_x_nc
    try:
        sub_f = dm_21[b_vars_h1 + ["tied", "co"]].dropna().copy()
        ok    = sub_f.groupby("co")["tied"].transform(lambda x: x.nunique() == 2)
        sub_f = sub_f[ok].copy()
        grps  = sub_f["co"].values
        exog  = sub_f[b_vars_h1].values.astype(float)
        endog = sub_f["tied"].values.astype(float)
        mask  = np.isfinite(exog).all(axis=1)
        for meth in ["bfgs","newton","lbfgs"]:
            try:
                res_mde = ConditionalLogit(endog[mask], exog[mask],
                                           groups=grps[mask]).fit(
                    method=meth, maxiter=600, disp=False)
                if np.isfinite(res_mde.bse).all():
                    mde_nc = (norm.ppf(0.975) + norm.ppf(0.80)) * res_mde.bse[1]
                    print(f"  MDE for fold×NC_t (80% power): {mde_nc:.4f}")
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"  MDE estimation error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# PART B — Step 3: Main model
# ═══════════════════════════════════════════════════════════════════════════
if not proceed:
    print("\n  FEASIBILITY FAILED — H1 not estimated.")
else:
    print("\n── Step 3: Main conditional logit ──")
    r_h1 = run_clogit(dm_21, b_vars_h1, "H1 main")
    if r_h1:
        res_h1, xv_h1, n_h1, ns_h1, sub_h1 = r_h1
        print(f"\n  N dyads: {n_h1:,}  |  N strata: {ns_h1:,}")
        print()
        print(f"  {'Variable':<28}  {'β':>9}  {'SE':>8}  {'p':>8}  {'sig':>4}")
        print("  " + "─" * 60)
        for i, v in enumerate(xv_h1):
            print(f"  {v:<28}  {res_h1.params[i]:+9.4f}  {res_h1.bse[i]:8.4f}  "
                  f"{res_h1.pvalues[i]:8.4f}  {sig(res_h1.pvalues[i])}")

        b_h1_nc = res_h1.params[1]; se_h1_nc = res_h1.bse[1]; p_h1_nc = res_h1.pvalues[1]
        b_h1_os = res_h1.params[2]; se_h1_os = res_h1.bse[2]; p_h1_os = res_h1.pvalues[2]
        mde_final = (norm.ppf(0.975) + norm.ppf(0.80)) * se_h1_nc

        # ── Step 4: Robustness ──────────────────────────────────────────────
        print("\n── Step 4: Robustness ──")

        # R-i: prior_activity quartiles
        print("\n  [R-i] Within prior-activity quartiles:")
        dm_21["pa_q"] = pd.qcut(dm_21["log_prior_deals_z"], 4,
                                  labels=False, duplicates="drop")
        q_labels = sorted(dm_21["pa_q"].dropna().unique())
        print(f"  {'Quartile':<8}  {'N dyads':>8}  {'β(fold×NC)':>12}  {'SE':>8}  {'p':>8}  {'sig':>4}")
        print("  " + "─" * 50)
        for q in q_labels:
            qlabel = f"Q{int(q)+1}"
            sub_q = dm_21[dm_21["pa_q"] == q]
            rq = run_clogit(sub_q, b_vars_h1, qlabel)
            if rq:
                res_q = rq[0]
                print(f"  {qlabel:<8}  {rq[2]:>8,}  {res_q.params[1]:+12.4f}  "
                      f"{res_q.bse[1]:8.4f}  {res_q.pvalues[1]:8.4f}  {sig(res_q.pvalues[1])}")
            else:
                print(f"  {qlabel:<8}  insufficient data")

        # R-ii: Full-network fold classification (existing Script 54 output)
        print("\n  [R-ii] Full-network fold classification (all years):")
        sf_full = pd.read_csv("output/structural_folds/bridge_type_folds_holes.csv")[
            ["investor", "classification"]
        ]
        fold_full = dict(zip(sf_full["investor"].str.lower(),
                              sf_full["classification"]))
        dm_21_full = dm_21.copy()
        dm_21_full["fold"] = dm_21_full["investor"].map(
            lambda inv: 1.0 if fold_full.get(inv) == "FOLD" else 0.0
        )
        dm_21_full["fold_x_nc"] = dm_21_full["fold"] * dm_21_full["nc_z"]
        dm_21_full["fold_x_os"] = dm_21_full["fold"] * dm_21_full["os_z"]
        n_fold_full = (dm_21_full["fold"] == 1).sum()
        print(f"  Full-network FOLD dyads in 2021+ sample: {int(n_fold_full):,}")
        r_full = run_clogit(dm_21_full, b_vars_h1, "full-net fold")
        if r_full:
            rf = r_full[0]
            print(f"  N dyads: {r_full[2]:,}  |  N strata: {r_full[3]:,}")
            print(f"  fold×NC_t: β={rf.params[1]:+.4f}  SE={rf.bse[1]:.4f}  "
                  f"p={rf.pvalues[1]:.4f}  {sig(rf.pvalues[1])}")
            print(f"  fold×OS_t: β={rf.params[2]:+.4f}  SE={rf.bse[2]:.4f}  "
                  f"p={rf.pvalues[2]:.4f}  {sig(rf.pvalues[2])}")

        # ── Verdict ────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("PLAIN VERDICT — H1: STRUCTURAL FOLDS × NICHE NOVELTY")
        print("=" * 70)
        print(f"\n  Pre-specified sign: fold × niche_cohesion_t  > 0")
        print(f"\n  Primary (fold × NC_t):  β={b_h1_nc:+.4f}  SE={se_h1_nc:.4f}  "
              f"p={p_h1_nc:.4f}  {sig(p_h1_nc)}")
        print(f"  Secondary (fold × OS_t):β={b_h1_os:+.4f}  SE={se_h1_os:.4f}  "
              f"p={p_h1_os:.4f}  {sig(p_h1_os)}")
        print(f"  MDE (80% power, fold×NC_t): {mde_final:.4f}")
        print(f"  N dyads: {n_h1:,}  |  N strata: {ns_h1:,}")
        print(f"  Fold-investor dyads: {int(n_fold_dyads):,}  "
              f"({100*n_fold_dyads/len(dm_21):.1f}% of 2021+ dyads)")

        if b_h1_nc > 0 and p_h1_nc < 0.05:
            verdict = "SUPPORTED"
        elif b_h1_nc > 0 and p_h1_nc < 0.10:
            verdict = "DIRECTIONALLY SUPPORTED (marginal)"
        elif b_h1_nc > 0:
            verdict = "DIRECTIONALLY CORRECT BUT NOT SIGNIFICANT"
        elif b_h1_nc < 0 and p_h1_nc < 0.05:
            verdict = "WRONG SIGN — SIGNIFICANT"
        else:
            verdict = "NULL — WRONG SIGN"
        print(f"\n  ** H1 VERDICT: {verdict} **")

print("\n" + "=" * 70)
print("SCRIPT 88 COMPLETE")
print("=" * 70)
