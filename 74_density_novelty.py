"""
Script 74: Multi-Scale Density Novelty (k-free)
================================================
niche_score = std(local_density) - std(regional_density)
High niche_score = tight local cluster, isolated from broader field.
This is the optimal-distinctiveness construct without dependence on k-means k.

Neighbour pairs tested: (3,30), (5,50), (10,100)
Primary: (5, 50)
"""
import pickle, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
OUT  = BASE / "output" / "two_level_novelty"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

def elapsed(): return f"{time.time()-t0:.1f}s"
def banner(s): print(f"\n{'='*72}\n{s}\n{'='*72}")
def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────
banner("LOADING DATA")
with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns = pickle.load(fh)
embs_raw = np.array(ns["embs_raw"], dtype=np.float32)
names    = [n.strip() for n in ns["names"]]
N        = len(names)
embs     = normalize(embs_raw, norm="l2")

meta = pd.read_csv(BASE/"output"/"novelty"/"novelty_scores.csv")
meta["name"] = meta["name"].str.strip()
meta = meta.set_index("name").reindex(names).reset_index()
meta.index = range(N)
desc_arr = meta["Description"].fillna("").values
desc_len = np.array([len(d) for d in desc_arr], dtype=float)
hdb_arr  = meta["specter_cluster"].values
is_noise = (hdb_arr == -1).astype(int)

# Script 73 measures for validation
v73_path = OUT / "two_level_novelty_v2.csv"
v73 = pd.read_csv(v73_path) if v73_path.exists() else None
if v73 is not None:
    v73 = v73.set_index("name").reindex(names).reset_index()
    v73.index = range(N)
    print(f"  Script 73 output loaded: {v73_path}")
else:
    print(f"  WARNING: Script 73 output not found — skipping cross-validation")

print(f"  Embeddings: {embs.shape}")

# ──────────────────────────────────────────────────────────────
# STEP 1 & 3: COMPUTE DENSITY PAIRS
# ──────────────────────────────────────────────────────────────
banner("STEP 1: LOCAL + REGIONAL DENSITY (all neighbour pairs)")

PAIRS = [(3, 30), (5, 50), (10, 100)]
PRIMARY = (5, 50)

# Build max_k NN once
max_k = max(r for _, r in PAIRS) + 1   # +1 because col 0 = self
print(f"  Building {max_k}-NN index ...", end=" ", flush=True)
nn_model = NearestNeighbors(n_neighbors=max_k, metric="cosine",
                             algorithm="brute", n_jobs=-1)
nn_model.fit(embs)
dist_mat, idx_mat = nn_model.kneighbors(embs)
# dist_mat[:,0] ≈ 0 (self); similarity = 1 - distance
sim_mat = 1.0 - dist_mat    # (N, max_k)  col 0 = self sim ≈ 1.0
print(f"done  [{elapsed()}]")

density_results = {}
for (loc_k, reg_k) in PAIRS:
    # skip self (col 0); cols 1..loc_k are the loc_k nearest
    local_sim    = sim_mat[:, 1:loc_k+1].mean(axis=1)    # (N,)
    regional_sim = sim_mat[:, 1:reg_k+1].mean(axis=1)    # (N,)
    density_results[(loc_k, reg_k)] = (local_sim, regional_sim)

# Report distributions for primary pair
loc5, reg50 = density_results[PRIMARY]
print(f"\n  PRIMARY pair: local={PRIMARY[0]}-NN, regional={PRIMARY[1]}-NN")
print(f"\n  {'Measure':<30} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8}")
print(f"  {'-'*62}")
for label, arr in [("local_density (5-NN sim)", loc5),
                   ("regional_density (50-NN sim)", reg50)]:
    print(f"  {label:<30} {arr.mean():>8.4f} {arr.std():>8.4f} "
          f"{arr.min():>8.4f} {arr.max():>8.4f}")

# ──────────────────────────────────────────────────────────────
# STEP 2: NICHE SCORE
# ──────────────────────────────────────────────────────────────
banner("STEP 2: niche_score = std(local_density) - std(regional_density)")

def compute_niche(local_sim, regional_sim):
    loc_z = (local_sim  - local_sim.mean())  / (local_sim.std()  + 1e-10)
    reg_z = (regional_sim - regional_sim.mean()) / (regional_sim.std() + 1e-10)
    return loc_z - reg_z

niche_scores = {}
for pair, (local_sim, regional_sim) in density_results.items():
    niche_scores[pair] = compute_niche(local_sim, regional_sim)

ns_primary = niche_scores[PRIMARY]

print(f"\n  niche_score distribution (primary 5/50):")
print(f"  mean={ns_primary.mean():+.4f}  sd={ns_primary.std():.4f}  "
      f"min={ns_primary.min():+.4f}  max={ns_primary.max():+.4f}")
for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    {q:>3}th pct: {np.percentile(ns_primary,q):+.4f}")

# Leverage / extreme points
print(f"\n  Leverage check: startups with |niche_score| > 4:")
extreme_mask = np.abs(ns_primary) > 4
n_extreme = extreme_mask.sum()
print(f"  N extreme (|z|>4): {n_extreme}  ({n_extreme/N:.1%})")
if n_extreme > 0 and n_extreme <= 30:
    for i in np.where(extreme_mask)[0]:
        print(f"    [{ns_primary[i]:+.3f}] {names[i][:55]}")
elif n_extreme > 30:
    top_i = np.argsort(ns_primary)[-5:]
    print(f"  Top 5 positive extremes:")
    for i in top_i[::-1]:
        print(f"    [{ns_primary[i]:+.3f}] {names[i][:55]}")

print(f"\n  SD of niche_score: {ns_primary.std():.4f}")
if ns_primary.std() < 0.5:
    print("  WARNING: low SD — measure may have limited discriminating power")
else:
    print("  OK: SD sufficient for regression use")

# ──────────────────────────────────────────────────────────────
# STEP 3: ROBUSTNESS TO NEIGHBOUR COUNTS
# ──────────────────────────────────────────────────────────────
banner("STEP 3: STABILITY ACROSS NEIGHBOUR PAIRS")

pair_keys = list(niche_scores.keys())
pair_names = [f"({l},{r})" for l, r in pair_keys]
print(f"  Spearman ρ of niche_score across pairs:")
print(f"  {'Pair A':<12} {'Pair B':<12} {'ρ':>8} {'p':>10} {'sig':>4}")
print(f"  {'-'*50}")
from itertools import combinations
for (p1, p2) in combinations(range(len(pair_keys)), 2):
    a, b = niche_scores[pair_keys[p1]], niche_scores[pair_keys[p2]]
    r, p = spearmanr(a, b)
    print(f"  {pair_names[p1]:<12} {pair_names[p2]:<12} {r:>+8.4f} {p:>10.4f} {sig(p):>4}")

# Also check: are the components (local / regional) each stable?
print(f"\n  Stability of components separately:")
for label, component_idx in [("local_density", 0), ("regional_density", 1)]:
    arrs = [density_results[pk][component_idx] for pk in pair_keys]
    for (p1, p2) in combinations(range(len(pair_keys)), 2):
        r, p = spearmanr(arrs[p1], arrs[p2])
        print(f"  {label:<20} {pair_names[p1]:<10} vs {pair_names[p2]:<10} ρ={r:+.4f}  {sig(p)}")

# ──────────────────────────────────────────────────────────────
# STEP 4: VALIDATION AGAINST SCRIPT 73
# ──────────────────────────────────────────────────────────────
banner("STEP 4: VALIDATION AGAINST SCRIPT 73 MEASURES")

print(f"  Correlations with niche_score (primary 5/50):")
print(f"  {'Measure':<40} {'ρ':>8} {'p':>10} {'sig':>4}")
print(f"  {'-'*65}")

# Description length
r, p = spearmanr(ns_primary, desc_len)
print(f"  {'description_length':<40} {r:>+8.4f} {p:>10.4f} {sig(p)}")

# HDBSCAN noise flag
r, p = spearmanr(ns_primary, is_noise)
print(f"  {'is_hdbscan_noise':<40} {r:>+8.4f} {p:>10.4f} {sig(p)}")

# Script 73 measures
if v73 is not None:
    for col in ["cat_novelty_cluster","isolation","std_distinct",
                "cluster_size_at_founding","cluster_size"]:
        if col not in v73.columns: continue
        arr = v73[col].values.astype(float)
        valid = ~np.isnan(arr)
        r, p = spearmanr(ns_primary[valid], arr[valid])
        print(f"  {col:<40} {r:>+8.4f} {p:>10.4f} {sig(p)}")

# Specter novelty from Script 11
specter_nov = meta["specter_novelty"].values if "specter_novelty" in meta.columns else None
if specter_nov is not None:
    valid = ~np.isnan(specter_nov.astype(float))
    r, p = spearmanr(ns_primary[valid], specter_nov[valid])
    print(f"  {'specter_novelty (Script 11)':<40} {r:>+8.4f} {p:>10.4f} {sig(p)}")

print(f"\n  Interpretation guide:")
print(f"    niche_score vs cat_novelty:   want ρ > 0.3 (same construct, different method)")
print(f"    niche_score vs std_distinct:  want ρ close to 0 (orthogonal levels)")
print(f"    niche_score vs desc_length:   want ρ near 0 (not driven by description quality)")

# ──────────────────────────────────────────────────────────────
# STEP 5: FACE VALIDITY
# ──────────────────────────────────────────────────────────────
banner("STEP 5: FACE VALIDITY — TOP 15 ON niche_score")

top15_idx = np.argsort(ns_primary)[-15:][::-1]
print(f"\n  Top 15 startups: HIGH niche_score (tight local cluster, isolated from field)")
print(f"  {'rank':>4} {'niche':>7} {'loc_sim':>8} {'reg_sim':>8}  {'name':<45}  description")
print(f"  {'-'*115}")
for rank, i in enumerate(top15_idx, 1):
    nm   = names[i][:44]
    desc = desc_arr[i][:70]
    loc  = loc5[i]
    reg  = reg50[i]
    ns_  = ns_primary[i]
    print(f"  {rank:>4} {ns_:>+7.3f} {loc:>8.4f} {reg:>8.4f}  {nm:<45}  {desc}")

bot15_idx = np.argsort(ns_primary)[:15]
print(f"\n  Bottom 15 startups: LOW niche_score (generic, embedded in dense region)")
print(f"  {'rank':>4} {'niche':>7} {'loc_sim':>8} {'reg_sim':>8}  {'name':<45}  description")
print(f"  {'-'*115}")
for rank, i in enumerate(bot15_idx, 1):
    nm   = names[i][:44]
    desc = desc_arr[i][:70]
    loc  = loc5[i]
    reg  = reg50[i]
    ns_  = ns_primary[i]
    print(f"  {rank:>4} {ns_:>+7.3f} {loc:>8.4f} {reg:>8.4f}  {nm:<45}  {desc}")

# Also: neighbourhood peek — show the 5 nearest neighbours for top 5
print(f"\n  5-NN neighbourhood for top 5 niche_score startups:")
for rank, i in enumerate(top15_idx[:5], 1):
    print(f"\n  #{rank} [{ns_primary[i]:+.3f}] {names[i]}")
    for j_rank, j in enumerate(idx_mat[i, 1:6], 1):
        print(f"    NN{j_rank}: [{1-dist_mat[i,j_rank]:.4f} sim]  {names[j][:65]}")

# ──────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────
banner("SAVING OUTPUT")

df_out = pd.DataFrame({
    "name":             names,
    "local_density_5":  density_results[(5,50)][0],
    "regional_density_50": density_results[(5,50)][1],
    "niche_score":      niche_scores[(5,50)],
    "local_density_3":  density_results[(3,30)][0],
    "regional_density_30": density_results[(3,30)][1],
    "niche_score_3_30": niche_scores[(3,30)],
    "local_density_10": density_results[(10,100)][0],
    "regional_density_100": density_results[(10,100)][1],
    "niche_score_10_100": niche_scores[(10,100)],
    "is_hdbscan_noise": is_noise,
    "desc_len":         desc_len,
})
out_path = OUT / "density_novelty.csv"
df_out.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(df_out):,} rows)")

print(f"\n  Final summary:")
print(f"  niche_score (5/50):   SD={niche_scores[(5,50)].std():.4f}  "
      f"range=[{niche_scores[(5,50)].min():+.3f}, {niche_scores[(5,50)].max():+.3f}]")
print(f"  niche_score (3/30):   SD={niche_scores[(3,30)].std():.4f}")
print(f"  niche_score (10/100): SD={niche_scores[(10,100)].std():.4f}")
print(f"\n  [{elapsed()}]")
