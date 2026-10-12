"""
Script 75: Clean and Finalise Density-Based Novelty Measures
=============================================================
1. Deduplicate: cosine sim > 0.99 pairs + boilerplate (desc < 100 chars)
2. Recompute niche_score and regional_sparsity on cleaned set
3. Quadrant analysis (mainstream / isolated niche / crowded outlier / lonely outlier)
4. Correlation matrix: regional_sparsity, niche_score, std_distinct,
                       specter_novelty, is_hdbscan_noise
"""
import pickle, warnings, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr

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
    ns_pkl = pickle.load(fh)
embs_raw = np.array(ns_pkl["embs_raw"], dtype=np.float32)
names    = [n.strip() for n in ns_pkl["names"]]
N        = len(names)
embs     = normalize(embs_raw, norm="l2")

meta = pd.read_csv(BASE/"output"/"novelty"/"novelty_scores.csv")
meta["name"] = meta["name"].str.strip()
meta = meta.set_index("name").reindex(names).reset_index()
meta.index = range(N)
desc_arr     = meta["Description"].fillna("").values
desc_len     = np.array([len(d) for d in desc_arr])
hdb_arr      = meta["specter_cluster"].values
is_noise_all = (hdb_arr == -1).astype(int)
specter_nov  = meta["specter_novelty"].values.astype(float)

# Script 72 std_distinct (k=60 primary)
v72 = pd.read_csv(OUT / "two_level_novelty.csv")
v72 = v72.set_index("name").reindex(names).reset_index()
v72.index = range(N)
std_dist_all = v72["std_distinct"].values.astype(float)

print(f"  Loaded {N} startups")

# ──────────────────────────────────────────────────────────────
# STEP 1: DEDUPLICATION
# ──────────────────────────────────────────────────────────────
banner("STEP 1: DEDUPLICATION — cosine sim > 0.99 + boilerplate filter")

# 1a. Near-duplicate detection via 30-NN search
print("  Running 30-NN search for near-duplicates ...", end=" ", flush=True)
nn30 = NearestNeighbors(n_neighbors=30, metric="cosine",
                         algorithm="brute", n_jobs=-1)
nn30.fit(embs)
d30, idx30 = nn30.kneighbors(embs)
print(f"done  [{elapsed()}]")

THRESH = 0.01   # cosine distance < 0.01 ↔ sim > 0.99
pairs = []
for i in range(N):
    for col in range(1, 30):
        j = idx30[i, col]
        if j <= i: continue
        if d30[i, col] < THRESH:
            pairs.append((i, j, float(1 - d30[i, col])))

print(f"\n  Pairs with cosine sim > 0.99: {len(pairs)}")

# Connected components (union-find)
parent = list(range(N))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(x, y):
    px, py = find(x), find(y)
    if px != py: parent[px] = py

for i, j, _ in pairs:
    union(i, j)

dup_groups = defaultdict(list)
for i in range(N):
    dup_groups[find(i)].append(i)
dup_groups = {k: v for k, v in dup_groups.items() if len(v) > 1}
print(f"  Duplicate groups (connected components): {len(dup_groups)}")

# Show 10 largest groups
sorted_groups = sorted(dup_groups.values(), key=len, reverse=True)
print(f"\n  All duplicate groups (sim>0.99):")
for gi, grp in enumerate(sorted_groups):
    shared_desc = desc_arr[grp[0]][:100].replace("\n"," ")
    print(f"\n  Group {gi+1}  (N={len(grp)}):")
    for idx_i in grp:
        print(f"    [{desc_len[idx_i]:>3} chars] {names[idx_i][:60]}")
    print(f"    Shared description: \"{shared_desc}\"")

# 1b. Boilerplate filter — descriptions < 100 chars
BOILERPLATE_PATTERNS = [
    "currently operating in stealth",
    "operating in stealth mode",
    "stealth mode",
]
def is_boilerplate(d, length):
    if length < 100: return True
    dl = d.lower()
    return any(pat in dl for pat in BOILERPLATE_PATTERNS)

boilerplate_mask = np.array([is_boilerplate(d, l)
                              for d, l in zip(desc_arr, desc_len)])
print(f"\n  Boilerplate/short descriptions (< 100 chars or stealth): "
      f"{boilerplate_mask.sum()}")
for i in np.where(boilerplate_mask)[0]:
    print(f"    [{desc_len[i]:>3}] {names[i][:55]:<55}  |  {desc_arr[i][:50]}")

# 1c. Build exclusion set: boilerplate + duplicates (keep first/longest in each group)
exclude_idx = set(np.where(boilerplate_mask)[0])

# For dup groups: keep the one with longest description; exclude rest
dup_drops = 0
for grp in sorted_groups:
    best = max(grp, key=lambda i: desc_len[i])
    for idx_i in grp:
        if idx_i != best:
            exclude_idx.add(idx_i)
            dup_drops += 1

print(f"\n  Exclusions:")
print(f"    Boilerplate/short:     {boilerplate_mask.sum()}")
print(f"    Dup redundant copies:  {dup_drops}  (keeping 1 per group)")
print(f"    Total unique to drop:  {len(exclude_idx)}")

keep_mask = np.ones(N, dtype=bool)
keep_mask[list(exclude_idx)] = False
N_clean = keep_mask.sum()
print(f"  Cleaned sample: {N_clean} startups ({N_clean/N:.1%} of original)")

keep_idx  = np.where(keep_mask)[0]
embs_c    = embs[keep_idx]
names_c   = [names[i] for i in keep_idx]
desc_arr_c = [desc_arr[i] for i in keep_idx]
desc_len_c = desc_len[keep_idx]
hdb_c     = is_noise_all[keep_idx]
snov_c    = specter_nov[keep_idx]
sdist_c   = std_dist_all[keep_idx]

# ──────────────────────────────────────────────────────────────
# STEP 2: RECOMPUTE ON CLEANED SET
# ──────────────────────────────────────────────────────────────
banner("STEP 2: RECOMPUTE ON CLEANED SET")

PAIRS_LIST = [(3, 30), (5, 50), (10, 100)]
PRIMARY    = (5, 50)

max_k_need = max(r for _, r in PAIRS_LIST) + 1
print(f"  Building {max_k_need}-NN on {N_clean} cleaned startups ...",
      end=" ", flush=True)
nn_c = NearestNeighbors(n_neighbors=max_k_need, metric="cosine",
                         algorithm="brute", n_jobs=-1)
nn_c.fit(embs_c)
dists_c, idx_c = nn_c.kneighbors(embs_c)
sim_c = 1.0 - dists_c   # col 0 = self
print(f"done  [{elapsed()}]")

density_results = {}
for (loc_k, reg_k) in PAIRS_LIST:
    local_sim    = sim_c[:, 1:loc_k+1].mean(axis=1)
    regional_sim = sim_c[:, 1:reg_k+1].mean(axis=1)
    density_results[(loc_k, reg_k)] = (local_sim, regional_sim)

loc5, reg50 = density_results[PRIMARY]
regional_sparsity = 1.0 - reg50   # high = isolated from broad neighbourhood

def std_arr(a):
    return (a - a.mean()) / (a.std() + 1e-10)

niche_scores = {}
for pair, (local_sim, regional_sim) in density_results.items():
    niche_scores[pair] = std_arr(local_sim) - std_arr(regional_sim)

ns_p = niche_scores[PRIMARY]

print(f"\n  Distributions on cleaned set (N={N_clean}):")
print(f"  {'Measure':<35} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8} {'p99':>8}")
for label, arr in [
    ("local_density (5-NN)",    loc5),
    ("regional_density (50-NN)", reg50),
    ("regional_sparsity",        regional_sparsity),
    ("niche_score",              ns_p),
]:
    print(f"  {label:<35} {arr.mean():>8.4f} {arr.std():>8.4f} "
          f"{arr.min():>8.4f} {arr.max():>8.4f} {np.percentile(arr,99):>8.4f}")

print(f"\n  Leverage check (|niche_score| > 3): "
      f"{(np.abs(ns_p) > 3).sum()} points")
if (np.abs(ns_p) > 3).sum() <= 10:
    for i in np.where(np.abs(ns_p) > 3)[0]:
        print(f"    [{ns_p[i]:+.3f}] {names_c[i][:60]}  |  {desc_arr_c[i][:60]}")

# Stability
print(f"\n  Stability of niche_score across neighbour pairs (cleaned):")
from itertools import combinations
pk = list(niche_scores.keys())
pn = [f"({l},{r})" for l,r in pk]
for (p1, p2) in combinations(range(len(pk)), 2):
    r, p = spearmanr(niche_scores[pk[p1]], niche_scores[pk[p2]])
    print(f"    {pn[p1]:<10} vs {pn[p2]:<10} ρ={r:+.4f}  {sig(p)}")

# ──────────────────────────────────────────────────────────────
# STEP 3: QUADRANT ANALYSIS
# ──────────────────────────────────────────────────────────────
banner("STEP 3: QUADRANT ANALYSIS (median splits)")

loc_med = np.median(loc5)
reg_med = np.median(reg50)

# Quadrant definitions
#   loc > med  AND  reg > med → mainstream (dense everywhere)
#   loc > med  AND  reg < med → isolated niche (tight group, sparse field)
#   loc < med  AND  reg > med → crowded outlier (loose local, dense field)
#   loc < med  AND  reg < med → lonely outlier (isolated everywhere)

labels_q = np.empty(N_clean, dtype=object)
labels_q[(loc5 >= loc_med) & (reg50 >= reg_med)] = "mainstream"
labels_q[(loc5 >= loc_med) & (reg50 <  reg_med)] = "isolated niche"
labels_q[(loc5 <  loc_med) & (reg50 >= reg_med)] = "crowded outlier"
labels_q[(loc5 <  loc_med) & (reg50 <  reg_med)] = "lonely outlier"

print(f"\n  Quadrant thresholds: local median={loc_med:.4f}, regional median={reg_med:.4f}")
print(f"\n  {'Quadrant':<20} {'N':>6} {'%':>6} {'local_mean':>12} {'regional_mean':>14}")
print(f"  {'-'*65}")
for quad in ["isolated niche", "mainstream", "crowded outlier", "lonely outlier"]:
    mask = labels_q == quad
    n = mask.sum()
    print(f"  {quad:<20} {n:>6} {n/N_clean:>5.1%} "
          f"{loc5[mask].mean():>12.4f} {reg50[mask].mean():>14.4f}")

# 8 examples per quadrant
print()
quad_order = ["isolated niche", "mainstream", "crowded outlier", "lonely outlier"]
quad_desc = {
    "isolated niche":  "tight local cluster, sparse broad field — THE TARGET CONSTRUCT",
    "mainstream":      "dense everywhere — established, crowded categories",
    "crowded outlier": "loose local, dense field — misfits in a crowded space",
    "lonely outlier":  "sparse everywhere — singletons, outliers",
}
for quad in quad_order:
    mask = np.where(labels_q == quad)[0]
    # sort by niche_score descending for isolated niche, desc_len for others
    if quad == "isolated niche":
        sorted_mask = mask[np.argsort(ns_p[mask])[::-1]]
    elif quad == "lonely outlier":
        sorted_mask = mask[np.argsort(ns_p[mask])]
    else:
        sorted_mask = mask[np.argsort(loc5[mask])[::-1]]
    examples = sorted_mask[:8]
    print(f"  --- {quad.upper()} ({quad_desc[quad]}) ---")
    for i in examples:
        desc_s = desc_arr_c[i][:65]
        print(f"    [{ns_p[i]:+.3f}] {names_c[i][:52]:<52}  {desc_s}")
    print()

# ──────────────────────────────────────────────────────────────
# STEP 4: CORRELATION MATRIX
# ──────────────────────────────────────────────────────────────
banner("STEP 4: CORRELATION MATRIX (Spearman ρ)")

measures = {
    "regional_sparsity": regional_sparsity,
    "niche_score":       ns_p,
    "std_distinct":      sdist_c,
    "specter_novelty":   snov_c,
    "is_hdbscan_noise":  hdb_c.astype(float),
    "desc_len":          desc_len_c.astype(float),
}
mkeys = list(measures.keys())

# Print header
col_w = 18
print(f"\n  {'':>{col_w}}", end="")
for k in mkeys:
    print(f"  {k[:col_w]:>{col_w}}", end="")
print()
print(f"  {'-' * (col_w + (col_w+2)*len(mkeys))}")

corr_grid = {}
for k1 in mkeys:
    print(f"  {k1:>{col_w}}", end="")
    for k2 in mkeys:
        a, b = measures[k1], measures[k2]
        valid = ~(np.isnan(a) | np.isnan(b))
        if k1 == k2:
            print(f"  {'1.000':>{col_w}}", end="")
        else:
            r, p = spearmanr(a[valid], b[valid])
            star = sig(p).replace("n.s.","").strip()
            cell = f"{r:+.3f}{star}"
            print(f"  {cell:>{col_w}}", end="")
    print()

# Interpretation
print(f"\n  Key findings:")
r_ns_sp, p_ns_sp = spearmanr(ns_p, regional_sparsity)
r_ns_sd, p_ns_sd = spearmanr(ns_p[~np.isnan(sdist_c)],
                               sdist_c[~np.isnan(sdist_c)])
r_sp_sn, _ = spearmanr(regional_sparsity[~np.isnan(snov_c)],
                         snov_c[~np.isnan(snov_c)])
r_ns_sn, _ = spearmanr(ns_p[~np.isnan(snov_c)], snov_c[~np.isnan(snov_c)])
print(f"    niche_score × regional_sparsity: ρ={r_ns_sp:+.3f} {sig(p_ns_sp)}")
print(f"    niche_score × std_distinct:      ρ={r_ns_sd:+.3f} {sig(p_ns_sd)}")
print(f"    regional_sparsity × specter_nov: ρ={r_sp_sn:+.3f}")
print(f"    niche_score × specter_novelty:   ρ={r_ns_sn:+.3f}")

if abs(r_ns_sd) < 0.15:
    print(f"\n  LEVEL ORTHOGONALITY: niche_score (category) and std_distinct (within-category)")
    print(f"  are near-uncorrelated (ρ={r_ns_sd:+.3f}). The two levels are genuinely distinct.")
else:
    print(f"\n  WARNING: niche_score and std_distinct correlate at ρ={r_ns_sd:+.3f}.")
    print(f"  The two levels are NOT fully orthogonal — report this constraint.")

# ──────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────
banner("SAVING OUTPUT")

df_out = pd.DataFrame({
    "name":                names_c,
    "local_density_5":     loc5,
    "regional_density_50": reg50,
    "regional_sparsity":   regional_sparsity,
    "niche_score":         ns_p,
    "niche_score_3_30":    niche_scores[(3,30)],
    "niche_score_10_100":  niche_scores[(10,100)],
    "quadrant":            labels_q,
    "std_distinct":        sdist_c,
    "specter_novelty":     snov_c,
    "is_hdbscan_noise":    hdb_c,
    "desc_len":            desc_len_c,
})
out_path = OUT / "density_novelty_clean.csv"
df_out.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(df_out):,} rows)")
print(f"  Dropped: {N - N_clean} startups (boilerplate + duplicates)")
print(f"\n  [{elapsed()}]")
