"""
Script 73: Fix the Two-Level Novelty Measure
==============================================
Three problems addressed:
  1. Finer clustering at k=150, 200, 300
  2. category_novelty = std(isolation) × std(smallness) per cluster
  3. Centroid instability check for clusters < 10 members
  4. Stability: ρ of category_novelty and std_distinct across k=150/200/300
  5. Effective N: ICC of failure within clusters + design effect
"""
import pickle, warnings, time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

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
embs     = normalize(embs_raw, norm="l2")   # L2-norm for cosine via dot product

meta = pd.read_csv(BASE/"output"/"novelty"/"novelty_scores.csv")
meta["name"] = meta["name"].str.strip()
meta = meta.set_index("name").reindex(names).reset_index()
meta.index = range(N)

year_arr  = meta["Year Founded"].values
hdb_arr   = meta["specter_cluster"].values
desc_arr  = meta["Description"].fillna("").values
is_noise  = (hdb_arr == -1).astype(int)

# Failure outcome for ICC (subset that appears in cpu dataset)
cdf = pd.read_csv(BASE/"output"/"cpu"/"cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
co_failed = dict(zip(cdf["co"], cdf["failed"]))
failed_arr = np.array([co_failed.get(n, np.nan) for n in names])

print(f"  Embeddings: {embs.shape}")
print(f"  Companies with failure outcome: {(~np.isnan(failed_arr)).sum():,} / {N:,}")

# ──────────────────────────────────────────────────────────────
# STEP 1: FINER CLUSTERING
# ──────────────────────────────────────────────────────────────
banner("STEP 1: FINER CLUSTERING (k = 150, 200, 300)")

RNG = 42
K_LIST = [150, 200, 300]
cluster_labels = {}

for k in K_LIST:
    print(f"  k-means k={k} ...", end=" ", flush=True)
    km = KMeans(n_clusters=k, random_state=RNG, n_init=10, max_iter=500)
    labels = km.fit_predict(embs)
    cluster_labels[f"km{k}"] = labels
    sizes = pd.Series(labels).value_counts()
    n_tiny = (sizes < 10).sum()
    print(f"done  median={sizes.median():.0f}  min={sizes.min()}  max={sizes.max()}  "
          f"clusters<10: {n_tiny}")

for k in K_LIST:
    print(f"  agglomerative k={k} ...", end=" ", flush=True)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(embs)
    cluster_labels[f"agg{k}"] = labels
    sizes = pd.Series(labels).value_counts()
    n_tiny = (sizes < 10).sum()
    print(f"done  median={sizes.median():.0f}  min={sizes.min()}  max={sizes.max()}  "
          f"clusters<10: {n_tiny}")

print(f"\n  Full size table:")
print(f"  {'Method':<10} {'k':>5} {'median':>8} {'min':>6} {'max':>6} {'<10':>6} {'<5':>6}")
for key, labels in cluster_labels.items():
    sizes = pd.Series(labels).value_counts()
    print(f"  {key:<10} {len(sizes):>5} {sizes.median():>8.0f} {sizes.min():>6} "
          f"{sizes.max():>6} {(sizes<10).sum():>6} {(sizes<5).sum():>6}")

# ──────────────────────────────────────────────────────────────
# HELPER: build centroids + isolation + smallness per clustering
# ──────────────────────────────────────────────────────────────
def build_cluster_measures(labels, embs, names, year_arr, window=3):
    labels = np.array(labels)
    unique_labs = np.unique(labels)
    K = len(unique_labs)

    # Cluster members and centroids
    members = {}
    centroids = np.zeros((K, embs.shape[1]), dtype=np.float32)
    for ci, lab in enumerate(unique_labs):
        idx = np.where(labels == lab)[0]
        members[lab] = idx
        c = embs[idx].mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-10)
        centroids[ci] = c

    lab_to_ci = {lab: ci for ci, lab in enumerate(unique_labs)}

    # Inter-centroid cosine distances
    sim_matrix = centroids @ centroids.T                # cosine sim
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, np.nan)

    # Per-cluster stats
    cluster_stats = {}
    for ci, lab in enumerate(unique_labs):
        idx = members[lab]
        sz  = len(idx)
        # isolation = mean cosine distance to other centroids
        iso = float(np.nanmean(dist_matrix[ci]))
        # per-member raw_distinct
        raw_d = 1.0 - (embs[idx] @ centroids[ci])
        cluster_stats[lab] = {
            "size": sz,
            "isolation": iso,
            "raw_distinct_mean": raw_d.mean(),
            "raw_distinct_sd": raw_d.std(),
            "centroid_idx": ci,
            "member_idx": idx,
        }

    # Standardise isolation and smallness → category_novelty per cluster
    all_iso = np.array([v["isolation"] for v in cluster_stats.values()])
    all_sz  = np.array([v["size"]      for v in cluster_stats.values()], dtype=float)
    smallness = 1.0 / np.log1p(all_sz)

    iso_z = (all_iso - all_iso.mean()) / (all_iso.std() + 1e-10)
    sml_z = (smallness - smallness.mean()) / (smallness.std() + 1e-10)
    cat_novelty_cluster = iso_z * sml_z   # per cluster

    for ci, (lab, v) in enumerate(cluster_stats.items()):
        v["smallness"] = smallness[ci]
        v["cat_novelty_cluster"] = cat_novelty_cluster[ci]

    # Propagate to startup level
    rows = []
    for ci, lab in enumerate(unique_labs):
        idx = members[lab]
        for i in idx:
            yr = year_arr[i]
            raw_d = 1.0 - float(embs[i] @ centroids[ci])
            # time-aware cluster_size_at_founding
            if not np.isnan(yr):
                same_yrs = year_arr[members[lab]]
                n_prior = int(((same_yrs >= yr - window) & (same_yrs < yr)).sum())
            else:
                n_prior = len(members[lab]) - 1
            rows.append({
                "idx": i,
                "cluster": lab,
                "isolation": cluster_stats[lab]["isolation"],
                "cluster_size": cluster_stats[lab]["size"],
                "smallness": cluster_stats[lab]["smallness"],
                "cat_novelty_cluster": cluster_stats[lab]["cat_novelty_cluster"],
                "raw_distinct": raw_d,
                "cluster_size_at_founding": n_prior,
            })

    df = pd.DataFrame(rows).set_index("idx").sort_index()

    # Within-cluster std_distinct (Haans 2019)
    for lab in unique_labs:
        idx = members[lab]
        vals = df.loc[idx, "raw_distinct"]
        mu = vals.mean(); sd = vals.std()
        if sd < 1e-10: sd = 1.0
        df.loc[idx, "std_distinct"] = (vals - mu) / sd

    return df, cluster_stats, centroids, dist_matrix, unique_labs, lab_to_ci

# ──────────────────────────────────────────────────────────────
# STEP 2: CATEGORY NOVELTY = std(isolation) x std(smallness)
# ──────────────────────────────────────────────────────────────
banner("STEP 2: CATEGORY NOVELTY = std(isolation) × std(smallness)")

results = {}
for key in [f"km{k}" for k in K_LIST] + [f"agg{k}" for k in K_LIST]:
    print(f"  Building {key} ...", end=" ", flush=True)
    df_out, cstats, centroids, dist_m, ulabs, l2ci = build_cluster_measures(
        cluster_labels[key], embs, names, year_arr)
    results[key] = (df_out, cstats, centroids, dist_m, ulabs, l2ci)
    print(f"done  [{elapsed()}]")

# Distributions for km200 as reference
PRIMARY = "km200"
df_p, cstats_p, centroids_p, dist_m_p, ulabs_p, l2ci_p = results[PRIMARY]

print(f"\n  Distributions (primary = {PRIMARY}):")
for col in ["isolation","smallness","cat_novelty_cluster","raw_distinct","std_distinct"]:
    s = df_p[col].dropna()
    print(f"    {col:<30} mean={s.mean():+.4f}  sd={s.std():.4f}  "
          f"min={s.min():+.4f}  max={s.max():+.4f}")

# Top 10 clusters by category_novelty
print(f"\n  Top 10 clusters by category_novelty (k=200):")
top_clusters = sorted(cstats_p.items(),
                      key=lambda x: x[1]["cat_novelty_cluster"], reverse=True)[:10]
for rank, (lab, v) in enumerate(top_clusters, 1):
    idx_members = v["member_idx"][:5]
    member_names = [names[i][:45] for i in idx_members]
    print(f"\n  {rank:>2}. Cluster {int(lab):>4}  "
          f"N={v['size']:>4}  iso={v['isolation']:.4f}  "
          f"cat_novelty={v['cat_novelty_cluster']:+.4f}")
    for nm in member_names:
        print(f"       {nm}")

# Bottom 10 (most crowded + similar to field)
print(f"\n  Bottom 10 clusters by category_novelty (largest, most central):")
bot_clusters = sorted(cstats_p.items(),
                      key=lambda x: x[1]["cat_novelty_cluster"])[:10]
for rank, (lab, v) in enumerate(bot_clusters, 1):
    idx_members = v["member_idx"][:3]
    member_names = [names[i][:45] for i in idx_members]
    print(f"  {rank:>2}. Cluster {int(lab):>4}  N={v['size']:>4}  "
          f"cat_novelty={v['cat_novelty_cluster']:+.4f}  members: "
          + "  |  ".join(member_names))

# Smallest and most isolated clusters — look for coherent niches
print(f"\n  5 smallest + most isolated clusters at k=200 (3 members each):")
tiny = sorted([(lab, v) for lab, v in cstats_p.items() if v["size"] < 15],
              key=lambda x: x[1]["isolation"], reverse=True)[:10]
for lab, v in tiny[:5]:
    idx_m = v["member_idx"]
    print(f"\n  Cluster {int(lab):>4}  N={v['size']:>3}  iso={v['isolation']:.4f}  "
          f"cat_novelty={v['cat_novelty_cluster']:+.4f}")
    for i in idx_m[:5]:
        desc_short = desc_arr[i][:70] if len(desc_arr[i]) > 0 else ""
        print(f"    {names[i][:55]:<55}  {desc_short}")

# ──────────────────────────────────────────────────────────────
# STEP 3: CENTROID INSTABILITY (leave-one-out) FOR SMALL CLUSTERS
# ──────────────────────────────────────────────────────────────
banner("STEP 3: CENTROID STABILITY — LEAVE-ONE-OUT FOR CLUSTERS < 10")

print(f"  Testing k=200 clusters with N < 10 members:")
tiny_labs = [lab for lab, v in cstats_p.items() if v["size"] < 10]
print(f"  Found {len(tiny_labs)} such clusters")

unstable_clusters = []
for lab in tiny_labs:
    idx = cstats_p[lab]["member_idx"]
    n_c = len(idx)
    full_centroid = embs[idx].mean(axis=0)
    full_centroid /= max(np.linalg.norm(full_centroid), 1e-10)
    full_dists = 1.0 - (embs[idx] @ full_centroid)
    full_std = full_dists.std() if len(full_dists) > 1 else 0.0

    # Leave one out
    loo_stds = []
    for drop_i in range(n_c):
        loo_idx = [idx[j] for j in range(n_c) if j != drop_i]
        if len(loo_idx) < 2:
            loo_stds.append(np.nan)
            continue
        loo_centroid = embs[loo_idx].mean(axis=0)
        loo_centroid /= max(np.linalg.norm(loo_centroid), 1e-10)
        loo_dists = 1.0 - (embs[loo_idx] @ loo_centroid)
        loo_stds.append(loo_dists.std())

    loo_stds_valid = [x for x in loo_stds if not np.isnan(x)]
    if not loo_stds_valid: continue
    max_change = max(abs(s - full_std) for s in loo_stds_valid)
    rel_change = max_change / (full_std + 1e-10)

    if rel_change > 0.20:  # >20% change in SD when one member dropped
        unstable_clusters.append((lab, n_c, full_std, max_change, rel_change))

unstable_clusters.sort(key=lambda x: x[4], reverse=True)
print(f"  Clusters with >20% LOO instability in std_distinct: {len(unstable_clusters)}")
if unstable_clusters:
    print(f"  {'cluster':>8} {'N':>4} {'full_sd':>10} {'max_change':>12} {'rel_change':>12}")
    for lab, n_c, fsd, mc, rc in unstable_clusters[:15]:
        idx = cstats_p[lab]["member_idx"]
        nm_list = [names[i][:35] for i in idx[:3]]
        print(f"  {int(lab):>8} {n_c:>4} {fsd:>10.4f} {mc:>12.4f} {rc:>11.1%}"
              f"  [{' | '.join(nm_list)}]")
    print(f"\n  FLAG: these {len(unstable_clusters)} clusters should be excluded from or "
          f"flagged in within-category distinctiveness regressions.")
else:
    print(f"  No clusters show >20% LOO instability.")

# Overall LOO pass rate
print(f"\n  Of {len(tiny_labs)} clusters < 10 members: "
      f"{len(tiny_labs)-len(unstable_clusters)} stable, {len(unstable_clusters)} unstable")

# ──────────────────────────────────────────────────────────────
# STEP 4: STABILITY ACROSS k
# ──────────────────────────────────────────────────────────────
banner("STEP 4: STABILITY OF MEASURES ACROSS k = 150, 200, 300")

key_pairs = [("km150","km200"),("km200","km300"),("km150","km300"),
             ("km200","agg200")]
for k1, k2 in key_pairs:
    df1, df2 = results[k1][0], results[k2][0]
    for col in ["cat_novelty_cluster","std_distinct","isolation"]:
        s = pd.DataFrame({"a": df1[col].values, "b": df2[col].values}).dropna()
        r, p = spearmanr(s["a"], s["b"])
        print(f"  {k1} vs {k2}  {col:<30} ρ={r:+.4f}  p={p:.4f} {sig(p)}  N={len(s):,}")

# ──────────────────────────────────────────────────────────────
# STEP 5: EFFECTIVE N — ICC AND DESIGN EFFECT
# ──────────────────────────────────────────────────────────────
banner("STEP 5: EFFECTIVE N — ICC OF FAILURE WITHIN CLUSTERS + DESIGN EFFECT")

valid = ~np.isnan(failed_arr)
y = failed_arr[valid]
print(f"  N startups with failure outcome: {valid.sum():,}")

print(f"\n  {'Method':<10} {'k':>5} {'ICC':>8} {'avg_clust_sz':>14} {'design_eff':>12} {'eff_N':>10}")
print(f"  {'-'*62}")

def icc_oneway(y, groups):
    """One-way random-effects ICC = between-cluster variance / total variance."""
    unique_g = np.unique(groups)
    grand_mean = y.mean()
    n_total = len(y)
    n_groups = len(unique_g)
    # Mean squares
    group_sizes = np.array([np.sum(groups == g) for g in unique_g])
    group_means = np.array([y[groups == g].mean() for g in unique_g])
    SS_between = np.sum(group_sizes * (group_means - grand_mean)**2)
    SS_within  = sum(np.sum((y[groups == g] - group_means[i])**2)
                     for i, g in enumerate(unique_g))
    df_b = n_groups - 1
    df_w = n_total - n_groups
    MS_b = SS_between / df_b if df_b > 0 else 0
    MS_w = SS_within  / df_w if df_w > 0 else 1
    n0   = (n_total - np.sum(group_sizes**2)/n_total) / (n_groups - 1)
    icc  = (MS_b - MS_w) / (MS_b + (n0 - 1) * MS_w + 1e-20)
    return float(np.clip(icc, 0, 1)), float(n0), MS_b, MS_w

for key in [f"km{k}" for k in K_LIST] + [f"agg{k}" for k in K_LIST]:
    labels_all = cluster_labels[key]
    labels_v   = labels_all[valid]
    icc, n0, msb, msw = icc_oneway(y, labels_v)
    # Design effect = 1 + (n_avg - 1) * ICC
    sizes_v = pd.Series(labels_v).value_counts()
    n_avg   = sizes_v.mean()
    deff    = 1.0 + (n_avg - 1.0) * icc
    eff_N   = valid.sum() / deff
    print(f"  {key:<10} {sizes_v.count():>5} {icc:>8.4f} {n_avg:>14.1f} {deff:>12.2f} {eff_N:>10.0f}")

# Interpretation
print(f"\n  Interpretation:")
print(f"  ICC ≈ 0.00–0.05: minimal cluster similarity, little design effect")
print(f"  ICC ≈ 0.05–0.15: moderate, SEs understated by ~10–30%")
print(f"  ICC ≈ 0.15+: material; standard errors must be clustered on cluster_id")

# ──────────────────────────────────────────────────────────────
# SAVE FIXED OUTPUT
# ──────────────────────────────────────────────────────────────
banner("SAVING FIXED OUTPUT")

df_save = results[PRIMARY][0].copy()
df_save["name"]             = names
df_save["is_hdbscan_noise"] = is_noise
df_save["desc_len"]         = [len(d) for d in desc_arr]
df_save["failed"]           = failed_arr

# Add km150 and km300 for stability appendix
df_save["cat_novelty_km150"]  = results["km150"][0]["cat_novelty_cluster"].values
df_save["std_distinct_km150"] = results["km150"][0]["std_distinct"].values
df_save["cat_novelty_km300"]  = results["km300"][0]["cat_novelty_cluster"].values
df_save["std_distinct_km300"] = results["km300"][0]["std_distinct"].values

# Add instability flag
unstable_set = {lab for lab, *_ in unstable_clusters}
df_save["centroid_unstable"] = df_save["cluster"].isin(unstable_set).astype(int)

out_path = OUT / "two_level_novelty_v2.csv"
df_save.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(df_save):,} rows)")
print(f"  Columns: {df_save.columns.tolist()}")
print(f"\n  Summary of final measures (k=200):")
for col in ["cat_novelty_cluster","std_distinct","isolation","smallness"]:
    s = df_save[col].dropna()
    print(f"    {col:<30} SD={s.std():.4f}  range=[{s.min():+.4f}, {s.max():+.4f}]")
print(f"\n  [{elapsed()}]")
