"""
Script 72: Two-Level Novelty Measure
======================================
Optimal distinctiveness framework (Haans 2019):
  Level 1 — Category novelty:       how isolated is the startup's cluster?
  Level 2 — Within-cat distinctiveness: how far is the startup from its cluster centroid?

PRIMARY: k-means at k=12,30,60,100 and agglomerative at k=60
ROBUSTNESS: existing HDBSCAN solution (26% noise → unsuitable as primary)
is_hdbscan_noise retained as its own binary signal.
"""
import pickle, warnings, re, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
OUT  = BASE / "output" / "two_level_novelty"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

def elapsed(): return f"{time.time()-t0:.1f}s"
def banner(s): print(f"\n{'='*72}\n{s}\n{'='*72}")
def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

# ──────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────
banner("LOADING DATA")
with open(BASE/"output"/"novelty"/"novelty_specter.pkl","rb") as fh:
    ns = pickle.load(fh)
embs_raw = np.array(ns["embs_raw"], dtype=np.float32)   # (9749, 768)
names    = ns["names"]
N        = len(names)

# L2-normalise for cosine distances via euclidean
embs = normalize(embs_raw, norm="l2")

# Meta
meta = pd.read_csv(BASE/"output"/"novelty"/"novelty_scores.csv")
meta["name"] = meta["name"].str.strip()
name_idx = {n.strip(): i for i, n in enumerate(names)}

# Align meta to embedding order
meta = meta.set_index("name").reindex([n.strip() for n in names]).reset_index()
meta.index = range(N)
assert len(meta) == N

year_arr = meta["Year Founded"].values          # float, NaN allowed
hdb_arr  = meta["specter_cluster"].values       # HDBSCAN labels, -1=noise
desc_arr = meta["Description"].fillna("").values
desc_len = np.array([len(d) for d in desc_arr], dtype=float)
is_noise = (hdb_arr == -1).astype(int)

print(f"  Embeddings: {embs.shape}  names: {N}")
print(f"  HDBSCAN noise points: {is_noise.sum()} ({is_noise.mean():.1%})")
print(f"  Year range: {int(np.nanmin(year_arr))}–{int(np.nanmax(year_arr))}")

# ──────────────────────────────────────────────────────────────
# STEP 1: CLUSTERING AT MULTIPLE GRANULARITIES
# ──────────────────────────────────────────────────────────────
banner("STEP 1: CLUSTERING AT MULTIPLE GRANULARITIES")

RNG = 42
K_LIST = [12, 30, 60, 100]
cluster_labels = {}   # k → array of labels

# k-means
for k in K_LIST:
    print(f"  k-means k={k} ...", end=" ", flush=True)
    km = KMeans(n_clusters=k, random_state=RNG, n_init=10, max_iter=500)
    labels = km.fit_predict(embs)
    cluster_labels[f"km{k}"] = labels
    sizes = pd.Series(labels).value_counts()
    print(f"done  median_size={sizes.median():.0f}  min={sizes.min()}  max={sizes.max()}")

# Agglomerative at k=60
print(f"  Agglomerative k=60 ...", end=" ", flush=True)
agg = AgglomerativeClustering(n_clusters=60, linkage="ward")
labels_agg = agg.fit_predict(embs)
cluster_labels["agg60"] = labels_agg
sizes_agg = pd.Series(labels_agg).value_counts()
print(f"done  median_size={sizes_agg.median():.0f}  min={sizes_agg.min()}  max={sizes_agg.max()}")

# HDBSCAN existing labels (excluding noise for size stats)
hdb_non_noise = hdb_arr[hdb_arr >= 0]
sizes_hdb = pd.Series(hdb_non_noise).value_counts()
cluster_labels["hdbscan"] = hdb_arr

print(f"\n  Cluster size distributions:")
print(f"  {'Method':<12} {'k':>5} {'n_clusters':>10} {'median':>8} {'min':>6} {'max':>6} {'noise':>8}")
print(f"  {'-'*62}")
for key, labels in cluster_labels.items():
    mask = labels >= 0
    k_eff = len(np.unique(labels[mask]))
    sizes = pd.Series(labels[mask]).value_counts()
    n_noise = (labels < 0).sum()
    print(f"  {key:<12} {len(np.unique(labels[mask])):>5} {k_eff:>10} {sizes.median():>8.0f} "
          f"{sizes.min():>6} {sizes.max():>6} {n_noise:>8}")

# 5 example companies from 5 random clusters at k=60
print(f"\n  Example startups — k-means k=60 (5 random clusters):")
np.random.seed(RNG)
sample_clusters = np.random.choice(60, size=5, replace=False)
for c in sorted(sample_clusters):
    idx_c = np.where(cluster_labels["km60"] == c)[0]
    sample_idx = idx_c[:5] if len(idx_c) >= 5 else idx_c
    print(f"\n  Cluster {c:>3} (N={len(idx_c):>4}):")
    for i in sample_idx:
        nm = str(meta.loc[i,"name"])[:55]
        desc = str(meta.loc[i,"Description"])[:80] if not pd.isna(meta.loc[i,"Description"]) else ""
        print(f"    [{i:>5}] {nm:<55}  {desc[:60]}...")

# ──────────────────────────────────────────────────────────────
# HELPER: compute two-level novelty for a label array
# ──────────────────────────────────────────────────────────────
def compute_two_level(labels, key, year_arr, embs, N, time_aware=True, window=3):
    """
    Returns DataFrame with:
      cluster_id, cluster_isolation, cluster_size_at_founding,
      raw_distinct, std_distinct
    For noise points (label=-1): cluster_id=-1, assign to nearest centroid
    for within-cat distance but flag separately.
    """
    labels = np.array(labels)
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    K = len(unique_labels)

    # Cluster centroids (L2-normalised, excluding noise)
    centroids = np.zeros((K, embs.shape[1]), dtype=np.float32)
    label_to_idx = {}  # cluster_id → row in centroids array
    for ci, lab in enumerate(unique_labels):
        mask = (labels == lab)
        centroids[ci] = embs[mask].mean(axis=0)
        label_to_idx[lab] = ci
    # Normalise centroids
    cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cnorms[cnorms < 1e-10] = 1.0
    centroids_n = centroids / cnorms

    # Inter-centroid cosine distances (K×K)
    # cosine distance = 1 - cosine_similarity; embs already L2-normed
    cent_cos_dist = 1.0 - (centroids_n @ centroids_n.T)  # (K, K)
    np.fill_diagonal(cent_cos_dist, np.nan)

    # Per-startup distances to own centroid
    rows = []
    for i in range(N):
        lab = labels[i]
        yr  = year_arr[i]

        # distance to own centroid (or nearest centroid for noise)
        if lab < 0:  # noise
            # nearest centroid
            dists_to_cents = 1.0 - (embs[i] @ centroids_n.T)
            nearest_cent   = int(np.argmin(dists_to_cents))
            nearest_lab    = unique_labels[nearest_cent]
            own_cent_idx   = nearest_cent
            own_cent_lab   = nearest_lab
            is_noise_pt    = 1
        else:
            own_cent_idx = label_to_idx[lab]
            own_cent_lab = lab
            is_noise_pt  = 0

        raw_distinct = float(1.0 - (embs[i] @ centroids_n[own_cent_idx]))

        rows.append({
            "idx": i,
            "label_assigned": own_cent_lab,
            "is_noise": is_noise_pt,
            "raw_distinct": raw_distinct,
            "year": yr,
        })

    df = pd.DataFrame(rows).set_index("idx")

    # Within-cluster standardised distinctiveness (Haans 2019)
    for lab in unique_labels:
        mask = (df["label_assigned"] == lab) & (df["is_noise"] == 0)
        vals = df.loc[mask, "raw_distinct"]
        mu = vals.mean(); sd = vals.std()
        if sd < 1e-10: sd = 1.0
        df.loc[mask, "std_distinct"] = (vals - mu) / sd
    # Noise points: standardised relative to their nearest centroid cluster
    for i in df.index[df["is_noise"]==1]:
        lab = df.loc[i,"label_assigned"]
        mask = (df["label_assigned"] == lab) & (df["is_noise"] == 0)
        vals = df.loc[mask, "raw_distinct"]
        mu = vals.mean(); sd = vals.std()
        if sd < 1e-10: sd = 1.0
        df.loc[i, "std_distinct"] = (df.loc[i,"raw_distinct"] - mu) / sd

    # Cluster isolation (mean dist to other centroids, time-aware)
    # For each startup at founding year t: use centroids of clusters that had ≥1 member in [t-window, t)
    # Simplified: use full-sample centroids but flag if time-aware yields different results
    # Time-aware: per cluster, size_at_founding = # startups in same cluster founded in [t-window, t)
    df["cluster_isolation"] = np.nan
    df["cluster_size_at_founding"] = np.nan

    for i in df.index:
        yr = df.loc[i, "year"]
        lab = df.loc[i, "label_assigned"]
        cent_idx = label_to_idx[lab] if lab in label_to_idx else None
        if cent_idx is None: continue

        # cluster_isolation: mean cosine dist from own centroid to all other centroids
        row_dists = cent_cos_dist[cent_idx, :]  # NaN on diagonal
        df.loc[i, "cluster_isolation"] = float(np.nanmean(row_dists))

        # cluster_size_at_founding: startups in same cluster with year in [yr-window, yr)
        if not np.isnan(yr) and time_aware:
            same_lab = (df["label_assigned"] == lab) & (df["is_noise"] == 0)
            years_same = df.loc[same_lab, "year"].values
            valid = years_same[(years_same >= yr - window) & (years_same < yr)]
            df.loc[i, "cluster_size_at_founding"] = len(valid)
        else:
            same_lab = (df["label_assigned"] == lab) & (df["is_noise"] == 0)
            df.loc[i, "cluster_size_at_founding"] = same_lab.sum()

    # Cluster dispersion (for reporting, à la Haans)
    cluster_disp = {}
    for lab in unique_labels:
        mask = (df["label_assigned"] == lab) & (df["is_noise"] == 0)
        vals = df.loc[mask, "raw_distinct"]
        cluster_disp[lab] = {"mean": vals.mean(), "sd": vals.std(), "n": mask.sum()}

    return df, cluster_disp, centroids_n

# ──────────────────────────────────────────────────────────────
# STEP 2 & 3: COMPUTE MEASURES FOR ALL CLUSTERINGS
# ──────────────────────────────────────────────────────────────
banner("STEPS 2-3: CATEGORY NOVELTY + WITHIN-CATEGORY DISTINCTIVENESS")

results = {}   # key → df with all measures
disp_tables = {}

for key in ["km12","km30","km60","km100","agg60"]:
    print(f"\n  Computing for {key} ...", end=" ", flush=True)
    labels = cluster_labels[key]
    df_out, cluster_disp, centroids_n = compute_two_level(
        labels, key, year_arr, embs, N)
    results[key] = df_out
    disp_tables[key] = cluster_disp
    print(f"done  [{elapsed()}]")

# Primary: k=60
PRIMARY = "km60"

print(f"\n  Distributions (primary = {PRIMARY}):")
df_p = results[PRIMARY]
for col in ["cluster_isolation","cluster_size_at_founding","raw_distinct","std_distinct"]:
    s = df_p[col].dropna()
    print(f"    {col:<35} N={len(s):>5}  mean={s.mean():+.4f}  sd={s.std():.4f}  "
          f"min={s.min():+.4f}  max={s.max():+.4f}")

# Cluster dispersion table (k=60, top/bottom 10 by sd)
print(f"\n  Cluster dispersion (k=60) — sorted by within-cluster SD of raw_distinct:")
disp_df = pd.DataFrame(disp_tables[PRIMARY]).T
disp_df = disp_df.sort_values("sd", ascending=False)
print(f"  {'cluster':>8} {'N':>6} {'mean_dist':>10} {'sd_dist':>10}  (tight → loose)")
for lab, row in disp_df.head(10).iterrows():
    print(f"  {int(lab):>8} {int(row['n']):>6} {row['mean']:>10.4f} {row['sd']:>10.4f}  LOOSE")
print(f"  ...")
for lab, row in disp_df.tail(5).iterrows():
    print(f"  {int(lab):>8} {int(row['n']):>6} {row['mean']:>10.4f} {row['sd']:>10.4f}  TIGHT")

# ──────────────────────────────────────────────────────────────
# STEP 4: NOISE POINTS
# ──────────────────────────────────────────────────────────────
banner("STEP 4: HDBSCAN NOISE POINTS")
print(f"  N noise = {is_noise.sum()} ({is_noise.mean():.1%})")

# For the primary k-means frame, compare noise vs non-noise
df_p_full = df_p.copy()
df_p_full["is_hdbscan_noise"] = is_noise

for col in ["cluster_isolation","raw_distinct","std_distinct"]:
    g1 = df_p_full[df_p_full["is_hdbscan_noise"]==1][col].dropna()
    g0 = df_p_full[df_p_full["is_hdbscan_noise"]==0][col].dropna()
    from scipy.stats import ttest_ind
    t, p = ttest_ind(g1, g0, equal_var=False)
    print(f"  {col:<35} noise mean={g1.mean():+.4f}  non-noise={g0.mean():+.4f}  "
          f"Δ={g1.mean()-g0.mean():+.4f}  p={p:.4f} {sig(p)}")

print(f"\n  Interpretation: is_hdbscan_noise = 1 → startup is in a sparse region")
print(f"  → this is itself a category-novelty signal (no dense peer cluster)")
print(f"  Retained as binary covariate; not dropped.")

# ──────────────────────────────────────────────────────────────
# STEP 5: VALIDATION
# ──────────────────────────────────────────────────────────────
banner("STEP 5: VALIDATION")

df_val = df_p_full.copy()
df_val["desc_len"] = desc_len
df_val["name"] = [names[i].strip() for i in range(N)]

# 5a: correlation with description word count
print(f"\n  5a. Correlation with description length (proxy for info richness):")
for col in ["cluster_isolation","cluster_size_at_founding","raw_distinct","std_distinct"]:
    s = df_val[[col,"desc_len"]].dropna()
    r, p = spearmanr(s[col], s["desc_len"])
    print(f"    {col:<35} ρ={r:+.4f}  p={p:.4f} {sig(p)}")

# 5b: correlation between the two novelty components
print(f"\n  5b. Correlation between novelty components (should be low):")
pairs = [("cluster_isolation","raw_distinct"),
         ("cluster_isolation","std_distinct"),
         ("cluster_size_at_founding","raw_distinct"),
         ("cluster_size_at_founding","std_distinct"),
         ("raw_distinct","std_distinct")]
for c1, c2 in pairs:
    s = df_val[[c1,c2]].dropna()
    r, p = spearmanr(s[c1], s[c2])
    print(f"    {c1:<35} × {c2:<35} ρ={r:+.4f}  p={p:.4f} {sig(p)}")

# 5c: stability across k=30, 60, 100
print(f"\n  5c. Stability of measures across k (Spearman ρ between k=30, k=60, k=100):")
pairs_k = [("km30","km60"),("km60","km100"),("km30","km100"),("km60","agg60")]
for k1, k2 in pairs_k:
    for col in ["cluster_isolation","raw_distinct","std_distinct"]:
        s = pd.DataFrame({
            "a": results[k1][col].values,
            "b": results[k2][col].values,
        }).dropna()
        r, p = spearmanr(s["a"], s["b"])
        print(f"    {k1} vs {k2}  {col:<35} ρ={r:+.4f}  p={p:.4f} {sig(p)}")

# 5d: top/bottom 10 on each component
print(f"\n  5d. TOP / BOTTOM 10 per component (primary k=60):")
df_val["desc_short"] = [str(d)[:80] for d in desc_arr]

for col, ascending, label in [
    ("cluster_isolation", False, "HIGH category novelty (isolated cluster)"),
    ("cluster_isolation", True,  "LOW category novelty (crowded cluster)"),
    ("std_distinct",      False, "HIGH within-cat distinctiveness"),
    ("std_distinct",      True,  "LOW within-cat distinctiveness (cluster centroid)"),
]:
    print(f"\n  --- {label} ---")
    sub = df_val[["name","desc_short",col]].dropna().sort_values(col, ascending=ascending)
    for rank, (_, row) in enumerate(sub.head(10).iterrows(), 1):
        print(f"  {rank:>2}. [{row[col]:+.4f}] {str(row['name'])[:52]}")
        print(f"      {str(row['desc_short'])[:75]}")

# ──────────────────────────────────────────────────────────────
# HDBSCAN AGREEMENT CHECK
# ──────────────────────────────────────────────────────────────
banner("HDBSCAN NOISE vs K-MEANS NOVELTY: AGREEMENT CHECK")

print(f"  Correlation of is_hdbscan_noise with k-means novelty components (primary k=60):")
for col in ["cluster_isolation","raw_distinct","std_distinct","cluster_size_at_founding"]:
    s = df_val[["is_hdbscan_noise",col]].dropna()
    r, p = spearmanr(s["is_hdbscan_noise"], s[col])
    print(f"    is_hdbscan_noise × {col:<35} ρ={r:+.4f}  p={p:.4f} {sig(p)}")

# Proportion of top-quartile k-means novelty that are HDBSCAN noise
q75_iso = df_val["cluster_isolation"].quantile(0.75)
q75_dis = df_val["std_distinct"].quantile(0.75)
n_noise_in_iso = ((df_val["cluster_isolation"]>=q75_iso) &
                   (df_val["is_hdbscan_noise"]==1)).sum()
n_noise_in_dis = ((df_val["std_distinct"]>=q75_dis) &
                   (df_val["is_hdbscan_noise"]==1)).sum()
top_q = len(df_val) // 4
print(f"\n  Of top-quartile cluster_isolation:   {n_noise_in_iso}/{top_q} ({n_noise_in_iso/top_q:.1%}) are HDBSCAN noise")
print(f"  Of top-quartile std_distinct:        {n_noise_in_dis}/{top_q} ({n_noise_in_dis/top_q:.1%}) are HDBSCAN noise")
if spearmanr(df_val["is_hdbscan_noise"].dropna(),
             df_val["cluster_isolation"].dropna())[0] > 0.3:
    print(f"\n  VERDICT: HDBSCAN noise and k-means cluster_isolation AGREE — noise points "
          f"fall in isolated/sparse regions. Both approaches flag the same startups.")
else:
    print(f"\n  VERDICT: HDBSCAN noise and k-means measures DO NOT strongly agree. "
          f"Report this discrepancy — the two approaches pick up different startups.")

# ──────────────────────────────────────────────────────────────
# SAVE OUTPUT
# ──────────────────────────────────────────────────────────────
banner("SAVING OUTPUT")

save_df = df_val[["name","cluster_isolation","cluster_size_at_founding",
                   "raw_distinct","std_distinct","is_hdbscan_noise","desc_len","year"]].copy()
save_df["name"] = [names[i].strip() for i in range(N)]

# Add km30 and km100 for robustness appendix
save_df["cluster_isolation_km30"] = results["km30"]["cluster_isolation"].values
save_df["std_distinct_km30"]      = results["km30"]["std_distinct"].values
save_df["cluster_isolation_km100"]= results["km100"]["cluster_isolation"].values
save_df["std_distinct_km100"]     = results["km100"]["std_distinct"].values

out_path = OUT / "two_level_novelty.csv"
save_df.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(save_df):,} rows)")
print(f"  Columns: {save_df.columns.tolist()}")
print(f"\n  [{elapsed()}]")
