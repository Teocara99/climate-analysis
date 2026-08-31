"""
Script 48: Investor Semantic Embeddings & Syndicate Diversity → Outcomes
=========================================================================

SPECTER2 embeddings of investor descriptions → syndicate semantic diversity
→ regression vs. failure/exits/patents, horse-race with cross-community.

Steps:
  1. Build investor descriptions (PitchBook priority, scraped fallback)
  2. SPECTER2 embeddings → data/investor_embeddings.npy
  3. Semantic similarity network + Louvain communities → NMI vs co-invest
  4. Company-level syndicate semantic diversity metrics
  5. Regression: diversity → outcomes (LPM, HC1)
  6. K-means thematic clustering (K=12) + TF-IDF naming
  7. UMAP 2D coordinates → output/semantic/umap_coords.csv
"""

import warnings; warnings.filterwarnings("ignore")
import sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine, cdist
from scipy.sparse import coo_matrix
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
import networkx as nx
import community as community_louvain
import umap

sys.path.insert(0, ".")
from load_data import load_deals

# ── Config ────────────────────────────────────────────────────────────────
OUT     = Path("output/semantic"); OUT.mkdir(parents=True, exist_ok=True)
DATA    = Path("data")
EMB_FILE  = DATA / "investor_embeddings.npy"
NAME_FILE = DATA / "investor_names.csv"
CTRL    = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: BUILD INVESTOR DESCRIPTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 1: BUILD INVESTOR DESCRIPTIONS")
print("="*72)

# Load PitchBook descriptions
pb = pd.read_excel("data/Investors_data.xlsx", sheet_name="Data", header=7)
pb = pb.dropna(subset=["Investors"])
pb["Investors"] = pb["Investors"].str.strip()
pb = pb.drop_duplicates("Investors")
pb_desc = pb.set_index("Investors")["Description"].dropna().to_dict()
print(f"  PitchBook descriptions: {len(pb_desc):,}")

# Load scraped website descriptions
with open("output/investor_descriptions.json") as f:
    scraped = json.load(f)
scraped_desc = {
    k.strip(): v["description"]
    for k, v in scraped.items()
    if isinstance(v, dict) and len(v.get("description", "").strip()) > 80
}
print(f"  Scraped descriptions (>80 chars): {len(scraped_desc):,}")

# Load network investors (the set we care about for NMI)
net = pd.read_csv("output/network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
network_investors = set(net["investor"])
inv_to_comm = net.set_index("investor")["community"].to_dict()
print(f"  Network investors: {len(network_investors):,}")

# Build best description per investor (PB priority)
desc_map   = {}  # investor → description text
source_map = {}  # investor → source
for inv in network_investors:
    if inv in pb_desc and len(str(pb_desc[inv]).strip()) > 30:
        desc_map[inv]   = str(pb_desc[inv]).strip()
        source_map[inv] = "pitchbook"
    elif inv in scraped_desc:
        desc_map[inv]   = scraped_desc[inv].strip()
        source_map[inv] = "scraped"

n_pb      = sum(1 for v in source_map.values() if v == "pitchbook")
n_scraped = sum(1 for v in source_map.values() if v == "scraped")
print(f"\n  Coverage within network investors:")
print(f"    PitchBook source : {n_pb:,}  ({n_pb/len(network_investors):.1%})")
print(f"    Scraped fallback : {n_scraped:,}  ({n_scraped/len(network_investors):.1%})")
print(f"    No description   : {len(network_investors)-len(desc_map):,}")

# Investors to embed: all with descriptions
inv_list  = sorted(desc_map.keys())
texts     = [desc_map[inv] for inv in inv_list]
print(f"\n  Investors to embed: {len(inv_list):,}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: SPECTER2 EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 2: SPECTER2 EMBEDDINGS")
print("="*72)

if EMB_FILE.exists() and NAME_FILE.exists():
    saved_names = pd.read_csv(NAME_FILE)["investor"].tolist()
    saved_embs  = np.load(EMB_FILE)
    # Check if we already have all needed investors
    saved_set = set(saved_names)
    need_new  = [inv for inv in inv_list if inv not in saved_set]
    print(f"  Found saved embeddings: {len(saved_names):,}")
    print(f"  Need to embed (new): {len(need_new):,}")

    if len(need_new) == 0:
        # Reorder to match inv_list
        idx_map = {n: i for i, n in enumerate(saved_names)}
        keep    = [inv for inv in inv_list if inv in idx_map]
        idxs    = [idx_map[inv] for inv in keep]
        inv_list = keep
        texts    = [desc_map[inv] for inv in keep]
        embeddings = saved_embs[idxs]
        print(f"  Using cached embeddings for {len(embeddings):,} investors.")
    else:
        # Embed new ones and concatenate
        from sentence_transformers import SentenceTransformer
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  Loading SPECTER2 (device={device})…")
        model = SentenceTransformer("allenai/specter2_base", device=device)
        new_texts = [desc_map[inv] for inv in need_new]
        print(f"  Embedding {len(new_texts):,} new investors (batch_size=64)…")
        new_embs = model.encode(new_texts, batch_size=64, show_progress_bar=True,
                                normalize_embeddings=False)
        # Combine
        all_names = saved_names + need_new
        all_embs  = np.vstack([saved_embs, new_embs])
        pd.DataFrame({"investor": all_names}).to_csv(NAME_FILE, index=False)
        np.save(EMB_FILE, all_embs)
        # Reorder
        idx_map = {n: i for i, n in enumerate(all_names)}
        keep    = [inv for inv in inv_list if inv in idx_map]
        idxs    = [idx_map[inv] for inv in keep]
        inv_list   = keep
        texts      = [desc_map[inv] for inv in keep]
        embeddings = all_embs[idxs]
        print(f"  Saved updated embeddings: {len(all_names):,}")
else:
    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading SPECTER2 (device={device})…")
    model = SentenceTransformer("allenai/specter2_base", device=device)
    print(f"  Embedding {len(texts):,} investors (batch_size=64)…")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=False)
    pd.DataFrame({"investor": inv_list}).to_csv(NAME_FILE, index=False)
    np.save(EMB_FILE, embeddings)
    print(f"  Saved → {EMB_FILE}  {EMB_FILE.stat().st_size/1e6:.1f} MB")

# L2-normalize for cosine similarity via dot product
emb_norm = normalize(embeddings, norm="l2")
emb_df   = pd.DataFrame({"investor": inv_list})
emb_df["comm_invest"] = emb_df["investor"].map(inv_to_comm)
print(f"\n  Embedding matrix: {emb_norm.shape}  (L2 normalized)")
print(f"  Norm check (mean): {np.linalg.norm(emb_norm, axis=1).mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: SEMANTIC SIMILARITY NETWORK + COMMUNITIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 3: SEMANTIC NETWORK + NMI")
print("="*72)

N = len(inv_list)

# -- Sample pairwise sims to calibrate threshold --
rng = np.random.default_rng(42)
sample_idx = rng.choice(N, size=min(2000, N), replace=False)
sample_emb  = emb_norm[sample_idx]
sample_sims = (sample_emb @ sample_emb.T).ravel()
sample_sims = sample_sims[sample_sims < 0.9999]  # exclude self-similarity
print(f"  Pairwise cosine similarity distribution (sample N=2000):")
for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"    P{int(q*100):3d}: {np.quantile(sample_sims, q):.4f}")

# Use k-NN instead of global threshold: 10 nearest neighbors per investor
# This gives a sparse, well-defined graph regardless of description similarity
K_NN = 10
print(f"\n  Building k-NN graph (k={K_NN}) in chunks…")
chunk_size = 2000
rows_list, cols_list, weights_list = [], [], []

for start in range(0, N, chunk_size):
    end = min(start + chunk_size, N)
    chunk = emb_norm[start:end]          # (C, D)
    sims  = chunk @ emb_norm.T           # (C, N)
    # Zero out self-connections
    for local_i in range(end - start):
        sims[local_i, start + local_i] = -1.0
    # Top-k per row
    top_k_idx = np.argpartition(sims, -K_NN, axis=1)[:, -K_NN:]
    for local_i in range(end - start):
        global_i = start + local_i
        for j in top_k_idx[local_i]:
            if j > global_i:  # upper triangle only
                w = float(sims[local_i, j])
                rows_list.append(global_i)
                cols_list.append(j)
                weights_list.append(w)
    if (start // chunk_size) % 3 == 0:
        print(f"    Progress: {end:,}/{N:,}")

print(f"  k-NN edges: {len(rows_list):,}")

# Build semantic network
G_sem = nx.Graph()
G_sem.add_nodes_from(range(N))
for i, j, w in zip(rows_list, cols_list, weights_list):
    if G_sem.has_edge(i, j):
        G_sem[i][j]["weight"] = max(G_sem[i][j]["weight"], w)
    else:
        G_sem.add_edge(i, j, weight=max(w, 0.0))

print(f"  Semantic k-NN network: {G_sem.number_of_nodes():,} nodes, {G_sem.number_of_edges():,} edges")
cc = list(nx.connected_components(G_sem))
print(f"  Connected components: {len(cc)}  (largest: {max(len(c) for c in cc):,})")

# Louvain on semantic network
part_sem = community_louvain.best_partition(G_sem, weight="weight", random_state=42)
sem_comms = pd.Series(part_sem)
n_sem_comms = sem_comms.nunique()
print(f"  Semantic Louvain communities: {n_sem_comms}")

# NMI: semantic communities vs co-investment communities
# Use only investors present in both
invest_comm_arr = emb_df["comm_invest"].values
sem_comm_arr    = np.array([part_sem.get(i, -1) for i in range(N)])
mask = pd.notna(invest_comm_arr) & (sem_comm_arr >= 0)
nmi = normalized_mutual_info_score(
    invest_comm_arr[mask].astype(int),
    sem_comm_arr[mask]
)
print(f"\n  NMI (semantic vs co-investment communities): {nmi:.4f}")
if nmi < 0.15:
    print("  → LOW NMI: semantic and co-investment communities capture DIFFERENT structure.")
    print("    Investors who talk alike do NOT necessarily invest together.")
    print("    Semantic diversity adds a DISTINCT dimension beyond network position.")
elif nmi < 0.40:
    print("  → MODERATE NMI: partial overlap — both dimensions carry unique information.")
else:
    print("  → HIGH NMI: communities are largely redundant — co-investment ≈ thematic.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: COMPANY-LEVEL SYNDICATE DIVERSITY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 4: COMPANY-LEVEL SYNDICATE SEMANTIC DIVERSITY")
print("="*72)

# Build embedding lookup: investor → embedding vector
emb_lookup = {inv: emb_norm[i] for i, inv in enumerate(inv_list)}

# Load deals
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))

# For each company: get all investors that appear in ANY round (we use unique)
co_investors = (d_exp.groupby("co")["inv"]
                .apply(lambda x: list(set(x)))
                .reset_index()
                .rename(columns={"inv": "investors"}))

diversity_rows = []
overall_centroid = emb_norm.mean(axis=0)  # global centroid

for _, row in co_investors.iterrows():
    co   = row["co"]
    invs = [i for i in row["investors"] if i in emb_lookup]

    if len(invs) < 2:
        diversity_rows.append({
            "co": co, "n_investors_embedded": len(invs),
            "semantic_diversity": np.nan,
            "semantic_range": np.nan,
            "semantic_centroid_dist": np.nan,
        })
        continue

    vecs = np.array([emb_lookup[i] for i in invs])          # (K, D)

    # Pairwise cosine distances (1 - cosine_sim)
    # Since vecs are L2-normalized, dot product = cosine similarity
    sim_matrix = vecs @ vecs.T                              # (K, K)
    dist_matrix = 1 - sim_matrix                           # cosine distance
    upper = dist_matrix[np.triu_indices(len(invs), k=1)]

    # Syndicate centroid
    centroid = vecs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    diversity_rows.append({
        "co": co,
        "n_investors_embedded": len(invs),
        "semantic_diversity": float(upper.mean()),     # 1 - mean pairwise cos sim
        "semantic_range": float(upper.max()),          # max pairwise distance
        "semantic_centroid_dist": float(
            1 - float(centroid @ overall_centroid)),   # distance from mainstream
    })

div_df = pd.DataFrame(diversity_rows)
has_div = div_df["semantic_diversity"].notna()
print(f"  Companies with ≥2 embedded investors: {has_div.sum():,}")
print(f"  Companies with <2 embedded investors: {(~has_div).sum():,}")
print(f"\n  Semantic diversity (mean pairwise cosine distance):")
d = div_df[has_div]["semantic_diversity"]
print(f"    Mean={d.mean():.4f}  Std={d.std():.4f}  P25={d.quantile(0.25):.4f}  "
      f"P75={d.quantile(0.75):.4f}")
print(f"\n  Semantic range (max pairwise distance):")
r = div_df[has_div]["semantic_range"]
print(f"    Mean={r.mean():.4f}  Std={r.std():.4f}")
print(f"\n  Centroid distance from overall investor centroid:")
c = div_df[has_div]["semantic_centroid_dist"]
print(f"    Mean={c.mean():.4f}  Std={c.std():.4f}")

div_df.to_csv(OUT / "company_semantic_diversity.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: REGRESSIONS — DOES SEMANTIC DIVERSITY PREDICT OUTCOMES?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 5: REGRESSIONS — SEMANTIC DIVERSITY → OUTCOMES")
print("="*72)

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]   = cdf["log_capital"]
cdf["nrounds"]   = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

# Merge diversity metrics
df = cdf.merge(div_df[["co","semantic_diversity","semantic_range","semantic_centroid_dist",
                         "n_investors_embedded"]], on="co", how="left")
df["has_diversity"] = df["semantic_diversity"].notna()
print(f"  Merged companies: {len(df):,}")
print(f"  With diversity metrics: {df['has_diversity'].sum():,}")

sub = df[df["has_diversity"]].copy()
sub["sem_div_std"] = (sub["semantic_diversity"] - sub["semantic_diversity"].mean()) \
                      / sub["semantic_diversity"].std()
sub["sem_range_std"] = (sub["semantic_range"] - sub["semantic_range"].mean()) \
                        / sub["semantic_range"].std()

print(f"\n  Cross-community vs semantic diversity correlation:")
corr = sub[["cross_community","semantic_diversity","semantic_range","semantic_centroid_dist"]].corr()
print(f"    cross_community × semantic_diversity: r={corr.loc['cross_community','semantic_diversity']:.4f}")
print(f"    cross_community × semantic_range:     r={corr.loc['cross_community','semantic_range']:.4f}")

# ── Failure Models ──
print(f"\n  {'─'*65}")
print(f"  OUTCOME: Failure (is_failed)")
print(f"  {'─'*65}")

models_fail = {
    "M1: cross_community only":
        f"is_failed ~ cross_community + {CTRL}",
    "M2: semantic_diversity only":
        f"is_failed ~ sem_div_std + {CTRL}",
    "M3: both (horse race)":
        f"is_failed ~ cross_community + sem_div_std + {CTRL}",
    "M4: +semantic_range":
        f"is_failed ~ cross_community + sem_div_std + sem_range_std + {CTRL}",
}

results = {}
for label, formula in models_fail.items():
    m = smf.ols(formula, data=sub.dropna(subset=["is_failed"])).fit(cov_type="HC1")
    results[label] = m
    b_cross = m.params.get("cross_community", np.nan)
    p_cross = m.pvalues.get("cross_community", 1.0)
    b_div   = m.params.get("sem_div_std", np.nan)
    p_div   = m.pvalues.get("sem_div_std", 1.0)
    n       = int(m.nobs)
    r2      = m.rsquared
    print(f"\n  [{label}]  N={n:,}  R²={r2:.4f}")
    if not np.isnan(b_cross):
        print(f"    cross_community   : β={b_cross:>+.4f}  {sig(p_cross)}")
    if not np.isnan(b_div):
        print(f"    sem_diversity(std): β={b_div:>+.4f}  {sig(p_div)}")

# ── Positive outcomes ──
for outcome_label, outcome_col in [
    ("log_employees", "log_employees"),
    ("has_patent", "has_patent"),
    ("exited", "exited"),
]:
    if outcome_col not in sub.columns:
        continue
    sub_o = sub.dropna(subset=[outcome_col]).copy()
    if len(sub_o) < 100:
        continue
    formula = f"{outcome_col} ~ cross_community + sem_div_std + {CTRL}"
    try:
        m = smf.ols(formula, data=sub_o).fit(cov_type="HC1")
        b_cross = m.params.get("cross_community", np.nan)
        p_cross = m.pvalues.get("cross_community", 1.0)
        b_div   = m.params.get("sem_div_std", np.nan)
        p_div   = m.pvalues.get("sem_div_std", 1.0)
        print(f"\n  OUTCOME: {outcome_label}  N={int(m.nobs):,}  R²={m.rsquared:.4f}")
        print(f"    cross_community   : β={b_cross:>+.4f}  {sig(p_cross)}")
        print(f"    sem_diversity(std): β={b_div:>+.4f}  {sig(p_div)}")
    except Exception as e:
        print(f"  [{outcome_label}] error: {e}")

# Centroid distance (niche vs mainstream)
m_cen = smf.ols(
    f"is_failed ~ cross_community + sem_div_std + semantic_centroid_dist + {CTRL}",
    data=sub.dropna(subset=["is_failed"])).fit(cov_type="HC1")
b_cen = m_cen.params.get("semantic_centroid_dist", np.nan)
p_cen = m_cen.pvalues.get("semantic_centroid_dist", 1.0)
print(f"\n  NICHE vs MAINSTREAM (centroid distance from mean investor):")
print(f"    sem_centroid_dist : β={b_cen:>+.4f}  {sig(p_cen)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: K-MEANS THEMATIC CLUSTERING + TF-IDF NAMING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 6: THEMATIC CLUSTERING (K=12) + TF-IDF NAMING")
print("="*72)

K = 12
print(f"  Fitting K-means (K={K}) on {emb_norm.shape[0]:,} embeddings…")
km = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(emb_norm)
print(f"  Done. Cluster sizes:")

# TF-IDF naming
tfidf = TfidfVectorizer(max_features=5000, stop_words="english",
                         min_df=3, ngram_range=(1, 2))
tfidf.fit(texts)
feat_names = np.array(tfidf.get_feature_names_out())

cluster_names = {}
cluster_info  = {}
for k in range(K):
    mask_k     = km_labels == k
    n_k        = mask_k.sum()
    texts_k    = [texts[i] for i in np.where(mask_k)[0]]
    # TF-IDF centroid for cluster
    X_k        = tfidf.transform(texts_k).toarray()
    centroid_k = X_k.mean(axis=0)
    top_idx    = centroid_k.argsort()[::-1][:8]
    top_terms  = feat_names[top_idx].tolist()
    name       = " / ".join(top_terms[:3])
    cluster_names[k] = name
    cluster_info[k] = {"size": int(n_k), "top_terms": top_terms, "name": name}
    print(f"    Cluster {k:2d} (N={n_k:5,}): {name}")

# Save cluster assignments
emb_df["thematic_cluster"] = km_labels
emb_df.to_csv(OUT / "investor_clusters.csv", index=False)

# ── Company-level thematic diversity ──
print(f"\n  Computing company-level thematic cluster diversity…")
inv_cluster = dict(zip(inv_list, km_labels))

thematic_rows = []
for _, row in co_investors.iterrows():
    co   = row["co"]
    invs = [i for i in row["investors"] if i in inv_cluster]
    if len(invs) == 0:
        thematic_rows.append({"co": co, "n_thematic_clusters": np.nan})
    else:
        clusters_present = set(inv_cluster[i] for i in invs)
        thematic_rows.append({
            "co": co,
            "n_thematic_clusters": len(clusters_present),
            "n_investors_clustered": len(invs),
        })

them_df = pd.DataFrame(thematic_rows)
df = df.merge(them_df, on="co", how="left")
df["log_n_thematic"] = np.log1p(df["n_thematic_clusters"].fillna(0))

# ── Horse race: n_thematic_clusters vs cross_community ──
sub2 = df[df["n_thematic_clusters"].notna() & df["semantic_diversity"].notna()].copy()
sub2["sem_div_std"] = (sub2["semantic_diversity"] - sub2["semantic_diversity"].mean()) \
                      / sub2["semantic_diversity"].std()
sub2["n_comm_std"] = (sub2["n_communities"] - sub2["n_communities"].mean()) \
                      / sub2["n_communities"].std() if "n_communities" in sub2 else 0.0

print(f"\n  {'─'*65}")
print(f"  HORSE RACE: Network diversity vs Thematic diversity vs Semantic diversity")
print(f"  Outcome: failure  N={sub2['is_failed'].notna().sum():,}")
print(f"  {'─'*65}")

for label, formula in [
    ("M1: cross_community only",
     f"is_failed ~ cross_community + {CTRL}"),
    ("M2: n_thematic_clusters only",
     f"is_failed ~ log_n_thematic + {CTRL}"),
    ("M3: semantic_diversity only",
     f"is_failed ~ sem_div_std + {CTRL}"),
    ("M4: all three",
     f"is_failed ~ cross_community + log_n_thematic + sem_div_std + {CTRL}"),
]:
    try:
        m = smf.ols(formula, data=sub2.dropna(subset=["is_failed"])).fit(cov_type="HC1")
        print(f"\n  [{label}]  R²={m.rsquared:.4f}  N={int(m.nobs):,}")
        for var, lbl in [
            ("cross_community","cross_community"),
            ("log_n_thematic","log(n_thematic_clusters)"),
            ("sem_div_std","semantic_diversity(std)"),
        ]:
            if var in m.params:
                print(f"    {lbl:<30}: β={m.params[var]:>+.4f}  {sig(m.pvalues[var])}")
    except Exception as e:
        print(f"  [{label}] ERR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: UMAP 2D COORDINATES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 7: UMAP 2D PROJECTION")
print("="*72)

print(f"  Fitting UMAP on {emb_norm.shape[0]:,} embeddings (n_neighbors=30)…")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                    metric="cosine", random_state=42, verbose=False)
coords  = reducer.fit_transform(emb_norm)

umap_df = pd.DataFrame({
    "investor":          inv_list,
    "umap_x":            coords[:, 0],
    "umap_y":            coords[:, 1],
    "invest_community":  [inv_to_comm.get(inv, -1) for inv in inv_list],
    "thematic_cluster":  km_labels,
    "source":            [source_map.get(inv, "none") for inv in inv_list],
})
umap_df.to_csv(OUT / "umap_coords.csv", index=False)
print(f"  UMAP saved → {OUT}/umap_coords.csv")

# ─── Summary ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("SUMMARY")
print("="*72)

m_base = results.get("M1: cross_community only")
m_sem  = results.get("M2: semantic_diversity only")
m_both = results.get("M3: both (horse race)")

b_cc_base = m_base.params.get("cross_community", np.nan) if m_base else np.nan
p_cc_base = m_base.pvalues.get("cross_community", 1.0)   if m_base else 1.0
b_cc_both = m_both.params.get("cross_community", np.nan) if m_both else np.nan
p_cc_both = m_both.pvalues.get("cross_community", 1.0)   if m_both else 1.0
b_sd_sem  = m_sem.params.get("sem_div_std", np.nan)      if m_sem else np.nan
p_sd_sem  = m_sem.pvalues.get("sem_div_std", 1.0)        if m_sem else 1.0
b_sd_both = m_both.params.get("sem_div_std", np.nan)     if m_both else np.nan
p_sd_both = m_both.pvalues.get("sem_div_std", 1.0)       if m_both else 1.0

print(f"""
  EMBEDDING COVERAGE
  ─────────────────────────────────────────────
  Network investors with embeddings: {len(inv_list):,}/{len(network_investors):,}  ({len(inv_list)/len(network_investors):.1%})
  PitchBook descriptions used: {n_pb:,}  Scraped fallback: {n_scraped:,}

  SEMANTIC NETWORK
  ─────────────────────────────────────────────
  NMI (semantic vs co-investment communities): {nmi:.4f}
  Semantic Louvain communities: {n_sem_comms}
  Co-investment Louvain communities: 46

  FAILURE REGRESSIONS (LPM, HC1)
  ─────────────────────────────────────────────
  M1 (cross_community alone):   β_cross={b_cc_base:>+.4f}  {sig(p_cc_base)}
  M2 (semantic_div alone):      β_div  ={b_sd_sem:>+.4f}  {sig(p_sd_sem)}
  M3 (horse race — both):
    cross_community:             β={b_cc_both:>+.4f}  {sig(p_cc_both)}
    semantic_diversity:          β={b_sd_both:>+.4f}  {sig(p_sd_both)}

  KEY VERDICT:
""")

div_sig    = p_sd_sem < 0.10
cross_surv = p_cc_both < 0.10
subsumes   = div_sig and not cross_surv

if not div_sig:
    print("  ❌ Semantic diversity does NOT independently predict failure.")
    print("     What investors SAY about themselves doesn't predict outcomes —")
    print("     what matters is WHO they invest with (network position).")
elif subsumes:
    print("  ⚠️  Semantic diversity predicts failure, AND cross-community drops out.")
    print("     Semantic diversity SUBSUMES cross-community — they measure the same thing.")
    print("     Cross-community investment was really capturing thematic heterogeneity.")
elif div_sig and cross_surv:
    print("  ✅ BOTH semantic diversity AND cross-community independently predict outcomes.")
    print("     Two distinct mechanisms: thematic diversity in descriptions AND")
    print("     structural bridging in the co-investment network both matter.")
    print("     → Paper can report them as complementary findings.")
else:
    print("  → Mixed result — check individual model outputs above.")

print(f"""
  THEMATIC CLUSTERS (K=12)
  ─────────────────────────────────────────────""")
for k, info in cluster_info.items():
    print(f"  Cluster {k:2d} (N={info['size']:5,}): {info['name']}")

print(f"""
  OUTPUT FILES
  ─────────────────────────────────────────────
  data/investor_embeddings.npy    — SPECTER2 embeddings
  data/investor_names.csv         — investor name order
  output/semantic/company_semantic_diversity.csv
  output/semantic/investor_clusters.csv
  output/semantic/umap_coords.csv
""")
