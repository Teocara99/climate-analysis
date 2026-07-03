"""
11a_novelty_compute.py — Novelty Index for Climate Tech Startups
================================================================
Phase 1 of 2: compute embeddings, novelty scores, clusters.
Outputs are cached so 11b_novelty_analysis.py can re-run instantly.

PIPELINE
--------
1. Load PitchBook descriptions (9,751 companies, avg 410 chars)
2. Embed with TWO models and compare:
   A. BAAI/bge-large-en-v1.5   — best general semantic similarity
      (validated in prior venture-client project, #1 on MTEB)
   B. allenai/specter2_base     — scientific/technical domain specialist
      (trained on citation graphs of research papers — captures
       technological differentiation rather than linguistic style)
3. ABTT preprocessing per model
   (All-But-The-Top: removes top D principal components that encode
    dataset-wide discourse bias — see Mu & Viswanath 2018 "All-but-the-Top")
4. Peer novelty = 1 − max cosine similarity to any other company
   High novelty ≈ the company is semantically furthest from all peers.
5. UMAP reduction
   • 10-D: input to HDBSCAN clustering
   • 2-D:  visualisation
6. HDBSCAN clustering (density-based, no k to specify)
7. c-TF-IDF cluster keywords (BERTopic-style)
8. Model comparison → select primary model by silhouette score

WHY THESE CHOICES (vs novelty_index.py defaults)
-------------------------------------------------
• BGE-large / SPECTER2 > all-MiniLM-L6-v2
  MiniLM trades quality for speed. For 9,751 static descriptions
  (no web scraping) encoding takes ~2 min per model regardless.

• HDBSCAN > K-means
  K-means forces every point into a cluster and requires guessing k.
  HDBSCAN finds natural density-based clusters and labels genuine
  outliers as noise — more honest for a corpus where some niches
  (e.g. carbon removal) are tight clusters and others sparse.

• UMAP > PCA
  PCA is a global linear projection; UMAP preserves local neighbourhood
  structure, producing maps where semantically similar companies stay
  close even when globally distant. The venture-client biodiversity
  analysis confirmed this produces more interpretable maps.

• ABTT preprocessing
  All scientific/technical text corpora share a "common direction"
  in embedding space (domain jargon, boilerplate). ABTT removes the
  top-D components of the mean-centred covariance matrix, sharpening
  the dimensions that carry genuine company-level signal.

OUTPUTS (saved to output/novelty/)
-----------------------------------
novelty_bge.pkl          — BGE embeddings + novelty df (full)
novelty_specter.pkl      — SPECTER2 embeddings + novelty df (full)
novelty_scores.csv       — company × {bge_novelty, specter_novelty,
                           cluster_bge, cluster_specter, keywords_*}
umap_2d_bge.npy          — 2-D UMAP coords (BGE)
umap_2d_specter.npy      — 2-D UMAP coords (SPECTER2)
model_comparison.txt     — silhouette, correlation, winner
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from load_data import load_companies

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = Path(__file__).parent / "output" / "novelty"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_A = "BAAI/bge-large-en-v1.5"
MODEL_B = "allenai/specter2_base"
ABTT_D  = 3      # top principal components to remove
MIN_DESC_LEN = 50


# ── helpers ──────────────────────────────────────────────────────────────────

def abtt(embs: np.ndarray, d: int = ABTT_D) -> np.ndarray:
    """Remove top-d principal components then L2-normalise (Mu & Viswanath 2018)."""
    mu = embs.mean(axis=0, keepdims=True)
    centred = embs - mu
    svd = TruncatedSVD(n_components=d, random_state=42)
    svd.fit(centred)
    for v in svd.components_:
        centred -= centred.dot(v[:, None]) * v[None, :]
    return normalize(centred, norm="l2")


def peer_novelty(embs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    novelty[i]  = 1 - max cosine_similarity(i, j≠i)
    density[i]  = mean cosine_similarity of 10 nearest neighbours
    """
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, -1.0)
    k = min(10, len(embs) - 1)
    novelty = 1 - sim.max(axis=1)
    density = np.sort(sim, axis=1)[:, -k:].mean(axis=1)
    nearest = sim.argmax(axis=1)
    return novelty, density, nearest


def run_umap(embs: np.ndarray, n_dim: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_dim, n_neighbors=n_neighbors,
        min_dist=min_dist, metric="cosine",
        random_state=42, n_jobs=1,
    )
    return reducer.fit_transform(embs)


def run_hdbscan(coords_10d: np.ndarray) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=25, min_samples=5,
        metric="euclidean", cluster_selection_method="eom",
    )
    return clusterer.fit_predict(coords_10d)


def ctfidf_keywords(descriptions: list[str], labels: np.ndarray, top_n: int = 8) -> dict:
    """BERTopic-style class TF-IDF to label each cluster."""
    stop = {"the","a","an","is","are","to","of","and","in","for","its","that","this",
            "with","by","on","it","has","which","be","as","or","their","also","have",
            "from","at","was","not","can","we","our","company","startup","develop",
            "developer","solution","solutions","platform","technology","technologies",
            "based","designed","help","provide","provides","enable","enables","use",
            "using","used","offers","offering","climate","tech","clean"}

    cluster_ids = sorted(set(labels))
    cluster_ids = [c for c in cluster_ids if c != -1]
    docs_per_cluster = {}
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        docs_per_cluster[c] = " ".join(descriptions[i] for i in idx)

    vec = CountVectorizer(stop_words=list(stop), ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform(list(docs_per_cluster.values()))
    terms = np.array(vec.get_feature_names_out())

    # c-TF-IDF
    tf = X.toarray() / (X.toarray().sum(axis=1, keepdims=True) + 1e-9)
    df_ = (X > 0).toarray().sum(axis=0)
    idf = np.log(1 + len(cluster_ids) / (df_ + 1))
    ctf = tf * idf

    keywords = {}
    for i, c in enumerate(cluster_ids):
        top_idx = ctf[i].argsort()[::-1][:top_n]
        keywords[c] = ", ".join(terms[top_idx])
    return keywords


# ── main ─────────────────────────────────────────────────────────────────────

def embed_and_score(model_name: str, tag: str,
                    names: list, descriptions: list) -> pd.DataFrame:
    cache = OUT / f"novelty_{tag}.pkl"

    if cache.exists():
        print(f"  [{tag}] loading cached embeddings...")
        with open(cache, "rb") as f:
            data = pickle.load(f)
        embs_raw = data["embs_raw"]
    else:
        print(f"  [{tag}] loading model {model_name}...")
        model = SentenceTransformer(model_name)
        print(f"  [{tag}] encoding {len(descriptions):,} descriptions...")
        embs_raw = model.encode(
            descriptions, batch_size=64, show_progress_bar=True,
            normalize_embeddings=False,
        )
        with open(cache, "wb") as f:
            pickle.dump({"embs_raw": embs_raw, "names": names}, f)
        print(f"  [{tag}] embeddings cached → {cache}")

    print(f"  [{tag}] ABTT preprocessing (d={ABTT_D})...")
    embs = abtt(np.array(embs_raw))

    print(f"  [{tag}] computing peer novelty...")
    novelty, density, nearest_idx = peer_novelty(embs)
    nearest_names = [names[i] for i in nearest_idx]

    print(f"  [{tag}] UMAP 10-D (for clustering)...")
    coords_10d = run_umap(embs, n_dim=10, n_neighbors=30, min_dist=0.0)

    print(f"  [{tag}] UMAP 2-D (for visualisation)...")
    coords_2d = run_umap(embs, n_dim=2, n_neighbors=15, min_dist=0.1)
    np.save(OUT / f"umap_2d_{tag}.npy", coords_2d)

    print(f"  [{tag}] HDBSCAN clustering...")
    labels = run_hdbscan(coords_10d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  [{tag}] {n_clusters} clusters, {n_noise} noise points")

    print(f"  [{tag}] c-TF-IDF keywords...")
    keywords = ctfidf_keywords(np.array(descriptions), labels)
    keyword_labels = [keywords.get(l, "noise") for l in labels]

    # silhouette (on 10D UMAP coords, excluding noise)
    mask = labels != -1
    sil = silhouette_score(coords_10d[mask], labels[mask]) if mask.sum() > 1 else 0.0
    print(f"  [{tag}] silhouette score: {sil:.4f}")

    df = pd.DataFrame({
        "name":              names,
        f"{tag}_novelty":    novelty.round(4),
        f"{tag}_density":    density.round(4),
        f"{tag}_nearest":    nearest_names,
        f"{tag}_cluster":    labels,
        f"{tag}_keywords":   keyword_labels,
        f"umap_{tag}_x":     coords_2d[:, 0],
        f"umap_{tag}_y":     coords_2d[:, 1],
    })
    df["_sil"] = sil
    return df


def main():
    print("Loading PitchBook company descriptions...")
    companies = load_companies()
    companies = companies[["Companies","Description","Verticals",
                            "HQ Global Region","Year Founded"]].copy()
    companies = companies[companies["Description"].str.len().fillna(0) >= MIN_DESC_LEN]
    names        = companies["Companies"].tolist()
    descriptions = companies["Description"].tolist()
    print(f"  {len(names):,} companies with valid descriptions")

    print("\n=== MODEL A: BGE-large ===")
    df_a = embed_and_score(MODEL_A, "bge", names, descriptions)

    print("\n=== MODEL B: SPECTER2 ===")
    df_b = embed_and_score(MODEL_B, "specter", names, descriptions)

    # merge
    df = df_a.merge(df_b.drop(columns=["_sil"]), on="name", how="inner")
    df = df.merge(companies.rename(columns={"Companies":"name"}), on="name", how="left")

    # correlation
    corr = df["bge_novelty"].corr(df["specter_novelty"])
    sil_a = df["_sil"].iloc[0]
    sil_b = df_b["_sil"].iloc[0]
    winner = "bge" if sil_a >= sil_b else "specter"

    df.drop(columns=["_sil"], inplace=True)
    df.to_csv(OUT / "novelty_scores.csv", index=False)
    print(f"\nSaved → {OUT / 'novelty_scores.csv'}")

    report = [
        "MODEL COMPARISON REPORT",
        "="*50,
        f"BGE-large silhouette score:  {sil_a:.4f}",
        f"SPECTER2 silhouette score:   {sil_b:.4f}",
        f"Novelty score correlation:   {corr:.4f}",
        f"Primary model:               {winner.upper()} (higher silhouette)",
        "",
        f"BGE-large clusters:   {df['bge_cluster'].nunique()} "
        f"(excl. noise={( df['bge_cluster']==-1).sum()})",
        f"SPECTER2 clusters:    {df['specter_cluster'].nunique()} "
        f"(excl. noise={(df['specter_cluster']==-1).sum()})",
        "",
        "Top 5 most novel (BGE):",
    ]
    for _, r in df.nlargest(5,"bge_novelty").iterrows():
        report.append(f"  {r['bge_novelty']:.3f}  {r['name']}  "
                     f"[{r['bge_keywords'][:60]}]")
    report += ["","Top 5 most novel (SPECTER2):"]
    for _, r in df.nlargest(5,"specter_novelty").iterrows():
        report.append(f"  {r['specter_novelty']:.3f}  {r['name']}  "
                     f"[{r['specter_keywords'][:60]}]")

    (OUT / "model_comparison.txt").write_text("\n".join(report))
    print("\n" + "\n".join(report))
    print(f"\nDone. Primary model: {winner.upper()}")


if __name__ == "__main__":
    main()
