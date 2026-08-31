"""
Script 53: Validate Semantic Embedding Approach
================================================
V1: SPECTER2 clusters vs PitchBook categories (Cramér's V)
V2: Description length bias in expert classification
V3: Model robustness (SPECTER2 vs all-MiniLM-L6-v2)
V4: Expert-by-embedding vs PitchBook Preferred Verticals
V5: Echo chamber with PitchBook-based expert definition
"""

import warnings; warnings.filterwarnings("ignore")
import sys, pickle, re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
import statsmodels.formula.api as smf
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

OUT = Path("output/validation"); OUT.mkdir(parents=True, exist_ok=True)
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ── Load core data ────────────────────────────────────────────────────────
print("Loading data…")

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]         = cdf["Companies"].str.strip()
cdf["is_failed"]  = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

co_sector = (deals.dropna(subset=["Primary PitchBook Industry Group"])
             .groupby("co")["Primary PitchBook Industry Group"]
             .agg(lambda x: x.mode()[0]).reset_index()
             .rename(columns={"Primary PitchBook Industry Group": "sector"}))
co_verticals = (deals.dropna(subset=["Verticals"])
                .groupby("co")["Verticals"]
                .agg(lambda x: x.mode()[0]).reset_index())

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    sp = pickle.load(f)
startup_names_raw = [n.strip() for n in sp["names"]]
startup_emb_raw   = sp["embs_raw"]
startup_emb_norm  = normalize(startup_emb_raw, norm="l2")
startup_emb_dict  = {n: startup_emb_norm[i] for i, n in enumerate(startup_names_raw)}

ns = pd.read_csv("output/novelty/novelty_scores.csv")
ns["co"] = ns["name"].str.strip()

inv_emb_raw  = np.load("data/investor_embeddings.npy")
inv_names_ser = pd.read_csv("data/investor_names.csv")["investor"]
inv_names    = inv_names_ser.tolist()
inv_emb_norm = normalize(inv_emb_raw, norm="l2")
inv_emb_dict = {n: inv_emb_norm[i] for i, n in enumerate(inv_names)}

net = pd.read_csv("output/network/network_metrics.csv")
inv_to_comm = net.set_index("investor")["community"].to_dict()

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv").assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""').dropna(subset=["Year"]))
co_all_inv = d_exp.groupby("co")["inv"].apply(set).to_dict()

# ── Investor data file (Preferred Verticals etc.) ─────────────────────────
print("Loading Investors_data.xlsx…")
inv_data = pd.read_excel("data/Investors_data.xlsx", skiprows=7)
inv_data["investor"] = inv_data["Investors"].str.strip()
inv_data = inv_data.drop_duplicates(subset=["investor"]).set_index("investor")
print(f"  Investor records: {len(inv_data):,}")
desc_col = "Description"
pref_vert_col = "Preferred Verticals"
pref_ind_col  = "Preferred PitchBook Industry"
print(f"  Description non-null: {inv_data[desc_col].notna().sum():,}")
print(f"  Preferred Verticals non-null: {inv_data[pref_vert_col].notna().sum():,}")

# ─────────────────────────────────────────────────────────────────────────
# FIT K-MEANS K=15 ON STARTUP EMBEDDINGS (same as Scripts 50-52)
# ─────────────────────────────────────────────────────────────────────────
print("Fitting startup K-means K=15…")
km = KMeans(n_clusters=15, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(startup_emb_norm)
startup_name_to_tc = {n: int(km_labels[i]) for i, n in enumerate(startup_names_raw)}

tc_labels = {
    0:"Carbon/Hydrogen",   1:"Solar/Renewable",   2:"Data/Monitoring",
    3:"AgTech",            4:"EV/Mobility",        5:"Grid/Energy Mgmt",
    6:"Carbon Data",       7:"Wind Power",         8:"Battery/Storage",
    9:"Waste/Water",      10:"Alt Protein",       11:"Plastics/Materials",
    12:"Heating/HVAC",    13:"Climate Services",  14:"Plant-Based Food",
}

tc_df = pd.DataFrame({
    "co": startup_names_raw,
    "tech_cluster": km_labels,
})
tc_df["tech_cluster_lbl"] = tc_df["tech_cluster"].map(tc_labels)
tc_df = tc_df.merge(co_sector, on="co", how="left")
tc_df = tc_df.merge(co_verticals, on="co", how="left")
tc_df = tc_df.merge(ns[["co","Description","specter_novelty"]], on="co", how="left")

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION 1: SPECTER2 CLUSTERS vs PITCHBOOK CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("VALIDATION 1: SPECTER2 CLUSTERS vs PITCHBOOK INDUSTRY CATEGORIES")
print("="*72)

tc_sec = tc_df.dropna(subset=["sector","tech_cluster"])
V = cramers_v(tc_sec["tech_cluster"], tc_sec["sector"])
nmi = normalized_mutual_info_score(tc_sec["tech_cluster"], tc_sec["sector"])
print(f"\n  Cramér's V (cluster ↔ PitchBook Industry Group) = {V:.4f}")
print(f"  NMI                                               = {nmi:.4f}")
print(f"  N with both labels: {len(tc_sec):,}")
if V > 0.4:
    print("  ✅ V>0.4: Clusters capture REAL technology differences")
elif V > 0.2:
    print("  🟡 0.2<V<0.4: Moderate alignment — clusters partially match PitchBook")
else:
    print("  ❌ V<0.2: Low alignment — clusters may be noise")

print(f"\n  Modal PitchBook sector per SPECTER2 cluster:")
print(f"  {'Cluster':<25} {'N':>5}  {'Top PitchBook Sector':<30} {'%':>6}  {'2nd Sector':<25} {'%':>6}")
print(f"  {'-'*105}")
for tc in range(15):
    s = tc_sec[tc_sec["tech_cluster"]==tc]
    if len(s) < 5: continue
    vc = s["sector"].value_counts()
    top = vc.index[0] if len(vc)>0 else "—"
    top_pct = vc.iloc[0]/len(s) if len(vc)>0 else 0
    sec2 = vc.index[1] if len(vc)>1 else "—"
    pct2 = vc.iloc[1]/len(s) if len(vc)>1 else 0
    lbl = tc_labels.get(tc, f"Cluster {tc}")
    print(f"  {lbl:<25} {len(s):>5}  {top:<30} {top_pct:>5.0%}  {sec2:<25} {pct2:>5.0%}")

# 20 random examples per cluster (first 3 clusters)
print(f"\n  Sample companies (20 per cluster, first 3 clusters shown):")
for tc in range(3):
    s = tc_sec[tc_sec["tech_cluster"]==tc].sample(min(20, len(tc_sec[tc_sec["tech_cluster"]==tc])),
                                                    random_state=42)
    lbl = tc_labels.get(tc, f"Cluster {tc}")
    print(f"\n  ── Cluster {tc}: {lbl} ──")
    print(f"  {'Company':<35} {'PitchBook Sector':<28} {'Verticals'}")
    print(f"  {'-'*100}")
    for _, row in s.iterrows():
        v = str(row.get("Verticals",""))[:30] if pd.notna(row.get("Verticals")) else "—"
        print(f"  {row['co'][:34]:<35} {str(row.get('sector',''))[:27]:<28} {v}")

tc_df.to_csv(OUT / "cluster_sector_alignment.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION 2: DESCRIPTION LENGTH BIAS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("VALIDATION 2: DESCRIPTION LENGTH BIAS IN EXPERT CLASSIFICATION")
print("="*72)

# Investor description lengths
inv_desc_raw = inv_data[desc_col].reindex(inv_names)
desc_lengths  = inv_desc_raw.apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
print(f"\n  Investor description lengths (word count):")
print(f"    N with description: {(desc_lengths > 0).sum():,} of {len(desc_lengths):,}")
print(f"    Median: {desc_lengths[desc_lengths>0].median():.0f} words")
print(f"    Mean:   {desc_lengths[desc_lengths>0].mean():.0f} words")
print(f"    Min/Max: {desc_lengths[desc_lengths>0].min()}/{desc_lengths[desc_lengths>0].max()}")

# Distance from cluster centroid
inv_clusters_df = pd.read_csv("output/semantic/investor_clusters.csv")
inv_clusters_df["investor"] = inv_clusters_df["investor"].str.strip()

# For each investor: distance from their cluster centroid
print("\n  Computing centroid distances…")
inv_emb_matrix = inv_emb_norm  # already normalized
inv_km = KMeans(n_clusters=12, random_state=42, n_init=10, max_iter=300)
inv_km_labels = inv_km.fit_predict(inv_emb_matrix)
centroids = inv_km.cluster_centers_  # not normalized
centroids_norm = normalize(centroids, norm="l2")
# Cosine similarity to own centroid
own_centroid_sim = np.array([
    float(inv_emb_matrix[i] @ centroids_norm[inv_km_labels[i]])
    for i in range(len(inv_names))
])
dist_from_centroid = 1 - own_centroid_sim  # cosine distance

desc_len_arr = np.array([desc_lengths.get(n, 0) for n in inv_names])
mask_has_desc = desc_len_arr > 0

r_len_dist, p_len_dist = stats.pearsonr(
    desc_len_arr[mask_has_desc], dist_from_centroid[mask_has_desc]
)
print(f"\n  Correlation: description_length × distance_from_centroid")
print(f"    r={r_len_dist:.4f}  p={p_len_dist:.4f}  {sig(p_len_dist)}")
if abs(r_len_dist) < 0.15:
    print(f"  ✅ Weak correlation — description length does NOT drive cluster membership")
elif r_len_dist < 0:
    print(f"  ⚠  Longer descriptions cluster TIGHTER (closer to centroid)")
else:
    print(f"  ⚠  Longer descriptions more PERIPHERAL — short descriptions cluster tighter")

# Short-description check: do short descriptions cluster together?
short_mask  = (desc_len_arr > 0) & (desc_len_arr < 20)
long_mask   = desc_len_arr >= 100
zero_mask   = desc_len_arr == 0
print(f"\n  Clustering of short descriptions:")
print(f"    Investors with 0 words (missing desc): {zero_mask.sum():,}")
print(f"    Investors with <20 words (very short):  {short_mask.sum():,}")
print(f"    Investors with ≥100 words (long):       {long_mask.sum():,}")

if short_mask.sum() > 10:
    short_clusters = inv_km_labels[short_mask]
    vc = pd.Series(short_clusters).value_counts()
    hhi = (vc / vc.sum()).pow(2).sum()
    print(f"    Short-desc cluster HHI = {hhi:.3f}  (>0.5 means concentrated in few clusters)")
    top_cluster = vc.index[0]
    print(f"    Most common cluster for short descriptions: {top_cluster} ({vc.iloc[0]/short_mask.sum():.0%} of them)")

# KEY TEST: is expert classification correlated with description length?
print(f"\n  KEY TEST: does 'expert' classification correlate with description length?")
print(f"  (Expert = investor cosine match with startup > median)")

# Build investor-startup match, compute per-investor average match score
inv_match_scores = defaultdict(list)
for co, investors in co_all_inv.items():
    emb = startup_emb_dict.get(co)
    if emb is None: continue
    for inv in investors:
        if inv in inv_emb_dict:
            sim = float(inv_emb_dict[inv] @ emb)
            inv_match_scores[inv].append(sim)
inv_avg_match = {inv: np.mean(scores) for inv, scores in inv_match_scores.items() if len(scores)>=3}

# Build dataframe linking desc_length and avg_match for investors
inv_match_df = pd.DataFrame({
    "investor": list(inv_avg_match.keys()),
    "avg_match": list(inv_avg_match.values()),
})
inv_match_df["desc_length"] = inv_match_df["investor"].map(
    lambda n: desc_lengths.get(n, np.nan))
inv_match_df = inv_match_df.dropna(subset=["desc_length"]).query("desc_length > 0")

r_exp, p_exp = stats.pearsonr(inv_match_df["desc_length"], inv_match_df["avg_match"])
print(f"    N investors with match score and description: {len(inv_match_df):,}")
print(f"    r(description_length, avg_match_score) = {r_exp:.4f}  p={p_exp:.4f}  {sig(p_exp)}")

# Regression: avg_match ~ description_length
m_desc = smf.ols("avg_match ~ np.log1p(desc_length)", data=inv_match_df).fit()
b_dl = m_desc.params.get("np.log1p(desc_length)", np.nan)
p_dl = m_desc.pvalues.get("np.log1p(desc_length)", 1.0)
print(f"    LPM: avg_match ~ log(desc_length): β={b_dl:>+.6f}  {sig(p_dl)}")
if abs(r_exp) < 0.2 and p_exp > 0.05:
    print(f"  ✅ CLEAR: Expert score is NOT driven by description length")
elif abs(r_exp) < 0.3:
    print(f"  🟡 MODERATE: Weak correlation — some length bias but not dominant")
else:
    print(f"  ❌ PROBLEM: Expert classification correlated with description length")

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION 3: MODEL ROBUSTNESS — SPECTER2 vs all-MiniLM-L6-v2
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("VALIDATION 3: MODEL ROBUSTNESS — SPECTER2 vs all-MiniLM-L6-v2")
print("="*72)

N_SAMPLE = 200
rng = np.random.default_rng(42)

# Pick 200 startups with descriptions
valid_startups = [n for n in startup_names_raw
                  if n in {r["co"] for _, r in ns.iterrows() if pd.notna(r.get("Description"))}]
sample_names = [valid_startups[i] for i in rng.choice(len(valid_startups), size=min(N_SAMPLE, len(valid_startups)), replace=False)]
sample_emb_specter = np.array([startup_emb_dict[n] for n in sample_names])

# Get their descriptions
ns_idx = ns.set_index("co")
sample_descs = [str(ns_idx.loc[n, "Description"]) if n in ns_idx.index and pd.notna(ns_idx.loc[n, "Description"]) else "" for n in sample_names]
valid_mask = [len(d) > 10 for d in sample_descs]
sample_names_v   = [n for n, v in zip(sample_names, valid_mask) if v]
sample_descs_v   = [d for d, v in zip(sample_descs, valid_mask) if v]
sample_emb_sp_v  = normalize(np.array([startup_emb_dict[n] for n in sample_names_v]), norm="l2")
print(f"\n  Sample size: {len(sample_names_v)} startups with descriptions")

# Pairwise SPECTER2 cosine similarity matrix
pairwise_sp = sample_emb_sp_v @ sample_emb_sp_v.T

# Embed with all-MiniLM-L6-v2
print("  Embedding with all-MiniLM-L6-v2…")
try:
    from sentence_transformers import SentenceTransformer
    mini = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb_mini = mini.encode(sample_descs_v, batch_size=64, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=True)
    pairwise_mini = emb_mini @ emb_mini.T

    # Correlate upper triangle of the two pairwise matrices
    idx = np.triu_indices(len(sample_names_v), k=1)
    sp_flat   = pairwise_sp[idx]
    mini_flat = pairwise_mini[idx]

    r_models, p_models = stats.pearsonr(sp_flat, mini_flat)
    r_spearman, p_spearman = stats.spearmanr(sp_flat, mini_flat)

    print(f"  Pairwise similarity matrix correlation (N pairs = {len(sp_flat):,}):")
    print(f"    Pearson  r = {r_models:.4f}  p={p_models:.2e}")
    print(f"    Spearman r = {r_spearman:.4f}  p={p_spearman:.2e}")

    if r_models > 0.8:
        print(f"  ✅ r>0.8: Embeddings are ROBUST — model choice doesn't drive results")
    elif r_models > 0.5:
        print(f"  🟡 0.5<r<0.8: Moderate robustness — some sensitivity to model choice")
    else:
        print(f"  ❌ r<0.5: Results are model-dependent — SPECTER2 choice matters")

    # Also compare cluster assignments
    km_mini = KMeans(n_clusters=15, random_state=42, n_init=5)
    labels_mini = km_mini.fit_predict(emb_mini)
    km_sp   = KMeans(n_clusters=15, random_state=42, n_init=5)
    labels_sp = km_sp.fit_predict(sample_emb_sp_v)
    nmi_models = normalized_mutual_info_score(labels_sp, labels_mini)
    V_models = cramers_v(pd.Series(labels_sp), pd.Series(labels_mini))
    print(f"\n  Cluster assignment agreement (K=15):")
    print(f"    NMI(SPECTER2, MiniLM) = {nmi_models:.4f}")
    print(f"    Cramér's V            = {V_models:.4f}")
    if nmi_models > 0.5:
        print(f"  ✅ Cluster structure is consistent across models")
    elif nmi_models > 0.3:
        print(f"  🟡 Partial cluster agreement — some structure is model-dependent")
    else:
        print(f"  ❌ Clusters are model-dependent")

except ImportError:
    print("  sentence_transformers not available — skipping model comparison")
except Exception as e:
    print(f"  Model comparison error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION 4: DOES 'EXPERT' MEAN ANYTHING REAL?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("VALIDATION 4: EXPERT-BY-EMBEDDING vs PITCHBOOK PREFERRED VERTICALS")
print("="*72)

# Build investor preferred verticals dict
pref_vert_series = inv_data[pref_vert_col].dropna()
pref_ind_series  = inv_data[pref_ind_col].dropna() if pref_ind_col in inv_data.columns else pd.Series(dtype=str)
print(f"\n  Investors with Preferred Verticals: {len(pref_vert_series):,}")
print(f"  Investors with Preferred Industry:  {len(pref_ind_series):,}")

def parse_set(s):
    if not isinstance(s, str): return set()
    return {x.strip().lower() for x in re.split(r"[,;|]", s) if x.strip()}

inv_pref_verticals = {n: parse_set(v) for n, v in pref_vert_series.items()}
inv_pref_industry  = {n: parse_set(v) for n, v in pref_ind_series.items()}

# Build per-company: list of expert investors (match > median) and non-expert
# Recompute from the embedding-based match (using all-sample median)
all_match_vals = []
for co, investors in co_all_inv.items():
    emb = startup_emb_dict.get(co)
    if emb is None: continue
    for inv in investors:
        if inv in inv_emb_dict:
            all_match_vals.append(float(inv_emb_dict[inv] @ emb))
global_median = np.median(all_match_vals)
print(f"\n  Global median investor-startup match: {global_median:.4f}")

# For each (company, investor) pair: is_expert, has_pref_vert_overlap
rows = []
for co, investors in co_all_inv.items():
    emb = startup_emb_dict.get(co)
    if emb is None: continue
    co_vert = parse_set(co_verticals.set_index("co").loc[co, "Verticals"]
                        if co in co_verticals.set_index("co").index else "")
    co_sec  = co_sector.set_index("co").loc[co, "sector"] if co in co_sector.set_index("co").index else ""
    co_sec_set = {co_sec.lower().strip()} if isinstance(co_sec, str) else set()

    for inv in investors:
        if inv not in inv_emb_dict: continue
        sim = float(inv_emb_dict[inv] @ emb)
        is_expert = int(sim > global_median)
        inv_pv   = inv_pref_verticals.get(inv, set())
        inv_pi   = inv_pref_industry.get(inv, set())
        vert_overlap = len(co_vert & inv_pv) > 0 if co_vert and inv_pv else np.nan
        ind_overlap  = len(co_sec_set & inv_pi) > 0 if co_sec_set and inv_pi else np.nan
        rows.append({
            "co": co, "investor": inv, "match_sim": sim,
            "is_expert": is_expert,
            "vert_overlap": vert_overlap,
            "ind_overlap": ind_overlap,
        })

pairs_df = pd.DataFrame(rows)
print(f"\n  Total investor-company pairs: {len(pairs_df):,}")
print(f"  Pairs with Preferred Verticals data: {pairs_df['vert_overlap'].notna().sum():,}")
print(f"  Pairs with Preferred Industry data:  {pairs_df['ind_overlap'].notna().sum():,}")

# % of expert vs non-expert with vertical overlap
for col, label in [("vert_overlap","Preferred Verticals overlap"),
                   ("ind_overlap",  "Preferred Industry overlap")]:
    sub = pairs_df.dropna(subset=[col])
    if len(sub) < 100: continue
    exp_rate  = sub[sub["is_expert"]==1][col].mean()
    nexp_rate = sub[sub["is_expert"]==0][col].mean()
    odds      = exp_rate / nexp_rate if nexp_rate > 0 else np.nan
    tstat, tpval = stats.ttest_ind(
        sub[sub["is_expert"]==1][col].astype(float),
        sub[sub["is_expert"]==0][col].astype(float)
    )
    print(f"\n  [{label}]")
    print(f"    Expert investors (embed match>median):     {exp_rate:.1%} have overlap")
    print(f"    Non-expert investors:                      {nexp_rate:.1%} have overlap")
    print(f"    Odds ratio: {odds:.2f}x   t={tstat:.2f}  p={tpval:.4f}  {sig(tpval)}")
    if exp_rate > nexp_rate * 1.3 and tpval < 0.05:
        print(f"  ✅ Expert-by-embedding has SIGNIFICANTLY more vertical overlap with startups")
    elif exp_rate > nexp_rate:
        print(f"  🟡 Expert-by-embedding has MORE vertical overlap but not strongly significant")
    else:
        print(f"  ❌ Expert-by-embedding does NOT predict PitchBook vertical overlap")

# Correlation: match similarity → vert_overlap (continuous)
sub_v = pairs_df.dropna(subset=["vert_overlap"])
if len(sub_v) > 100:
    pt, pp = stats.pointbiserialr(sub_v["vert_overlap"].astype(float), sub_v["match_sim"])
    print(f"\n  Point-biserial r(vert_overlap, match_sim) = {pt:.4f}  p={pp:.4f}  {sig(pp)}")
    print(f"  {'Positive = higher cosine match → more likely to have stated vertical preference' if pt>0 else 'Negative = mismatch'}")

pairs_df.to_csv(OUT / "expert_validation_pairs.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION 5: ECHO CHAMBER WITH PITCHBOOK-BASED EXPERT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("VALIDATION 5: ECHO CHAMBER WITH PITCHBOOK-BASED EXPERT DEFINITION")
print("="*72)
print("  expert_pitchbook = 1 if ≥1 investor in syndicate has Preferred")
print("  Verticals overlapping with the startup's Verticals/Industry Group")

# Per-company: does the syndicate contain at least one PitchBook-expert investor?
pb_expert_rows = []
for co, investors in co_all_inv.items():
    co_vert = parse_set(co_verticals.set_index("co").loc[co, "Verticals"]
                        if co in co_verticals.set_index("co").index else "")
    co_sec_raw = co_sector.set_index("co").loc[co, "sector"] \
                 if co in co_sector.set_index("co").index else ""
    co_sec_set = {co_sec_raw.lower().strip()} if isinstance(co_sec_raw, str) else set()
    n_pb_expert = 0
    n_with_pref = 0
    for inv in investors:
        inv_pv = inv_pref_verticals.get(inv, set())
        inv_pi = inv_pref_industry.get(inv, set())
        if inv_pv or inv_pi:
            n_with_pref += 1
            if (co_vert & inv_pv) or (co_sec_set & inv_pi):
                n_pb_expert += 1
    pb_expert_rows.append({
        "co": co,
        "n_investors": len(investors),
        "n_with_pref_data": n_with_pref,
        "n_pb_expert": n_pb_expert,
        "pct_pb_expert": n_pb_expert / n_with_pref if n_with_pref > 0 else np.nan,
        "has_pb_expert": int(n_pb_expert > 0),
        "majority_pb_expert": int(n_pb_expert > n_with_pref / 2) if n_with_pref > 0 else np.nan,
    })

pb_df = pd.DataFrame(pb_expert_rows)
print(f"\n  Companies with ≥1 investor with preference data: {(pb_df['n_with_pref_data']>0).sum():,}")
print(f"  Companies with ≥1 PB-expert investor:           {pb_df['has_pb_expert'].sum():,}")
print(f"  Mean % PB-expert investors per company:         {pb_df['pct_pb_expert'].dropna().mean():.1%}")

# Merge with cdf for outcome data
pb_full = cdf.merge(pb_df, on="co", how="inner")
pb_full = pb_full.dropna(subset=["has_pb_expert","is_failed","n_with_pref_data"])
pb_full = pb_full[pb_full["n_with_pref_data"] > 0]

# Rebuild 2×2 with PitchBook-based expert
pb_full["pb_expert"]       = pb_full["has_pb_expert"].astype(float)
pb_full["network_diverse"] = pb_full["cross_community"].astype(float)
pb_full["type_pb"] = np.where(
    pb_full["pb_expert"]==1,
    np.where(pb_full["network_diverse"]==1, "A2_PbExp_Div", "B2_PbExp_Sam"),
    np.where(pb_full["network_diverse"]==1, "C2_NxDiv", "D2_NxSam"),
)
pb_full["type_A2"] = (pb_full["type_pb"]=="A2_PbExp_Div").astype(float)
pb_full["type_B2"] = (pb_full["type_pb"]=="B2_PbExp_Sam").astype(float)
pb_full["type_C2"] = (pb_full["type_pb"]=="C2_NxDiv").astype(float)

sub_pb = pb_full.dropna(subset=["is_failed"]).copy()
print(f"\n  Analysis sample (PB expert): N={len(sub_pb):,}")
print(f"\n  2×2 CELL SIZES AND FAILURE RATES (PitchBook-based expert):")
print(f"  {'Cell':<35} {'N':>6} {'Fail':>6} {'Fail%':>7}  Flag")
print(f"  {'-'*65}")
for t, lbl in [("A2_PbExp_Div","A2: PB-Expert+Diverse"),
               ("B2_PbExp_Sam","B2: PB-Expert+Same  ← echo?"),
               ("C2_NxDiv",    "C2: Non-expert+Diverse"),
               ("D2_NxSam",    "D2: Non-expert+Same  ← ref")]:
    s = sub_pb[sub_pb["type_pb"]==t]
    n = len(s); fn = s["is_failed"].sum(); fp = s["is_failed"].mean() if n>0 else np.nan
    flag = " ⚠ FRAGILE" if n < 50 else ""
    print(f"  {lbl:<35} {n:>6} {fn:>6.0f} {fp:>7.1%}{flag}")

# Echo penalty
fa2 = sub_pb[sub_pb["type_pb"]=="A2_PbExp_Div"]["is_failed"].mean()
fb2 = sub_pb[sub_pb["type_pb"]=="B2_PbExp_Sam"]["is_failed"].mean()
fc2 = sub_pb[sub_pb["type_pb"]=="C2_NxDiv"]["is_failed"].mean()
fd2 = sub_pb[sub_pb["type_pb"]=="D2_NxSam"]["is_failed"].mean()
echo_pb = fb2 - fa2
print(f"\n  Echo penalty (B2−A2): {echo_pb:>+.1%}")
print(f"  Ordering: A2={fa2:.1%}  B2={fb2:.1%}  C2={fc2:.1%}  D2={fd2:.1%}")

# Regression
try:
    m_pb = smf.ols(f"is_failed ~ type_A2 + type_B2 + type_C2 + {CTRL}",
                   data=sub_pb).fit(cov_type="HC1")
    print(f"\n  Regression (N={int(m_pb.nobs):,}  R²={m_pb.rsquared:.4f}):")
    for var, lbl in [("type_A2","A2: PB-Expert+Diverse"),
                     ("type_B2","B2: PB-Expert+Same"),
                     ("type_C2","C2: Non-expert+Diverse")]:
        if var in m_pb.params:
            b = m_pb.params[var]; p = m_pb.pvalues[var]
            print(f"    {lbl:<30}: β={b:>+.4f}  {sig(p)}")

    # Wald: β_B2 ≠ β_A2
    w = m_pb.f_test("type_B2 = type_A2")
    f_w, p_w = float(w.statistic), float(w.pvalue)
    delta_pb = m_pb.params.get("type_B2",0) - m_pb.params.get("type_A2",0)
    print(f"\n  Wald (B2=A2): F={f_w:.3f}  p={p_w:.4f}  {sig(p_w)}  Δ={delta_pb:>+.4f}")

    if p_w < 0.05:
        print(f"  ✅ ECHO CHAMBER CONFIRMED with PitchBook expert definition")
    elif p_w < 0.10:
        print(f"  † MARGINAL echo chamber with PitchBook expert definition")
    else:
        print(f"  ❌ Echo chamber NOT confirmed with PitchBook expert definition")

except Exception as e:
    print(f"  Regression error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("VALIDATION SUMMARY")
print("="*72)
print(f"""
  V1 CLUSTER VALIDITY
     Cramér's V (SPECTER2 cluster vs PitchBook Industry) = {V:.3f}
     NMI = {nmi:.3f}
     {'✅ Clusters capture real technology differentiation' if V>0.4 else '🟡 Moderate alignment' if V>0.2 else '❌ Low alignment'}

  V2 DESCRIPTION LENGTH BIAS
     r(desc_length, centroid_distance) = {r_len_dist:.3f}  {sig(p_len_dist)}
     r(desc_length, expert_score)      = {r_exp:.3f}  {sig(p_exp)}
     {'✅ No significant bias' if abs(r_exp)<0.2 and p_exp>0.05 else '❌ Bias detected: expert score correlates with description length'}

  V3 MODEL ROBUSTNESS (SPECTER2 vs MiniLM)
     {'r = '+f'{r_models:.3f}' if 'r_models' in dir() else 'Not computed'}
     {'✅ Robust' if 'r_models' in dir() and r_models>0.8 else '🟡 Moderate' if 'r_models' in dir() and r_models>0.5 else '❌ Fragile' if 'r_models' in dir() else '—'}

  V4 EXPERT VALIDITY (vs PitchBook Preferred Verticals)
     {'See rates above — expert investors are ' + ('meaningfully more aligned' if 'exp_rate' in dir() and exp_rate>nexp_rate*1.3 else 'somewhat more aligned')}

  V5 ECHO CHAMBER (PitchBook-based expert)
     Echo penalty B2−A2 = {echo_pb:>+.1%}
     Wald p = {p_w:.4f}  {sig(p_w)}
     {'✅ Echo chamber replicated with PitchBook expert definition' if p_w<0.05 else '❌ Not replicated with PitchBook expert definition'}

  OUTPUT FILES:
    {OUT}/cluster_sector_alignment.csv
    {OUT}/expert_validation_pairs.csv
""")
