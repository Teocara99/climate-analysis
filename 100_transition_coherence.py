"""
Script 100: Transition-coherence analysis. Descriptive only.

0. Noise audit: 50 samples, classification, novelty comparison
1. TRL coverage by year (first rounds + dollars)
2. T1 — Readiness trend: TRL bands × year, capital-weighted TRL with CIs
3. T2 — IPCC WG3 mapping (MAPPING ONLY — stop before computing)
4. Descriptive context: total deals and dollars by year
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress

warnings.filterwarnings("ignore")

OUT_DIR = "output/transition_coherence"
os.makedirs(OUT_DIR, exist_ok=True)

def sep(title=""):
    print(f"\n{'─'*68}")
    if title:
        print(f"  {title}")
        print(f"{'─'*68}")

print("=" * 68)
print("SCRIPT 100 — TRANSITION COHERENCE ANALYSIS")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────
# Load shared data
# ─────────────────────────────────────────────────────────────────────
sep("Loading data")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl["year"]      = dl["Deal Date"].dt.year
dl["co"]        = dl["Companies"].str.strip()
dl["deal_size"] = pd.to_numeric(dl["Deal Size (USD M)"], errors="coerce")
deals_v = dl.dropna(subset=["year"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

nov = pd.read_csv("output/novelty/novelty_scores.csv")
trl_map = pd.read_csv("output/h6_trl/cluster_trl_mapping.csv")

co_cluster  = dict(zip(nov["name"], nov["specter_cluster"].astype(int)))
cl_trl_dict = dict(zip(trl_map["cluster"].astype(int),
                        trl_map["trl"]))    # NaN where unassigned
cl_conf_dict = dict(zip(trl_map["cluster"].astype(int),
                         trl_map["confidence"]))

# First rounds
fr = (deals_v.sort_values("Deal Date")
      .groupby("co").first()
      .reset_index()
      .rename(columns={"Deal Date":"first_date","year":"first_year"}))
fr["cluster"] = fr["co"].map(co_cluster)
fr["trl_raw"] = fr["cluster"].map(cl_trl_dict)      # float or NaN
fr["trl"]     = pd.to_numeric(fr["trl_raw"], errors="coerce")
fr["trl_known"] = fr["trl"].notna()

print(f"  First-round companies: {len(fr):,}")
print(f"  With TRL:             {fr['trl_known'].sum():,}  ({100*fr['trl_known'].mean():.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 0. Noise audit
# ─────────────────────────────────────────────────────────────────────
sep("0. Noise audit (specter_cluster == -1)")

noise = nov[nov["specter_cluster"] == -1].copy()
print(f"  Total noise-cluster companies: {len(noise):,}")
print(f"  Mean specter_novelty — noise:    {noise['specter_novelty'].mean():.4f}  "
      f"(SD={noise['specter_novelty'].std():.4f})")
print(f"  Mean specter_novelty — clustered:{nov[nov['specter_cluster']>=0]['specter_novelty'].mean():.4f}  "
      f"(SD={nov[nov['specter_cluster']>=0]['specter_novelty'].std():.4f})")

# Load SPECTER2 embeddings and compute cluster centroids
with open("output/novelty/novelty_specter.pkl","rb") as f:
    spk = pickle.load(f)
all_embs  = np.array(spk["embs_raw"], dtype=np.float32)
all_names = spk["names"]
name_to_idx = {n:i for i,n in enumerate(all_names)}

# Normalise for cosine similarity
norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
norms[norms == 0] = 1e-8
all_embs_norm = all_embs / norms

# Cluster centroids (excluding noise)
clustered_ids = nov[nov["specter_cluster"]>=0]["specter_cluster"].values
unique_cls = sorted(set(clustered_ids))
centroid_embs = np.zeros((len(unique_cls), all_embs_norm.shape[1]), dtype=np.float32)
for i, cl in enumerate(unique_cls):
    names_cl = nov[nov["specter_cluster"]==cl]["name"].tolist()
    idxs_cl  = [name_to_idx[n] for n in names_cl if n in name_to_idx]
    if idxs_cl:
        centroid_embs[i] = all_embs_norm[idxs_cl].mean(axis=0)
        centroid_embs[i] /= (np.linalg.norm(centroid_embs[i]) + 1e-8)

# For noise companies, find nearest cluster centroid
noise_sample = noise.sample(50, random_state=42).copy()
noise_results = []
for _, row in noise_sample.iterrows():
    nm  = row["name"]
    idx = name_to_idx.get(nm)
    if idx is None:
        noise_results.append({"name":nm,"nearest_cl":None,"sim":None,
                               "trl":None,"conf":None,"desc":""})
        continue
    emb = all_embs_norm[idx]
    sims = centroid_embs @ emb
    best_i  = int(np.argmax(sims))
    best_cl = unique_cls[best_i]
    best_sim= float(sims[best_i])
    trl_v   = cl_trl_dict.get(best_cl, np.nan)
    conf_v  = cl_conf_dict.get(best_cl, "-")
    noise_results.append({
        "name":      nm,
        "nearest_cl": best_cl,
        "sim":        best_sim,
        "trl":        trl_v,
        "conf":       conf_v,
        "desc":       str(row.get("Description",""))[:120],
    })

noise_df = pd.DataFrame(noise_results)

# Within-cluster similarity baseline (75th pctile of within-cluster cosine sim)
# Use a fast approximation: mean pairwise sim for a random subset of each cluster
within_sims = []
rng = np.random.default_rng(42)
for i, cl in enumerate(unique_cls[:20]):  # sample first 20 clusters
    names_cl = nov[nov["specter_cluster"]==cl]["name"].tolist()
    idxs_cl  = [name_to_idx[n] for n in names_cl if n in name_to_idx]
    if len(idxs_cl) < 2: continue
    samp = rng.choice(idxs_cl, size=min(20, len(idxs_cl)), replace=False)
    subembs = all_embs_norm[samp]
    pairwise = subembs @ subembs.T
    upper = pairwise[np.triu_indices(len(samp),k=1)]
    within_sims.extend(upper.tolist())
threshold = np.percentile(within_sims, 25) if within_sims else 0.85
print(f"\n  Within-cluster cosine similarity (25th pctile, used as 'assignable' threshold): {threshold:.4f}")

noise_df["assignable"] = noise_df["sim"].apply(
    lambda s: "YES" if (s is not None and s > threshold) else "NO")

print(f"\n  Sample of 50 noise companies — nearest cluster and similarity:")
print(f"  {'Name':<40}  {'Cl':>4}  {'Sim':>6}  {'TRL':>5}  {'Assign'}  Description")
print(f"  {'─'*40}  {'─'*4}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*50}")
for _, row in noise_df.sort_values("sim", ascending=False).iterrows():
    cl_s  = f"{int(row['nearest_cl'])}" if pd.notna(row['nearest_cl']) else "—"
    sim_s = f"{row['sim']:.4f}" if pd.notna(row['sim']) else "—"
    trl_s = f"{int(row['trl'])}" if pd.notna(row['trl']) else "  —"
    print(f"  {str(row['name']):<40}  {cl_s:>4}  {sim_s:>6}  {trl_s:>5}  "
          f"{row['assignable']:<6}  {row['desc'][:55]}")

# Summary
n_assign = (noise_df["assignable"]=="YES").sum()
n_noise_w_trl = noise_df[noise_df["assignable"]=="YES"]["trl"].notna().sum()
print(f"\n  Summary (N=50 sample):")
print(f"    Assignable (sim ≥ {threshold:.3f}): {n_assign}/50  ({100*n_assign/50:.0f}%)")
print(f"    Of those, with TRL in matched cluster: {n_noise_w_trl}/{n_assign}")
print(f"    Not assignable (genuinely distinct):   {50-n_assign}/50")

# Nearest-cluster distribution for full noise set
print(f"\n  Full noise set ({len(noise):,}) — nearest cluster distribution (top 15):")
full_noise_names = noise["name"].tolist()
noise_nearest = []
for nm in full_noise_names:
    idx = name_to_idx.get(nm)
    if idx is None: continue
    emb = all_embs_norm[idx]
    sims = centroid_embs @ emb
    best_cl = unique_cls[int(np.argmax(sims))]
    noise_nearest.append(best_cl)
nc_dist = pd.Series(noise_nearest).value_counts().head(15)
kw_by_cl_map = dict(zip(trl_map["cluster"].astype(int),
                         trl_map["top_keywords"].str[:45]))
for cl, cnt in nc_dist.items():
    trl_v = cl_trl_dict.get(int(cl), "—")
    kw_v  = kw_by_cl_map.get(int(cl), "?")
    print(f"    Cluster {int(cl):>3}  n={cnt:>4}  TRL={str(trl_v):<4}  {kw_v}")

# Save
noise_df.to_csv(f"{OUT_DIR}/noise_audit_sample50.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 1. TRL coverage by year
# ─────────────────────────────────────────────────────────────────────
sep("1. TRL coverage by year (first rounds, 2016-2025)")

YEAR_MIN, YEAR_MAX = 2016, 2025
fr_y = fr[fr["first_year"].between(YEAR_MIN, YEAR_MAX)].copy()

# TRL band
def trl_band(t):
    if pd.isna(t): return "Unknown"
    t = int(t)
    if t <= 6:  return "4-6 Emerging"
    if t <= 8:  return "7-8 Demonstration"
    return "9-11 Mature"

fr_y["trl_band"] = fr_y["trl"].apply(trl_band)

# Deal share
cov_n = (fr_y.groupby(["first_year","trl_band"])
         .size().unstack(fill_value=0))
BANDS_ORDER = ["4-6 Emerging","7-8 Demonstration","9-11 Mature","Unknown"]
cov_n = cov_n.reindex(columns=[b for b in BANDS_ORDER if b in cov_n.columns], fill_value=0)
cov_n["Total"] = cov_n.sum(axis=1)
cov_pct = cov_n.drop(columns="Total").div(cov_n["Total"], axis=0) * 100

print(f"\n  Deal count share (%) by TRL band × year:")
out = cov_pct.copy(); out["N"] = cov_n["Total"].astype(int)
print(out.round(1).to_string())

# Dollar share (all-rounds level for deal size)
# Join all deals with TRL for dollar coverage
deals_trl = deals_v[deals_v["year"].between(YEAR_MIN,YEAR_MAX)].copy()
deals_trl["cluster"] = deals_trl["co"].map(co_cluster)
deals_trl["trl"]     = pd.to_numeric(deals_trl["cluster"].map(cl_trl_dict), errors="coerce")
deals_trl["trl_band"]= deals_trl["trl"].apply(trl_band)

cov_d = (deals_trl.groupby(["year","trl_band"])["deal_size"]
         .sum().unstack(fill_value=0))
cov_d = cov_d.reindex(columns=[b for b in BANDS_ORDER if b in cov_d.columns], fill_value=0)
cov_d["Total_M"] = cov_d.sum(axis=1)
cov_dpct = cov_d.drop(columns="Total_M").div(cov_d["Total_M"].replace(0,np.nan), axis=0) * 100

print(f"\n  Dollar share (%) by TRL band × year (all rounds):")
out2 = cov_dpct.copy(); out2["USD_M"] = cov_d["Total_M"].round(0).astype(int)
print(out2.fillna(0).round(1).to_string())

# ─────────────────────────────────────────────────────────────────────
# 2. T1 — Readiness trend
# ─────────────────────────────────────────────────────────────────────
sep("2. T1 — Readiness trend")

# Capital-weighted mean TRL by year (first rounds with known TRL)
fr_known = fr_y[fr_y["trl"].notna()].copy()
fr_known["deal_size"] = fr_known["co"].map(
    deals_v.groupby("co")["deal_size"].max())

def cap_wt_trl(df):
    """Return capital-weighted mean TRL. Falls back to unweighted if no deal size."""
    if len(df) == 0: return np.nan, np.nan, np.nan
    w = pd.to_numeric(df["deal_size"], errors="coerce").fillna(0)
    if w.sum() == 0:
        w = np.ones(len(df))
    w = w.values / w.sum()
    t = df["trl"].values
    mean_t = float(np.dot(w, t))
    # Bootstrap CI
    rng2 = np.random.default_rng(0)
    boots = [np.dot(rng2.choice(w, size=len(w), replace=True), t) for _ in range(999)]
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    return mean_t, ci_lo, ci_hi

rows_trl = []
for yr in range(YEAR_MIN, YEAR_MAX+1):
    sub = fr_known[fr_known["first_year"]==yr]
    if len(sub) < 3: continue
    m, lo, hi = cap_wt_trl(sub)
    rows_trl.append({"year":yr, "n":len(sub), "cwt_trl":m,
                      "ci_lo":lo, "ci_hi":hi,
                      "mean_trl":sub["trl"].mean(), "sd_trl":sub["trl"].std()})
cwt_df = pd.DataFrame(rows_trl)

print(f"\n  Capital-weighted TRL by year (first rounds, TRL-known only):")
print(f"  {'Year'}  {'N':>4}  {'CW-TRL':>8}  {'CI lo':>7}  {'CI hi':>7}  {'Unweighted':>11}")
print(f"  {'─'*4}  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*11}")
for _, r in cwt_df.iterrows():
    print(f"  {int(r.year)}  {int(r.n):>4}  {r.cwt_trl:>8.3f}  "
          f"{r.ci_lo:>7.3f}  {r.ci_hi:>7.3f}  {r.mean_trl:>11.3f}")

# Linear trend
sl, ic, rv, pv, _ = linregress(cwt_df["year"].values, cwt_df["cwt_trl"].values)
print(f"\n  CW-TRL trend: {sl:+.4f}/yr  r={rv:.3f}  p={pv:.4f}")
sl2, _, rv2, pv2, _ = linregress(cwt_df["year"].values, cwt_df["mean_trl"].values)
print(f"  Unweighted trend: {sl2:+.4f}/yr  r={rv2:.3f}  p={pv2:.4f}")

# Plot: stacked bar shares + line for CW-TRL
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)

BAND_CLRS = {"4-6 Emerging":"#D55E00","7-8 Demonstration":"#F0E442",
             "9-11 Mature":"#0072B2","Unknown":"#CCCCCC"}
yrs_plot = cov_pct.index.tolist()
bot = np.zeros(len(yrs_plot))
for band in BANDS_ORDER:
    if band in cov_pct.columns:
        vals = cov_pct[band].values
        ax1.bar(yrs_plot, vals, bottom=bot, color=BAND_CLRS[band],
                label=band, width=0.65, alpha=0.9)
        bot += vals
ax1.set_ylim(0, 100)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
ax1.set_title("Deal share by TRL band — first rounds", fontsize=10)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

ax2.plot(cwt_df["year"], cwt_df["cwt_trl"], color="#0072B2",
         linewidth=2, marker="o", ms=5, label="Cap-wtd TRL")
ax2.fill_between(cwt_df["year"], cwt_df["ci_lo"], cwt_df["ci_hi"],
                 alpha=0.2, color="#0072B2")
ax2.plot(cwt_df["year"], cwt_df["mean_trl"], color="grey",
         linewidth=1.3, linestyle="--", marker="s", ms=3, label="Unweighted TRL")
ax2.axhline(6, color="#D55E00", linewidth=0.8, linestyle=":", alpha=0.6)
ax2.axhline(8, color="#F0E442", linewidth=0.8, linestyle=":", alpha=0.6)
ax2.set_ylabel("Mean TRL"); ax2.set_xlabel("Year")
ax2.set_title("Capital-weighted mean TRL (with 95% bootstrap CI)", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_trl_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_trl_trend.png")

# Regional split
sep("  T1 — Regional split")
REGIONS = ["Americas","Europe","Asia"]
def region_bucket(r):
    r = str(r)
    if "Americas" in r:  return "Americas"
    if "Europe" in r:    return "Europe"
    if "Asia" in r:      return "Asia"
    return "Other"

fr_known["region"] = fr_known["HQ Global Region"].apply(region_bucket)
print(f"\n  Capital-weighted mean TRL by region (full 2016-2025 average):")
print(f"  {'Region':<12}  {'N':>5}  {'CW-TRL':>8}  {'Unweighted':>11}  {'SD':>6}")
for reg in REGIONS:
    sub = fr_known[fr_known["region"]==reg]
    if len(sub) < 5: continue
    m, lo, hi = cap_wt_trl(sub)
    print(f"  {reg:<12}  {len(sub):>5}  {m:>8.3f}  {sub['trl'].mean():>11.3f}  {sub['trl'].std():>6.3f}")

# By region × year (unweighted mean for legibility)
print(f"\n  Unweighted mean TRL by region × year:")
reg_yr = (fr_known[fr_known["region"].isin(REGIONS)]
          .groupby(["first_year","region"])["trl"]
          .mean().unstack()
          .reindex(columns=REGIONS))
print(reg_yr.round(2).to_string())

# ─────────────────────────────────────────────────────────────────────
# 3. T2 — IPCC WG3 mapping (PROPOSAL ONLY — no computation yet)
# ─────────────────────────────────────────────────────────────────────
sep("3. T2 — IPCC WG3 mitigation mapping (PROPOSAL FOR REVIEW)")

# IPCC AR6 WG3 mitigation potential values
# Source: IPCC AR6 WG3 (2022), Chapters 6/9/10/11; SPM Fig SPM.7
# Values: approximate median, <$100/tCO2, 2030 horizon, in GtCO2e/yr
# Confidence: H=well-established in literature, M=moderate, L=high uncertainty

IPCC_MAP = [
    # cl   IEA technology                      IPCC option                   IPCC ch  GtCO2 conf  note
    ( 40, "Solar photovoltaics",               "Solar PV (utility+distributed)",   6,  4.0, "H", "Ch6 Fig6.19"),
    ( 17, "Offshore wind",                     "Wind energy (onshore+offshore)",   6,  2.8, "H", "Ch6 Fig6.19"),
    (  0, "Advanced nuclear (SMR+fusion)",     "Nuclear power",                    6,  0.5, "H", "Ch6 Fig6.19"),
    (  2, "Green hydrogen via electrolysis",   "Clean hydrogen (energy/industry)", 6,  0.8, "M", "Ch6+Ch11; cross-sector"),
    ( 37, "Biomass to energy / Biochar",       "Bioenergy (BECCS incl.)",          6,  0.7, "M", "Ch6; BECCS uncertain"),
    (  5, "Enhanced geothermal systems",       "Geothermal power",                 6,  0.2, "M", "Ch6; smaller potential"),
    ( 15, "Ocean wave and tidal energy",       "Ocean energy",                     6,  0.05,"L", "Ch6; pre-commercial"),
    ( 13, "Carbon capture (CCUS / DAC)",       "CO2 capture and storage (CCS)",    6,  1.0, "M", "Ch6; range 0.7-2.0"),
    ( 55, "Li-ion and next-gen batteries",     "Energy storage (enabling)",        6,  0.5, "M", "Ch6; indirect via RE"),
    ( 54, "Thermal energy storage",            "Energy storage (enabling)",        6,  0.2, "M", "Ch6; minor direct"),
    ( 56, "Grid-scale long-duration storage",  "Energy storage (enabling)",        6,  0.5, "M", "Ch6; same as above"),
    ( 57, "Battery storage (grid-connected)",  "Energy storage (enabling)",        6,  0.5, "M", "Ch6; combined"),
    ( 59, "Smart grid / distributed energy",   "Demand-side flexibility / DSM",    6,  0.4, "M", "Ch6; grid flexibility"),
    (  4, "Battery electric vehicles",         "Road transport electrification",  10,  2.0, "H", "Ch10; LDV dominant"),
    (  3, "EV charging infrastructure",        "Road transport electrification",  10,  2.0, "H", "Ch10; enabling infra"),
    ( 16, "Electric aviation (eVTOL)",         "Aviation decarbonisation",        10,  0.2, "L", "Ch10; early-stage"),
    ( 21, "Zero-emission shipping (H2/NH3)",   "Shipping decarbonisation",        10,  0.4, "M", "Ch10"),
    ( 60, "Heat pumps (space heating)",        "Building electrification (HP)",    9,  0.8, "H", "Ch9"),
    ( 62, "Efficient cooling and HVAC",        "Building cooling efficiency",      9,  0.5, "H", "Ch9; HFC+energy"),
    ( 31, "Methane abatement and monitoring",  "Methane abatement (O&G+agri)",     6,  1.5, "H", "Ch6+Ch12; high urgency"),
    ( 42, "Li-ion battery recycling",          "Industrial circular economy",     11,  0.2, "L", "Ch11; enabling"),
    ( 43, "Low-carbon cement and concrete",    "Low-carbon cement production",    11,  0.5, "H", "Ch11"),
    ( 45, "Bio-based materials (lignin/fibers)","Bio-based materials (industry)", 11,  0.3, "M", "Ch11"),
    ( 46, "Chemical plastics recycling",       "Plastic waste management",        11,  0.2, "M", "Ch11"),
    ( 47, "Bio-based/compostable packaging",   "Material substitution (packaging)",11, 0.1, "L", "Ch11"),
    ( 49, "Advanced waste sorting/recycling",  "Waste management + recycling",    11,  0.3, "M", "Ch11+Ch12"),
    ( 51, "Low-carbon construction materials", "Low-carbon construction",         11,  0.3, "M", "Ch11"),
]

print(f"""
  PROPOSAL: cluster → IPCC WG3 mitigation option mapping
  Source: IPCC AR6 WG3 (2022). GtCO2e/yr = approximate median at <$100/tCO2, 2030.
  STOP — awaiting user review before computing capital-allocation vs mitigation scatter.

  {'Cl':>4}  {'IEA technology':<38}  {'IPCC option':<35}  {'Ch':>3}  {'GtCO2':>7}  {'Conf'}  Notes
  {'─'*4}  {'─'*38}  {'─'*35}  {'─'*3}  {'─'*7}  {'─'*4}  {'─'*30}""")

for row in IPCC_MAP:
    cl, iea, ipcc_opt, ch, gt, conf, note = row
    print(f"  {cl:>4}  {iea:<38}  {ipcc_opt:<35}  {ch:>3}  {gt:>7.2f}  {conf:<4}  {note}")

# Group by IPCC option (some clusters map to same option)
ipcc_df = pd.DataFrame(IPCC_MAP,
    columns=["cluster","iea","ipcc_option","ipcc_ch","gt_co2","conf","note"])
print(f"\n  Aggregated IPCC options (combining clusters sharing the same option):")
agg = (ipcc_df.groupby("ipcc_option")
       .agg(n_clusters=("cluster","count"),
            clusters=("cluster",lambda x: list(x)),
            gt_co2=("gt_co2","first"),
            conf=("conf","first"),
            ipcc_ch=("ipcc_ch","first"))
       .sort_values("gt_co2", ascending=False))
print(f"\n  {'IPCC option':<35}  {'Cl-count':>8}  {'GtCO2':>7}  {'Ch':>4}  Clusters")
print(f"  {'─'*35}  {'─'*8}  {'─'*7}  {'─'*4}  {'─'*20}")
for opt, row in agg.iterrows():
    print(f"  {opt:<35}  {int(row.n_clusters):>8}  {row.gt_co2:>7.2f}  "
          f"{row.ipcc_ch:>4}  {row.clusters}")

ipcc_df.to_csv(f"{OUT_DIR}/ipcc_mapping_proposal.csv", index=False)
print(f"\n  Saved ipcc_mapping_proposal.csv")
print(f"\n  *** STOP: T2 computation awaits review of the mapping above. ***")

# ─────────────────────────────────────────────────────────────────────
# 4. Descriptive context: total deals and dollars by year
# ─────────────────────────────────────────────────────────────────────
sep("4. Descriptive context: total deals and dollars by year")

# All rounds
ctx_all = (deals_v[deals_v["year"].between(YEAR_MIN-2, YEAR_MAX)]
           .groupby("year")
           .agg(n_deals=("co","count"),
                usd_m=("deal_size","sum"),
                n_companies=("co","nunique"))
           .reset_index())

# First rounds
ctx_fr = (fr[fr["first_year"].between(YEAR_MIN-2, YEAR_MAX)]
          .groupby("first_year")
          .agg(n_first=("co","count"),
               usd_fr=("deal_size", lambda x: pd.to_numeric(x,errors="coerce").sum()))
          .reset_index()
          .rename(columns={"first_year":"year"}))

ctx = ctx_all.merge(ctx_fr, on="year", how="left")

print(f"\n  {'Year'}  {'All deals':>9}  {'First rnd':>9}  {'USD_M (all)':>11}  "
      f"{'USD_M (1st)':>11}  {'Companies':>9}")
print(f"  {'─'*4}  {'─'*9}  {'─'*9}  {'─'*11}  {'─'*11}  {'─'*9}")
for _, r in ctx.iterrows():
    print(f"  {int(r.year)}  {int(r.n_deals):>9,}  {int(r.n_first) if pd.notna(r.n_first) else '':>9}  "
          f"{r.usd_m:>11,.0f}  "
          f"{r.usd_fr:>11,.0f}  "
          f"{int(r.n_companies):>9,}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,7), sharex=True)
ax1.bar(ctx["year"], ctx["n_deals"], color="#0072B2", alpha=0.7, label="All rounds")
ax1.bar(ctx["year"], ctx["n_first"].fillna(0), color="#D55E00", alpha=0.9, label="First rounds")
ax1.set_ylabel("N deals"); ax1.set_title("Deal volume by year")
ax1.legend(fontsize=8); ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

ax2.bar(ctx["year"], ctx["usd_m"].fillna(0)/1000, color="#0072B2", alpha=0.7, label="All rounds")
ax2.bar(ctx["year"], ctx["usd_fr"].fillna(0)/1000, color="#D55E00", alpha=0.9, label="First rounds")
ax2.set_ylabel("Total USD (billion)"); ax2.set_xlabel("Year")
ax2.set_title("Capital deployed by year (USD billion)")
ax2.legend(fontsize=8); ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_volume_context.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved fig_volume_context.png")

# Save tables
ctx.to_csv(f"{OUT_DIR}/volume_context.csv", index=False)
cov_pct.round(2).to_csv(f"{OUT_DIR}/trl_deal_shares.csv")
cov_dpct.fillna(0).round(2).to_csv(f"{OUT_DIR}/trl_dollar_shares.csv")
cwt_df.to_csv(f"{OUT_DIR}/cwt_trl_by_year.csv", index=False)
reg_yr.round(3).to_csv(f"{OUT_DIR}/trl_by_region_year.csv")

print()
print("=" * 68)
print("SCRIPT 100 COMPLETE")
print("=" * 68)
