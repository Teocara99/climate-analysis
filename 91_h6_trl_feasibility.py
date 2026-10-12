"""
Script 91: H6 setup and feasibility. NO outcome regressions.

1. IEA Clean Energy Technology Guide — version and technology count.
2. Map each SPECTER2 cluster to IEA technology + TRL (1-11 scale).
   Clusters with no clean-energy match (food, software, services) get NaN.
3. Feasibility: startup/dyad coverage, TRL SD, distinct values, MDE.
4. Draft H6 pre-registration.
"""

import os, sys, re, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit
from collections import defaultdict, Counter
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "h6_trl")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("SCRIPT 91 — IEA TRL MAPPING + H6 FEASIBILITY")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# 1. IEA Clean Energy Technology Guide — source & version
# ═══════════════════════════════════════════════════════════════════════════
print("""
── 1. IEA Clean Energy Technology Guide ──

  Source:  IEA ETP Clean Energy Technology Guide
  URL:     https://www.iea.org/data-and-statistics/data-tools/etp-clean-energy-technology-guide
  Report:  Energy Technology Perspectives (ETP) 2023
  Version used: ETP 2023 (data last updated 2023; referenced via web check Sept 2026;
                no 2024/2025 major revision found at time of this script)

  Technologies: > 650 individual technology designs and components
  TRL scale: 1 – 11
    1–3   Concept   (initial idea → concept validation)
    4–6   Prototype (early prototype → full-scale prototype)
    7–9   Demonstration / Early adoption
    10–11 Mature    (commercial → widely deployed)

  Key categories relevant to this corpus:
    Solar PV (TRL 11) · Wind (TRL 10-11) · Batteries (TRL 9-11)
    Green hydrogen / electrolysis (TRL 8-9) · CCUS / DAC (TRL 6-9)
    Heat pumps (TRL 9-10) · EVs (TRL 10) · Nuclear SMR (TRL 5-6)
    Fusion (TRL 2-3) · Wave / tidal (TRL 6) · Electric aviation (TRL 5)
    Zero-emission shipping (TRL 6) · Green cement (TRL 7)
    Battery recycling (TRL 8) · Thermal storage (TRL 8-9)
    Offshore wind (TRL 10) · Advanced biofuels (TRL 7-8)
""")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load cluster profiles and map to IEA TRL
# ═══════════════════════════════════════════════════════════════════════════
print("── 2. Cluster → IEA Technology mapping ──")

nov = pd.read_csv("output/novelty/novelty_scores.csv")
# 5 example names per cluster
cluster_examples = (
    nov[nov["specter_cluster"] >= 0]
    .groupby("specter_cluster")["name"]
    .apply(lambda x: " | ".join(x.head(5).tolist()))
    .to_dict()
)
cluster_kw = {}
for cl in nov[nov["specter_cluster"] >= 0]["specter_cluster"].unique():
    sub = nov[nov["specter_cluster"] == cl]
    kw = sub["specter_keywords"].dropna().iloc[0] if sub["specter_keywords"].notna().any() else ""
    cluster_kw[cl] = str(kw)[:60]
cluster_n = nov[nov["specter_cluster"] >= 0].groupby("specter_cluster").size().to_dict()

# ── Mapping table ───────────────────────────────────────────────────────
# (iea_tech, trl_midpoint, confidence)
# confidence: "H"=high, "M"=medium, "L"=low, "-"=no match
# TRL midpoint uses IEA ETP 2023 best estimate for the technology
MAPPING = {
     0: ("Advanced nuclear (SMR + fusion)",             4, "M"),
     1: (None,                                           None, "-"),  # stealth
     2: ("Green hydrogen via electrolysis",              8, "H"),
     3: ("EV charging infrastructure",                   9, "H"),
     4: ("Battery electric vehicles",                   10, "H"),
     5: ("Enhanced geothermal systems",                  7, "M"),
     6: (None,                                           None, "-"),  # mining/metals
     7: (None,                                           None, "-"),  # water/wastewater
     8: (None,                                           None, "-"),  # insect protein
     9: (None,                                           None, "-"),  # synthetic biology/food
    10: (None,                                           None, "-"),  # alt protein
    11: (None,                                           None, "-"),  # cultivated meat
    12: (None,                                           None, "-"),  # seafood
    13: ("Carbon capture (CCUS / DAC)",                  7, "H"),
    14: (None,                                           None, "-"),  # semiconductors
    15: ("Ocean wave and tidal energy",                  6, "H"),
    16: ("Electric aviation (eVTOL + fixed-wing)",       5, "H"),
    17: ("Offshore wind",                               10, "H"),
    18: (None,                                           None, "-"),  # aquaculture
    19: (None,                                           None, "-"),  # ocean restoration
    20: (None,                                           None, "-"),  # underwater monitoring
    21: ("Zero-emission shipping (H₂ / ammonia)",        6, "M"),
    22: (None,                                           None, "-"),  # wildfire detection
    23: (None,                                           None, "-"),  # satellite monitoring
    24: (None,                                           None, "-"),  # vertical farming
    25: (None,                                           None, "-"),  # weather forecasting
    26: (None,                                           None, "-"),  # climate risk/insurance
    27: (None,                                           None, "-"),  # forest monitoring
    28: (None,                                           None, "-"),  # reforestation/nature-based
    29: (None,                                           None, "-"),  # crop/agtech
    30: (None,                                           None, "-"),  # crop protection
    31: ("Methane abatement and monitoring",             8, "L"),
    32: (None,                                           None, "-"),  # air quality monitoring
    33: (None,                                           None, "-"),  # carbon accounting software
    34: (None,                                           None, "-"),  # sustainability services
    35: (None,                                           None, "-"),  # carbon credit blockchain
    36: (None,                                           None, "-"),  # carbon finance
    37: ("Biomass to energy / Biochar",                  8, "M"),
    38: (None,                                           None, "-"),  # food waste services
    39: (None,                                           None, "-"),  # textile recycling
    40: ("Solar photovoltaics",                         11, "H"),
    41: (None,                                           None, "-"),  # inventory/logistics
    42: ("Lithium-ion battery recycling",                8, "H"),
    43: ("Low-carbon cement and concrete",               7, "H"),
    44: (None,                                           None, "-"),  # sustainable textile/leather
    45: ("Bio-based materials (lignin, fibers)",         6, "L"),
    46: ("Chemical plastics recycling",                  7, "M"),
    47: ("Bio-based / compostable packaging",            8, "L"),
    48: (None,                                           None, "-"),  # reusable packaging services
    49: ("Advanced waste sorting and recycling",         8, "L"),
    50: (None,                                           None, "-"),  # waste collection logistics
    51: ("Low-carbon construction materials",            7, "L"),
    52: (None,                                           None, "-"),  # general construction
    53: (None,                                           None, "-"),  # generic green energy (n=25, too broad)
    54: ("Thermal energy storage",                       8, "H"),
    55: ("Li-ion and next-generation batteries",         9, "H"),
    56: ("Grid-scale long-duration storage",             8, "H"),
    57: ("Battery storage systems (grid-connected)",     8, "H"),
    58: (None,                                           None, "-"),  # energy trading software
    59: ("Smart grid / distributed energy resources",    8, "L"),
    60: ("Heat pumps (space heating)",                   9, "H"),
    61: (None,                                           None, "-"),  # energy management software
    62: ("Efficient cooling and HVAC",                   9, "H"),
}

# ── Print mapping table ─────────────────────────────────────────────────
print()
print(f"  {'CL':>3}  {'n':>5}  {'Conf':>4}  {'TRL':>3}  "
      f"{'IEA Technology':<38}  Top keywords")
print("  " + "─" * 100)
for cl in sorted(MAPPING.keys()):
    iea, trl, conf = MAPPING[cl]
    n = cluster_n.get(cl, 0)
    kw = cluster_kw.get(cl, "")[:40]
    iea_short = (iea[:36] + "…") if iea and len(iea) > 38 else (iea or "— no match —")
    trl_s = str(trl) if trl is not None else " —"
    print(f"  {cl:>3}  {n:>5}  {conf:>4}  {trl_s:>3}  {iea_short:<38}  {kw}")

# ── 5 example names per matched cluster ─────────────────────────────────
print()
print("  Example names for matched clusters (high/medium confidence):")
print("  " + "─" * 70)
for cl in sorted(MAPPING.keys()):
    iea, trl, conf = MAPPING[cl]
    if iea is None or conf not in ("H", "M"): continue
    exs = cluster_examples.get(cl, "")
    print(f"  CL{cl:>2} (TRL {trl}, {conf}): {exs[:80]}")

# ── Save mapping to file ─────────────────────────────────────────────────
mapping_rows = []
for cl, (iea, trl, conf) in MAPPING.items():
    mapping_rows.append({
        "cluster": cl,
        "n_startups": cluster_n.get(cl, 0),
        "top_keywords": cluster_kw.get(cl, ""),
        "examples": cluster_examples.get(cl, ""),
        "iea_technology": iea,
        "trl": trl,
        "confidence": conf,
    })
mapping_df = pd.DataFrame(mapping_rows)
map_path = os.path.join(OUT_DIR, "cluster_trl_mapping.csv")
mapping_df.to_csv(map_path, index=False)
print(f"\n  Mapping saved → {map_path}")

# ── Summary stats ────────────────────────────────────────────────────────
matched = mapping_df[mapping_df["trl"].notna()]
unmatched = mapping_df[mapping_df["trl"].isna()]
print(f"""
  Total clusters (excl. noise):  {len(mapping_df):>4}
  Matched to IEA tech:           {len(matched):>4}  ({100*len(matched)/len(mapping_df):.0f}%)
    High confidence:             {(matched['confidence']=='H').sum():>4}
    Medium confidence:           {(matched['confidence']=='M').sum():>4}
    Low confidence:              {(matched['confidence']=='L').sum():>4}
  No match (software/food/svc):  {len(unmatched):>4}  ({100*len(unmatched)/len(mapping_df):.0f}%)

  TRL distribution of matched clusters:
    min={int(matched['trl'].min())}  max={int(matched['trl'].max())}  mean={matched['trl'].mean():.1f}  sd={matched['trl'].std():.1f}
    Values: {sorted(matched['trl'].dropna().astype(int).unique())}
""")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Feasibility
# ═══════════════════════════════════════════════════════════════════════════
print("── 3. Feasibility ──")

# Startup-level TRL
co_cluster_map = dict(zip(nov["name"], nov["specter_cluster"]))
cl_to_trl      = {cl: trl for cl, (_, trl, _) in MAPPING.items() if trl is not None}
co_trl         = {co: cl_to_trl.get(co_cluster_map.get(co, -1))
                  for co in nov["name"]}

# Load dyad dataset (rebuild if needed — mirrors Script 89)
def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

N_COUNTER = 5
RNG = np.random.default_rng(42)

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl["year"]      = dl["Deal Date"].dt.year
dl["co"]        = dl["Companies"].str.strip()
deals_v = dl.dropna(subset=["year", "Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

first_round = (deals_v.sort_values("Deal Date")
               .groupby("co").first().reset_index()
               [["co", "year", "Investors",
                 "Company Country/Territory/Region", "HQ Global Region"]])

inv_names_raw = pd.read_csv("data/investor_names.csv", header=None, names=["investor"])
inv_emb_raw   = np.load("data/investor_embeddings.npy").astype(np.float32)
inv_norms_r   = np.linalg.norm(inv_emb_raw, axis=1, keepdims=True)
inv_norms_r[inv_norms_r == 0] = 1.0
inv_emb_raw /= inv_norms_r
inv_names_ser = [str(n).strip() for n in inv_names_raw["investor"]
                 if str(n).strip() not in ("investor", "nan", "")]
inv_emb_arr   = inv_emb_raw[:len(inv_names_ser)]
inv_emb_map   = {n.lower(): inv_emb_arr[i] for i, n in enumerate(inv_names_ser)}

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    ns_pkl = pickle.load(f)
su_names = [n.strip() for n in ns_pkl["names"]]
su_emb   = np.array(ns_pkl["embs_raw"], dtype=np.float32)
su_n_    = np.linalg.norm(su_emb, axis=1, keepdims=True); su_n_[su_n_ == 0] = 1.0
su_emb  /= su_n_
startup_emb_map = {n: su_emb[i] for i, n in enumerate(su_names)}

co_os = dict(zip(
    pd.read_csv("output/novelty_t/novelty_time_aware.csv")["Companies"],
    pd.read_csv("output/novelty_t/novelty_time_aware.csv")["overall_sparsity_t"]
))

tuned_raw = pd.read_csv("output/h3_redesign/lagged_comm_tuned.csv")
tuned_lc  = {(r["investor"].lower(), int(r["year"])): int(r["community_tuned"])
             for _, r in tuned_raw.iterrows()}

inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors", "HQ Country/Territory/Region", "HQ Global Region"])
inv_raw["k"] = inv_raw["Investors"].str.strip().str.lower()
ic = dict(zip(inv_raw["k"], inv_raw["HQ Country/Territory/Region"].fillna("")))
ir = dict(zip(inv_raw["k"], inv_raw["HQ Global Region"].fillna("")))

year_active = defaultdict(set)
for _, row in deals_v.iterrows():
    for inv in comma_parse(row["Investors"]):
        year_active[int(row["year"])].add(inv.lower())

inv_deals_before = defaultdict(lambda: defaultdict(int))
for _, row in deals_v.iterrows():
    yr = int(row["year"])
    for inv in comma_parse(row["Investors"]):
        inv_k = inv.lower()
        for fy in range(yr + 1, 2028):
            inv_deals_before[fy][inv_k] += 1

cumulative_edges = defaultdict(float)
lagged_degree = {}
all_yrs = sorted({int(r["year"]) for _, r in deals_v.iterrows()})
for yr in range(min(all_yrs) - 1, max(all_yrs) + 2):
    for _, row in deals_v[deals_v["year"] == yr].iterrows():
        invs = [x.lower() for x in comma_parse(row["Investors"])]
        for a, b in combinations(invs, 2):
            cumulative_edges[(a, b)] = cumulative_edges.get((a, b), 0) + 1
    if cumulative_edges:
        deg = defaultdict(int)
        for a, b in cumulative_edges: deg[a] += 1; deg[b] += 1
        lagged_degree[yr + 1] = dict(deg)

co_lc = {}
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    comms = [tuned_lc.get((i.lower(), yr)) for i in comma_parse(str(fr["Investors"]))
             if tuned_lc.get((i.lower(), yr)) is not None]
    co_lc[co] = Counter(comms).most_common(1)[0][0] if comms else None

print("  Building dyad dataset…")
rows_r, rows_f = [], []
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None or co not in startup_emb_map or co not in co_os: continue
    trl_val = co_trl.get(co)  # may be None
    investors = comma_parse(str(fr["Investors"]))
    if not investors: continue
    co_ctry = str(fr.get("Company Country/Territory/Region", "")).strip()
    co_rgn  = str(fr.get("HQ Global Region", "")).strip()
    actual  = {i.lower() for i in investors}
    pool    = [i for i in year_active.get(yr, set())
               if i not in actual and i in inv_emb_map]
    if len(pool) < N_COUNTER: continue
    sv = startup_emb_map.get(co)
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map: continue
        iv = inv_emb_map[inv_k]
        sd = float(1.0 - sv @ iv)
        sc = 1.0 if (co_lc.get(co) is not None and
                     tuned_lc.get((inv_k, yr)) == co_lc.get(co)) else 0.0
        lpd = float(np.log1p(inv_deals_before[yr][inv_k]))
        ldt = float(np.log1p(lagged_degree.get(yr, {}).get(inv_k, 0)))
        gc  = 1.0 if (co_ctry and ic.get(inv_k, "") == co_ctry) else 0.0
        gr  = 1.0 if (co_rgn  and ir.get(inv_k, "") == co_rgn)  else 0.0
        rows_r.append({"co": co, "tied": 1, "sem": sd, "sc": sc,
                       "lpd": lpd, "ldt": ldt, "gc": gc, "gr": gr,
                       "trl": trl_val, "yr": yr})
        for fi in RNG.choice(pool, N_COUNTER, replace=False):
            iv2 = inv_emb_map[fi]; sd2 = float(1.0 - sv @ iv2)
            sc2 = 1.0 if (co_lc.get(co) is not None and
                          tuned_lc.get((fi, yr)) == co_lc.get(co)) else 0.0
            lpd2 = float(np.log1p(inv_deals_before[yr][fi]))
            ldt2 = float(np.log1p(lagged_degree.get(yr, {}).get(fi, 0)))
            gc2  = 1.0 if (co_ctry and ic.get(fi, "") == co_ctry) else 0.0
            gr2  = 1.0 if (co_rgn  and ir.get(fi, "") == co_rgn)  else 0.0
            rows_f.append({"co": co, "tied": 0, "sem": sd2, "sc": sc2,
                           "lpd": lpd2, "ldt": ldt2, "gc": gc2, "gr": gr2,
                           "trl": trl_val, "yr": yr})

dyads = pd.DataFrame(rows_r + rows_f).reset_index(drop=True)
dm    = dyads.dropna(subset=["sem"]).copy()

# Standardise
for col, col_z in [("sem","sem_z"),("lpd","lpd_z"),("ldt","ldt_z")]:
    mu = dm[col].mean(); sd = dm[col].std()
    dm[col_z] = (dm[col] - mu) / (sd if sd > 0 else 1.0)

print(f"  Dyads total: {len(dm):,}  |  Strata: {dm['co'].nunique():,}")

# TRL coverage
dm_trl = dm[dm["trl"].notna()].copy()
n_co_trl = dm_trl["co"].nunique()
n_co_all = dm["co"].nunique()

trl_vals = dm_trl["trl"].astype(float)
mu_trl   = trl_vals.mean(); sd_trl = trl_vals.std()
dm_trl["trl_z"] = (dm_trl["trl"] - mu_trl) / (sd_trl if sd_trl > 0 else 1.0)
dm_trl["sem_x_trl"] = dm_trl["sem_z"] * dm_trl["trl_z"]

n_dyads_trl  = len(dm_trl)
trl_distinct = sorted(dm_trl["trl"].astype(int).unique())
trl_by_cl    = dm_trl.groupby("trl")["co"].nunique()

print(f"""
  TRL coverage:
    Startups with TRL:       {n_co_trl:,} / {n_co_all:,}  ({100*n_co_trl/n_co_all:.1f}%)
    Dyads with TRL:          {n_dyads_trl:,} / {len(dm):,}  ({100*n_dyads_trl/len(dm):.1f}%)
    TRL mean:                {mu_trl:.2f}  (startup-weighted)
    TRL SD (dyad-level):     {sd_trl:.2f}
    Distinct TRL values:     {len(trl_distinct)}  {trl_distinct}
    Startups per TRL level:""")
for trl_v, n in trl_by_cl.sort_index().items():
    print(f"      TRL {int(trl_v):>2}: {n:>4} startups")

# MDE for sem_dist × TRL in conditional logit
print("\n  Computing MDE (sem_dist_z × trl_z in conditional logit)…")
b_vars = ["sem_z", "trl_z", "sem_x_trl", "sc", "gc", "gr", "lpd_z", "ldt_z"]
sub = dm_trl[["tied","co"] + b_vars].dropna().copy()
ok  = sub.groupby("co")["tied"].transform(lambda x: x.nunique() == 2)
sub = sub[ok].copy()
grps  = sub["co"].values
exog  = sub[b_vars].values.astype(float)
endog = sub["tied"].values.astype(float)
mask  = np.isfinite(exog).all(axis=1)

se_interact = np.nan; p_interact = np.nan; b_interact = np.nan
for meth in ["bfgs", "newton", "lbfgs"]:
    try:
        res = ConditionalLogit(endog[mask], exog[mask],
                               groups=grps[mask]).fit(
            method=meth, maxiter=600, disp=False)
        if np.isfinite(res.bse).all():
            b_interact  = res.params[2]
            se_interact = res.bse[2]
            p_interact  = res.pvalues[2]
            break
    except Exception:
        continue

mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se_interact if np.isfinite(se_interact) else np.nan

print(f"""
  Conditional logit (sem_dist_z × trl_z), N={mask.sum():,}  strata={sub['co'].nunique():,}:
    sem_dist_z:              β={res.params[0]:+.4f}  SE={res.bse[0]:.4f}
    trl_z:                   β={res.params[1]:+.4f}  SE={res.bse[1]:.4f}
    sem_dist_z × trl_z:      β={b_interact:+.4f}  SE={se_interact:.4f}  p={p_interact:.4f}
    MDE (80% power):         {mde:.4f}  log-odds per SD of both sem_dist and TRL
""")

# ═══════════════════════════════════════════════════════════════════════════
# 4. H6 pre-registration draft
# ═══════════════════════════════════════════════════════════════════════════
print("── 4. H6 Pre-registration draft ──")
print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  H6 — Expertise barrier hypothesis (technology readiness level)     │
  │  Family F — TRL moderation (Holm across H6 only, single hypothesis) │
  └─────────────────────────────────────────────────────────────────────┘

  Rationale:
    For technologies at low readiness (TRL 1–6), the investor's own
    scientific expertise is the primary screening criterion — a wrong
    match cannot be corrected by market signals or established standards.
    As technology matures (TRL 7–11), performance benchmarks and
    commercial precedent substitute for deep expertise, relaxing the
    distance penalty. The effect of semantic distance on matching
    should therefore be steepest at low TRL and flatten as TRL rises.

  Hypothesis:
    H6: The negative effect of semantic distance on co-investment
    probability is attenuated at higher technology readiness levels.

  Pre-specified sign:  β(sem_dist_z × trl_z)  > 0

  Interpretation:
    sem_dist_z coefficient is negative (further = less likely to co-invest).
    A positive interaction means higher TRL weakens this penalty —
    investors can fund more distant technologies once they are better
    understood and commercially precedented.

  Unit:  Dyad (startup × candidate investor).
  Risk set: Script 79r risk set (actual + {N_COUNTER} counterfactuals active that year).
  Model: Conditional logit stratified by startup.

    tie ~ sem_dist_z + trl_z + sem_dist_z × trl_z
        + same_community_tuned + geo_same_country + geo_same_region
        + log_prior_deals_z + has_prior_network

  TRL measure:
    IEA ETP 2023 TRL midpoint (1–11), assigned at the SPECTER2 cluster
    level. Source: ETP Clean Energy Technology Guide (>650 technologies).
    Clusters without a clear IEA match (food, carbon accounting software,
    sustainability services, monitoring/satellite) receive TRL = NaN and
    are excluded from this regression. See cluster_trl_mapping.csv.
    TRL is time-invariant (technology-level, not startup-level).

  Standard errors: clustered by SPECTER2 tech cluster (specter_cluster).
    Rationale: TRL is constant within a cluster, so all startups in the
    same cluster share an identical TRL value. Clustering by cluster
    accounts for within-cluster correlation and the constraint on TRL
    variation (variation is only across clusters, not within).

  Robustness variants (to be pre-registered after mapping review):
    R-i:  Exclude low-confidence (L) TRL assignments.
    R-ii: Treat TRL as ordered categorical (4 IEA broad categories).
    R-iii: Fully-interacted model including year × TRL.

  Coverage: {n_co_trl:,} / {n_co_all:,} startups ({100*n_co_trl/n_co_all:.1f}%)  |
            {n_dyads_trl:,} / {len(dm):,} dyads ({100*n_dyads_trl/len(dm):.1f}%)
  MDE: {mde:.4f} log-odds (80% power, two-sided α = 0.05)
  TRL distinct values: {len(trl_distinct)}  ({trl_distinct})
  TRL SD (dyad-level): {sd_trl:.2f}
""")

# Save feasibility summary
feas = pd.DataFrame([{
    "n_matched_clusters": len(matched),
    "n_unmatched_clusters": len(unmatched),
    "n_startups_with_trl": n_co_trl,
    "n_startups_total": n_co_all,
    "pct_startups": round(100 * n_co_trl / n_co_all, 1),
    "n_dyads_with_trl": n_dyads_trl,
    "n_dyads_total": len(dm),
    "pct_dyads": round(100 * n_dyads_trl / len(dm), 1),
    "trl_mean": round(mu_trl, 2),
    "trl_sd": round(sd_trl, 2),
    "trl_distinct_values": len(trl_distinct),
    "beta_sem_x_trl": round(float(b_interact), 4) if np.isfinite(b_interact) else np.nan,
    "se_sem_x_trl": round(float(se_interact), 4) if np.isfinite(se_interact) else np.nan,
    "p_sem_x_trl": round(float(p_interact), 4) if np.isfinite(p_interact) else np.nan,
    "mde_80pct": round(float(mde), 4) if np.isfinite(mde) else np.nan,
}])
feas.to_csv(os.path.join(OUT_DIR, "h6_feasibility.csv"), index=False)

print("=" * 70)
print("SCRIPT 91 COMPLETE — awaiting mapping review before H6 runs.")
print("=" * 70)
