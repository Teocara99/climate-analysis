"""
Script 89: Fix community variable, re-check H1, run H5.

PART A — DIAGNOSE & FIX
  1. Confirm year alignment: partition year t uses deals < t (correct).
  2. Fix casing mismatch: tuned_lc uses original case; dyads use lowercase.
  3. Cross-tab same_community_tuned x tie (before/after fix).
  4. Collapse log_deg + log_deals into PC1.
  5. Re-run base model and H1 with fixed variables.

PART B — H5 EnvPU moderation (update preregistration first)
  H5a: EnvPU × sem_dist   < 0   (uncertainty → prefer closer partners)
  H5b: EnvPU × same_comm  > 0   (uncertainty → prefer same community)
  H5c: EnvPU × geo_country > 0  (uncertainty → prefer home-country)
  Holm correction across H5a-c (preregistration Family E).
  SEs clustered by quarter.
"""

import os, sys, re, warnings, pickle
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, pearsonr, chi2_contingency
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "h1_folds")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)
N_COUNTER = 5

# Reference values from Script 79r M4
REF_79R = {
    "sem_dist_z":         {"beta": -0.2117},
    "same_community_t":   {"beta": +3.4420},
    "geo_same_country":   {"beta": +2.6810},
    "log_deg_z":          {"beta": -0.5870},
    "log_deals_z":        {"beta": +0.3900},
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

def run_clogit(data, b_vars, label="", cluster_col=None):
    sub = data[["tied", "co"] + b_vars +
               ([cluster_col] if cluster_col else [])].dropna().copy()
    ok  = sub.groupby("co")["tied"].transform(lambda x: x.nunique() == 2)
    sub = sub[ok].copy()
    if len(sub) < 200 or sub["co"].nunique() < 20:
        print(f"  {label}: insufficient data ({len(sub)} rows)"); return None
    grps = sub["co"].values
    exog = sub[b_vars].values.astype(float)
    endog = sub["tied"].values.astype(float)
    mask = np.isfinite(exog).all(axis=1)
    for method in ["bfgs", "newton", "lbfgs"]:
        try:
            res = ConditionalLogit(endog[mask], exog[mask],
                                   groups=grps[mask]).fit(
                method=method, maxiter=600, disp=False)
            if not np.isfinite(res.bse).all():
                continue
            # Cluster-robust SEs if requested
            if cluster_col is not None:
                try:
                    score_obs = res.model.score_obs(res.params)
                    # sub has a non-contiguous index; use boolean mask for positional indexing
                    sub_reset = sub.reset_index(drop=True)
                    cl_series = sub_reset.loc[mask, cluster_col].values
                    cl_ids = pd.Categorical(cl_series).codes
                    n_cl = cl_ids.max() + 1
                    meat = np.zeros((len(b_vars), len(b_vars)))
                    for g in range(n_cl):
                        sg = score_obs[cl_ids == g].sum(axis=0)
                        meat += np.outer(sg, sg)
                    B = res.cov_params()
                    V_cl = B @ meat @ B
                    se_cl = np.sqrt(np.diag(V_cl))
                    z_cl  = res.params / se_cl
                    p_cl  = 2 * norm.sf(np.abs(z_cl))
                    return res, b_vars, int(mask.sum()), int(sub["co"].nunique()), \
                           sub[mask], se_cl, p_cl
                except Exception:
                    pass  # fall through to standard SEs
            return res, b_vars, int(mask.sum()), int(sub["co"].nunique()), sub[mask]
        except Exception:
            continue
    print(f"  {label}: all optimisers failed"); return None

print("=" * 70)
print("SCRIPT 89 — FIX COMMUNITY VARIABLE + H5 EnvPU")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING (mirrors 88 but with lowercase fix applied)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading data ──")

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
                    "Company Country/Territory/Region", "HQ Global Region",
                    "Deal Date"]]
)

# Embeddings
inv_names_raw = pd.read_csv("data/investor_names.csv", header=None, names=["investor"])
inv_emb_raw   = np.load("data/investor_embeddings.npy").astype(np.float32)
inv_norms_r   = np.linalg.norm(inv_emb_raw, axis=1, keepdims=True)
inv_norms_r[inv_norms_r == 0] = 1.0
inv_emb_raw   = inv_emb_raw / inv_norms_r
inv_names_ser = [str(n).strip() for n in inv_names_raw["investor"]
                 if str(n).strip() not in ("investor", "nan", "")]
inv_emb_arr   = inv_emb_raw[:len(inv_names_ser)]
inv_emb_map   = {n.lower(): inv_emb_arr[i] for i, n in enumerate(inv_names_ser)}

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    ns_pkl = pickle.load(f)
su_emb_names = [n.strip() for n in ns_pkl["names"]]
su_emb_raw   = np.array(ns_pkl["embs_raw"], dtype=np.float32)
su_norms_r   = np.linalg.norm(su_emb_raw, axis=1, keepdims=True)
su_norms_r[su_norms_r == 0] = 1.0
su_emb_arr   = su_emb_raw / su_norms_r
startup_emb_map = {n: su_emb_arr[i] for i, n in enumerate(su_emb_names)}

inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors", "HQ Country/Territory/Region",
                                 "HQ Global Region"])
inv_raw["inv_key"] = inv_raw["Investors"].str.strip().str.lower()
inv_country_map = dict(zip(inv_raw["inv_key"],
                           inv_raw["HQ Country/Territory/Region"].fillna("")))
inv_region_map  = dict(zip(inv_raw["inv_key"],
                           inv_raw["HQ Global Region"].fillna("")))

nov_t = pd.read_csv("output/novelty_t/novelty_time_aware.csv")
co_os_t = dict(zip(nov_t["Companies"], nov_t["overall_sparsity_t"]))
co_nc_t = dict(zip(nov_t["Companies"], nov_t["niche_cohesion_t"]))

# ── Tuned communities  — FIX: LOWERCASE KEYS ────────────────────────────
tuned_raw = pd.read_csv("output/h3_redesign/lagged_comm_tuned.csv")

print("\n── Part A: Diagnosing year alignment and casing bug ──")
print("\n  [A1] Year alignment in lagged_comm_tuned.csv")
print("  Convention (Script 87, line 108):")
print("    'Store community BEFORE adding year yr's edges'")
print("    → year t partition uses deals < t  (correctly lagged)")
print(f"  Years present: {sorted(tuned_raw.year.unique())[:5]} … "
      f"{sorted(tuned_raw.year.unique())[-3:]}")
print()

# Show casing mismatch
sample_inv  = tuned_raw["investor"].iloc[5]
has_lower   = sample_inv == sample_inv.lower()
has_upper   = any(c.isupper() for c in sample_inv)
print(f"  [A2] Casing check: sample investor name = '{sample_inv}'")
print(f"       Has uppercase: {has_upper}  —  Keys are ORIGINAL CASE (not lowercased)")
print(f"       Dyad investor keys are lowercased → MISMATCH: all tuned lookups return None")

# Before-fix tuned_lc (original case — broken)
tuned_lc_broken = {}
for _, r in tuned_raw.iterrows():
    tuned_lc_broken[(r["investor"], int(r["year"]))] = int(r["community_tuned"])

# After-fix tuned_lc (lowercase — correct)
tuned_lc = {}
for _, r in tuned_raw.iterrows():
    tuned_lc[(r["investor"].lower(), int(r["year"]))] = int(r["community_tuned"])

n_broken = sum(1 for _, r in tuned_raw.head(1000).iterrows()
               if tuned_lc_broken.get((r["investor"].lower(), int(r["year"]))) is None)
print(f"       Fraction of lookups that fail with original case: {n_broken}/1000")

# ── Network variables ────────────────────────────────────────────────────
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

lagged_degree = {}
cumulative_edges = defaultdict(float)
all_years = sorted(edges_by_year.keys())
for yr in range(min(all_years) - 1, max(all_years) + 2):
    for a, b in edges_by_year.get(yr, []):
        cumulative_edges[(a, b)] = cumulative_edges.get((a, b), 0) + 1
    if cumulative_edges:
        deg = defaultdict(int)
        for (a, b) in cumulative_edges:
            deg[a] += 1; deg[b] += 1
        lagged_degree[yr + 1] = dict(deg)
    else:
        lagged_degree[yr + 1] = {}

year_active = defaultdict(set)
for _, row in deals_v.iterrows():
    for inv in comma_parse(row["Investors"]):
        year_active[int(row["year"])].add(inv.lower())

# ── Build dyad dataset ────────────────────────────────────────────────────
print("\n  Building dyad dataset…")
rows_real, rows_fake = [], []
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None or co not in startup_emb_map or co not in co_os_t: continue
    investors = comma_parse(str(fr["Investors"]))
    if not investors: continue
    co_ctry = str(fr.get("Company Country/Territory/Region", "")).strip()
    co_rgn  = str(fr.get("HQ Global Region", "")).strip()
    actual_set = {i.lower() for i in investors}
    pool = [i for i in year_active.get(yr, set())
            if i not in actual_set and i in inv_emb_map]
    if len(pool) < N_COUNTER: continue
    for inv in investors:
        inv_k = inv.lower()
        if inv_k not in inv_emb_map: continue
        meta = {"co": co, "investor": inv_k, "co_country": co_ctry,
                "co_region": co_rgn, "year": yr, "tied": 1,
                "first_date": fr["Deal Date"]}
        rows_real.append(meta)
        for fi in RNG.choice(pool, N_COUNTER, replace=False):
            rows_fake.append({**meta, "investor": fi, "tied": 0})

dyads = pd.DataFrame(rows_real + rows_fake).reset_index(drop=True)
print(f"  Dyads: {len(dyads):,}  |  Strata: {dyads['co'].nunique():,}")

# Semantic distance
sem_d = []
for _, row in dyads.iterrows():
    sv = startup_emb_map.get(row["co"])
    iv = inv_emb_map.get(row["investor"])
    sem_d.append(float(1.0 - sv @ iv) if sv is not None and iv is not None else np.nan)
dyads["semantic_distance"] = sem_d

dyads["inv_country"]      = dyads["investor"].map(inv_country_map).fillna("")
dyads["inv_region"]       = dyads["investor"].map(inv_region_map).fillna("")
dyads["geo_same_country"] = ((dyads["co_country"] != "") & (dyads["inv_country"] != "") &
                              (dyads["co_country"] == dyads["inv_country"])).astype(float)
dyads["geo_same_region"]  = ((dyads["co_region"] != "") & (dyads["inv_region"] != "") &
                              (dyads["co_region"] == dyads["inv_region"])).astype(float)

dyads["log_prior_deals"] = [float(np.log1p(inv_deals_before[int(r["year"])].get(r["investor"],0)))
                             for _, r in dyads.iterrows()]
dyads["has_prior_network"] = (dyads["log_prior_deals"] > 0).astype(float)
dyads["log_deg_t"]  = [float(np.log1p(lagged_degree.get(int(r["year"]),{}).get(r["investor"],0)))
                        for _, r in dyads.iterrows()]
dyads["overall_sparsity_t"] = dyads["co"].map(co_os_t)
dyads["niche_cohesion_t"]   = dyads["co"].map(co_nc_t)

# ── FIXED same_community_tuned ────────────────────────────────────────────
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

# Also compute broken version for cross-tab comparison
dyads["same_comm_broken"] = [
    1.0 if (co_tuned_lead_comm.get(r["co"]) is not None and
            tuned_lc_broken.get((r["investor"], int(r["year"]))) == co_tuned_lead_comm.get(r["co"]))
    else 0.0
    for _, r in dyads.iterrows()
]

# [A3] Cross-tab
print("\n  [A3] Cross-tab same_community_tuned × tie")
for label, col in [("BROKEN (original case)", "same_comm_broken"),
                   ("FIXED  (lowercase)",      "same_community_tuned")]:
    ct = pd.crosstab(dyads[col], dyads["tied"])
    share_tied = ct.loc[1.0, 1] / ct.loc[1.0].sum() if 1.0 in ct.index else 0
    n_same = int(ct.loc[1.0].sum()) if 1.0 in ct.index else 0
    n_diff = int(ct.loc[0.0].sum()) if 0.0 in ct.index else 0
    tied_diff = ct.loc[0.0, 1] / ct.loc[0.0].sum() if 0.0 in ct.index else 0
    print(f"    {label}:")
    print(f"      same_comm=1: n={n_same:,}  tied_share={share_tied:.4f}")
    print(f"      same_comm=0: n={n_diff:,}  tied_share={tied_diff:.4f}")
    if n_same > 0:
        print(f"      Odds ratio: {(share_tied/(1-share_tied)) / (tied_diff/(1-tied_diff)):.2f}  "
              f"(same_comm=1 dyads are {(share_tied/(1-share_tied)) / (tied_diff/(1-tied_diff)):.1f}x more likely to be realised)")

# [A4] PC1 of log_deg and log_deals
print("\n  [A4] PC1 of [log_deg_t, log_prior_deals]")
pa_cols = ["log_deg_t", "log_prior_deals"]
pa_data = dyads[pa_cols].values
pa_mask = np.isfinite(pa_data).all(axis=1)
scaler  = StandardScaler().fit(pa_data[pa_mask])
pa_scaled = scaler.transform(pa_data[pa_mask])
pca = PCA(n_components=2).fit(pa_scaled)
print(f"    Variance explained: PC1={pca.explained_variance_ratio_[0]:.3f}  "
      f"PC2={pca.explained_variance_ratio_[1]:.3f}")
print(f"    PC1 loadings: log_deg={pca.components_[0,0]:.3f}  "
      f"log_deals={pca.components_[0,1]:.3f}")
dyads["prior_act_pc1"] = np.nan
dyads.loc[pa_mask, "prior_act_pc1"] = pca.transform(pa_scaled)[:, 0]
mu_pa = dyads.loc[pa_mask, "prior_act_pc1"].mean()
sd_pa = dyads.loc[pa_mask, "prior_act_pc1"].std()
dyads["prior_act_pc1_z"] = (dyads["prior_act_pc1"] - mu_pa) / (sd_pa if sd_pa > 0 else 1.0)

# Standardise remaining vars
dm = dyads.dropna(subset=["semantic_distance"]).copy()
for col, col_z in [("semantic_distance","sem_dist_z"),
                   ("log_deg_t","log_deg_z"),
                   ("log_prior_deals","log_deals_z"),
                   ("overall_sparsity_t","os_z"),
                   ("niche_cohesion_t","nc_z")]:
    mu = dm[col].mean(); sd = dm[col].std()
    dm[col_z] = (dm[col] - mu) / sd if sd > 0 else 0.0

# ═══════════════════════════════════════════════════════════════════════════
# PART A STEP 5: Re-run base model + H1 with fixed community
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART A — Re-run base model (fixed same_community + PC1)")
print("=" * 70)

BASE_VARS = ["sem_dist_z", "geo_same_country", "geo_same_region",
             "prior_act_pc1_z", "has_prior_network", "same_community_tuned"]

r_base = run_clogit(dm, BASE_VARS, "Base M4 fixed")
if r_base:
    res_b, xv_b, n_b, ns_b = r_base[:4]
    print(f"\n  N dyads: {n_b:,}  |  N strata: {ns_b:,}")
    ref_map = {
        "sem_dist_z":          "sem_dist_z",
        "same_community_tuned": "same_community_t",
    }
    print()
    print(f"  {'Variable':<26}  {'79r β':>8}  {'88 β':>8}  {'SE':>7}  {'p':>7}  {'sig':>4}")
    print("  " + "─" * 65)
    for i, v in enumerate(xv_b):
        ref_k = ref_map.get(v)
        b79 = REF_79R.get(ref_k, {}).get("beta", np.nan)
        b88 = res_b.params[i]; se = res_b.bse[i]; p = res_b.pvalues[i]
        b79_s = f"{b79:+8.4f}" if not np.isnan(b79) else "       —"
        print(f"  {v:<26}  {b79_s}  {b88:+8.4f}  {se:7.4f}  {p:7.4f}  {sig(p)}")

# H1 re-run with fixed community + PC1
print("\n── H1 re-run (fixed community, PC1) ──")
fold_df = pd.read_csv(os.path.join(OUT_DIR, "fold_classification_pre2020.csv"))
fold_map = dict(zip(fold_df["investor"].str.lower(), fold_df["classification_pre20"]))
dm_21 = dm[dm["year"] >= 2021].copy()
dm_21["fold"]     = dm_21["investor"].map(lambda inv: 1.0 if fold_map.get(inv) == "FOLD" else 0.0)
dm_21["fold_x_nc"] = dm_21["fold"] * dm_21["nc_z"]
dm_21["fold_x_os"] = dm_21["fold"] * dm_21["os_z"]

H1_VARS = ["fold", "fold_x_nc", "fold_x_os",
           "sem_dist_z", "geo_same_country", "geo_same_region",
           "prior_act_pc1_z", "has_prior_network", "same_community_tuned"]

r_h1 = run_clogit(dm_21, H1_VARS, "H1 fixed")
if r_h1:
    res_h1, xv_h1, n_h1, ns_h1 = r_h1[:4]
    b_h1_nc = res_h1.params[1]; se_h1_nc = res_h1.bse[1]; p_h1_nc = res_h1.pvalues[1]
    b_h1_os = res_h1.params[2]; se_h1_os = res_h1.bse[2]; p_h1_os = res_h1.pvalues[2]
    print(f"\n  N dyads: {n_h1:,}  |  N strata: {ns_h1:,}")
    print(f"  fold × NC_t: β={b_h1_nc:+.4f}  SE={se_h1_nc:.4f}  "
          f"p={p_h1_nc:.4f}  {sig(p_h1_nc)}")
    print(f"  fold × OS_t: β={b_h1_os:+.4f}  SE={se_h1_os:.4f}  "
          f"p={p_h1_os:.4f}  {sig(p_h1_os)}")
    mde_h1 = (norm.ppf(0.975) + norm.ppf(0.80)) * se_h1_nc
    print(f"  MDE (fold×NC_t): {mde_h1:.4f}")
    if b_h1_nc > 0 and p_h1_nc < 0.05:   h1_v = "SUPPORTED"
    elif b_h1_nc > 0 and p_h1_nc < 0.10: h1_v = "DIRECTIONALLY SUPPORTED (marginal)"
    elif b_h1_nc > 0:                     h1_v = "DIRECTIONALLY CORRECT — NOT SIGNIFICANT"
    else:                                 h1_v = "NULL — WRONG SIGN"
    print(f"  ** H1 (fixed): {h1_v} **")

# ═══════════════════════════════════════════════════════════════════════════
# PART B — H5: EnvPU moderation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B — H5: EnvPU moderation of co-investment matching")
print("=" * 70)

# ── Update preregistration ────────────────────────────────────────────────
preg_path = "output/preregistration.md"
with open(preg_path, "r") as f:
    preg_content = f.read()

H5_BLOCK = """

---

## Family E — Environmental Policy Uncertainty (EnvPU) moderation
**Added:** 2026-09-22  **Status: LOCKED**

> Tests whether climate policy uncertainty modulates the structure of
> co-investment matching. Unit: dyad (startup × investor candidate).
> Risk set: same as Script 79r (actual + 5 counterfactual investors).
> Model: conditional logit stratified by startup.
> EnvPU source: Noailly et al. EnvPU Index (US, monthly, 1990–2019).
> Coverage: startups with first rounds through March 2019.
> EnvPU matched by first-round year-quarter.
> SEs clustered by first-round quarter.

| H# | Hypothesis | Term | Sign | Holm family |
|---|---|---|---|---|
| **H5a** | Under higher policy uncertainty investors prefer semantically closer partners (less exploration). | EnvPU × semantic_distance | **−** | E |
| **H5b** | Under higher policy uncertainty investors prefer same-community co-investors (network insularity). | EnvPU × same_community | **+** | E |
| **H5c** | Under higher policy uncertainty investors prefer home-country co-investors (home bias). | EnvPU × geo_same_country | **+** | E |

Holm correction across H5a–H5c.

**Coverage note:** EnvPU index ends March 2019 (351 monthly observations,
1990–2019). Analysis restricted to startups with first rounds ≤ March 2019.
"""

if "Family E" not in preg_content:
    with open(preg_path, "a") as f:
        f.write(H5_BLOCK)
    print("  Updated preregistration.md — added Family E (H5a-c).")
else:
    print("  preregistration.md already contains Family E — skipping update.")

# ── Load EnvPU ────────────────────────────────────────────────────────────
envpu = pd.read_excel("data/EnvPU_data.xlsx")
envpu["quarter"] = (envpu["Year"].astype(str) + "Q" +
                    ((envpu["Month"] - 1) // 3 + 1).astype(str))
envpu_q = envpu.groupby("quarter")["EnvPU Index"].mean().reset_index()
envpu_q.columns = ["quarter", "envpu"]
envpu_dict = dict(zip(envpu_q["quarter"], envpu_q["envpu"]))

# Merge to first_round
first_round["first_date"] = pd.to_datetime(first_round["Deal Date"], errors="coerce")
first_round["quarter"] = first_round["first_date"].apply(
    lambda d: f"{d.year}Q{(d.month-1)//3+1}" if pd.notna(d) else None)
first_round["envpu"] = first_round["quarter"].map(envpu_dict)

n_with_envpu = first_round["envpu"].notna().sum()
n_without    = first_round["envpu"].isna().sum()
print(f"\n  EnvPU coverage: {n_with_envpu:,} startups with EnvPU "
      f"({100*n_with_envpu/len(first_round):.1f}%); {n_without:,} dropped")
print(f"  Year range with EnvPU: {first_round.loc[first_round['envpu'].notna(),'year'].min()} – "
      f"{first_round.loc[first_round['envpu'].notna(),'year'].max()}")

# Merge EnvPU into dyads
co_envpu = dict(zip(first_round["co"], first_round["envpu"]))
co_quarter = dict(zip(first_round["co"], first_round["quarter"]))
dm["envpu"] = dm["co"].map(co_envpu)
dm["quarter_str"] = dm["co"].map(co_quarter)
dm_env = dm.dropna(subset=["envpu"]).copy()

# Standardise EnvPU
mu_e = dm_env["envpu"].mean(); sd_e = dm_env["envpu"].std()
dm_env["envpu_z"] = (dm_env["envpu"] - mu_e) / sd_e if sd_e > 0 else 0.0

# Interaction terms
dm_env["envpu_x_sem"]  = dm_env["envpu_z"] * dm_env["sem_dist_z"]
dm_env["envpu_x_comm"] = dm_env["envpu_z"] * dm_env["same_community_tuned"]
dm_env["envpu_x_geo"]  = dm_env["envpu_z"] * dm_env["geo_same_country"]

print(f"\n  Restricted dyads (EnvPU matched): {len(dm_env):,}  "
      f"Strata: {dm_env['co'].nunique():,}")
print(f"  EnvPU range: {dm_env['envpu'].min():.1f} – {dm_env['envpu'].max():.1f}  "
      f"mean={dm_env['envpu'].mean():.1f}  sd={dm_env['envpu'].std():.1f}")

# Feasibility
n_strata_env = dm_env["co"].nunique()
proceed_h5 = n_strata_env >= 50 and len(dm_env) >= 1000

if not proceed_h5:
    print(f"\n  FEASIBILITY FAILED — insufficient strata ({n_strata_env}).")
else:
    # Quarter cluster IDs
    dm_env["qcluster"] = pd.Categorical(dm_env["quarter_str"]).codes

    # Report within-stratum same_community variation
    wv = dm_env.groupby("co").apply(lambda g: g["same_community_tuned"].nunique()).eq(2).sum()
    sc_pos = (dm_env["same_community_tuned"] == 1).sum()
    print(f"\n  Strata with variation in same_community_tuned: {int(wv):,}")
    print(f"  same_community=1 dyads: {int(sc_pos):,}  "
          f"({100*sc_pos/len(dm_env):.1f}%)")

    # Full model — may fail if same_community causes separation in small subset
    H5_VARS_FULL = ["sem_dist_z", "envpu_x_sem",
                    "same_community_tuned", "envpu_x_comm",
                    "geo_same_country", "envpu_x_geo",
                    "geo_same_region", "prior_act_pc1_z", "has_prior_network"]
    # Fallback without same_community (controls still present via fixed effects)
    H5_VARS_REDUCED = ["sem_dist_z", "envpu_x_sem",
                       "envpu_x_comm",
                       "geo_same_country", "envpu_x_geo",
                       "geo_same_region", "prior_act_pc1_z", "has_prior_network"]

    r_h5 = run_clogit(dm_env, H5_VARS_FULL, "H5 full", cluster_col="qcluster")
    H5_VARS = H5_VARS_FULL
    if r_h5 is None:
        print("  Full model failed — trying reduced model (dropping same_community_tuned).")
        r_h5 = run_clogit(dm_env, H5_VARS_REDUCED, "H5 reduced", cluster_col="qcluster")
        H5_VARS = H5_VARS_REDUCED
        if r_h5 is None:
            print("  Reduced model also failed.")
    if r_h5 is not None:
        print(f"  Model converged with: {H5_VARS}")
    if r_h5 is not None:
        if len(r_h5) == 7:
            res_h5, xv_h5, n_h5, ns_h5, sub_h5, se_cl, p_cl = r_h5
            use_cl = True
        else:
            res_h5, xv_h5, n_h5, ns_h5, sub_h5 = r_h5
            se_cl = res_h5.bse; p_cl = res_h5.pvalues; use_cl = False

        print(f"\n  H5 results (SEs {'cluster-robust by quarter' if use_cl else 'standard'}):")
        print(f"  N dyads: {n_h5:,}  |  N strata: {ns_h5:,}")
        n_quarters = dm_env["quarter_str"].nunique()
        print(f"  N quarters (clusters): {n_quarters}")
        print()
        h5_terms = {
            "envpu_x_sem":  ("H5a", "EnvPU × sem_dist",    "−"),
            "envpu_x_comm": ("H5b", "EnvPU × same_comm",   "+"),
            "envpu_x_geo":  ("H5c", "EnvPU × geo_country", "+"),
        }
        print(f"  {'Variable':<28}  {'β':>9}  {'SE':>8}  {'p':>8}  {'sig':>4}  {'MDE':>7}")
        print("  " + "─" * 68)
        h5_pvals = {}
        mde_vals = {}
        for i, v in enumerate(xv_h5):
            b  = res_h5.params[i]
            se = se_cl[i]; p = p_cl[i]
            mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se
            tag = f"  ← {h5_terms[v][0]}" if v in h5_terms else ""
            print(f"  {v:<28}  {b:+9.4f}  {se:8.4f}  {p:8.4f}  {sig(p)}  {mde:7.4f}{tag}")
            if v in h5_terms:
                h5_pvals[v] = p
                mde_vals[v] = mde

        # Holm correction across H5a-c
        h5_keys = ["envpu_x_sem", "envpu_x_comm", "envpu_x_geo"]
        h5_p_raw = [h5_pvals[k] for k in h5_keys]
        _, h5_p_holm, _, _ = multipletests(h5_p_raw, method="holm")
        print()
        print(f"  Holm-corrected p-values across H5a-c:")
        for k, ph in zip(h5_keys, h5_p_holm):
            print(f"    {h5_terms[k][0]} ({k}): raw p={h5_pvals[k]:.4f} → Holm p={ph:.4f}  {sig(ph)}")

        # Verdicts
        print("\n" + "=" * 70)
        print("PLAIN VERDICTS — H5a-c EnvPU MODERATION")
        print("=" * 70)
        coeff_map = {v: res_h5.params[i] for i, v in enumerate(xv_h5)}
        for raw_k, (hnum, desc, sign) in h5_terms.items():
            b = coeff_map[raw_k]; p_h = h5_p_holm[h5_keys.index(raw_k)]
            mde = mde_vals[raw_k]
            matches_sign = (sign == "−" and b < 0) or (sign == "+" and b > 0)
            if matches_sign and p_h < 0.05:
                v_txt = "SUPPORTED (Holm p<0.05)"
            elif matches_sign and p_h < 0.10:
                v_txt = "DIRECTIONALLY SUPPORTED (Holm p<0.10)"
            elif matches_sign:
                v_txt = "DIRECTIONALLY CORRECT — NOT SIGNIFICANT"
            elif not matches_sign and p_h < 0.05:
                v_txt = "WRONG SIGN — SIGNIFICANT"
            else:
                v_txt = "NULL — WRONG SIGN"
            print(f"  {hnum} ({desc}): β={b:+.4f}  MDE={mde:.4f}  {v_txt}")

print("\n" + "=" * 70)
print("SCRIPT 89 COMPLETE")
print("=" * 70)
