"""
Script 92: H6 execution — TRL × semantic distance in co-investment matching.

1. Lock mapping + note pilot in preregistration.md
2. Primary: sem_dist_z × trl_z — standard SE, cluster SE (sandwich), wild bootstrap
3. Secondary: add sem_dist_z × overall_sparsity_t
4. Robustness: LOCO, drop CL 0, drop low-conf, TRL as 3 categories
"""

import os, sys, re, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from statsmodels.discrete.conditional_models import ConditionalLogit
from collections import defaultdict, Counter
from itertools import combinations

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = "output/h6_trl"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("SCRIPT 92 — H6 EXECUTION: TRL × SEMANTIC DISTANCE")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────
# 1.  Preregistration: lock mapping, note pilot
# ─────────────────────────────────────────────────────────────────────────
PREREG = "output/preregistration.md"
with open(PREREG) as f:
    content = f.read()

OLD = "**Status: LOCKED pending mapping review**"
NEW = ("**Status: LOCKED — mapping confirmed unchanged 2026-09-22**\n\n"
       "**Pilot estimate (Script 91):** β(sem_dist_z × trl_z) = +0.094 "
       "(SE=0.034, p=0.006). Directionally consistent with H6. "
       "Cluster-to-TRL mapping reviewed and confirmed unchanged after observing "
       "this pilot. Execution in Script 92 was fully planned before the pilot.")

if OLD in content:
    with open(PREREG, "w") as f:
        f.write(content.replace(OLD, NEW))
    print("  preregistration.md updated — mapping locked, pilot noted.")
else:
    print("  NOTE: status string already updated or not found.")

# ─────────────────────────────────────────────────────────────────────────
# 2.  TRL mapping (locked — identical to Script 91)
# ─────────────────────────────────────────────────────────────────────────
MAPPING = {
     0: ("Advanced nuclear (SMR + fusion)",          4, "M"),
     2: ("Green hydrogen via electrolysis",           8, "H"),
     3: ("EV charging infrastructure",                9, "H"),
     4: ("Battery electric vehicles",                10, "H"),
     5: ("Enhanced geothermal systems",               7, "M"),
    13: ("Carbon capture (CCUS / DAC)",               7, "H"),
    15: ("Ocean wave and tidal energy",               6, "H"),
    16: ("Electric aviation (eVTOL + fixed-wing)",    5, "H"),
    17: ("Offshore wind",                            10, "H"),
    21: ("Zero-emission shipping (H2 / ammonia)",     6, "M"),
    31: ("Methane abatement and monitoring",          8, "L"),
    37: ("Biomass to energy / Biochar",               8, "M"),
    40: ("Solar photovoltaics",                      11, "H"),
    42: ("Lithium-ion battery recycling",             8, "H"),
    43: ("Low-carbon cement and concrete",            7, "H"),
    45: ("Bio-based materials (lignin, fibers)",      6, "L"),
    46: ("Chemical plastics recycling",               7, "M"),
    47: ("Bio-based / compostable packaging",         8, "L"),
    49: ("Advanced waste sorting and recycling",      8, "L"),
    51: ("Low-carbon construction materials",         7, "L"),
    54: ("Thermal energy storage",                    8, "H"),
    55: ("Li-ion and next-generation batteries",      9, "H"),
    56: ("Grid-scale long-duration storage",          8, "H"),
    57: ("Battery storage systems (grid-connected)",  8, "H"),
    59: ("Smart grid / distributed energy",           8, "L"),
    60: ("Heat pumps (space heating)",                9, "H"),
    62: ("Efficient cooling and HVAC",                9, "H"),
}
cl_to_trl      = {cl: trl for cl, (_, trl, _) in MAPPING.items()}
LOW_CONF       = {cl for cl, (_, _, c) in MAPPING.items() if c == "L"}
print(f"  Matched clusters: {len(MAPPING)}  |  Low-conf: {sorted(LOW_CONF)}")

# ─────────────────────────────────────────────────────────────────────────
# 3.  Load data
# ─────────────────────────────────────────────────────────────────────────
print("\n── Loading data ──", flush=True)

nov = pd.read_csv("output/novelty/novelty_scores.csv")
co_cluster = {r["name"]: int(r["specter_cluster"]) for _, r in nov.iterrows()}
co_trl     = {co: cl_to_trl.get(cl) for co, cl in co_cluster.items()}

nat    = pd.read_csv("output/novelty_t/novelty_time_aware.csv")
co_os  = dict(zip(nat["Companies"], nat["overall_sparsity_t"]))

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl["year"] = dl["Deal Date"].dt.year
dl["co"]   = dl["Companies"].str.strip()
deals_v = dl.dropna(subset=["year", "Investors"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

first_round = (deals_v.sort_values("Deal Date")
               .groupby("co").first().reset_index()
               [["co","year","Investors",
                 "Company Country/Territory/Region","HQ Global Region"]])

# Investor embeddings
inv_names_raw = pd.read_csv("data/investor_names.csv", header=None, names=["investor"])
inv_emb_raw   = np.load("data/investor_embeddings.npy").astype(np.float32)
nrm = np.linalg.norm(inv_emb_raw, axis=1, keepdims=True); nrm[nrm==0]=1.0
inv_emb_raw  /= nrm
inv_names_ser = [str(n).strip() for n in inv_names_raw["investor"]
                 if str(n).strip() not in ("investor","nan","")]
inv_emb_map   = {n.lower(): inv_emb_raw[:len(inv_names_ser)][i]
                 for i, n in enumerate(inv_names_ser)}

# Startup embeddings
with open("output/novelty/novelty_specter.pkl","rb") as f:
    ns = pickle.load(f)
su_emb = np.array(ns["embs_raw"], dtype=np.float32)
sn = np.linalg.norm(su_emb, axis=1, keepdims=True); sn[sn==0]=1.0
su_emb /= sn
startup_emb_map = {str(n).strip(): su_emb[i] for i, n in enumerate(ns["names"])}

# Tuned communities (lowercase fix)
tuned_raw = pd.read_csv("output/h3_redesign/lagged_comm_tuned.csv")
tuned_lc  = {(r["investor"].lower(), int(r["year"])): int(r["community_tuned"])
             for _, r in tuned_raw.iterrows()}

# Investor geo
inv_raw = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors",
                                 "HQ Country/Territory/Region",
                                 "HQ Global Region"])
inv_raw["k"] = inv_raw["Investors"].str.strip().str.lower()
ic = dict(zip(inv_raw["k"], inv_raw["HQ Country/Territory/Region"].fillna("")))
ir = dict(zip(inv_raw["k"], inv_raw["HQ Global Region"].fillna("")))

print(f"  Investor embeddings: {len(inv_emb_map):,}  "
      f"Startup embeddings: {len(startup_emb_map):,}")

# ─────────────────────────────────────────────────────────────────────────
# 4.  Build lagged network objects
# ─────────────────────────────────────────────────────────────────────────
print("── Building network objects ──", flush=True)

year_active = defaultdict(set)
for _, row in deals_v.iterrows():
    for inv in comma_parse(row["Investors"]):
        year_active[int(row["year"])].add(inv.lower())

# Lagged degree and deal counts (built incrementally)
sorted_years      = sorted(deals_v["year"].unique())
inv_cumdeals      = defaultdict(int)
inv_cumdeg        = defaultdict(int)
lagged_degree     = {}
inv_deals_before  = {}
# first_coinv[(a,b)] = first year a and b appeared in the same deal (a<b lex)
first_coinv       = {}

current_adj = defaultdict(set)
for yr in sorted_years:
    lagged_degree[yr]    = dict(inv_cumdeg)
    inv_deals_before[yr] = dict(inv_cumdeals)
    for _, row in deals_v[deals_v["year"] == yr].iterrows():
        invs = [x.lower() for x in comma_parse(row["Investors"])]
        for a in invs:
            inv_cumdeals[a] += 1
        for a, b in combinations(invs, 2):
            if b not in current_adj[a]:
                inv_cumdeg[a] += 1; inv_cumdeg[b] += 1
            current_adj[a].add(b); current_adj[b].add(a)
            key = (a, b) if a < b else (b, a)
            if key not in first_coinv:
                first_coinv[key] = yr

# Plurality community per startup (for same_community_tuned)
co_lc = {}
for _, fr in first_round.iterrows():
    co = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None: continue
    comms = [tuned_lc.get((i.lower(), yr))
             for i in comma_parse(str(fr["Investors"]))
             if tuned_lc.get((i.lower(), yr)) is not None]
    co_lc[co] = Counter(comms).most_common(1)[0][0] if comms else None

print("  Done.")

# ─────────────────────────────────────────────────────────────────────────
# 5.  Build dyad dataset
# ─────────────────────────────────────────────────────────────────────────
print("── Building dyad dataset ──", flush=True)
N_COUNTER = 5
RNG = np.random.default_rng(42)
rows = []

for _, fr in first_round.iterrows():
    co  = fr["co"]; yr = int(fr["year"]) if pd.notna(fr["year"]) else None
    if yr is None or co not in startup_emb_map: continue
    trl_v = co_trl.get(co)
    if trl_v is None: continue          # H6 subsample only

    os_v  = co_os.get(co)
    cl_v  = co_cluster.get(co, -1)
    investors = comma_parse(str(fr["Investors"]))
    if not investors: continue
    actual_lc = {i.lower() for i in investors}
    pool = [i for i in year_active.get(yr, set())
            if i not in actual_lc and i in inv_emb_map]
    if len(pool) < N_COUNTER: continue

    sv       = startup_emb_map[co]
    co_ctry  = str(fr.get("Company Country/Territory/Region","")).strip()
    co_rgn   = str(fr.get("HQ Global Region","")).strip()
    # has_prior_network: candidate has co-invested with any actual investor before yr

    def mkrow(inv_k, tied):
        if inv_k not in inv_emb_map: return None
        iv  = inv_emb_map[inv_k]
        sd  = float(1.0 - sv @ iv)
        sc  = 1.0 if (co_lc.get(co) is not None and
                      tuned_lc.get((inv_k, yr)) == co_lc[co]) else 0.0
        lpd = float(np.log1p(inv_deals_before.get(yr, {}).get(inv_k, 0)))
        ldt = float(np.log1p(lagged_degree.get(yr, {}).get(inv_k, 0)))
        gc  = 1.0 if (co_ctry and ic.get(inv_k,"") == co_ctry) else 0.0
        gr  = 1.0 if (co_rgn  and ir.get(inv_k,"") == co_rgn)  else 0.0
        hpn = 0.0
        for a in actual_lc:
            if a == inv_k: continue
            k2 = (inv_k, a) if inv_k < a else (a, inv_k)
            if first_coinv.get(k2, 9999) < yr:
                hpn = 1.0; break
        return dict(co=co, tied=tied, sem=sd, sc=sc, lpd=lpd, ldt=ldt,
                    gc=gc, gr=gr, hpn=hpn, trl=float(trl_v), os=os_v,
                    cl=cl_v, yr=yr)

    for inv in investors:
        r = mkrow(inv.lower(), 1)
        if r: rows.append(r)
    for fi in RNG.choice(pool, N_COUNTER, replace=False):
        r = mkrow(fi, 0)
        if r: rows.append(r)

dm = pd.DataFrame(rows).dropna(subset=["sem","trl"]).reset_index(drop=True)

# Standardise
for col, col_z in [("sem","sem_z"),("lpd","lpd_z"),("ldt","ldt_z"),("os","os_z")]:
    mu = dm[col].mean(); s = dm[col].std()
    dm[col_z] = (dm[col] - mu) / (s if s > 0 else 1.0)

# PC1 of (ldt_z, lpd_z)
pca = PCA(n_components=1)
dm["pc1_raw"] = pca.fit_transform(dm[["ldt_z","lpd_z"]].values)[:,0]
dm["pc1_z"]   = (dm["pc1_raw"] - dm["pc1_raw"].mean()) / dm["pc1_raw"].std()

# TRL standardised + interactions
trl_mu = dm["trl"].mean(); trl_sd = dm["trl"].std()
dm["trl_z"]      = (dm["trl"] - trl_mu) / (trl_sd if trl_sd>0 else 1.0)
dm["sem_x_trl"]  = dm["sem_z"] * dm["trl_z"]
dm["sem_x_os"]   = dm["sem_z"] * dm["os_z"]

# TRL categorical (4-6=1, 7-8=2, 9-11=3)
dm["trl_cat"] = pd.cut(dm["trl"], bins=[0,6,8,12],
                        labels=[1,2,3]).astype(float)
cat_mu = dm["trl_cat"].mean(); cat_sd = dm["trl_cat"].std()
dm["trl_cat_z"]     = (dm["trl_cat"] - cat_mu) / (cat_sd if cat_sd>0 else 1.0)
dm["sem_x_trlcat"]  = dm["sem_z"] * dm["trl_cat_z"]

# Valid strata
ok = dm.groupby("co")["tied"].transform(lambda x: x.nunique() == 2)
dm = dm[ok].reset_index(drop=True)

print(f"  Dyads: {len(dm):,}  |  Strata: {dm['co'].nunique():,}  "
      f"|  Tech clusters: {dm['cl'].nunique()}")
print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  has_prior_network: {dm['hpn'].mean():.1%}")
print(f"  TRL mean={dm['trl'].mean():.2f}  sd={dm['trl'].std():.2f}  "
      f"distinct={sorted(dm['trl'].astype(int).unique())}")

# ─────────────────────────────────────────────────────────────────────────
# 6.  Model utilities
# ─────────────────────────────────────────────────────────────────────────
INTERACT_IDX = 1   # position of the primary interaction (trl_z dropped — absorbed by startup FE)

def fit_clogit(sub, xvars):
    s  = sub[["tied","co","cl"] + xvars].dropna().copy()
    ok = s.groupby("co")["tied"].transform(lambda x: x.nunique() == 2)
    s  = s[ok].reset_index(drop=True)
    en = s["tied"].values.astype(float)
    ex = s[xvars].values.astype(float)
    gr = s["co"].values
    mk = np.isfinite(ex).all(axis=1)
    for mth in ["bfgs","newton","lbfgs"]:
        try:
            res = ConditionalLogit(en[mk], ex[mk],
                                   groups=gr[mk]).fit(
                method=mth, maxiter=800, disp=False)
            if np.isfinite(res.bse).all():
                return res, s[mk].reset_index(drop=True)
        except Exception:
            continue
    return None, None


def compute_cluster_se_and_G(result, sub_m, xvars):
    """
    Sandwich cluster SE and per-cluster score sums G_c.
    Scores approximated as (y_i - p_i)*x_i with p_i = softmax(eta) within stratum.
    Exact for k_s=1; good approximation for k_s>1.
    """
    params   = result.params
    endog    = sub_m["tied"].values.astype(float)
    exog     = sub_m[xvars].values.astype(float)
    grps     = sub_m["co"].values
    cls      = sub_m["cl"].values

    # Per-stratum softmax conditional probabilities
    lin_pred = exog @ params
    p_cond   = np.zeros(len(endog))
    for g in np.unique(grps):
        mk    = grps == g
        eta   = lin_pred[mk] - lin_pred[mk].max()
        p_cond[mk] = np.exp(eta) / np.exp(eta).sum()

    scores = (endog - p_cond)[:, None] * exog  # (n, k)

    # Aggregate to tech-cluster level
    unique_cls = sorted(set(cls))
    G = {c: scores[cls == c].sum(axis=0) for c in unique_cls}

    meat  = sum(np.outer(g, g) for g in G.values())
    bread = result.cov_params()
    V_cl  = bread @ meat @ bread
    se_cl = np.sqrt(np.maximum(0.0, np.diag(V_cl)))
    return se_cl, G, V_cl


def wild_bootstrap_p(result, G_dict, interact_idx, n_boot=999, seed=2026):
    """
    Linearised wild cluster bootstrap (Webb 6-point weights).
    p = P(|beta* - mean(beta*)| >= |beta_hat|) — null imposed by centering.
    """
    params = result.params
    cov    = result.cov_params()
    G_mat  = np.array([G_dict[c] for c in sorted(G_dict)])  # (n_cl, k)
    n_cl   = len(G_mat)
    webb   = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                        np.sqrt(0.5),  1.0,  np.sqrt(1.5)])
    rng_b  = np.random.default_rng(seed)
    boots  = np.empty(n_boot)
    for b in range(n_boot):
        eps      = rng_b.choice(webb, size=n_cl)
        perturb  = eps @ G_mat
        boots[b] = (params + cov @ perturb)[interact_idx]
    b_hat = params[interact_idx]
    # impose null by centering bootstrap distribution
    d = boots - boots.mean()
    p = float(np.mean(np.abs(d) >= abs(b_hat)))
    return p, boots


def print_model(result, xvars, label=""):
    print(f"  {label}")
    for i, v in enumerate(xvars):
        b = result.params[i]; se = result.bse[i]; p = result.pvalues[i]
        stars = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
        print(f"    {v:<18} β={b:+.4f}  SE={se:.4f}  p={p:.4f}  {stars}")


# ─────────────────────────────────────────────────────────────────────────
# 7.  Primary model
# ─────────────────────────────────────────────────────────────────────────
print("\n── PRIMARY: sem_dist_z × trl_z ──")
# trl_z excluded: constant within startup strata → absorbed by startup FE → near-singular SE=670
# Only the interaction sem_x_trl is identified (sem_z varies by candidate investor)
XVARS_PRI = ["sem_z","sem_x_trl","sc","gc","gr","pc1_z","hpn"]
res_pri, sub_pri = fit_clogit(dm, XVARS_PRI)

if res_pri is None:
    print("  ERROR: primary model failed."); sys.exit(1)

print_model(res_pri, XVARS_PRI, "Standard SEs:")

print("\n  Computing cluster-robust SEs (sandwich)…")
se_cl, G, V_cl = compute_cluster_se_and_G(res_pri, sub_pri, XVARS_PRI)

print(f"\n  Cluster-robust SEs ({len(G)} tech clusters):")
for i, v in enumerate(XVARS_PRI):
    b = res_pri.params[i]; sc_ = se_cl[i]
    p = 2*(1-norm.cdf(abs(b/sc_))) if sc_>0 else np.nan
    stars = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
    print(f"    {v:<18} β={b:+.4f}  SE_cl={sc_:.4f}  p_cl={p:.4f}  {stars}")

print("\n  Running wild cluster bootstrap (Webb, 999 reps)…")
p_boot, beta_boots = wild_bootstrap_p(res_pri, G, INTERACT_IDX, n_boot=999)

b_hat     = res_pri.params[INTERACT_IDX]
se_std    = res_pri.bse[INTERACT_IDX]
se_cl_int = se_cl[INTERACT_IDX]
p_std     = res_pri.pvalues[INTERACT_IDX]
p_cl_int  = 2*(1-norm.cdf(abs(b_hat/se_cl_int))) if se_cl_int>0 else np.nan
boot_se   = float(beta_boots.std())

print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  H6 PRIMARY RESULT  (sem_dist_z × trl_z)                    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  β                   =  {b_hat:+.4f}                           ║
  ║  SE  (standard)      =  {se_std:.4f}                           ║
  ║  SE  (cluster)       =  {se_cl_int:.4f}                           ║
  ║  p   (standard)      =  {p_std:.4f}                           ║
  ║  p   (cluster)       =  {p_cl_int:.4f}                           ║
  ║  p   (wild bootstrap)=  {p_boot:.4f}                           ║
  ║  Bootstrap SE        =  {boot_se:.4f}                           ║
  ╚══════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────────────────────
# 8.  Secondary: add sem_dist_z × overall_sparsity_t
# ─────────────────────────────────────────────────────────────────────────
print("── SECONDARY: add sem_dist_z × overall_sparsity_t ──")
XVARS_SEC = ["sem_z","sem_x_trl","sem_x_os","sc","gc","gr","pc1_z","hpn"]
res_sec, _ = fit_clogit(dm.dropna(subset=["os_z"]), XVARS_SEC)

if res_sec is not None:
    print_model(res_sec, XVARS_SEC, "Secondary (standard SEs):")
    b_trl_sec = res_sec.params[1]  # sem_x_trl (trl_z removed from spec)
    b_os_sec  = res_sec.params[2]  # sem_x_os
    print(f"\n  sem_x_trl:  {b_hat:+.4f} → {b_trl_sec:+.4f}  "
          f"(Δ = {b_trl_sec - b_hat:+.4f})")
    print(f"  sem_x_os:   β={b_os_sec:+.4f}  SE={res_sec.bse[2]:.4f}  "
          f"p={res_sec.pvalues[2]:.4f}")
else:
    print("  WARNING: secondary model did not converge.")
    b_trl_sec = b_os_sec = np.nan

# ─────────────────────────────────────────────────────────────────────────
# 9.  Robustness
# ─────────────────────────────────────────────────────────────────────────
print("\n── ROBUSTNESS ──")

# R-i: LOCO (single pass — collect beta + p)
print("\n  R-i: Leave-one-cluster-out…")
all_cls    = sorted(dm["cl"].unique())
loco_betas = {}
loco_pvals = {}
for cl_drop in all_cls:
    r_loco, _ = fit_clogit(dm[dm["cl"] != cl_drop], XVARS_PRI)
    if r_loco is not None:
        loco_betas[cl_drop] = float(r_loco.params[INTERACT_IDX])
        loco_pvals[cl_drop] = float(r_loco.pvalues[INTERACT_IDX])

if loco_betas:
    vals      = list(loco_betas.values())
    inf_cl    = max(loco_betas, key=lambda c: abs(loco_betas[c] - b_hat))
    n_sig     = sum(1 for p in loco_pvals.values() if p < 0.05)
    print(f"    β range: [{min(vals):+.4f}, {max(vals):+.4f}]  (full: {b_hat:+.4f})")
    print(f"    Most influential: CL{inf_cl}  β_without={loco_betas[inf_cl]:+.4f}  "
          f"Δ={loco_betas[inf_cl]-b_hat:+.4f}")
    print(f"    All positive:  {all(v>0 for v in vals)}")
    print(f"    Significant (std SE, p<0.05) in {n_sig}/{len(all_cls)} LOCO variants")

# R-ii: Drop cluster 0 (nuclear/fusion, TRL=4)
print("\n  R-ii: Drop CL 0 (nuclear/fusion)…")
res_nc0, _ = fit_clogit(dm[dm["cl"] != 0], XVARS_PRI)
if res_nc0:
    b0 = res_nc0.params[INTERACT_IDX]
    print(f"    β={b0:+.4f}  SE={res_nc0.bse[INTERACT_IDX]:.4f}  "
          f"p={res_nc0.pvalues[INTERACT_IDX]:.4f}")
else:
    b0 = np.nan; print("    WARNING: did not converge.")

# R-iii: Drop low-confidence clusters
print(f"\n  R-iii: Drop low-confidence clusters {sorted(LOW_CONF)}…")
res_hc, _ = fit_clogit(dm[~dm["cl"].isin(LOW_CONF)], XVARS_PRI)
if res_hc:
    bhc = res_hc.params[INTERACT_IDX]
    print(f"    β={bhc:+.4f}  SE={res_hc.bse[INTERACT_IDX]:.4f}  "
          f"p={res_hc.pvalues[INTERACT_IDX]:.4f}")
else:
    bhc = np.nan; print("    WARNING: did not converge.")

# R-iv: TRL as 3 ordered categories
print("\n  R-iv: TRL as 3 categories (4–6=1, 7–8=2, 9–11=3)…")
XVARS_CAT = ["sem_z","sem_x_trlcat","sc","gc","gr","pc1_z","hpn"]
res_cat, _ = fit_clogit(dm.dropna(subset=["trl_cat"]), XVARS_CAT)
if res_cat:
    bcat = res_cat.params[INTERACT_IDX]
    print(f"    β={bcat:+.4f}  SE={res_cat.bse[INTERACT_IDX]:.4f}  "
          f"p={res_cat.pvalues[INTERACT_IDX]:.4f}")
else:
    bcat = np.nan; print("    WARNING: did not converge.")

# ─────────────────────────────────────────────────────────────────────────
# 10.  Summary table + verdict
# ─────────────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  H6 RESULTS SUMMARY  (sem_dist_z × TRL_interaction, β and SE_std)  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Specification                β       SE_std  p_std  p_cl    p_boot ║
╠══════════════════════════════════════════════════════════════════════╣
║  Primary (TRL continuous)  {b_hat:+.4f}  {se_std:.4f}  {p_std:.4f}  {p_cl_int:.4f}  {p_boot:.4f} ║
║  Drop CL 0 (nuclear)       {b0:+.4f}  {res_nc0.bse[2] if res_nc0 else np.nan:.4f}  {res_nc0.pvalues[2] if res_nc0 else np.nan:.4f}  —       —       ║
║  High-conf only            {bhc:+.4f}  {res_hc.bse[2] if res_hc else np.nan:.4f}  {res_hc.pvalues[2] if res_hc else np.nan:.4f}  —       —       ║
║  TRL categorical (3 cats)  {bcat:+.4f}  {res_cat.bse[2] if res_cat else np.nan:.4f}  {res_cat.pvalues[2] if res_cat else np.nan:.4f}  —       —       ║""")
if loco_betas:
    print(f"║  LOCO range                [{min(vals):+.4f}, {max(vals):+.4f}]  all>0: {all(v>0 for v in vals)}  ║")
print("╚══════════════════════════════════════════════════════════════════════╝")

print("\n── VERDICT ──")
if p_boot < 0.05:
    verdict = ("SUPPORTED at 5% (wild bootstrap). "
               "The expertise-barrier penalty is significantly attenuated at higher TRL. "
               "H6 confirmed.")
elif p_boot < 0.10:
    verdict = ("MARGINAL (wild bootstrap p < 0.10). "
               "Direction consistent with H6 but below 5% threshold. "
               "Result is suggestive, not confirmatory.")
elif b_hat > 0:
    verdict = ("DIRECTIONAL ONLY. β > 0 throughout, but wild bootstrap p >= 0.10. "
               "H6 not confirmed at conventional levels.")
else:
    verdict = "NOT SUPPORTED. β wrong sign or insignificant."

print(f"\n  {verdict}")
print(f"\n  sem_dist_z (base effect): β={res_pri.params[0]:+.4f}  SE={res_pri.bse[0]:.4f}")
print(f"  sem_dist_z × trl_z:      β={b_hat:+.4f}  SE_std={se_std:.4f}  "
      f"SE_cl={se_cl_int:.4f}  p_boot={p_boot:.4f}")

# Secondary note
if res_sec is not None:
    print(f"\n  Secondary: sem_x_os β={b_os_sec:+.4f}  p={res_sec.pvalues[2]:.4f}  "
          f"| sem_x_trl after adding OS: β={b_trl_sec:+.4f}")

# ─────────────────────────────────────────────────────────────────────────
# 11.  Save
# ─────────────────────────────────────────────────────────────────────────
specs = ["Primary","Drop CL 0","High-conf only","TRL categorical"]
betas = [b_hat, b0, bhc, bcat]
se_s  = [res_pri.bse[2],
         res_nc0.bse[2] if res_nc0 else np.nan,
         res_hc.bse[2]  if res_hc  else np.nan,
         res_cat.bse[2] if res_cat else np.nan]
ps    = [res_pri.pvalues[2],
         res_nc0.pvalues[2] if res_nc0 else np.nan,
         res_hc.pvalues[2]  if res_hc  else np.nan,
         res_cat.pvalues[2] if res_cat else np.nan]
extra = [p_boot, np.nan, np.nan, np.nan]

pd.DataFrame({"spec": specs, "beta": betas, "se_std": se_s,
              "p_std": ps, "p_wild_boot": extra}).to_csv(
    f"{OUT_DIR}/h6_results.csv", index=False)

print(f"\n  Results saved → {OUT_DIR}/h6_results.csv")
print("=" * 70)
print("SCRIPT 92 COMPLETE")
print("=" * 70)
