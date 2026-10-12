"""
Script 78: Scrutinise std_distinct_resid → failure
====================================================
Four checks:
  1. Exact p-values + Holm/BH across the three novelty axes
  2. Cohort test: first-deal years 2014-2016 / 2017-2020 / 2021+
  3. Syndicate-size stratification: n_invs = 1, 2, 3-4, 5-9, 10+
  4. Curvature (SDR²) + moderation by cluster dispersion (Haans heterogeneity)
"""
import re, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."
def banner(s): print(f"\n{'='*72}\n{s}\n{'='*72}")

# ── HC1 OLS ──────────────────────────────────────────────────
def ols_hc1(y, X):
    N, k = X.shape
    try: beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except: return None
    res     = y - X @ beta
    XtXinv  = np.linalg.pinv(X.T @ X)
    meat    = (X * res[:,None]).T @ (X * res[:,None]) * N / (N-k)
    V       = XtXinv @ meat @ XtXinv
    ses     = np.sqrt(np.clip(np.diag(V), 0, None))
    ts      = beta / np.where(ses > 1e-20, ses, np.nan)
    ps      = np.where(~np.isnan(ts), f_dist.sf(ts**2, 1, N-k), np.nan)
    return beta, ses, ps

def run_ols(df_sub, focal_vars, outcome="failed"):
    """OLS with year+country FE; returns (N, beta_focal, se_focal, p_focal) for each focal var."""
    cols = focal_vars + [outcome, "log_capital", "n_rounds", "deal_year", "hq_country"]
    sub = df_sub[cols].dropna().copy()
    if len(sub) < 50: return None
    y   = sub[outcome].values.astype(float)
    yr  = pd.get_dummies(sub["deal_year"],  prefix="yr", drop_first=True).astype(float)
    co  = pd.get_dummies(sub["hq_country"], prefix="co", drop_first=True).astype(float)
    X   = np.column_stack([np.ones(len(sub)),
                           sub[["log_capital","n_rounds"]].values.astype(float),
                           yr.values, co.values,
                           sub[focal_vars].values.astype(float)])
    out = ols_hc1(y, X)
    if out is None: return None
    beta, ses, ps = out
    n_f = len(focal_vars)
    return {"N": len(y),
            "beta": beta[-n_f:], "se": ses[-n_f:], "p": ps[-n_f:]}

def fmt_row(r, vi=0):
    if r is None: return f"{'—':>36}"
    b, se, p = r["beta"][vi], r["se"][vi], r["p"][vi]
    return f"β={b:+.4f} SE={se:.4f} p={p:.4f} {sig(p)}"

# ── Holm and BH helpers ──────────────────────────────────────
def holm(ps):
    m = len(ps); order = np.argsort(ps)
    adj = np.empty(m); adj[order] = [min(1, ps[order[i]]*(m-i)) for i in range(m)]
    for i in range(1, m): adj[order[i]] = max(adj[order[i]], adj[order[i-1]])
    return adj

def bh(ps, alpha=0.05):
    m = len(ps); order = np.argsort(ps)
    adj = np.empty(m)
    for rank, i in enumerate(order, 1): adj[i] = min(1, ps[i]*m/rank)
    for i in range(m-2, -1, -1): adj[order[i]] = min(adj[order[i]], adj[order[i+1]])
    return adj

# ── LOAD DATA ────────────────────────────────────────────────
banner("LOADING & MERGING DATA")

nf  = pd.read_csv("output/two_level_novelty/novelty_final.csv")
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
nf["co"]  = nf["name"].str.strip()
cdf["co"] = cdf["Companies"].str.strip()
df = cdf.merge(nf, on="co", how="inner", suffixes=("","_nov"))
if "std_distinct_nov" in df.columns: df["std_distinct"] = df["std_distinct_nov"]

# n_invs and first_year from deals
from load_data import load_deals
def comma_parse(s):
    if pd.isna(s) or not str(s).strip(): return []
    return [re.sub(r"\s*\(.*?\)\s*$","",p).strip()
            for p in re.split(r",\s*|\n", str(s)) if p.strip()]
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()
deals["Deal Date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
first_round = (deals.sort_values("Deal Date")
               .groupby("co").first().reset_index()[["co","Investors","Deal Date"]])
first_round["n_invs"]     = first_round["Investors"].apply(
    lambda x: len(comma_parse(str(x))))
first_round["first_year"] = first_round["Deal Date"].dt.year
df = df.merge(first_round[["co","n_invs","first_year"]], on="co", how="left")

# Cluster dispersion (Haans heterogeneity) via k=60 on cleaned embeddings
print("  Building cluster dispersion (k=60)...", end=" ", flush=True)
with open("output/novelty/novelty_specter.pkl","rb") as fh:
    ns_pkl = pickle.load(fh)
all_names = [n.strip() for n in ns_pkl["names"]]
all_embs  = normalize(np.array(ns_pkl["embs_raw"], dtype=np.float32), norm="l2")

# use the cleaned set (drop boilerplate from Script 75)
clean_df = pd.read_csv("output/two_level_novelty/density_novelty_clean.csv")
clean_names_set = set(clean_df["name"].values)
keep_mask = np.array([n in clean_names_set for n in all_names])
embs_c  = all_embs[keep_mask]
names_c = [n for n, k in zip(all_names, keep_mask) if k]

km60 = KMeans(n_clusters=60, random_state=42, n_init=10, max_iter=500)
c_labels = km60.fit_predict(embs_c)
# raw_distinct per startup
centroids = np.zeros((60, embs_c.shape[1]), dtype=np.float32)
for ci in range(60):
    mask = c_labels == ci
    c = embs_c[mask].mean(axis=0)
    centroids[ci] = c / max(np.linalg.norm(c), 1e-10)
raw_dist = np.array([1.0 - float(embs_c[i] @ centroids[c_labels[i]])
                     for i in range(len(embs_c))])
# per-cluster dispersion (SD of raw_distinct)
clust_disp = {}
for ci in range(60):
    mask = c_labels == ci
    clust_disp[ci] = raw_dist[mask].std()
cluster_disp_arr = np.array([clust_disp[c_labels[i]] for i in range(len(embs_c))])

# map back to startups
name_to_disp = dict(zip(names_c, cluster_disp_arr))
df["cluster_disp"] = df["co"].map(name_to_disp)
print("done")

# Residualise std_distinct on overall_sparsity
valid_sd = ~(np.isnan(df["std_distinct"]) | np.isnan(df["overall_sparsity"]))
X_res = np.column_stack([np.ones(valid_sd.sum()),
                         df.loc[valid_sd,"overall_sparsity"].values])
y_res = df.loc[valid_sd,"std_distinct"].values
b_res = np.linalg.lstsq(X_res, y_res, rcond=None)[0]
df.loc[valid_sd,"sdr"] = y_res - X_res @ b_res

print(f"  Analysis sample: N={len(df):,}  failure rate={df['failed'].mean():.3f}")
print(f"  cluster_disp: mean={df['cluster_disp'].mean():.4f}  sd={df['cluster_disp'].std():.4f}")

# ── STEP 1: EXACT P-VALUES + MULTIPLE-TESTING CORRECTION ────
banner("STEP 1: EXACT p-VALUES + HOLM / BH ACROSS THREE AXES")

axes = [("overall_sparsity",["overall_sparsity"]),
        ("niche_cohesion",  ["niche_cohesion"]),
        ("sdr",             ["sdr"])]

raw_ps = []
print(f"\n  {'Axis':<25} {'β':>9} {'SE':>8} {'p_raw':>9}  Holm    BH")
print(f"  {'-'*65}")
for label, fv in axes:
    r = run_ols(df, fv)
    if r:
        b, se, p = r["beta"][0], r["se"][0], r["p"][0]
        raw_ps.append((label, b, se, p, r["N"]))
    else:
        raw_ps.append((label, np.nan, np.nan, np.nan, 0))

ps_arr = np.array([x[3] for x in raw_ps])
holm_ps = holm(ps_arr)
bh_ps   = bh(ps_arr)

for i, (label, b, se, p, n) in enumerate(raw_ps):
    if np.isnan(p):
        print(f"  {label:<25} {'—':>9}")
        continue
    print(f"  {label:<25} {b:>+9.5f} {se:>8.5f} {p:>9.5f}  "
          f"{holm_ps[i]:.5f} {sig(holm_ps[i])}  {bh_ps[i]:.5f} {sig(bh_ps[i])}  "
          f"(N={n:,})")

# All three cuts for sdr
print(f"\n  sdr exact p across sample cuts:")
os_p99 = np.percentile(df["overall_sparsity"].dropna(), 99)
os_p98 = np.percentile(df["overall_sparsity"].dropna(), 98)
nc_p01 = np.percentile(df["niche_cohesion"].dropna(),    1)
for label, sub in [
    ("(a) full",          df),
    ("(b) excl OS>p99",   df[df["overall_sparsity"] <= os_p99]),
    ("(c) excl OS>p98 & NC<p01",
     df[(df["overall_sparsity"]<=os_p98)&(df["niche_cohesion"]>=nc_p01)]),
]:
    r = run_ols(sub, ["sdr"])
    if r:
        b,se,p = r["beta"][0],r["se"][0],r["p"][0]
        print(f"  {label:<35} β={b:+.5f}  SE={se:.5f}  p={p:.5f} {sig(p)}  N={r['N']:,}")

# ── STEP 2: COHORT TEST ──────────────────────────────────────
banner("STEP 2: COHORT TEST (description-convergence check)")
print("  Hypothesis: if older cohorts are more resolved (success/failure decided),")
print("  a spurious effect concentrates there. True effect should appear in recent cohorts too.")

cohorts = [
    ("2014-2016", (df["first_year"]>=2014)&(df["first_year"]<=2016)),
    ("2017-2020", (df["first_year"]>=2017)&(df["first_year"]<=2020)),
    ("2021+",     df["first_year"]>=2021),
]
print(f"\n  {'Cohort':<15} {'N':>6} {'fail%':>7} {'β(sdr)':>9} {'SE':>8} {'p':>9}")
print(f"  {'-'*60}")
for label, mask in cohorts:
    sub = df[mask]
    n_total = mask.sum()
    fail_r  = sub["failed"].mean() if n_total > 0 else np.nan
    r = run_ols(sub, ["sdr"])
    if r:
        b,se,p = r["beta"][0],r["se"][0],r["p"][0]
        print(f"  {label:<15} {r['N']:>6} {fail_r:>6.1%} {b:>+9.4f} {se:>8.4f} {p:>9.4f} {sig(p)}")
    else:
        print(f"  {label:<15} {n_total:>6} {fail_r:>6.1%}  (too few obs)")

# ── STEP 3: SYNDICATE-SIZE STRATIFICATION ───────────────────
banner("STEP 3: SYNDICATE-SIZE STRATIFICATION")
print("  Cross_community collapsed within N=2 stratum; check if sdr does the same.")

strata = [
    ("N=1",   df["n_invs"]==1),
    ("N=2",   df["n_invs"]==2),
    ("N=3-4", (df["n_invs"]>=3)&(df["n_invs"]<=4)),
    ("N=5-9", (df["n_invs"]>=5)&(df["n_invs"]<=9)),
    ("N=10+", df["n_invs"]>=10),
]
print(f"\n  {'Stratum':<10} {'N':>6} {'fail%':>7} {'β(sdr)':>9} {'SE':>8} {'p':>9}  MDE")
print(f"  {'-'*70}")
for label, mask in strata:
    sub = df[mask]
    n_total = mask.sum()
    fail_r  = sub["failed"].mean() if n_total > 0 else np.nan
    r = run_ols(sub, ["sdr"])
    if r:
        b,se,p = r["beta"][0],r["se"][0],r["p"][0]
        mde = 1.645*se + 0.842*se   # 80% power, one-sided
        print(f"  {label:<10} {r['N']:>6} {fail_r:>6.1%} {b:>+9.4f} {se:>8.4f} {p:>9.4f} {sig(p)}  "
              f"MDE≈{mde:.4f}")
    else:
        print(f"  {label:<10} {n_total:>6} {fail_r:>6.1%}  (too few obs for FE)")

# ── STEP 4: CURVATURE AND MODERATION ────────────────────────
banner("STEP 4: CURVATURE AND MODERATION BY CLUSTER DISPERSION")

df["sdr2"]       = df["sdr"]**2
df["sdr_z"]      = (df["sdr"] - df["sdr"].mean()) / df["sdr"].std()
df["cdisp_z"]    = (df["cluster_disp"] - df["cluster_disp"].mean()) / df["cluster_disp"].std()
df["sdr_x_disp"] = df["sdr_z"] * df["cdisp_z"]

print(f"\n  4a. Curvature: sdr + sdr²")
r_curve = run_ols(df, ["sdr","sdr2"])
if r_curve:
    for vi, var in enumerate(["sdr","sdr²"]):
        b,se,p = r_curve["beta"][vi],r_curve["se"][vi],r_curve["p"][vi]
        print(f"    {var:<10} β={b:+.5f}  SE={se:.5f}  p={p:.5f} {sig(p)}  N={r_curve['N']:,}")
    # turning point
    b_lin, b_sq = r_curve["beta"][0], r_curve["beta"][1]
    if abs(b_sq) > 1e-10:
        tp = -b_lin / (2*b_sq)
        print(f"    Turning point: sdr = {tp:.3f}  "
              f"({'within observed range' if df['sdr'].quantile(.05)<tp<df['sdr'].quantile(.95) else 'outside p5-p95'})")

print(f"\n  4b. Moderation by cluster dispersion (Haans category heterogeneity)")
print(f"       cluster_disp: mean={df['cluster_disp'].mean():.4f}  sd={df['cluster_disp'].std():.4f}")
r_mod = run_ols(df, ["sdr_z","cdisp_z","sdr_x_disp"])
if r_mod:
    for vi, var in enumerate(["sdr_z","cdisp_z","sdr_z × cdisp_z"]):
        b,se,p = r_mod["beta"][vi],r_mod["se"][vi],r_mod["p"][vi]
        print(f"    {var:<22} β={b:+.5f}  SE={se:.5f}  p={p:.5f} {sig(p)}")
    print(f"    N={r_mod['N']:,}")
    if r_mod["p"][2] < 0.10:
        print(f"    Interaction significant → sdr effect depends on cluster dispersion.")
        print(f"    Tight clusters (low cdisp): sdr main effect at cdisp_z=−1: "
              f"β≈{r_mod['beta'][0]+r_mod['beta'][2]*(-1):+.4f}")
        print(f"    Loose clusters (high cdisp): sdr main effect at cdisp_z=+1: "
              f"β≈{r_mod['beta'][0]+r_mod['beta'][2]*(+1):+.4f}")
    else:
        print(f"    Interaction NOT significant → sdr effect does not depend on cluster dispersion.")

# ── OVERALL VERDICT ──────────────────────────────────────────
banner("VERDICT")
sdr_full_p = raw_ps[2][3] if len(raw_ps)>2 else np.nan
sdr_holm_p = holm_ps[2]   if len(raw_ps)>2 else np.nan
print(f"  sdr raw p     = {sdr_full_p:.5f} {sig(sdr_full_p)}")
print(f"  sdr Holm p    = {sdr_holm_p:.5f} {sig(sdr_holm_p)}")
print(f"  sdr BH p      = {bh_ps[2]:.5f}   {sig(bh_ps[2])}")
print(f"  Cohort: effect present in 2021+ cohort?  (see table above)")
print(f"  Syndicate: effect survives N=1 stratum?  (see table above)")
print(f"  Curvature: significant quadratic?        (see table above)")
print(f"  Moderation: cluster dispersion matters?  (see table above)")
