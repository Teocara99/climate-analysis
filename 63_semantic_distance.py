"""
Script 63: Semantic distance between home and bridge communities as a predictor
of startup failure.

PRE-SPECIFIED PREDICTIONS (registered before running):
  P1: comm_distance (home ↔ bridge community semantic distance) → LOWER failure
      Hypothesis: greater conceptual distance means more novel information
      crosses the bridge → lower failure. Expected β < 0.
  P2: bridge_expertise (cosine similarity of bridge centroid to startup embedding)
      → LOWER failure when bridge community is LESS expert in the startup domain.
      A bridge from an orthogonal domain brings diverse resources.
      Expected β < 0 (lower expertise = lower failure).

SAMPLE: cross-community companies only, restricted to those where BOTH home and
bridge community have ≥10 embedded investors in SPECTER2.
"""

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.spatial.distance import cosine as scipy_cosine
from collections import Counter

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT  = BASE / "output" / "semantic_distance"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()


def elapsed():
    return f"{time.time() - t0:.1f}s"


def banner(s):
    print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")


def parse_investors(s):
    if pd.isna(s):
        return []
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip().lower() for p in parts if p.strip()]


def cosine_sim(a, b):
    """Cosine similarity; returns NaN if either vector is zero-length."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


# ════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════
banner("LOADING DATA")

# SPECTER2 embeddings (14409 × 768)
emb_matrix = np.load(BASE / "data" / "investor_embeddings.npy")        # float32
inv_names   = pd.read_csv(BASE / "data" / "investor_names.csv")["investor"].str.lower()
emb_map     = {name: emb_matrix[i] for i, name in enumerate(inv_names)}  # lowercase → vec

# Network community assignments
nm       = pd.read_csv(BASE / "output" / "network" / "network_metrics.csv")
nm_lower = nm.copy()
nm_lower["inv_lower"] = nm_lower["investor"].str.lower()
inv_comm = nm_lower.set_index("inv_lower")["community"].to_dict()        # lowercase → community

# Company dataset
cdf = pd.read_csv(BASE / "output" / "cpu" / "cpu_company_dataset.csv")

# Deals (for round-level investor identification)
sys.path.insert(0, str(BASE))
from load_data import load_deals
deals = load_deals()
deals = deals[deals["Deal Date"].notna() & (deals["Deal Date"].dt.year >= 2012)].copy()
deals["ymd"] = deals["Deal Date"].dt.date

print(f"Embeddings: {emb_matrix.shape}  |  investors with community: {len(inv_comm)}")
print(f"Company dataset: {cdf.shape}  |  Cross-community: {(cdf['cross_community']==1).sum()}")
print(f"Deals: {deals.shape}  [{elapsed()}]")

# ════════════════════════════════════════════════════════════════════
# STEP 1: COMMUNITY CENTROIDS FROM SPECTER2
# ════════════════════════════════════════════════════════════════════
banner("STEP 1 — COMMUNITY CENTROIDS (SPECTER2)")

# For each community, collect embeddings of members that have SPECTER2 vectors
community_vecs = {}   # community_id → list of embedding vectors
for inv, comm in inv_comm.items():
    vec = emb_map.get(inv)
    if vec is not None:
        community_vecs.setdefault(comm, []).append(vec)

# Report coverage
print(f"{'Comm':>6} {'N_embed':>8} {'Eligible':>8}")
n_eligible = 0
eligible_comms = set()
for comm in sorted(community_vecs):
    n = len(community_vecs[comm])
    ok = "YES" if n >= 10 else "NO "
    if n >= 10:
        n_eligible += 1
        eligible_comms.add(comm)
    print(f"  {comm:>4}   {n:>6}   {ok}")

print(f"\nCommunities with ≥10 embedded members: {n_eligible} / 46")

# Build centroids (L2-normalised mean)
community_centroids = {}
for comm in eligible_comms:
    vecs = np.stack(community_vecs[comm], axis=0)
    centroid = vecs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        community_centroids[comm] = centroid / norm   # unit-normalised

print(f"Centroids built for {len(community_centroids)} communities  [{elapsed()}]")

# ════════════════════════════════════════════════════════════════════
# STEP 2: IDENTIFY HOME vs BRIDGE COMMUNITY PER COMPANY
# ════════════════════════════════════════════════════════════════════
banner("STEP 2 — HOME and BRIDGE COMMUNITY IDENTIFICATION")

# Group deals by company, sorted by date
company_deals = (
    deals[deals["Investors"].notna()]
    .groupby("Companies", group_keys=False)
    .apply(lambda d: d.sort_values("Deal Date"))
)

records = []
n_no_deals = 0
n_no_round1_comm = 0
n_no_bridge = 0
n_ok = 0

cross_companies = set(cdf[cdf["cross_community"] == 1]["Companies"])

for company, grp in company_deals.groupby("Companies"):
    if company not in cross_companies:
        continue

    grp = grp.sort_values("Deal Date")
    first_date = grp["Deal Date"].iloc[0]

    # Round 1 = all deals within 30 days of the first deal
    round1 = grp[grp["Deal Date"] <= first_date + pd.Timedelta(days=30)]
    r1_investors = []
    for inv_str in round1["Investors"]:
        r1_investors.extend(parse_investors(inv_str))

    # Home community = modal community of round-1 investors that have known community
    r1_comms = [inv_comm.get(i) for i in r1_investors if inv_comm.get(i) is not None]
    if not r1_comms:
        n_no_round1_comm += 1
        continue
    home_comm = Counter(r1_comms).most_common(1)[0][0]

    # Bridge = first investor from a different community, in chronological order
    bridge_comm = None
    bridge_date = None
    for _, row in grp.iterrows():
        for inv in parse_investors(row["Investors"]):
            c = inv_comm.get(inv)
            if c is not None and c != home_comm:
                bridge_comm = c
                bridge_date = row["Deal Date"]
                break
        if bridge_comm is not None:
            break

    if bridge_comm is None:
        n_no_bridge += 1
        continue

    # Startup embedding = mean of ALL embedded investors (across all rounds)
    all_invs = []
    for inv_str in grp["Investors"]:
        all_invs.extend(parse_investors(inv_str))
    startup_vecs = [emb_map[i] for i in all_invs if i in emb_map]
    startup_emb = np.stack(startup_vecs, axis=0).mean(axis=0) if startup_vecs else None

    records.append({
        "company": company,
        "home_comm": home_comm,
        "bridge_comm": bridge_comm,
        "bridge_date": bridge_date,
        "n_startup_invs_embedded": len(startup_vecs),
        "startup_emb": startup_emb,
    })
    n_ok += 1

print(f"Cross-community companies:            {len(cross_companies)}")
print(f"  No round-1 community assignment:    {n_no_round1_comm}")
print(f"  No bridge community found:          {n_no_bridge}")
print(f"  Resolved home+bridge:               {n_ok}  [{elapsed()}]")

rec_df = pd.DataFrame(records)

# ════════════════════════════════════════════════════════════════════
# STEP 3: COMPUTE INDEPENDENT VARIABLES
# ════════════════════════════════════════════════════════════════════
banner("STEP 3 — COMM_DISTANCE and BRIDGE_EXPERTISE")

# Restrict to companies where BOTH communities have ≥10 embedded members
before_restriction = len(rec_df)
rec_df = rec_df[
    rec_df["home_comm"].isin(community_centroids) &
    rec_df["bridge_comm"].isin(community_centroids)
].copy()
after_restriction = len(rec_df)
print(f"Before ≥10-member restriction: {before_restriction}")
print(f"After  ≥10-member restriction: {after_restriction}  "
      f"(dropped {before_restriction - after_restriction})")

# A) comm_distance = 1 − cosine(home_centroid, bridge_centroid)
rec_df["comm_sim"] = rec_df.apply(
    lambda r: cosine_sim(community_centroids[r["home_comm"]],
                         community_centroids[r["bridge_comm"]]), axis=1
)
rec_df["comm_distance_raw"] = 1 - rec_df["comm_sim"]

# B) bridge_expertise = cosine(bridge_centroid, startup_embedding)
def bridge_exp(row):
    if row["startup_emb"] is None:
        return np.nan
    return cosine_sim(community_centroids[row["bridge_comm"]], row["startup_emb"])

rec_df["bridge_expertise_raw"] = rec_df.apply(bridge_exp, axis=1)

# Standardise
cd_mean, cd_sd = rec_df["comm_distance_raw"].mean(), rec_df["comm_distance_raw"].std()
be_mean, be_sd = rec_df["bridge_expertise_raw"].mean(), rec_df["bridge_expertise_raw"].std()
rec_df["comm_distance"]    = (rec_df["comm_distance_raw"]    - cd_mean) / cd_sd
rec_df["bridge_expertise"] = (rec_df["bridge_expertise_raw"] - be_mean) / be_sd

# Report descriptives
print(f"\ncomm_distance (raw):  mean={cd_mean:.4f}  SD={cd_sd:.4f}  "
      f"min={rec_df['comm_distance_raw'].min():.4f}  max={rec_df['comm_distance_raw'].max():.4f}")
print(f"bridge_expertise (raw): mean={be_mean:.4f}  SD={be_sd:.4f}  "
      f"min={rec_df['bridge_expertise_raw'].min():.4f}  max={rec_df['bridge_expertise_raw'].max():.4f}")

# Range check
cd_range = rec_df["comm_distance_raw"].max() - rec_df["comm_distance_raw"].min()
if cd_range < 0.15:
    print(f"\n  *** RANGE WARNING: comm_distance span = {cd_range:.4f} ***")
    print(f"  SPECTER2 similarity is compressed (typical: 0.80–0.98). A null β may")
    print(f"  reflect insufficient variation, not absence of effect.")
else:
    print(f"  Range span: {cd_range:.4f}  (adequate variation)")

corr_ab = rec_df[["comm_distance_raw", "bridge_expertise_raw"]].corr().iloc[0, 1]
print(f"\nCorrelation(comm_distance, bridge_expertise): {corr_ab:.4f}")

# Histogram bins for comm_distance
bins = np.linspace(rec_df["comm_distance_raw"].min(), rec_df["comm_distance_raw"].max(), 11)
hist, edges = np.histogram(rec_df["comm_distance_raw"].dropna(), bins=bins)
print("\ncomm_distance histogram:")
for i, (lo, hi, cnt) in enumerate(zip(edges[:-1], edges[1:], hist)):
    bar = "█" * int(cnt / max(hist) * 30)
    print(f"  [{lo:.3f}, {hi:.3f})  {cnt:4d}  {bar}")

# ════════════════════════════════════════════════════════════════════
# MERGE WITH COMPANY DATASET FOR CONTROLS + OUTCOME
# ════════════════════════════════════════════════════════════════════
# Compute syndicate size and log_n_investors from deals
banner("BUILDING ANALYSIS DATASET")

# Syndicate size = unique investors in round 1
synsize_rows = []
for company, grp in company_deals.groupby("Companies"):
    if company not in set(rec_df["company"]):
        continue
    grp = grp.sort_values("Deal Date")
    first_date = grp["Deal Date"].iloc[0]
    round1 = grp[grp["Deal Date"] <= first_date + pd.Timedelta(days=30)]
    r1_invs = set()
    for s in round1["Investors"]:
        r1_invs.update(parse_investors(s))
    all_invs = set()
    for s in grp["Investors"]:
        all_invs.update(parse_investors(s))
    synsize_rows.append({
        "company": company,
        "syndicate_size": len(r1_invs),
        "n_investors_total": len(all_invs),
    })

syn_df = pd.DataFrame(synsize_rows)

# Merge
adf = (
    rec_df.drop(columns=["startup_emb"], errors="ignore")
    .merge(cdf[["Companies", "failed", "log_capital", "n_rounds",
                "hq_country", "deal_year", "first_ym"]],
           left_on="company", right_on="Companies", how="inner")
    .merge(syn_df, on="company", how="left")
    .dropna(subset=["comm_distance", "failed", "log_capital", "n_rounds",
                    "hq_country", "deal_year"])
)

adf["log_n_investors"] = np.log1p(adf["n_investors_total"].fillna(1))
adf["syndicate_size"]  = adf["syndicate_size"].fillna(1)
adf["log_syndicate"]   = np.log1p(adf["syndicate_size"])

print(f"Final analysis N = {len(adf)}")
print(f"  of which: failed={int(adf['failed'].sum())}  "
      f"({adf['failed'].mean()*100:.1f}%)")
print(f"  dropped from {after_restriction}: {after_restriction - len(adf)} "
      f"(missing controls)")
print(f"  countries: {adf['hq_country'].nunique()}  "
      f"  years: {sorted(adf['deal_year'].dropna().unique().astype(int).tolist())}")

FINAL_N = len(adf)

# ════════════════════════════════════════════════════════════════════
# STEP 4: REGRESSIONS M1 – M3
# ════════════════════════════════════════════════════════════════════
banner("STEP 4 — REGRESSIONS M1 / M2 / M3")

CONTROLS = ("log_capital + n_rounds + log_syndicate + log_n_investors"
            " + C(deal_year) + C(hq_country)")

def run_model(formula, data, label):
    try:
        m = smf.ols(formula, data=data).fit(cov_type="HC1")
        key = formula.split("~")[1].split("+")[0].strip().lstrip("f(").rstrip(")")
        # Pull the first substantive coefficient
        sub_cols = [c for c in m.params.index
                    if not c.startswith("C(") and c != "Intercept"]
        if not sub_cols:
            return None
        results = []
        for col in sub_cols[:3]:   # report up to 3 main terms
            results.append({
                "model": label, "variable": col,
                "beta": m.params[col], "SE": m.bse[col],
                "p": m.pvalues[col], "N": int(m.nobs),
                "R2": m.rsquared,
            })
        return results, m
    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        return None, None

res_rows = []
models = {}

for label, formula in [
    ("M1", f"failed ~ comm_distance + {CONTROLS}"),
    ("M2", f"failed ~ bridge_expertise + {CONTROLS}"),
    ("M3", f"failed ~ comm_distance + bridge_expertise + {CONTROLS}"),
]:
    res = run_model(formula, adf, label)
    if res and res[0]:
        rows, m = res
        for r in rows:
            res_rows.append(r)
        models[label] = m

print(f"\n{'Model':<4} {'Variable':<22} {'β':>10} {'SE':>8} {'p':>8} {'N':>6} {'R²':>7}")
print("-" * 65)
for r in res_rows:
    sig = "***" if r["p"] < 0.001 else "** " if r["p"] < 0.01 else "*  " if r["p"] < 0.05 else "   "
    print(f"{r['model']:<4} {r['variable']:<22} {r['beta']:>10.6f} {r['SE']:>8.5f} "
          f"{r['p']:>8.4f}{sig} {r['N']:>6} {r['R2']:>7.4f}")

pd.DataFrame(res_rows).to_csv(OUT / "regression_results.csv", index=False)

# ════════════════════════════════════════════════════════════════════
# STEP 5: NON-LINEARITY CHECK
# ════════════════════════════════════════════════════════════════════
banner("STEP 5 — NON-LINEARITY (M4 + TERTILES)")

adf["comm_distance_sq"] = adf["comm_distance"] ** 2

res_m4 = run_model(
    f"failed ~ comm_distance + comm_distance_sq + {CONTROLS}",
    adf, "M4"
)
if res_m4 and res_m4[0]:
    for r in res_m4[0]:
        sig = "***" if r["p"] < 0.001 else "** " if r["p"] < 0.01 else "*  " if r["p"] < 0.05 else "   "
        print(f"{r['model']:<4} {r['variable']:<22} {r['beta']:>10.6f} {r['SE']:>8.5f} "
              f"{r['p']:>8.4f}{sig} {r['N']:>6} {r['R2']:>7.4f}")

# Tertile failure rates
adf["cd_tertile"] = pd.qcut(adf["comm_distance_raw"], q=3,
                              labels=["Low", "Mid", "High"], duplicates="drop")
print(f"\nTertile failure rates (by comm_distance_raw):")
print(f"  {'Tertile':<8} {'N':>6} {'N_fail':>8} {'Fail%':>8}")
for t, grp in adf.groupby("cd_tertile"):
    print(f"  {str(t):<8} {len(grp):>6} {int(grp['failed'].sum()):>8} "
          f"{grp['failed'].mean()*100:>7.1f}%")

# ════════════════════════════════════════════════════════════════════
# STEP 6: POWER AND FRAGILITY
# ════════════════════════════════════════════════════════════════════
banner("STEP 6 — POWER AND FRAGILITY")

cd_sd_raw = adf["comm_distance_raw"].std()
print(f"SD of comm_distance (raw):  {cd_sd_raw:.4f}")
print(f"Range (raw):                {adf['comm_distance_raw'].min():.4f} – "
      f"{adf['comm_distance_raw'].max():.4f}")

if cd_sd_raw < 0.05:
    print(f"\n  *** COMPRESSED RANGE WARNING ***")
    print(f"  SD = {cd_sd_raw:.4f} is very small. SPECTER2 community centroids often")
    print(f"  cluster within 0.85–0.98 cosine similarity. A null β is plausible")
    print(f"  due to range compression, not absence of a structural effect.")

# MDE at 80% power (two-sided t-test, α=0.05)
# With N obs, proportion p_fail, and OLS on binary outcome:
p_fail = adf["failed"].mean()
sd_y   = np.sqrt(p_fail * (1 - p_fail))
# Approximate: ignoring FE df loss, R2 of controls ≈ 0.05 conservative
# SE ≈ sd_y / (sd_x * sqrt(N) * sqrt(1 - R2_x)) where R2_x ≈ partial R2
# Simple approximation: MDE = 1.96 * 2 * sd_y / sqrt(N)  (conservative, ignores controls)
from scipy.stats import norm as snorm
z_alpha = snorm.ppf(0.975)
z_beta  = snorm.ppf(0.80)
mde_standardised = (z_alpha + z_beta) / np.sqrt(FINAL_N / 4)  # conservative (binary both)
mde_raw = mde_standardised * sd_y / cd_sd_raw if cd_sd_raw > 0 else np.nan
print(f"\nMDE at 80% power, α=0.05 (conservative):")
print(f"  N={FINAL_N}  failure rate={p_fail:.3f}  SD(comm_distance)={cd_sd_raw:.4f}")
print(f"  MDE (standardised β): {mde_standardised:.4f}")
print(f"  MDE (unstandardised β on comm_distance_raw): {mde_raw:.5f}")

# Trimmed regression: drop top/bottom 5% of comm_distance
lo5, hi5 = adf["comm_distance_raw"].quantile([0.05, 0.95])
adf_trim = adf[(adf["comm_distance_raw"] >= lo5) & (adf["comm_distance_raw"] <= hi5)].copy()
# Re-standardise within trimmed sample
cd_t_mean = adf_trim["comm_distance_raw"].mean()
cd_t_sd   = adf_trim["comm_distance_raw"].std()
adf_trim["comm_distance"] = (adf_trim["comm_distance_raw"] - cd_t_mean) / cd_t_sd

print(f"\nTrimmed M1 (drop top/bottom 5% of comm_distance, N={len(adf_trim)}):")
res_trim = run_model(f"failed ~ comm_distance + {CONTROLS}", adf_trim, "M1_trim")
if res_trim and res_trim[0]:
    for r in res_trim[0]:
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        print(f"  β={r['beta']:.6f}  SE={r['SE']:.5f}  p={r['p']:.4f}{sig}  N={r['N']}")

# ════════════════════════════════════════════════════════════════════
# STEP 7: ROBUSTNESS — all-MiniLM-L6-v2
# ════════════════════════════════════════════════════════════════════
banner("STEP 7 — ROBUSTNESS: all-MiniLM-L6-v2 EMBEDDINGS")

try:
    from sentence_transformers import SentenceTransformer

    print("Loading investor descriptions for MiniLM embedding ...")
    inv_data = pd.read_excel(BASE / "data" / "Investors_data.xlsx", header=7)
    # Build lowercase name → description map
    inv_col  = next(c for c in inv_data.columns if "investor" in c.lower()
                    and "type" not in c.lower() and "former" not in c.lower()
                    and "aka" not in c.lower() and "legal" not in c.lower()), None
    if isinstance(inv_col, tuple):
        inv_col = inv_col[0]

    inv_data["inv_lower"] = inv_data[inv_col].str.lower().str.strip()
    inv_data["desc_clean"] = inv_data["Description"].fillna("").str.strip()

    # Keep only investors in our analysis + community-assigned investors
    relevant_investors = set(inv_comm.keys())
    inv_sub = (inv_data[inv_data["inv_lower"].isin(relevant_investors) &
                        (inv_data["desc_clean"] != "")]
               .drop_duplicates("inv_lower"))

    print(f"  Investors with descriptions: {len(inv_sub)} / {len(relevant_investors)}")

    # Embed with MiniLM (384-dim)
    print("  Embedding with all-MiniLM-L6-v2 ...")
    mini = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    texts = inv_sub["desc_clean"].tolist()
    names_sub = inv_sub["inv_lower"].tolist()
    mini_vecs  = mini.encode(texts, batch_size=256, show_progress_bar=False,
                              convert_to_numpy=True).astype(np.float32)
    mini_map   = {n: mini_vecs[i] for i, n in enumerate(names_sub)}
    print(f"  MiniLM embeddings: {mini_vecs.shape}  [{elapsed()}]")

    # Build MiniLM community centroids (same eligible_comms filter)
    mini_comm_vecs = {}
    for inv, comm in inv_comm.items():
        vec = mini_map.get(inv)
        if vec is not None:
            mini_comm_vecs.setdefault(comm, []).append(vec)

    mini_centroids = {}
    for comm in eligible_comms:   # same ≥10 filter from SPECTER2
        if comm in mini_comm_vecs and len(mini_comm_vecs[comm]) >= 10:
            vecs = np.stack(mini_comm_vecs[comm], axis=0)
            c = vecs.mean(axis=0)
            n = np.linalg.norm(c)
            if n > 1e-10:
                mini_centroids[comm] = c / n
    print(f"  MiniLM centroids with ≥10 members: {len(mini_centroids)}")

    # Recompute comm_distance for analysis sample
    adf_mini = adf.copy()
    adf_mini["comm_dist_mini_raw"] = adf_mini.apply(
        lambda r: (1 - cosine_sim(mini_centroids[r["home_comm"]],
                                   mini_centroids[r["bridge_comm"]]))
        if r["home_comm"] in mini_centroids and r["bridge_comm"] in mini_centroids
        else np.nan,
        axis=1,
    )
    adf_mini = adf_mini.dropna(subset=["comm_dist_mini_raw"])
    cd_mini_mean = adf_mini["comm_dist_mini_raw"].mean()
    cd_mini_sd   = adf_mini["comm_dist_mini_raw"].std()
    adf_mini["comm_distance"] = (adf_mini["comm_dist_mini_raw"] - cd_mini_mean) / cd_mini_sd

    print(f"  MiniLM comm_distance: mean={cd_mini_mean:.4f}  SD={cd_mini_sd:.4f}  "
          f"N={len(adf_mini)}")

    res_mini = run_model(f"failed ~ comm_distance + {CONTROLS}", adf_mini, "M1_MiniLM")
    if res_mini and res_mini[0]:
        for r in res_mini[0]:
            sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else "n.s."
            print(f"  MiniLM M1: β={r['beta']:.6f}  SE={r['SE']:.5f}  p={r['p']:.4f} {sig}  N={r['N']}")
    mini_result = res_mini[0][0] if (res_mini and res_mini[0]) else None

except Exception as e:
    print(f"  MiniLM step failed: {e}")
    mini_result = None

print(f"\nStep 7 complete [{elapsed()}]")

# ════════════════════════════════════════════════════════════════════
# FULL RESULTS TABLE + VERDICTS
# ════════════════════════════════════════════════════════════════════
banner("FULL RESULTS TABLE + VERDICTS")

print(f"""
SAMPLE CONSTRUCTION
  Cross-community companies (cpu_company_dataset):   {len(cross_companies)}
  Resolved home + bridge community:                  {n_ok}
  Both communities have ≥10 embedded members:        {after_restriction}
  After merging controls (country, year, capital...): {FINAL_N}

DESCRIPTIVES
  comm_distance (raw):  mean={cd_mean:.4f}  SD={cd_sd:.4f}
                        min={adf['comm_distance_raw'].min():.4f}  max={adf['comm_distance_raw'].max():.4f}
  bridge_expertise (raw): mean={be_mean:.4f}  SD={be_sd:.4f}
  Correlation(A, B): {corr_ab:.4f}
""")

print(f"REGRESSION RESULTS (HC1 standard errors)")
print(f"{'Model':<4} {'Variable':<22} {'β':>10} {'SE':>8} {'p':>8} {'':3} {'N':>6} {'R²':>7}")
print("-" * 70)
for r in res_rows:
    sig = "***" if r["p"] < 0.001 else "** " if r["p"] < 0.01 else "*  " if r["p"] < 0.05 else "   "
    print(f"{r['model']:<4} {r['variable']:<22} {r['beta']:>10.6f} {r['SE']:>8.5f} "
          f"{r['p']:>8.4f} {sig} {r['N']:>6} {r['R2']:>7.4f}")

if res_m4 and res_m4[0]:
    for r in res_m4[0]:
        sig = "***" if r["p"] < 0.001 else "** " if r["p"] < 0.01 else "*  " if r["p"] < 0.05 else "   "
        print(f"{r['model']:<4} {r['variable']:<22} {r['beta']:>10.6f} {r['SE']:>8.5f} "
              f"{r['p']:>8.4f} {sig} {r['N']:>6} {r['R2']:>7.4f}")

# Verdicts
m1_row = next((r for r in res_rows if r["model"] == "M1" and r["variable"] == "comm_distance"), None)
m2_row = next((r for r in res_rows if r["model"] == "M2" and r["variable"] == "bridge_expertise"), None)

p1_holds = m1_row and m1_row["p"] < 0.05 and m1_row["beta"] < 0
p2_holds = m2_row and m2_row["p"] < 0.05 and m2_row["beta"] < 0

print(f"""
POWER / FRAGILITY NOTES
  SD(comm_distance raw) = {cd_sd_raw:.4f}  |  Range = {adf['comm_distance_raw'].max() - adf['comm_distance_raw'].min():.4f}
  MDE (standardised, 80% power) = {mde_standardised:.4f}  → detects β ≥ {mde_raw:.5f} on raw scale
  Trimmed M1 (p5–p95 sample, N={len(adf_trim)}): see above

MINIML ROBUSTNESS
  {f"β={mini_result['beta']:.6f}  p={mini_result['p']:.4f}" if mini_result else "Not available (see Step 7 output)"}

TOTAL RUNTIME: {elapsed()}
""")

print("=" * 60)
print("PLAIN VERDICTS")
print("=" * 60)
m1_b = m1_row["beta"] if m1_row else float("nan")
m1_p = m1_row["p"]    if m1_row else float("nan")
m2_b = m2_row["beta"] if m2_row else float("nan")
m2_p = m2_row["p"]    if m2_row else float("nan")

print(f"\nP1 (comm_distance → lower failure, β < 0):")
print(f"   M1: β = {m1_b:.6f}  p = {m1_p:.4f}")
if p1_holds:
    print("   P1 HOLDS: significant negative effect confirmed")
elif m1_row and m1_row["beta"] < 0:
    print("   P1 PARTIALLY SUPPORTED: sign is negative but not significant")
elif m1_row:
    print("   P1 DOES NOT HOLD: sign is positive or null")
else:
    print("   P1 NOT TESTABLE: model failed")

print(f"\nP2 (bridge_expertise → lower failure, β < 0):")
print(f"   M2: β = {m2_b:.6f}  p = {m2_p:.4f}")
if p2_holds:
    print("   P2 HOLDS: significant negative effect confirmed")
elif m2_row and m2_row["beta"] < 0:
    print("   P2 PARTIALLY SUPPORTED: sign is negative but not significant")
elif m2_row:
    print("   P2 DOES NOT HOLD: sign is positive or null")
else:
    print("   P2 NOT TESTABLE: model failed")

print(f"\nOutputs saved to {OUT}/")
print(f"Total runtime: {elapsed()}")
