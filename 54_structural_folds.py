"""
Script 54: Vedres & Stark (2010) Structural Folds vs Structural Holes
======================================================================

Splits "bridge investors" (those whose co-investors span 2+ Louvain
communities) into:
  - HOLE  (Burt-style): touches multiple communities but with LOW ego-network
            density inside each — loose bridging ties
  - FOLD  (Vedres-Stark "multiple insider"): dense cohesive insider in EACH of
            the communities they touch simultaneously — high ego-network density
            within every community they reach

Steps:
  1. Inspect schemas (already done externally — verified above)
  2. Build co-investment edge set from deals (needed for ego-density)
  3. Per bridge investor: compute within-community ego-network density
  4. Classify HOLE / FOLD via median split
  5. Aggregate to company level: which companies have FOLD vs HOLE bridges
  6. Re-run 2SLS (replicating Script 30 spec) separately + interaction term
  7. Save outputs + markdown summary

Key flag: investors with <3 neighbors in a community → density degenerate
          (reported separately in Step 6 diagnostics)
"""

import warnings; warnings.filterwarnings("ignore")
import sys, pickle, re
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

try:
    from linearmodels.iv import IV2SLS
    HAS_LM = True
except ImportError:
    HAS_LM = False
    print("  linearmodels not available — will use 2-stage OLS fallback")

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

def parse_investors(s):
    if not isinstance(s, str) or not s.strip(): return []
    parts = [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]
    return [re.sub(r"\s*\([^)]*\)\s*$", "", p).strip() for p in parts if p]

OUT = Path("output/structural_folds"); OUT.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 0: SCHEMA REPORT
# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("STEP 0: SCHEMA INSPECTION")
print("=" * 72)

bt = pd.read_csv("output/community_anatomy/bridge_types.csv")
nm = pd.read_csv("output/network/network_metrics.csv")
bi = pd.read_csv("output/network/bridge_investors.csv")

print(f"\n  bridge_types.csv  shape={bt.shape}")
print(f"  columns: {list(bt.columns)}")
print(f"  NOTE: this is COMPANY-level (bridge_type = dimensional characterization)")
print(f"  bridge_type values: {bt['bridge_type'].value_counts().head(5).to_dict()}")

print(f"\n  network_metrics.csv  shape={nm.shape}")
print(f"  columns: {list(nm.columns)}")
print(f"  community range: [{nm['community'].min()}, {nm['community'].max()}]  "
      f"n_communities={nm['community'].nunique()}")
print(f"  degree: median={nm['degree'].median():.0f}  max={nm['degree'].max()}")

print(f"\n  bridge_investors.csv  shape={bi.shape}")
print(f"  columns: {list(bi.columns)}")
print(f"  cross_type==True (bridge across green/trad): {bi['cross_type'].sum():,}")

print(f"\n  NOTE: Louvain assigns each investor to exactly ONE community.")
print(f"  'Bridge investor' here = investor whose CO-INVESTORS span 2+ communities.")
print(f"  Ego-density is computed among the investor's neighbors within each community.")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 1: LOAD DATA & BUILD CO-INVESTMENT GRAPH")
print("=" * 72)

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["company_key"] = cdf["Companies"].str.lower().str.strip()
cdf["is_failed"]   = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]     = np.log1p(cdf["Total Raised"].fillna(0))
cdf["deal_year"]   = cdf["deal_year"].fillna(2018).astype(int)

deals = load_deals()
deals["company_key"] = deals["Companies"].str.lower().str.strip()
# Use Deal Date (40k+ non-null) rather than Announced Date (only 5.9k)
deals["date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals_dated  = deals.dropna(subset=["date"]).copy()   # 40k+ rows for IV
# All deals (no date filter) for building the co-investment graph
deals_all    = deals.dropna(subset=["Investors"]).copy()

# Company age (use full deals for widest coverage)
year_founded = (deals.dropna(subset=["Year Founded"])
                .groupby("company_key")["Year Founded"].first().astype(int))
cdf["year_founded"] = cdf["company_key"].map(year_founded)
cdf["company_age"]  = (cdf["deal_year"] - cdf["year_founded"]).clip(0, 30)

# Investor → community mapping (lower-cased keys)
inv_to_comm = dict(zip(nm["investor"].str.lower().str.strip(),
                       nm["community"].astype(int)))

# Investor → geo_focus (for IV_B)
ip = pd.read_csv("output/investor_profiles.csv")
inv_to_geo = dict(zip(ip["investor"].str.lower().str.strip(), ip["geo_focus"]))

print(f"  Companies: {len(cdf):,}  with company_age: {cdf['company_age'].notna().sum():,}")
print(f"  Deals (all): {len(deals):,}  Deals with date: {len(deals_dated):,}  Investors in network: {len(inv_to_comm):,}")

# ── Build co-investment graph (adjacency as set of frozensets) ────────────
print("\n  Building co-investment edge set…")

# One group per company: all investors who appeared in any deal (no date filter)
co_investors_map = (
    deals_all
    .assign(inv=lambda df: df["Investors"].str.split(r",\s*"))
    .explode("inv")
    .assign(inv=lambda df: df["inv"].str.strip()
                                    .str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
                                    .str.strip()
                                    .str.lower())
    .query('inv != ""')
    .groupby("company_key")["inv"]
    .apply(set)
)

# Build edge set and adjacency dict simultaneously
edge_set = set()          # frozenset({inv_i, inv_j})
adj = defaultdict(set)    # investor → set of co-investors

for co, invs in co_investors_map.items():
    invs = list(invs)
    for i in range(len(invs)):
        for j in range(i + 1, len(invs)):
            a, b = invs[i], invs[j]
            edge_set.add(frozenset((a, b)))
            adj[a].add(b)
            adj[b].add(a)

print(f"  Edges: {len(edge_set):,}")
print(f"  Nodes with adjacency: {len(adj):,}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: IDENTIFY BRIDGE INVESTORS & COMPUTE EGO-NETWORK DENSITY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 2: EGO-NETWORK DENSITY PER BRIDGE INVESTOR")
print("=" * 72)

# For each investor: group neighbors by community, compute density within each
fold_rows = []
n_total = len(adj)
n_bridge = 0
n_degenerate = 0

for inv, neighbors in adj.items():
    own_comm = inv_to_comm.get(inv)

    # Group neighbors by their community (excluding those with no community assignment)
    comm_to_neighbors = defaultdict(set)
    for nb in neighbors:
        c = inv_to_comm.get(nb)
        if c is not None:
            comm_to_neighbors[c].add(nb)

    touched_communities = set(comm_to_neighbors.keys())

    # Bridge investor = has neighbors in 2+ communities
    if len(touched_communities) < 2:
        continue

    n_bridge += 1

    # For each community touched: compute ego-network density
    densities = {}
    flags = {}
    for c, nbs in comm_to_neighbors.items():
        n = len(nbs)
        if n < 2:
            # Degenerate: can't compute density with <2 neighbors in community
            densities[c] = np.nan
            flags[c] = "degenerate_n<2"
        elif n == 2:
            # Only one possible edge — density is 0 or 1
            nbs_list = list(nbs)
            has_edge = frozenset((nbs_list[0], nbs_list[1])) in edge_set
            densities[c] = 1.0 if has_edge else 0.0
            flags[c] = "low_n_2"
        else:
            # Proper density computation
            nbs_list = list(nbs)
            actual = sum(1 for a, b in combinations(nbs_list, 2)
                         if frozenset((a, b)) in edge_set)
            possible = n * (n - 1) // 2
            densities[c] = actual / possible
            flags[c] = "ok" if n >= 5 else "low_n<5"
            if n < 3:
                n_degenerate += 1

    valid_densities = [d for d in densities.values() if not np.isnan(d)]
    mean_density   = np.mean(valid_densities) if valid_densities else np.nan
    min_density    = np.min(valid_densities) if valid_densities else np.nan
    n_comms_ok     = sum(1 for d in densities.values() if not np.isnan(d))
    any_degenerate = any(f != "ok" for f in flags.values())

    fold_rows.append({
        "investor":            inv,
        "own_community":       own_comm,
        "n_communities_touched": len(touched_communities),
        "communities_touched": str(sorted(touched_communities)),
        "density_per_community": str({c: round(d, 4) if not np.isnan(d) else "NaN"
                                       for c, d in densities.items()}),
        "mean_within_density": mean_density,
        "min_within_density":  min_density,
        "n_comms_with_valid_density": n_comms_ok,
        "has_degenerate":      any_degenerate,
    })

fold_df = pd.DataFrame(fold_rows)
print(f"  Total investors with adjacency data: {n_total:,}")
print(f"  Bridge investors (neighbors in 2+ communities): {n_bridge:,}")
print(f"  Coverage: {n_bridge/n_total:.1%} of investors are bridges")
print(f"\n  Distribution of communities touched:")
vc = fold_df["n_communities_touched"].value_counts().sort_index()
for k, v in vc.head(8).items():
    print(f"    Touches {k} communities: {v:,} investors ({v/len(fold_df):.1%})")

print(f"\n  Density distribution (mean within-community density):")
valid = fold_df["mean_within_density"].dropna()
print(f"    N with valid density: {len(valid):,}")
print(f"    Mean: {valid.mean():.4f}  Median: {valid.median():.4f}")
print(f"    p25: {valid.quantile(.25):.4f}  p75: {valid.quantile(.75):.4f}")
print(f"    Min: {valid.min():.4f}  Max: {valid.max():.4f}")
print(f"    Has-degenerate (any community with n<2): {fold_df['has_degenerate'].sum():,} "
      f"({fold_df['has_degenerate'].mean():.1%})")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: CLASSIFY HOLE vs FOLD
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 3: CLASSIFY HOLE vs FOLD (MEDIAN SPLIT ON mean_within_density)")
print("=" * 72)

# Median computed only on valid (non-degenerate or with ok density)
density_median = fold_df["mean_within_density"].median()
print(f"\n  Global median within-community density: {density_median:.4f}")
print(f"  (Investors above median → FOLD / multiple insider)")
print(f"  (Investors below median → HOLE / structural hole)")

fold_df["classification"] = np.where(
    fold_df["mean_within_density"].isna(), "UNCLASSIFIED",
    np.where(fold_df["mean_within_density"] >= density_median, "FOLD", "HOLE")
)

cls_counts = fold_df["classification"].value_counts()
print(f"\n  Classification split:")
for cls, n in cls_counts.items():
    print(f"    {cls:<15}: {n:>5,}  ({n/len(fold_df):.1%})")

# Flag: investors with ALL degenerate communities
all_degen = fold_df["mean_within_density"].isna().sum()
print(f"\n  ⚠ UNCLASSIFIED (all communities degenerate, n<2 neighbors): {all_degen:,}")
print(f"  These investors touch 2+ communities but always through single ties.")
print(f"  By Burt logic these are HOLES; we exclude them from regression but flag them.")

# Also provide quartile-based classification for robustness
q25 = fold_df["mean_within_density"].quantile(0.25)
q75 = fold_df["mean_within_density"].quantile(0.75)
fold_df["classification_strict"] = np.where(
    fold_df["mean_within_density"].isna(), "UNCLASSIFIED",
    np.where(fold_df["mean_within_density"] >= q75, "FOLD_STRONG",
    np.where(fold_df["mean_within_density"] <= q25, "HOLE_STRONG", "AMBIGUOUS"))
)
print(f"\n  Quartile-based strict classification:")
for cls, n in fold_df["classification_strict"].value_counts().items():
    print(f"    {cls:<15}: {n:>5,}  ({n/len(fold_df):.1%})")

# Save investor-level file
fold_df.to_csv(OUT / "bridge_type_folds_holes.csv", index=False)
print(f"\n  Saved: {OUT}/bridge_type_folds_holes.csv  ({len(fold_df):,} rows)")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: AGGREGATE TO COMPANY LEVEL
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 4: COMPANY-LEVEL BRIDGE TYPE ASSIGNMENT")
print("=" * 72)

fold_lookup   = dict(zip(fold_df["investor"], fold_df["classification"]))
density_lookup = dict(zip(fold_df["investor"], fold_df["mean_within_density"]))

co_bridge_rows = []
for co, invs in co_investors_map.items():
    invs_lower = {i.lower() for i in invs}
    # Bridge investors in this syndicate = those classified as HOLE or FOLD
    bridge_invs = [i for i in invs_lower if fold_lookup.get(i) in ("FOLD","HOLE")]
    n_fold = sum(1 for i in bridge_invs if fold_lookup.get(i) == "FOLD")
    n_hole = sum(1 for i in bridge_invs if fold_lookup.get(i) == "HOLE")
    n_unclass = sum(1 for i in invs_lower if fold_lookup.get(i) == "UNCLASSIFIED")
    avg_density = np.mean([density_lookup[i] for i in bridge_invs
                           if not np.isnan(density_lookup.get(i, np.nan))]
                          ) if bridge_invs else np.nan

    # Dominant bridge type
    if n_fold == 0 and n_hole == 0:
        dom_type = "NO_BRIDGE"
    elif n_fold > n_hole:
        dom_type = "FOLD_DOM"
    elif n_hole > n_fold:
        dom_type = "HOLE_DOM"
    else:
        dom_type = "MIXED"

    co_bridge_rows.append({
        "company_key": co,
        "n_bridge_investors": len(bridge_invs),
        "n_fold": n_fold,
        "n_hole": n_hole,
        "n_unclassified": n_unclass,
        "avg_bridge_density": avg_density,
        "dominant_bridge_type": dom_type,
        "has_fold": int(n_fold > 0),
        "has_hole": int(n_hole > 0),
    })

co_bridge_df = pd.DataFrame(co_bridge_rows)
print(f"\n  Companies with bridge-investor data: {len(co_bridge_df):,}")
print(f"\n  Dominant bridge type distribution:")
for t, n in co_bridge_df["dominant_bridge_type"].value_counts().items():
    print(f"    {t:<15}: {n:>5,}  ({n/len(co_bridge_df):.1%})")

# Merge with main analysis frame
df = (cdf.merge(co_bridge_df, on="company_key", how="left")
         .copy())
df["is_fold_dom"] = (df["dominant_bridge_type"] == "FOLD_DOM").astype(float)
df["is_hole_dom"] = (df["dominant_bridge_type"] == "HOLE_DOM").astype(float)
df["has_fold"]    = df["has_fold"].fillna(0)
df["has_hole"]    = df["has_hole"].fillna(0)

# Cross-community × bridge type
df_cross = df[df["cross_community"] == 1].copy()
print(f"\n  Cross-community companies: {df['cross_community'].sum():,}")
print(f"  of which FOLD-dominated: {(df_cross['dominant_bridge_type']=='FOLD_DOM').sum():,}")
print(f"  of which HOLE-dominated: {(df_cross['dominant_bridge_type']=='HOLE_DOM').sum():,}")
print(f"  of which MIXED:          {(df_cross['dominant_bridge_type']=='MIXED').sum():,}")
print(f"  of which NO_BRIDGE:      {(df_cross['dominant_bridge_type']=='NO_BRIDGE').sum():,}")

# Failure rates by bridge type (raw)
print(f"\n  Failure rates by bridge type (cross-community companies only):")
print(f"  {'Type':<20} {'N':>6}  {'Fail%':>7}")
print(f"  {'-'*40}")
for t in ["FOLD_DOM","HOLE_DOM","MIXED","NO_BRIDGE"]:
    s = df_cross[df_cross["dominant_bridge_type"]==t]["is_failed"]
    if len(s) < 5: continue
    print(f"  {t:<20} {len(s):>6}  {s.mean():>7.1%}")

# Also: non-cross-community baseline
nc = df[df["cross_community"]==0]["is_failed"]
print(f"  {'NON-CROSS (ref)':<20} {len(nc):>6}  {nc.mean():>7.1%}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: 2SLS MODELS — REPLICATE SCRIPT 30 SPEC + FOLD/HOLE SPLIT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 5: 2SLS REGRESSION (Script 30 spec + FOLD/HOLE interaction)")
print("=" * 72)

# ── Build IV_A (lead investor prior community reach) ──────────────────────
print("\n  Building IV_A …")
investor_history = defaultdict(list)
for _, row in deals_dated.iterrows():
    inv_list = parse_investors(str(row.get("Investors", "")))
    if not inv_list: continue
    inv_lower = [i.lower().strip() for i in inv_list if i.strip()]
    date = row["date"]
    comms_for_row = {i: inv_to_comm.get(i) for i in inv_lower}
    for inv in inv_lower:
        co_comms = frozenset(c for i, c in comms_for_row.items()
                             if i != inv and c is not None)
        investor_history[inv].append((date, co_comms))
for inv in investor_history:
    investor_history[inv].sort(key=lambda x: x[0])

deals_sorted = (deals_dated.sort_values("date")
                .dropna(subset=["Investors"])
                .drop_duplicates("company_key", keep="first"))

def get_lead(row):
    lc = row.get("Lead/Sole Investors")
    if pd.notna(lc) and str(lc).strip():
        invs = parse_investors(str(lc))
        if invs: return invs[0].lower().strip()
    invs = parse_investors(str(row.get("Investors","")))
    return invs[0].lower().strip() if invs else None

company_lead = {row["company_key"]: (get_lead(row), row["date"])
                for _, row in deals_sorted.iterrows()}

def prior_communities(lead_inv, deal_date, lookback=2):
    hist = investor_history.get(lead_inv, [])
    if not hist: return 0
    cutoff = deal_date - pd.DateOffset(years=lookback)
    prior_comms = set()
    for d, co_comms in hist:
        if d >= deal_date: break
        if d >= cutoff: prior_comms.update(co_comms)
    return len(prior_comms)

iv_a_records = []
for ck, (lead, date) in company_lead.items():
    iv_a_records.append({"company_key": ck,
                         "iv_a": prior_communities(lead, date) if lead else np.nan})
iv_a_df = pd.DataFrame(iv_a_records)
print(f"    IV_A coverage: {iv_a_df['iv_a'].notna().sum():,}  "
      f"mean={iv_a_df['iv_a'].mean():.2f}  sd={iv_a_df['iv_a'].std():.2f}")

# ── Build IV_B (geographic distance between investor HQs) ─────────────────
print("  Building IV_B …")
geo_order = ["Americas","Europe","Asia","Oceania","Middle East","Africa"]
geo_idx   = {g: i for i, g in enumerate(geo_order)}

iv_b_records = []
for _, row in deals_dated.sort_values("date").drop_duplicates("company_key", keep="first").iterrows():
    ck = row["company_key"]
    inv_list = parse_investors(str(row.get("Investors","")))
    inv_lower = [i.lower().strip() for i in inv_list if i.strip()]
    geos = [geo_idx[inv_to_geo[i]] for i in inv_lower
            if inv_to_geo.get(i) in geo_idx]
    if len(geos) < 2:
        iv_b_records.append({"company_key": ck, "iv_b": np.nan})
    else:
        dists = [abs(geos[a] - geos[b]) for a in range(len(geos))
                 for b in range(a+1, len(geos))]
        iv_b_records.append({"company_key": ck, "iv_b": np.mean(dists)})
iv_b_df = pd.DataFrame(iv_b_records)
print(f"    IV_B coverage: {iv_b_df['iv_b'].notna().sum():,}  "
      f"mean={iv_b_df['iv_b'].mean():.3f}")

# ── Assemble analysis frame ────────────────────────────────────────────────
reg = (df.merge(iv_a_df, on="company_key", how="left")
          .merge(iv_b_df, on="company_key", how="left"))
reg["iv_a_z"] = (reg["iv_a"] - reg["iv_a"].mean()) / reg["iv_a"].std()
reg["iv_b_z"] = (reg["iv_b"] - reg["iv_b"].mean()) / reg["iv_b"].std()

# Country FE (same as before)
reg["country_fe"] = reg["hq_country"].where(
    reg["hq_country"].isin(reg["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

required = ["is_failed","cross_community","log_cap","specter_novelty","company_age","iv_a"]
reg = reg.dropna(subset=required).copy()
reg["const"] = 1.0
print(f"\n  Final regression sample: N={len(reg):,}")

# Year + region dummies
year_dummies   = pd.get_dummies(reg["deal_year"], prefix="yr", drop_first=True)
region_dummies = pd.get_dummies(reg["hq_region"], prefix="reg", drop_first=True)
exog_cols = ["log_cap","specter_novelty","company_age"]
exog_base = pd.concat([reg[["const"] + exog_cols].reset_index(drop=True),
                        year_dummies.reset_index(drop=True),
                        region_dummies.reset_index(drop=True)], axis=1)
exog_base.index = reg.index

def run_2sls(y, exog, endog, instruments, label):
    idx = (y.dropna().index & endog.dropna().index
           & instruments.dropna().index & exog.dropna().index)
    y_    = y.loc[idx]
    exog_ = exog.loc[idx].copy()
    endog_= endog.loc[idx]
    inst_ = instruments.loc[idx]
    # Drop zero-variance columns (dummies that are all 0 or all 1 in this subset)
    nonconstant = exog_.columns[exog_.std() > 0]
    if "const" not in nonconstant:
        nonconstant = ["const"] + list(nonconstant)
    exog_ = exog_[nonconstant]
    n     = len(y_)
    lm_ok = False
    if HAS_LM:
        try:
            mod = IV2SLS(dependent=y_, exog=exog_, endog=endog_, instruments=inst_)
            res = mod.fit(cov_type="robust")
            b   = res.params.get("cross_community", np.nan)
            se  = res.std_errors.get("cross_community", np.nan)
            t   = res.tstats.get("cross_community", np.nan)
            p   = res.pvalues.get("cross_community", 1.0)
            try:
                fs = res.first_stage
                fs_res = fs.results.get("cross_community") or list(fs.results.values())[0]
                fs_f = fs_res.f_statistic.stat if hasattr(fs_res, 'f_statistic') else np.nan
            except:
                fs_f = np.nan
            lm_ok = True
        except Exception as e:
            print(f"      [IV2SLS failed ({e}), falling back to manual 2-stage OLS]")
    if not lm_ok:
        # 2-stage OLS fallback using numpy (avoids patsy column-name issues)
        X   = exog_.values.astype(float)          # [N × K]
        Z   = inst_.values.astype(float)           # [N × L]
        D   = endog_.values.astype(float)          # [N × 1]
        Y   = y_.values.astype(float).reshape(-1)  # [N]
        XZ  = np.hstack([X, Z])                    # [N × K+L]
        # Stage 1: D ~ X + Z
        D_hat, _, _, _ = np.linalg.lstsq(XZ, D, rcond=None)
        D_fitted = XZ @ D_hat                      # [N, 1]
        # Stage 2: Y ~ X + D_fitted
        XD = np.hstack([X, D_fitted.reshape(-1, 1)])
        coefs, _, _, _ = np.linalg.lstsq(XD, Y, rcond=None)
        b = coefs[-1]                              # coefficient on cross_community
        # HC1 SE for stage-2 coefficient on D_fitted
        resid = Y - XD @ coefs
        n2, k2 = XD.shape
        scale = (resid @ resid) / (n2 - k2)
        # Sandwich: (X'X)^{-1} X' diag(e^2) X (X'X)^{-1} times n/(n-k)
        try:
            XtX_inv = np.linalg.pinv(XD.T @ XD)
            meat = XD.T @ np.diag(resid**2) @ XD
            vcov = (n2 / (n2 - k2)) * (XtX_inv @ meat @ XtX_inv)
            se   = float(np.sqrt(vcov[-1, -1]))
        except Exception:
            se = np.nan
        t    = b / se if se > 0 else np.nan
        p    = float(2 * (1 - stats.t.cdf(abs(t), df=n2-k2))) if not np.isnan(t) else np.nan
        # First-stage F (partial F for Z in stage 1)
        try:
            D1_hat, _, _, _ = np.linalg.lstsq(X, D, rcond=None)
            D1_resid = D - X @ D1_hat
            D2_hat_resid = D_fitted.reshape(-1) - (X @ np.linalg.lstsq(X, D_fitted, rcond=None)[0]).reshape(-1)
            fs_f = float((D2_hat_resid @ D2_hat_resid / Z.shape[1]) /
                         (D1_resid.reshape(-1) @ D1_resid.reshape(-1) / (n - XZ.shape[1])))
        except Exception:
            fs_f = np.nan
    return {"label": label, "N": n, "beta": b, "se": se, "t": t, "p": p, "fs_F": fs_f}

results = []
print(f"\n  {'─'*72}")
print(f"  Running 2SLS models (IV_A + IV_B, replicating Script 30 spec):")

# M0: FULL SAMPLE (baseline replication)
y     = reg["is_failed"]
endog = reg[["cross_community"]].astype(float)
inst  = reg[["iv_a_z","iv_b_z"]].fillna(0)
inst_a = reg[["iv_a_z"]]
r0 = run_2sls(y, exog_base, endog, inst_a, "M0: Full sample (IV_A only)")
results.append(r0)

# M1: FOLD-dominated companies only
mask_fold = reg["dominant_bridge_type"] == "FOLD_DOM"
if mask_fold.sum() >= 50:
    r1 = run_2sls(y[mask_fold], exog_base[mask_fold], endog[mask_fold],
                  inst_a[mask_fold], "M1: FOLD-bridge companies")
    results.append(r1)
    print(f"    M1 FOLD sample: N={mask_fold.sum()}")
else:
    print(f"    ⚠ FOLD_DOM too small ({mask_fold.sum()}) — skipping subsample")
    results.append({"label":"M1: FOLD (insufficient N)","N":int(mask_fold.sum()),
                    "beta":np.nan,"se":np.nan,"t":np.nan,"p":np.nan,"fs_F":np.nan})

# M2: HOLE-dominated companies only
mask_hole = reg["dominant_bridge_type"] == "HOLE_DOM"
if mask_hole.sum() >= 50:
    r2 = run_2sls(y[mask_hole], exog_base[mask_hole], endog[mask_hole],
                  inst_a[mask_hole], "M2: HOLE-bridge companies")
    results.append(r2)
    print(f"    M2 HOLE sample: N={mask_hole.sum()}")
else:
    print(f"    ⚠ HOLE_DOM too small ({mask_hole.sum()}) — skipping subsample")
    results.append({"label":"M2: HOLE (insufficient N)","N":int(mask_hole.sum()),
                    "beta":np.nan,"se":np.nan,"t":np.nan,"p":np.nan,"fs_F":np.nan})

# M3: NO-BRIDGE cross-community companies (neither FOLD nor HOLE)
mask_nobr = reg["dominant_bridge_type"] == "NO_BRIDGE"
if mask_nobr.sum() >= 50:
    r3 = run_2sls(y[mask_nobr], exog_base[mask_nobr], endog[mask_nobr],
                  inst_a[mask_nobr], "M3: NO-BRIDGE companies")
    results.append(r3)

# M4: Pooled OLS interaction term (cross_community × has_fold)
print(f"\n  M4: OLS interaction (pooled) — cross_community × has_fold")
reg_int = reg.copy()
reg_int["cc_x_fold"] = reg_int["cross_community"] * reg_int["has_fold"]
reg_int["cc_x_hole"] = reg_int["cross_community"] * reg_int["has_hole"]
ctrl = "log_cap + specter_novelty + company_age + C(deal_year) + C(hq_region)"
try:
    m4 = smf.ols(f"is_failed ~ cross_community + has_fold + has_hole + "
                 f"cc_x_fold + cc_x_hole + {ctrl}",
                 data=reg_int.dropna(subset=["is_failed"])).fit(cov_type="HC1")
    print(f"    N={int(m4.nobs):,}  R²={m4.rsquared:.4f}")
    print(f"    {'Variable':<30} {'β':>9}  {'p':>7}  {'Sig'}")
    print(f"    {'-'*58}")
    for v, lbl in [
        ("cross_community",    "cross_community (base)"),
        ("has_fold",           "has_fold"),
        ("has_hole",           "has_hole"),
        ("cc_x_fold",          "cross_comm × has_fold  ← KEY"),
        ("cc_x_hole",          "cross_comm × has_hole  ← KEY"),
    ]:
        if v in m4.params:
            b = m4.params[v]; p = m4.pvalues[v]
            print(f"    {lbl:<30} {b:>+9.4f}  {p:>7.4f}  {sig(p)}")

    # Wald test: cc_x_fold ≠ cc_x_hole
    try:
        w = m4.f_test("cc_x_fold = cc_x_hole")
        fw, pw = float(w.statistic), float(w.pvalue)
        print(f"\n    Wald test cc_x_fold = cc_x_hole: F={fw:.3f}  p={pw:.4f}  {sig(pw)}")
        if pw < 0.05:
            print(f"    ✅ FOLD and HOLE bridges have SIGNIFICANTLY DIFFERENT effects")
        else:
            print(f"    ❌ Cannot distinguish FOLD vs HOLE interaction effect (p={pw:.3f})")
    except Exception as e:
        print(f"    Wald test error: {e}")
except Exception as e:
    print(f"    M4 error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6: RESULTS TABLE + DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("STEP 6: RESULTS SUMMARY TABLE")
print("=" * 72)

print(f"\n  {'Model':<38} {'N':>6}  {'β(cross_comm)':>14}  {'SE':>7}  {'p':>7}  {'Sig':>5}  {'FS-F':>7}")
print(f"  {'-'*95}")
for r in results:
    b  = r['beta']
    se = r['se']
    p  = r['p']
    ff = r['fs_F']
    print(f"  {r['label']:<38} {r['N']:>6}  {f'{b:>+.4f}' if not np.isnan(b) else '  N/A  ':>14}  "
          f"{f'{se:.4f}' if not np.isnan(se) else '  N/A ':>7}  "
          f"{f'{p:.4f}' if not np.isnan(p) else '  N/A ':>7}  "
          f"{sig(p) if not np.isnan(p) else '':>5}  "
          f"{f'{ff:.1f}' if ff is not None and not np.isnan(ff) else '  N/A ':>7}")

print(f"\n  ⚠ SPECIFICATION FLAGS:")
print(f"    1. UNCLASSIFIED investors (all communities have <2 neighbors): "
      f"{fold_df['classification'].eq('UNCLASSIFIED').sum():,} "
      f"({fold_df['classification'].eq('UNCLASSIFIED').mean():.1%})")
print(f"       → By Burt logic these are HOLES. Conservative: assign them HOLE for robustness check.")

low_n = fold_df[fold_df["has_degenerate"] & (fold_df["classification"] != "UNCLASSIFIED")]
print(f"    2. Classified investors with ≥1 degenerate community (n<2 neighbors in it): "
      f"{len(low_n):,}")
print(f"       → Density for those communities is NaN; mean density uses only valid communities.")
print(f"       → If most communities per investor are degenerate, the mean is unstable.")

fold_small = fold_df[fold_df["n_comms_with_valid_density"] <= 1]
print(f"    3. Investors where valid density is based on only 1 community: "
      f"{len(fold_small):,}  ({len(fold_small)/len(fold_df):.1%})")
print(f"       → These cannot truly be 'multiple insiders' — FOLD classification is weak for them.")
print(f"       → Recommendation: require valid density in ≥2 communities for FOLD label.")

fold_strict = fold_df[
    (fold_df["n_comms_with_valid_density"] >= 2) &
    (fold_df["classification"] == "FOLD")
]
print(f"    4. FOLD investors with valid density in ≥2 communities (strict criterion): "
      f"{len(fold_strict):,}  ({len(fold_strict)/len(fold_df):.1%})")

# ════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE MARKDOWN SUMMARY
# ════════════════════════════════════════════════════════════════════════════

baseline_r = results[0]
fold_r     = results[1] if len(results) > 1 else {}
hole_r     = results[2] if len(results) > 2 else {}

fold_n = reg["dominant_bridge_type"].eq("FOLD_DOM").sum()
hole_n = reg["dominant_bridge_type"].eq("HOLE_DOM").sum()
nobr_n = reg["dominant_bridge_type"].eq("NO_BRIDGE").sum()

fold_fail = df_cross[df_cross["dominant_bridge_type"]=="FOLD_DOM"]["is_failed"].mean()
hole_fail = df_cross[df_cross["dominant_bridge_type"]=="HOLE_DOM"]["is_failed"].mean()

def fmt(r, key):
    v = r.get(key, np.nan) if isinstance(r, dict) else np.nan
    return f"{v:+.4f}" if not np.isnan(v) else "N/A"

md = f"""# Structural Folds vs Structural Holes — Regression Results
*(Vedres & Stark 2010, AJS — adapted for single-assignment Louvain communities)*

## Setup

**Adaptation note:** PitchBook Louvain detection assigns each investor to exactly **one** community (46 communities, N=17,790 investors). "Bridge investors" here are investors whose **co-investors** span 2+ communities — they sit at the boundary but are formally assigned to one. Ego-network density is computed *among the investor's neighbors* within each community they reach.

- **HOLE (Burt-style):** mean within-community ego-density < global median ({density_median:.4f})
- **FOLD (Vedres-Stark multiple insider):** mean within-community ego-density ≥ global median

**2SLS spec (replicating Script 30):**
- Endogenous: `cross_community`
- Instrument: IV_A = lead investor's prior community reach (# unique communities touched in prior 2 years)
- Controls: `log_cap`, `specter_novelty`, `company_age`, year FE, region FE
- Outcome: `is_failed`

---

## Step 2–3: Bridge Investor Classification

| Metric | Value |
|---|---|
| Total investors with co-investment data | {n_total:,} |
| Bridge investors (neighbors in ≥2 communities) | {n_bridge:,} ({n_bridge/n_total:.1%}) |
| Global median within-community ego-density | {density_median:.4f} |
| FOLD (≥ median density) | {fold_df['classification'].eq('FOLD').sum():,} ({fold_df['classification'].eq('FOLD').mean():.1%}) |
| HOLE (< median density) | {fold_df['classification'].eq('HOLE').sum():,} ({fold_df['classification'].eq('HOLE').mean():.1%}) |
| UNCLASSIFIED (all communities degenerate) | {fold_df['classification'].eq('UNCLASSIFIED').sum():,} ({fold_df['classification'].eq('UNCLASSIFIED').mean():.1%}) |

---

## Step 4: Company-Level Bridge Type

Cross-community companies only (N={df_cross.shape[0]:,}):

| Bridge Type | N companies | Failure Rate |
|---|---|---|
| FOLD-dominated | {(df_cross['dominant_bridge_type']=='FOLD_DOM').sum():,} | {fold_fail:.1%} |
| HOLE-dominated | {(df_cross['dominant_bridge_type']=='HOLE_DOM').sum():,} | {hole_fail:.1%} |
| MIXED | {(df_cross['dominant_bridge_type']=='MIXED').sum():,} | — |
| NO_BRIDGE (cross-comm, no classified bridge investor) | {(df_cross['dominant_bridge_type']=='NO_BRIDGE').sum():,} | — |
| Non-cross-community (reference) | {df[df['cross_community']==0].shape[0]:,} | {df[df['cross_community']==0]['is_failed'].mean():.1%} |

---

## Step 5: 2SLS Results

| Model | N | β(cross_community) | SE | p | First-stage F |
|---|---|---|---|---|---|
| M0: Full sample | {baseline_r.get('N','—')} | {fmt(baseline_r,'beta')} | {fmt(baseline_r,'se')} | {f"{baseline_r.get('p',np.nan):.4f}" if not np.isnan(baseline_r.get('p',np.nan)) else 'N/A'} | {f"{baseline_r.get('fs_F',np.nan):.1f}" if baseline_r.get('fs_F') is not None and not np.isnan(baseline_r.get('fs_F',np.nan)) else 'N/A'} |
| M1: FOLD-bridge companies | {fold_r.get('N','—')} | {fmt(fold_r,'beta')} | {fmt(fold_r,'se')} | {f"{fold_r.get('p',np.nan):.4f}" if isinstance(fold_r,dict) and not np.isnan(fold_r.get('p',np.nan)) else 'N/A'} | {f"{fold_r.get('fs_F',np.nan):.1f}" if isinstance(fold_r,dict) and fold_r.get('fs_F') is not None and not np.isnan(fold_r.get('fs_F',np.nan)) else 'N/A'} |
| M2: HOLE-bridge companies | {hole_r.get('N','—')} | {fmt(hole_r,'beta')} | {fmt(hole_r,'se')} | {f"{hole_r.get('p',np.nan):.4f}" if isinstance(hole_r,dict) and not np.isnan(hole_r.get('p',np.nan)) else 'N/A'} | {f"{hole_r.get('fs_F',np.nan):.1f}" if isinstance(hole_r,dict) and hole_r.get('fs_F') is not None and not np.isnan(hole_r.get('fs_F',np.nan)) else 'N/A'} |

**OLS pooled interaction (M4):** `is_failed ~ cross_community + has_fold + has_hole + cross_community×has_fold + cross_community×has_hole + controls`
*(see console output for full M4 coefficients and Wald test)*

---

## Step 6: Specification Flags

1. **UNCLASSIFIED investors** (all communities have <2 neighbors in them): cannot compute ego-density. These are structural singles — by Burt's logic they are HOLES. Conservative robustness check: recode UNCLASSIFIED → HOLE.

2. **Density based on n=2 neighbors** in a community: density is trivially 0 or 1 (one possible edge). Affects {fold_df['has_degenerate'].mean():.1%} of bridge investors.

3. **Valid density from only 1 community**: {len(fold_small):,} investors ({len(fold_small)/len(fold_df):.1%}). These cannot satisfy the Vedres-Stark criterion of being an insider in *multiple* communities. Strict FOLD requires valid density in ≥2 communities: N={len(fold_strict):,} ({len(fold_strict)/len(fold_df):.1%}).

4. **First-stage F < 10** (if observed): weak instruments warning. The IV was validated in Script 30 on the full sample; subgroup F-statistics may be lower due to smaller N.

## Output Files

- `output/structural_folds/bridge_type_folds_holes.csv` — investor-level classification (N={len(fold_df):,})
- `output/structural_folds/structural_folds_test_results.md` — this file
"""

with open(OUT / "structural_folds_test_results.md", "w") as f:
    f.write(md)

print(f"\n  Saved: {OUT}/structural_folds_test_results.md")
print(f"  Saved: {OUT}/bridge_type_folds_holes.csv")
print("\n  ✅ Script 54 complete.")
