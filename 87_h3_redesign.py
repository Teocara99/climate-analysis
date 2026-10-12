"""
Script 87: Redesign H3 — niche imprinting. NO outcome regressions.

1. Per-year lagged network: LCC only. Tune Louvain resolution for 40-60
   communities per year. Report communities per year, share retained.
2. Validate against static 46-community solution: NMI and geographic
   homogeneity (share of communities >=70% one region).
3. Restrict niches to founding year >= 2018. Recompute founding communities.
4. Feasibility: N niches, N later entrants, N dyads, N dyads with
   in_founding_community=1. Compute MDE. Proceed only if MDE < 1.0.
5. Update preregistration.md. Lock. Stop.
"""

import os, sys, re, warnings, time
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from statsmodels.discrete.conditional_models import ConditionalLogit
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "h3_redesign")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

print("=" * 70)
print("SCRIPT 87 — H3 REDESIGN: TUNED LOUVAIN & NICHE FEASIBILITY")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD INPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading inputs ──")

deals_df = load_deals()
deals_df["Deal Date"] = pd.to_datetime(deals_df["Deal Date"], errors="coerce")
deals_df = deals_df.dropna(subset=["Deal Date"])
deals_df["year"] = deals_df["Deal Date"].dt.year

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

# Edges sorted by year
print("  Building edge list…")
edges_by_year = {}   # year → list of (inv_a, inv_b)
for _, row in deals_df.iterrows():
    if pd.isna(row.get("Investors")):
        continue
    invs = comma_parse(row["Investors"])
    yr   = int(row["year"])
    for i in range(len(invs)):
        for j in range(i + 1, len(invs)):
            edges_by_year.setdefault(yr, []).append((invs[i], invs[j]))

# Static 46-community partition
net_m = pd.read_csv("output/network/network_metrics.csv")[["investor", "community"]]
static_part = net_m.dropna().set_index("investor")["community"].astype(int).to_dict()
N_STATIC = len(set(static_part.values()))
print(f"  Static partition: {N_STATIC} communities, {len(static_part):,} investors")

# Investor HQ country → global region (rough lookup)
inv_xl = pd.read_excel("data/Investors_data.xlsx", header=7,
                       usecols=["Investors", "HQ Country/Territory/Region"])
inv_xl = inv_xl.rename(columns={"Investors": "investor",
                                 "HQ Country/Territory/Region": "inv_country"})
inv_xl = inv_xl.dropna(subset=["investor"]).set_index("investor")["inv_country"].to_dict()

COUNTRY_TO_REGION = {
    "United States": "N America", "Canada": "N America", "Mexico": "N America",
    "United Kingdom": "W Europe", "Germany": "W Europe", "France": "W Europe",
    "Netherlands": "W Europe", "Sweden": "W Europe", "Spain": "W Europe",
    "Denmark": "W Europe", "Finland": "W Europe", "Norway": "W Europe",
    "Switzerland": "W Europe", "Austria": "W Europe", "Belgium": "W Europe",
    "Ireland": "W Europe", "Italy": "W Europe", "Portugal": "W Europe",
    "China": "Asia", "Japan": "Asia", "South Korea": "Asia",
    "India": "Asia", "Singapore": "Asia", "Taiwan": "Asia",
    "Hong Kong": "Asia", "Australia": "APAC (other)", "New Zealand": "APAC (other)",
    "Brazil": "LatAm", "Chile": "LatAm", "Colombia": "LatAm",
    "Israel": "Middle East / Africa", "UAE": "Middle East / Africa",
    "Saudi Arabia": "Middle East / Africa", "South Africa": "Middle East / Africa",
}
def inv_region(investor):
    ctry = inv_xl.get(investor)
    if ctry:
        return COUNTRY_TO_REGION.get(ctry, "Other")
    return None

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1  Tune Louvain per year
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 1: Tune Louvain (LCC, target 40-60 communities) ──")

TARGET_LO, TARGET_HI = 40, 60
YEARS = sorted(set(edges_by_year.keys()))

G_cumul = nx.Graph()
tuned_partitions = {}   # year → {investor: community}
year_stats        = []

for yr in YEARS:
    # Store community BEFORE adding year yr's edges (lagged: based on deals < yr)
    if G_cumul.number_of_nodes() >= 10:
        # LCC
        lcc_nodes = max(nx.connected_components(G_cumul), key=len)
        G_lcc = G_cumul.subgraph(lcc_nodes).copy()
        n_full = G_cumul.number_of_nodes()
        n_lcc  = G_lcc.number_of_nodes()
        pct_lcc = 100 * n_lcc / n_full

        # Binary search for resolution in [0.5, 4.0]
        lo_r, hi_r = 0.5, 4.0
        best_res = 1.0
        best_part = None
        best_n_c  = None
        for _ in range(15):
            mid_r = (lo_r + hi_r) / 2
            part  = community_louvain.best_partition(
                G_lcc, weight="weight", resolution=mid_r, random_state=42
            )
            n_c = len(set(part.values()))
            if n_c < TARGET_LO:
                lo_r = mid_r
            elif n_c > TARGET_HI:
                hi_r = mid_r
            else:
                best_res = mid_r
                best_part = part
                best_n_c  = n_c
                break
            if best_part is None or abs(n_c - (TARGET_LO + TARGET_HI) / 2) < abs(best_n_c - 50 if best_n_c else 999):
                best_res = mid_r
                best_part = part
                best_n_c  = n_c

        # Store ALL investors in the LCC with community; non-LCC get None
        tuned_partitions[yr] = best_part   # {investor: community}
        year_stats.append({
            "year": yr, "n_full": n_full, "n_lcc": n_lcc,
            "pct_lcc": pct_lcc, "resolution": best_res, "n_communities": best_n_c
        })
    else:
        tuned_partitions[yr] = {}
        year_stats.append({"year": yr, "n_full": G_cumul.number_of_nodes(), "n_lcc": 0,
                           "pct_lcc": 0, "resolution": np.nan, "n_communities": 0})

    # Add this year's edges
    for inv_a, inv_b in edges_by_year.get(yr, []):
        if G_cumul.has_edge(inv_a, inv_b):
            G_cumul[inv_a][inv_b]["weight"] += 1
        else:
            G_cumul.add_edge(inv_a, inv_b, weight=1)

stats_df = pd.DataFrame(year_stats)
print(f"\n  {'Year':>4}  {'Full':>6}  {'LCC':>6}  {'%LCC':>5}  {'res':>5}  {'n_comm':>6}")
print("  " + "─" * 42)
for _, r in stats_df.iterrows():
    flag = "" if pd.isna(r["n_communities"]) else (
        "  ← in range" if TARGET_LO <= r["n_communities"] <= TARGET_HI else
        "  ← below target" if r["n_communities"] < TARGET_LO else "  ← above target"
    )
    nc = f"{r['n_communities']:.0f}" if not pd.isna(r["n_communities"]) and r["n_communities"] else "—"
    res_s = f"{r['resolution']:.2f}" if not pd.isna(r.get("resolution")) else "—"
    print(f"  {int(r['year']):>4}  {r['n_full']:>6}  {r['n_lcc']:>6}  "
          f"{r['pct_lcc']:>4.0f}%  {res_s:>5}  {nc:>6}{flag}")

# Build fast lookup: (investor, year) → tuned community
tuned_lc_lookup = {}
for yr, part in tuned_partitions.items():
    for inv, comm in part.items():
        tuned_lc_lookup[(inv, yr)] = comm

# Save tuned communities
tuned_rows = []
for yr, part in tuned_partitions.items():
    for inv, comm in part.items():
        tuned_rows.append({"investor": inv, "year": yr, "community_tuned": comm})
tuned_df = pd.DataFrame(tuned_rows)
tuned_path = os.path.join(OUT_DIR, "lagged_comm_tuned.csv")
tuned_df.to_csv(tuned_path, index=False)
print(f"\n  Tuned community table saved → {tuned_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2  Validate: NMI vs static partition + geographic homogeneity
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: Validation ──")

# NMI: compare the most recent year's tuned partition with static
VALID_YR = 2024   # lagged = deals up to 2023
tuned_2024 = tuned_partitions.get(VALID_YR, {})
if tuned_2024:
    common = sorted(set(tuned_2024.keys()) & set(static_part.keys()))
    if len(common) > 100:
        labels_t = [tuned_2024[i]   for i in common]
        labels_s = [static_part[i]  for i in common]
        nmi = normalized_mutual_info_score(labels_s, labels_t)
        print(f"  NMI (tuned yr<{VALID_YR} vs static 46-community): {nmi:.4f}  "
              f"(N common investors: {len(common):,})")
    else:
        print(f"  NMI: too few common investors ({len(common)})")
else:
    print(f"  Tuned partition for year {VALID_YR} not available")

# Geographic homogeneity: for the most recent tuned year
print()
if tuned_2024:
    # Map investor → region
    comm_regions = {}   # community → list of regions
    for inv, comm in tuned_2024.items():
        reg = inv_region(inv)
        if reg:
            comm_regions.setdefault(comm, []).append(reg)

    homog_counts = []
    for comm, regions in comm_regions.items():
        if len(regions) >= 3:
            from collections import Counter
            top_share = max(Counter(regions).values()) / len(regions)
            homog_counts.append(top_share)

    if homog_counts:
        pct_homog = 100 * np.mean(np.array(homog_counts) >= 0.70)
        print(f"  Communities ≥70% one region: {pct_homog:.1f}%  "
              f"(N communities with ≥3 located investors: {len(homog_counts):,})")
        print(f"  Mean top-region share: {np.mean(homog_counts):.3f}  "
              f"Median: {np.median(homog_counts):.3f}")
    else:
        print("  Geographic homogeneity: insufficient coverage")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3  Rebuild niches restricted to founding year >= 2018
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 3: Rebuild niches (founding year ≥ 2018) ──")

niche_df  = pd.read_csv("output/niches/niche_profiles.csv")
member_df = pd.read_csv("output/niches/niche_assignments.csv",
                        parse_dates=["first_date"])

# First-round investors per company
first_round_inv = {}
for cid, grp in deals_df.sort_values("Deal Date").groupby("Company ID"):
    row = grp.iloc[0]
    first_round_inv[cid] = comma_parse(row["Investors"]) if pd.notna(row.get("Investors")) else []

# Filter founders to ≥ 2018
founders = member_df[member_df["position"] == "founder"].copy()
founders_18 = founders[founders["first_year"] >= 2018]

print(f"  Original niches: {niche_df['niche_id'].nunique():,}")
print(f"  Niches with founding year ≥ 2018: {len(founders_18):,}")

valid_niche_ids_18 = set(founders_18["niche_id"])
member_18 = member_df[member_df["niche_id"].isin(valid_niche_ids_18)].copy()

# Recompute founding communities using TUNED partitions
niche_profiles_18 = []
for nid in sorted(valid_niche_ids_18):
    niche_members = member_18[member_18["niche_id"] == nid].sort_values("first_date")
    if len(niche_members) < 2:
        continue
    founder_row  = niche_members.iloc[0]
    founder_name = founder_row["Companies"]
    founder_cid  = founder_row["Company ID"]
    t0           = int(founder_row["first_year"])

    # Founding community: tuned partition at year t0
    founder_investors = first_round_inv.get(founder_cid, [])
    comm_votes = {}
    for inv in founder_investors:
        c = tuned_lc_lookup.get((inv, t0))
        if c is not None:
            comm_votes[c] = comm_votes.get(c, 0) + 1
    if comm_votes:
        founding_community = max(comm_votes, key=comm_votes.get)
        n_with_comm = sum(comm_votes.values())
    else:
        founding_community = None
        n_with_comm = 0

    niche_profiles_18.append({
        "niche_id":           nid,
        "founding_startup":   founder_name,
        "founding_year":      t0,
        "founding_community": founding_community,
        "n_founder_inv":      len(founder_investors),
        "n_founder_inv_w_comm": n_with_comm,
        "n_members":          len(niche_members),
        "members":            "; ".join(niche_members["Companies"].tolist()),
    })

niche_18_df = pd.DataFrame(niche_profiles_18)
print(f"  Niches with ≥2 dated members: {len(niche_18_df):,}")
print(f"  Niches with founding community (tuned): "
      f"{niche_18_df['founding_community'].notna().sum():,}")

# Founding-investor coverage
fc_total = niche_18_df["n_founder_inv"].sum()
fc_with  = niche_18_df["n_founder_inv_w_comm"].sum()
print(f"  Founding investor community coverage: "
      f"{fc_with}/{fc_total} = {100*fc_with/max(fc_total,1):.1f}%")

# Save
niche_18_path = os.path.join(OUT_DIR, "niche_profiles_2018plus.csv")
niche_18_df.to_csv(niche_18_path, index=False)
print(f"  Saved → {niche_18_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4  Feasibility: build dyad dataset, compute MDE
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 4: Feasibility check ──")

import pickle
# Startup embeddings
with open("output/novelty/novelty_specter.pkl", "rb") as f:
    spec_data = pickle.load(f)
su_emb_names = spec_data["names"]
su_emb_raw   = np.array(spec_data["embs_raw"], dtype=np.float32)
su_norms     = np.linalg.norm(su_emb_raw, axis=1, keepdims=True)
su_norms[su_norms == 0] = 1.0
su_emb_arr   = su_emb_raw / su_norms
su_name_to_idx = {n: i for i, n in enumerate(su_emb_names)}

# Investor embeddings
inv_names_df = pd.read_csv("data/investor_names.csv", header=None, names=["investor"])
inv_emb_arr  = np.load("data/investor_embeddings.npy").astype(np.float32)
inv_emb_norms = np.linalg.norm(inv_emb_arr, axis=1, keepdims=True)
inv_emb_norms[inv_emb_norms == 0] = 1.0
inv_emb_arr  = inv_emb_arr / inv_emb_norms
inv_name_to_idx = {n: i for i, n in enumerate(inv_names_df["investor"])}
if "investor" in inv_name_to_idx:
    del inv_name_to_idx["investor"]

# Active investors by year
active_by_year = {}
for yr, grp in deals_df.groupby("year"):
    invs = set()
    for v in grp["Investors"].dropna():
        invs.update(comma_parse(v))
    active_by_year[int(yr)] = invs

# Startup country
startup_country = (
    deals_df.sort_values("Deal Date")
    .groupby("Company ID").first()["Company Country/Territory/Region"].to_dict()
)

# Investor prior deals count
inv_deals_before = {}
for yr, grp in sorted(deals_df.groupby("year")):
    for v in grp["Investors"].dropna():
        for inv in comma_parse(v):
            inv_deals_before.setdefault(inv, []).append(int(yr))

def prior_deals(investor, year):
    return sum(1 for y in inv_deals_before.get(investor, []) if y < year)

# Usable niches: founding community assigned, ≥3 members
usable_18 = niche_18_df.dropna(subset=["founding_community"]).copy()
usable_18["founding_community"] = usable_18["founding_community"].astype(int)
usable_18 = usable_18[usable_18["n_members"] >= 3]
print(f"  Usable niches (FC assigned, ≥3 members): {len(usable_18):,}")

niche_fc_18  = usable_18.set_index("niche_id")["founding_community"].to_dict()
niche_fy_18  = usable_18.set_index("niche_id")["founding_year"].to_dict()

# Founder Company IDs
niche_founder_cid_18 = {}
for nid, row in usable_18.iterrows():
    founder_name = row["founding_startup"]
    matches = member_18[(member_18["niche_id"] == nid) &
                        (member_18["position"] == "founder")]["Company ID"]
    if len(matches):
        niche_founder_cid_18[row["niche_id"]] = matches.iloc[0]

# Later entrants in usable niches
later_18 = member_18[
    (member_18["position"] == "later") &
    (member_18["niche_id"].isin(niche_fc_18))
].copy()
print(f"  Later entrants in usable niches: {len(later_18):,}")

# Build dyad dataset
dyad_rows = []
n_skip_no_inv = 0
n_skip_pool   = 0

for _, su_row in later_18.iterrows():
    nid   = int(su_row["niche_id"])
    cid   = su_row["Company ID"]
    name  = su_row["Companies"]
    t_k   = int(su_row["first_year"])
    fc    = niche_fc_18.get(nid)

    realised = first_round_inv.get(cid, [])
    if not realised:
        n_skip_no_inv += 1
        continue

    # Exclude founding startup's investors
    founder_cid = niche_founder_cid_18.get(nid)
    founder_inv = set(first_round_inv.get(founder_cid, [])) if founder_cid else set()
    realised_ok = [i for i in realised if i not in founder_inv]
    if not realised_ok:
        n_skip_no_inv += 1
        continue

    pool = [i for i in active_by_year.get(t_k, set())
            if i not in set(realised) and i not in founder_inv]
    if len(pool) < 5:
        n_skip_pool += 1
        continue

    counterf = list(RNG.choice(pool, size=5, replace=False))
    su_idx   = su_name_to_idx.get(name)
    su_ctry  = startup_country.get(cid)

    for investor, label in ([(i, 1) for i in realised_ok] +
                             [(i, 0) for i in counterf]):
        # in_founding_community: investor's tuned community at t_k == fc
        inv_comm = tuned_lc_lookup.get((investor, t_k))
        ifc      = int(inv_comm == fc) if (inv_comm is not None and fc is not None) else 0

        # semantic distance
        if su_idx is not None:
            inv_idx = inv_name_to_idx.get(investor)
            sem_d   = float(1.0 - su_emb_arr[su_idx] @ inv_emb_arr[inv_idx]) if inv_idx is not None else np.nan
        else:
            sem_d = np.nan

        # geo
        inv_ctry  = inv_xl.get(investor)
        geo_ctry  = int(su_ctry == inv_ctry) if (su_ctry and inv_ctry) else 0

        # prior activity
        pd_count  = prior_deals(investor, t_k)
        log_pd    = float(np.log1p(pd_count))
        hpn       = float(pd_count > 0)

        dyad_rows.append({
            "startup":               cid,
            "investor":              investor,
            "niche_id":              nid,
            "tie":                   label,
            "in_founding_community": ifc,
            "semantic_distance":     sem_d,
            "geo_same_country":      geo_ctry,
            "log_prior_deals":       log_pd,
            "has_prior_network":     hpn,
            "year":                  t_k,
        })

dm = pd.DataFrame(dyad_rows)
dm_ok = dm.dropna(subset=["semantic_distance"]).copy()
for col in ["semantic_distance", "log_prior_deals"]:
    mu = dm_ok[col].mean(); sd = dm_ok[col].std()
    dm_ok[f"{col}_z"] = (dm_ok[col] - mu) / sd if sd > 0 else 0.0

def has_both(grp):
    return grp["tie"].nunique() == 2

valid = dm_ok.groupby("startup").filter(has_both)

n_pos         = dm["in_founding_community"].sum()
n_pos_valid   = valid["in_founding_community"].sum()
n_strata      = valid["startup"].nunique()
n_dyads_valid = len(valid)

print(f"\n  Dyad dataset:")
print(f"    Total rows:         {len(dm):>6,}")
print(f"    With sem_distance:  {len(dm_ok):>6,}")
print(f"    Valid strata:       {n_strata:>6,}")
print(f"    Valid dyads:        {n_dyads_valid:>6,}")
print(f"    in_fc = 1 (all):    {int(n_pos):>6,}  ({100*dm['in_founding_community'].mean():.1f}%)")
print(f"    in_fc = 1 (valid):  {int(n_pos_valid):>6,}  ({100*n_pos_valid/max(n_dyads_valid,1):.1f}%)")

# MDE estimation: run univariate conditional logit on in_founding_community only
b_vars_feas = ["in_founding_community", "semantic_distance_z",
               "geo_same_country", "log_prior_deals_z", "has_prior_network"]

mde = np.nan
se_ifc = np.nan
proceed = False

if n_pos_valid >= 10 and n_strata >= 30:
    try:
        endog = valid["tie"].values.astype(float)
        exog  = valid[b_vars_feas].values.astype(float)
        mask  = np.isfinite(exog).all(axis=1)
        grps  = valid["startup"].values[mask]
        model = ConditionalLogit(endog[mask], exog[mask], groups=grps)
        for method in ["newton", "bfgs", "lbfgs"]:
            try:
                res = model.fit(method=method, maxiter=400, disp=False)
                if hasattr(res, "bse") and np.isfinite(res.bse).all():
                    se_ifc = float(res.bse[0])
                    break
            except Exception:
                continue
        if np.isfinite(se_ifc):
            mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se_ifc
            proceed = mde < 1.0
    except Exception as e:
        print(f"  MDE estimation failed: {e}")
else:
    print(f"  Insufficient positive cases (n_pos_valid={n_pos_valid}) for MDE estimation.")

print(f"\n  MDE RESULT:")
print(f"    SE(in_founding_community): {se_ifc:.4f}" if np.isfinite(se_ifc) else "    SE: unavailable")
print(f"    MDE (80% power, α=0.05):   {mde:.4f}" if np.isfinite(mde) else "    MDE: unavailable")
print(f"    Threshold:                 1.0 log-odds")
if np.isfinite(mde):
    print(f"    Decision:  {'PROCEED — MDE < 1.0, H3 is testable' if proceed else 'STOP — MDE >= 1.0, H3 underpowered'}")
else:
    print(f"    Decision:  STOP — MDE could not be computed (insufficient data)")

# Save dyad dataset for Script 88
dyad_path = os.path.join(OUT_DIR, "dyad_dataset_18plus.csv")
dm_ok.to_csv(dyad_path, index=False)
print(f"\n  Dyad dataset saved → {dyad_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5  Update pre-registration
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 5: Update preregistration.md ──")

PREREG_PATH = os.path.join(os.path.dirname(__file__), "output", "preregistration.md")
with open(PREREG_PATH) as f:
    prereg = f.read()

from datetime import datetime
ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

H3_ADDENDUM = f"""
---

## Script 86 result and H3 redesign rationale ({ts})

### Script 86 result (original design)

| Quantity | Value |
|---|---|
| Niches with founding community | 133 / 628 (21%) |
| Valid dyads | 1,701 |
| in_founding_community = 1 | 23 dyads (0.7%) |
| β (in_founding_community) | −1.408 (SE=1.115, p=0.207) |
| Permutation p (500 draws) | 0.180 |
| MDE (80% power) | 3.12 log-odds |
| Verdict | NULL — WRONG SIGN |

**Why H3 failed:** The lagged community approach requires the founding
startup's investors to already have a co-investment history before year t₀.
For niches founded before 2016, almost no investor has prior deal history,
so founding community coverage was only 26.1% of founding investors. This
produced only 23 dyads with in_founding_community=1, making the test
uninformative (MDE >> 1.0).

### Redesign specification

**Changes:**
1. **LCC restriction**: lagged network built on the largest connected
   component only, not the full graph. Isolates the main exchange network.
2. **Tuned resolution**: Louvain resolution parameter tuned per year by
   binary search to yield 40–60 communities. This produces consistent
   community granularity across years rather than letting the resolution
   fall wherever modularity peaks.
3. **Founding-year cutoff**: restrict to niches whose founding startup's
   first round is ≥ 2018. By 2018 the LCC covers {
     round(stats_df[stats_df['year']==2018]['pct_lcc'].values[0]) if len(stats_df[stats_df['year']==2018]) > 0 else 'N/A'
}% of investors, so community coverage is substantially higher.

**Hypothesis prediction:** unchanged. **POSITIVE** sign on
in_founding_community.

**MDE gate:** feasibility check must confirm MDE < 1.0 log-odds before
Script 88 (H3 execution) is authorised.
Script 87 feasibility result:
  - N usable niches (FC assigned, ≥3 members): {len(usable_18)}
  - N later entrants: {len(later_18)}
  - N valid dyads: {n_dyads_valid}
  - in_fc = 1 in valid dyads: {int(n_pos_valid)} ({100*n_pos_valid/max(n_dyads_valid,1):.1f}%)
  - SE: {f'{se_ifc:.4f}' if np.isfinite(se_ifc) else 'unavailable'}
  - MDE: {f'{mde:.4f}' if np.isfinite(mde) else 'unavailable'}
  - Decision: {'PROCEED' if proceed else 'STOP'}
"""

prereg_updated = prereg + H3_ADDENDUM
prereg_updated = prereg_updated.replace(
    "**Revision timestamp:**",
    f"**Revision timestamp:** {ts}\n**Previous revision timestamp:**"
)

with open(PREREG_PATH, "w") as f:
    f.write(prereg_updated)
print(f"  Pre-registration updated → {PREREG_PATH}")

print("\n" + "=" * 70)
print("SCRIPT 87 COMPLETE")
if np.isfinite(mde):
    print(f"MDE = {mde:.4f} log-odds  →  {'PROCEED to Script 88' if proceed else 'STOP — H3 underpowered'}")
else:
    print("MDE unavailable — STOP")
print("=" * 70)
