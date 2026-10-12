"""
Script 83: Setup and pre-registration.
NO outcome regressions.

1. Persist overall_sparsity_t and niche_cohesion_t (time-aware, within-year standardised)
   → output/novelty_t/novelty_time_aware.csv
   → output/novelty_t/lagged_communities.csv  (investor → community by year)

2. Propose definitions (for author review):
   a) IRA_tech: k=60 cluster → IRA credit category mapping with justification
   b) hardware: hardware-heavy cluster proposals
   c) niches: mutual 5-NN groups ≥3 members with above-median niche_cohesion_t

3. Pre-registration file → output/preregistration.md
   H1–H10 with unit, DV, key variable, predicted sign, family (A/B/C/D)
   Holm correction within each family

4. Feasibility for H4
   Cell counts: US/non-US × IRA_tech/not × pre/post 2022-08-16
   Flag cells < 100

Stop after this. Do NOT run H4.
"""

import os, sys, pickle, re, warnings, textwrap
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import community as community_louvain
import networkx as nx

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "novelty_t")
os.makedirs(OUT_DIR, exist_ok=True)

IRA_DATE = pd.Timestamp("2022-08-16")   # IRA signing date

# ── helpers ──────────────────────────────────────────────────────────────────
def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

print("=" * 70)
print("SCRIPT 83 — SETUP & PRE-REGISTRATION")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0  Load shared inputs
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading inputs ──")
deals = load_deals()
deals["Deal Date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals = deals.dropna(subset=["Deal Date"])
deals["year"] = deals["Deal Date"].dt.year

# Startup first-round date  (same logic as 79r)
first_round = (
    deals.sort_values("Deal Date")
    .groupby("Company ID")
    .first()[["Companies", "Deal Date"]]
    .rename(columns={"Deal Date": "first_date"})
)
first_round["first_year"] = first_round["first_date"].dt.year

# Startup SPECTER2 embeddings
with open("output/novelty/novelty_specter.pkl", "rb") as f:
    spec_data = pickle.load(f)
emb_names = spec_data["names"]
emb_raw   = spec_data["embs_raw"]          # (9749, 768)
emb_arr   = np.array(emb_raw, dtype=np.float32)
norms     = np.linalg.norm(emb_arr, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb_arr   = emb_arr / norms                 # L2-normalised

# Map embedding index → Company name
emb_name_to_idx = {n: i for i, n in enumerate(emb_names)}

# Merge company IDs into embedding roster
id_map = first_round.reset_index()[["Company ID", "Companies", "first_date", "first_year"]]
id_map["emb_idx"] = id_map["Companies"].map(emb_name_to_idx)
id_map = id_map.dropna(subset=["emb_idx"]).copy()
id_map["emb_idx"] = id_map["emb_idx"].astype(int)
id_map = id_map.sort_values("first_year").reset_index(drop=True)

print(f"  Embeddings: {len(emb_names):,} startups")
print(f"  Matched to deals (have first-round date): {len(id_map):,}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1  Time-aware novelty  (overall_sparsity_t, niche_cohesion_t)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 1: Time-aware novelty ──")

MIN_PRIOR  = 100
K_LOC, K_REG = 5, 50

sorted_rows = id_map.sort_values("first_year").to_dict("records")
N_total = len(sorted_rows)

prior_embs = []
results    = []

# Group by year for within-year batch processing
from itertools import groupby
from operator import itemgetter

for yr, yr_group in groupby(sorted_rows, key=itemgetter("first_year")):
    yr_batch = list(yr_group)
    yr_indices = [r["emb_idx"] for r in yr_batch]
    yr_embs = emb_arr[yr_indices]              # (n_yr, 768)

    n_prior = len(prior_embs)
    if n_prior < MIN_PRIOR:
        # Not enough prior startups — mark as dropped
        for r in yr_batch:
            results.append({**r, "loc5_t": np.nan, "reg50_t": np.nan, "dropped": True})
        prior_embs.extend(yr_embs.tolist())
        continue

    prior_arr = np.array(prior_embs, dtype=np.float32)
    k_max = min(K_REG + 1, n_prior)
    nn = NearestNeighbors(n_neighbors=k_max, metric="cosine",
                          algorithm="brute", n_jobs=-1).fit(prior_arr)
    dists, _ = nn.kneighbors(yr_embs, n_neighbors=k_max)
    sims = 1.0 - dists                        # cosine similarity

    for i, r in enumerate(yr_batch):
        loc5  = float(sims[i, :min(K_LOC,  sims.shape[1])].mean()) if sims.shape[1] >= K_LOC  else np.nan
        reg50 = float(sims[i, :min(K_REG,  sims.shape[1])].mean()) if sims.shape[1] >= K_REG  else np.nan
        results.append({**r, "loc5_t": loc5, "reg50_t": reg50, "dropped": False})

    prior_embs.extend(yr_embs.tolist())

tanov = pd.DataFrame(results)
n_dropped = tanov["dropped"].sum()
print(f"  Dropped (year ≤ threshold, <{MIN_PRIOR} prior): {n_dropped:,}")

# Within-year standardisation
tanov_ok = tanov[~tanov["dropped"]].copy()
for col_z, col_raw in [("z_loc", "loc5_t"), ("z_reg", "reg50_t")]:
    for yr, grp in tanov_ok.groupby("first_year"):
        mu = grp[col_raw].mean()
        sd = grp[col_raw].std()
        if sd > 0:
            tanov_ok.loc[grp.index, col_z] = (grp[col_raw] - mu) / sd
        else:
            tanov_ok.loc[grp.index, col_z] = 0.0

tanov_ok["overall_sparsity_t"] = -(tanov_ok["z_loc"] + tanov_ok["z_reg"]) / 2
tanov_ok["niche_cohesion_t"]   =   tanov_ok["z_loc"] - tanov_ok["z_reg"]

print(f"  Retained (have time-aware novelty): {len(tanov_ok):,}")
print(f"  overall_sparsity_t: mean={tanov_ok['overall_sparsity_t'].mean():.3f}  "
      f"sd={tanov_ok['overall_sparsity_t'].std():.3f}")
print(f"  niche_cohesion_t:   mean={tanov_ok['niche_cohesion_t'].mean():.3f}  "
      f"sd={tanov_ok['niche_cohesion_t'].std():.3f}")

# Correlation with static versions
static = pd.read_csv("output/two_level_novelty/novelty_final.csv")[
    ["name", "overall_sparsity", "niche_cohesion"]
]
merged_check = tanov_ok.merge(static, left_on="Companies", right_on="name", how="inner")
if len(merged_check) > 50:
    r_os = merged_check["overall_sparsity_t"].corr(merged_check["overall_sparsity"])
    r_nc = merged_check["niche_cohesion_t"].corr(merged_check["niche_cohesion"])
    print(f"  ρ(OS_t, static OS) = {r_os:.3f}")
    print(f"  ρ(NC_t, static NC) = {r_nc:.3f}")

# Save
save_cols = ["Company ID", "Companies", "first_date", "first_year",
             "loc5_t", "reg50_t", "z_loc", "z_reg",
             "overall_sparsity_t", "niche_cohesion_t"]
out_path = os.path.join(OUT_DIR, "novelty_time_aware.csv")
tanov_ok[save_cols].to_csv(out_path, index=False)
print(f"  Saved → {out_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2  Time-aware (lagged) community assignments
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: Lagged Louvain communities ──")

ALL_EDGES = []
for _, row in deals.iterrows():
    if pd.isna(row.get("Investors")):
        continue
    investors = comma_parse(row["Investors"])
    yr = int(row["year"])
    for i in range(len(investors)):
        for j in range(i + 1, len(investors)):
            ALL_EDGES.append((investors[i], investors[j], yr))

years_sorted = sorted(set(e[2] for e in ALL_EDGES))
G_cumul = nx.Graph()
lagged_comm = {}    # year t → community map (built from deals in years < t)

for yr in years_sorted:
    # Store community BEFORE adding this year's edges (lagged)
    if G_cumul.number_of_nodes() > 0:
        try:
            partition = community_louvain.best_partition(
                G_cumul, weight="weight", random_state=42
            )
        except Exception:
            partition = {}
    else:
        partition = {}
    lagged_comm[yr] = partition

    # Add this year's edges
    for inv_a, inv_b, edge_yr in ALL_EDGES:
        if edge_yr == yr:
            if G_cumul.has_edge(inv_a, inv_b):
                G_cumul[inv_a][inv_b]["weight"] += 1
            else:
                G_cumul.add_edge(inv_a, inv_b, weight=1)

print(f"  Years processed: {years_sorted[0]}–{years_sorted[-1]}")
print(f"  Investors in final network: {G_cumul.number_of_nodes():,}")

# Flatten to tidy CSV: investor, year, community
comm_rows = []
for yr, partition in lagged_comm.items():
    for inv, comm in partition.items():
        comm_rows.append({"investor": inv, "year": yr, "community_lagged": comm})
comm_df = pd.DataFrame(comm_rows)
comm_path = os.path.join(OUT_DIR, "lagged_communities.csv")
comm_df.to_csv(comm_path, index=False)
print(f"  Lagged community rows: {len(comm_df):,}")
print(f"  Saved → {comm_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3  Proposed definitions (for author review)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 3: Proposed definitions ──")

ns = pd.read_csv("output/novelty/novelty_scores.csv")
cluster_profiles = (
    ns[ns["specter_cluster"] >= 0]
    .groupby("specter_cluster")
    .agg(
        n=("name", "count"),
        keywords=("specter_keywords", "first"),
        examples=("name", lambda x: "; ".join(x.head(5))),
    )
    .sort_values("n", ascending=False)
    .reset_index()
)

# ── 3a  IRA_tech cluster proposals ──────────────────────────────────────────
IRA_PROPOSALS = {
    # IRA credit → (cluster_ids, rationale)
    "45X (Advanced Manufacturing / Battery Components)": {
        "clusters": [55, 42],
        "rationale": (
            "Cluster 55: lithium-ion batteries and energy storage packs "
            "(Romeo Power, Hithium). "
            "Cluster 42: lithium battery recycling and lithium extraction "
            "(Redwood Materials, Northvolt). "
            "Both map to the 45X manufacturing credit for battery cells, "
            "modules, electrode active materials, and critical minerals."
        ),
    },
    "Solar (ITC/PTC clean electricity)": {
        "clusters": [40],
        "rationale": (
            "Cluster 40 is the largest cluster (N=673) and centres on "
            "'solar, photovoltaic, renewable energy' (CHINT, DAS Solar, Gokin). "
            "Clean electricity production tax credit under 45Y and investment "
            "tax credit under 48E both cover solar generation."
        ),
    },
    "Wind (ITC/PTC clean electricity)": {
        "clusters": [17],
        "rationale": (
            "Cluster 17: 'wind turbines, offshore wind energy' "
            "(GE Vernova, Delijia). "
            "Maps to 45Y/48E clean electricity credits for wind."
        ),
    },
    "45V (Clean Hydrogen)": {
        "clusters": [2],
        "rationale": (
            "Cluster 2: 'hydrogen, green hydrogen, hydrogen production, "
            "fuel cell' (TSK, ZHGAS). "
            "45V directly targets green hydrogen production."
        ),
    },
    "45Q (Carbon Capture, Utilisation & Storage)": {
        "clusters": [13],
        "rationale": (
            "Cluster 13: 'carbon dioxide capture, direct air capture' "
            "(Solugen, Compact Carbon Capture). "
            "45Q applies to CCUS projects above statutory capture thresholds."
        ),
    },
    "30D / 45W (Clean Vehicles)": {
        "clusters": [4, 3],
        "rationale": (
            "Cluster 4: 'electric vehicles, mobility' (Li Auto, ZEEKR). "
            "Cluster 3: 'EV charging stations, vehicle charging' "
            "(Star Charge, Wallbox). "
            "30D covers consumer clean vehicle purchases; 45W covers "
            "commercial vehicles. Charging infrastructure also qualifies "
            "under 30C Alternative Fuel Refuelling Property."
        ),
    },
    "48E (Clean Electricity Storage)": {
        "clusters": [56, 57, 54],
        "rationale": (
            "Cluster 56: 'energy storage, power storage systems' (Trina Storage). "
            "Cluster 57: 'battery storage, grid-scale storage' (Energy Vault). "
            "Cluster 54: 'thermal energy storage, heat storage' (Energy Dome, Accenta). "
            "48E (investment tax credit) covers standalone energy storage ≥5 kWh "
            "connected to the grid, including battery and thermal storage."
        ),
    },
    "Geothermal (45Y/48C)": {
        "clusters": [5],
        "rationale": (
            "Cluster 5: 'geothermal energy, drilling, wells' "
            "(HyperSciences, Lithium de France). "
            "Geothermal qualifies under 45Y clean electricity PTC "
            "and 48C Advanced Energy Manufacturing credit."
        ),
    },
    "Nuclear (45Y / advanced nuclear)": {
        "clusters": [0],
        "rationale": (
            "Cluster 0: 'nuclear, fusion, nuclear power' "
            "(Naarea, Standard Nuclear, Marvel Fusion). "
            "Advanced nuclear qualifies for 45Y clean electricity PTC."
        ),
    },
}

print("\n  [3a] IRA_tech cluster proposals")
print("  " + "─" * 66)
for credit, info in IRA_PROPOSALS.items():
    cids = info["clusters"]
    cnts = cluster_profiles[cluster_profiles["specter_cluster"].isin(cids)]["n"].sum()
    kw_list = []
    for cid in cids:
        row_kw = cluster_profiles[cluster_profiles["specter_cluster"] == cid]["keywords"]
        if len(row_kw):
            kw_list.append(f"[{cid}] {row_kw.values[0]}")
    print(f"\n  {credit}  (N={cnts:,})")
    print(f"    Clusters: {cids}")
    for kw in kw_list:
        print(f"    Keywords: {kw}")
    for line in textwrap.wrap(info["rationale"], width=64):
        print(f"    {line}")

# IRA_tech cluster ID set
IRA_CLUSTER_IDS = sorted({c for v in IRA_PROPOSALS.values() for c in v["clusters"]})
print(f"\n  IRA_tech clusters (proposed): {IRA_CLUSTER_IDS}")
ira_n = cluster_profiles[cluster_profiles["specter_cluster"].isin(IRA_CLUSTER_IDS)]["n"].sum()
print(f"  Startups in IRA_tech clusters: {ira_n:,} "
      f"/ {cluster_profiles['n'].sum():,} ({100*ira_n/cluster_profiles['n'].sum():.1f}%)")

# ── 3b  Hardware-heavy cluster proposals ────────────────────────────────────
print("\n  [3b] Hardware-heavy cluster proposals")
print("  " + "─" * 66)

HARDWARE_PROPOSALS = {
    55: "Battery packs / cell manufacturing — fabrication-intensive",
    42: "Battery recycling / lithium extraction — physical process",
    40: "Solar PV — module & panel manufacturing",
    17: "Wind turbines — mechanical/electrical hardware",
     2: "Hydrogen electrolysers — hardware plant",
    13: "Direct-air capture — capital-intensive physical plant",
     3: "EV charging hardware — grid-connected devices",
    57: "Grid-scale battery storage — rack/inverter systems",
    54: "Thermal energy storage — physical heat reservoirs",
     5: "Geothermal drilling — subsurface capital equipment",
    62: "HVAC / heat-pump hardware — mechanical systems",
    60: "Heat-pump installations — physical hardware + installation",
    16: "eVTOL / electric aircraft — aerospace hardware",
    21: "Maritime propulsion — naval engineering hardware",
    43: "Concrete/cement decarbonisation — materials manufacturing",
    45: "Bio-fibres and lignin materials — physical material production",
    51: "Low-carbon construction materials — physical manufacturing",
}

for cid, rationale in HARDWARE_PROPOSALS.items():
    row = cluster_profiles[cluster_profiles["specter_cluster"] == cid]
    if len(row):
        n = row["n"].values[0]
        kw = row["keywords"].values[0]
        ex = row["examples"].values[0]
        print(f"\n  Cluster {cid}  (N={n})")
        print(f"    Keywords: {kw}")
        print(f"    Examples: {ex}")
        print(f"    Rationale: {rationale}")

HARDWARE_CLUSTER_IDS = sorted(HARDWARE_PROPOSALS.keys())
hw_n = cluster_profiles[cluster_profiles["specter_cluster"].isin(HARDWARE_CLUSTER_IDS)]["n"].sum()
print(f"\n  Hardware clusters (proposed): {HARDWARE_CLUSTER_IDS}")
print(f"  Startups: {hw_n:,}")

# ── 3c  Niche identification (mutual 5-NN, ≥3 members, above-median NC_t) ──
print("\n  [3c] Niche identification (mutual 5-NN cliques ≥3, above-median niche_cohesion_t)")
print("  " + "─" * 66)

# Use the full embedding matrix for 5-NN (need enough startups)
# Find mutual 5-NN pairs: i and j are mutual 5-NN if i ∈ 5-NN(j) AND j ∈ 5-NN(i)
K_NICHE = 5
nn_full = NearestNeighbors(n_neighbors=K_NICHE + 1, metric="cosine",
                           algorithm="brute", n_jobs=-1).fit(emb_arr)
dists_all, idxs_all = nn_full.kneighbors(emb_arr)

# Build mutual adjacency
n_emb = len(emb_names)
mutual_adj = {i: set() for i in range(n_emb)}
for i in range(n_emb):
    nbrs = set(idxs_all[i, 1:])   # exclude self
    for j in nbrs:
        if i in set(idxs_all[j, 1:]):
            mutual_adj[i].add(j)
            mutual_adj[j].add(i)

# Find connected components in mutual graph (proxy for mutual NN cliques)
# Then keep components of size ≥3
visited = [False] * n_emb
components = []
for start in range(n_emb):
    if not visited[start]:
        stack = [start]
        comp = []
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                comp.append(node)
                stack.extend(mutual_adj[node] - set(comp))
        if len(comp) >= 3:
            components.append(comp)

# Filter to above-median niche_cohesion_t
# Map emb index → NC_t
emb_to_nc = tanov_ok.set_index("emb_idx")["niche_cohesion_t"].to_dict()
nc_median = np.nanmedian(list(emb_to_nc.values()))

niche_list = []
for comp in components:
    nc_vals = [emb_to_nc.get(idx, np.nan) for idx in comp]
    nc_vals_clean = [v for v in nc_vals if not np.isnan(v)]
    if len(nc_vals_clean) == 0:
        continue
    mean_nc = np.mean(nc_vals_clean)
    if mean_nc > nc_median:
        niche_list.append({
            "size": len(comp),
            "mean_nc": mean_nc,
            "members": [emb_names[i] for i in comp],
        })

niche_list = sorted(niche_list, key=lambda x: -x["mean_nc"])

print(f"  Mutual 5-NN components ≥3 members (any NC): {len(components):,}")
print(f"  Retained (above-median NC_t = {nc_median:.3f}): {len(niche_list):,}")
sizes = [n["size"] for n in niche_list]
if sizes:
    print(f"  Size distribution: min={min(sizes)}, median={np.median(sizes):.0f}, "
          f"max={max(sizes)}, mean={np.mean(sizes):.1f}")
print("\n  Top 10 niches by mean niche_cohesion_t:")
for i, niche in enumerate(niche_list[:10]):
    members_str = "; ".join(niche["members"][:4])
    if len(niche["members"]) > 4:
        members_str += f" (+{len(niche['members'])-4} more)"
    print(f"  {i+1:2d}. size={niche['size']:3d}  NC_t={niche['mean_nc']:.3f}  {members_str}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4  Pre-registration file
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 4: Pre-registration file ──")

timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

preregistration = f"""# Pre-registration: IRA Natural Experiment — Climate-Tech Novelty & Network Structure

**Timestamp:** {timestamp}
**Study:** How did the Inflation Reduction Act (IRA, signed 2022-08-16) reshape the
novelty profile and investor network structure of climate-tech startups?
**Repository:** Breaking the Echo Chamber (climate_analysis)
**Contact:** teocara98@gmail.com

---

## Design

**Natural experiment:** IRA signed 2022-08-16. Treated group = IRA_tech startups
(mapped from SPECTER2 cluster IDs below). Control = non-IRA_tech climate-tech startups.

**Pre/post cutoff:** 2022-08-16 (IRA signing date). Startups whose first funding
round occurred on or after this date are "post-IRA".

**Geographic dimension:** US startups (HQ Country = United States) vs. non-US.
US startups are the intended beneficiaries of IRA credits; non-US startups serve
as a geographic comparison arm.

**IRA_tech clusters (proposed; not final until author approval):**
```
Credit category                    SPECTER2 cluster IDs
───────────────────────────────────────────────────────
45X  Battery manufacturing/components  55, 42
Solar  (45Y / 48E)                     40
Wind   (45Y / 48E)                     17
45V  Clean hydrogen                     2
45Q  Carbon capture (CCUS)             13
30D/45W  Clean vehicles & charging     4, 3
48E  Standalone energy storage         56, 57, 54
Geothermal  (45Y / 48C)                5
Nuclear  (45Y advanced)                0
```

**Multiple-testing correction:** Holm sequential within each family (A–D).
**Software:** statsmodels ≥ 0.14, scikit-learn ≥ 1.3, Python ≥ 3.10.

---

## Hypotheses

### Family A — Entry and scale (DiD: unit = startup)

> Primary estimand: difference-in-differences.
> Post = first round ≥ 2022-08-16. Treated = IRA_tech cluster.

| Hypothesis | Unit | Dependent variable | Key variable | Predicted sign | Family |
|---|---|---|---|---|---|
| **H1** IRA accelerates entry into IRA_tech categories. US IRA_tech startups see a larger post-IRA increase in annual founding counts than US non-IRA-tech startups, and larger than non-US IRA_tech startups. | Startup | Annual entry count (agg.) | US × IRA_tech × post-IRA | **+** | A |
| **H2** IRA attracts larger syndicated rounds. US IRA_tech startups raised more capital per round after IRA than before, relative to non-IRA-tech startups. | Startup-round | log_capital | US × IRA_tech × post-IRA | **+** | A |
| **H3** IRA broadens the syndicate. US IRA_tech startups post-IRA have more investors per round than the comparison cells. | Startup-round | log(# investors) | US × IRA_tech × post-IRA | **+** | A |

Holm correction across H1–H3.

---

### Family B — Innovation novelty (DiD: unit = startup at first round)

> Primary outcome: **overall_sparsity_t** (time-aware, higher = more novel / isolated).
> Secondary: **niche_cohesion_t** (higher = niche specialist).
> Both computed as of the startup's first-round year t, using only prior startups.

| Hypothesis | Unit | Dependent variable | Key variable | Predicted sign | Family |
|---|---|---|---|---|---|
| **H4** IRA shifts novelty of US IRA_tech entrants upward. The IRA's large and legible subsidies attract bold novel entrants filling unexplored technological niches rather than incremental imitators. Post-IRA US IRA_tech startups show higher overall_sparsity_t than pre-IRA US IRA_tech startups, controlling for the secular novelty trend (non-IRA-tech and non-US comparison cells). | Startup | **overall_sparsity_t** | US × IRA_tech × post-IRA | **+** | B |
| **H5** IRA increases niche specialisation among US IRA_tech entrants. Subsidy certainty allows startups to commit to tight technology niches. Post-IRA US IRA_tech startups show higher niche_cohesion_t relative to comparison cells. | Startup | niche_cohesion_t | US × IRA_tech × post-IRA | **+** | B |

Holm correction across H4–H5.

---

### Family C — Network structure (DiD: unit = startup × investor dyad or startup level)

> Tests whether IRA reshaped the investor network around IRA_tech startups.

| Hypothesis | Unit | Dependent variable | Key variable | Predicted sign | Family |
|---|---|---|---|---|---|
| **H6** IRA attracts heterogeneous investor types to IRA_tech. After IRA, US IRA_tech startups see a larger increase in pct_ivc (independent VC participation) relative to pre-IRA and relative to non-IRA-tech. (Alternative: pct_cvc if corporate investors respond to supply-chain incentives.) | Startup | pct_ivc | US × IRA_tech × post-IRA | **+** | C |
| **H7** IRA reduces community siloing in IRA_tech investment. Post-IRA US IRA_tech syndicates span more Louvain communities (excess_spread, Script 81) than pre-IRA syndicates and non-IRA-tech post-IRA syndicates. | Startup | excess_spread (Script 81) | US × IRA_tech × post-IRA | **+** | C |
| **H8** IRA's network effect is concentrated in hardware clusters. The cross-community syndication increase (H7) is larger for hardware-intensive IRA_tech startups (45X, 48E, geothermal) than for software/service IRA_tech startups. | Startup | excess_spread | hardware × post-IRA (within IRA_tech) | **+** | C |

Holm correction across H6–H8.

---

### Family D — Heterogeneity and robustness

| Hypothesis | Unit | Dependent variable | Key variable | Predicted sign | Family |
|---|---|---|---|---|---|
| **H9** IRA novelty effect is concentrated in physically intensive credits. The H4 novelty increase is larger for 45X (batteries), 45V (hydrogen), and 45Q (CCUS) — credits with specific technical requirements and minimum efficiency thresholds — than for the broader clean electricity credits (45Y/48E). | Startup | overall_sparsity_t | credit-category × post-IRA (within US IRA_tech) | 45X>45Y | D |
| **H10** Non-US startups show no IRA novelty effect. Since IRA credits require US domestic production, non-US startups in IRA_tech categories should show no change in novelty profile around 2022-08-16. This is a placebo test for H4. | Startup | overall_sparsity_t | non-US × IRA_tech × post-IRA | **≈0** (placebo) | D |

Holm correction across H9–H10.

---

## Estimating equation (H4 primary, analogous for others)

```
overall_sparsity_t = β₀
    + β₁ · post_IRA
    + β₂ · IRA_tech
    + β₃ · US
    + β₄ · (US × IRA_tech)
    + β₅ · (US × post_IRA)
    + β₆ · (IRA_tech × post_IRA)
    + **β₇ · (US × IRA_tech × post_IRA)**    ← H4 coefficient
    + γ · log_capital
    + year FE + country FE
    + ε
```

Standard errors: HC1 (OLS) or, if sample large enough, clustered by first-round year × country.

---

## Variables

| Variable | Source | Notes |
|---|---|---|
| overall_sparsity_t | output/novelty_t/novelty_time_aware.csv | Time-aware; higher = more isolated |
| niche_cohesion_t | output/novelty_t/novelty_time_aware.csv | Time-aware; higher = niche specialist |
| IRA_tech | SPECTER2 cluster → IRA category mapping (above) | Binary indicator |
| hardware | Hardware cluster mapping (Step 3b) | Sub-indicator of IRA_tech |
| US | HQ Country = "United States" | Binary |
| post_IRA | first_date ≥ 2022-08-16 | Binary |
| log_capital | log(1 + Total Raised USD M) | From cpu_company_dataset |
| excess_spread | output from Script 81 pipeline | Monte Carlo community spread |
| pct_ivc | pct_ivc from cpu_company_dataset | Fraction of round from IVC investors |
| lagged_community | output/novelty_t/lagged_communities.csv | Year-lagged Louvain community ID |

---

## What is NOT pre-registered here

- Any model-selection decisions after seeing the data (covariate choice, functional form)
- Survival / failure analyses (separate analysis family)
- Script 79r moderation hypotheses (P1–P3; separately pre-registered in Scripts 79–80)
"""

prereg_path = os.path.join(os.path.dirname(__file__), "output", "preregistration.md")
with open(prereg_path, "w") as f:
    f.write(preregistration)
print(f"  Pre-registration saved → {prereg_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5  Feasibility check for H4
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 5: H4 feasibility — cell counts ──")
print("  (US/non-US × IRA_tech/not × pre/post 2022-08-16)")

# Build working sample: startups with time-aware novelty + cluster label
ns_for_feas = ns[["name", "specter_cluster"]].rename(columns={"name": "Companies"})
cpu = pd.read_csv("output/cpu/cpu_company_dataset.csv")[
    ["Companies", "is_us", "hq_country"]
]

# Merge
feas = tanov_ok[["Company ID", "Companies", "first_date", "overall_sparsity_t"]].copy()
feas = feas.merge(cpu, on="Companies", how="left")
feas = feas.merge(ns_for_feas, on="Companies", how="left")

# IRA_tech flag
feas["IRA_tech"] = feas["specter_cluster"].isin(IRA_CLUSTER_IDS).astype(int)

# Noise / unmatched → not IRA_tech
feas.loc[feas["specter_cluster"].isna(), "IRA_tech"] = 0
feas.loc[feas["specter_cluster"] == -1, "IRA_tech"] = 0

# US flag (from cpu dataset; fall back to HQ country)
feas["is_US"] = feas["is_us"].fillna(
    (feas["hq_country"] == "United States").astype(float)
).astype(int)

# Post-IRA flag
feas["post_IRA"] = (feas["first_date"] >= IRA_DATE).astype(int)

# Cell counts
cell_counts = (
    feas.groupby(["is_US", "IRA_tech", "post_IRA"])
    .size()
    .reset_index(name="N")
)
cell_counts["US_label"]      = cell_counts["is_US"].map({1: "US", 0: "non-US"})
cell_counts["IRA_label"]     = cell_counts["IRA_tech"].map({1: "IRA_tech", 0: "non-IRA_tech"})
cell_counts["post_IRA_label"]= cell_counts["post_IRA"].map({1: "post-IRA (≥2022-08-16)", 0: "pre-IRA (<2022-08-16)"})

print()
print(f"  {'Geography':<12}  {'IRA_tech':<14}  {'Period':<28}  {'N':>6}  {'Flag'}")
print("  " + "─" * 72)
for _, row in cell_counts.iterrows():
    flag = "  *** UNDER 100 ***" if row["N"] < 100 else ""
    print(f"  {row['US_label']:<12}  {row['IRA_label']:<14}  "
          f"{row['post_IRA_label']:<28}  {row['N']:>6}{flag}")

print()
n_flagged = (cell_counts["N"] < 100).sum()
print(f"  Cells flagged (N < 100): {n_flagged} / {len(cell_counts)}")
print()
print("  Summary totals:")
print(f"    Total startups in feasibility sample:  {len(feas):,}")
print(f"    IRA_tech startups:                     {feas['IRA_tech'].sum():,}")
print(f"    US startups:                           {feas['is_US'].sum():,}")
print(f"    US × IRA_tech startups:                {(feas['is_US'] & feas['IRA_tech'].astype(bool)).sum():,}")
print(f"    US × IRA_tech × post-IRA:              {(feas['is_US'] & feas['IRA_tech'].astype(bool) & feas['post_IRA'].astype(bool)).sum():,}")

# Also report by specific IRA credit category for H9 heterogeneity
print("\n  H9 breakdown — US startups by credit category × post-IRA:")
print(f"  {'Credit category':<44}  {'Pre-IRA':>8}  {'Post-IRA':>9}")
print("  " + "─" * 65)
us_feas = feas[feas["is_US"] == 1].copy()
for credit, info in IRA_PROPOSALS.items():
    cids = info["clusters"]
    mask = us_feas["specter_cluster"].isin(cids)
    pre  = (mask & (us_feas["post_IRA"] == 0)).sum()
    post = (mask & (us_feas["post_IRA"] == 1)).sum()
    flag = "  *" if min(pre, post) < 30 else ""
    cat_short = credit[:43]
    print(f"  {cat_short:<44}  {pre:>8}  {post:>9}{flag}")

print()
print("=" * 70)
print("SCRIPT 83 COMPLETE — review output/preregistration.md before proceeding")
print("=" * 70)
