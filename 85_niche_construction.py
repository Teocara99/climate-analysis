"""
Script 85: Finalise niches and check H3 feasibility.
NO outcome regressions.

1. NICHE CONSTRUCTION: mutual 3-NN components; keep those with
   mean internal cosine similarity > 90th percentile of all pairwise
   similarities; size cap 30.
2. For each niche, order by first-round date. Founding community =
   plurality lagged community of the founding startup's first-round
   investors.
3. FEASIBILITY: niches with >=2 later entrants; later-entrant dyads;
   share of founding investors with a lagged community.
4. Update preregistration.md with niche rule and H3 spec. Lock it.
"""

import os, sys, re, warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

NICHE_DIR = os.path.join(os.path.dirname(__file__), "output", "niches")
os.makedirs(NICHE_DIR, exist_ok=True)

K_MNN   = 3   # mutual nearest-neighbours
SIZE_CAP = 30  # max niche size

print("=" * 70)
print("SCRIPT 85 — NICHE CONSTRUCTION & H3 FEASIBILITY")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD INPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading inputs ──")

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    spec_data = pickle.load(f)
emb_names = spec_data["names"]
emb_raw   = np.array(spec_data["embs_raw"], dtype=np.float32)
norms = np.linalg.norm(emb_raw, axis=1, keepdims=True); norms[norms == 0] = 1.0
emb_arr = emb_raw / norms          # L2 normalised, cosine sim = dot product
n_emb   = len(emb_names)
emb_name_to_idx = {n: i for i, n in enumerate(emb_names)}

nov_t = pd.read_csv("output/novelty_t/novelty_time_aware.csv",
                    parse_dates=["first_date"])

lagged_comm = pd.read_csv("output/novelty_t/lagged_communities.csv")
# investor × year → community_lagged  (quick lookup dict)
lc_lookup = {}   # (investor, year) → community
for _, row in lagged_comm.iterrows():
    lc_lookup[(row["investor"], int(row["year"]))] = int(row["community_lagged"])

deals_df = load_deals()
deals_df["Deal Date"] = pd.to_datetime(deals_df["Deal Date"], errors="coerce")
deals_df = deals_df.dropna(subset=["Deal Date"])
deals_df["year"] = deals_df["Deal Date"].dt.year

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

# First-round investors per company
first_round_investors = {}
for cid, grp in deals_df.sort_values("Deal Date").groupby("Company ID"):
    row = grp.iloc[0]
    if pd.notna(row.get("Investors")):
        first_round_investors[cid] = comma_parse(row["Investors"])
    else:
        first_round_investors[cid] = []

print(f"  Startup embeddings: {n_emb:,}")
print(f"  nov_t rows: {len(nov_t):,}")
print(f"  Lagged community entries: {len(lagged_comm):,}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1  Global 90th percentile of pairwise cosine similarities
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 1: Global similarity distribution ──")

np.random.seed(42)
idx_a = np.random.choice(n_emb, 100_000, replace=True)
idx_b = np.random.choice(n_emb, 100_000, replace=True)
mask  = idx_a != idx_b
global_sims = np.einsum("ij,ij->i", emb_arr[idx_a[mask]], emb_arr[idx_b[mask]])
SIM_THRESHOLD = float(np.percentile(global_sims, 90))
print(f"  p50={np.percentile(global_sims,50):.4f}  p75={np.percentile(global_sims,75):.4f}  "
      f"p90={SIM_THRESHOLD:.4f}  p95={np.percentile(global_sims,95):.4f}")
print(f"  Niche similarity threshold (p90): {SIM_THRESHOLD:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2  Mutual 3-NN graph and connected components
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: Mutual 3-NN components ──")

nn = NearestNeighbors(n_neighbors=K_MNN + 1, metric="cosine",
                      algorithm="brute", n_jobs=-1).fit(emb_arr)
dists, idxs = nn.kneighbors(emb_arr)   # (n_emb, K_MNN+1); first is self (dist=0)

# Mutual adjacency: i–j edge if j ∈ K_MNN(i) AND i ∈ K_MNN(j)
nbr_sets = [set(idxs[i, 1:]) for i in range(n_emb)]   # exclude self
mutual_adj = [set() for _ in range(n_emb)]
for i in range(n_emb):
    for j in nbr_sets[i]:
        if i in nbr_sets[j]:
            mutual_adj[i].add(j)

# Connected components via BFS
visited = np.zeros(n_emb, dtype=bool)
raw_components = []
for start in range(n_emb):
    if visited[start]:
        continue
    comp = []
    queue = [start]
    while queue:
        node = queue.pop()
        if visited[node]:
            continue
        visited[node] = True
        comp.append(node)
        queue.extend(mutual_adj[node] - set(comp))
    if len(comp) >= 3:
        raw_components.append(comp)

print(f"  Mutual 3-NN components of size ≥ 3: {len(raw_components):,}")
sizes_raw = sorted(len(c) for c in raw_components)
print(f"  Size distribution: min={min(sizes_raw)}, median={np.median(sizes_raw):.0f}, "
      f"max={max(sizes_raw)}, >30: {sum(1 for s in sizes_raw if s > 30)}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3  Filter by similarity threshold and size cap
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 3: Apply similarity threshold and size cap ──")

niches = []
for comp in raw_components:
    if len(comp) > SIZE_CAP:
        continue  # size backstop
    comp_emb = emb_arr[comp]
    # Mean pairwise similarity (all pairs within component)
    sim_mat = comp_emb @ comp_emb.T
    np.fill_diagonal(sim_mat, 0.0)
    n_c = len(comp)
    mean_sim = sim_mat.sum() / (n_c * (n_c - 1)) if n_c > 1 else 0.0
    if mean_sim >= SIM_THRESHOLD:
        niches.append({"indices": comp, "mean_sim": mean_sim, "n": n_c})

print(f"  After size cap (≤{SIZE_CAP}): {sum(1 for c in raw_components if len(c) <= SIZE_CAP):,} components")
print(f"  After similarity filter (mean_sim ≥ {SIM_THRESHOLD:.4f}): {len(niches):,} niches")

# Sort by mean_sim descending
niches = sorted(niches, key=lambda x: -x["mean_sim"])

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4  Assign niche IDs; founding startup; founding community
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 4: Founding startup and founding community ──")

# Index: Company ID → first_date, first_year (from nov_t)
nov_t_idx = nov_t.set_index("Companies")[["Company ID", "first_date", "first_year",
                                           "overall_sparsity_t", "niche_cohesion_t"]].copy()

niche_records   = []   # one row per niche (profile)
member_records  = []   # one row per startup (assignment)

for niche_id, niche in enumerate(niches):
    comp      = niche["indices"]
    mean_sim  = niche["mean_sim"]
    members   = [emb_names[i] for i in comp]

    # Get first-round dates for members that have them
    dated = []
    for name in members:
        if name in nov_t_idx.index:
            row = nov_t_idx.loc[name]
            dated.append({
                "company": name,
                "Company ID": row["Company ID"],
                "first_date": row["first_date"],
                "first_year": int(row["first_year"]),
                "OS_t": row["overall_sparsity_t"],
                "NC_t": row["niche_cohesion_t"],
            })
    if len(dated) < 2:
        # Need at least founder + 1 later entrant
        continue
    dated = sorted(dated, key=lambda x: x["first_date"])

    # Founding startup = earliest first-round date
    founder   = dated[0]
    founder_id = founder["Company ID"]
    t0        = founder["first_year"]

    # Founding community = plurality lagged community of founder's investors at t0
    founder_investors = first_round_investors.get(founder_id, [])
    comm_votes = {}
    for inv in founder_investors:
        c = lc_lookup.get((inv, t0))
        if c is not None:
            comm_votes[c] = comm_votes.get(c, 0) + 1
    if comm_votes:
        founding_community = max(comm_votes, key=comm_votes.get)
        n_with_comm = sum(comm_votes.values())
        n_founder_inv = len(founder_investors)
    else:
        founding_community = None
        n_with_comm = 0
        n_founder_inv = len(founder_investors)

    # Member records
    for rank, d in enumerate(dated):
        position = "founder" if rank == 0 else "later"
        member_records.append({
            "niche_id":           niche_id,
            "Companies":          d["company"],
            "Company ID":         d["Company ID"],
            "first_date":         d["first_date"],
            "first_year":         d["first_year"],
            "position":           position,
            "overall_sparsity_t": d["OS_t"],
            "niche_cohesion_t":   d["NC_t"],
        })

    niche_records.append({
        "niche_id":           niche_id,
        "mean_sim":           mean_sim,
        "n_members":          len(dated),    # with first-round dates
        "n_all":              len(members),  # total emb members
        "founding_startup":   founder["company"],
        "founding_year":      t0,
        "founding_community": founding_community,
        "n_founder_inv":      n_founder_inv,
        "n_founder_inv_w_comm": n_with_comm,
        "members":            "; ".join(m["company"] for m in dated),
    })

niche_df  = pd.DataFrame(niche_records)
member_df = pd.DataFrame(member_records)

print(f"  Niches with ≥2 dated members: {len(niche_df):,}")
print(f"  Niches with founding community assigned: "
      f"{niche_df['founding_community'].notna().sum():,}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5  Report niche examples
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 5: Niche summary ──")

sizes = niche_df["n_members"].values
print(f"  N niches: {len(niche_df):,}")
print(f"  Size distribution: "
      f"min={sizes.min()}, median={np.median(sizes):.0f}, max={sizes.max()}, "
      f"mean={sizes.mean():.1f}")

print("\n  Top 10 niches by mean internal similarity:")
print(f"  {'ID':>4}  {'n':>3}  {'sim':>6}  {'Founding yr':>10}  Members (first 4)")
print("  " + "─" * 72)
for _, row in niche_df.head(10).iterrows():
    members_short = "; ".join(row["members"].split("; ")[:4])
    if row["n_members"] > 4:
        members_short += f" (+{row['n_members']-4})"
    print(f"  {int(row['niche_id']):>4}  {row['n_members']:>3}  "
          f"{row['mean_sim']:.4f}  {row['founding_year']:>10}  {members_short}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6  Feasibility check
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 6: H3 feasibility ──")

later_mask    = member_df["position"] == "later"
niches_w2plus = niche_df[niche_df["n_members"] >= 3]  # ≥2 later entrants = ≥3 members
later_df      = member_df[later_mask].copy()

# Estimate dyad count (each later entrant × risk set ~6 = 1 real + 5 counterfactual)
n_dyads_est = len(later_df) * 6

# Founding investor community coverage
total_inv       = niche_df["n_founder_inv"].sum()
inv_with_comm   = niche_df["n_founder_inv_w_comm"].sum()
pct_with_comm   = 100 * inv_with_comm / max(total_inv, 1)

print(f"\n  Niches with ≥2 later entrants (≥3 members total): {len(niches_w2plus):,}")
print(f"  Later entrants: {len(later_df):,}")
print(f"  Estimated dyads (×6 risk set): ~{n_dyads_est:,}")
print(f"  Founding investor community coverage: "
      f"{inv_with_comm}/{total_inv} = {pct_with_comm:.1f}%")

# By niche size
print(f"\n  Niche size distribution:")
for sz in sorted(niche_df["n_members"].unique()):
    cnt = (niche_df["n_members"] == sz).sum()
    print(f"    size {sz}: {cnt:>4} niches")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
niche_path  = os.path.join(NICHE_DIR, "niche_profiles.csv")
member_path = os.path.join(NICHE_DIR, "niche_assignments.csv")
niche_df.to_csv(niche_path,  index=False)
member_df.to_csv(member_path, index=False)
print(f"\n  Niche profiles → {niche_path}")
print(f"  Member assignments → {member_path}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7  Update pre-registration
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 7: Updating preregistration.md ──")

PREREG_PATH = os.path.join(os.path.dirname(__file__), "output", "preregistration.md")
with open(PREREG_PATH) as f:
    prereg = f.read()

NICHE_RULE = f"""
---

## Niche definition (finalised 2026-09-21, before H3 runs)

A niche is a connected component of the mutual 3-nearest-neighbour graph on
SPECTER2 startup embeddings satisfying:
- Mean internal cosine similarity ≥ {SIM_THRESHOLD:.4f} (90th percentile of
  100,000 randomly sampled pairwise similarities in the corpus)
- Size ≤ 30 (backstop; drops the main-stream mega-components)

This yielded **{len(niche_df)} niches** with ≥2 dated members (total dated
member rows: {len(member_df)}).

**Founding startup** = niche member with the earliest first-round date.
**Founding community** = plurality lagged Louvain community of the founding
startup's first-round investors (lagged to deals before year t₀).

### H3 specification (locked)

Unit: dyad (later entrant × candidate investor).
Dependent variable: tie (1 = realised, 0 = counterfactual).
Risk set: per startup — actual first-round investors + 5 counterfactual
  investors active in that year, drawn from the population of investors
  with any deal in that year; founding startup's first-round investors
  are excluded from both realised and counterfactual sets.
Model: conditional logit stratified by startup.

```
tie ~ in_founding_community + semantic_distance
    + geo_same_country + geo_same_region
    + same_community_t + log_prior_deals + has_prior_network
```

Pre-specified sign on in_founding_community: **POSITIVE**.
Placebo: replace founding community with random community of similar size,
  500 draws; report permutation p.
"""

# Insert niche rule before "## What is NOT pre-registered"
INSERT_BEFORE = "## What is NOT pre-registered here"
prereg_updated = prereg.replace(INSERT_BEFORE, NICHE_RULE + "\n\n---\n\n" + INSERT_BEFORE)

# Update lock timestamp
from datetime import datetime
ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
prereg_updated = prereg_updated.replace(
    "**Revision timestamp:** 2026-09-21 16:42 UTC",
    f"**Revision timestamp:** {ts}"
)

with open(PREREG_PATH, "w") as f:
    f.write(prereg_updated)
print(f"  Pre-registration updated and locked → {PREREG_PATH}")

print("\n" + "=" * 70)
print("SCRIPT 85 COMPLETE")
print("=" * 70)
