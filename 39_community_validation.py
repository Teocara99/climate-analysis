"""
Script 39: Community Validation — Are Louvain Communities Real?
===============================================================

5 tests:
  TEST 1 — Temporal stability (NMI across time splits)
  TEST 2 — Algorithm robustness (Louvain/Leiden/Infomap at multiple resolutions)
  TEST 3 — Modularity vs. random (configuration-model null)
  TEST 4 — Geographic validation (communities recover geography)
  TEST 5 — Behavioral similarity (within > across community)
"""

import warnings; warnings.filterwarnings("ignore")
import sys, time, random, itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import normalized_mutual_info_score
import community as community_louvain   # python-louvain
import leidenalg, igraph as ig

sys.path.insert(0, ".")
from load_data import load_deals, load_companies

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ── Helpers ──────────────────────────────────────────────────────────────────
def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

def build_graph(deals_df, inv_col="Investors", min_cooccur=2):
    """Build investor co-investment graph from a deals DataFrame."""
    inv_sets = (deals_df.dropna(subset=[inv_col])
                .assign(inv=lambda d: d[inv_col].str.split(";"))
                .explode("inv")
                .assign(inv=lambda d: d["inv"].str.strip())
                .query("inv != ''")
                .groupby("Companies")["inv"].apply(list))

    edges = defaultdict(int)
    for investors in inv_sets:
        investors = list(set(investors))
        for a, b in itertools.combinations(sorted(investors), 2):
            edges[(a, b)] += 1

    G = nx.Graph()
    for (a, b), w in edges.items():
        if w >= min_cooccur:
            G.add_edge(a, b, weight=w)
    return G


def louvain_partition(G, resolution=1.0, seed=SEED):
    """Returns dict {node: community_id}."""
    return community_louvain.best_partition(G, resolution=resolution, random_state=seed, weight="weight")


def nx_to_igraph(G):
    """Convert networkx graph to igraph for Leiden/Infomap."""
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    ig_g = ig.Graph(n=len(nodes))
    ig_g.vs["name"] = nodes
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    ig_g.add_edges(edges)
    ig_g.es["weight"] = weights
    return ig_g, nodes


def igraph_to_partition(membership, nodes):
    return {nodes[i]: membership[i] for i in range(len(nodes))}


def nmi_from_partitions(p1, p2):
    """NMI for nodes present in both partitions."""
    common = set(p1) & set(p2)
    if not common:
        return np.nan
    a = [p1[n] for n in common]
    b = [p2[n] for n in common]
    return normalized_mutual_info_score(a, b, average_method="arithmetic")


def modularity_nx(G, partition):
    """Compute modularity using networkx."""
    communities = defaultdict(set)
    for node, comm in partition.items():
        communities[comm].add(node)
    return nx.community.modularity(G, list(communities.values()), weight="weight")


# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")
deals = deals.dropna(subset=["Year"])
deals["Year"] = deals["Year"].astype(int)

companies = load_companies()
companies["co"] = companies["Companies"].str.strip()

net = pd.read_csv("output/network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()

# Full network (all years)
print("Building full network...")
G_full = build_graph(deals)
print(f"  Full graph: {G_full.number_of_nodes():,} investors, {G_full.number_of_edges():,} edges")
partition_full = louvain_partition(G_full)
n_comm = len(set(partition_full.values()))
print(f"  Louvain communities: {n_comm}")

print()
print("="*72)
print("TEST 1: TEMPORAL STABILITY")
print("="*72)
t0 = time.time()

deals_early = deals[deals["Year"].between(2012, 2018)]
deals_late  = deals[deals["Year"].between(2019, 2026)]
print(f"  Early (2012–2018): {len(deals_early):,} deals")
print(f"  Late  (2019–2026): {len(deals_late):,} deals")

G_early = build_graph(deals_early)
G_late  = build_graph(deals_late)
print(f"  Early graph: {G_early.number_of_nodes():,} nodes, {G_early.number_of_edges():,} edges")
print(f"  Late  graph: {G_late.number_of_nodes():,} nodes, {G_late.number_of_edges():,} edges")

p_early = louvain_partition(G_early)
p_late  = louvain_partition(G_late)

nmi_temporal = nmi_from_partitions(p_early, p_late)
common_inv   = set(p_early) & set(p_late)
same_comm    = sum(p_early[n] == p_late[n] for n in common_inv)
# Note: raw label match understates stability since labels are arbitrary.
# Use NMI as the primary measure; also report majority-community retention.
# Better measure: for each early community, what fraction of members go to
# the same late community?
early_to_late = defaultdict(list)
for n in common_inv:
    early_to_late[p_early[n]].append(p_late[n])

retention_rates = []
for ec, late_labels in early_to_late.items():
    majority_frac = max(late_labels.count(l) for l in set(late_labels)) / len(late_labels)
    retention_rates.append(majority_frac)
avg_retention = np.mean(retention_rates)

print(f"\n  NMI (early vs late)         : {nmi_temporal:.3f}")
print(f"  Investors in both periods   : {len(common_inv):,}")
print(f"  Avg majority-community retention: {avg_retention:.1%}")
if nmi_temporal > 0.6:
    verdict1 = "PASS — high temporal stability (NMI>0.6)"
elif nmi_temporal > 0.4:
    verdict1 = "BORDERLINE — moderate stability (NMI 0.4–0.6)"
else:
    verdict1 = "FAIL — low temporal stability (NMI<0.4)"
print(f"  Verdict: {verdict1}")
print(f"  [{time.time()-t0:.1f}s]")

print()
print("="*72)
print("TEST 2: ALGORITHM & RESOLUTION ROBUSTNESS")
print("="*72)
t0 = time.time()

# Louvain at multiple resolutions
resolutions = [0.5, 1.0, 1.5, 2.0]
partitions  = {}
for r in resolutions:
    p = louvain_partition(G_full, resolution=r)
    partitions[f"Louvain_{r}"] = p
    n_c = len(set(p.values()))
    print(f"  Louvain r={r:.1f}: {n_c} communities")

# Leiden
ig_g, nodes = nx_to_igraph(G_full)
try:
    leiden_part = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition,
                                            seed=SEED, weights="weight")
    partitions["Leiden"] = igraph_to_partition(leiden_part.membership, nodes)
    print(f"  Leiden:        {len(set(leiden_part.membership))} communities")
except Exception as e:
    print(f"  Leiden: FAILED ({e})")

# Infomap (igraph built-in)
try:
    infomap_res = ig_g.community_infomap(edge_weights="weight")
    partitions["Infomap"] = igraph_to_partition(infomap_res.membership, nodes)
    print(f"  Infomap:       {len(set(infomap_res.membership))} communities")
except Exception as e:
    print(f"  Infomap: FAILED ({e})")

# NMI matrix
methods = list(partitions.keys())
print(f"\n  NMI matrix ({len(methods)} methods):")
header = "".join(f"{m:>14}" for m in methods)
print(f"  {'':20}" + header)
nmi_vals = []
for mi in methods:
    row = f"  {mi:<20}"
    for mj in methods:
        nmi = nmi_from_partitions(partitions[mi], partitions[mj])
        if mi == mj:
            row += f"{'1.000':>14}"
        else:
            row += f"{nmi:>14.3f}"
            if mi < mj:
                nmi_vals.append(nmi)
    print(row)

avg_nmi = np.mean(nmi_vals)
min_nmi = np.min(nmi_vals)
print(f"\n  Mean pairwise NMI: {avg_nmi:.3f}   Min: {min_nmi:.3f}")
if avg_nmi > 0.7:
    verdict2 = "PASS — core structure consistent across algorithms (mean NMI>0.7)"
elif avg_nmi > 0.5:
    verdict2 = "BORDERLINE — moderate consistency (mean NMI 0.5–0.7)"
else:
    verdict2 = "FAIL — structure is algorithm-dependent (mean NMI<0.5)"
print(f"  Verdict: {verdict2}")
print(f"  [{time.time()-t0:.1f}s]")

print()
print("="*72)
print("TEST 3: MODULARITY vs. RANDOM (Configuration Model)")
print("="*72)
t0 = time.time()

Q_actual = modularity_nx(G_full, partition_full)
print(f"  Actual modularity Q: {Q_actual:.4f}")

N_RAND = 500   # 500 random networks for speed (z-score stabilises by then)
print(f"  Generating {N_RAND} random networks (configuration model)…")
Q_random = []
for i in range(N_RAND):
    G_rand = nx.configuration_model(list(dict(G_full.degree()).values()), seed=SEED+i)
    G_rand = nx.Graph(G_rand)   # remove multi-edges
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    p_rand = louvain_partition(G_rand, resolution=1.0, seed=SEED+i)
    Q_random.append(modularity_nx(G_rand, p_rand))
    if (i+1) % 100 == 0:
        print(f"    {i+1}/{N_RAND}  mean Q_rand so far: {np.mean(Q_random):.4f}")

Q_rand_mean = np.mean(Q_random)
Q_rand_std  = np.std(Q_random)
z_score     = (Q_actual - Q_rand_mean) / Q_rand_std
p_val       = stats.norm.sf(z_score)   # one-tailed (actual > random)

print(f"\n  Q_actual:   {Q_actual:.4f}")
print(f"  Q_random:   {Q_rand_mean:.4f} ± {Q_rand_std:.4f}")
print(f"  Z-score:    {z_score:.1f}")
print(f"  p-value:    {p_val:.2e}")
if z_score > 5:
    verdict3 = "PASS — modularity far exceeds random expectation (z>5)"
elif z_score > 2:
    verdict3 = "BORDERLINE — above random but marginal (z 2–5)"
else:
    verdict3 = "FAIL — not significantly better than random (z<2)"
print(f"  Verdict: {verdict3}")
print(f"  [{time.time()-t0:.1f}s]")

print()
print("="*72)
print("TEST 4: GEOGRAPHIC VALIDATION")
print("="*72)
t0 = time.time()

# Map investor → country via portfolio companies' headquarters
# Use HQ country from companies file
co_country = companies.set_index("co")["HQ Country/Territory/Region"].dropna().to_dict()

# Map investor → set of portfolio company countries
inv_countries = defaultdict(list)
for _, row in (deals.dropna(subset=["Investors"])
               .assign(inv=lambda d: d["Investors"].str.split(";"))
               .explode("inv")
               .assign(inv=lambda d: d["inv"].str.strip())
               .query("inv != ''")
               .iterrows()):
    country = co_country.get(row["co"])
    if country and pd.notna(country):
        inv_countries[row["inv"]].append(country)

# Dominant country per investor = most common portfolio company country
inv_dom_country = {}
for inv, countries in inv_countries.items():
    if countries:
        inv_dom_country[inv] = max(set(countries), key=countries.count)

# Region mapping (broad)
REGION = {
    "United States": "North America", "Canada": "North America", "Mexico": "Latin America",
    "Brazil": "Latin America", "Colombia": "Latin America", "Chile": "Latin America",
    "Argentina": "Latin America", "Peru": "Latin America",
    "United Kingdom": "Europe", "Germany": "Europe", "France": "Europe",
    "Netherlands": "Europe", "Sweden": "Europe", "Denmark": "Europe",
    "Norway": "Europe", "Finland": "Europe", "Spain": "Europe", "Italy": "Europe",
    "Switzerland": "Europe", "Austria": "Europe", "Belgium": "Europe",
    "Poland": "Europe", "Portugal": "Europe", "Ireland": "Europe",
    "Israel": "Middle East/Africa", "United Arab Emirates": "Middle East/Africa",
    "Saudi Arabia": "Middle East/Africa", "South Africa": "Middle East/Africa",
    "India": "Asia-Pacific", "China": "Asia-Pacific", "Japan": "Asia-Pacific",
    "Singapore": "Asia-Pacific", "Australia": "Asia-Pacific",
    "South Korea": "Asia-Pacific", "Indonesia": "Asia-Pacific",
}
def to_region(c):
    return REGION.get(c, "Other")

inv_region = {inv: to_region(c) for inv, c in inv_dom_country.items()}

# Per-community geographic homogeneity
comm_regions = defaultdict(list)
for inv, comm in partition_full.items():
    reg = inv_region.get(inv)
    if reg:
        comm_regions[comm].append(reg)

geo_homog = []
for comm, regions in comm_regions.items():
    if len(regions) >= 3:
        majority_frac = max(regions.count(r) for r in set(regions)) / len(regions)
        geo_homog.append(majority_frac)

pct_gt70 = np.mean([x >= 0.70 for x in geo_homog]) * 100
pct_gt80 = np.mean([x >= 0.80 for x in geo_homog]) * 100
print(f"  Communities with ≥3 mapped investors: {len(geo_homog)}")
print(f"  % communities where dominant region ≥70% of members: {pct_gt70:.1f}%")
print(f"  % communities where dominant region ≥80% of members: {pct_gt80:.1f}%")
print(f"  Mean geographic homogeneity per community: {np.mean(geo_homog):.1%}")

# Chi-square: community × region
records = [(partition_full[inv], inv_region[inv])
           for inv in partition_full if inv in inv_region]
df_geo = pd.DataFrame(records, columns=["community","region"])
ct = pd.crosstab(df_geo["community"], df_geo["region"])
chi2, p_chi2, dof, _ = chi2_contingency(ct)
cramers_v = np.sqrt(chi2 / (len(df_geo) * (min(ct.shape)-1)))
print(f"\n  Chi-square (community × region): χ²={chi2:.1f}, df={dof}, p={p_chi2:.2e}")
print(f"  Cramér's V (effect size):        {cramers_v:.3f}")

# Compare to random partition
# Random: shuffle community labels, compute homogeneity
rand_homog_vals = []
inv_list = [inv for inv in partition_full if inv in inv_region]
comm_labels = [partition_full[inv] for inv in inv_list]
for _ in range(200):
    rand_labels = comm_labels.copy()
    random.shuffle(rand_labels)
    rc = defaultdict(list)
    for inv, rl in zip(inv_list, rand_labels):
        rc[rl].append(inv_region[inv])
    rvals = [max(v.count(r) for r in set(v))/len(v) for v in rc.values() if len(v)>=3]
    rand_homog_vals.append(np.mean(rvals))

rand_mean = np.mean(rand_homog_vals)
rand_std  = np.std(rand_homog_vals)
geo_z = (np.mean(geo_homog) - rand_mean) / rand_std
print(f"\n  Actual  mean geographic homogeneity: {np.mean(geo_homog):.3f}")
print(f"  Random  mean geographic homogeneity: {rand_mean:.3f} ± {rand_std:.3f}")
print(f"  Z-score vs random:                   {geo_z:.1f}")

if pct_gt70 > 60 and cramers_v > 0.3:
    verdict4 = "PASS — communities are geographically concentrated far beyond chance"
elif pct_gt70 > 40:
    verdict4 = "BORDERLINE — partial geographic concentration"
else:
    verdict4 = "FAIL — communities not geographically homogeneous"
print(f"  Verdict: {verdict4}")
print(f"  [{time.time()-t0:.1f}s]")

print()
print("="*72)
print("TEST 5: BEHAVIORAL SIMILARITY (Within vs. Across Community)")
print("="*72)
t0 = time.time()

# Build investor profiles from deals
print("  Building investor profiles...")

# Stage preference
stage_col = next((c for c in deals.columns if "stage" in c.lower() or "round" in c.lower()), None)
if stage_col is None:
    stage_col = "Deal Type"  # fallback

# Sector preference — use Primary PitchBook Industry Group from companies
co_sector = companies.set_index("co")["Primary PitchBook Industry Group"].dropna().to_dict()

# Build per-investor profiles
inv_deals = defaultdict(list)   # investor → list of (co, year, stage, sector)
for _, row in (deals.dropna(subset=["Investors"])
               .assign(inv=lambda d: d["Investors"].str.split(";"))
               .explode("inv")
               .assign(inv=lambda d: d["inv"].str.strip())
               .query("inv != ''")
               .iterrows()):
    inv = row["inv"]
    co  = row["co"]
    yr  = row["Year"]
    stage = str(row.get(stage_col, ""))
    sector = co_sector.get(co, "Unknown")
    inv_deals[inv].append((co, yr, stage, sector))

# Only keep investors in the full-network partition
inv_in_net = set(partition_full.keys())

def portfolio_set(inv):
    return set(d[0] for d in inv_deals.get(inv, []))

def stage_vector(inv, all_stages):
    stages = [d[2] for d in inv_deals.get(inv, [])]
    v = np.array([stages.count(s) for s in all_stages], dtype=float)
    return v / v.sum() if v.sum() > 0 else v

def sector_vector(inv, all_sectors):
    sectors = [d[3] for d in inv_deals.get(inv, [])]
    v = np.array([sectors.count(s) for s in all_sectors], dtype=float)
    return v / v.sum() if v.sum() > 0 else v

def timing_vector(inv, all_quarters):
    quarters = [f"{d[1]}Q{((pd.Timestamp(f'{int(d[1])}-01-01').month-1)//3)+1}" for d in inv_deals.get(inv, []) if pd.notna(d[1])]
    v = np.array([quarters.count(q) for q in all_quarters], dtype=float)
    return v / v.sum() if v.sum() > 0 else v

def cosine(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0

def jaccard(s1, s2):
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

# Gather useful investors (≥3 deals, in partition)
useful_inv = [inv for inv in inv_in_net
              if len(inv_deals.get(inv, [])) >= 3]
print(f"  Investors with ≥3 deals and in network: {len(useful_inv):,}")

# Precompute vectors
all_stages  = list(set(d[2] for inv in useful_inv for d in inv_deals.get(inv,[])))
all_sectors = list(set(d[3] for inv in useful_inv for d in inv_deals.get(inv,[])))
# Quarters: e.g. "2015Q1"
all_quarters = sorted(set(
    f"{d[1]}Q{1+((d[1]-1)%4)}"  # approximate
    for inv in useful_inv for d in inv_deals.get(inv,[])
    if pd.notna(d[1])
))

print(f"  Computing investor feature vectors…")
p_sets    = {inv: portfolio_set(inv)          for inv in useful_inv}
s_vecs    = {inv: stage_vector(inv, all_stages)   for inv in useful_inv}
sec_vecs  = {inv: sector_vector(inv, all_sectors) for inv in useful_inv}

# Sample pairs: 2000 within-community, 2000 across-community
comm_members = defaultdict(list)
for inv in useful_inv:
    comm_members[partition_full[inv]].append(inv)

within_pairs, across_pairs = [], []
rng = np.random.default_rng(SEED)

# Within
for comm, members in comm_members.items():
    if len(members) >= 2:
        n_sample = min(50, len(members)*(len(members)-1)//2)
        for _ in range(n_sample):
            a, b = rng.choice(members, 2, replace=False)
            within_pairs.append((a, b))
within_pairs = within_pairs[:2000]

# Across
comm_list = [c for c in comm_members if len(comm_members[c]) >= 1]
for _ in range(2000):
    c1, c2 = rng.choice(comm_list, 2, replace=False)
    a = rng.choice(comm_members[c1])
    b = rng.choice(comm_members[c2])
    across_pairs.append((a, b))

print(f"  Sampled {len(within_pairs):,} within-community pairs, {len(across_pairs):,} across-community pairs")

def score_pairs(pairs, label):
    port_sim, stage_sim, sector_sim = [], [], []
    for a, b in pairs:
        port_sim.append(jaccard(p_sets[a], p_sets[b]))
        stage_sim.append(cosine(s_vecs[a], s_vecs[b]))
        sector_sim.append(cosine(sec_vecs[a], sec_vecs[b]))
    return {"label": label,
            "portfolio_overlap": np.mean(port_sim),
            "stage_similarity":  np.mean(stage_sim),
            "sector_similarity": np.mean(sector_sim),
            "n": len(pairs)}

within_scores = score_pairs(within_pairs, "within")
across_scores = score_pairs(across_pairs, "across")

print(f"\n  {'Dimension':<28} {'Within':>10} {'Across':>10} {'Ratio':>8} {'Better?':>8}")
print(f"  {'-'*62}")
tests_pass = 0
for dim in ["portfolio_overlap","stage_similarity","sector_similarity"]:
    w = within_scores[dim]; a = across_scores[dim]
    ratio = w / a if a > 0 else float('inf')
    better = "YES ✓" if w > a else "NO ✗"
    if w > a:
        tests_pass += 1
    print(f"  {dim:<28} {w:>10.4f} {a:>10.4f} {ratio:>8.2f}x {better:>8}")

# Statistical test on portfolio overlap (most discriminating)
within_port = [jaccard(p_sets[a], p_sets[b]) for a, b in within_pairs]
across_port  = [jaccard(p_sets[a], p_sets[b]) for a, b in across_pairs]
t_stat, p_val5 = stats.mannwhitneyu(within_port, across_port, alternative="greater")
print(f"\n  Portfolio overlap Mann-Whitney U: p={p_val5:.2e} {sig(p_val5)}")

if tests_pass >= 3:
    verdict5 = f"PASS — within-community investors more similar on all {tests_pass}/3 dimensions"
elif tests_pass >= 2:
    verdict5 = f"BORDERLINE — within-community more similar on {tests_pass}/3 dimensions"
else:
    verdict5 = "FAIL — no consistent behavioral similarity within communities"
print(f"  Verdict: {verdict5}")
print(f"  [{time.time()-t0:.1f}s]")

print()
print("="*72)
print("SUMMARY: ARE LOUVAIN COMMUNITIES REAL INVESTOR ECOSYSTEMS?")
print("="*72)
verdicts = [verdict1, verdict2, verdict3, verdict4, verdict5]
pass_count = sum("PASS" in v for v in verdicts)
tests = ["TEST 1 — Temporal stability   ",
         "TEST 2 — Algorithm robustness ",
         "TEST 3 — Modularity vs random ",
         "TEST 4 — Geographic validation",
         "TEST 5 — Behavioral similarity"]
for t, v in zip(tests, verdicts):
    icon = "✓" if "PASS" in v else ("~" if "BORDERLINE" in v else "✗")
    print(f"  [{icon}] {t}: {v}")

print(f"\n  OVERALL: {pass_count}/5 tests passed")
if pass_count >= 4:
    print("  → STRONG EVIDENCE: Communities are real investor ecosystems,")
    print("    not algorithmic artifacts.")
elif pass_count >= 3:
    print("  → MODERATE EVIDENCE: Communities reflect real structure,")
    print("    with some method-dependence in boundaries.")
else:
    print("  → WEAK EVIDENCE: Community structure may be partly artifactual.")

print()
print("KEY STATISTICS FOR PAPER:")
print(f"  NMI temporal stability     : {nmi_temporal:.3f}")
print(f"  Mean algorithm NMI         : {avg_nmi:.3f}")
print(f"  Modularity z-score         : {z_score:.1f}  (p={p_val:.2e})")
print(f"  % geog. homogeneous (≥70%) : {pct_gt70:.1f}%")
print(f"  Cramér's V (comm × region) : {cramers_v:.3f}")
print(f"  Portfolio overlap ratio    : {within_scores['portfolio_overlap']/across_scores['portfolio_overlap']:.2f}x (within/across)")
print(f"  Portfolio overlap p-value  : {p_val5:.2e}")
