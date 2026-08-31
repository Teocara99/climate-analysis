"""
Script 40: Cross-Community Investment Paths
===========================================

6 analyses:
  1. Community × community deal matrix + outcome heatmap
  2. Geographic paths → outcome table + regression
  3. Concrete examples (top 5 paths, 3 companies each)
  4. What flows through the bridge? (capital, timing, follow-on)
  5. Path success regression (distance, overlap, complementarity)
  6. Narrative community profiles (top 10 communities)
"""

import warnings; warnings.filterwarnings("ignore")
import sys, itertools, math
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

sys.path.insert(0, ".")
from load_data import load_deals, load_companies

# ── Helpers ──────────────────────────────────────────────────────────────────
def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

REGION_COORDS = {          # approximate centroids (lat, lon)
    "Americas":    (40.0, -95.0),
    "Europe":      (50.0,  15.0),
    "Asia":        (30.0, 110.0),
    "Oceania":    (-25.0, 135.0),
    "Middle East": (25.0,  45.0),
    "Africa":       (0.0,  25.0),
}

def haversine_km(r1, r2):
    lat1, lon1 = map(math.radians, REGION_COORDS.get(r1, (0,0)))
    lat2, lon2 = map(math.radians, REGION_COORDS.get(r2, (0,0)))
    dlat = lat2-lat1; dlon = lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

def geo_path(r1, r2):
    if r1 == r2:
        return f"Within-{r1}"
    pair = tuple(sorted([r1, r2]))
    labels = {
        ("Americas","Europe"):     "Transatlantic (Americas↔Europe)",
        ("Americas","Asia"):       "Transpacific (Americas↔Asia)",
        ("Asia","Europe"):         "Euro-Asia",
        ("Africa","Americas"):     "Americas↔Africa",
        ("Africa","Europe"):       "Europe↔Africa",
        ("Africa","Asia"):         "Asia↔Africa",
        ("Americas","Oceania"):    "Americas↔Oceania",
        ("Europe","Oceania"):      "Europe↔Oceania",
        ("Asia","Oceania"):        "Asia↔Oceania",
        ("Africa","Oceania"):      "Africa↔Oceania",
        ("Americas","Middle East"):"Americas↔Middle East",
        ("Europe","Middle East"):  "Europe↔Middle East",
        ("Asia","Middle East"):    "Asia↔Middle East",
        ("Africa","Middle East"):  "Africa↔Middle East",
        ("Middle East","Oceania"): "Middle East↔Oceania",
    }
    return labels.get(pair, f"{r1}↔{r2}")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data…")
cdf  = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["is_exited"]  = cdf["exited"].astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

net = pd.read_csv("output/network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm = net.set_index("investor")["community"].to_dict()
inv_to_type  = net.set_index("investor")["investor_type"].to_dict()
inv_to_green = net.set_index("investor")["green_focus"].to_dict()

companies = load_companies()
companies["co"] = companies["Companies"].str.strip()
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")

# ── Map companies → investor communities ─────────────────────────────────────
print("Mapping companies → investor communities (via deals file)…")
# Use deals "Investors" column — same company naming as cdf
co_investors = defaultdict(list)
co_communities = defaultdict(set)
for _, row in deals.dropna(subset=["Investors"]).iterrows():
    co   = row["co"]
    invs = [i.strip() for i in str(row["Investors"]).split(";") if i.strip()]
    for inv in invs:
        comm = inv_to_comm.get(inv)
        co_investors[co].append(inv)
        if comm is not None:
            co_communities[co].add(comm)

# Deduplicate investor lists
co_investors = {co: list(set(invs)) for co, invs in co_investors.items()}

# Map community → dominant HQ region (via portfolio companies in cdf)
comm_regions = defaultdict(list)
co_region = cdf.set_index("co")["hq_region"].to_dict()
for co, comms in co_communities.items():
    reg = co_region.get(co)
    if reg:
        for comm in comms:
            comm_regions[comm].append(reg)

comm_dom_region = {}
for comm, regs in comm_regions.items():
    comm_dom_region[comm] = max(set(regs), key=regs.count)

# Sector map (built early so Analysis 3 can use it)
sector_col = next((c for c in deals.columns if "primary industry" in c.lower() and "sector" in c.lower()), None)
if sector_col is None:
    sector_col = next((c for c in deals.columns if "industry" in c.lower()), None)
co_sector_map = {}
if sector_col:
    co_sector_map = (deals.dropna(subset=[sector_col])
                     .sort_values("Year", ascending=False)
                     .drop_duplicates("co")
                     .set_index("co")[sector_col]
                     .to_dict())

print(f"  {len(comm_dom_region)} communities with region assignment")
region_counts = pd.Series(comm_dom_region).value_counts()
print(f"  Region distribution:\n{region_counts.to_string()}")

# ── Community × Community matrix ─────────────────────────────────────────────
print("\nBuilding community pair matrix…")
pair_data = defaultdict(list)   # (ci,cj) → list of co names

for _, row in cdf.iterrows():
    co   = row["co"]
    comms = list(co_communities.get(co, set()))
    if len(comms) < 2:
        continue
    for ci, cj in itertools.combinations(sorted(comms), 2):
        pair_data[(ci, cj)].append(co)

# Build pair DataFrame
records = []
for (ci, cj), cos_list in pair_data.items():
    sub = cdf[cdf["co"].isin(cos_list)]
    n       = len(sub)
    fail_r  = sub["is_failed"].mean()
    exit_r  = sub["is_exited"].mean()
    ri = comm_dom_region.get(ci, "Unknown")
    rj = comm_dom_region.get(cj, "Unknown")
    records.append({
        "comm_i": ci, "comm_j": cj,
        "n": n,
        "failure_rate": fail_r,
        "exit_rate":    exit_r,
        "region_i": ri, "region_j": rj,
        "path": geo_path(ri, rj),
        "same_region": ri == rj,
    })

pairs_df = pd.DataFrame(records).sort_values("n", ascending=False)
print(f"  Total community pairs with ≥1 deal: {len(pairs_df):,}")

print()
print("="*72)
print("ANALYSIS 1: TOP 20 COMMUNITY PAIRS BY DEAL VOLUME")
print("="*72)
# within-community baseline
wc_sub = cdf[cdf["cross_community"]==0]
wc_fail = wc_sub["is_failed"].mean()
wc_exit = wc_sub["is_exited"].mean()
print(f"\n  BASELINE — within-community: fail={wc_fail:.1%}  exit={wc_exit:.1%}  N={len(wc_sub):,}")
print()

top20 = pairs_df[pairs_df["n"] >= 5].head(20)
print(f"  {'Pair':>12}  {'N':>5}  {'Fail%':>7}  {'Exit%':>7}  {'vs Base':>9}  Path")
print(f"  {'-'*72}")
for _, r in top20.iterrows():
    delta = (r["failure_rate"] - wc_fail) * 100
    delta_str = f"{delta:>+.1f}pp"
    print(f"  {r['comm_i']:>3}↔{r['comm_j']:<6}   {r['n']:>5}  {r['failure_rate']:>7.1%}  "
          f"{r['exit_rate']:>7.1%}  {delta_str:>9}  {r['path']}")

print()
print("="*72)
print("ANALYSIS 2: GEOGRAPHIC PATHS → OUTCOMES")
print("="*72)

# Assign each cross-community COMPANY a path (using the most common pair)
co_path = {}
for _, row in cdf[cdf["cross_community"]==1].iterrows():
    co    = row["co"]
    comms = list(co_communities.get(co, set()))
    if len(comms) < 2:
        continue
    regs = [comm_dom_region.get(c) for c in comms if comm_dom_region.get(c)]
    regs = [r for r in regs if r]
    if len(regs) < 2:
        continue
    # Get most distant pair of regions for this company
    best_path = None
    best_dist = -1
    for ri, rj in itertools.combinations(regs, 2):
        d = haversine_km(ri, rj)
        if d > best_dist:
            best_dist = d
            best_path = geo_path(ri, rj)
    if best_path:
        co_path[co] = best_path

cdf["geo_path"] = cdf["co"].map(co_path)
cdf.loc[cdf["cross_community"]==0, "geo_path"] = "Within-community (same)"

path_stats = (cdf.dropna(subset=["geo_path"])
              .groupby("geo_path")
              .agg(
                  n         =("is_failed","count"),
                  fail_rate =("is_failed","mean"),
                  exit_rate =("is_exited","mean"),
                  avg_cap   =("log_capital","mean"),
              )
              .sort_values("n", ascending=False))

# Filter to paths with ≥10 companies
path_stats = path_stats[path_stats["n"] >= 10]
print(f"\n  {'Path':<40}  {'N':>5}  {'Fail%':>7}  {'Exit%':>7}  {'vs Base':>9}")
print(f"  {'-'*75}")
for path, r in path_stats.iterrows():
    delta = (r["fail_rate"] - wc_fail) * 100
    print(f"  {path:<40}  {r['n']:>5}  {r['fail_rate']:>7.1%}  "
          f"{r['exit_rate']:>7.1%}  {delta:>+.1f}pp")

# Regression with path dummies
print("\n  Regression: failure ~ path_type + controls")
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"
df_reg = cdf.dropna(subset=["geo_path","log_cap","deal_year"]).copy()
# Collapse small paths into "Other cross-community"
top_paths = path_stats[path_stats["n"] >= 20].index.tolist()
df_reg["path_reg"] = df_reg["geo_path"].where(
    df_reg["geo_path"].isin(top_paths + ["Within-community (same)"]),
    "Other cross-community"
)
df_reg["path_reg"] = df_reg["path_reg"].str.replace(r"[↔\(\)/\s\-]", "_", regex=True)
ref_cat = "Within_community__same_"   # reference will be set by patsy ordering
try:
    m_path = smf.ols(f"is_failed ~ C(path_reg) + {CTRL}", data=df_reg).fit(cov_type="HC1")
    path_coefs = [(k, v, m_path.pvalues[k])
                  for k, v in m_path.params.items() if "path_reg" in k]
    path_coefs.sort(key=lambda x: x[1])
    print(f"\n  {'Path coefficient':<55}  {'β':>8}  {'p':>8}")
    print(f"  {'-'*75}")
    for name, b, p in path_coefs:
        label = name.replace("C(path_reg)[T.","").rstrip("]").replace("_"," ")
        print(f"  {label:<55}  {b:>+8.4f}  {sig(p):>8}")
except Exception as e:
    print(f"  Regression failed: {e}")

print()
print("="*72)
print("ANALYSIS 3: CONCRETE EXAMPLES — TOP 5 PROTECTIVE PATHS")
print("="*72)

# Rank paths by failure rate (lower = more protective), min 10 companies, exclude within
ranked_paths = (path_stats
                [~path_stats.index.str.startswith("Within")]
                .sort_values("fail_rate")
                .head(5))

for rank, (path, prow) in enumerate(ranked_paths.iterrows(), 1):
    print(f"\n  #{rank}: {path}  (N={int(prow['n'])}, fail={prow['fail_rate']:.1%}, exit={prow['exit_rate']:.1%})")
    # Get companies on this path, sample 3 (prefer exited > active > failed)
    path_cos = cdf[cdf["geo_path"]==path].copy()
    path_cos = pd.concat([
        path_cos[path_cos["is_exited"]==1].head(2),
        path_cos[path_cos["is_failed"]==0].head(2),
        path_cos[path_cos["is_failed"]==1].head(1),
    ]).drop_duplicates("co").head(3)

    for _, row in path_cos.iterrows():
        co    = row["co"]
        invs  = co_investors.get(co, [])
        # Get community info per investor
        inv_comm_info = [(inv, inv_to_comm.get(inv), comm_dom_region.get(inv_to_comm.get(inv),"?"),
                          inv_to_type.get(inv,"?"))
                         for inv in invs if inv_to_comm.get(inv) is not None]
        # Group by community
        by_comm = defaultdict(list)
        for inv, comm, reg, typ in inv_comm_info:
            by_comm[(comm, reg)].append(f"{inv} ({typ})")

        cap = row.get("Total Raised", "?")
        cap_str = f"${cap:.0f}M" if pd.notna(cap) and cap != "?" else "N/A"
        outcome_str = row["outcome"]
        sector = co_sector_map.get(co, "?")
        print(f"\n    [{outcome_str}] {co}  |  Sector: {sector}  |  Capital: {cap_str}")
        print(f"    Investor communities:")
        for (comm, reg), inv_list in sorted(by_comm.items()):
            inv_str = "; ".join(inv_list[:3]) + ("…" if len(inv_list)>3 else "")
            print(f"      Community {comm} ({reg}): {inv_str}")

print()
print("="*72)
print("ANALYSIS 4: WHAT FLOWS THROUGH THE BRIDGE?")
print("="*72)

# For cross-community companies: define "home community" (most investors)
# and "bridge community" (second most)
print("\n  A) Capital: do bridge-community investors contribute more capital?")
print("     (Approximated by: does cross-community status predict MORE total capital?)")
CTRL2 = "nrounds + specter_novelty + C(country_fe) + C(deal_year)"
sub_cap = cdf.dropna(subset=["log_cap","deal_year"]).copy()
m_cap = smf.ols(f"log_cap ~ cross_community + {CTRL2}", data=sub_cap).fit(cov_type="HC1")
b_cap = m_cap.params["cross_community"]
print(f"     β_cross on log_capital = {b_cap:+.4f}  {sig(m_cap.pvalues['cross_community'])}")
print(f"     → Cross-community companies raise exp({b_cap:.2f}) ≈ {(np.exp(b_cap)-1)*100:+.1f}% more capital")

# B) Timing: do bridge investors come first or later?
print("\n  B) Timing: does the bridge community come in EARLY or LATER?")
# Get earliest deal year per company per investor community
co_comm_timing = defaultdict(lambda: defaultdict(list))
for _, row in deals.dropna(subset=["Investors","Year"]).iterrows():
    co  = row["co"]
    yr  = row["Year"]
    invs = [i.strip() for i in str(row["Investors"]).split(";") if i.strip()]
    for inv in invs:
        comm = inv_to_comm.get(inv)
        if comm is not None:
            co_comm_timing[co][comm].append(yr)

timing_records = []
for co, comm_years in co_comm_timing.items():
    comms = sorted(comm_years.keys())
    if len(comms) < 2:
        continue
    first_yr = {c: min(yrs) for c, yrs in comm_years.items()}
    sorted_comms = sorted(first_yr.items(), key=lambda x: x[1])
    home_comm, home_yr = sorted_comms[0]
    for bridge_comm, bridge_yr in sorted_comms[1:]:
        lag = bridge_yr - home_yr
        timing_records.append({
            "co": co,
            "home_comm": home_comm,
            "bridge_comm": bridge_comm,
            "home_yr": home_yr,
            "bridge_yr": bridge_yr,
            "lag_years": lag,
        })

timing_df = pd.DataFrame(timing_records)
if len(timing_df):
    print(f"     Cross-community timing analysis (N={len(timing_df):,} community-pair entries):")
    print(f"     Bridge community arrives: mean {timing_df['lag_years'].mean():+.1f} years after home community")
    print(f"     Median lag: {timing_df['lag_years'].median():+.1f} years")
    print(f"     % where bridge arrives FIRST (same year or before): "
          f"{(timing_df['lag_years'] <= 0).mean():.1%}")
    print(f"     % where bridge arrives LATER (>1 year after home): "
          f"{(timing_df['lag_years'] > 1).mean():.1%}")
    lag_by_outcome = timing_df.merge(cdf[["co","is_failed","is_exited"]], on="co", how="left")
    grp = lag_by_outcome.groupby("is_failed")["lag_years"].mean()
    print(f"\n     Mean lag — survived: {grp.get(0.0, np.nan):+.1f} yr  |  failed: {grp.get(1.0, np.nan):+.1f} yr")

# C) Follow-on: proxy via n_rounds
print("\n  C) Follow-on: do cross-community companies get more rounds?")
m_rounds = smf.ols(f"nrounds ~ cross_community + {CTRL2}", data=sub_cap).fit(cov_type="HC1")
b_r = m_rounds.params["cross_community"]
print(f"     β_cross on log(n_rounds+1) = {b_r:+.4f}  {sig(m_rounds.pvalues['cross_community'])}")
print(f"     → {(np.exp(b_r)-1)*100:+.1f}% more rounds for cross-community companies")

# D) Exit network
print("\n  D) Exit: do cross-community companies exit MORE?")
m_exit = smf.ols(f"is_exited ~ cross_community + {CTRL}", data=cdf.dropna(subset=["log_cap","deal_year"])).fit(cov_type="HC1")
b_e = m_exit.params["cross_community"]
print(f"     β_cross on exit = {b_e:+.4f}  {sig(m_exit.pvalues['cross_community'])}")
print(f"     → {b_e*100:+.1f}pp exit rate change for cross-community companies")

print()
print("="*72)
print("ANALYSIS 5: WHAT MAKES A PATH SUCCESSFUL?")
print("="*72)

# Build pair-level characteristics
print("\n  Building pair-level features…")
# Community-level features
comm_size = net.groupby("community").size().to_dict()
comm_green_pct = net.groupby("community")["green_focus"].apply(
    lambda x: (x=="GREEN_VC").mean()).to_dict()
comm_types = net.groupby("community")["investor_type"].apply(
    lambda x: x.value_counts(normalize=True).to_dict()).to_dict()

# Sector overlap between communities
comm_sectors = defaultdict(list)
for co, comms in co_communities.items():
    sector = co_sector_map.get(co, "Unknown")
    if sector and sector not in ("Unknown", "nan"):
        for comm in comms:
            comm_sectors[comm].append(sector)

def jaccard_sectors(c1, c2):
    s1 = set(comm_sectors.get(c1, []))
    s2 = set(comm_sectors.get(c2, []))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

pair_features = []
for _, r in pairs_df[pairs_df["n"] >= 5].iterrows():
    ci, cj = r["comm_i"], r["comm_j"]
    ri, rj = comm_dom_region.get(ci, "?"), comm_dom_region.get(cj, "?")
    geo_dist = haversine_km(ri, rj) if ri != "?" and rj != "?" else np.nan
    size_i, size_j = comm_size.get(ci, 1), comm_size.get(cj, 1)
    size_ratio = max(size_i, size_j) / max(min(size_i, size_j), 1)
    green_gap  = abs(comm_green_pct.get(ci, 0) - comm_green_pct.get(cj, 0))
    sec_overlap = jaccard_sectors(ci, cj)
    # Type complementarity: are there different dominant types?
    type_i = max(comm_types.get(ci, {"?":1}), key=comm_types[ci].get) if ci in comm_types else "?"
    type_j = max(comm_types.get(cj, {"?":1}), key=comm_types[cj].get) if cj in comm_types else "?"
    type_compl = float(type_i != type_j)
    pair_features.append({
        "comm_i": ci, "comm_j": cj,
        "failure_rate": r["failure_rate"],
        "exit_rate":    r["exit_rate"],
        "n":            r["n"],
        "geo_dist_km":  geo_dist,
        "size_ratio":   np.log1p(size_ratio),
        "green_gap":    green_gap,
        "sector_overlap": sec_overlap,
        "type_compl":   type_compl,
        "same_region":  float(ri == rj),
    })

feat_df = pd.DataFrame(pair_features).dropna(subset=["geo_dist_km"])
print(f"  Pair-level dataset: {len(feat_df):,} pairs with full features")

if len(feat_df) >= 20:
    # WLS: weight by n (larger pairs are more reliable)
    m_pair = smf.wls(
        "failure_rate ~ geo_dist_km + size_ratio + green_gap + sector_overlap + type_compl",
        data=feat_df, weights=feat_df["n"]
    ).fit()
    print(f"\n  WLS: pair_failure_rate ~ path characteristics  (N={len(feat_df)} pairs, weighted by deal count)")
    print(f"\n  {'Feature':<25}  {'β':>10}  {'p':>8}")
    print(f"  {'-'*48}")
    for k in ["geo_dist_km","size_ratio","green_gap","sector_overlap","type_compl"]:
        b = m_pair.params.get(k, np.nan)
        p = m_pair.pvalues.get(k, np.nan)
        print(f"  {k:<25}  {b:>+10.6f}  {sig(p):>8}")
    print(f"\n  R² = {m_pair.rsquared:.3f}")
    print("\n  Interpretation:")
    if m_pair.params.get("geo_dist_km",0) < 0:
        print("  → Geographic distance REDUCES failure (more distant = better outcome)")
    else:
        print("  → Geographic distance INCREASES failure (more distant = harder)")
    if m_pair.params.get("green_gap",0) < 0:
        print("  → Green gap REDUCES failure (complementary green/traditional mix = better)")
    if m_pair.params.get("sector_overlap",0) < 0:
        print("  → Sector overlap REDUCES failure (common sector knowledge = better)")
    if m_pair.params.get("type_compl",0) < 0:
        print("  → Type complementarity REDUCES failure (mixed VC types = better)")
else:
    print("  Insufficient pairs for regression")

print()
print("="*72)
print("ANALYSIS 6: NARRATIVE COMMUNITY PROFILES (TOP 10 BY SIZE)")
print("="*72)

top10_comms = net.groupby("community").size().nlargest(10).index.tolist()

# Per-community stats
for comm in top10_comms:
    members = net[net["community"]==comm]
    n_inv   = len(members)
    dom_reg = comm_dom_region.get(comm, "Unknown")
    green_pct = (members["green_focus"]=="GREEN_VC").mean() * 100
    types_dist = members["investor_type"].value_counts(normalize=True)
    dom_type   = types_dist.index[0] if len(types_dist) else "?"
    top5_inv   = members.nlargest(5,"degree")["investor"].tolist()
    top5_str   = ", ".join(top5_inv[:3])

    # Portfolio outcomes
    comm_cos = [co for co, comms in co_communities.items() if comm in comms and co in cdf["co"].values]
    if comm_cos:
        sub = cdf[cdf["co"].isin(comm_cos)]
        fail_r = sub["is_failed"].mean()
        exit_r = sub["is_exited"].mean()
        n_cos  = len(sub)

        # Dominant sectors (from deals-derived map)
        sectors = [co_sector_map.get(co, "Unknown") for co in comm_cos]
        sec_counts = pd.Series(sectors).value_counts()
        top_sec = sec_counts.index[:3].tolist()

        # Most common cross-community partners
        partner_counts = defaultdict(int)
        for _, r in pairs_df[(pairs_df["comm_i"]==comm) | (pairs_df["comm_j"]==comm)].iterrows():
            partner = r["comm_j"] if r["comm_i"]==comm else r["comm_i"]
            partner_counts[partner] += r["n"]
        top_partners = sorted(partner_counts.items(), key=lambda x:-x[1])[:3]
        partner_str = ", ".join([f"Comm {p} ({n})" for p,n in top_partners])

        print(f"\n  ─── Community {comm}: {dom_reg} Hub ─────────────────────────────")
        print(f"  {n_inv} investors  |  {green_pct:.0f}% GREEN_VC  |  Dominant type: {dom_type}")
        print(f"  Portfolio: {n_cos} companies  |  Fail: {fail_r:.1%}  |  Exit: {exit_r:.1%}")
        print(f"  Top sectors:   {', '.join(top_sec)}")
        print(f"  Key investors: {top5_str}")
        print(f"  Bridge partners: {partner_str}")
    else:
        print(f"\n  ─── Community {comm}: {dom_reg} ───")
        print(f"  {n_inv} investors  |  {green_pct:.0f}% GREEN_VC — no linked portfolio companies")

print()
print("="*72)
print("SUMMARY: CROSS-COMMUNITY PATHS")
print("="*72)
best_path = ranked_paths.iloc[0]
print(f"\n  Most protective path:  {best_path.name}  (fail={best_path['fail_rate']:.1%} vs baseline {wc_fail:.1%})")
print(f"  Timing:  bridge community arrives {timing_df['lag_years'].mean():+.1f} yr after home community on average")
print(f"  Capital: cross-community → {(np.exp(b_cap)-1)*100:+.1f}% more capital raised")
print(f"  Rounds:  cross-community → {(np.exp(b_r)-1)*100:+.1f}% more follow-on rounds")
print(f"  Exit:    cross-community → {b_e*100:+.1f}pp exit rate")
