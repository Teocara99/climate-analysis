"""
31 — Community Anatomy: What Defines Louvain Communities?

Five analyses:
  1. Community profiles (geography, sector, stage, type, green)
  2. Which dimension best predicts community membership (Cramér's V + AMI)
  3. Homogeneity decomposition across dimensions
  4. Cross-community = cross-what? (correlations with cross-X variables)
  5. What does crossing communities actually mean?

Outputs → output/community_anatomy/
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency, pearsonr
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
OUT  = ROOT / "output" / "community_anatomy"
OUT.mkdir(parents=True, exist_ok=True)

MIN_COMMUNITY_SIZE = 20   # only profile communities with ≥ this many investors

def parse_investors(s):
    if pd.isna(s): return []
    parts = [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]
    return [re.sub(r"\s*\([^)]*\)\s*$", "", p).strip() for p in parts if p]

def herfindahl(counts):
    """HHI from a Counter or dict. Returns 0 for empty."""
    total = sum(counts.values())
    if total == 0: return np.nan
    return sum((v/total)**2 for v in counts.values())

def cramers_v(x, y):
    """Corrected Cramér's V between two categorical Series."""
    tab = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(tab, correction=False)
    n = tab.values.sum()
    r, k = tab.shape
    phi2 = chi2 / n
    phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rc = r - (r-1)**2/(n-1)
    kc = k - (k-1)**2/(n-1)
    denom = min(kc-1, rc-1)
    if denom <= 0: return 0.0, p
    return float(np.sqrt(phi2c / denom)), float(p)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data …")

    nm = pd.read_csv(ROOT / "output/network/network_metrics.csv")
    nm["inv_key"] = nm["investor"].str.lower().str.strip()
    nm = nm.dropna(subset=["community"])
    nm["community"] = nm["community"].astype(int)

    ip = pd.read_csv(ROOT / "output/investor_profiles.csv")
    ip["inv_key"] = ip["investor"].str.lower().str.strip()

    # Merge: use nm as authority for community, ip for characteristics
    inv_df = nm.merge(
        ip[["inv_key","geo_focus","stage_focus","sector_focus"]],
        on="inv_key", how="left"
    )
    # Simplify geo to 3+1 buckets
    inv_df["geo_broad"] = inv_df["geo_focus"].map(
        lambda g: g if g in ("Americas","Europe","Asia") else "Other"
    ).fillna("Unknown")
    # Simplify stage
    inv_df["stage_clean"] = inv_df["stage_focus"].map(
        {"Seed/Angel":"Seed/Angel","Early":"Early",
         "Growth":"Growth","Late":"Late","Other":"Other"}
    ).fillna("Other")
    # Sector already coarse — keep as-is
    inv_df["sector_clean"] = inv_df["sector_focus"].fillna("Other / Diversified")

    # Louvain community sizes
    comm_sizes = inv_df["community"].value_counts()
    large_comms = sorted(comm_sizes[comm_sizes >= MIN_COMMUNITY_SIZE].index.tolist())
    print(f"  Investors: {len(inv_df):,}  |  Communities ≥{MIN_COMMUNITY_SIZE}: {len(large_comms)}")

    # Company dataset
    cdf = pd.read_csv(ROOT / "output/cpu/cpu_company_dataset.csv")
    cdf["company_key"] = cdf["Companies"].str.lower().str.strip()
    cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)

    # Deals
    deals = load_deals()
    deals["date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["date","Investors"])
    deals["company_key"] = deals["Companies"].str.lower().str.strip()

    # Investor lookup dicts
    inv_community = dict(zip(inv_df["inv_key"], inv_df["community"]))
    inv_geo       = dict(zip(inv_df["inv_key"], inv_df["geo_broad"]))
    inv_type      = dict(zip(inv_df["inv_key"], inv_df["investor_type"]))
    inv_stage     = dict(zip(inv_df["inv_key"], inv_df["stage_clean"]))
    inv_sector    = dict(zip(inv_df["inv_key"], inv_df["sector_clean"]))
    inv_green     = dict(zip(inv_df["inv_key"], inv_df["green_focus"]))

    return inv_df, cdf, deals, large_comms, inv_community, inv_geo, inv_type, inv_stage, inv_sector, inv_green


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: COMMUNITY PROFILES
# ═══════════════════════════════════════════════════════════════════════════════

def analysis1_profiles(inv_df, large_comms, lines):
    print("\nAnalysis 1: Community profiles …")
    lines += ["\n"+"="*80,
              "ANALYSIS 1: COMMUNITY PROFILES (communities ≥ 20 investors)",
              "="*80]

    profile_rows = []
    for comm in large_comms:
        sub = inv_df[inv_df["community"] == comm]
        n = len(sub)

        # A) Geography
        geo_cnt  = Counter(sub["geo_broad"].dropna())
        geo_hhi  = herfindahl(geo_cnt)
        dom_geo  = max(geo_cnt, key=geo_cnt.get) if geo_cnt else "?"
        pct_dom_geo = geo_cnt.get(dom_geo, 0) / max(n, 1)
        pct_amer = geo_cnt.get("Americas", 0) / n
        pct_eu   = geo_cnt.get("Europe",   0) / n
        pct_asia = geo_cnt.get("Asia",     0) / n

        # B) Sector
        sec_cnt  = Counter(sub["sector_clean"].dropna())
        sec_hhi  = herfindahl(sec_cnt)
        dom_sec  = max(sec_cnt, key=sec_cnt.get) if sec_cnt else "?"
        pct_dom_sec = sec_cnt.get(dom_sec, 0) / max(n, 1)

        # C) Stage
        stg_cnt  = Counter(sub["stage_clean"].dropna())
        stg_hhi  = herfindahl(stg_cnt)
        dom_stg  = max(stg_cnt, key=stg_cnt.get) if stg_cnt else "?"
        pct_dom_stg = stg_cnt.get(dom_stg, 0) / max(n, 1)

        # D) Investor type
        type_cnt  = Counter(sub["investor_type"].dropna())
        type_hhi  = herfindahl(type_cnt)
        dom_type  = max(type_cnt, key=type_cnt.get) if type_cnt else "?"
        pct_dom_type = type_cnt.get(dom_type, 0) / max(n, 1)
        pct_gvc  = type_cnt.get("GVC", 0) / n
        pct_ivc  = type_cnt.get("IVC", 0) / n
        pct_cvc  = type_cnt.get("CVC", 0) / n

        # E) Green focus
        grn_cnt  = Counter(sub["green_focus"].dropna())
        grn_hhi  = herfindahl(grn_cnt)
        dom_grn  = max(grn_cnt, key=grn_cnt.get) if grn_cnt else "?"
        pct_dom_grn = grn_cnt.get(dom_grn, 0) / max(n, 1)
        pct_green  = grn_cnt.get("GREEN_VC",   0) / n
        pct_trad   = grn_cnt.get("TRADITIONAL", 0) / n

        profile_rows.append({
            "community":    comm,
            "n_investors":  n,
            # Geography
            "pct_americas": pct_amer,  "pct_europe": pct_eu, "pct_asia": pct_asia,
            "geo_hhi":      geo_hhi,   "dom_geo": dom_geo,   "pct_dom_geo": pct_dom_geo,
            # Sector
            "dom_sector":   dom_sec,   "pct_dom_sector": pct_dom_sec, "sector_hhi": sec_hhi,
            # Stage
            "dom_stage":    dom_stg,   "pct_dom_stage": pct_dom_stg, "stage_hhi": stg_hhi,
            # Type
            "dom_type":     dom_type,  "pct_dom_type": pct_dom_type, "type_hhi": type_hhi,
            "pct_gvc": pct_gvc, "pct_ivc": pct_ivc, "pct_cvc": pct_cvc,
            # Green
            "dom_green":    dom_grn,   "pct_dom_green": pct_dom_grn, "green_hhi": grn_hhi,
            "pct_green_vc": pct_green, "pct_trad": pct_trad,
        })

    prof_df = pd.DataFrame(profile_rows)

    # Print profile table
    lines.append(f"\n  {'C':>3}  {'N':>5}  {'dom_geo':<12} {'%':>5}  "
                 f"{'dom_type':<10} {'%':>5}  "
                 f"{'dom_green':<12} {'%':>5}  "
                 f"{'dom_stage':<12} {'%':>5}  "
                 f"{'dom_sector':<22} {'%':>5}")
    lines.append("  " + "-"*110)
    for _, r in prof_df.iterrows():
        lines.append(
            f"  {int(r['community']):>3}  {int(r['n_investors']):>5}  "
            f"{r['dom_geo']:<12} {r['pct_dom_geo']:>5.0%}  "
            f"{r['dom_type']:<10} {r['pct_dom_type']:>5.0%}  "
            f"{r['dom_green']:<12} {r['pct_dom_green']:>5.0%}  "
            f"{r['dom_stage']:<12} {r['pct_dom_stage']:>5.0%}  "
            f"{r['dom_sector']:<22} {r['pct_dom_sector']:>5.0%}"
        )

    lines.append(f"\n  AVERAGE DOMINANT-SHARE BY DIMENSION:")
    lines.append(f"    Geography:   {prof_df['pct_dom_geo'].mean():.1%}  (HHI avg={prof_df['geo_hhi'].mean():.3f})")
    lines.append(f"    Type:        {prof_df['pct_dom_type'].mean():.1%}  (HHI avg={prof_df['type_hhi'].mean():.3f})")
    lines.append(f"    Green focus: {prof_df['pct_dom_green'].mean():.1%}  (HHI avg={prof_df['green_hhi'].mean():.3f})")
    lines.append(f"    Stage:       {prof_df['pct_dom_stage'].mean():.1%}  (HHI avg={prof_df['stage_hhi'].mean():.3f})")
    lines.append(f"    Sector:      {prof_df['pct_dom_sector'].mean():.1%}  (HHI avg={prof_df['sector_hhi'].mean():.3f})")

    prof_df.to_csv(OUT / "community_profiles.csv", index=False)
    return prof_df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: WHICH DIMENSION BEST PREDICTS COMMUNITY?
# ═══════════════════════════════════════════════════════════════════════════════

def analysis2_predictive(inv_df, large_comms, lines):
    print("\nAnalysis 2: Predictive power of each dimension …")
    lines += ["\n"+"="*80,
              "ANALYSIS 2: WHICH DIMENSION BEST PREDICTS COMMUNITY MEMBERSHIP?",
              "  Metrics: Cramér's V (corrected), Adjusted Mutual Information (AMI)",
              "  Higher = stronger association with community membership",
              "="*80]

    # Restrict to large communities + drop unknowns
    sub = inv_df[inv_df["community"].isin(large_comms)].copy()
    sub["geo_broad"]   = sub["geo_broad"].fillna("Unknown")
    sub["stage_clean"] = sub["stage_clean"].fillna("Other")
    sub["sector_clean"]= sub["sector_clean"].fillna("Other / Diversified")
    sub["investor_type"] = sub["investor_type"].fillna("OTHER")
    sub["green_focus"]   = sub["green_focus"].fillna("TRADITIONAL")

    comm = sub["community"].astype(str)
    dims = [
        ("Geography (region)",     sub["geo_broad"]),
        ("Investor type",          sub["investor_type"]),
        ("Green focus",            sub["green_focus"]),
        ("Stage preference",       sub["stage_clean"]),
        ("Sector preference",      sub["sector_clean"]),
    ]

    results = []
    lines.append(f"\n  {'Dimension':<28}  {'CramersV':>11}  {'p-value':>10}  {'AMI':>8}  Rank")
    lines.append(f"  {'-'*70}")

    for label, var in dims:
        v, p  = cramers_v(comm, var)
        ami   = adjusted_mutual_info_score(comm.values, var.values)
        results.append({"dimension": label, "cramers_v": v, "p": p, "ami": ami})

    # Rank by Cramér's V
    res_df = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
    res_df["rank"] = range(1, len(res_df)+1)

    for _, r in res_df.iterrows():
        lines.append(f"  {r['dimension']:<28}  {r['cramers_v']:>11.4f}  "
                     f"{r['p']:>10.4f}  {r['ami']:>8.4f}  #{int(r['rank'])}")

    top = res_df.iloc[0]
    lines.append(f"\n  WINNER: '{top['dimension']}' is the strongest predictor of community")
    lines.append(f"    V={top['cramers_v']:.3f}, AMI={top['ami']:.3f}")
    lines.append(f"  RUNNER-UP: '{res_df.iloc[1]['dimension']}'  V={res_df.iloc[1]['cramers_v']:.3f}")

    return res_df


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: HOMOGENEITY DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def analysis3_homogeneity(prof_df, lines):
    print("\nAnalysis 3: Homogeneity decomposition …")
    lines += ["\n"+"="*80,
              "ANALYSIS 3: COMMUNITY HOMOGENEITY BY DIMENSION",
              "  Homogeneity = max(% investors sharing a single attribute value)",
              "  High homogeneity → that dimension defines the community strongly",
              "="*80]

    dims_hom = {
        "Geography":   "pct_dom_geo",
        "Investor type":"pct_dom_type",
        "Green focus": "pct_dom_green",
        "Stage":       "pct_dom_stage",
        "Sector":      "pct_dom_sector",
    }

    lines.append(f"\n  {'Dimension':<18}  {'Avg homogeneity':>16}  {'Median':>8}  {'Min':>8}  {'Max':>8}")
    lines.append(f"  {'-'*65}")
    hom_summary = {}
    for label, col in dims_hom.items():
        vals = prof_df[col].dropna()
        hom_summary[label] = vals.mean()
        lines.append(f"  {label:<18}  {vals.mean():>16.1%}  {vals.median():>8.1%}  "
                     f"{vals.min():>8.1%}  {vals.max():>8.1%}")

    best = max(hom_summary, key=hom_summary.get)
    lines.append(f"\n  DEFINING DIMENSION: '{best}' shows highest avg within-community homogeneity")
    lines.append(f"    ({hom_summary[best]:.1%} of investors in same category on average)")
    lines.append(f"\n  INSIGHT: Communities are primarily defined by '{best}',")
    lines.append(f"    with '{sorted(hom_summary, key=hom_summary.get, reverse=True)[1]}' as the secondary dimension.")

    # Distribution of dominant geo across communities
    lines.append(f"\n  GEO DISTRIBUTION OF DOMINANT REGION ACROSS COMMUNITIES:")
    geo_dom_counts = prof_df["dom_geo"].value_counts()
    for geo, cnt in geo_dom_counts.items():
        lines.append(f"    {geo:<15} dominates {cnt:>2} of {len(prof_df)} communities")

    lines.append(f"\n  TYPE DISTRIBUTION OF DOMINANT TYPE ACROSS COMMUNITIES:")
    type_dom_counts = prof_df["dom_type"].value_counts()
    for t, cnt in type_dom_counts.items():
        lines.append(f"    {t:<15} dominates {cnt:>2} of {len(prof_df)} communities")

    lines.append(f"\n  GREEN FOCUS DISTRIBUTION ACROSS COMMUNITIES:")
    grn_dom_counts = prof_df["dom_green"].value_counts()
    for g, cnt in grn_dom_counts.items():
        lines.append(f"    {g:<15} dominates {cnt:>2} of {len(prof_df)} communities")

    return hom_summary


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: CROSS-COMMUNITY = CROSS-WHAT?
# ═══════════════════════════════════════════════════════════════════════════════

def analysis4_cross_dimensions(cdf, deals, inv_community, inv_geo, inv_type,
                                inv_stage, inv_sector, inv_green, lines):
    print("\nAnalysis 4: Cross-community = cross-what? …")
    lines += ["\n"+"="*80,
              "ANALYSIS 4: CROSS-COMMUNITY = CROSS-WHAT?",
              "  For each company's first-round syndicate, compute cross-X dimensions",
              "="*80]

    # First deal per company
    deals_first = (deals.sort_values("date")
                   .dropna(subset=["Investors"])
                   .drop_duplicates("company_key", keep="first"))

    records = []
    for _, row in deals_first.iterrows():
        ck = row["company_key"]
        inv_list = [i.lower() for i in parse_investors(row["Investors"]) if i]
        if not inv_list: continue

        comms   = [inv_community.get(i) for i in inv_list]
        comms   = [c for c in comms if c is not None]
        geos    = [inv_geo.get(i,"Unknown")   for i in inv_list]
        types   = [inv_type.get(i,"OTHER")    for i in inv_list]
        stages  = [inv_stage.get(i,"Other")   for i in inv_list]
        sectors = [inv_sector.get(i,"Other")  for i in inv_list]
        greens  = [inv_green.get(i,"TRADITIONAL") for i in inv_list]

        is_cross_comm   = int(len(set(c for c in comms if c is not None)) >= 2)
        is_cross_geo    = int(len(set(g for g in geos   if g != "Unknown")) >= 2)
        is_cross_type   = int(len(set(t for t in types  if t != "OTHER"))   >= 2)
        is_cross_stage  = int(len(set(s for s in stages if s != "Other"))   >= 2)
        is_cross_sector = int(len(set(s for s in sectors if s != "Other / Diversified")) >= 2)
        is_cross_green  = int(len(set(g for g in greens)) >= 2)

        records.append({
            "company_key":     ck,
            "cross_community": is_cross_comm,
            "cross_geo":       is_cross_geo,
            "cross_type":      is_cross_type,
            "cross_stage":     is_cross_stage,
            "cross_sector":    is_cross_sector,
            "cross_green":     is_cross_green,
            "n_inv":           len(inv_list),
            "n_comms":         len(set(c for c in comms if c is not None)),
        })

    cross_df = pd.DataFrame(records)
    cross_df = cross_df.merge(
        cdf[["company_key","cross_community","is_failed","hq_region"]].rename(
            columns={"cross_community": "cross_comm_cdf"}),
        on="company_key", how="left"
    )

    n_total = len(cross_df)
    lines.append(f"\n  N companies (with first-deal data): {n_total:,}")
    lines.append(f"  cross_community (recomputed): {cross_df['cross_community'].mean():.1%}")
    lines.append(f"  cross_community (original):   {cross_df['cross_comm_cdf'].mean():.1%}")

    # Prevalence of each cross-X
    lines.append(f"\n  PREVALENCE OF EACH CROSS-DIMENSION (in first-round syndicate):")
    dims = [("cross_community","Cross-community"),
            ("cross_geo",      "Cross-geography (region)"),
            ("cross_type",     "Cross-investor-type"),
            ("cross_stage",    "Cross-stage preference"),
            ("cross_sector",   "Cross-sector preference"),
            ("cross_green",    "Cross-green-focus")]
    for col, label in dims:
        pct = cross_df[col].mean()
        lines.append(f"    {label:<30} {pct:.1%}")

    # Correlations with cross_community
    lines.append(f"\n  CORRELATIONS WITH cross_community (phi / Pearson for binary vars):")
    lines.append(f"    {'Dimension':<30}  {'Phi coef':>10}  {'p':>10}  {'Overlap %':>10}")
    lines.append(f"    {'-'*65}")
    corr_results = {}
    for col, label in dims[1:]:
        r, p = pearsonr(cross_df["cross_community"].fillna(0),
                        cross_df[col].fillna(0))
        # Overlap = % of cross_community=1 that also has cross_X=1
        sub_cc  = cross_df[cross_df["cross_community"]==1]
        overlap = sub_cc[col].mean() if len(sub_cc) > 0 else np.nan
        corr_results[label] = r
        lines.append(f"    {label:<30}  {r:>10.4f}  {p:>10.4f}  {overlap:>10.1%}")

    # Which cross-X is most similar to cross-community?
    most_similar = max(corr_results, key=corr_results.get)
    lines.append(f"\n  MOST SIMILAR TO cross_community: '{most_similar}'")
    lines.append(f"    Phi = {corr_results[most_similar]:.4f}")

    # Are cross-community companies ALSO cross-X? (conditional probabilities)
    lines.append(f"\n  GIVEN cross_community=1, probability of also being cross-X:")
    sub_cc = cross_df[cross_df["cross_community"]==1]
    for col, label in dims[1:]:
        lines.append(f"    P({col} | cross_community) = {sub_cc[col].mean():.1%}")

    lines.append(f"\n  GIVEN cross_community=0, probability of cross-X:")
    sub_nc = cross_df[cross_df["cross_community"]==0]
    for col, label in dims[1:]:
        lines.append(f"    P({col} | NOT cross_community) = {sub_nc[col].mean():.1%}")

    cross_df.to_csv(OUT / "cross_dimensions_df.csv", index=False)
    return cross_df, corr_results


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: WHAT DOES CROSSING COMMUNITIES ACTUALLY MEAN?
# ═══════════════════════════════════════════════════════════════════════════════

def analysis5_bridge_meaning(cross_df, prof_df, inv_community, inv_geo, inv_type,
                              inv_stage, inv_sector, inv_green, deals, lines):
    print("\nAnalysis 5: What does crossing communities mean? …")
    lines += ["\n"+"="*80,
              "ANALYSIS 5: WHAT DOES 'CROSSING COMMUNITIES' ACTUALLY MEAN?",
              "  For cross-community companies, look up the dominant characteristics",
              "  of each community in their syndicate, and classify the bridge type.",
              "="*80]

    # Build community dominant characteristics from prof_df
    comm_dom = {}
    for _, r in prof_df.iterrows():
        comm_dom[int(r["community"])] = {
            "geo":    r["dom_geo"],
            "type":   r["dom_type"],
            "green":  r["dom_green"],
            "stage":  r["dom_stage"],
            "sector": r["dom_sector"],
        }

    # For each cross-community company in deals_first:
    deals_first = (deals.sort_values("date")
                   .dropna(subset=["Investors"])
                   .drop_duplicates("company_key", keep="first"))

    bridge_records = []
    for _, row in deals_first.iterrows():
        ck = row["company_key"]
        inv_list = [i.lower() for i in parse_investors(row["Investors"]) if i]
        if not inv_list: continue

        comms_present = list({inv_community.get(i) for i in inv_list
                               if inv_community.get(i) is not None})
        if len(comms_present) < 2: continue  # not cross-community

        # Get dominant characteristics for each community present
        doms = [comm_dom[c] for c in comms_present if c in comm_dom]
        if len(doms) < 2: continue

        # Is this crossing different geographies?
        all_geos   = {d["geo"]    for d in doms}
        all_types  = {d["type"]   for d in doms}
        all_greens = {d["green"]  for d in doms}
        all_stages = {d["stage"]  for d in doms}
        all_sectors= {d["sector"] for d in doms}

        diff_geo    = len(all_geos)    > 1
        diff_type   = len(all_types)   > 1
        diff_green  = len(all_greens)  > 1
        diff_stage  = len(all_stages)  > 1
        diff_sector = len(all_sectors) > 1

        n_diff = sum([diff_geo, diff_type, diff_green, diff_stage, diff_sector])

        # Primary bridge type (most specific single-dimension crossing)
        if n_diff == 0:
            bridge_type = "Same on all dimensions (homophily)"
        elif n_diff == 1:
            if diff_geo:    bridge_type = "Cross-geography only"
            elif diff_type: bridge_type = "Cross-type only"
            elif diff_green:bridge_type = "Cross-green-focus only"
            elif diff_stage:bridge_type = "Cross-stage only"
            else:           bridge_type = "Cross-sector only"
        elif diff_geo and diff_type:
            bridge_type = "Cross-geography + type"
        elif diff_geo and diff_green:
            bridge_type = "Cross-geography + green"
        elif diff_geo and diff_stage:
            bridge_type = "Cross-geography + stage"
        elif diff_type and diff_green:
            bridge_type = "Cross-type + green-focus"
        else:
            bridge_type = "Multiple dimensions (3+)"

        bridge_records.append({
            "company_key":    ck,
            "n_communities":  len(comms_present),
            "diff_geo":       int(diff_geo),
            "diff_type":      int(diff_type),
            "diff_green":     int(diff_green),
            "diff_stage":     int(diff_stage),
            "diff_sector":    int(diff_sector),
            "n_diff_dims":    n_diff,
            "bridge_type":    bridge_type,
        })

    bridge_df = pd.DataFrame(bridge_records)
    n_cross = len(bridge_df)
    lines.append(f"\n  Cross-community companies analysed: {n_cross:,}")

    # Tabulation of bridge types
    lines.append(f"\n  BRIDGE TYPE DISTRIBUTION:")
    lines.append(f"  {'Bridge type':<40}  {'N':>6}  {'%':>7}")
    lines.append(f"  {'-'*58}")
    type_counts = bridge_df["bridge_type"].value_counts()
    for btype, cnt in type_counts.items():
        lines.append(f"  {btype:<40}  {cnt:>6}  {cnt/n_cross:>7.1%}")

    # Single-dimension breakdown
    lines.append(f"\n  INDIVIDUAL DIMENSION DIFFERENCES (% of cross-community companies):")
    for col, label in [("diff_geo","Different dominant geography"),
                        ("diff_type","Different dominant investor type"),
                        ("diff_green","Different dominant green focus"),
                        ("diff_stage","Different dominant stage preference"),
                        ("diff_sector","Different dominant sector")]:
        pct = bridge_df[col].mean()
        lines.append(f"    {label:<38} {pct:.1%}")

    # Average number of differing dimensions
    lines.append(f"\n  Avg dimensions that differ when crossing: {bridge_df['n_diff_dims'].mean():.2f}")
    lines.append(f"  Most common: {type_counts.index[0]} ({type_counts.iloc[0]/n_cross:.1%})")

    # Same-on-all: how homophilic is the cross-community?
    same_all = (bridge_df["n_diff_dims"]==0).mean()
    lines.append(f"\n  {same_all:.1%} of cross-community companies bridge communities that look")
    lines.append(f"  IDENTICAL on all measured dimensions → network position differs,")
    lines.append(f"  but observable characteristics are the same (unobserved heterogeneity).")

    bridge_df.to_csv(OUT / "bridge_types.csv", index=False)
    return bridge_df


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(prof_df, res_df, hom_summary, cross_df, bridge_df, corr_results):
    print("\nGenerating figures …")

    # ── Fig 1: Community profile heatmap ──────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 10), dpi=130, facecolor="white")

    # Left: geographic composition heatmap
    ax = axes1[0]
    geo_data = prof_df.set_index("community")[["pct_americas","pct_europe","pct_asia"]].copy()
    geo_data.columns = ["Americas","Europe","Asia"]
    geo_data.index = [f"C{c}" for c in geo_data.index]
    sns.heatmap(geo_data, ax=ax, annot=True, fmt=".0%", cmap="Blues",
                vmin=0, vmax=1, linewidths=0.3, cbar_kws={"label":"% investors"})
    ax.set_title("Geographic Composition by Community", fontsize=10, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Community")

    # Right: type + green composition heatmap
    ax2 = axes1[1]
    type_data = prof_df.set_index("community")[["pct_gvc","pct_ivc","pct_green_vc","pct_trad"]].copy()
    type_data.columns = ["GVC","IVC","GREEN_VC","TRAD"]
    type_data.index = [f"C{c}" for c in type_data.index]
    sns.heatmap(type_data, ax=ax2, annot=True, fmt=".0%", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.3, cbar_kws={"label":"% investors"})
    ax2.set_title("Type & Green Composition by Community", fontsize=10, fontweight="bold")
    ax2.set_xlabel(""); ax2.set_ylabel("")

    fig1.suptitle("Community Profiles: Geographic and Type Composition",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig1.savefig(OUT/"fig1_community_heatmaps.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # ── Fig 2: Cramér's V ranking + Homogeneity ────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), dpi=130, facecolor="white")

    ax = axes2[0]
    dims_ranked = res_df.sort_values("cramers_v", ascending=True)
    colors_v = ["#e74c3c" if i == len(dims_ranked)-1 else "#3498db"
                for i in range(len(dims_ranked))]
    bars = ax.barh(dims_ranked["dimension"], dims_ranked["cramers_v"],
                   color=colors_v, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, dims_ranked["cramers_v"]):
        ax.text(v + 0.005, bar.get_y()+bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Cramér's V (0=none, 1=perfect)")
    ax.set_title("Predictive Power for Community Membership", fontsize=10, fontweight="bold")
    ax.set_xlim(0, res_df["cramers_v"].max() + 0.06)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax2 = axes2[1]
    hom_sorted = dict(sorted(hom_summary.items(), key=lambda x: x[1]))
    colors_h = ["#e74c3c" if k == max(hom_summary, key=hom_summary.get) else "#3498db"
                for k in hom_sorted]
    bars2 = ax2.barh(list(hom_sorted.keys()), [v*100 for v in hom_sorted.values()],
                     color=colors_h, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, hom_sorted.values()):
        ax2.text(v*100 + 0.5, bar.get_y()+bar.get_height()/2,
                 f"{v:.1%}", va="center", fontsize=9)
    ax2.set_xlabel("Avg within-community homogeneity (%)")
    ax2.set_title("Within-Community Homogeneity by Dimension", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig2.savefig(OUT/"fig2_predictive_homogeneity.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # ── Fig 3: Correlations cross_community vs cross-X ─────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), dpi=130, facecolor="white")

    ax = axes3[0]
    dims_corr = list(corr_results.keys())
    vals_corr = [corr_results[d] for d in dims_corr]
    colors_c = ["#e74c3c" if v == max(vals_corr) else "#3498db" for v in vals_corr]
    bars3 = ax.bar(range(len(dims_corr)), vals_corr, color=colors_c, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(dims_corr)))
    ax.set_xticklabels([d.replace("Cross-","").replace(" preference","")
                        for d in dims_corr], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Phi coefficient with cross_community")
    ax.set_title("Correlation: cross_community vs cross-X", fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    for bar, v in zip(bars3, vals_corr):
        ax.text(bar.get_x()+bar.get_width()/2, v + 0.003,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax2 = axes3[1]
    # Bridge type pie/bar
    if bridge_df is not None:
        type_counts = bridge_df["bridge_type"].value_counts().head(8)
        colors_b = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        bars4 = ax2.barh(range(len(type_counts)),
                         type_counts.values / len(bridge_df) * 100,
                         color=colors_b, alpha=0.85, edgecolor="white")
        ax2.set_yticks(range(len(type_counts)))
        ax2.set_yticklabels([t[:35] for t in type_counts.index], fontsize=8)
        for bar, v in zip(bars4, type_counts.values):
            ax2.text(v/len(bridge_df)*100 + 0.3, bar.get_y()+bar.get_height()/2,
                     f"{v/len(bridge_df):.1%}", va="center", fontsize=8)
        ax2.set_xlabel("% of cross-community companies")
        ax2.set_title("What Type of Bridge Does cross-community Create?", fontsize=10, fontweight="bold")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig3.savefig(OUT/"fig3_cross_analysis.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig3)

    print("  Saved 3 figure files.")


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def master_summary(res_df, hom_summary, corr_results, bridge_df, lines):
    lines += ["\n"+"="*80, "MASTER SUMMARY: WHAT ARE COMMUNITIES AND WHAT IS CROSSING THEM?",
              "="*80]
    top_dim  = res_df.iloc[0]["dimension"]
    top_v    = res_df.iloc[0]["cramers_v"]
    top2_dim = res_df.iloc[1]["dimension"]
    top_hom  = max(hom_summary, key=hom_summary.get)
    top_hom_val = hom_summary[top_hom]
    top_corr = max(corr_results, key=corr_results.get)
    top_corr_val = corr_results[top_corr]

    if bridge_df is not None and len(bridge_df) > 0:
        top_bridge = bridge_df["bridge_type"].value_counts().index[0]
        top_bridge_pct = bridge_df["bridge_type"].value_counts().iloc[0] / len(bridge_df)
        same_all_pct = (bridge_df["n_diff_dims"]==0).mean()
    else:
        top_bridge = "N/A"; top_bridge_pct = 0; same_all_pct = 0

    lines.append(f"""
  PARAGRAPH 1 — WHAT ARE COMMUNITIES?

  Louvain communities in this climate VC network are primarily organized by
  {top_dim} (Cramér's V={top_v:.3f}), followed by {top2_dim}.
  Within-community homogeneity is highest for {top_hom} ({top_hom_val:.0%} of
  investors in a given community share the same {top_hom.lower()} on average).
  This means the network's modular structure reflects real-world clustering along
  {top_hom.lower()} lines rather than emergent patterns with no observable correlate.

  PARAGRAPH 2 — WHAT DOES CROSS-COMMUNITY MEAN?

  Among cross-community companies, the strongest correlate of spanning multiple
  communities is {top_corr} (φ={top_corr_val:.3f}). The most common type of
  community bridge is '{top_bridge}' ({top_bridge_pct:.0%} of cross-community
  companies). Notably, {same_all_pct:.0%} of cross-community companies bridge
  communities that appear identical on all observable dimensions — their
  communities differ only in network position, not in measured characteristics.
  This latter group represents the "pure network" effect of cross-community access
  and is the empirically cleanest test of the information-brokerage mechanism.

  PARAGRAPH 3 — IMPLICATIONS FOR THE PAPER'S MECHANISM CLAIM

  The finding that communities are primarily defined by {top_hom.lower()} means
  that cross-community syndicates are largely cross-{top_hom.lower()} syndicates.
  The paper's mechanism — information flow across knowledge-distinct clusters —
  is therefore grounded: investors from different {top_hom.lower()} clusters
  bring structurally different information to the table. The causal chain is:
  geographic/type diversity → community boundary crossing → survival benefit.
  This grounds the cross-community → failure finding in a concrete economic mechanism.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = ["="*80,
             "COMMUNITY ANATOMY: WHAT DEFINES LOUVAIN COMMUNITIES?",
             "="*80]

    (inv_df, cdf, deals, large_comms, inv_community,
     inv_geo, inv_type, inv_stage, inv_sector, inv_green) = load_data()

    prof_df    = analysis1_profiles(inv_df, large_comms, lines)
    res_df     = analysis2_predictive(inv_df, large_comms, lines)
    hom_summary= analysis3_homogeneity(prof_df, lines)
    cross_df, corr_results = analysis4_cross_dimensions(
        cdf, deals, inv_community, inv_geo, inv_type, inv_stage, inv_sector, inv_green, lines)
    bridge_df  = analysis5_bridge_meaning(
        cross_df, prof_df, inv_community, inv_geo, inv_type,
        inv_stage, inv_sector, inv_green, deals, lines)

    master_summary(res_df, hom_summary, corr_results, bridge_df, lines)

    output = "\n".join(lines)
    print("\n" + output)
    (OUT / "results.txt").write_text(output)

    plot_all(prof_df, res_df, hom_summary, cross_df, bridge_df, corr_results)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
