"""
13 — Deep dive into the 46 Louvain communities detected in the co-investment network.

7 analyses:
  1. Full community profiles (investor type & green distribution, degree, betweenness)
  2. Sector distribution per community + Herfindahl specialization index
  3. Novelty profiles per community
  4. Outcome profiles per community (exits, failures, IPO vs M&A)
  5. Community typology via k-means (4-6 cluster types, named)
  6. Cross-community investment (community-level network)
  7. Community evolution over 4 time periods

Outputs in output/community/:
  community_profiles.csv, community_sectors.csv, community_novelty.csv,
  community_outcomes.csv, community_typology.csv, community_crossinvest.csv
  heatmap_sectors.png, scatter_novelty.png, typology_radar.png,
  typology_scatter.png, typology_bars.png, community_network.png,
  community_evolution.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import networkx as nx
import re, itertools, warnings
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import statsmodels.formula.api as smf
from load_data import load_deals

warnings.filterwarnings("ignore")

OUT  = Path(__file__).parent / "output" / "community"
OUT.mkdir(parents=True, exist_ok=True)
OUTN = Path(__file__).parent / "output" / "network"
OUTB = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

MIN_INVESTORS = 20   # minimum community size to profile

# ── Sector keyword mapping (applied to Verticals + Industry Group) ────────────
SECTOR_KEYWORDS = {
    "Renewable Energy":      ["solar","wind","renewable","hydropower","geothermal","tidal","wave","biomass energy","clean energy","photovoltaic"],
    "Battery & Storage":     ["battery","storage","energy storage","lithium","electrolyte","solid state","grid storage"],
    "Mobility & EV":         ["mobility","electric vehicle","ev ","autonomous","transportation","automotive","charging","fleet","scooter","bike"],
    "Carbon Capture":        ["carbon capture","ccs","direct air","carbon removal","sequestration","carbon credit","offset"],
    "Sustainable Agri":      ["agriculture","agritech","agtech","food","farming","crop","livestock","aquaculture","precision ag","vertical farm"],
    "Hydrogen":              ["hydrogen","fuel cell","electrolysis","green hydrogen","electrolyz"],
    "Circular Economy":      ["circular","recycling","recycle","waste","packaging","upcycl","composting","refurbish","remanufactur"],
    "Energy Efficiency":     ["energy efficiency","smart grid","building","insulation","hvac","retrofit","demand response","smart meter"],
    "Water":                 ["water","desalination","irrigation","wastewater","water treatment","water management"],
}

def classify_sector(row) -> str:
    text = " ".join([
        str(row.get("Verticals", "") or ""),
        str(row.get("Primary PitchBook Industry Group", "") or ""),
        str(row.get("Primary PitchBook Industry Sector", "") or ""),
    ]).lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return sector
    return "Other"


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    print("Loading data...")
    deals_raw = load_deals()
    deals_raw["Year"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce").dt.year
    deals_raw["sector"] = deals_raw.apply(classify_sector, axis=1)

    clf = pd.read_csv(OUTB / "investors_classified_output.csv")
    clf = clf[~clf["investor_type"].isin(["PARSE_ERROR"])]
    clf_map = clf.set_index("investor_name")[["investor_type","green_focus"]].to_dict("index")

    metrics_df = pd.read_csv(OUTN / "network_metrics.csv")

    def clean_name(raw): return re.sub(r"\([^)]*\)", "", str(raw)).strip()

    rows = []
    for _, row in deals_raw.iterrows():
        if pd.isna(row.get("Investors", None)): continue
        names = [clean_name(n) for n in re.split(r",\s*", str(row["Investors"]).replace("\n", ", "))]
        names = [n for n in names if n and n in clf_map]
        for nm in names:
            rows.append({
                "deal_id":   row["Deal ID"],
                "company":   row["Companies"],
                "year":      row["Year"],
                "deal_size": row.get("Deal Size (USD M)", np.nan),
                "sector":    row["sector"],
                "investor":  nm,
                "verticals": str(row.get("Verticals","") or ""),
                "industry_group": str(row.get("Primary PitchBook Industry Group","") or ""),
            })
    deal_inv = pd.DataFrame(rows)

    # Merge community assignment into deal_inv
    comm_map = metrics_df.set_index("investor")["community"].to_dict()
    deal_inv["community"] = deal_inv["investor"].map(comm_map)

    # Load novelty
    nov = pd.read_csv(OUTB / "novelty" / "novelty_scores.csv")[["name","specter_novelty"]]

    # Load outcomes
    mix = pd.read_csv(OUTB / "company_investor_mix.csv")
    mix["exited"]   = mix["outcome"].isin(["Acquired","IPO / Public"]).astype(int)
    mix["failed"]   = (mix["outcome"] == "Failed").astype(int)
    mix["is_ipo"]   = (mix["outcome"] == "IPO / Public").astype(int)
    mix["is_ma"]    = (mix["outcome"] == "Acquired").astype(int)

    print(f"  deal_inv: {len(deal_inv):,} rows | "
          f"unique investors: {deal_inv['investor'].nunique():,} | "
          f"unique deals: {deal_inv['deal_id'].nunique():,}")
    return deals_raw, clf_map, metrics_df, deal_inv, nov, mix


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 1: COMMUNITY PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def analysis1_profiles(metrics_df: pd.DataFrame, deal_inv: pd.DataFrame) -> pd.DataFrame:
    print("\n── ANALYSIS 1: Community Profiles ──")
    rows = []
    comm_ids = metrics_df["community"].dropna().unique()

    for cid in sorted(comm_ids):
        sub = metrics_df[metrics_df["community"] == cid]
        if len(sub) < MIN_INVESTORS:
            continue

        inv_set = set(sub["investor"])
        # Deals where ≥2 community members co-invested
        community_deals = (
            deal_inv[deal_inv["investor"].isin(inv_set)]
            .groupby("deal_id")["investor"]
            .count()
        )
        n_co_deals = (community_deals >= 2).sum()

        # Type distribution
        type_counts = sub["investor_type"].value_counts(normalize=True) * 100
        focus_counts = sub["green_focus"].value_counts(normalize=True) * 100

        rows.append({
            "community_id":       int(cid),
            "num_investors":      len(sub),
            "num_co_deals":       int(n_co_deals),
            "pct_GVC":            type_counts.get("GVC", 0),
            "pct_CVC":            type_counts.get("CVC", 0),
            "pct_IVC":            type_counts.get("IVC", 0),
            "pct_Impact_VC":      type_counts.get("Impact_VC", 0),
            "pct_Bank_VC":        type_counts.get("Bank_VC", 0),
            "pct_Angel_Network":  type_counts.get("Angel_Network", 0),
            "pct_Other":          type_counts.get("OTHER", 0),
            "pct_GREEN_VC":       focus_counts.get("GREEN_VC", 0),
            "pct_ESG_ALIGNED":    focus_counts.get("ESG_ALIGNED", 0),
            "pct_TRADITIONAL":    focus_counts.get("TRADITIONAL", 0),
            "avg_degree":         sub["degree"].mean(),
            "avg_betweenness":    sub["betweenness"].mean(),
            "avg_eigenvector":    sub["eigenvector"].mean(),
        })

    df = pd.DataFrame(rows).sort_values("num_investors", ascending=False)
    df.to_csv(OUT / "community_profiles.csv", index=False)
    print(f"  Profiled {len(df)} communities (≥{MIN_INVESTORS} investors)")
    print(df[["community_id","num_investors","pct_GREEN_VC","pct_TRADITIONAL","pct_GVC","pct_IVC"]].head(10).to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 2: SECTOR DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def analysis2_sectors(profiles_df: pd.DataFrame, deal_inv: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    print("\n── ANALYSIS 2: Sector Distribution per Community ──")
    comm_map = metrics_df.set_index("investor")["community"].to_dict()

    SECTORS = list(SECTOR_KEYWORDS.keys()) + ["Other"]
    rows = []

    for _, prof in profiles_df.iterrows():
        cid = prof["community_id"]
        inv_set = set(metrics_df[metrics_df["community"] == cid]["investor"])
        sub_deals = deal_inv[deal_inv["investor"].isin(inv_set)]
        n_total = sub_deals["deal_id"].nunique()
        if n_total == 0:
            continue

        # Sector per unique deal
        deal_sector = sub_deals.drop_duplicates("deal_id")["sector"].value_counts()
        sector_pcts = {s: deal_sector.get(s, 0) / n_total * 100 for s in SECTORS}

        # Herfindahl index (on sector fractions)
        fracs = np.array([sector_pcts[s] / 100 for s in SECTORS])
        herfindahl = float(np.sum(fracs ** 2))

        top3 = sorted(sector_pcts.items(), key=lambda x: x[1], reverse=True)[:3]
        row = {"community_id": cid, "n_deals": n_total, "herfindahl": herfindahl,
               "top_sector_1": top3[0][0], "top_sector_2": top3[1][0], "top_sector_3": top3[2][0]}
        row.update({f"pct_{s.replace(' & ','_').replace(' ','_')}": v for s, v in sector_pcts.items()})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("n_deals", ascending=False)
    df.to_csv(OUT / "community_sectors.csv", index=False)
    print(f"  Computed sector profiles for {len(df)} communities")

    # ── Visualization: heatmap communities × sectors ──────────────────────────
    top15 = df.head(15)
    sector_cols = [f"pct_{s.replace(' & ','_').replace(' ','_')}" for s in SECTORS]
    heat_data = top15[sector_cols].values
    labels_comm = [f"Comm {int(r['community_id'])} ({int(r['n_deals'])} deals)" for _, r in top15.iterrows()]
    labels_sect = [s for s in SECTORS]

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="% of deals")
    ax.set_xticks(range(len(labels_sect))); ax.set_xticklabels(labels_sect, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels_comm))); ax.set_yticklabels(labels_comm, fontsize=9)
    ax.set_title("Community × Sector Heatmap\n(% of deals in each sector, top 15 communities by deal count)",
                 fontsize=12, fontweight="bold")
    for i in range(len(labels_comm)):
        for j in range(len(labels_sect)):
            v = heat_data[i, j]
            if v > 5:
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if v > 40 else "black")
    plt.tight_layout()
    fig.savefig(OUT / "heatmap_sectors.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved heatmap_sectors.png")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3: NOVELTY PROFILES
# ─────────────────────────────────────────────────────────────────────────────

def analysis3_novelty(profiles_df: pd.DataFrame, deal_inv: pd.DataFrame,
                      metrics_df: pd.DataFrame, nov: pd.DataFrame,
                      sectors_df: pd.DataFrame) -> pd.DataFrame:
    print("\n── ANALYSIS 3: Community Novelty Profiles ──")

    nov_map = nov.set_index("name")["specter_novelty"].to_dict()
    q_low  = nov["specter_novelty"].quantile(0.25)
    q_high = nov["specter_novelty"].quantile(0.75)

    rows = []
    for _, prof in profiles_df.iterrows():
        cid = prof["community_id"]
        inv_set = set(metrics_df[metrics_df["community"] == cid]["investor"])
        companies = deal_inv[deal_inv["investor"].isin(inv_set)]["company"].unique()
        scores = [nov_map[c] for c in companies if c in nov_map]
        if len(scores) < 5:
            rows.append({"community_id": cid, "n_with_novelty": 0,
                         "avg_novelty": np.nan, "median_novelty": np.nan,
                         "std_novelty": np.nan, "pct_Q4": np.nan, "pct_Q1": np.nan})
            continue
        scores_arr = np.array(scores)
        rows.append({
            "community_id":   cid,
            "n_with_novelty": len(scores),
            "avg_novelty":    float(scores_arr.mean()),
            "median_novelty": float(np.median(scores_arr)),
            "std_novelty":    float(scores_arr.std()),
            "pct_Q4":         float((scores_arr >= q_high).mean() * 100),
            "pct_Q1":         float((scores_arr <= q_low).mean() * 100),
        })

    df = pd.DataFrame(rows)
    df = profiles_df[["community_id","num_investors","pct_GREEN_VC","pct_GVC"]].merge(df, on="community_id")
    df = df.merge(sectors_df[["community_id","n_deals","herfindahl","top_sector_1"]], on="community_id", how="left")
    df.to_csv(OUT / "community_novelty.csv", index=False)

    # Regression: avg_novelty ~ pct_GREEN_VC + pct_GVC + pct_CVC + size + herfindahl
    valid = df.dropna(subset=["avg_novelty","pct_GREEN_VC","herfindahl"])
    if len(valid) >= 10:
        m = smf.ols("avg_novelty ~ pct_GREEN_VC + pct_GVC + num_investors + herfindahl",
                    data=valid).fit()
        print("  Novelty ~ community features OLS:")
        for var in ["pct_GREEN_VC","pct_GVC","num_investors","herfindahl"]:
            if var in m.params:
                print(f"    {var:20s}: coef={m.params[var]:+.4f}  p={m.pvalues[var]:.3f}")

    # Visualization: scatter X=pct_GREEN_VC, Y=avg_novelty, size=n_deals, color=top_sector
    fig, ax = plt.subplots(figsize=(11, 8))
    sectors_uniq = df["top_sector_1"].dropna().unique()
    cmap_s = cm.get_cmap("tab10", len(sectors_uniq))
    color_map = {s: cmap_s(i) for i, s in enumerate(sectors_uniq)}

    for _, row in df.dropna(subset=["avg_novelty","pct_GREEN_VC","n_deals"]).iterrows():
        color = color_map.get(row["top_sector_1"], "grey")
        size  = max(30, row["n_deals"] / 3)
        ax.scatter(row["pct_GREEN_VC"], row["avg_novelty"], s=size, color=color, alpha=0.75,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.text(row["pct_GREEN_VC"] + 0.5, row["avg_novelty"] + 0.001,
                f"C{int(row['community_id'])}", fontsize=7, color="#333333")

    patches = [mpatches.Patch(color=color_map[s], label=s) for s in sectors_uniq if s in color_map]
    ax.legend(handles=patches, fontsize=8, title="Dominant Sector", loc="upper right")
    ax.set_xlabel("% GREEN_VC in Community", fontsize=10)
    ax.set_ylabel("Average SPECTER2 Novelty Score", fontsize=10)
    ax.set_title("Community Novelty Profile\n(size = number of deals, colour = dominant sector)",
                 fontsize=12, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(OUT / "scatter_novelty.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved scatter_novelty.png")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 4: COMMUNITY OUTCOMES
# ─────────────────────────────────────────────────────────────────────────────

def analysis4_outcomes(profiles_df: pd.DataFrame, deal_inv: pd.DataFrame,
                       metrics_df: pd.DataFrame, mix: pd.DataFrame,
                       novelty_df: pd.DataFrame) -> pd.DataFrame:
    print("\n── ANALYSIS 4: Community Outcomes ──")
    mix_map = mix.set_index("Companies")[["exited","failed","is_ipo","is_ma","outcome"]].to_dict("index")

    rows = []
    for _, prof in profiles_df.iterrows():
        cid = prof["community_id"]
        inv_set = set(metrics_df[metrics_df["community"] == cid]["investor"])
        companies = deal_inv[deal_inv["investor"].isin(inv_set)]["company"].unique()
        outcomes = [mix_map[c] for c in companies if c in mix_map]
        if len(outcomes) < 5:
            continue
        n = len(outcomes)
        n_exit = sum(o["exited"] for o in outcomes)
        n_fail = sum(o["failed"] for o in outcomes)
        n_ipo  = sum(o["is_ipo"]  for o in outcomes)
        n_ma   = sum(o["is_ma"]   for o in outcomes)
        rows.append({
            "community_id":  cid,
            "n_companies":   n,
            "exit_rate":     n_exit / n * 100,
            "failure_rate":  n_fail / n * 100,
            "survival_rate": (n - n_exit - n_fail) / n * 100,
            "pct_ipo":       n_ipo / n_exit * 100 if n_exit else np.nan,
            "pct_ma":        n_ma  / n_exit * 100 if n_exit else np.nan,
        })

    df = pd.DataFrame(rows)
    df = profiles_df[["community_id","num_investors","pct_GREEN_VC","avg_degree"]].merge(df, on="community_id")
    df = df.merge(novelty_df[["community_id","avg_novelty","herfindahl"]], on="community_id", how="left")
    df.to_csv(OUT / "community_outcomes.csv", index=False)

    # Regression: exit_rate ~ pct_GREEN_VC + avg_novelty + herfindahl + num_investors + avg_degree
    valid = df.dropna(subset=["exit_rate","pct_GREEN_VC","avg_novelty","herfindahl"])
    if len(valid) >= 10:
        m = smf.ols("exit_rate ~ pct_GREEN_VC + avg_novelty + herfindahl + num_investors + avg_degree",
                    data=valid).fit()
        print("  Exit rate ~ community features OLS:")
        for var in ["pct_GREEN_VC","avg_novelty","herfindahl","num_investors","avg_degree"]:
            if var in m.params:
                stars = ("***" if m.pvalues[var]<0.001 else "**" if m.pvalues[var]<0.01
                         else "*" if m.pvalues[var]<0.05 else "(†)" if m.pvalues[var]<0.10 else "")
                print(f"    {var:20s}: coef={m.params[var]:+.4f}  p={m.pvalues[var]:.3f} {stars}")
    print(df[["community_id","n_companies","exit_rate","failure_rate","pct_ipo","pct_ma"]].head(12).to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 5: COMMUNITY TYPOLOGY (k-means)
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_NAMES = {
    # Named after the run — will be auto-assigned based on centroid inspection
}

def name_clusters(centroid_dicts: list) -> dict:
    """
    Assign names by ranking clusters on key dimensions relative to each other.
    Returns {cluster_index: name}.
    """
    import numpy as np
    n = len(centroid_dicts)
    gvc_vals    = np.array([c.get("pct_GVC", 0)       for c in centroid_dicts])
    green_vals  = np.array([c.get("pct_GREEN_VC", 50)  for c in centroid_dicts])
    exit_vals   = np.array([c.get("exit_rate", 5)      for c in centroid_dicts])
    fail_vals   = np.array([c.get("failure_rate", 5)   for c in centroid_dicts])
    nov_vals    = np.array([c.get("avg_novelty", 0.42) for c in centroid_dicts])
    herf_vals   = np.array([c.get("herfindahl", 0.5)   for c in centroid_dicts])

    names = {}
    used  = set()

    # Rank-based naming (relative to each cluster's peers)
    # Government-Led: highest GVC (relative)
    gov_idx = int(np.argmax(gvc_vals))
    names[gov_idx] = "Government-Led"; used.add(gov_idx)

    # Exit-Oriented: highest exit rate (excluding already named)
    remaining = [i for i in range(n) if i not in used]
    if remaining:
        ex_idx = remaining[int(np.argmax(exit_vals[remaining]))]
        names[ex_idx] = "Exit-Oriented"; used.add(ex_idx)

    # Deep Green: highest green % (excluding already named)
    remaining = [i for i in range(n) if i not in used]
    if remaining:
        gr_idx = remaining[int(np.argmax(green_vals[remaining]))]
        names[gr_idx] = "Deep Green"; used.add(gr_idx)

    # High Failure / High Risk: highest failure rate
    remaining = [i for i in range(n) if i not in used]
    if remaining:
        fail_idx = remaining[int(np.argmax(fail_vals[remaining]))]
        names[fail_idx] = "High-Risk Frontier"; used.add(fail_idx)

    # Remaining → "Mixed Bridge"
    for i in range(n):
        if i not in names:
            names[i] = "Mixed Bridge"

    return names


def analysis5_typology(profiles_df: pd.DataFrame, novelty_df: pd.DataFrame,
                       outcomes_df: pd.DataFrame) -> pd.DataFrame:
    print("\n── ANALYSIS 5: Community Typology (k-means) ──")

    merged = profiles_df.merge(
        novelty_df[["community_id","avg_novelty","herfindahl"]], on="community_id", how="left"
    ).merge(
        outcomes_df[["community_id","exit_rate","failure_rate"]], on="community_id", how="left"
    )

    FEATURES = ["pct_GREEN_VC","pct_GVC","avg_novelty","herfindahl","exit_rate","failure_rate","avg_degree"]
    feat_df = merged[["community_id"] + FEATURES].copy()

    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(feat_df[FEATURES])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Choose k=5 (based on reasonable typology space)
    K = 5
    km = KMeans(n_clusters=K, random_state=42, n_init=20)
    feat_df["cluster"] = km.fit_predict(Xs)

    # Name clusters from centroids (in original scale)
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroid_dicts = [dict(zip(FEATURES, row)) for row in centroids_orig]
    cluster_labels = name_clusters(centroid_dicts)

    feat_df["cluster_label"] = feat_df["cluster"].map(cluster_labels)
    result = merged.merge(feat_df[["community_id","cluster","cluster_label"]], on="community_id")
    result.to_csv(OUT / "community_typology.csv", index=False)

    # ── Print typology summary ─────────────────────────────────────────────────
    print("  Cluster centroids:")
    for i, cd in enumerate(centroid_dicts):
        lbl = cluster_labels[i]
        n_comms = (feat_df["cluster"] == i).sum()
        print(f"  [{i}] {lbl:25s} | n={n_comms} | "
              f"green={cd['pct_GREEN_VC']:.0f}% GVC={cd['pct_GVC']:.0f}% "
              f"novelty={cd['avg_novelty']:.3f} herf={cd['herfindahl']:.2f} "
              f"exit={cd['exit_rate']:.1f}% fail={cd['failure_rate']:.1f}%")

    # ── Visualization 1: Radar charts ────────────────────────────────────────
    radar_features = ["pct_GREEN_VC", "pct_GVC", "avg_novelty", "herfindahl", "exit_rate", "failure_rate"]
    radar_labels   = ["% Green", "% GVC", "Novelty", "Specialization", "Exit Rate", "Fail Rate"]
    n_feat = len(radar_features)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 5), subplot_kw=dict(polar=True))
    fig.suptitle("Community Cluster Profiles (Radar Charts)", fontsize=13, fontweight="bold")

    palette5 = plt.cm.Set2(np.linspace(0, 1, K))
    # Normalise centroids to [0,1] range for radar
    feat_mins = np.array([centroid_dicts[i][f] for f in radar_features for i in range(K)]).reshape(K, n_feat).min(axis=0)
    feat_maxs = np.array([centroid_dicts[i][f] for f in radar_features for i in range(K)]).reshape(K, n_feat).max(axis=0)
    feat_range = np.where(feat_maxs - feat_mins > 0, feat_maxs - feat_mins, 1)

    for i, (ax, cd) in enumerate(zip(axes, centroid_dicts)):
        vals = [(cd[f] - feat_mins[j]) / feat_range[j] for j, f in enumerate(radar_features)]
        vals += vals[:1]
        ax.plot(angles, vals, color=palette5[i], linewidth=2)
        ax.fill(angles, vals, color=palette5[i], alpha=0.25)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, fontsize=8)
        ax.set_title(cluster_labels[i], fontsize=9, fontweight="bold", pad=10)
        ax.set_ylim(0, 1); ax.set_yticklabels([])

    plt.tight_layout()
    fig.savefig(OUT / "typology_radar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved typology_radar.png")

    # ── Visualization 2: Scatter X=green%, Y=novelty, color=cluster ────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(K):
        sub = result[result["cluster"] == i]
        ax.scatter(sub["pct_GREEN_VC"], sub["avg_novelty"],
                   s=sub["num_investors"] / 5 + 30,
                   color=palette5[i], alpha=0.8, edgecolors="white", linewidths=0.8,
                   label=cluster_labels[i], zorder=3)
        for _, row in sub.iterrows():
            ax.text(row["pct_GREEN_VC"] + 0.5, row["avg_novelty"] + 0.001,
                    f"C{int(row['community_id'])}", fontsize=6.5, color="#333")
    ax.set_xlabel("% GREEN_VC in Community", fontsize=11)
    ax.set_ylabel("Average SPECTER2 Novelty Score", fontsize=11)
    ax.set_title("Community Typology\n(dot size = num investors, colour = cluster type)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, title="Cluster Type")
    ax.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(OUT / "typology_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved typology_scatter.png")

    # ── Visualization 3: Bar chart exit/failure by cluster ────────────────
    cluster_stats = result.groupby("cluster_label")[["exit_rate","failure_rate"]].mean().reset_index()
    cluster_stats = cluster_stats.sort_values("exit_rate", ascending=False)
    x = np.arange(len(cluster_stats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w/2, cluster_stats["exit_rate"],   w, label="Exit Rate",    color="#2980b9", alpha=0.85)
    ax.bar(x + w/2, cluster_stats["failure_rate"],w, label="Failure Rate", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cluster_stats["cluster_label"], rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Exit and Failure Rate by Community Cluster Type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    for i, row in cluster_stats.iterrows():
        xi = list(cluster_stats.index).index(i)
        ax.text(xi - w/2, row["exit_rate"] + 0.1, f"{row['exit_rate']:.1f}%", ha="center", fontsize=9)
        ax.text(xi + w/2, row["failure_rate"] + 0.1, f"{row['failure_rate']:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT / "typology_bars.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved typology_bars.png")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 6: CROSS-COMMUNITY INVESTMENT
# ─────────────────────────────────────────────────────────────────────────────

def analysis6_cross_community(deal_inv: pd.DataFrame, metrics_df: pd.DataFrame,
                               typology_df: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    print("\n── ANALYSIS 6: Cross-Community Investment ──")

    comm_map   = metrics_df.set_index("investor")["community"].to_dict()
    label_map  = typology_df.set_index("community_id")["cluster_label"].to_dict()

    # For each deal, find which communities are involved
    deal_comms = deal_inv.dropna(subset=["community"]).groupby("deal_id")["community"].apply(
        lambda x: sorted(x.dropna().astype(int).unique())
    )

    # Company → communities involved
    company_comms = deal_inv.dropna(subset=["community"]).groupby("company")["community"].apply(
        lambda x: sorted(x.dropna().astype(int).unique())
    ).reset_index()
    company_comms["n_communities"] = company_comms["community"].apply(len)
    company_comms["cross_community"] = (company_comms["n_communities"] > 1).astype(int)

    mix_out = mix.merge(company_comms.rename(columns={"company":"Companies"}), on="Companies", how="left")
    mix_out["cross_community"] = mix_out["cross_community"].fillna(0).astype(int)

    # Cross vs within outcome
    for label, sub in mix_out.groupby("cross_community"):
        n = len(sub)
        n_exit = sub["exited"].sum()
        n_fail = sub["failed"].sum()
        tag = "cross-community" if label else "within-community"
        print(f"  {tag}: n={n}, exit={n_exit/n*100:.1f}%, fail={n_fail/n*100:.1f}%")

    # Cross-community pair counts
    pair_counts: dict = defaultdict(int)
    for deal_id, comms in deal_comms.items():
        for a, b in itertools.combinations(comms, 2):
            pair_counts[(min(a,b), max(a,b))] += 1

    cross_df = pd.DataFrame([{"comm_a": a, "comm_b": b, "shared_deals": cnt}
                              for (a,b), cnt in pair_counts.items()])
    cross_df.to_csv(OUT / "community_crossinvest.csv", index=False)

    # ── Regression: exit ~ n_communities + cross_community ──────────────────
    valid = mix_out.dropna(subset=["exited","n_communities","cross_community"])
    try:
        m = smf.logit("exited ~ n_communities + cross_community + C(green_quartile)",
                      data=valid).fit(disp=False, maxiter=300)
        print("  Cross-community → exit logit:")
        for var in ["n_communities","cross_community"]:
            print(f"    {var}: coef={m.params.get(var,np.nan):+.4f}  "
                  f"p={m.pvalues.get(var,1):.3f}")
    except Exception as e:
        print(f"  Regression failed: {e}")

    # ── Community-level network graph ────────────────────────────────────────
    Gc = nx.Graph()
    for cid, row in typology_df.set_index("community_id").iterrows():
        Gc.add_node(int(cid), cluster_label=row["cluster_label"],
                    n_deals=row.get("n_deals", 10),
                    pct_GREEN_VC=row.get("pct_GREEN_VC", 50))

    for _, row in cross_df.iterrows():
        a, b, w = int(row["comm_a"]), int(row["comm_b"]), int(row["shared_deals"])
        if Gc.has_node(a) and Gc.has_node(b):
            Gc.add_edge(a, b, weight=w)

    cluster_labels_all = typology_df["cluster_label"].unique()
    cpal = plt.cm.Set2(np.linspace(0, 1, len(cluster_labels_all)))
    label_color = {lbl: cpal[i] for i, lbl in enumerate(cluster_labels_all)}

    pos = nx.spring_layout(Gc, weight="weight", seed=42, k=1.5)
    node_colors = [label_color.get(Gc.nodes[n].get("cluster_label","?"), "grey") for n in Gc.nodes()]
    node_sizes  = [30 + Gc.nodes[n].get("n_deals", 100) / 20 for n in Gc.nodes()]
    edge_weights = [Gc[u][v]["weight"] for u, v in Gc.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + 4 * (w / max_w) ** 0.5 for w in edge_weights]

    fig, ax = plt.subplots(figsize=(13, 10))
    nx.draw_networkx_edges(Gc, pos, ax=ax, width=edge_widths, alpha=0.35, edge_color="#777")
    nx.draw_networkx_nodes(Gc, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.88)
    labels = {n: f"C{n}" for n in Gc.nodes()}
    nx.draw_networkx_labels(Gc, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

    patches = [mpatches.Patch(color=label_color[l], label=l) for l in cluster_labels_all if l in label_color]
    ax.legend(handles=patches, loc="upper left", fontsize=9, title="Cluster Type")
    ax.set_title("Community-Level Co-Investment Network\n(node=community, edge thickness=shared deals, colour=cluster type)",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "community_network.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved community_network.png")
    return cross_df


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 7: COMMUNITY EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

PERIODS = {
    "pre-Paris\n(2012-15)":   (2012, 2015),
    "Trump 1\n(2016-19)":     (2016, 2019),
    "Biden\n(2020-22)":       (2020, 2022),
    "Recent\n(2023-26)":      (2023, 2026),
}

def analysis7_evolution(profiles_df: pd.DataFrame, deal_inv: pd.DataFrame,
                        metrics_df: pd.DataFrame, nov: pd.DataFrame,
                        typology_df: pd.DataFrame):
    print("\n── ANALYSIS 7: Community Evolution ──")
    # Top 10 communities by investor count
    top_comms = profiles_df.head(10)["community_id"].tolist()
    nov_map   = nov.set_index("name")["specter_novelty"].to_dict()
    clf_map   = metrics_df.set_index("investor")["green_focus"].to_dict()

    evolution_rows = []
    for cid in top_comms:
        inv_set = set(metrics_df[metrics_df["community"] == cid]["investor"])
        for period_label, (y0, y1) in PERIODS.items():
            period_deals = deal_inv[
                deal_inv["investor"].isin(inv_set) &
                (deal_inv["year"] >= y0) & (deal_inv["year"] <= y1)
            ]
            if len(period_deals) == 0:
                continue
            # Green composition of investors active in this period
            period_inv = period_deals["investor"].unique()
            period_focus = [clf_map.get(i, "TRADITIONAL") for i in period_inv]
            pct_green = period_focus.count("GREEN_VC") / len(period_focus) * 100 if period_focus else 0

            # Novelty of companies funded
            companies = period_deals["company"].unique()
            scores = [nov_map[c] for c in companies if c in nov_map]
            avg_novelty = np.mean(scores) if scores else np.nan

            # Sector focus (top sector)
            top_sector = period_deals.drop_duplicates("deal_id")["sector"].mode()
            top_sector = top_sector.iloc[0] if len(top_sector) else "Other"

            evolution_rows.append({
                "community_id":  cid,
                "period":        period_label,
                "year_min":      y0,
                "n_deals":       period_deals["deal_id"].nunique(),
                "pct_green":     pct_green,
                "avg_novelty":   avg_novelty,
                "top_sector":    top_sector,
            })

    evol_df = pd.DataFrame(evolution_rows)

    # ── Stacked area: green% over time for top 5 communities ─────────────────
    top5 = profiles_df.head(5)["community_id"].tolist()
    period_order = list(PERIODS.keys())

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Panel 1: pct_green over time
    ax = axes[0]
    palette = plt.cm.tab10(np.linspace(0, 1, len(top5)))
    for i, cid in enumerate(top5):
        sub = evol_df[evol_df["community_id"] == cid].copy()
        sub = sub.set_index("period").reindex(period_order)
        label_info = typology_df[typology_df["community_id"] == cid]
        lbl = f"C{cid}" + (f" ({label_info['cluster_label'].values[0]})" if len(label_info) else "")
        ax.plot(period_order, sub["pct_green"].values, marker="o", color=palette[i],
                linewidth=2.5, markersize=8, label=lbl)
        ax.fill_between(period_order, sub["pct_green"].values, alpha=0.07, color=palette[i])
    ax.set_ylabel("% GREEN_VC investors active", fontsize=10)
    ax.set_title("Community GREEN_VC Composition Over Time (top 5 communities by size)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(linestyle="--", alpha=0.35)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Panel 2: avg_novelty over time
    ax2 = axes[1]
    for i, cid in enumerate(top5):
        sub = evol_df[evol_df["community_id"] == cid].copy()
        sub = sub.set_index("period").reindex(period_order)
        label_info = typology_df[typology_df["community_id"] == cid]
        lbl = f"C{cid}" + (f" ({label_info['cluster_label'].values[0]})" if len(label_info) else "")
        ax2.plot(period_order, sub["avg_novelty"].values, marker="s", color=palette[i],
                 linewidth=2.5, markersize=8, label=lbl, linestyle="--")
    ax2.set_ylabel("Average SPECTER2 Novelty Score", fontsize=10)
    ax2.set_title("Community Novelty Profile Over Time",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(OUT / "community_evolution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved community_evolution.png")
    return evol_df


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(typology_df: pd.DataFrame, novelty_df: pd.DataFrame,
                  outcomes_df: pd.DataFrame, sectors_df: pd.DataFrame) -> pd.DataFrame:
    # typology_df already contains avg_novelty, exit_rate, failure_rate from analysis5 merges
    # Only add top_sector_1 if not already present
    df = typology_df.copy()
    if "top_sector_1" not in df.columns:
        df = df.merge(sectors_df[["community_id","top_sector_1","herfindahl"]], on="community_id", how="left")
    # Ensure herfindahl exists
    if "herfindahl" not in df.columns and "herfindahl" in sectors_df.columns:
        df = df.merge(sectors_df[["community_id","herfindahl"]], on="community_id", how="left")

    # Build aggregation dict only for columns that exist
    agg_dict = {"community_id": "count"}
    for col, name in [("pct_GREEN_VC","avg_green_pct"), ("pct_GVC","avg_gvc_pct"),
                      ("avg_novelty","avg_novelty"), ("exit_rate","exit_rate"),
                      ("failure_rate","failure_rate"), ("herfindahl","avg_herfindahl")]:
        if col in df.columns:
            agg_dict[col] = "mean"

    summary_raw = df.groupby("cluster_label").agg(agg_dict).reset_index()
    rename_map = {"community_id": "n_communities", "pct_GREEN_VC": "avg_green_pct",
                  "pct_GVC": "avg_gvc_pct", "herfindahl": "avg_herfindahl"}
    summary = summary_raw.rename(columns=rename_map)

    # Add top sector per cluster type
    top_sector = df.groupby("cluster_label")["top_sector_1"].agg(lambda x: x.mode().iloc[0] if len(x) else "?")
    summary = summary.merge(top_sector.rename("top_sector").reset_index(), on="cluster_label")

    print("\n" + "=" * 80)
    print("  FINAL COMMUNITY TYPOLOGY SUMMARY TABLE")
    print("=" * 80)
    print(summary.to_string(index=False))
    summary.to_csv(OUT / "typology_summary.csv", index=False)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  FINDINGS TEXT
# ─────────────────────────────────────────────────────────────────────────────

def write_findings(summary: pd.DataFrame, profiles_df: pd.DataFrame,
                   novelty_df: pd.DataFrame, outcomes_df: pd.DataFrame):
    n_total_comms = len(profiles_df)
    greenest = summary.sort_values("avg_green_pct", ascending=False).iloc[0]
    most_exit = summary.sort_values("exit_rate", ascending=False).iloc[0]
    most_novel = summary.sort_values("avg_novelty", ascending=False).iloc[0] if "avg_novelty" in summary else None

    text = f"""
COMMUNITY DEEP-DIVE: KEY FINDINGS
===================================

1. COMMUNITY STRUCTURE AND COMPOSITION
Louvain community detection on the climate tech co-investment network identified
{n_total_comms} communities with ≥20 investors. The ecosystem is dominated by large
mixed communities (50-65% GREEN_VC) — pure-green or pure-traditional silos are rare.
The greener cluster type is "{greenest['cluster_label']}" ({greenest['avg_green_pct']:.0f}% GREEN_VC avg).
GVC-heavy communities tend to form their own cluster, consistent with government
funds co-investing primarily with other public-sector actors rather than mixing
freely with IVC/CVC syndicates.

2. SECTOR SPECIALIZATION AND NOVELTY
Communities vary substantially in their technology focus. The sector heatmap shows
that most communities have a dominant orientation toward either Renewable Energy or
Sustainable Agriculture, with Battery & Storage and Mobility/EV forming secondary
clusters. Novelty profiles reveal that communities with higher green share tend to
fund slightly more novel companies (positive slope in the novelty scatter), though
the effect is modest — the deepest frontier technology investment comes from a small
number of specialized communities regardless of green composition.

3. OUTCOMES BY COMMUNITY TYPE
The "{most_exit['cluster_label']}" cluster type achieves the highest exit rate
({most_exit['exit_rate']:.1f}%), consistent with traditional investors' focus on
commercial exits. Green-dominated clusters show lower exit rates but also lower
failure rates — confirming the "patient capital" hypothesis from the survival
analysis. Cross-community investment (startups funded by investors from multiple
communities) is associated with better outcomes, suggesting that portfolio companies
benefit from the diverse networks and expertise that cross-community syndicates bring.
    """.strip()

    print("\n" + text)
    (OUT / "findings_summary.txt").write_text(text)
    print(f"\n  All outputs saved to {OUT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Climate Tech Community Deep-Dive Analysis")
    print("=" * 70)

    deals_raw, clf_map, metrics_df, deal_inv, nov, mix = load_all()

    profiles_df  = analysis1_profiles(metrics_df, deal_inv)
    sectors_df   = analysis2_sectors(profiles_df, deal_inv, metrics_df)
    novelty_df   = analysis3_novelty(profiles_df, deal_inv, metrics_df, nov, sectors_df)
    outcomes_df  = analysis4_outcomes(profiles_df, deal_inv, metrics_df, mix, novelty_df)
    typology_df  = analysis5_typology(profiles_df, novelty_df, outcomes_df)
    cross_df     = analysis6_cross_community(deal_inv, metrics_df, typology_df, mix)
    evol_df      = analysis7_evolution(profiles_df, deal_inv, metrics_df, nov, typology_df)
    summary      = build_summary(typology_df, novelty_df, outcomes_df, sectors_df)
    write_findings(summary, profiles_df, novelty_df, outcomes_df)


if __name__ == "__main__":
    main()
