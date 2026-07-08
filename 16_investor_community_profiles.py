"""
16 — Investor profiles + community naming.

For each of 24,356 investors, derive from deal data:
  stage_focus, sector_focus, geographic_focus, deal_count,
  avg_deal_size, novelty_preference, exit_rate, failure_rate,
  co_investor_diversity

Then aggregate per community and name each community based on
its dominant profile features.

Outputs:
  output/investor_profiles.csv      — one row per investor
  output/community_named.csv        — 46 communities with derived names
  output/community_named.png        — visual summary
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict, Counter
from load_data import load_deals

OUT  = Path(__file__).parent / "output"
OUTN = OUT / "network"
OUTB = OUT

# ── Sector keyword map (same as scripts 13/15) ────────────────────────────────
SECTOR_KW = {
    "Renewable Energy":   ["solar","wind","renewable","hydropower","geothermal","tidal","biomass energy","clean energy","photovoltaic"],
    "Battery & Storage":  ["battery","storage","energy storage","lithium","solid state","grid storage","electrolyte"],
    "Mobility & EV":      ["mobility","electric vehicle"," ev ","autonomous","transportation","automotive","charging","fleet"],
    "Carbon Capture":     ["carbon capture","ccs","direct air","carbon removal","sequestration","carbon credit"],
    "Sustainable Agri":   ["agriculture","agritech","agtech","food","farming","crop","livestock","aquaculture","vertical farm"],
    "Hydrogen":           ["hydrogen","fuel cell","electrolysis","green hydrogen"],
    "Circular Economy":   ["circular","recycling","recycle","waste","packaging","upcycl","compost","refurbish"],
    "Energy Efficiency":  ["energy efficiency","smart grid","building","insulation","hvac","retrofit","smart meter"],
    "Water":              ["water","desalination","irrigation","wastewater"],
}

def classify_sector(row) -> str:
    text = " ".join([
        str(row.get("Verticals","") or ""),
        str(row.get("Primary PitchBook Industry Group","") or ""),
    ]).lower()
    for sector, kws in SECTOR_KW.items():
        if any(kw in text for kw in kws):
            return sector
    return "Other / Diversified"

def stage_bucket(vc_round: str) -> str:
    r = str(vc_round or "").strip()
    if r in ("1st Round","Angel","Seed"): return "Seed/Angel"
    if r in ("2nd Round","3rd Round"):    return "Early"
    if r in ("4th Round","5th Round","6th Round"): return "Growth"
    if r.endswith("Round") and r[0].isdigit() and int(r[0]) >= 7: return "Late"
    return "Other"


def build_investor_profiles() -> pd.DataFrame:
    print("Loading raw deals...")
    deals = load_deals()
    deals["Year"] = pd.to_datetime(deals["Deal Date"], errors="coerce").dt.year
    deals["sector"] = deals.apply(classify_sector, axis=1)
    deals["stage"]  = deals["VC Round"].apply(stage_bucket)

    clf = pd.read_csv(OUTB / "investors_classified_output.csv")
    clf = clf[~clf["investor_type"].isin(["PARSE_ERROR"])]
    clf_map = clf.set_index("investor_name")[["investor_type","green_focus"]].to_dict("index")

    metrics = pd.read_csv(OUTN / "network_metrics.csv")
    inv_comm = metrics.set_index("investor")["community"].to_dict()

    # Company outcomes
    mix = pd.read_csv(OUTB / "company_investor_mix.csv")
    mix["exited"] = mix["outcome"].isin(["Acquired","IPO / Public"]).astype(int)
    mix["failed"] = (mix["outcome"] == "Failed").astype(int)
    company_exit = mix.set_index("Companies")["exited"].to_dict()
    company_fail = mix.set_index("Companies")["failed"].to_dict()

    # Company novelty
    nov = pd.read_csv(OUTB / "novelty" / "novelty_scores.csv")[["name","specter_novelty"]]
    company_nov = nov.set_index("name")["specter_novelty"].to_dict()

    def clean(raw): return re.sub(r"\([^)]*\)", "", str(raw)).strip()

    print("Building investor → deal table (may take ~30s)...")
    # For each deal, parse investors → list of (investor, deal_id, company, size, stage, sector, region, year)
    inv_deals: dict[str, list] = defaultdict(list)
    deal_investor_list: dict[str, list] = defaultdict(list)

    for _, row in deals.iterrows():
        if pd.isna(row.get("Investors")): continue
        names = [clean(n) for n in re.split(r",\s*", str(row["Investors"]).replace("\n",", "))]
        names = [n for n in names if n and n in clf_map]
        deal_id = row["Deal ID"]
        company = row["Companies"]
        size    = row.get("Deal Size (USD M)", np.nan)
        stage   = row["stage"]
        sector  = row["sector"]
        region  = str(row.get("HQ Global Region","") or "Unknown")
        year    = row["Year"]
        for nm in names:
            inv_deals[nm].append({
                "deal_id": deal_id, "company": company,
                "size": size, "stage": stage, "sector": sector,
                "region": region, "year": year,
            })
            deal_investor_list[deal_id].append(nm)

    print(f"  {len(inv_deals):,} investors with ≥1 deal")

    # Per-investor aggregation
    print("Computing investor profiles...")
    rows = []
    for inv, deal_list in inv_deals.items():
        df_i = pd.DataFrame(deal_list)
        companies = df_i["company"].unique()

        # Stage focus
        stage_dist = df_i["stage"].value_counts()
        stage_focus = stage_dist.index[0] if len(stage_dist) else "Other"

        # Sector focus
        sec_dist = df_i["sector"].value_counts()
        sector_focus = sec_dist.index[0] if len(sec_dist) else "Other / Diversified"

        # Geographic focus
        reg_dist = df_i["region"].value_counts()
        geo_focus = reg_dist.index[0] if len(reg_dist) else "Unknown"

        # Deal metrics
        deal_count   = df_i["deal_id"].nunique()
        avg_deal_size = df_i["size"].mean()

        # Novelty preference (avg SPECTER2 across portfolio companies with scores)
        nov_scores = [company_nov[c] for c in companies if c in company_nov]
        novelty_pref = np.mean(nov_scores) if nov_scores else np.nan

        # Outcomes
        exits  = [company_exit.get(c, 0) for c in companies]
        fails  = [company_fail.get(c, 0) for c in companies]
        exit_rate    = np.mean(exits) if exits else 0.0
        failure_rate = np.mean(fails) if fails else 0.0

        # Co-investor diversity
        co_investors = set()
        for did in df_i["deal_id"].unique():
            for other in deal_investor_list.get(did, []):
                if other != inv:
                    co_investors.add(other)
        co_diversity = len(co_investors)

        info = clf_map.get(inv, {})
        rows.append({
            "investor":          inv,
            "investor_type":     info.get("investor_type","?"),
            "green_focus":       info.get("green_focus","?"),
            "community":         inv_comm.get(inv, np.nan),
            "deal_count":        deal_count,
            "avg_deal_size_M":   avg_deal_size,
            "stage_focus":       stage_focus,
            "sector_focus":      sector_focus,
            "geo_focus":         geo_focus,
            "novelty_pref":      novelty_pref,
            "exit_rate":         exit_rate,
            "failure_rate":      failure_rate,
            "co_investor_diversity": co_diversity,
        })

    profiles = pd.DataFrame(rows)
    # Add network centrality metrics
    profiles = profiles.merge(
        pd.read_csv(OUTN / "network_metrics.csv")[["investor","degree","betweenness","eigenvector"]],
        on="investor", how="left"
    )
    profiles.to_csv(OUT / "investor_profiles.csv", index=False)
    print(f"  Saved investor_profiles.csv ({len(profiles):,} rows)")
    return profiles


def name_community(row: pd.Series) -> str:
    """Generate a descriptive name from community profile features."""
    geo    = row.get("top_geo", "")
    stage  = row.get("top_stage", "")
    sector = row.get("top_sector", "")
    green  = row.get("pct_GREEN_VC", 50)
    gvc    = row.get("pct_GVC", 0)
    ivc    = row.get("pct_IVC", 0)
    exit_r = row.get("exit_rate_pct", 5)
    fail_r = row.get("failure_rate_pct", 7)
    nov    = row.get("avg_novelty", 0.42)
    n_inv  = row.get("n_investors", 100)

    # Geography prefix
    geo_tag = {
        "Americas": "US",
        "Europe":   "EU",
        "Asia":     "Asia",
    }.get(geo, "Global")

    # Stage tag
    stage_map = {
        "Seed/Angel": "Seed",
        "Early":      "Early-Stage",
        "Growth":     "Growth",
        "Late":       "Late-Stage",
        "Other":      "Multi-Stage",
    }
    stage_tag = stage_map.get(stage, "Multi-Stage")

    # Sector tag (shorten)
    sector_short = {
        "Renewable Energy":   "Renewables",
        "Battery & Storage":  "Storage",
        "Mobility & EV":      "Mobility",
        "Carbon Capture":     "Carbon",
        "Sustainable Agri":   "AgriTech",
        "Hydrogen":           "Hydrogen",
        "Circular Economy":   "Circular",
        "Energy Efficiency":  "Efficiency",
        "Water":              "Water",
        "Other / Diversified":"Diversified",
    }.get(sector, "Diversified")

    # Outcome modifier
    if exit_r >= 8:
        outcome_tag = "Exit-Active"
    elif fail_r >= 12:
        outcome_tag = "High-Risk"
    elif fail_r <= 4:
        outcome_tag = "Resilient"
    else:
        outcome_tag = None

    # Investor character modifier
    if gvc >= 55:
        char_tag = "Gov-Led"
    elif green >= 70 and nov >= 0.44:
        char_tag = "Green Frontier"
    elif green >= 70:
        char_tag = "Deep Green"
    elif green <= 35:
        char_tag = "Traditional"
    elif ivc >= 35:
        char_tag = "VC-Driven"
    else:
        char_tag = "Mixed"

    # Compose name: Character · Stage · Sector (Geo)
    parts = [char_tag, stage_tag, sector_short]
    if outcome_tag:
        parts.append(f"[{outcome_tag}]")
    name = f"{' · '.join(parts)} ({geo_tag})"
    return name


def build_community_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    print("\nAggregating per community...")
    typology = pd.read_csv(OUT / "community" / "community_typology.csv")

    rows = []
    for cid in sorted(profiles["community"].dropna().unique()):
        sub = profiles[profiles["community"] == cid]
        if len(sub) < 5: continue

        # Modal values
        top_stage  = sub["stage_focus"].value_counts().index[0] if len(sub) else "Other"
        top_sector = sub["sector_focus"].value_counts().index[0] if len(sub) else "Other / Diversified"
        top_geo    = sub["geo_focus"].value_counts().index[0] if len(sub) else "Unknown"

        # Distributions (%)
        stage_dist  = sub["stage_focus"].value_counts(normalize=True).mul(100).round(1)
        sector_dist = sub["sector_focus"].value_counts(normalize=True).mul(100).round(1)
        geo_dist    = sub["geo_focus"].value_counts(normalize=True).mul(100).round(1)

        # Averages
        avg_deal_size  = sub["avg_deal_size_M"].mean()
        avg_novelty    = sub["novelty_pref"].mean()
        exit_rate_pct  = sub["exit_rate"].mean() * 100
        fail_rate_pct  = sub["failure_rate"].mean() * 100
        avg_co_div     = sub["co_investor_diversity"].mean()
        avg_deal_count = sub["deal_count"].mean()

        # Green / GVC from typology
        typ_row = typology[typology["community_id"] == int(cid)]
        pct_green = typ_row["pct_GREEN_VC"].values[0] if len(typ_row) else 50
        pct_gvc   = typ_row["pct_GVC"].values[0] if len(typ_row) else 0
        pct_ivc   = typ_row["pct_IVC"].values[0] if len(typ_row) else 0
        cluster   = typ_row["cluster_label"].values[0] if len(typ_row) else "Unknown"

        row_data = {
            "community_id":     int(cid),
            "n_investors":      len(sub),
            "cluster_label":    cluster,
            "top_stage":        top_stage,
            "top_sector":       top_sector,
            "top_geo":          top_geo,
            "pct_GREEN_VC":     pct_green,
            "pct_GVC":          pct_gvc,
            "pct_IVC":          pct_ivc,
            "avg_deal_size_M":  avg_deal_size,
            "avg_novelty":      avg_novelty,
            "exit_rate_pct":    exit_rate_pct,
            "failure_rate_pct": fail_rate_pct,
            "avg_co_diversity": avg_co_div,
            "avg_deals_per_inv":avg_deal_count,
        }
        # Stage %
        for s in ["Seed/Angel","Early","Growth","Late","Other"]:
            row_data[f"pct_stage_{s.replace('/','_')}"] = stage_dist.get(s, 0)
        # Sector %
        for s, short in [("Renewable Energy","Renew"),("Battery & Storage","Battery"),
                          ("Mobility & EV","Mobility"),("Carbon Capture","Carbon"),
                          ("Sustainable Agri","Agri"),("Hydrogen","H2"),
                          ("Circular Economy","Circular"),("Energy Efficiency","Effic"),
                          ("Water","Water"),("Other / Diversified","Other")]:
            row_data[f"pct_sec_{short}"] = sector_dist.get(s, 0)
        # Top geo %
        for g in ["Americas","Europe","Asia","Oceania","Middle East","Africa"]:
            row_data[f"pct_geo_{g.replace(' ','_')}"] = geo_dist.get(g, 0)

        rows.append(row_data)

    comm_df = pd.DataFrame(rows)

    # Generate names
    comm_df["community_name"] = comm_df.apply(name_community, axis=1)

    # Save
    comm_df.to_csv(OUT / "community_named.csv", index=False)

    # Print summary
    display_cols = ["community_id","n_investors","cluster_label","community_name",
                    "top_stage","top_sector","top_geo","pct_GREEN_VC",
                    "exit_rate_pct","failure_rate_pct"]
    print(comm_df[display_cols].sort_values("n_investors",ascending=False).to_string(index=False))
    return comm_df


def plot_community_named(comm_df: pd.DataFrame):
    print("\nGenerating visualization...")

    CLUSTER_COLORS = {
        "Exit-Oriented":     "#2980b9",
        "Mixed Bridge":      "#27ae60",
        "High-Risk Frontier":"#e67e22",
        "Government-Led":    "#8e44ad",
        "Deep Green":        "#16a085",
        "Unknown":           "#95a5a6",
    }

    top = comm_df.nlargest(30, "n_investors")
    n   = len(top)
    fig, ax = plt.subplots(figsize=(18, max(10, n * 0.45)))
    fig.suptitle("Climate Tech Co-Investment Communities — Named Profiles\n"
                 "(sorted by community size; bar colour = cluster type)",
                 fontsize=13, fontweight="bold")

    y = np.arange(n)[::-1]
    colors = [CLUSTER_COLORS.get(r["cluster_label"],"#95a5a6") for _, r in top.iterrows()]
    bars = ax.barh(y, top["n_investors"], color=colors, alpha=0.85, edgecolor="white", height=0.7)

    # Right-side labels: community name + key stats
    for i, (_, r) in enumerate(top.iterrows()):
        yi = y[i]
        label = (f"  C{int(r['community_id'])}: {r['community_name']}"
                 f"  |  exit {r['exit_rate_pct']:.1f}%  fail {r['failure_rate_pct']:.1f}%"
                 f"  |  green {r['pct_GREEN_VC']:.0f}%")
        ax.text(r["n_investors"] + 5, yi, label, va="center", fontsize=7.5, color="#222")

    ax.set_yticks(y)
    ax.set_yticklabels([f"C{int(r['community_id'])}" for _, r in top.iterrows()], fontsize=9)
    ax.set_xlabel("Number of Investors in Community", fontsize=10)
    ax.set_xlim(0, top["n_investors"].max() * 2.8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, c in CLUSTER_COLORS.items() if l != "Unknown"]
    ax.legend(handles=handles, loc="lower right", fontsize=9, title="Cluster type")

    plt.tight_layout()
    path = OUT / "community_named.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def print_community_cards(comm_df: pd.DataFrame):
    """Print detailed profile cards for the top 15 communities by size."""
    print("\n" + "="*70)
    print("  COMMUNITY PROFILE CARDS (top 15 by size)")
    print("="*70)

    stage_cols  = [c for c in comm_df.columns if c.startswith("pct_stage_")]
    sector_cols = [c for c in comm_df.columns if c.startswith("pct_sec_")]
    geo_cols    = [c for c in comm_df.columns if c.startswith("pct_geo_")]

    for _, r in comm_df.nlargest(15,"n_investors").iterrows():
        print(f"\n  ── C{int(r['community_id'])}: {r['community_name']} ──")
        print(f"     Cluster: {r['cluster_label']}  |  {int(r['n_investors'])} investors")
        print(f"     Green: {r['pct_GREEN_VC']:.0f}%  GVC: {r['pct_GVC']:.0f}%  IVC: {r['pct_IVC']:.0f}%")
        print(f"     Outcomes: exit={r['exit_rate_pct']:.1f}%  fail={r['failure_rate_pct']:.1f}%")
        print(f"     Avg deal size: ${r['avg_deal_size_M']:.1f}M  "
              f"Avg novelty: {r['avg_novelty']:.3f}  "
              f"Avg co-investor diversity: {r['avg_co_diversity']:.0f}")

        # Stage breakdown (top 3)
        stage_vals = [(c.replace("pct_stage_",""), r[c]) for c in stage_cols]
        stage_top  = sorted(stage_vals, key=lambda x: x[1], reverse=True)[:3]
        print("     Stage: " + "  ".join([f"{s}={v:.0f}%" for s,v in stage_top if v > 0]))

        # Sector breakdown (top 3)
        sec_vals = [(c.replace("pct_sec_",""), r[c]) for c in sector_cols]
        sec_top  = sorted(sec_vals, key=lambda x: x[1], reverse=True)[:3]
        print("     Sector: " + "  ".join([f"{s}={v:.0f}%" for s,v in sec_top if v > 0]))

        # Geo breakdown (top 3)
        geo_vals = [(c.replace("pct_geo_",""), r[c]) for c in geo_cols]
        geo_top  = sorted(geo_vals, key=lambda x: x[1], reverse=True)[:3]
        print("     Geography: " + "  ".join([f"{g}={v:.0f}%" for g,v in geo_top if v > 0]))


def main():
    print("="*65)
    print("  Investor Profiles + Community Naming")
    print("="*65)

    profiles = build_investor_profiles()

    print("\nProfile summary:")
    print(f"  Stage focus distribution:\n{profiles['stage_focus'].value_counts().to_string()}")
    print(f"\n  Sector focus distribution:\n{profiles['sector_focus'].value_counts().to_string()}")
    print(f"\n  Geographic focus:\n{profiles['geo_focus'].value_counts().head(6).to_string()}")
    print(f"\n  Avg deal count per investor: {profiles['deal_count'].mean():.1f}")
    print(f"  Avg co-investor diversity:   {profiles['co_investor_diversity'].mean():.1f}")
    print(f"  Avg novelty preference:      {profiles['novelty_pref'].mean():.3f}")

    comm_df = build_community_profiles(profiles)
    print_community_cards(comm_df)
    plot_community_named(comm_df)

    print(f"\n  investor_profiles.csv   : {len(profiles):,} rows")
    print(f"  community_named.csv     : {len(comm_df)} communities")
    print(f"  community_named.png     : saved")


if __name__ == "__main__":
    main()
