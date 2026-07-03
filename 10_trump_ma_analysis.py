"""
10 — Trump 1 M&A surge deep-dive: sector, acquirer type, geography, stage.

Analyses:
  1. Which sectors had the largest M&A increase during Trump 1?
  2. Who acquired climate tech companies — and did that change?
  3. Geographic flow: did foreign buyers increase during Trump 1?
  4. Stage at acquisition: early-stage fire-sales vs late-stage strategic?

All outputs saved to output/trump_ma_*.png and output/trump_ma_tables.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import re
from pathlib import Path
from load_data import load_deals, load_companies

OUT  = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

TRUMP1_START = pd.Timestamp("2016-11-08")
TRUMP1_END   = pd.Timestamp("2021-01-19")
PRE_START    = pd.Timestamp("2012-01-01")
PRE_END      = pd.Timestamp("2016-11-07")

# ── Sector keyword mapping ──────────────────────────────────────────────────
SECTOR_RULES = [
    ("Battery / Storage",        ["battery","storage","energy storage","electrolyzer","fuel cell"]),
    ("Solar & Wind",             ["solar","wind","photovoltaic","pv energy","offshore wind"]),
    ("Mobility / EV",            ["mobility","electric vehicle"," ev ","evs","autonomous","transportation","automotive"]),
    ("Carbon Capture / Removal", ["carbon capture","carbon removal","ccs","ccus","carbon sequestration","carbon credit","net zero","carbon offset"]),
    ("Sustainable Agriculture",  ["agri","agtech","food tech","precision farming","sustainable food","alternative protein","aquaculture"]),
    ("Water Tech",               ["water","wastewater","desalination","water treatment"]),
    ("Circular Economy",         ["circular","recycling","waste","upcycling","plastic","biomass"]),
    ("Energy Efficiency",        ["efficiency","smart building","building tech","insulation","hvac","led","demand response"]),
    ("Other Renewables",         ["geothermal","hydro","tidal","wave energy","bioenergy","biomethane"]),
]


def classify_sector(verticals: str) -> str:
    if pd.isna(verticals):
        return "Other Climate Tech"
    v = verticals.lower()
    for sector, keywords in SECTOR_RULES:
        if any(k in v for k in keywords):
            return sector
    return "Other Climate Tech"


# ── Acquirer type mapping ───────────────────────────────────────────────────
ENERGY_MAJORS = {
    "shell","bp","total","equinor","exxon","chevron","conocophillips","eni","repsol",
    "petrobras","aramco","cnooc","engie","rwe","e.on","iberdrola","duke energy",
    "nextera","edf","vattenfall","ørsted","enel","orsted","nrg","exelon","dominion",
    "constellation","sempra","southern company","centrica","national grid","avangrid",
    "innogy","drax","fortum","verbund","eon","sse","endesa","acciona"
}
TECH_GIANTS = {
    "google","alphabet","microsoft","amazon","apple","meta","facebook","ibm",
    "salesforce","oracle","sap","intel","cisco","qualcomm","nvidia","tesla",
    "twitter","uber","lyft","airbnb","spotify","palantir","snowflake","databricks"
}
INDUSTRIALS = {
    "siemens","ge ","general electric","honeywell","abb","schneider","bosch","mitsubishi",
    "hitachi","danfoss","johnson controls","carrier","emerson","rockwell","3m","caterpillar",
    "thyssenkrupp","voith","fortive","danaher","roper","dover","xylem","veolia","suez",
    "atlas copco","grundfos","sulzer","alfa laval"
}
FINANCIAL = {
    "capital","partners","private equity","fund","investment","asset management",
    "holdings","ventures","equity","acquisition","spac","merger","group",
    "blackstone","kkr","carlyle","tpg","apollo","warburg","advent","permira",
    "bain capital","ares","brookfield"
}


def classify_acquirer(name: str) -> str:
    if pd.isna(name) or str(name).strip() == "":
        return "Unknown / Undisclosed"
    n = name.lower()
    if any(k in n for k in ENERGY_MAJORS):
        return "Energy Major"
    if any(k in n for k in TECH_GIANTS):
        return "Tech Giant"
    if any(k in n for k in INDUSTRIALS):
        return "Industrial / Manufacturing"
    if any(k in n for k in FINANCIAL):
        return "Financial Buyer (PE/SPAC)"
    return "Other / Climate Tech"


# ── Load & prepare data ──────────────────────────────────────────────────────
def load_data():
    deals = load_deals()
    deals = deals.dropna(subset=["Year","Deal Date"]).query("2012 <= Year <= 2026")

    companies = load_companies()
    companies["is_acquired"] = companies["Ownership Status"].str.contains("Acquired", na=False).astype(int)
    companies["sector"]      = companies["Verticals"].apply(classify_sector)
    companies["acquirer_type"] = companies["Acquirers"].apply(
        lambda x: classify_acquirer(x.split(",")[0].strip()) if pd.notna(x) else "Unknown / Undisclosed"
    )

    def comp_outcome(row):
        own = str(row.get("Ownership Status",""))
        if "Publicly Held" in own or "IPO" in own: return "ipo"
        if "Acquired" in own:                       return "acquired"
        if "Out of Business" in own:                return "failed"
        return "operating"
    companies["outcome"] = companies.apply(comp_outcome, axis=1)

    comp_map = companies.drop_duplicates("Companies").set_index("Companies")[
        ["outcome","sector","acquirer_type","Acquirers","HQ Global Region"]
    ].to_dict("index")

    deals["outcome"]       = deals["Companies"].map(lambda c: comp_map.get(c,{}).get("outcome","operating"))
    deals["sector"]        = deals["Companies"].map(lambda c: comp_map.get(c,{}).get("sector","Other Climate Tech"))
    deals["acquirer_type"] = deals["Companies"].map(lambda c: comp_map.get(c,{}).get("acquirer_type","Unknown / Undisclosed"))
    deals["acquirer_raw"]  = deals["Companies"].map(lambda c: comp_map.get(c,{}).get("Acquirers",""))
    deals["is_ma"]         = (deals["outcome"] == "acquired").astype(int)

    deals["era"] = "Other"
    deals.loc[(deals["Deal Date"] >= PRE_START)    & (deals["Deal Date"] <= PRE_END),    "era"] = "Pre-Trump"
    deals.loc[(deals["Deal Date"] >= TRUMP1_START) & (deals["Deal Date"] <= TRUMP1_END), "era"] = "Trump 1"

    # Stage classification from VC Round
    def classify_stage(vc_round):
        if pd.isna(vc_round): return "Unknown"
        r = str(vc_round)
        if r in ("Angel","1st Round"): return "Seed / Angel"
        if r in ("2nd Round","3rd Round"): return "Early (A/B)"
        if r in ("4th Round","5th Round"): return "Late (C/D)"
        if r in ("6th Round","7th Round","8th Round"): return "Growth (E+)"
        return "Other"
    deals["stage"] = deals["VC Round"].apply(classify_stage)

    return deals[deals["era"].isin(["Pre-Trump","Trump 1"])], companies


def era_ma_rate(df, groupby_col):
    """Return table: col | Pre-Trump M&A% | Trump-1 M&A% | Delta | N (Trump1)."""
    rows = []
    for val in df[groupby_col].unique():
        sub = df[df[groupby_col] == val]
        pre    = sub[sub["era"]=="Pre-Trump"]["is_ma"]
        trump1 = sub[sub["era"]=="Trump 1"]["is_ma"]
        if len(pre) < 10 or len(trump1) < 10:
            continue
        rows.append({
            groupby_col:           val,
            "Pre-Trump M&A%":      pre.mean()*100,
            "Trump 1 M&A%":        trump1.mean()*100,
            "Delta (pp)":          trump1.mean()*100 - pre.mean()*100,
            "N (Pre-Trump)":       len(pre),
            "N (Trump 1)":         len(trump1),
        })
    return pd.DataFrame(rows).sort_values("Delta (pp)", ascending=False)


def main():
    df, companies = load_data()
    print(f"Deals in analysis (Pre-Trump + Trump 1): {len(df):,}")
    print(f"  Pre-Trump: {(df['era']=='Pre-Trump').sum():,} | Trump 1: {(df['era']=='Trump 1').sum():,}")
    print(f"  M&A rate  — Pre: {df[df['era']=='Pre-Trump']['is_ma'].mean()*100:.1f}%  "
          f"Trump 1: {df[df['era']=='Trump 1']['is_ma'].mean()*100:.1f}%")

    # ── Analysis 1: Sectors ────────────────────────────────────────────────
    print("\n=== ANALYSIS 1: SECTOR M&A RATES ===")
    t1 = era_ma_rate(df, "sector")
    print(t1.round(2).to_string(index=False))

    # ── Analysis 2: Acquirer types ─────────────────────────────────────────
    print("\n=== ANALYSIS 2: ACQUIRER TYPE ===")
    t2 = era_ma_rate(df[df["is_ma"]==1], "acquirer_type") if False else pd.DataFrame()
    # Better: just count acquirer types per era
    acq_counts = df[df["is_ma"]==1].groupby(["era","acquirer_type"]).size().unstack(fill_value=0)
    acq_pct    = acq_counts.div(acq_counts.sum(axis=1), axis=0)*100
    print(acq_pct.round(1).to_string())

    # ── Analysis 3: Geography ─────────────────────────────────────────────
    print("\n=== ANALYSIS 3: GEOGRAPHIC DISTRIBUTION ===")
    geo = df[df["is_ma"]==1].groupby(["era","HQ Global Region"]).size().unstack(fill_value=0)
    geo_pct = geo.div(geo.sum(axis=1), axis=0)*100
    print(geo_pct.round(1).to_string())

    # ── Analysis 4: Stage ─────────────────────────────────────────────────
    print("\n=== ANALYSIS 4: STAGE AT ACQUISITION ===")
    t4 = era_ma_rate(df, "stage")
    print(t4.round(2).to_string(index=False))

    # Save tables
    with pd.ExcelWriter(OUT / "trump_ma_tables.xlsx") as w:
        t1.round(2).to_excel(w, sheet_name="Sectors", index=False)
        acq_pct.round(1).to_excel(w, sheet_name="Acquirer Types")
        geo_pct.round(1).to_excel(w, sheet_name="Geography")
        t4.round(2).to_excel(w, sheet_name="Stages", index=False)
    print(f"\nSaved → {OUT / 'trump_ma_tables.xlsx'}")

    # ── Charts ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Trump 1 M&A Surge Deep-Dive (2016–2021)\nClimate Tech Acquisitions: Sector, Acquirer, Geography, Stage",
                 fontsize=13, fontweight="bold", y=1.01)

    # 1. Sector chart
    ax = axes[0,0]
    t1_plot = t1[t1["N (Trump 1)"] >= 20].head(10)
    colors = ["#e74c3c" if d > 0 else "#3498db" for d in t1_plot["Delta (pp)"]]
    bars = ax.barh(t1_plot["sector"], t1_plot["Delta (pp)"], color=colors, edgecolor="white", alpha=0.9)
    ax.bar_label(bars, labels=[f"{d:+.1f}pp" for d in t1_plot["Delta (pp)"]], padding=4, fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Change in M&A rate (pp)\nTrump 1 vs Pre-Trump")
    ax.set_title("1. Sector M&A Increase During Trump 1\n(red = increase)", fontsize=10, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # 2. Acquirer type chart
    ax = axes[0,1]
    acq_order = acq_pct.columns.tolist()
    x = np.arange(len(acq_order))
    w = 0.35
    era_colors = {"Pre-Trump":"#95a5a6","Trump 1":"#e74c3c"}
    for i, era in enumerate(["Pre-Trump","Trump 1"]):
        if era not in acq_pct.index: continue
        vals = [acq_pct.loc[era, c] if c in acq_pct.columns else 0 for c in acq_order]
        bars = ax.bar(x + (i-0.5)*w, vals, w, label=era,
                     color=era_colors[era], edgecolor="white", alpha=0.9)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=2, fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(" / ","\n").replace(" (", "\n(") for a in acq_order],
                       fontsize=8)
    ax.set_ylabel("Share of M&A exits (%)")
    ax.set_title("2. Acquirer Type Mix\nPre-Trump vs Trump 1", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    # 3. Geography chart
    ax = axes[1,0]
    geo_plot = geo_pct.T.reindex(index=[c for c in geo_pct.columns if geo_pct[c].sum()>0])
    geo_plot_filtered = geo_plot[geo_plot.sum(axis=1)>0]
    x = np.arange(len(geo_plot_filtered))
    for i, era in enumerate(["Pre-Trump","Trump 1"]):
        if era not in geo_plot_filtered.columns: continue
        vals = geo_plot_filtered[era].values
        bars = ax.bar(x + (i-0.5)*w, vals, w, label=era,
                     color=era_colors[era], edgecolor="white", alpha=0.9)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(geo_plot_filtered.index, fontsize=9)
    ax.set_ylabel("Share of M&A exits (%)")
    ax.set_title("3. Geography of Acquired Startups\nPre-Trump vs Trump 1", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    # 4. Stage chart
    ax = axes[1,1]
    stage_order = ["Seed / Angel","Early (A/B)","Late (C/D)","Growth (E+)","Other","Unknown"]
    t4_plot = t4[t4["stage"].isin(stage_order)].set_index("stage").reindex(
        [s for s in stage_order if s in t4.set_index("stage").index]
    ).reset_index()
    x = np.arange(len(t4_plot))
    w2 = 0.35
    b1 = ax.bar(x-w2/2, t4_plot["Pre-Trump M&A%"], w2, label="Pre-Trump",
               color="#95a5a6", edgecolor="white", alpha=0.9)
    b2 = ax.bar(x+w2/2, t4_plot["Trump 1 M&A%"], w2, label="Trump 1",
               color="#e74c3c", edgecolor="white", alpha=0.9)
    ax.bar_label(b1, labels=[f"{v:.1f}%" for v in t4_plot["Pre-Trump M&A%"]], padding=2, fontsize=8)
    ax.bar_label(b2, labels=[f"{v:.1f}%" for v in t4_plot["Trump 1 M&A%"]], padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(t4_plot["stage"], fontsize=9)
    ax.set_ylabel("M&A exit rate (%)")
    ax.set_title("4. Stage at Acquisition\nPre-Trump vs Trump 1", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    plt.tight_layout()
    fig.savefig(OUT / "trump_ma_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT / 'trump_ma_analysis.png'}")


if __name__ == "__main__":
    main()
