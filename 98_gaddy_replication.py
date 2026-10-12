"""
Script 98: Replicate and extend Gaddy et al. Figure 8.
Descriptive only — no regressions.

Gaddy et al. (2017) "Venture Capital and Cleantech: The Wrong Model for
Clean Energy Innovation." MIT Energy Initiative. Section 3.1 categories:
  MCP  — Materials / Chemicals / Processes (novel science, high tech risk)
  HW   — Hardware Integration (system integration, moderate risk)
  SW   — Software / Software Appliance (low capital, monitoring, platforms)
  DF   — Deployment / Finance (financial engineering, project deployment)
  Other — does not fit above
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

OUT_DIR = "output/gaddy_replication"
os.makedirs(OUT_DIR, exist_ok=True)

def sep(title=""):
    print(f"\n{'─'*68}")
    if title:
        print(f"  {title}")
        print(f"{'─'*68}")

print("=" * 68)
print("SCRIPT 98 — GADDY REPLICATION + EXTENSION")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────
# 1. Cluster → Gaddy category mapping
# ─────────────────────────────────────────────────────────────────────
# Columns: cluster_id, gaddy_cat, confidence (H/M/L), rationale
#   MCP = Materials/Chemicals/Processes
#   HW  = Hardware Integration
#   SW  = Software/Software Appliance
#   DF  = Deployment/Finance
#   Other = unclassifiable

GADDY_MAP = {
    #cl  keywords (abbreviated)                  cat     conf
     0: ("MCP", "H", "nuclear/fusion — fundamental nuclear R&D"),
     1: ("Other","L", "stealth / unknown descriptors"),
     2: ("MCP", "H", "green hydrogen, electrolysis chemistry"),
     3: ("HW",  "H", "EV charging infrastructure hardware"),
     4: ("HW",  "H", "electric vehicles, e-bikes, mobility hardware"),
     5: ("HW",  "M", "geothermal drilling systems"),
     6: ("Other","M","mining / metals — upstream extraction"),
     7: ("HW",  "M", "water/wastewater treatment systems"),
     8: ("MCP", "M", "insect protein, novel animal feed"),
     9: ("MCP", "H", "precision fermentation, synthetic biology"),
    10: ("Other","H","alt-protein food brands — brand/market, not tech"),
    11: ("MCP", "H", "cultivated meat — cell-culture bioprocess"),
    12: ("Other","M","plant-based seafood — mostly food brand"),
    13: ("MCP", "H", "CCUS / direct air capture chemistry"),
    14: ("HW",  "M", "power semiconductors, photonics hardware"),
    15: ("HW",  "H", "wave and tidal energy hardware"),
    16: ("HW",  "H", "electric aviation, eVTOL hardware"),
    17: ("HW",  "H", "offshore wind turbines and systems"),
    18: ("Other","M","aquaculture / seaweed farming"),
    19: ("Other","M","ocean/reef restoration — conservation"),
    20: ("HW",  "M", "underwater robotics and marine monitoring HW"),
    21: ("HW",  "H", "zero-emission shipping propulsion"),
    22: ("SW",  "M", "wildfire detection, remote sensing SW"),
    23: ("SW",  "H", "satellite/geospatial analytics platform"),
    24: ("HW",  "H", "vertical/indoor farming systems"),
    25: ("SW",  "H", "weather forecasting analytics"),
    26: ("DF",  "H", "climate risk, flood insurance, financial"),
    27: ("Other","M","forest/biodiversity credits — NbS"),
    28: ("Other","M","forest carbon, reforestation projects — NbS"),
    29: ("MCP", "M", "crop protection, novel fertilizers"),
    30: ("MCP", "M", "crop science, plant biology inputs"),
    31: ("SW",  "H", "methane monitoring, predictive maintenance SW"),
    32: ("HW",  "M", "air quality monitoring and purification HW"),
    33: ("SW",  "H", "carbon/ESG reporting and data platforms"),
    34: ("SW",  "M", "sustainability engagement, impact platforms"),
    35: ("DF",  "M", "blockchain carbon credits, verification"),
    36: ("DF",  "H", "carbon project finance, climate finance"),
    37: ("MCP", "H", "biomass-to-energy, biochar chemistry"),
    38: ("Other","M","food waste reduction — retail/logistics service"),
    39: ("Other","M","textile recycling — service/circular"),
    40: ("HW",  "H", "solar PV installation and systems"),
    41: ("Other","L","inventory/logistics — unclear cleantech link"),
    42: ("MCP", "H", "battery recycling, hydromet chemistry"),
    43: ("MCP", "H", "low-carbon cement/concrete chemistry"),
    44: ("MCP", "H", "bio-based leather and cellulose materials"),
    45: ("MCP", "H", "lignin/bio-fiber materials chemistry"),
    46: ("MCP", "H", "chemical/catalytic plastics recycling"),
    47: ("MCP", "H", "bio-based/compostable packaging materials"),
    48: ("Other","M","reusable packaging — service/logistics"),
    49: ("HW",  "H", "waste sorting and recycling hardware"),
    50: ("Other","M","waste collection services — logistics/ops"),
    51: ("MCP", "H", "low-carbon construction materials"),
    52: ("HW",  "M", "sustainable construction, green buildings"),
    53: ("DF",  "M", "generic green energy developer/utility"),
    54: ("HW",  "H", "thermal energy storage hardware"),
    55: ("MCP", "H", "next-gen battery chemistry and storage"),
    56: ("HW",  "H", "grid-scale long-duration storage systems"),
    57: ("HW",  "H", "grid-connected battery storage systems"),
    58: ("DF",  "H", "energy asset management and trading"),
    59: ("SW",  "H", "smart grid / distributed energy SW"),
    60: ("HW",  "H", "heat pumps — space/water heating HW"),
    61: ("SW",  "H", "energy management and consumption SW"),
    62: ("HW",  "H", "HVAC, efficient cooling hardware"),
}

CATS  = ["MCP","HW","SW","DF","Other"]
CLRS  = {"MCP":"#D55E00","HW":"#0072B2","SW":"#009E73","DF":"#CC79A7","Other":"#999999"}
LBLS  = {"MCP":"Materials/Chem/Process","HW":"Hardware Integration",
         "SW":"Software/SW Appliance","DF":"Deployment/Finance","Other":"Other"}

# Print classification table
sep("1. Cluster classification")
print(f"  {'Cl':>4}  {'N':>5}  {'Cat':<6}  {'Conf'}  Keywords (first 55 chars)")
print(f"  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*55}")

nov = pd.read_csv("output/novelty/novelty_scores.csv")
n_by_cl = nov[nov["specter_cluster"]>=0].groupby("specter_cluster").size()
kw_by_cl = (nov[nov["specter_cluster"]>=0]
            .groupby("specter_cluster")["specter_keywords"].first())
ex_by_cl = (nov[nov["specter_cluster"]>=0]
            .groupby("specter_cluster")["name"]
            .apply(lambda x: " | ".join(x.head(5).tolist())))

summary_rows = []
for cl in sorted(GADDY_MAP):
    cat, conf, rat = GADDY_MAP[cl]
    n   = n_by_cl.get(cl, 0)
    kw  = str(kw_by_cl.get(cl,""))[:55]
    ex  = str(ex_by_cl.get(cl,""))
    print(f"  {cl:>4}  {n:>5}  {cat:<6}  {conf:<4}  {kw}")
    summary_rows.append({"cluster":cl,"cat":cat,"conf":conf,"n_startups":n,
                          "keywords":kw_by_cl.get(cl,""),"examples":ex})

cls_df = pd.DataFrame(summary_rows)

# Category counts
print(f"\n  Category summary:")
for cat in CATS:
    nc = (cls_df["cat"]==cat).sum()
    ns = cls_df[cls_df["cat"]==cat]["n_startups"].sum()
    print(f"    {LBLS[cat]:<30}  clusters={nc:>3}  startups={ns:>4}")

cls_df.to_csv(f"{OUT_DIR}/gaddy_cluster_classification.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# Load deals data and attach Gaddy category
# ─────────────────────────────────────────────────────────────────────
sep("Loading deals and attaching categories")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl["year"]      = dl["Deal Date"].dt.year
dl["co"]        = dl["Companies"].str.strip()
dl["deal_size"] = pd.to_numeric(dl["Deal Size (USD M)"], errors="coerce")

deals_v = dl.dropna(subset=["year"]).copy()
deals_v["year"] = deals_v["year"].astype(int)

# co → specter_cluster
co_cluster = dict(zip(nov["name"], nov["specter_cluster"].astype(int)))
# cluster → Gaddy cat
cl_to_cat = {cl: v[0] for cl, v in GADDY_MAP.items()}

deals_v["cluster"] = deals_v["co"].map(co_cluster)
deals_v["gaddy"]   = deals_v["cluster"].map(cl_to_cat).fillna("Other")

# Region
def region_bucket(r):
    r = str(r)
    if r in ("Americas",):                       return "Americas"
    if r in ("Europe",):                         return "Europe"
    if r in ("Asia",):                           return "Asia"
    if r in ("Middle East & Africa","Africa"):   return "MEA"
    return "Other/Unknown"

deals_v["region"] = deals_v["HQ Global Region"].apply(region_bucket)

# Novelty
nat  = pd.read_csv("output/novelty_t/novelty_time_aware.csv")
co_os = dict(zip(nat["Companies"], nat["overall_sparsity_t"]))
deals_v["os"] = deals_v["co"].map(co_os)

print(f"  Total deals (with year): {len(deals_v):,}")
print(f"  Classified to a tech cluster:  {deals_v['cluster'].notna().sum():,}  "
      f"({100*deals_v['cluster'].notna().mean():.1f}%)")
print(f"  Gaddy distribution:")
print(deals_v["gaddy"].value_counts().to_string())

# First rounds: earliest deal date per company
first_round = (deals_v.sort_values("Deal Date")
               .groupby("co").first()
               .reset_index()
               .rename(columns={"Deal Date":"first_date",
                                 "year":"first_year"}))

# Filter 2012-2026
YEAR_MIN, YEAR_MAX = 2012, 2026

# ─────────────────────────────────────────────────────────────────────
# 2. Share of deals and dollars by year
# ─────────────────────────────────────────────────────────────────────
sep("2. Year-by-year share of deals and dollars")

def yearly_shares(df, year_col, size_col, label):
    df = df[(df[year_col] >= YEAR_MIN) & (df[year_col] <= YEAR_MAX)].copy()
    years = sorted(df[year_col].unique())

    # Deal shares
    deal_piv = (df.groupby([year_col, "gaddy"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=CATS, fill_value=0))
    deal_piv = deal_piv.loc[deal_piv.index.isin(years)]
    deal_sh  = deal_piv.div(deal_piv.sum(axis=1), axis=0)

    # Dollar shares
    dol_piv = (df.groupby([year_col, "gaddy"])[size_col]
                 .sum()
                 .unstack(fill_value=0)
                 .reindex(columns=CATS, fill_value=0))
    dol_piv = dol_piv.loc[dol_piv.index.isin(years)]
    dol_sh  = dol_piv.div(dol_piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    totals_n = deal_piv.sum(axis=1).rename("n_deals")
    totals_d = dol_piv.sum(axis=1).rename("usd_M")

    print(f"\n  {label} — deal count shares (%):")
    tbl = (deal_sh * 100).round(1)
    tbl["N"]   = totals_n.astype(int)
    print(tbl.to_string())

    print(f"\n  {label} — dollar shares (%):")
    tbl2 = (dol_sh * 100).round(1)
    tbl2["USD_M"] = totals_d.round(0).astype(int)
    print(tbl2.to_string())

    return deal_sh, dol_sh, totals_n, totals_d

# All rounds
deal_sh_all, dol_sh_all, n_all, usd_all = yearly_shares(
    deals_v, "year", "deal_size", "ALL ROUNDS")

# First rounds only
fr_slice = deals_v.sort_values("Deal Date").groupby("co").first().reset_index()
fr_slice["year"] = fr_slice["Deal Date"].dt.year
deal_sh_fr, dol_sh_fr, n_fr, usd_fr = yearly_shares(
    fr_slice, "year", "deal_size", "FIRST ROUNDS ONLY")

# ─────────────────────────────────────────────────────────────────────
# 3. Figures
# ─────────────────────────────────────────────────────────────────────
sep("3. Plots")

# Gaddy's approximate Figure 8 values (2004-2014, deal share %)
# Source: Gaddy et al. 2017, MIT EERI, Fig 8 — read from paper
# These are approximate digitised values for the five categories
GADDY_ORIG = {
    "year":  [2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014],
    "MCP":   [45,  45,  43,  42,  40,  36,  34,  32,  30,  28,  26 ],
    "HW":    [28,  27,  27,  28,  28,  26,  24,  22,  22,  21,  20 ],
    "SW":    [ 8,   9,  10,  11,  13,  14,  16,  18,  19,  22,  24 ],
    "DF":    [10,  10,  11,  11,  11,  14,  15,  16,  16,  16,  17 ],
    "Other": [ 9,   9,   9,   8,   8,  10,  11,  12,  13,  13,  13 ],
}
gaddy_df = pd.DataFrame(GADDY_ORIG).set_index("year")

def make_stacked_area(deal_sh, dol_sh, title_sfx, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Gaddy categories — {title_sfx}", fontsize=12, fontweight="bold")

    for ax, sh, metric in zip(axes, [deal_sh, dol_sh], ["Deal share","Dollar share"]):
        yrs = sh.index.tolist()
        bot = np.zeros(len(yrs))
        for cat in CATS:
            vals = sh[cat].values * 100
            ax.bar(yrs, vals, bottom=bot, color=CLRS[cat], label=LBLS[cat],
                   width=0.7, alpha=0.9)
            bot += vals

        # Gaddy overlay (deal share only, for overlap years 2012+)
        if metric == "Deal share":
            ov = gaddy_df.reindex([y for y in gaddy_df.index if y >= 2012])
            if len(ov) > 0:
                for cat in CATS:
                    ax.plot(ov.index, ov[cat].values, linestyle="--",
                            color=CLRS[cat], linewidth=1.4, alpha=0.6)
                ax.plot([], [], linestyle="--", color="grey",
                        linewidth=1.4, label="Gaddy (2017) est.")

        ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
        ax.set_xlabel("Year")
        ax.set_title(metric, fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")

make_stacked_area(deal_sh_all, dol_sh_all, "all rounds",    "fig_all_rounds.png")
make_stacked_area(deal_sh_fr,  dol_sh_fr,  "first rounds",  "fig_first_rounds.png")

# Gaddy vs our 2012-2014 comparison table
sep("  Gaddy (2017) vs our sample: overlap years 2012-2014")
ov_yrs = [y for y in deal_sh_all.index if 2012 <= y <= 2014]
if ov_yrs:
    our_avg   = (deal_sh_all.loc[ov_yrs] * 100).mean().round(1)
    gaddy_avg = gaddy_df.reindex(ov_yrs).mean().round(1)
    comp = pd.DataFrame({"Gaddy_2017 (%)": gaddy_avg,
                          "Our sample (%)": our_avg})
    print(f"\n  Average deal share 2012-2014:")
    print(comp.to_string())

# ─────────────────────────────────────────────────────────────────────
# 4. Regional split (deal share, all rounds)
# ─────────────────────────────────────────────────────────────────────
sep("4. Regional split (deal share, all rounds, 2012-2026)")

REGIONS = ["Americas","Europe","Asia"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("Gaddy categories by region — deal share, all rounds", fontsize=11, fontweight="bold")

for ax, region in zip(axes, REGIONS):
    sub = deals_v[(deals_v["region"]==region) &
                  (deals_v["year"]>=YEAR_MIN) &
                  (deals_v["year"]<=YEAR_MAX)]
    piv = (sub.groupby(["year","gaddy"])
              .size()
              .unstack(fill_value=0)
              .reindex(columns=CATS, fill_value=0))
    sh  = piv.div(piv.sum(axis=1), axis=0)
    yrs = sh.index.tolist()
    bot = np.zeros(len(yrs))
    for cat in CATS:
        vals = sh[cat].values * 100
        ax.bar(yrs, vals, bottom=bot, color=CLRS[cat], label=LBLS[cat],
               width=0.7, alpha=0.9)
        bot += vals

    ax.set_title(f"{region}  (N={len(sub):,} deals)", fontsize=10)
    ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    ax.set_xlabel("Year")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

axes[0].legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_regional.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig_regional.png")

# Print tables
print(f"\n  Deal share by category and region (2012-2026 average):")
print(f"  {'Category':<28}  {'Americas':>9}  {'Europe':>9}  {'Asia':>9}")
print(f"  {'─'*28}  {'─'*9}  {'─'*9}  {'─'*9}")
for cat in CATS:
    row = []
    for region in REGIONS:
        sub = deals_v[(deals_v["region"]==region) &
                      (deals_v["year"].between(YEAR_MIN,YEAR_MAX))]
        sh = (sub["gaddy"]==cat).mean() * 100
        row.append(f"{sh:>9.1f}%")
    print(f"  {LBLS[cat]:<28}  {'  '.join(row)}")

print(f"\n  N deals by region (2012-2026):")
for region in REGIONS:
    n = ((deals_v["region"]==region) & (deals_v["year"].between(YEAR_MIN,YEAR_MAX))).sum()
    print(f"    {region:<12}  {n:>5}")

# ─────────────────────────────────────────────────────────────────────
# 5. Mean novelty by year among newly-funded startups
# ─────────────────────────────────────────────────────────────────────
sep("5. Mean novelty (overall_sparsity_t) of newly-funded startups by year")

fr_nov = fr_slice[(fr_slice["year"].between(YEAR_MIN, YEAR_MAX)) &
                  (fr_slice["os"].notna())].copy()

yr_nov = (fr_nov.groupby("year")
          .agg(n=("os","count"),
               os_mean=("os","mean"),
               os_sd=("os","std"),
               os_p25=("os", lambda x: x.quantile(.25)),
               os_p75=("os", lambda x: x.quantile(.75)))
          .reset_index())

print(f"\n  {'Year'}  {'N':>5}  {'Mean OS':>9}  {'SD':>7}  {'P25':>7}  {'P75':>7}")
print(f"  {'─'*4}  {'─'*5}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}")
for _, r in yr_nov.iterrows():
    print(f"  {int(r.year)}  {int(r.n):>5}  "
          f"{r.os_mean:>+9.4f}  {r.os_sd:>7.4f}  "
          f"{r.os_p25:>+7.4f}  {r.os_p75:>+7.4f}")

# Plot novelty trend
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(yr_nov["year"], yr_nov["os_mean"], color="#0072B2", linewidth=2, marker="o", ms=5)
ax.fill_between(yr_nov["year"],
                yr_nov["os_mean"] - yr_nov["os_sd"]/np.sqrt(yr_nov["n"]),
                yr_nov["os_mean"] + yr_nov["os_sd"]/np.sqrt(yr_nov["n"]),
                alpha=0.2, color="#0072B2")
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("First-round year"); ax.set_ylabel("Mean overall_sparsity_t (standardised)")
ax.set_title("Novelty of newly-funded startups by year (first rounds)", fontsize=11)
ax.grid(linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_novelty_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved fig_novelty_trend.png")

# By Gaddy category
print(f"\n  Mean OS by year and Gaddy category (first rounds, 2014-2026 only for brevity):")
cat_nov = (fr_nov[fr_nov["year"]>=2014]
           .groupby(["year","gaddy"])["os"]
           .mean()
           .unstack()
           .reindex(columns=CATS))
print(cat_nov.round(3).to_string())

# Trend test: linear regression of mean_os on year
from scipy.stats import linregress
yr_arr = yr_nov["year"].values.astype(float)
os_arr = yr_nov["os_mean"].values
slope, intercept, r, p, se = linregress(yr_arr, os_arr)
print(f"\n  Linear trend in mean OS vs year:")
print(f"    slope = {slope:+.5f}/yr  (p={p:.4f}  r={r:.3f})")
if p < 0.05:
    print(f"    → Significant {'decline' if slope<0 else 'rise'} in novelty among entrants.")
else:
    print(f"    → No significant trend.")

# Save tables
yr_nov.to_csv(f"{OUT_DIR}/novelty_by_year.csv", index=False)
cat_nov.to_csv(f"{OUT_DIR}/novelty_by_year_category.csv")
(deal_sh_all * 100).round(1).to_csv(f"{OUT_DIR}/deal_share_all.csv")
(dol_sh_all  * 100).round(1).to_csv(f"{OUT_DIR}/dollar_share_all.csv")
(deal_sh_fr  * 100).round(1).to_csv(f"{OUT_DIR}/deal_share_first.csv")

print()
print("=" * 68)
print("SCRIPT 98 COMPLETE")
print("=" * 68)
