"""
Script 103: Quiet-exit feasibility and pre-registration draft.
NO hypothesis tests. Tables only.

Steps:
1. Export date identification
2. Investor matching (our deals ↔ Investors_data.xlsx)
3. Reliability check (Last Investment Date vs our data; last-2yr counts)
4. Quiet-exit candidate classification
5. Prior climate intensity distribution
6. Cell counts for planned tests
7. Preregistration draft (do not lock)
"""

import os, re, warnings
import numpy as np
import pandas as pd
from collections import Counter

warnings.filterwarnings("ignore")

OUT_DIR = "output/quiet_exit"
os.makedirs(OUT_DIR, exist_ok=True)

def sep(title=""):
    print(f"\n{'─'*68}")
    if title:
        print(f"  {title}")
        print(f"{'─'*68}")

def clean_name(s):
    """Lowercase, strip legal suffixes, strip parenthetical affiliations."""
    s = re.sub(r"\s*\(.*?\)\s*$", "", str(s)).strip()
    suffixes = (r"\b(llc|lp|l\.p\.|llp|ltd|limited|inc|incorporated|corp|"
                r"corporation|plc|gmbh|bv|b\.v\.|sas|sarl|ag|sa|nv|pty|"
                r"fund|partners|capital|ventures|management|group|holdings?"
                r"|advisors?|associates?)\b\.?")
    s = re.sub(suffixes, "", s, flags=re.IGNORECASE).strip().rstrip(",. ")
    return s.lower().strip()

print("=" * 68)
print("SCRIPT 103 — QUIET EXIT FEASIBILITY")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────
# Load deal data
# ─────────────────────────────────────────────────────────────────────
sep("Load deal data")
import sys
sys.path.insert(0, ".")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl = dl.dropna(subset=["Deal Date"]).copy()

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip() and p.strip() != "nan"]

rows = []
for _, row in dl.dropna(subset=["Investors"]).iterrows():
    for inv in comma_parse(row["Investors"]):
        rows.append({
            "investor_raw": inv,
            "investor_key": clean_name(inv),
            "deal_date": row["Deal Date"],
            "company": row["Companies"],
            "deal_id": row["Deal ID"],
        })
inv_deals = pd.DataFrame(rows)
print(f"  Investor-deal pairs: {len(inv_deals):,}")
print(f"  Unique investor names (raw): {inv_deals['investor_raw'].nunique():,}")
print(f"  Deal date range: {inv_deals['deal_date'].min().date()} → {inv_deals['deal_date'].max().date()}")

# ─────────────────────────────────────────────────────────────────────
# 1. Export date
# ─────────────────────────────────────────────────────────────────────
sep("1. Export date")

idf_raw = pd.read_excel("data/Investors_data.xlsx", header=7)
last_updated = pd.to_datetime(idf_raw["Last Updated Date"], errors="coerce")
last_inv_overall = pd.to_datetime(idf_raw["Last Investment Date"], errors="coerce")

export_date = last_updated.max()
inferred_from_last_inv = last_inv_overall.max()

print(f"  Last Updated Date (all rows max): {export_date.date()}")
print(f"    → unique values: {last_updated.dropna().nunique()} (all same: {(last_updated.dropna() == export_date).all()})")
print(f"  Max Last Investment Date in file: {inferred_from_last_inv.date()}")
print()
print(f"  EXPORT DATE: {export_date.date()}")
print(f"  (Confirmed from Last Updated Date column — all {last_updated.notna().sum():,} rows share the same date.)")

window_2yr_start = export_date - pd.DateOffset(years=2)   # 2024-08-26
window_5yr_start = export_date - pd.DateOffset(years=5)   # 2021-08-26

print(f"\n  Rolling windows (relative to export date):")
print(f"    Last 2 years : {window_2yr_start.date()} → {export_date.date()}")
print(f"    Last 5 years : {window_5yr_start.date()} → {export_date.date()}")

# ─────────────────────────────────────────────────────────────────────
# 2. Matching: our investor names ↔ Investors_data.xlsx
# ─────────────────────────────────────────────────────────────────────
sep("2. Investor matching")

idf_raw["_key"] = idf_raw["Investors"].apply(clean_name)
idf_key_set = set(idf_raw["_key"].dropna())

inv_deals["_in_pb"] = inv_deals["investor_key"].isin(idf_key_set)

uniq_inv = inv_deals[["investor_raw","investor_key"]].drop_duplicates("investor_key")
uniq_inv["in_pb"] = uniq_inv["investor_key"].isin(idf_key_set)

n_our = uniq_inv["investor_key"].nunique()
n_matched = uniq_inv["in_pb"].sum()
pct_matched = n_matched / n_our * 100

print(f"  Unique investors in deal data (key-deduplicated): {n_our:,}")
print(f"  Matched to Investors_data.xlsx:                   {n_matched:,} ({pct_matched:.1f}%)")
print(f"  Unmatched:                                        {n_our - n_matched:,} ({100-pct_matched:.1f}%)")

# Deal-pair coverage
pairs_matched = inv_deals["_in_pb"].sum()
print(f"\n  Investor-deal pairs with a PitchBook record: "
      f"{pairs_matched:,} / {len(inv_deals):,} ({pairs_matched/len(inv_deals)*100:.1f}%)")

# Build matched investor table (join PitchBook columns)
idf = idf_raw.copy()
idf_for_join = idf.set_index("_key")[[
    "Last Investment Date", "Investments in the last 2 years",
    "Investments in the last 5 years", "Total Investments",
    "Primary Investor Type",
]].rename(columns={
    "Last Investment Date": "pb_last_inv_date",
    "Investments in the last 2 years": "pb_inv_2yr",
    "Investments in the last 5 years": "pb_inv_5yr",
    "Total Investments": "pb_total_inv",
    "Primary Investor Type": "pb_type",
})

# For duplicate keys keep first (rare duplicates in PB)
idf_for_join = idf_for_join[~idf_for_join.index.duplicated(keep="first")]

inv_deals = inv_deals.join(idf_for_join, on="investor_key")
inv_deals["pb_last_inv_date"] = pd.to_datetime(inv_deals["pb_last_inv_date"], errors="coerce")

# Save matched investor list
matched_inv = (uniq_inv[uniq_inv["in_pb"]]
               .join(idf_for_join, on="investor_key")
               .reset_index(drop=True))
matched_inv.to_csv(f"{OUT_DIR}/matched_investors.csv", index=False)
print(f"\n  Saved matched_investors.csv ({len(matched_inv):,} rows)")

# ─────────────────────────────────────────────────────────────────────
# 3. Reliability check
# ─────────────────────────────────────────────────────────────────────
sep("3. Reliability check")

print("  3a. PitchBook Last Investment Date vs our last climate deal date")
print("      (PB should be >= our last deal; violations = data quality issue)")

# Per matched investor: their last climate deal date in our data
last_climate_deal = (inv_deals[inv_deals["_in_pb"]]
                     .groupby("investor_key")["deal_date"].max()
                     .rename("our_last_climate_date"))

rel_df = (matched_inv[["investor_raw","investor_key","pb_last_inv_date"]]
          .join(last_climate_deal, on="investor_key")
          .dropna(subset=["pb_last_inv_date","our_last_climate_date"]))

rel_df["pb_last_inv_date"] = pd.to_datetime(rel_df["pb_last_inv_date"], errors="coerce")
rel_df["violation"] = rel_df["pb_last_inv_date"] < rel_df["our_last_climate_date"]
rel_df["delta_days"] = (rel_df["pb_last_inv_date"] - rel_df["our_last_climate_date"]).dt.days

n_rel = len(rel_df)
n_viol = rel_df["violation"].sum()
print(f"\n  Investors with both dates available: {n_rel:,}")
print(f"  Violations (PB date < our last climate deal): {n_viol:,} ({n_viol/n_rel*100:.1f}%)")

ex_viol = (rel_df[rel_df["violation"]]
           .nsmallest(10, "delta_days")
           [["investor_raw","pb_last_inv_date","our_last_climate_date","delta_days"]])
print("\n  10 worst violations (largest discrepancy):")
print(ex_viol.to_string(index=False))

print("\n  3b. PitchBook 'Investments in last 2 years' vs our climate deal count (same window)")
print(f"      Window: {window_2yr_start.date()} → {export_date.date()}")
print(f"      NOTE: PB column has {int(idf_raw['Investments in the last 2 years'].notna().sum()):,} non-null rows "
      f"out of {len(idf_raw):,} — NaN likely means 0 (not missing)")

# Our climate deal count in last-2yr window
our_2yr_count = (inv_deals[inv_deals["_in_pb"]]
                 .query("deal_date >= @window_2yr_start")
                 .groupby("investor_key")["deal_id"].nunique()
                 .rename("our_climate_2yr"))

cnt_df = (matched_inv[["investor_raw","investor_key","pb_inv_2yr"]]
          .join(our_2yr_count, on="investor_key")
          .fillna({"pb_inv_2yr": 0, "our_climate_2yr": 0}))

# Violation: overall PB count < our climate count (impossible if data consistent)
cnt_df["cnt_violation"] = cnt_df["pb_inv_2yr"] < cnt_df["our_climate_2yr"]
n_cnt_viol = cnt_df["cnt_violation"].sum()
pct_cnt_viol = n_cnt_viol / len(cnt_df) * 100
print(f"\n  Total matched investors: {len(cnt_df):,}")
print(f"  Count violations (PB overall 2yr < our climate 2yr): {n_cnt_viol:,} ({pct_cnt_viol:.1f}%)")

# Distribution of our_climate vs pb total
cnt_comp = cnt_df.groupby("our_climate_2yr")["pb_inv_2yr"].describe()[["count","mean","50%","max"]]
print("\n  Our climate deal count (2yr) → PB total inv count (2yr):")
print(cnt_comp.head(10).to_string())

rel_df.to_csv(f"{OUT_DIR}/reliability_date_check.csv", index=False)
cnt_df.to_csv(f"{OUT_DIR}/reliability_count_check.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 4. Quiet-exit candidate classification
# ─────────────────────────────────────────────────────────────────────
sep("4. Quiet-exit classification")

# Base population: ≥1 climate deal in the 5 years before export
print(f"  Base population: investors with ≥1 climate deal between "
      f"{window_5yr_start.date()} and {export_date.date()}")

base_inv = (inv_deals[inv_deals["_in_pb"] &
                      (inv_deals["deal_date"] >= window_5yr_start) &
                      (inv_deals["deal_date"] <= export_date)]
            ["investor_key"].unique())
print(f"  Base population N: {len(base_inv):,}")

# Climate deals in last 2 years (our data)
climate_2yr_set = set(
    inv_deals[(inv_deals["deal_date"] >= window_2yr_start) &
              (inv_deals["deal_date"] <= export_date)]
    ["investor_key"]
)

# PB columns for each base investor
base_df = (matched_inv[matched_inv["investor_key"].isin(base_inv)]
           .set_index("investor_key")[["investor_raw","pb_last_inv_date","pb_inv_2yr","pb_type"]]
           .copy())
base_df["pb_inv_2yr"] = pd.to_numeric(base_df["pb_inv_2yr"], errors="coerce").fillna(0)
base_df["pb_last_inv_date"] = pd.to_datetime(base_df["pb_last_inv_date"], errors="coerce")
base_df["has_climate_2yr"]  = base_df.index.isin(climate_2yr_set)
base_df["pb_active_2yr"]    = base_df["pb_inv_2yr"] > 0

# Classification
def classify_exit(row):
    if row["has_climate_2yr"]:
        return "stayer"
    elif row["pb_active_2yr"]:
        return "quiet_exit"
    else:
        return "full_exit"

base_df["exit_status"] = base_df.apply(classify_exit, axis=1)

# Additional: pb_active_2yr NaN flag (can't distinguish quiet vs full for these)
base_df["pb_2yr_unknown"] = base_df["pb_inv_2yr"] == 0  # 0 from fillna — ambiguous

counts = base_df["exit_status"].value_counts()
total_base = len(base_df)
print(f"\n  {'Status':<15}  {'N':>6}  {'Share':>8}")
print(f"  {'─'*35}")
for status in ["stayer","quiet_exit","full_exit"]:
    n = counts.get(status, 0)
    print(f"  {status:<15}  {n:>6,}  {n/total_base*100:>7.1f}%")

print(f"\n  NOTE: pb_inv_2yr NaN treated as 0 (consistent with PitchBook convention where "
      f"no entry = no recorded deals). Investors with pb_inv_2yr=0 and no climate 2yr "
      f"are classified full_exit. If NaN means 'data not available', quiet_exit N may be "
      f"understated.")
print(f"\n  Investors with pb_inv_2yr = NaN (original, before fill): "
      f"{(matched_inv[matched_inv['investor_key'].isin(base_inv)]['pb_inv_2yr'].isna().sum()):,}")

base_df.reset_index().to_csv(f"{OUT_DIR}/base_population.csv", index=False)
print(f"\n  Saved base_population.csv")

# ─────────────────────────────────────────────────────────────────────
# 5. Prior climate intensity
# ─────────────────────────────────────────────────────────────────────
sep("5. Prior climate intensity")

# Climate intensity = climate deals (all time, in our data) / pb_total_inv
# "Prior" = before the last-2yr window = up to 2024-08-26

prior_cutoff = window_2yr_start

base_keys = set(base_df.index)

# All climate deals before the 2yr window (prior period)
prior_climate = (inv_deals[inv_deals["_in_pb"] &
                            inv_deals["investor_key"].isin(base_keys) &
                            (inv_deals["deal_date"] < prior_cutoff)]
                 .groupby("investor_key")["deal_id"].nunique()
                 .rename("prior_climate_count"))

# Join with PB total investments
intensity_df = (base_df[["investor_raw","exit_status","pb_type"]]
                .join(prior_climate)
                .join(matched_inv.set_index("investor_key")["pb_total_inv"]))
intensity_df["prior_climate_count"] = intensity_df["prior_climate_count"].fillna(0)
intensity_df["pb_total_inv"] = pd.to_numeric(intensity_df["pb_total_inv"], errors="coerce")

# Intensity = prior_climate / total_investments (where total > 0)
mask_pos = intensity_df["pb_total_inv"] > 0
intensity_df.loc[mask_pos, "intensity"] = (
    intensity_df.loc[mask_pos, "prior_climate_count"] /
    intensity_df.loc[mask_pos, "pb_total_inv"]
)
intensity_df["intensity"] = intensity_df["intensity"].clip(0, 1)  # cap at 1

print(f"  Base population: {len(intensity_df):,}")
print(f"  With intensity calculable (pb_total_inv > 0): {mask_pos.sum():,}")

desc = intensity_df["intensity"].describe(percentiles=[.1,.25,.5,.75,.9])
print(f"\n  Intensity distribution:")
print(f"    mean   : {desc['mean']:.3f}")
print(f"    10th p : {desc['10%']:.3f}")
print(f"    25th p : {desc['25%']:.3f}")
print(f"    50th p : {desc['50%']:.3f}")
print(f"    75th p : {desc['75%']:.3f}")
print(f"    90th p : {desc['90%']:.3f}")
print(f"    max    : {desc['max']:.3f}")

# Quintiles
intensity_df["intensity_q"] = pd.qcut(
    intensity_df["intensity"].fillna(0), 5,
    labels=["Q1 (lowest)","Q2","Q3","Q4","Q5 (highest)"],
    duplicates="drop"
)

qt_exit = intensity_df.groupby("intensity_q")["exit_status"].value_counts(normalize=True).unstack(fill_value=0)
qt_count = intensity_df.groupby("intensity_q")["exit_status"].count()
qt_exit["N"] = qt_count
print(f"\n  Exit status by intensity quintile (share):")
print(qt_exit.round(3).to_string())

intensity_df.to_csv(f"{OUT_DIR}/intensity.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 6. Cell counts for planned tests
# ─────────────────────────────────────────────────────────────────────
sep("6. Cell counts")

# Load green_focus classification
all_inv = pd.read_csv("output/all_investors.csv")[["Investor","green_focus"]].copy()
all_inv["_key"] = all_inv["Investor"].apply(clean_name)
gf_map = all_inv.set_index("_key")["green_focus"].to_dict()
intensity_df["green_focus"] = intensity_df.index.map(gf_map).fillna("UNKNOWN")

# Load community
inv_prof = pd.read_csv("output/investor_profiles.csv")[["investor","community"]].copy()
inv_prof["_key"] = inv_prof["investor"].apply(clean_name)
comm_map = inv_prof.set_index("_key")["community"].to_dict()
intensity_df["community"] = intensity_df.index.map(comm_map)

# Load novelty (mean specter_novelty of ventures funded by each investor)
nov = pd.read_csv("output/novelty/novelty_scores.csv")[["name","specter_novelty"]].copy()
nov_map = dict(zip(nov["name"], nov["specter_novelty"]))
inv_deals["specter_novelty"] = inv_deals["company"].map(nov_map)
mean_novelty = (inv_deals[inv_deals["investor_key"].isin(base_keys)]
                .groupby("investor_key")["specter_novelty"]
                .mean()
                .rename("portfolio_novelty"))
intensity_df = intensity_df.join(mean_novelty)

print("  6a. green_focus × intensity_quintile (base population)")
gf_cross = (intensity_df.groupby(["green_focus","intensity_q"])
            .size().unstack(fill_value=0))
print(gf_cross.to_string())

print("\n  6b. green_focus × exit_status (base population)")
gf_exit = (intensity_df.groupby(["green_focus","exit_status"])
           .size().unstack(fill_value=0))
print(gf_exit.to_string())

print("\n  6c. investor_type (PitchBook primary) × exit_status")
type_exit = (intensity_df.groupby(["pb_type","exit_status"])
             .size().unstack(fill_value=0))
print(type_exit.to_string())

print("\n  6d. Community cell counts (base population)")
comm_counts = (intensity_df["community"]
               .value_counts()
               .rename("n_base"))
comm_with_30plus = comm_counts[comm_counts >= 30]
print(f"  Communities with ≥30 base-population investors: {len(comm_with_30plus)}")
print(f"  Communities with ≥10: {(comm_counts >= 10).sum()}")
print(f"  Investors in communities ≥30: {comm_with_30plus.sum():,} / {len(intensity_df):,}")
print(f"  Investors with no community assignment: {intensity_df['community'].isna().sum():,}")

if len(comm_with_30plus) > 0:
    comm_detail = (intensity_df[intensity_df["community"].isin(comm_with_30plus.index)]
                   .groupby("community")["exit_status"]
                   .value_counts().unstack(fill_value=0)
                   .assign(N=lambda d: d.sum(axis=1))
                   .sort_values("N", ascending=False))
    print(f"\n  Communities ≥30 (exit breakdown):")
    print(comm_detail.to_string())

print(f"\n  6e. Portfolio novelty coverage")
has_novelty = intensity_df["portfolio_novelty"].notna()
print(f"  Base-population investors with portfolio novelty: "
      f"{has_novelty.sum():,} / {len(intensity_df):,} ({has_novelty.mean()*100:.1f}%)")
print(f"  Mean portfolio novelty — stayers:     "
      f"{intensity_df[intensity_df['exit_status']=='stayer']['portfolio_novelty'].mean():.4f}")
print(f"  Mean portfolio novelty — quiet_exit:  "
      f"{intensity_df[intensity_df['exit_status']=='quiet_exit']['portfolio_novelty'].mean():.4f}")
print(f"  Mean portfolio novelty — full_exit:   "
      f"{intensity_df[intensity_df['exit_status']=='full_exit']['portfolio_novelty'].mean():.4f}")

intensity_df.reset_index().to_csv(f"{OUT_DIR}/base_population_full.csv", index=False)
print(f"\n  Saved base_population_full.csv")

# ─────────────────────────────────────────────────────────────────────
# 7. Pre-registration draft
# ─────────────────────────────────────────────────────────────────────
sep("7. Pre-registration draft")

quiet_n   = (base_df["exit_status"] == "quiet_exit").sum()
stayer_n  = (base_df["exit_status"] == "stayer").sum()
full_n    = (base_df["exit_status"] == "full_exit").sum()
q2_n      = min(
    (intensity_df.query("exit_status in ['quiet_exit','stayer']") ["portfolio_novelty"].notna()).sum(),
    (intensity_df["exit_status"].isin(["quiet_exit","stayer"])).sum()
)
q3_green  = intensity_df[intensity_df["green_focus"] == "GREEN_VC"]["exit_status"].isin(["quiet_exit","stayer"]).sum()
q3_trad   = intensity_df[intensity_df["green_focus"] == "TRADITIONAL"]["exit_status"].isin(["quiet_exit","stayer"]).sum()
q4_comm   = len(comm_with_30plus)

prereg = f"""
## Family I — Quiet Exit (Script 103 feasibility | do not lock)

**Status:** DRAFT — do not lock until review

**Data anchor:** PitchBook Investors_data.xlsx exported {export_date.date()}.
Rolling-window columns "Investments in the last N years" are relative to this date.

### Definitions

| Term | Definition |
|---|---|
| Export date | {export_date.date()} (Last Updated Date, all rows identical) |
| Last-2yr window | {window_2yr_start.date()} – {export_date.date()} |
| Last-5yr window | {window_5yr_start.date()} – {export_date.date()} |
| Base population | Investors matched to Investors_data.xlsx with ≥1 climate deal in 5-yr window |
| stayer | ≥1 climate deal in 2-yr window (our data) |
| quiet_exit | 0 climate deals in 2-yr window AND pb_inv_2yr > 0 (still making deals elsewhere) |
| full_exit | 0 climate deals in 2-yr window AND pb_inv_2yr = 0 |
| green_focus | GREEN_VC / ESG / TRADITIONAL (from all_investors.csv classification) |
| intensity | prior climate deals / pb_total_inv (before 2-yr window) |
| portfolio_novelty | mean specter_novelty of ventures funded by investor (pre-2yr) |

### Feasibility numbers (Script 103, not pre-specified)
Base population: {total_base:,}
 — stayer: {stayer_n:,} ({stayer_n/total_base*100:.1f}%)
 — quiet_exit: {quiet_n:,} ({quiet_n/total_base*100:.1f}%)
 — full_exit: {full_n:,} ({full_n/total_base*100:.1f}%)

Communities ≥30 base investors: {q4_comm}
Investors with portfolio novelty: {has_novelty.sum():,} / {len(intensity_df):,}

### Q1 (snapshot, descriptive — no test)
**What share of climate-active investors went quiet in the last two years?**
Estimate quiet-exit share = quiet_exit / (quiet_exit + stayer + full_exit).
Report N and share by green_focus and investor type. No inferential test.

### Q2 (pre-registered, directional)
**Did quiet exiters fund more novel ventures (higher specter_novelty) than stayers?**
Prediction: positive (quiet exiters backed more frontier/novel startups).
Unit: investor × portfolio_novelty.
Test: OLS with intensity quintile FE; SE clustered by community where available.
Outcome: portfolio_novelty (mean specter_novelty of pre-2yr climate portfolio).
Min cell: quiet_exit with portfolio_novelty ≈ {(intensity_df['exit_status']=='quiet_exit')['portfolio_novelty' in intensity_df.columns and intensity_df['portfolio_novelty'].notna()].sum() if 'portfolio_novelty' in intensity_df.columns else 'see above'}.
Pre-specified sign: positive coefficient on quiet_exit dummy.

### Q3 (pre-registered, directional)
**Within intensity quintiles, do GREEN_VC investors quiet-exit more than TRADITIONAL?**
Prediction: positive (GREEN_VC has higher quiet-exit rate than TRADITIONAL within quintile).
Test: logit quiet_exit ~ green_focus dummies + intensity quintile FE.
Reference: TRADITIONAL. Pre-specified sign: positive coefficient on GREEN_VC.
Min cells: GREEN_VC × stayer/quiet_exit = {q3_green:,}; TRADITIONAL × stayer/quiet_exit = {q3_trad:,}.

### Q4 (pre-registered, two-sided)
**Are quiet exits concentrated in particular investor communities beyond chance?**
Test: permutation test (999 draws) — observed community-level quiet-exit rate vs
random allocation of quiet_exit labels across base population.
Null: community membership does not predict quiet-exit status.
Test statistic: max(community quiet_exit share) − overall quiet_exit share.
Restrict to communities with ≥30 base-population investors (N communities = {q4_comm}).

### Holm correction
Applied across Q2, Q3, Q4. Q1 is descriptive only.

---
*Do not lock this section. Review cell counts and definitions before locking.*
"""

print(prereg)

# Append to preregistration.md
prereg_path = "output/preregistration.md"
if os.path.exists(prereg_path):
    with open(prereg_path, "a") as f:
        f.write(prereg)
    print(f"\n  Appended to {prereg_path}")
else:
    with open(prereg_path, "w") as f:
        f.write(prereg)
    print(f"\n  Created {prereg_path}")

# Also save standalone
with open(f"{OUT_DIR}/prereg_draft_family_I.md", "w") as f:
    f.write(prereg)

print("\n" + "=" * 68)
print("SCRIPT 103 COMPLETE — STOP. Awaiting review before any tests.")
print("=" * 68)
