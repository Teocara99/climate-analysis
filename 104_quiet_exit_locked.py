"""
Script 104: Fix quiet-exit classification and lock preregistration.

1. Deals export date — establish window endpoints
2. Reclassify using Last Investment Date (not pb_inv_2yr)
3. Name-collision filtering (country check + single-word exclusion)
4. Quiet-exit validation (Last Investment Type + investee company)
5. Re-report intensity-quintile table
6. Lock preregistration (Family I, Q2–Q4)
"""

import os, re, warnings, datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = "output/quiet_exit"
os.makedirs(OUT_DIR, exist_ok=True)

def sep(title=""):
    print(f"\n{'─'*68}")
    if title:
        print(f"  {title}")
        print(f"{'─'*68}")

# ── name normalisation ──────────────────────────────────────────────
def clean_name(s):
    s = re.sub(r"\s*\(.*?\)\s*$", "", str(s)).strip()
    suffixes = (r"\b(llc|lp|l\.p\.|llp|ltd|limited|inc|incorporated|corp|"
                r"corporation|plc|gmbh|bv|b\.v\.|sas|sarl|ag|sa|nv|pty|"
                r"fund|partners|capital|ventures|management|group|holdings?"
                r"|advisors?|associates?)\b\.?")
    s = re.sub(suffixes, "", s, flags=re.IGNORECASE).strip().rstrip(",. ")
    return s.lower().strip()

def word_count(key):
    return len(key.split())

# ── country → broad region ───────────────────────────────────────────
def country_to_region(c):
    c = str(c).strip()
    americas = {"United States","Canada","Brazil","Mexico","Argentina","Chile",
                "Colombia","Peru","Uruguay","Costa Rica"}
    europe   = {"United Kingdom","Germany","France","Sweden","Netherlands",
                "Switzerland","Spain","Italy","Denmark","Finland","Norway",
                "Belgium","Austria","Portugal","Ireland","Luxembourg",
                "Poland","Czech Republic","Estonia","Latvia","Lithuania"}
    asia     = {"China","India","Japan","South Korea","Singapore","Israel",
                "Hong Kong","Taiwan","Indonesia","Malaysia","Vietnam","Thailand"}
    oceania  = {"Australia","New Zealand"}
    mena     = {"United Arab Emirates","Saudi Arabia","Egypt","Qatar","Kuwait",
                "Bahrain","Jordan","Turkey"}
    africa   = {"South Africa","Kenya","Nigeria","Ghana"}
    if c in americas: return "Americas"
    if c in europe:   return "Europe"
    if c in asia:     return "Asia"
    if c in oceania:  return "Oceania"
    if c in mena:     return "Middle East"
    if c in africa:   return "Africa"
    return "Other"

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip()
            for p in parts if p.strip() and p.strip() != "nan"]

print("=" * 68)
print("SCRIPT 104 — QUIET EXIT LOCKED")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
sep("Load data")

import sys
sys.path.insert(0, ".")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl = dl.dropna(subset=["Deal Date"]).copy()

# Explode investor-deal pairs
rows = []
for _, row in dl.dropna(subset=["Investors"]).iterrows():
    for inv in comma_parse(row["Investors"]):
        rows.append({
            "investor_raw": inv,
            "investor_key": clean_name(inv),
            "deal_date": row["Deal Date"],
            "company": row["Companies"],
            "deal_id": row["Deal ID"],
            "deal_country": row.get("Company Country/Territory/Region", np.nan),
        })
inv_deals = pd.DataFrame(rows)

# PitchBook investor table
idf_raw = pd.read_excel("data/Investors_data.xlsx", header=7, usecols=[
    "Investors", "Primary Investor Type",
    "HQ Country/Territory/Region",
    "Last Investment Date",
    "Last Investment Type",
    "Last Investment Company",
])
idf_raw["_key"] = idf_raw["Investors"].apply(clean_name)
idf_raw = idf_raw[~idf_raw["_key"].duplicated(keep="first")].copy()
idf_raw["Last Investment Date"] = pd.to_datetime(idf_raw["Last Investment Date"], errors="coerce")
idf_raw["pb_region"] = idf_raw["HQ Country/Territory/Region"].apply(country_to_region)

# Investor geo_focus from profiles
ip = pd.read_csv("output/investor_profiles.csv")[["investor","geo_focus","community"]].copy()
ip["_key"] = ip["investor"].apply(clean_name)
geo_map   = ip.drop_duplicates("_key").set_index("_key")["geo_focus"].to_dict()
comm_map  = ip.drop_duplicates("_key").set_index("_key")["community"].to_dict()

# Green_focus
all_inv   = pd.read_csv("output/all_investors.csv")[["Investor","green_focus"]].copy()
all_inv["_key"] = all_inv["Investor"].apply(clean_name)
gf_map    = all_inv.drop_duplicates("_key").set_index("_key")["green_focus"].to_dict()

# Novelty
nov       = pd.read_csv("output/novelty/novelty_scores.csv")[["name","specter_novelty"]].copy()
nov_map   = dict(zip(nov["name"], nov["specter_novelty"]))
inv_deals["specter_novelty"] = inv_deals["company"].map(nov_map)

print(f"  Investor-deal pairs: {len(inv_deals):,}")

# ─────────────────────────────────────────────────────────────────────
# 1. Deals export date and window adjustment
# ─────────────────────────────────────────────────────────────────────
sep("1. Deals export date")

deals_max  = dl["Deal Date"].max()
pb_export  = pd.Timestamp("2026-08-26")
window_end = min(deals_max, pb_export)

monthly_25_26 = (dl[dl["Deal Date"].dt.year >= 2025]
                 .groupby(dl["Deal Date"].dt.to_period("M")).size()
                 .rename("deals"))
print(f"  Latest Deal Date in file: {deals_max.date()}")
print(f"  PitchBook export date:    {pb_export.date()}")
print(f"  → Window end (min of both): {window_end.date()}")
print()
print("  Deal counts by month (2025 onward):")
print(monthly_25_26.to_string())

window_2yr_start = window_end - pd.DateOffset(years=2)
window_5yr_start = window_end - pd.DateOffset(years=5)

print(f"\n  Restated windows (end = deals cutoff {window_end.date()}):")
print(f"    Last 2 years : {window_2yr_start.date()} → {window_end.date()}")
print(f"    Last 5 years : {window_5yr_start.date()} → {window_end.date()}")
print(f"  NOTE: PB Last Investment Date may include deals between {window_end.date()} "
      f"and {pb_export.date()}.  These are treated as evidence of general activity but "
      f"cannot be verified for climate content in our data.")

# ─────────────────────────────────────────────────────────────────────
# 2. Pre-collision matching and initial classification
# ─────────────────────────────────────────────────────────────────────
sep("2. Initial classification (pre-collision filter)")

idf_idx = idf_raw.set_index("_key")

uniq_inv = (inv_deals[["investor_raw","investor_key"]]
            .drop_duplicates("investor_key").copy())
uniq_inv["in_pb_raw"] = uniq_inv["investor_key"].isin(idf_raw["_key"].values)

n_our = len(uniq_inv)
n_matched_raw = uniq_inv["in_pb_raw"].sum()
print(f"  Unique investors in deal data: {n_our:,}")
print(f"  Matched (pre-filter):          {n_matched_raw:,} ({n_matched_raw/n_our*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 3. Name-collision filter
# ─────────────────────────────────────────────────────────────────────
sep("3. Name-collision filter")

# Rule A: exclude single-word names (after suffix stripping)
uniq_inv["word_count"]    = uniq_inv["investor_key"].apply(word_count)
uniq_inv["is_single_word"] = uniq_inv["word_count"] == 1

# Rule B: country mismatch
# Our-data region: from geo_focus (Americas / Europe / Asia / Oceania / Middle East / Africa)
# PB region: country_to_region(HQ Country)
# Conflict = both available AND they disagree

# Map geo_focus region label to our 5-bucket system (they already use same labels)
uniq_inv["our_region"] = uniq_inv["investor_key"].map(geo_map)
uniq_inv["pb_region"]  = uniq_inv["investor_key"].map(
    idf_raw.set_index("_key")["pb_region"].to_dict()
)

# Conflict: both non-null, non-Other, and disagree
both_avail = (
    uniq_inv["our_region"].notna() &
    uniq_inv["pb_region"].notna() &
    (uniq_inv["our_region"] != "Other") &
    (uniq_inv["pb_region"] != "Other")
)
uniq_inv["country_conflict"] = both_avail & (uniq_inv["our_region"] != uniq_inv["pb_region"])

# Apply: exclude if single-word OR country conflict
uniq_inv["exclude"] = uniq_inv["is_single_word"] | uniq_inv["country_conflict"]

n_excl_single = (uniq_inv["in_pb_raw"] & uniq_inv["is_single_word"]).sum()
n_excl_ctry   = (uniq_inv["in_pb_raw"] & uniq_inv["country_conflict"] & ~uniq_inv["is_single_word"]).sum()
n_excl_total  = (uniq_inv["in_pb_raw"] & uniq_inv["exclude"]).sum()
n_matched_clean = (uniq_inv["in_pb_raw"] & ~uniq_inv["exclude"]).sum()

print(f"  Excluded — single-word name:   {n_excl_single:,}")
print(f"  Excluded — country mismatch (multi-word): {n_excl_ctry:,}")
print(f"  Total excluded:                {n_excl_total:,}")
print(f"  Retained (clean matches):      {n_matched_clean:,} ({n_matched_clean/n_our*100:.1f}%)")

# Examples of excluded by each rule
excl_sw = uniq_inv[uniq_inv["in_pb_raw"] & uniq_inv["is_single_word"]].head(15)
print(f"\n  Sample single-word exclusions ({min(15, len(excl_sw))}):")
print(excl_sw[["investor_raw","investor_key","our_region","pb_region"]].to_string(index=False))

excl_ct = uniq_inv[uniq_inv["in_pb_raw"] & uniq_inv["country_conflict"] & ~uniq_inv["is_single_word"]].head(15)
if len(excl_ct):
    print(f"\n  Sample country-mismatch exclusions ({min(15, len(excl_ct))}):")
    print(excl_ct[["investor_raw","investor_key","our_region","pb_region"]].to_string(index=False))

# Build clean match set
clean_match_keys = set(uniq_inv[uniq_inv["in_pb_raw"] & ~uniq_inv["exclude"]]["investor_key"])
inv_deals["in_pb_clean"] = inv_deals["investor_key"].isin(clean_match_keys)

# Join PB columns to clean-matched deals
inv_deals_clean = inv_deals[inv_deals["in_pb_clean"]].copy()
pb_join = idf_idx[[
    "Last Investment Date","Last Investment Type","Last Investment Company","pb_region"
]].rename(columns={
    "Last Investment Date": "pb_last_inv_date",
    "Last Investment Type": "pb_last_inv_type",
    "Last Investment Company": "pb_last_inv_co",
})
inv_deals_clean = inv_deals_clean.join(pb_join, on="investor_key")
inv_deals_clean["pb_last_inv_date"] = pd.to_datetime(inv_deals_clean["pb_last_inv_date"], errors="coerce")

# ─────────────────────────────────────────────────────────────────────
# 4. Reclassify using Last Investment Date
# ─────────────────────────────────────────────────────────────────────
sep("4. Reclassification using Last Investment Date")

# Base population: ≥1 climate deal in 5yr window
base_keys_5yr = set(
    inv_deals_clean[
        (inv_deals_clean["deal_date"] >= window_5yr_start) &
        (inv_deals_clean["deal_date"] <= window_end)
    ]["investor_key"].unique()
)

# Per investor: last PB investment date and climate deal presence in 2yr window
pb_dates = (inv_deals_clean[["investor_key","pb_last_inv_date"]]
            .drop_duplicates("investor_key")
            .set_index("investor_key")["pb_last_inv_date"])

climate_2yr_keys = set(
    inv_deals_clean[
        (inv_deals_clean["deal_date"] >= window_2yr_start) &
        (inv_deals_clean["deal_date"] <= window_end)
    ]["investor_key"].unique()
)

print(f"  Base population (≥1 climate deal in 5yr, clean matches): {len(base_keys_5yr):,}")

# Build base_df
base_rows = []
for key in base_keys_5yr:
    pb_date = pb_dates.get(key, pd.NaT)
    has_climate_2yr = key in climate_2yr_keys
    active_overall = (not pd.isna(pb_date)) and (pb_date >= window_2yr_start)

    if has_climate_2yr:
        status = "stayer"
    elif active_overall:
        status = "quiet_exit"
    else:
        status = "full_exit"

    base_rows.append({
        "investor_key": key,
        "pb_last_inv_date": pb_date,
        "active_overall": active_overall,
        "has_climate_2yr": has_climate_2yr,
        "exit_status": status,
    })

base_df = pd.DataFrame(base_rows).set_index("investor_key")

counts = base_df["exit_status"].value_counts()
total_base = len(base_df)
print(f"\n  {'Status':<15}  {'N':>6}  {'Share':>8}")
print(f"  {'─'*35}")
for status in ["stayer","quiet_exit","full_exit"]:
    n = counts.get(status, 0)
    print(f"  {status:<15}  {n:>6,}  {n/total_base*100:>7.1f}%")

print(f"\n  Classification logic:")
print(f"    active_overall = Last Investment Date >= {window_2yr_start.date()}")
print(f"    stayer     = ≥1 climate deal in 2yr window")
print(f"    quiet_exit = active_overall AND 0 climate deals in 2yr window")
print(f"    full_exit  = Last Investment Date < {window_2yr_start.date()}")
print(f"\n  Investors with missing Last Investment Date (→ full_exit): "
      f"{base_df['pb_last_inv_date'].isna().sum():,}")

# ─────────────────────────────────────────────────────────────────────
# 5. Quiet-exit validation
# ─────────────────────────────────────────────────────────────────────
sep("5. Quiet-exit validation")

qe_keys = set(base_df[base_df["exit_status"] == "quiet_exit"].index)
qe_df   = (inv_deals_clean[["investor_key","pb_last_inv_type","pb_last_inv_co"]]
           [inv_deals_clean["investor_key"].isin(qe_keys)]
           .drop_duplicates("investor_key"))

print(f"  Quiet exiters: {len(qe_keys):,}")

# Last investment type distribution
print(f"\n  Last Investment Type (PitchBook) for quiet exiters:")
lt = qe_df["pb_last_inv_type"].value_counts()
print(lt.to_string())

# Climate/cleantech keywords in last investee company name
cleantech_kw = re.compile(
    r"solar|wind|energy|climate|clean|battery|ev|electric|hydrogen|carbon|"
    r"thermal|grid|storage|renewabl|sustainab|green|biogas|biomass|"
    r"geotherm|nuclear|fuel.cell|water|environ|emission|recycl",
    re.IGNORECASE
)

def is_ct(company_str):
    if pd.isna(company_str):
        return False
    # Remove parenthetical country
    name_part = re.sub(r"\s*\(.*?\)\s*$", "", str(company_str)).strip()
    return bool(cleantech_kw.search(name_part))

qe_df = qe_df.copy()
qe_df["last_co_is_ct"] = qe_df["pb_last_inv_co"].apply(is_ct)

# Also: is the last investee company IN our sample?
our_companies = set(dl["Companies"].dropna().str.strip().str.lower())
qe_df["last_co_in_our_sample"] = qe_df["pb_last_inv_co"].apply(
    lambda x: re.sub(r"\s*\(.*?\)\s*$","",str(x)).strip().lower() in our_companies
    if not pd.isna(x) else False
)

n_qe = len(qe_df)
n_ct_last = qe_df["last_co_is_ct"].sum()
n_in_sample = qe_df["last_co_in_our_sample"].sum()

print(f"\n  Of {n_qe:,} quiet exiters with PB data:")
print(f"    Last investee has cleantech keywords: {n_ct_last:,} ({n_ct_last/n_qe*100:.1f}%)")
print(f"    Last investee IS in our sample:        {n_in_sample:,} ({n_in_sample/n_qe*100:.1f}%)")

# Potential misclassifications: last investee has CT keywords but NOT in our sample
# These could be investors active in climate but outside our data coverage
misclass = qe_df[qe_df["last_co_is_ct"] & ~qe_df["last_co_in_our_sample"]]
print(f"\n  Potential misclassifications (CT last investee not in our sample): "
      f"{len(misclass):,} ({len(misclass)/n_qe*100:.1f}%)")
if len(misclass) > 0:
    print(f"  Examples:")
    print(misclass[["investor_key","pb_last_inv_type","pb_last_inv_co"]].head(20).to_string(index=False))

qe_df.to_csv(f"{OUT_DIR}/quiet_exit_validation.csv", index=False)
print(f"\n  Saved quiet_exit_validation.csv")

# ─────────────────────────────────────────────────────────────────────
# 6. Enrich base_df and re-report intensity quintile table
# ─────────────────────────────────────────────────────────────────────
sep("6. Intensity quintile table (post-fix)")

# Prior climate intensity
prior_cutoff = window_2yr_start
prior_climate = (inv_deals_clean[
    inv_deals_clean["investor_key"].isin(base_df.index) &
    (inv_deals_clean["deal_date"] < prior_cutoff)
].groupby("investor_key")["deal_id"].nunique().rename("prior_climate_count"))

# PB total investments
pb_total_map = idf_raw.set_index("_key")["Total Investments"].to_dict() if "Total Investments" in idf_raw.columns else {}
# Re-read with total investments
idf_tot = pd.read_excel("data/Investors_data.xlsx", header=7,
                        usecols=["Investors","Total Investments"]).copy()
idf_tot["_key"] = idf_tot["Investors"].apply(clean_name)
idf_tot = idf_tot.drop_duplicates("_key")
pb_total_map = dict(zip(idf_tot["_key"], pd.to_numeric(idf_tot["Total Investments"], errors="coerce")))

base_df["prior_climate_count"] = base_df.index.map(prior_climate).fillna(0)
base_df["pb_total_inv"] = base_df.index.map(pb_total_map)
base_df["pb_total_inv"] = pd.to_numeric(base_df["pb_total_inv"], errors="coerce")
mask_pos = base_df["pb_total_inv"] > 0
base_df.loc[mask_pos, "intensity"] = (
    base_df.loc[mask_pos,"prior_climate_count"] / base_df.loc[mask_pos,"pb_total_inv"]
).clip(0, 1)

base_df["green_focus"] = base_df.index.map(gf_map).fillna("UNKNOWN")
base_df["community"]   = base_df.index.map(comm_map)

mean_nov = (inv_deals_clean[inv_deals_clean["investor_key"].isin(base_df.index)]
            .groupby("investor_key")["specter_novelty"]
            .mean().rename("portfolio_novelty"))
base_df["portfolio_novelty"] = base_df.index.map(mean_nov)

base_df["intensity_q"] = pd.qcut(
    base_df["intensity"].fillna(0), 5,
    labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop"
)

qt_exit = (base_df.groupby(["intensity_q","exit_status"])
           .size().unstack(fill_value=0))
qt_exit["N"] = qt_exit.sum(axis=1)
for col in ["stayer","quiet_exit","full_exit"]:
    if col in qt_exit.columns:
        qt_exit[f"sh_{col}"] = (qt_exit[col] / qt_exit["N"]).round(3)

print("  Exit status by intensity quintile:")
print(qt_exit.to_string())

# Summary statistics
print(f"\n  Overall quiet-exit share: "
      f"{counts.get('quiet_exit',0)/total_base*100:.1f}%")
print(f"\n  Portfolio novelty (mean specter_novelty):")
for s in ["stayer","quiet_exit","full_exit"]:
    v = base_df[base_df["exit_status"]==s]["portfolio_novelty"].mean()
    print(f"    {s:<15}: {v:.4f}")

# Cell counts for tests
comm_counts = base_df["community"].value_counts()
comm_30plus = comm_counts[comm_counts >= 30]
print(f"\n  Communities ≥30 base investors: {len(comm_30plus)}")
print(f"  Investors with portfolio_novelty: "
      f"{base_df['portfolio_novelty'].notna().sum():,} / {len(base_df):,}")

gf_exit = (base_df.groupby(["green_focus","exit_status"])
           .size().unstack(fill_value=0))
print(f"\n  green_focus × exit_status:")
print(gf_exit.to_string())

base_df.reset_index().to_csv(f"{OUT_DIR}/base_population_final.csv", index=False)
print(f"\n  Saved base_population_final.csv")

# ─────────────────────────────────────────────────────────────────────
# 7. Lock preregistration
# ─────────────────────────────────────────────────────────────────────
sep("7. Lock preregistration — Family I")

qe_n      = counts.get("quiet_exit", 0)
stayer_n  = counts.get("stayer", 0)
full_n    = counts.get("full_exit", 0)
has_nov   = int(base_df["portfolio_novelty"].notna().sum())
qe_nov    = int((base_df["exit_status"]=="quiet_exit") & base_df["portfolio_novelty"].notna()).sum() \
            if False else int(base_df.loc[base_df["exit_status"]=="quiet_exit","portfolio_novelty"].notna().sum())
misclass_share = len(misclass) / n_qe * 100 if n_qe > 0 else 0

gvc_stayer_qe = int(gf_exit.loc["GREEN_VC", ["stayer","quiet_exit"]].sum()) if "GREEN_VC" in gf_exit.index else 0
trad_stayer_qe = int(gf_exit.loc["TRADITIONAL", ["stayer","quiet_exit"]].sum()) if "TRADITIONAL" in gf_exit.index else 0

qe_nov_mean  = base_df.loc[base_df["exit_status"]=="quiet_exit","portfolio_novelty"].mean()
sta_nov_mean = base_df.loc[base_df["exit_status"]=="stayer","portfolio_novelty"].mean()

prereg = f"""
## Family I — Quiet Exit  [LOCKED {datetime.date.today()}]

**Status: LOCKED**

---

### Data and windows

| Item | Value |
|---|---|
| PitchBook export date | 2026-08-26 (Last Updated Date, all rows) |
| Deals-file cutoff | {window_end.date()} (max Deal Date) |
| Window end used | {window_end.date()} (deals-file cutoff — earlier than PB export) |
| Last-2yr window | {window_2yr_start.date()} – {window_end.date()} |
| Last-5yr window | {window_5yr_start.date()} – {window_end.date()} |

### Matching and collision filter

| Step | N investors | Share of deal-data |
|---|---|---|
| Unique investors in deal data | {n_our:,} | 100% |
| Matched to PB (pre-filter) | {n_matched_raw:,} | {n_matched_raw/n_our*100:.1f}% |
| Excluded — single-word name | {n_excl_single:,} | — |
| Excluded — country mismatch | {n_excl_ctry:,} | — |
| Retained (clean matches) | {n_matched_clean:,} | {n_matched_clean/n_our*100:.1f}% |

### Definitions

| Term | Definition |
|---|---|
| active_overall | PB Last Investment Date ≥ {window_2yr_start.date()} |
| stayer | ≥1 climate deal in 2yr window (our data) |
| quiet_exit | active_overall AND 0 climate deals in 2yr window |
| full_exit | Last Investment Date < {window_2yr_start.date()} |
| Base population | clean-matched investors with ≥1 climate deal in 5yr window |
| intensity | prior_climate_deals / pb_total_inv (deals before {window_2yr_start.date()}) |
| portfolio_novelty | mean specter_novelty of ventures funded (pre-2yr window) |
| green_focus | GREEN_VC / ESG / TRADITIONAL (all_investors.csv) |

### Feasibility (Script 104, not pre-specified)

Base population: {total_base:,}
 — stayer    : {stayer_n:,} ({stayer_n/total_base*100:.1f}%)
 — quiet_exit: {qe_n:,} ({qe_n/total_base*100:.1f}%)
 — full_exit : {full_n:,} ({full_n/total_base*100:.1f}%)

Quiet-exit validation: {misclass_share:.1f}% of quiet exiters have a cleantech keyword
in their PB last investee company but that company is NOT in our sample
(potential missed coverage, not a classification error per se).

Portfolio novelty — stayers: {sta_nov_mean:.4f} | quiet_exit: {qe_nov_mean:.4f}

---

### Q1 — Snapshot (descriptive, no test)

**What share of climate-active investors went quiet in the last two years?**

Estimand: quiet_exit / base_population.
Report separately by green_focus and PB investor type.
No inferential test. Script 103/104 provide the numbers.

---

### Q2 — Portfolio novelty (pre-registered, directional)

**Did quiet exiters fund more novel ventures than stayers?**

Outcome: portfolio_novelty (mean specter_novelty across investor's pre-2yr climate portfolio).
Unit of observation: investor.
Comparison: quiet_exit vs stayer (exclude full_exit).
Model: OLS portfolio_novelty ~ quiet_exit_dummy + intensity_quintile_FEs.
   SE clustered by community (min-size ≥30) where available; HC3 otherwise.
Pre-specified sign: **positive** coefficient on quiet_exit_dummy.
   (Theoretical prediction: investors who exit climate backed more novel, harder-to-scale
   startups, consistent with the echo-chamber mechanism.)

Min cell: quiet_exit with portfolio_novelty = {qe_nov:,}.
Feasibility note: observed gap = {qe_nov_mean - sta_nov_mean:+.4f}. Power likely low.

---

### Q3 — GREEN_VC quiet-exit rate (pre-registered, directional)

**Within intensity quintiles, do GREEN_VC investors quiet-exit at higher rates than
TRADITIONAL investors?**

Outcome: binary quiet_exit (1) vs stayer (0); exclude full_exit.
Unit: investor.
Model: logit quiet_exit ~ green_focus_dummies + intensity_quintile_FEs.
   Reference: TRADITIONAL.
Pre-specified sign: **positive** coefficient on GREEN_VC dummy
   (GREEN_VC investors are specialised; when they exit climate, they exit entirely,
   whereas TRADITIONAL investors merely reduce exposure).
Comparison is within intensity quintiles (quintile FEs control for baseline commitment).

Min cells: GREEN_VC stayer+quiet_exit = {gvc_stayer_qe:,};
           TRADITIONAL stayer+quiet_exit = {trad_stayer_qe:,}.

---

### Q4 — Community concentration (pre-registered, two-sided)

**Are quiet exits concentrated in particular investor communities beyond chance?**

Test: permutation test (999 draws).
Statistic: max(community quiet_exit share) − overall quiet_exit share,
   across communities with ≥30 base-population investors.
   (N qualifying communities = {len(comm_30plus)}.)
Null: community membership does not predict quiet-exit status.
   p-value = share of permuted statistics ≥ observed.

---

### Holm correction

Applied across Q2, Q3, Q4 (three pre-registered tests).
Q1 is descriptive and excluded from Holm.

---

*Locked by Script 104 on {datetime.date.today()}. No further changes without a dated amendment.*
"""

print(prereg)

# Append locked section to preregistration.md
prereg_path = "output/preregistration.md"
with open(prereg_path, "a") as f:
    f.write(prereg)
print(f"\n  Appended locked section to {prereg_path}")

with open(f"{OUT_DIR}/prereg_family_I_LOCKED.md", "w") as f:
    f.write(prereg)
print(f"  Saved prereg_family_I_LOCKED.md")

print("\n" + "=" * 68)
print(f"SCRIPT 104 COMPLETE — PREREGISTRATION LOCKED {datetime.date.today()}")
print("=" * 68)
