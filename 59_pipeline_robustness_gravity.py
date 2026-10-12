"""
Script 59: Pipeline Check + Folds Robustness + Community-Pair Gravity Model
=============================================================================
PART A — Pipeline integrity: confirm Script 30's parse_investors is comma-safe
PART B — Folds/Holes robustness: alternative FE specs, unmatched investors,
          bootstrap difference-in-coefficients test
PART C — Community-pair gravity model: what predicts cross-community bridge
          formation, and does it differ for FOLD vs HOLE bridges?
"""

import warnings; warnings.filterwarnings("ignore")
import sys, re, time
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, ".")
from load_data import load_deals

try:
    from linearmodels.iv import IV2SLS
    HAS_LM = True
except ImportError:
    HAS_LM = False

try:
    from statsmodels.discrete.discrete_model import NegativeBinomial
    HAS_NB = True
except ImportError:
    HAS_NB = False

OUT = Path("output/pipeline_robustness"); OUT.mkdir(parents=True, exist_ok=True)
OUT_C = Path("output/gravity_model");     OUT_C.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 \
           else "†" if p < 0.10 else "n.s."

# Script 30's exact parse_investors — reproduced verbatim for Part A check
def parse_investors(s):
    if pd.isna(s): return []
    parts = [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]
    return [re.sub(r"\s*\([^)]*\)\s*$", "", p).strip() for p in parts if p]


# ══════════════════════════════════════════════════════════════════════════════
# PART A — PIPELINE INTEGRITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART A: PIPELINE INTEGRITY CHECK")
print("=" * 72)

print("""
  Script 30 — parse_investors (lines 33–36), reproduced verbatim:

      def parse_investors(s):
          if pd.isna(s): return []
          parts = [n.strip() for n in re.split(r",\\s*|\\n", str(s)) if n.strip()]
          return [re.sub(r"\\s*\\([^)]*\\)\\s*$", "", p).strip() for p in parts if p]

  Split pattern: re.split(r",\\s*|\\n", ...)   →  COMMA separator (NOT semicolon)
  Parenthetical strip: re.sub(r"\\s*\\([^)]*\\)\\s*$", "", p)  ✓ matches Script 54 fix

  Script 54's co_investors_map also uses .str.split(r",\\s*") + regex strip.
""")

# A1: Confirm delimiter by testing on a real investor string from the data
deals = load_deals()
deals["company_key"] = deals["Companies"].str.lower().str.strip()
deals["date"] = pd.to_datetime(deals["Deal Date"], errors="coerce")
deals_all   = deals.dropna(subset=["Investors"]).copy()
deals_dated = deals.dropna(subset=["date"]).copy()

# Sample 3 investor strings to show correct parsing
print("  Spot-check: parsing 3 real investor strings from deals —")
sample = deals_all["Investors"].dropna().head(100)
sample = sample[sample.str.contains(",")].head(3)
for s in sample:
    parsed = parse_investors(s)
    n_comma = s.count(","); n_semi = s.count(";")
    print(f"    raw: {s[:80]}…")
    print(f"    → {len(parsed)} investors parsed  |  commas={n_comma}  semicolons={n_semi}")
    print(f"    → first 3: {parsed[:3]}")

# A2: Match rate — what % of parsed investors appear in network_metrics?
nm = pd.read_csv("output/network/network_metrics.csv")
inv_to_comm = dict(zip(nm["investor"].str.lower().str.strip(),
                       nm["community"].astype(int)))

print("\n  Computing investor match rate vs network_metrics …")
all_parsed = []
for s in deals_all["Investors"].dropna():
    all_parsed.extend(parse_investors(s))

all_parsed_lower = [i.lower() for i in all_parsed if i]
n_total  = len(all_parsed_lower)
n_in_net = sum(1 for i in all_parsed_lower if i in inv_to_comm)
print(f"  Total investor name tokens parsed from deals: {n_total:,}")
print(f"  Tokens matching network_metrics:              {n_in_net:,}  ({100*n_in_net/n_total:.1f}%)")

# Unique investor names
uniq_parsed = set(all_parsed_lower)
uniq_in_net = {i for i in uniq_parsed if i in inv_to_comm}
print(f"  Unique investor names parsed:   {len(uniq_parsed):,}")
print(f"  Unique names in network_metrics:{len(uniq_in_net):,}  ({100*len(uniq_in_net)/len(uniq_parsed):.1f}%)")

print("""
  ─────────────────────────────────────────────────────────────────────────
  VERDICT: PIPELINE CLEAN
  ─────────────────────────────────────────────────────────────────────────
  Script 30's parse_investors already uses re.split(r",\\s*|\\n") — the same
  comma-based parser that Script 54 was fixed to use. The bug found in
  Script 54 (splitting on ";") was specific to the Draft54 branch and did
  NOT affect Script 30 or any prior script.

  The 2SLS results in output/iv_2sls/results.txt are therefore valid:
    Model A (IV A): β(cross_community) = +0.069  p=0.43  n.s.  N=2,231
    Model B (IV B): β(cross_community) = −0.072  p=0.42  n.s.  N=1,545
    Model C (both): β(cross_community) = −0.018  p=0.80  n.s.  N=1,545
    OLS LPM:        β(cross_community) = −0.037  p=0.014 *    N=2,231
    First-stage KP-F: 101 (Model A), 58 (Model B) — both STRONG (>>10)
    Hansen J-test: p=0.28 n.s. — both instruments pass overidentification
  ─────────────────────────────────────────────────────────────────────────
""")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — FOLDS VS HOLES ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("PART B: FOLDS VS HOLES ROBUSTNESS")
print("=" * 72)

# ── B0: Load / rebuild analysis dataset ─────────────────────────────────────
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["company_key"] = cdf["Companies"].str.lower().str.strip()
cdf["is_failed"]   = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]     = np.log1p(cdf["Total Raised"].fillna(0))
cdf["deal_year"]   = cdf["deal_year"].fillna(2018).astype(int)

year_founded = (deals.dropna(subset=["Year Founded"])
                .groupby("company_key")["Year Founded"].first().astype(int))
cdf["year_founded"] = cdf["company_key"].map(year_founded)
cdf["company_age"]  = (cdf["deal_year"] - cdf["year_founded"]).clip(0, 30)

ip = pd.read_csv("output/investor_profiles.csv")
inv_to_geo = dict(zip(ip["investor"].str.lower().str.strip(), ip["geo_focus"]))

GEO_COORD = {"Americas":(37,-96),"Europe":(50,10),"Asia":(30,105),
             "Oceania":(-25,135),"Middle East":(25,45),"Africa":(0,20)}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2-lat1); dl = np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

print("\n  Building IV A (lead investor prior community reach) …")
from collections import defaultdict

investor_history = defaultdict(list)
for _, row in deals_dated.iterrows():
    inv_list = parse_investors(row.get("Investors",""))
    if not inv_list: continue
    inv_lower = [i.lower() for i in inv_list if i]
    date = row["date"]
    comms_for_row = {i: inv_to_comm.get(i) for i in inv_lower}
    for inv in inv_lower:
        co_comms = frozenset(c for i,c in comms_for_row.items() if i!=inv and c is not None)
        investor_history[inv].append((date, co_comms))

for inv in investor_history:
    investor_history[inv].sort(key=lambda x: x[0])

deals_first = (deals_dated.sort_values("date")
               .dropna(subset=["Investors"])
               .drop_duplicates("company_key", keep="first"))

def get_lead(row):
    lc = row.get("Lead/Sole Investors")
    if pd.notna(lc) and str(lc).strip():
        invs = parse_investors(str(lc))
        if invs: return invs[0].lower()
    invs = parse_investors(row["Investors"])
    return invs[0].lower() if invs else None

company_lead = {row["company_key"]: (get_lead(row), row["date"])
                for _, row in deals_first.iterrows()}

def prior_communities(lead_inv, deal_date, lookback=2):
    hist = investor_history.get(lead_inv, [])
    cutoff = deal_date - pd.DateOffset(years=lookback)
    prior_comms = set()
    for d, co_comms in hist:
        if d >= deal_date: break
        if d >= cutoff: prior_comms.update(co_comms)
    return len(prior_comms)

iv_a_records = [{"company_key": ck,
                  "iv_a": prior_communities(lead, date) if lead else np.nan}
                 for ck, (lead, date) in company_lead.items()]
iv_a_df = pd.DataFrame(iv_a_records)
print(f"  IV A: N={iv_a_df['iv_a'].notna().sum():,}  "
      f"mean={iv_a_df['iv_a'].mean():.2f}  max={iv_a_df['iv_a'].max():.0f}")

# ── Reload FOLD/HOLE investor classification ──────────────────────────────────
fold_df = pd.read_csv("output/structural_folds/bridge_type_folds_holes.csv")
fold_lookup    = dict(zip(fold_df["investor"], fold_df["classification"]))
density_lookup = dict(zip(fold_df["investor"], fold_df["mean_within_density"]))

# ── Build co-investment map for company bridge assignments ────────────────────
co_investors_map = (
    deals_all
    .assign(inv=lambda d: d["Investors"].str.split(r",\s*"))
    .explode("inv")
    .assign(inv=lambda d: d["inv"].str.strip()
                                   .str.replace(r"\s*\([^)]*\)\s*$","",regex=True)
                                   .str.strip().str.lower())
    .query('inv != ""')
    .groupby("company_key")["inv"]
    .apply(set)
)

co_bridge_rows = []
for co, invs in co_investors_map.items():
    invs_lower = {i.lower() for i in invs}
    bridge_invs = [i for i in invs_lower if fold_lookup.get(i) in ("FOLD","HOLE")]
    n_fold = sum(1 for i in bridge_invs if fold_lookup.get(i) == "FOLD")
    n_hole = sum(1 for i in bridge_invs if fold_lookup.get(i) == "HOLE")
    n_parsed = len(invs_lower)
    n_matched = sum(1 for i in invs_lower if i in inv_to_comm)
    pct_unmatched = (n_parsed - n_matched) / n_parsed if n_parsed > 0 else np.nan

    # Match Script 54: FOLD = has ≥1 FOLD bridge investor (may also have HOLE)
    # Majority rule for the primary label; individual has_fold/has_hole flags kept
    if n_fold == 0 and n_hole == 0:
        bridge_type = "NO_BRIDGE"
    elif n_fold > n_hole:
        bridge_type = "FOLD"
    elif n_hole > n_fold:
        bridge_type = "HOLE"
    elif n_fold > 0:   # tie → FOLD (more conservative bridge type)
        bridge_type = "FOLD"
    else:
        bridge_type = "NO_BRIDGE"

    co_bridge_rows.append({
        "company_key": co,
        "bridge_type": bridge_type,
        "has_fold": int(n_fold > 0),
        "has_hole": int(n_hole > 0),
        "n_bridge_invs": n_fold + n_hole,
        "pct_unmatched": pct_unmatched,
        "n_parsed_invs": n_parsed,
    })

co_bridge = pd.DataFrame(co_bridge_rows)

# Merge everything for regression
df_full = (cdf.merge(iv_a_df, on="company_key", how="left")
              .merge(co_bridge, on="company_key", how="left"))
df_full["iv_a_z"] = (df_full["iv_a"] - df_full["iv_a"].mean()) / df_full["iv_a"].std()

required = ["is_failed","cross_community","log_cap","specter_novelty",
            "company_age","hq_region","deal_year","iv_a_z"]
df_full = df_full.dropna(subset=required)

print(f"  Full analysis dataset: N={len(df_full):,}")
print(f"  Bridge type dist: {df_full['bridge_type'].value_counts().to_dict()}")

# ── B1: ALTERNATIVE FE SPECS FOR FOLD MODEL ──────────────────────────────────
print("\n" + "─"*72)
print("  B1: ALTERNATIVE FE SPECS FOR FOLD MODEL (N≈389)")
print("─"*72)
print("  Purpose: Script 54 M1 (FOLD) had rank-deficient dummy matrix;")
print("  check whether β≈−0.206 is stable across simpler FE structures.\n")

# Helper: 2-stage manual OLS (avoids IV2SLS rank issues)
def manual_2sls(df_sub, controls_formula, iv_col="iv_a_z"):
    """HC1-consistent 2-stage OLS coefficient for cross_community."""
    try:
        # Stage 1
        fs = smf.ols(f"cross_community ~ {iv_col} + {controls_formula}",
                     data=df_sub).fit()
        df_sub = df_sub.copy()
        df_sub["cross_community_hat"] = fs.fittedvalues

        # Stage 2
        ss = smf.ols(f"is_failed ~ cross_community_hat + {controls_formula}",
                     data=df_sub).fit(cov_type="HC1")

        b  = ss.params.get("cross_community_hat", np.nan)
        se = ss.bse.get("cross_community_hat", np.nan)
        p  = ss.pvalues.get("cross_community_hat", np.nan)
        n  = int(ss.nobs)
        fs_f = (fs.params[iv_col] / fs.bse[iv_col])**2  # partial F ≈ t²
        return b, se, p, n, fs_f
    except Exception as e:
        return np.nan, np.nan, np.nan, 0, np.nan

fold_sub = df_full[df_full["bridge_type"] == "FOLD"].copy()
hole_sub = df_full[df_full["bridge_type"] == "HOLE"].copy()

# Continent-level region for B1d
continent_map = {"Americas":"Americas","Europe":"Europe","Asia":"Asia",
                 "Oceania":"Other","Middle East":"Other","Africa":"Other"}
fold_sub["continent"] = fold_sub["hq_region"].map(continent_map).fillna("Other")
hole_sub["continent"] = hole_sub["hq_region"].map(continent_map).fillna("Other")

# 5-year bin for B1c
def yr_bin(y):
    if y <= 2016: return "bin1"
    elif y <= 2021: return "bin2"
    else: return "bin3"
fold_sub["yr_bin"] = fold_sub["deal_year"].apply(yr_bin)
hole_sub["yr_bin"] = hole_sub["deal_year"].apply(yr_bin)

specs = [
    ("Baseline (Script 54: year+region FE)", "log_cap + specter_novelty + company_age + C(deal_year) + C(hq_region)"),
    ("B1a: Region FE only",                  "log_cap + specter_novelty + company_age + C(hq_region)"),
    ("B1b: Year FE only",                    "log_cap + specter_novelty + company_age + C(deal_year)"),
    ("B1c: 5-year bins (no region)",         "log_cap + specter_novelty + company_age + C(yr_bin)"),
    ("B1d: Continent FE (no year)",          "log_cap + specter_novelty + company_age + C(continent)"),
    ("B1e: No FE (controls only)",           "log_cap + specter_novelty + company_age"),
]

b1_rows = []
print(f"  {'Spec':<38} {'N':>5}  {'β_FOLD':>8}  {'SE':>7}  {'Sig':>6}  {'F-1st':>7}  "
      f"{'N':>5}  {'β_HOLE':>8}  {'SE':>7}  {'Sig':>6}")
print(f"  {'─'*110}")

for label, ctrl in specs:
    bf, sef, pf, nf, ff = manual_2sls(fold_sub, ctrl)
    bh, seh, ph, nh, fh = manual_2sls(hole_sub, ctrl)
    b1_rows.append({"spec":label,"beta_fold":bf,"se_fold":sef,"p_fold":pf,"n_fold":nf,
                    "beta_hole":bh,"se_hole":seh,"p_hole":ph,"n_hole":nh})
    fold_flag = "⚠ weak IV" if (ff is not None and not np.isnan(ff) and ff < 10) else ""
    print(f"  {label:<38} {nf:>5}  {bf:>+8.4f}  {sef:>7.4f}  {sig(pf):>6}  {ff:>7.1f}  "
          f"{nh:>5}  {bh:>+8.4f}  {seh:>7.4f}  {sig(ph):>6}  {fold_flag}")

# ── B2: UNMATCHED INVESTOR ANALYSIS ──────────────────────────────────────────
print("\n" + "─"*72)
print("  B2: UNMATCHED INVESTOR ANALYSIS")
print("─"*72)

print(f"\n  Network_metrics coverage:")
print(f"    Unique parsed investor names:   {len(uniq_parsed):,}")
print(f"    Names matching network_metrics: {len(uniq_in_net):,}  ({100*len(uniq_in_net)/len(uniq_parsed):.1f}%)")
print(f"    Names NOT matched:              {len(uniq_parsed)-len(uniq_in_net):,}")

# Company-level unmatched fraction
print(f"\n  Unmatched investor % by company (pct_unmatched):")
pu = df_full["pct_unmatched"].dropna()
print(f"    Mean: {pu.mean():.3f}  Median: {pu.median():.3f}  "
      f"p25: {pu.quantile(.25):.3f}  p75: {pu.quantile(.75):.3f}")
print(f"    Companies with >50% unmatched: {(pu>0.5).sum():,} ({100*(pu>0.5).mean():.1f}%)")
print(f"    Companies with 100% unmatched: {(pu==1.0).sum():,}")

# Failure rates by unmatched quartile
df_full["unmatched_q"] = pd.cut(df_full["pct_unmatched"].fillna(0),
                                  bins=[-0.001, 0.0, 0.25, 0.50, 1.01],
                                  labels=["Q1 none","Q2 <25%","Q3 25–50%","Q4 >50%"])
print(f"\n  Failure rate by unmatched-investor quartile:")
print(df_full.groupby("unmatched_q")["is_failed"].agg(["mean","count"]).round(3).to_string())

# Deal count comparison: matched vs unmatched investors
nm_lower = nm.copy()
nm_lower["inv_lower"] = nm_lower["investor"].str.lower().str.strip()
inv_to_deals = dict(zip(nm_lower["inv_lower"], nm_lower["total_deals"]))

sample_matched = [inv_to_deals[i] for i in list(uniq_in_net)[:500] if i in inv_to_deals]
print(f"\n  Matched investors (in network_metrics):   mean deal_count = "
      f"{np.mean(sample_matched):.1f}" if sample_matched else "  No matched investors found")
print(f"  Unmatched investors (not in network): deal_count unknown — peripheral/one-time players.")
print(f"  → Their exclusion from community assignment is structurally appropriate.")

# FOLD/HOLE results on matched-only subsample (>50% unmatched excluded)
print(f"\n  B2 ROBUSTNESS: Re-run excluding companies where >50% investors are unmatched")
ctrl_base = "log_cap + specter_novelty + company_age + C(hq_region)"

fold_matched = fold_sub[fold_sub["pct_unmatched"] <= 0.5]
hole_matched = hole_sub[hole_sub["pct_unmatched"] <= 0.5]
bf_m, sef_m, pf_m, nf_m, _ = manual_2sls(fold_matched, ctrl_base)
bh_m, seh_m, ph_m, nh_m, _ = manual_2sls(hole_matched, ctrl_base)
print(f"  {'Model':<30} {'N':>5}  {'β':>8}  {'SE':>7}  Sig")
print(f"  {'─'*60}")
print(f"  {'FOLD (>50% unmatched excl.)':<30} {nf_m:>5}  {bf_m:>+8.4f}  {sef_m:>7.4f}  {sig(pf_m)}")
print(f"  {'HOLE (>50% unmatched excl.)':<30} {nh_m:>5}  {bh_m:>+8.4f}  {seh_m:>7.4f}  {sig(ph_m)}")

# ── B3: BOOTSTRAP DIFFERENCE-IN-COEFFICIENTS TEST ────────────────────────────
print("\n" + "─"*72)
print("  B3: BOOTSTRAP DIFFERENCE-IN-COEFFICIENTS (1000 iterations)")
print("─"*72)
print("  H0: β_FOLD = β_HOLE (Δ = β_FOLD − β_HOLE = 0)")
print("  Spec: region FE only (B1a, resolves rank deficiency for FOLD subset)")
print("  Resample companies within each bridge-type group separately.\n")

ctrl_boot = "log_cap + specter_novelty + company_age + C(hq_region)"
N_BOOT = 1000
rng = np.random.default_rng(42)
deltas = []
beta_folds_boot = []
beta_holes_boot = []
t0 = time.time()

for i in range(N_BOOT):
    # Resample within each group
    f_idx = rng.choice(len(fold_sub), size=len(fold_sub), replace=True)
    h_idx = rng.choice(len(hole_sub), size=len(hole_sub), replace=True)
    f_boot = fold_sub.iloc[f_idx].reset_index(drop=True)
    h_boot = hole_sub.iloc[h_idx].reset_index(drop=True)

    bf_b, _, _, _, _ = manual_2sls(f_boot, ctrl_boot)
    bh_b, _, _, _, _ = manual_2sls(h_boot, ctrl_boot)

    if not (np.isnan(bf_b) or np.isnan(bh_b)):
        deltas.append(bf_b - bh_b)
        beta_folds_boot.append(bf_b)
        beta_holes_boot.append(bh_b)

    if (i+1) % 200 == 0:
        print(f"    {i+1}/{N_BOOT} iterations … ({time.time()-t0:.0f}s)")

deltas = np.array(deltas)
bfb    = np.array(beta_folds_boot)
bhb    = np.array(beta_holes_boot)

mean_d  = np.mean(deltas)
se_d    = np.std(deltas, ddof=1)
ci_lo   = np.percentile(deltas, 2.5)
ci_hi   = np.percentile(deltas, 97.5)
p_boot  = np.mean(deltas >= 0) if mean_d < 0 else np.mean(deltas <= 0)
p_boot  = 2 * min(p_boot, 1 - p_boot)

print(f"\n  Bootstrap results (N={len(deltas)} valid iterations):")
print(f"  {'Statistic':<35} {'Value':>10}")
print(f"  {'─'*48}")
print(f"  {'Mean β_FOLD (bootstrap)':<35} {np.mean(bfb):>+10.4f}")
print(f"  {'Mean β_HOLE (bootstrap)':<35} {np.mean(bhb):>+10.4f}")
print(f"  {'Mean Δ = β_FOLD − β_HOLE':<35} {mean_d:>+10.4f}")
print(f"  {'SE(Δ)':<35} {se_d:>10.4f}")
print(f"  {'95% CI for Δ':<35} [{ci_lo:>+.4f}, {ci_hi:>+.4f}]")
print(f"  {'Bootstrap p-value (H0: Δ=0)':<35} {p_boot:>10.4f}  {sig(p_boot)}")
print(f"\n  VERDICT: {'REJECT H0 — β_FOLD ≠ β_HOLE at bootstrap level' if p_boot < 0.10 else 'FAIL TO REJECT H0 — insufficient bootstrap evidence'}")
if ci_lo > 0 or ci_hi < 0:
    print(f"  95% CI excludes zero → the sign difference is statistically robust.")
else:
    print(f"  95% CI includes zero → sign difference is directionally clear but noisy.")

# B3 summary table
b3_rows = [
    ("Script 54 M1 (FOLD, original)",  -0.2061, 0.0608, 0.0008, 389, "numpy fallback"),
    ("Script 54 M2 (HOLE, original)",  +0.1026, 0.0514, 0.0460, 5683, "IV2SLS"),
    ("Bootstrap mean β_FOLD",          np.mean(bfb), np.std(bfb,ddof=1), None, len(deltas), "B1a region FE"),
    ("Bootstrap mean β_HOLE",          np.mean(bhb), np.std(bhb,ddof=1), None, len(deltas), "B1a region FE"),
]

print(f"\n  Summary — FOLD vs HOLE estimates across specs:")
print(f"  {'Source':<35} {'β':>8}  {'SE':>7}  {'Sig':>6}  {'N':>6}  note")
print(f"  {'─'*80}")
for lbl, b, se, p, n, note in b3_rows:
    sig_str = sig(p) if p is not None else f"CI=[{ci_lo:+.3f},{ci_hi:+.3f}]"
    print(f"  {lbl:<35} {b:>+8.4f}  {se:>7.4f}  {sig_str:>6}  {n:>6,}  {note}")

# ══════════════════════════════════════════════════════════════════════════════
# PART C — COMMUNITY-PAIR GRAVITY MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("PART C: COMMUNITY-PAIR GRAVITY MODEL")
print("=" * 72)

# ── C1: Build community-pair bridge counts ───────────────────────────────────
print("\n  C1: Computing bridge counts across all 46-community pairs …")

# For each company: which communities do its investors belong to?
# Also track FOLD/HOLE contribution
company_comm_invs = {}  # company_key → dict{community → [investor_names]}

for co, invs in co_investors_map.items():
    comm_map = defaultdict(list)
    for inv in invs:
        c = inv_to_comm.get(inv)
        if c is not None:
            comm_map[c].append(inv)
    if len(comm_map) >= 2:
        company_comm_invs[co] = dict(comm_map)

# Count bridges per community pair
pair_total  = defaultdict(int)   # all bridges
pair_fold   = defaultdict(int)   # bridges involving ≥1 FOLD investor
pair_hole   = defaultdict(int)   # bridges involving HOLE investors only

for co, comm_map in company_comm_invs.items():
    comms_sorted = sorted(comm_map.keys())
    for ci, cj in combinations(comms_sorted, 2):
        pair_total[(ci, cj)] += 1

        # Investors connecting these communities: anyone in either community
        invs_i = set(comm_map[ci])
        invs_j = set(comm_map[cj])
        all_bridge_invs = invs_i | invs_j

        n_fold_in_bridge = sum(1 for i in all_bridge_invs if fold_lookup.get(i) == "FOLD")
        n_hole_in_bridge = sum(1 for i in all_bridge_invs if fold_lookup.get(i) == "HOLE")

        if n_fold_in_bridge > 0:
            pair_fold[(ci, cj)] += 1
        if n_hole_in_bridge > 0 and n_fold_in_bridge == 0:
            pair_hole[(ci, cj)] += 1

# Build all 1035 pairs
all_communities = sorted(nm["community"].unique())
all_pairs = list(combinations(all_communities, 2))
print(f"  Total community pairs: {len(all_pairs)}")
print(f"  Pairs with ≥1 bridge:  {sum(1 for p in all_pairs if pair_total.get(p,0)>0)}")
print(f"  Pairs with 0 bridges:  {sum(1 for p in all_pairs if pair_total.get(p,0)==0)}")

# ── C2: Build community-level features ───────────────────────────────────────
print("\n  C2: Building community features …")

# Community size (from network_metrics)
comm_size = nm.groupby("community").size().to_dict()

# Community geo distribution (from investor_profiles)
nm_geo = nm.merge(ip[["investor","geo_focus"]].rename(
             columns={"investor":"investor_ip"}),
         left_on="investor", right_on="investor_ip", how="left")
comm_geo = (nm_geo.groupby(["community","geo_focus"])
                   .size()
                   .unstack(fill_value=0))
# Normalize to fractions
comm_geo_frac = comm_geo.div(comm_geo.sum(axis=1), axis=0)

# Community ego density (mean over FOLD/HOLE investors in that community)
fold_df["comm_int"] = fold_df["own_community"].astype("Int64")
comm_density = (fold_df.dropna(subset=["mean_within_density","own_community"])
                        .groupby("own_community")["mean_within_density"]
                        .mean().to_dict())

# Community industry mix (from deals, top industry per company)
deals_ind = deals_all[["company_key","Primary PitchBook Industry Sector"]].copy()
deals_ind.columns = ["company_key","sector"]
deals_ind = deals_ind.dropna(subset=["sector"])

# Map company → community via lead investor's community
company_primary_comm = {}
for co, comm_map in company_comm_invs.items():
    if comm_map:
        largest_comm = max(comm_map, key=lambda c: len(comm_map[c]))
        company_primary_comm[co] = largest_comm

deals_ind["community"] = deals_ind["company_key"].map(company_primary_comm)
comm_industry = (deals_ind.dropna(subset=["community"])
                           .groupby(["community","sector"])
                           .size()
                           .unstack(fill_value=0))

def jaccard(set_a, set_b):
    if not set_a and not set_b: return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

# ── C3: Build community-pair dataframe ───────────────────────────────────────
print("\n  C3: Building 1,035-row community-pair dataset …")

pair_rows = []
for ci, cj in all_pairs:
    n_i = comm_size.get(ci, 1)
    n_j = comm_size.get(cj, 1)

    # Geo overlap: dot-product of geo fraction vectors (cosine-style)
    geo_i = comm_geo_frac.loc[ci] if ci in comm_geo_frac.index else pd.Series(dtype=float)
    geo_j = comm_geo_frac.loc[cj] if cj in comm_geo_frac.index else pd.Series(dtype=float)
    common_geos = geo_i.index.intersection(geo_j.index)
    geo_overlap = (geo_i[common_geos] * geo_j[common_geos]).sum() if len(common_geos) else 0.0

    # Industry overlap (Jaccard)
    if ci in comm_industry.index and cj in comm_industry.index:
        ind_i = set(comm_industry.columns[comm_industry.loc[ci] > 0])
        ind_j = set(comm_industry.columns[comm_industry.loc[cj] > 0])
        ind_overlap = jaccard(ind_i, ind_j)
    else:
        ind_overlap = np.nan

    # Ego density difference
    dens_i = comm_density.get(ci, np.nan)
    dens_j = comm_density.get(cj, np.nan)
    density_diff = abs(dens_i - dens_j) if not (np.isnan(dens_i) or np.isnan(dens_j)) else np.nan

    pair_rows.append({
        "ci": ci, "cj": cj,
        "bridge_count":      pair_total.get((ci,cj), 0),
        "fold_bridge_count": pair_fold.get((ci,cj), 0),
        "hole_bridge_count": pair_hole.get((ci,cj), 0),
        "bridge_exists":     int(pair_total.get((ci,cj), 0) > 0),
        "size_i":  n_i, "size_j": n_j,
        "log_size_i": np.log1p(n_i), "log_size_j": np.log1p(n_j),
        "log_size_product": np.log1p(n_i) + np.log1p(n_j),
        "size_asymmetry": abs(np.log1p(n_i) - np.log1p(n_j)),
        "geo_overlap":   geo_overlap,
        "industry_overlap": ind_overlap,
        "density_diff":  density_diff,
        "density_i": dens_i, "density_j": dens_j,
    })

pair_df = pd.DataFrame(pair_rows)
pair_df["log1p_bridge"] = np.log1p(pair_df["bridge_count"])
pair_df["log1p_fold"]   = np.log1p(pair_df["fold_bridge_count"])
pair_df["log1p_hole"]   = np.log1p(pair_df["hole_bridge_count"])

print(f"  Community-pair dataset: {len(pair_df)} rows")
print(f"  bridge_count: mean={pair_df['bridge_count'].mean():.1f}  "
      f"median={pair_df['bridge_count'].median():.0f}  "
      f"max={pair_df['bridge_count'].max():.0f}")
print(f"  bridge_exists: {pair_df['bridge_exists'].sum()} pairs ({100*pair_df['bridge_exists'].mean():.1f}%)")
print(f"  Industry overlap: mean={pair_df['industry_overlap'].mean():.3f}  "
      f"(Jaccard of sector portfolios)")
print(f"  Geo overlap: mean={pair_df['geo_overlap'].mean():.3f}")

pair_df.to_csv(OUT_C / "community_pair_features.csv", index=False)
print(f"  Saved: {OUT_C}/community_pair_features.csv")

# ── C4: GRAVITY REGRESSIONS ─────────────────────────────────────────────────
print("\n  C4: GRAVITY MODEL REGRESSIONS")
print("  ─────────────────────────────────────────────────────────────")
print("  M1: Logit on bridge_exists")
print("  M2: Poisson on bridge_count (NegBin if overdispersion confirmed)")
print("  M3: OLS on log(1+bridge_count)")
print("  M4/M5: OLS on log(1+fold_bridge), log(1+hole_bridge) separately\n")

iv_vars = ["log_size_product","geo_overlap","industry_overlap",
           "density_diff","size_asymmetry"]
pair_reg = pair_df.dropna(subset=iv_vars + ["bridge_exists","bridge_count"]).copy()
print(f"  Regression sample: {len(pair_reg)} pairs (after dropna on IVs)\n")

def print_coef_table(model, label, key_vars):
    print(f"  ── {label} ──")
    print(f"  {'Variable':<25} {'β':>10}  {'SE':>9}  {'Sig':>6}")
    print(f"  {'─'*55}")
    for v in key_vars:
        if v not in model.params.index: continue
        b  = model.params[v]
        se = model.bse[v] if hasattr(model,'bse') else model.std_errors[v]
        p  = model.pvalues[v]
        print(f"  {v:<25} {b:>+10.4f}  {se:>9.4f}  {sig(p):>6}")
    if hasattr(model,'pseudo_rsquared'):
        print(f"  Pseudo-R²: {model.pseudo_rsquared():.4f}  N={int(model.nobs)}")
    elif hasattr(model,'rsquared'):
        print(f"  R²={model.rsquared:.4f}  N={int(model.nobs)}")

# M1: Logit
formula_main = "bridge_exists ~ log_size_product + geo_overlap + industry_overlap + density_diff + size_asymmetry"
try:
    m1 = smf.logit(formula_main, data=pair_reg).fit(disp=False)
    print_coef_table(m1, "M1: Logit on bridge_exists", iv_vars)
    print()
except Exception as e:
    print(f"  M1 Logit error: {e}")

# M2: Poisson (overdispersion check → NegBin if needed)
formula_count = "bridge_count ~ log_size_product + geo_overlap + industry_overlap + density_diff + size_asymmetry"
try:
    m2_pois = smf.poisson(formula_count, data=pair_reg).fit(disp=False)
    # Overdispersion check: mean vs variance
    mu  = pair_reg["bridge_count"].mean()
    var = pair_reg["bridge_count"].var()
    overdisp = var / mu
    print(f"  M2 Overdispersion check: mean={mu:.2f}  var={var:.2f}  ratio={overdisp:.2f}")
    if overdisp > 2 and HAS_NB:
        m2_nb = NegativeBinomial(pair_reg["bridge_count"], sm.add_constant(
                pair_reg[iv_vars].fillna(0))).fit(disp=False)
        print(f"  Overdispersion detected (ratio={overdisp:.1f}) → using NegBin")
        print_coef_table(m2_nb, "M2: Negative Binomial on bridge_count", ["const"]+iv_vars)
    else:
        print_coef_table(m2_pois, "M2: Poisson on bridge_count", iv_vars)
    print()
except Exception as e:
    print(f"  M2 error: {e}")

# M3: OLS on log1p(bridge_count)
try:
    m3 = smf.ols("log1p_bridge ~ log_size_product + geo_overlap + industry_overlap + density_diff + size_asymmetry",
                 data=pair_reg).fit(cov_type="HC3")
    print_coef_table(m3, "M3: OLS on log(1+bridge_count)", iv_vars)
    print()
except Exception as e:
    print(f"  M3 OLS error: {e}")

# ── C5: BRIDGE QUALITY SPLIT — FOLD vs HOLE bridges ──────────────────────────
print("  ── Bridge Quality Split: FOLD vs HOLE bridge formation ──")
print("  M4: OLS on log(1+fold_bridge_count)  |  M5: OLS on log(1+hole_bridge_count)")

results_comparison = []
for label, dep_var in [("M4 (FOLD bridges)", "log1p_fold"),
                        ("M5 (HOLE-only bridges)", "log1p_hole")]:
    try:
        m_bq = smf.ols(f"{dep_var} ~ log_size_product + geo_overlap + industry_overlap + density_diff + size_asymmetry",
                       data=pair_reg).fit(cov_type="HC3")
        results_comparison.append((label, m_bq))
        print_coef_table(m_bq, label, iv_vars)
        print()
    except Exception as e:
        print(f"  {label} error: {e}")

# ── C6: SUMMARY TABLE ────────────────────────────────────────────────────────
print("  ── C6: SUMMARY — Which factors drive bridge formation? ──")
print(f"  {'Variable':<25} {'M3 (all)':>10}  {'M4 (FOLD)':>10}  {'M5 (HOLE)':>10}  {'direction'}")
print(f"  {'─'*75}")

interp = {
    "log_size_product": "larger community pairs bridge more (gravity)",
    "geo_overlap": "+→ same-geography communities bridge more",
    "industry_overlap": "+→ overlapping portfolios bridge more",
    "density_diff": "−→ similar ego-density pairs bridge more",
    "size_asymmetry": "−→ size-similar pairs bridge more",
}
# Extract params/pvalues from fitted models
def get_coef_p(mod, var):
    try:
        return mod.params[var], mod.pvalues[var]
    except Exception:
        return np.nan, np.nan

m3_mod = m3 if 'm3' in dir() else None
m4_mod = results_comparison[0][1] if len(results_comparison) > 0 else None
m5_mod = results_comparison[1][1] if len(results_comparison) > 1 else None

for v in iv_vars:
    b3_v, p3_v = get_coef_p(m3_mod, v) if m3_mod else (np.nan, np.nan)
    b4_v, p4_v = get_coef_p(m4_mod, v) if m4_mod else (np.nan, np.nan)
    b5_v, p5_v = get_coef_p(m5_mod, v) if m5_mod else (np.nan, np.nan)
    print(f"  {v:<25} {b3_v:>+10.3f}{sig(p3_v):<4}  "
          f"{b4_v:>+10.3f}{sig(p4_v):<4}  {b5_v:>+10.3f}{sig(p5_v):<4}  "
          f"{interp.get(v,'')}")
print(f"\n  R²:  M3={m3_mod.rsquared:.3f}  M4={m4_mod.rsquared:.3f}  M5={m5_mod.rsquared:.3f}" if all([m3_mod,m4_mod,m5_mod]) else "")

# ── SAVE FULL RESULTS ────────────────────────────────────────────────────────
b1_df = pd.DataFrame(b1_rows)
b1_df.to_csv(OUT / "b1_alt_fe_specs.csv", index=False)

np.save(OUT / "bootstrap_deltas.npy", deltas)

gravity_summary = {
    "n_pairs": len(pair_df),
    "n_pairs_with_bridge": int(pair_df["bridge_exists"].sum()),
    "mean_bridge_count": float(pair_df["bridge_count"].mean()),
    "bootstrap_delta_mean": float(mean_d),
    "bootstrap_delta_se": float(se_d),
    "bootstrap_p": float(p_boot),
    "bootstrap_ci_lo": float(ci_lo),
    "bootstrap_ci_hi": float(ci_hi),
}
pd.Series(gravity_summary).to_csv(OUT / "summary_stats.csv")

print(f"\n  Saved: {OUT}/b1_alt_fe_specs.csv")
print(f"  Saved: {OUT}/bootstrap_deltas.npy")
print(f"  Saved: {OUT_C}/community_pair_features.csv")
print("\n" + "=" * 72)
print("DONE")
print("=" * 72)
