"""
Script 41: Selection vs. Treatment — Cross-Community Causal Tests
=================================================================

5 tests addressing the round-2 selection concern:
  1. Round-1 vs later cross-community (separate effects)
  2. Pre-bridge performance comparison (observable selection)
  3. IV restricted to sequential subsample
  4. What the bridge community provides (capital, type, rounds, timing)
  5. Bridge community follow-on pipeline
"""

import warnings; warnings.filterwarnings("ignore")
import sys, itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from linearmodels.iv import IV2SLS

sys.path.insert(0, ".")
from load_data import load_deals, load_companies

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data…")
cdf  = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]   = cdf["log_capital"]
cdf["nrounds"]   = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

net = pd.read_csv("output/network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm  = net.set_index("investor")["community"].to_dict()
inv_to_type  = net.set_index("investor")["investor_type"].to_dict()
comm_size    = net.groupby("community").size().to_dict()

deals = load_deals()
deals["co"]   = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")

# Expand deals to investor level
print("Expanding deals…")
d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))
d_exp["community"] = d_exp["inv"].map(inv_to_comm)
d_exp["inv_type"]  = d_exp["inv"].map(inv_to_type)
d_exp_comm = d_exp.dropna(subset=["community"])

# ── Build round-level community structure ────────────────────────────────────
print("Building round-level community structure…")
cross_cos   = set(cdf[cdf["cross_community"] == 1]["co"])
within_cos  = set(cdf[cdf["cross_community"] == 0]["co"])

co_timing = {}   # co → {year → set(communities)}
for co, grp in d_exp_comm.dropna(subset=["Year"]).groupby("co"):
    co_timing[co] = grp.groupby("Year")["community"].apply(set).to_dict()

def classify_timing(co):
    """Returns (bridge_in_r1, bridge_later, n_comms_r1, first_bridge_round)."""
    yt = co_timing.get(co, {})
    if not yt:
        return None
    sorted_years = sorted(yt.keys())
    # Assign round index 1,2,3…
    yr_to_rnd = {yr: i+1 for i, yr in enumerate(sorted_years)}
    # Track when each community first appears
    seen, comm_first = set(), {}
    for yr in sorted_years:
        rnd = yr_to_rnd[yr]
        for c in yt[yr]:
            if c not in comm_first:
                comm_first[c] = rnd
        seen |= yt[yr]
    if len(comm_first) < 2:
        return None
    sorted_entries = sorted(comm_first.items(), key=lambda x: x[1])
    first_bridge_round = sorted_entries[1][1]
    comms_r1 = yt.get(sorted_years[0], set())
    bridge_in_r1 = len(comms_r1) >= 2
    return {
        "bridge_in_r1":      bridge_in_r1,
        "bridge_later":      not bridge_in_r1,
        "first_bridge_round": first_bridge_round,
        "n_comms_r1":        len(comms_r1),
        "n_comms_total":     len(comm_first),
        "n_rounds_obs":      len(sorted_years),
        "year_r1":           sorted_years[0],
        "home_comm":         sorted_entries[0][0],
        "bridge_comm":       sorted_entries[1][0],
    }

timing_records = []
for co in cross_cos:
    t = classify_timing(co)
    if t:
        t["co"] = co
        timing_records.append(t)

timing_df = pd.DataFrame(timing_records)
print(f"  Cross-community with timing data: {len(timing_df):,}")
print(f"  Bridge in round 1 (co-lead):      {timing_df['bridge_in_r1'].sum():,}  "
      f"({timing_df['bridge_in_r1'].mean():.1%})")
print(f"  Bridge in round 2+:               {timing_df['bridge_later'].sum():,}  "
      f"({timing_df['bridge_later'].mean():.1%})")

# Merge back to cdf
cdf2 = cdf.merge(timing_df[["co","bridge_in_r1","bridge_later","first_bridge_round",
                              "n_rounds_obs","year_r1","home_comm","bridge_comm"]],
                 on="co", how="left")
cdf2["bridge_in_r1"]  = cdf2["bridge_in_r1"].fillna(False).astype(float)
cdf2["bridge_later"]  = cdf2["bridge_later"].fillna(False).astype(float)
cdf2["within_only"]   = ((cdf2["cross_community"] == 0)).astype(float)

# ── TEST 1: Round-1 vs Later Cross-Community ─────────────────────────────────
print()
print("="*72)
print("TEST 1: ROUND-1 vs LATER CROSS-COMMUNITY EFFECTS")
print("="*72)

sub1 = cdf2.dropna(subset=["log_cap","deal_year"]).copy()

m_base  = smf.ols(f"is_failed ~ cross_community + {CTRL}", data=sub1).fit(cov_type="HC1")
m_split = smf.ols(f"is_failed ~ bridge_in_r1 + bridge_later + {CTRL}", data=sub1).fit(cov_type="HC1")

b_r1   = m_split.params["bridge_in_r1"]
p_r1   = m_split.pvalues["bridge_in_r1"]
b_lat  = m_split.params["bridge_later"]
p_lat  = m_split.pvalues["bridge_later"]

n_r1   = int(cdf2["bridge_in_r1"].sum())
n_lat  = int(cdf2["bridge_later"].sum())
n_wc   = int((cdf2["cross_community"]==0).sum())

print(f"\n  Reference: within-community only  (N={n_wc:,}  fail={cdf[cdf['cross_community']==0]['is_failed'].mean():.1%})")
print(f"\n  β_cross_r1   (co-lead at founding):  {b_r1:>+.4f}  {sig(p_r1)}  "
      f"(N={n_r1:,}  fail={cdf2[cdf2['bridge_in_r1']==1]['is_failed'].mean():.1%})")
print(f"  β_cross_later (sequential round 2+): {b_lat:>+.4f}  {sig(p_lat)}  "
      f"(N={n_lat:,}  fail={cdf2[cdf2['bridge_later']==1]['is_failed'].mean():.1%})")

# Test if coefficients are significantly different from each other
cov_m = m_split.cov_params()
se_diff = np.sqrt(float(cov_m.loc["bridge_in_r1","bridge_in_r1"]) +
                  float(cov_m.loc["bridge_later","bridge_later"]) -
                  2*float(cov_m.loc["bridge_in_r1","bridge_later"]))
t_diff  = (b_r1 - b_lat) / se_diff
p_diff  = 2 * stats.t.sf(abs(t_diff), df=m_split.df_resid)
print(f"\n  Wald test (β_r1 = β_later): t={t_diff:.3f}  p={p_diff:.3f}  {sig(p_diff)}")

if p_r1 < 0.05 and p_lat < 0.05:
    verdict1 = "BOTH significant → diversity helps at BOTH founding AND follow-on stages"
elif p_r1 < 0.10 and not (p_lat < 0.05):
    verdict1 = "Only round-1 marginally significant → founding diversity is the driver"
elif p_lat < 0.05 and not (p_r1 < 0.10):
    verdict1 = "Only round 2+ significant → selection at follow-on may explain effect"
else:
    verdict1 = "Neither individually significant after split (power issue)"
print(f"  → {verdict1}")

# ── TEST 2: PRE-BRIDGE PERFORMANCE ───────────────────────────────────────────
print()
print("="*72)
print("TEST 2: PRE-BRIDGE PERFORMANCE (Observable Selection)")
print("="*72)

# For sequential (bridge_later) companies vs. within-community:
# Compare round-1 characteristics
# Round 1 = first year of deals for a company

def get_r1_stats(co):
    sub = d_exp[d_exp["co"] == co].dropna(subset=["Year"])
    if sub.empty:
        return {}
    yr1 = sub["Year"].min()
    r1_deals = sub[sub["Year"] == yr1]
    # Syndicate size = unique investors in r1
    invs_r1 = r1_deals["inv"].unique()
    n_inv_r1 = len(invs_r1)
    # Capital raised in r1
    cap_r1 = (deals[(deals["co"]==co) & (deals["Year"]==yr1)]["Deal Size (USD M)"]
              .dropna().sum())
    # Deal type in r1
    dtype_r1 = (deals[(deals["co"]==co) & (deals["Year"]==yr1)]["Deal Type"]
                .mode().iloc[0] if len(deals[(deals["co"]==co) & (deals["Year"]==yr1)]) else "Unknown")
    # Has any IVC in r1?
    has_ivc_r1 = any(inv_to_type.get(inv) == "IVC" for inv in invs_r1)
    return {
        "co": co,
        "n_inv_r1": n_inv_r1,
        "cap_r1": cap_r1 if cap_r1 > 0 else np.nan,
        "has_ivc_r1": float(has_ivc_r1),
        "year_r1": yr1,
    }

print("  Computing round-1 characteristics for sequential vs within-community companies…")
target_cos = list(timing_df[timing_df["bridge_later"]==True]["co"]) + \
             list(cdf[cdf["cross_community"]==0]["co"])

r1_records = []
for co in target_cos[:5000]:   # cap at 5000 for speed
    r = get_r1_stats(co)
    if r:
        r1_records.append(r)

r1_df = pd.DataFrame(r1_records)
r1_df = r1_df.merge(cdf[["co","cross_community","is_failed"]], on="co", how="left")

seq_mask = r1_df["co"].isin(timing_df[timing_df["bridge_later"]==True]["co"])
r1_df["is_sequential"] = seq_mask.astype(float)

print(f"\n  {'Characteristic':<30}  {'Sequential (bridge R2+)':>22}  {'Within-community':>18}  {'p-value':>10}")
print(f"  {'-'*80}")

metrics = [
    ("Round-1 syndicate size",  "n_inv_r1"),
    ("Round-1 capital (USD M)", "cap_r1"),
    ("Has IVC in round 1",      "has_ivc_r1"),
]
all_same = True
for label, col in metrics:
    seq_vals  = r1_df[r1_df["is_sequential"]==1][col].dropna()
    wc_vals   = r1_df[r1_df["is_sequential"]==0][col].dropna()
    if len(seq_vals) < 10 or len(wc_vals) < 10:
        continue
    _, pval = stats.mannwhitneyu(seq_vals, wc_vals, alternative="two-sided")
    s_mean = seq_vals.mean(); w_mean = wc_vals.mean()
    direction = "↑" if s_mean > w_mean else "↓"
    print(f"  {label:<30}  {s_mean:>22.2f}  {w_mean:>18.2f}  {pval:>8.4f} {sig(pval)} {direction}")
    if pval < 0.05:
        all_same = False

if all_same:
    verdict2 = "Pre-bridge characteristics IDENTICAL → selection on observables is NOT driving it"
else:
    verdict2 = "Pre-bridge characteristics DIFFER → some selection on observables"
print(f"\n  → {verdict2}")

# ── TEST 3: IV ON SEQUENTIAL SUBSAMPLE ───────────────────────────────────────
print()
print("="*72)
print("TEST 3: IV (2SLS) RESTRICTED TO SEQUENTIAL SUBSAMPLE")
print("="*72)

# IV: min community size instrument — rebuild for sequential subsample
# Instrument: companies whose investors' smallest community is smaller → more likely
# to reach cross-community follow-on. Use same IV as main paper: min_community_size.

print("  Building min_community_size IV…")
# Get all investors per company (from deals)
co_inv_sets = (d_exp_comm.groupby("co")["community"]
               .apply(lambda x: list(set(x))).to_dict())

iv_records = []
for co in cdf2["co"]:
    comms = co_inv_sets.get(co, [])
    if comms:
        min_cs = min(comm_size.get(c, 9999) for c in comms)
        iv_records.append({"co": co, "min_comm_size": min_cs})

iv_df = pd.DataFrame(iv_records)
sub_iv = cdf2.merge(iv_df, on="co", how="left")
sub_iv["log_min_cs"] = np.log1p(sub_iv["min_comm_size"].fillna(sub_iv["min_comm_size"].median()))

# Full sample IV first (sanity check)
sub_full = sub_iv.dropna(subset=["log_cap","deal_year","log_min_cs"]).copy()
exog_cols = ["log_cap","nrounds","specter_novelty"]
endog     = "cross_community"
instr     = "log_min_cs"

# Country FE dummies
sub_full = pd.get_dummies(sub_full, columns=["country_fe","deal_year"], drop_first=True)
fe_cols  = [c for c in sub_full.columns if c.startswith("country_fe_") or c.startswith("deal_year_")]
all_exog = exog_cols + fe_cols

verdict3 = "IV skipped"
try:
    from linearmodels.iv import IV2SLS as IVOLS
    import statsmodels.api as sm

    def run_iv(df, label):
        df = df.copy()
        # Absorb FE via within-transformation (demean by country×year group)
        df["g"] = df["country_fe"].astype(str) + "_" + df["deal_year"].astype(str)
        for col in ["is_failed","cross_community","log_cap","nrounds","specter_novelty","log_min_cs"]:
            gm = df.groupby("g")[col].transform("mean")
            df[col+"_dm"] = df[col] - gm
        # 2SLS via statsmodels manual
        X_fs = sm.add_constant(df[["log_cap_dm","nrounds_dm","specter_novelty_dm","log_min_cs_dm"]].astype(float))
        fs   = sm.OLS(df["cross_community_dm"].astype(float), X_fs).fit()
        kpf  = (fs.params["log_min_cs_dm"] / fs.bse["log_min_cs_dm"])**2
        df["cc_hat"] = fs.fittedvalues
        X_ss = sm.add_constant(df[["cc_hat","log_cap_dm","nrounds_dm","specter_novelty_dm"]].astype(float))
        ss   = sm.OLS(df["is_failed_dm"].astype(float), X_ss).fit(cov_type="HC1")
        b    = ss.params["cc_hat"]; p = ss.pvalues["cc_hat"]
        print(f"  {label} — First-stage F: {kpf:.1f}  |  β_cross_IV: {b:>+.4f}  {sig(p)}  N={len(df):,}")
        return b, p, kpf

    sub_iv_num = sub_iv.dropna(subset=["log_cap","deal_year","log_min_cs","is_failed","cross_community","specter_novelty"]).copy()
    b_full, p_full, kpf_full = run_iv(sub_iv_num, "Full sample")

    sub_seq_num = sub_iv_num[sub_iv_num["bridge_later"]==1].copy()
    b_seq, p_seq, kpf_seq = run_iv(sub_seq_num, "Sequential only")

    if p_seq < 0.05:
        verdict3 = "IV significant on sequential subsample → forced cross-community STILL helps"
    elif p_seq < 0.10:
        verdict3 = "IV marginally significant on sequential → suggestive treatment effect"
    else:
        verdict3 = f"IV n.s. on sequential subsample (F={kpf_seq:.0f}, power limited by N={len(sub_seq_num):,})"
except Exception as e:
    print(f"  IV failed: {e}")

# Sequential subsample IV
sub_seq = sub_iv[sub_iv["bridge_later"]==1].dropna(
    subset=["log_cap","deal_year","log_min_cs"]).copy()
sub_seq = pd.get_dummies(sub_seq, columns=["country_fe","deal_year"], drop_first=True)
for col in sub_seq.columns:
    sub_seq[col] = pd.to_numeric(sub_seq[col], errors="coerce")
sub_seq = sub_seq.dropna(subset=["is_failed", endog, instr])
fe_cols_s = [c for c in sub_seq.columns if c.startswith("country_fe_") or c.startswith("deal_year_")]
all_exog_s = [c for c in exog_cols if c in sub_seq.columns] + fe_cols_s

try:
    X_fs_s = sm.add_constant(sub_seq[[c for c in all_exog_s if c in sub_seq.columns] + [instr]].astype(float))
    fs_s   = sm.OLS(sub_seq[endog].astype(float), X_fs_s).fit(cov_type="HC1")
    kp_f_s = (fs_s.params.get(instr, 0) / fs_s.bse.get(instr, 1))**2
    sub_seq["cross_hat"] = fs_s.fittedvalues
    X_ss_s = sm.add_constant(sub_seq[["cross_hat"] + [c for c in all_exog_s if c in sub_seq.columns]])
    ss_s   = sm.OLS(sub_seq["is_failed"], X_ss_s).fit(cov_type="HC1")
    b_iv_seq = ss_s.params.get("cross_hat", np.nan)
    p_iv_seq = ss_s.pvalues.get("cross_hat", 1.0)
    print(f"\n  Sequential subsample — First-stage F: {kp_f_s:.1f}")
    print(f"  IV β_cross (sequential only): {b_iv_seq:>+.4f}  {sig(p_iv_seq)}  N={len(sub_seq):,}")
    if p_iv_seq < 0.05:
        verdict3 = "IV significant on sequential subsample → forced sequential cross-community STILL helps"
    elif p_iv_seq < 0.10:
        verdict3 = "IV marginally significant on sequential → suggestive treatment effect"
    else:
        verdict3 = "IV not significant on sequential subsample (power limited by smaller N)"
    print(f"  → {verdict3}")
except Exception as e:
    print(f"  Sequential IV failed: {e}")

# ── TEST 4: WHAT DOES THE BRIDGE PROVIDE? ────────────────────────────────────
print()
print("="*72)
print("TEST 4: WHAT DOES THE BRIDGE COMMUNITY PROVIDE?")
print("="*72)

# For companies with bridge at round 2 specifically
bridge_r2_cos = set(timing_df[timing_df["first_bridge_round"]==2]["co"])
within_cos_set = set(cdf[cdf["cross_community"]==0]["co"])

def get_round_stats(co, round_n):
    """Stats for a specific round (by year rank)."""
    yt = co_timing.get(co, {})
    if not yt:
        return {}
    sorted_years = sorted(yt.keys())
    if round_n > len(sorted_years):
        return {}
    yr = sorted_years[round_n - 1]
    r_deals = deals[(deals["co"]==co) & (deals["Year"]==yr)]
    cap     = r_deals["Deal Size (USD M)"].dropna().sum()
    n_inv   = r_deals["# Investors"].dropna().sum()
    types   = set()
    for inv in d_exp[(d_exp["co"]==co) & (d_exp["Year"]==yr)]["inv"]:
        t = inv_to_type.get(inv)
        if t:
            types.add(t)
    return {"cap": cap if cap > 0 else np.nan, "n_inv": n_inv if n_inv > 0 else np.nan,
            "types": types, "year": yr}

# A) Capital injection at round 2: bridge vs within-community round 2
print("\n  A) Capital at round 2: bridge companies vs within-community companies")
bridge_r2_caps, within_r2_caps = [], []
for co in list(bridge_r2_cos)[:2000]:
    r2 = get_round_stats(co, 2)
    if r2.get("cap"):
        bridge_r2_caps.append(r2["cap"])
for co in list(within_cos_set)[:2000]:
    r2 = get_round_stats(co, 2)
    if r2.get("cap"):
        within_r2_caps.append(r2["cap"])

if bridge_r2_caps and within_r2_caps:
    _, p_cap = stats.mannwhitneyu(bridge_r2_caps, within_r2_caps, alternative="two-sided")
    print(f"     Bridge R2 median capital:   ${np.median(bridge_r2_caps):.1f}M  (N={len(bridge_r2_caps)})")
    print(f"     Within R2 median capital:   ${np.median(within_r2_caps):.1f}M  (N={len(within_r2_caps)})")
    print(f"     Mann-Whitney p={p_cap:.4f} {sig(p_cap)}")

# B) Investor type change: is bridge investor a different type than round 1?
print("\n  B) Investor type change at bridge entry")
type_change = 0; type_same = 0; type_total = 0
for co in bridge_r2_cos:
    yt = co_timing.get(co, {})
    if not yt:
        continue
    years = sorted(yt.keys())
    if len(years) < 2:
        continue
    # Types in round 1 vs round 2
    t1 = set(inv_to_type.get(inv) for inv in
             d_exp[(d_exp["co"]==co) & (d_exp["Year"]==years[0])]["inv"] if inv_to_type.get(inv))
    t2 = set(inv_to_type.get(inv) for inv in
             d_exp[(d_exp["co"]==co) & (d_exp["Year"]==years[1])]["inv"] if inv_to_type.get(inv))
    if t1 and t2:
        if t1 != t2:
            type_change += 1
        else:
            type_same += 1
        type_total += 1
if type_total:
    print(f"     Bridge brings NEW investor type:  {type_change:,}  ({type_change/type_total:.1%})")
    print(f"     Same investor type as round 1:    {type_same:,}  ({type_same/type_total:.1%})")

# C) Subsequent rounds after bridge
print("\n  C) Rounds raised AFTER bridge enters (at round 2)")
# Compare companies with bridge at R2 vs within-community at R2
def rounds_after_r2(co):
    yt = co_timing.get(co, {})
    if not yt:
        return np.nan
    return max(0, len(yt) - 2)   # rounds after year 2

bridge_post = [rounds_after_r2(co) for co in bridge_r2_cos if not np.isnan(rounds_after_r2(co))]
within_post = [rounds_after_r2(co) for co in within_cos_set if not np.isnan(rounds_after_r2(co))]
bridge_post = [x for x in bridge_post if x is not None]
within_post = [x for x in within_post if x is not None]

if bridge_post and within_post:
    _, p_post = stats.mannwhitneyu(bridge_post, within_post, alternative="greater")
    print(f"     Bridge companies — mean rounds after R2: {np.mean(bridge_post):.2f}  (N={len(bridge_post)})")
    print(f"     Within companies — mean rounds after R2: {np.mean(within_post):.2f}  (N={len(within_post)})")
    print(f"     Mann-Whitney (bridge > within): p={p_post:.4f} {sig(p_post)}")

# D) Time to round 3
print("\n  D) Time from round 2 to round 3 (years)")
def time_r2_to_r3(co):
    yt = co_timing.get(co, {})
    if not yt:
        return np.nan
    years = sorted(yt.keys())
    if len(years) < 3:
        return np.nan
    return years[2] - years[1]

bridge_t3 = [time_r2_to_r3(co) for co in bridge_r2_cos]
within_t3  = [time_r2_to_r3(co) for co in within_cos_set]
bridge_t3  = [x for x in bridge_t3 if x is not None and not np.isnan(x)]
within_t3  = [x for x in within_t3 if x is not None and not np.isnan(x)]

if bridge_t3 and within_t3:
    _, p_t3 = stats.mannwhitneyu(bridge_t3, within_t3, alternative="less")
    print(f"     Bridge companies — median gap R2→R3: {np.median(bridge_t3):.1f} years  (N={len(bridge_t3)})")
    print(f"     Within companies — median gap R2→R3: {np.median(within_t3):.1f} years  (N={len(within_t3)})")
    print(f"     Mann-Whitney (bridge faster): p={p_t3:.4f} {sig(p_t3)}")

# Regression: more rounds after bridge controlling for total capital & time
cdf2["rounds_post_r2"] = cdf2["co"].apply(rounds_after_r2)
sub4 = cdf2.dropna(subset=["rounds_post_r2","log_cap","deal_year"]).copy()
sub4 = sub4[sub4["rounds_post_r2"] >= 0]
m4 = smf.ols(f"rounds_post_r2 ~ bridge_later + {CTRL}", data=sub4).fit(cov_type="HC1")
b4 = m4.params.get("bridge_later", np.nan)
p4 = m4.pvalues.get("bridge_later", 1.0)
print(f"\n  Regression: rounds_after_r2 ~ bridge_later + controls")
print(f"     β_bridge_later = {b4:>+.4f}  {sig(p4)}  → "
      f"{b4:+.2f} additional rounds after bridge enters")

# ── TEST 5: BRIDGE COMMUNITY FOLLOW-ON PIPELINE ───────────────────────────────
print()
print("="*72)
print("TEST 5: BRIDGE COMMUNITY FOLLOW-ON PIPELINE")
print("="*72)

print("  For each bridge company: does bridge community invest again in round 3+?")
pipeline_records = []
for _, row in timing_df[timing_df["first_bridge_round"]==2].iterrows():
    co         = row["co"]
    bridge_c   = row["bridge_comm"]
    yt = co_timing.get(co, {})
    if not yt:
        continue
    years = sorted(yt.keys())
    if len(years) < 3:
        # Company only raised 1-2 rounds — can't measure pipeline
        pipeline_records.append({"co": co, "got_r3": False, "bridge_followon": False,
                                  "n_later_rounds": 0})
        continue
    # Round 3+ investor communities
    later_comms = set()
    for yr in years[2:]:
        later_comms |= yt.get(yr, set())
    bridge_followon = bridge_c in later_comms
    pipeline_records.append({
        "co": co,
        "got_r3": True,
        "bridge_followon": bridge_followon,
        "n_later_rounds": len(years) - 2,
    })

pip_df = pd.DataFrame(pipeline_records)
got_r3 = pip_df[pip_df["got_r3"]]

print(f"\n  Bridge companies that raised round 3+: {len(got_r3):,}  "
      f"({len(got_r3)/len(pip_df):.1%} of bridge companies)")
if len(got_r3):
    followon_rate = got_r3["bridge_followon"].mean()
    print(f"  Of those, bridge community invests again in R3+: "
          f"{got_r3['bridge_followon'].sum():,}  ({followon_rate:.1%})")
    print(f"  → {'Bridge OPENS a pipeline (reinvestment >30%)' if followon_rate > 0.30 else 'Bridge is mostly one-time entry (reinvestment ≤30%)'}")

# Compare failure rates: bridge with follow-on pipeline vs bridge without
pip_df2 = pip_df.merge(cdf[["co","is_failed"]], on="co", how="left")
if len(pip_df2[pip_df2["bridge_followon"]==True]) >= 20:
    f_pipe = pip_df2[pip_df2["bridge_followon"]==True]["is_failed"].mean()
    f_nopipe = pip_df2[pip_df2["bridge_followon"]==False]["is_failed"].mean()
    print(f"\n  Failure rate — bridge with pipeline:    {f_pipe:.1%}")
    print(f"  Failure rate — bridge without pipeline: {f_nopipe:.1%}")
    print(f"  Δ = {(f_pipe - f_nopipe)*100:+.1f}pp")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print()
print("="*72)
print("SUMMARY: SELECTION vs. TREATMENT")
print("="*72)
print(f"""
  TEST 1 — {verdict1}
  TEST 2 — {verdict2}
  TEST 3 — {verdict3}

  Interpretation:
  - If round-1 cross-community ALSO reduces failure (Test 1):
    → Some effect is causal, not purely selection
    → At founding stage there's no "track record" to select on
  - If pre-bridge R1 performance is identical (Test 2):
    → Observable selection is not the mechanism
  - If IV works on sequential subsample (Test 3):
    → Even exogenously induced sequential cross-community helps
  - If bridge companies raise more rounds (Test 4C):
    → Bridge creates ongoing capital access, not just one check
  - If bridge reinvests (Test 5):
    → Pipeline effect: bridge is a gateway to a full ecosystem
""")
