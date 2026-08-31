"""
Script 49: Signal Independence Theory of Cross-Community Investment
===================================================================

THEORY: Within-community investors herd (correlated signals).
Cross-community investors evaluate independently (orthogonal signals).
Agreement by independent evaluators is more informative about true quality.

Tests:
  1. Co-investment herding (familiarity/prior co-investments)
  2. Portfolio overlap (signal correlation proxy)
  3. Signal independence → outcomes regression
  4. Herding vs independent syndicate failure rates
  5. Within-community concentration → failure
  6. Round-2 "second opinion from a stranger"
"""

import warnings; warnings.filterwarnings("ignore")
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

OUT  = Path("output/signal_independence"); OUT.mkdir(parents=True, exist_ok=True)
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ── Load base data ─────────────────────────────────────────────────────────
print("Loading data…")
deals = load_deals()
deals["co"]   = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")

net = pd.read_csv("output/network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm = net.set_index("investor")["community"].to_dict()

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]         = cdf["Companies"].str.strip()
cdf["is_failed"]  = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")
cdf["deal_year"]  = pd.to_numeric(cdf["deal_year"], errors="coerce")

# Explode deals → investor-company pairs
d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""')
         .dropna(subset=["Year"]))
d_exp["community"] = d_exp["inv"].map(inv_to_comm)

# Investor → set of companies invested in (all years)
inv_portfolio: dict[str, set] = defaultdict(set)
for _, row in d_exp.iterrows():
    inv_portfolio[row["inv"]].add(row["co"])

# Company → set of all investors
co_all_investors: dict[str, set] = (
    d_exp.groupby("co")["inv"].apply(set).to_dict())

# Company → first deal year
co_first_year: dict[str, int] = (
    d_exp.groupby("co")["Year"].min().astype(int).to_dict())

print(f"  Investors with portfolios: {len(inv_portfolio):,}")
print(f"  Companies with investor lists: {len(co_all_investors):,}")

# ═══════════════════════════════════════════════════════════════════════════
# BUILD PAIR METRICS
# Core: for each co-investing pair, compute community overlap, familiarity,
#       portfolio Jaccard
# ═══════════════════════════════════════════════════════════════════════════
print("\nBuilding co-investment pair metrics…")

pair_data = []  # one row per (company, investor1, investor2)

for co, investors in co_all_investors.items():
    inv_list = sorted([i for i in investors if i in inv_to_comm])
    if len(inv_list) < 2:
        continue

    co_year = co_first_year.get(co, 2015)

    for i, j in combinations(inv_list, 2):
        comm_i = inv_to_comm[i]
        comm_j = inv_to_comm[j]
        same_comm = int(comm_i == comm_j)

        # Portfolio overlap (Jaccard), excluding current company
        port_i = inv_portfolio[i] - {co}
        port_j = inv_portfolio[j] - {co}
        inter  = len(port_i & port_j)
        union  = len(port_i | port_j)
        jaccard = inter / union if union > 0 else 0.0

        # Prior co-investments: times they appeared in same company BEFORE co_year
        prior_cos = {c for c in port_i & port_j
                     if co_first_year.get(c, 9999) < co_year}
        prior_count = len(prior_cos)

        pair_data.append({
            "co":            co,
            "inv1":          i,
            "inv2":          j,
            "same_community": same_comm,
            "portfolio_jaccard": jaccard,
            "prior_coinvest": prior_count,
            "n_shared_total": inter,
        })

pairs = pd.DataFrame(pair_data)
print(f"  Total co-investment pairs: {len(pairs):,}")
print(f"  Same-community pairs:  {pairs['same_community'].sum():,}  ({pairs['same_community'].mean():.1%})")
print(f"  Cross-community pairs: {(~pairs['same_community'].astype(bool)).sum():,}  ({1-pairs['same_community'].mean():.1%})")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: CO-INVESTMENT HERDING — PRIOR CO-INVESTMENTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("TEST 1: CO-INVESTMENT HERDING — PRIOR CO-INVESTMENTS BY PAIR TYPE")
print("="*72)

within = pairs[pairs["same_community"] == 1]["prior_coinvest"]
cross  = pairs[pairs["same_community"] == 0]["prior_coinvest"]

tstat, pval = stats.mannwhitneyu(within, cross, alternative="greater")
_, pval_t = stats.ttest_ind(within, cross, alternative="greater")

print(f"\n  {'Pair type':<28} {'N':>7} {'Mean':>7} {'Median':>8} {'P75':>8} {'P90':>8}")
print(f"  {'-'*65}")
print(f"  {'Same-community':28} {len(within):>7,} {within.mean():>7.2f} {within.median():>8.2f} "
      f"{within.quantile(.75):>8.2f} {within.quantile(.90):>8.2f}")
print(f"  {'Cross-community':28} {len(cross):>7,} {cross.mean():>7.2f} {cross.median():>8.2f} "
      f"{cross.quantile(.75):>8.2f} {cross.quantile(.90):>8.2f}")
print(f"\n  Ratio (within/cross): {within.mean()/max(cross.mean(), 0.001):.1f}×")
print(f"  Mann-Whitney U (within>cross): p={pval:.4f}  {sig(pval)}")
print(f"  T-test (within>cross): p={pval_t:.4f}  {sig(pval_t)}")

pct_repeat_within = (within > 0).mean()
pct_repeat_cross  = (cross  > 0).mean()
print(f"\n  Pairs with ANY prior co-investment:")
print(f"    Same-community:  {pct_repeat_within:.1%}")
print(f"    Cross-community: {pct_repeat_cross:.1%}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: PORTFOLIO OVERLAP (SIGNAL CORRELATION)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("TEST 2: PORTFOLIO OVERLAP — SIGNAL CORRELATION")
print("="*72)

w_jac = pairs[pairs["same_community"] == 1]["portfolio_jaccard"]
c_jac = pairs[pairs["same_community"] == 0]["portfolio_jaccard"]

tstat_j, pval_j = stats.mannwhitneyu(w_jac, c_jac, alternative="greater")

print(f"\n  {'Pair type':<28} {'N':>7} {'Mean Jaccard':>13} {'Median':>8} {'P75':>8}")
print(f"  {'-'*68}")
print(f"  {'Same-community':28} {len(w_jac):>7,} {w_jac.mean():>13.4f} {w_jac.median():>8.4f} {w_jac.quantile(.75):>8.4f}")
print(f"  {'Cross-community':28} {len(c_jac):>7,} {c_jac.mean():>13.4f} {c_jac.median():>8.4f} {c_jac.quantile(.75):>8.4f}")
print(f"\n  Ratio (within/cross): {w_jac.mean()/max(c_jac.mean(),0.0001):.1f}×")
print(f"  Mann-Whitney U (within>cross): p={pval_j:.4f}  {sig(pval_j)}")

pct_zero_within = (w_jac == 0).mean()
pct_zero_cross  = (c_jac == 0).mean()
print(f"\n  Pairs with ZERO portfolio overlap:")
print(f"    Same-community:  {pct_zero_within:.1%}")
print(f"    Cross-community: {pct_zero_cross:.1%}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: SIGNAL INDEPENDENCE → OUTCOMES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("TEST 3: SIGNAL INDEPENDENCE → OUTCOMES (LPM)")
print("="*72)

# Company-level: mean portfolio overlap of all investor pairs
co_metrics = (pairs.groupby("co")
              .agg(
                  signal_independence   = ("portfolio_jaccard", lambda x: 1 - x.mean()),
                  mean_prior_coinvest   = ("prior_coinvest",    "mean"),
                  pct_same_community    = ("same_community",    "mean"),
                  n_pairs               = ("co",                "count"),
              )
              .reset_index())

print(f"\n  Companies with pair metrics: {len(co_metrics):,}")
print(f"  Signal independence (1-mean_jaccard): mean={co_metrics['signal_independence'].mean():.4f} ± {co_metrics['signal_independence'].std():.4f}")

df = cdf.merge(co_metrics, on="co", how="left")
df["has_pairs"] = df["n_pairs"].notna()
sub = df[df["has_pairs"]].copy()
sub["sig_indep_std"] = (sub["signal_independence"] - sub["signal_independence"].mean()) \
                        / sub["signal_independence"].std()
sub["prior_ci_std"]  = (sub["mean_prior_coinvest"] - sub["mean_prior_coinvest"].mean()) \
                        / sub["mean_prior_coinvest"].std()

print(f"\n  Merged with outcomes: N={len(sub):,}")
print(f"  Correlation (signal_independence × cross_community): "
      f"r={sub['signal_independence'].corr(sub['cross_community']):.4f}")
print(f"  Correlation (pct_same_community × cross_community): "
      f"r={sub['pct_same_community'].corr(sub['cross_community']):.4f}")

print(f"\n  {'─'*65}")
print(f"  LPM: failure ~ signal_independence (horse race with cross_community)")
print(f"  {'─'*65}")
for label, formula in [
    ("M1: cross_community",
     f"is_failed ~ cross_community + {CTRL}"),
    ("M2: signal_independence",
     f"is_failed ~ sig_indep_std + {CTRL}"),
    ("M3: both (horse race)",
     f"is_failed ~ cross_community + sig_indep_std + {CTRL}"),
    ("M4: +prior co-invest herding",
     f"is_failed ~ cross_community + sig_indep_std + prior_ci_std + {CTRL}"),
    ("M5: pct_same_community instead",
     f"is_failed ~ pct_same_community + cross_community + {CTRL}"),
]:
    m = smf.ols(formula, data=sub.dropna(subset=["is_failed"])).fit(cov_type="HC1")
    print(f"\n  [{label}]  R²={m.rsquared:.4f}  N={int(m.nobs):,}")
    for var, lbl in [
        ("cross_community",   "cross_community"),
        ("sig_indep_std",     "signal_independence(std)"),
        ("prior_ci_std",      "prior_coinvest(std)"),
        ("pct_same_community","pct_same_community"),
    ]:
        if var in m.params:
            b, p = m.params[var], m.pvalues[var]
            print(f"    {lbl:<28}: β={b:>+.4f}  {sig(p)}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: HERDING vs INDEPENDENT SYNDICATE FAILURE RATES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("TEST 4: HERDING vs INDEPENDENT SYNDICATE FAILURE RATES")
print("="*72)

# Classify syndicates by median splits
med_prior  = sub["mean_prior_coinvest"].median()
med_overlap = (1 - sub["signal_independence"]).median()

sub["high_familiarity"] = (sub["mean_prior_coinvest"]  > med_prior).astype(int)
sub["high_overlap"]     = ((1 - sub["signal_independence"]) > med_overlap).astype(int)

# 4a: by familiarity (prior co-investments)
print(f"\n  Split by syndicate familiarity (median prior co-investments = {med_prior:.2f}):")
print(f"  {'Type':<35} {'N':>6} {'Fail%':>8} {'Exit%':>8}")
print(f"  {'-'*60}")
for label, mask in [
    ("High-familiarity (herding)", sub["high_familiarity"]==1),
    ("Low-familiarity (independent)", sub["high_familiarity"]==0),
]:
    s = sub[mask].dropna(subset=["is_failed"])
    fail_r = s["is_failed"].mean()
    exit_r = s["exited"].mean() if "exited" in s else np.nan
    print(f"  {label:<35} {len(s):>6,} {fail_r:>8.1%} {exit_r:>8.1%}")

# Chi-squared test on failure
ct = pd.crosstab(sub.dropna(subset=["is_failed"])["high_familiarity"],
                 sub.dropna(subset=["is_failed"])["is_failed"])
chi2, p_chi, *_ = stats.chi2_contingency(ct)
print(f"  χ²={chi2:.3f}  p={p_chi:.4f}  {sig(p_chi)}")

# LPM for familiarity → failure
m_fam = smf.ols(f"is_failed ~ high_familiarity + {CTRL}", data=sub.dropna(subset=["is_failed"])).fit(cov_type="HC1")
b_fam = m_fam.params.get("high_familiarity", np.nan)
p_fam = m_fam.pvalues.get("high_familiarity", 1.0)
print(f"\n  LPM: β(high_familiarity→failure) = {b_fam:>+.4f}  {sig(p_fam)}")

# 4b: by portfolio overlap
print(f"\n  Split by portfolio overlap (median = {med_overlap:.4f}):")
print(f"  {'Type':<35} {'N':>6} {'Fail%':>8} {'Exit%':>8}")
print(f"  {'-'*60}")
for label, mask in [
    ("High-overlap (correlated signals)", sub["high_overlap"]==1),
    ("Low-overlap (independent signals)", sub["high_overlap"]==0),
]:
    s = sub[mask].dropna(subset=["is_failed"])
    fail_r = s["is_failed"].mean()
    exit_r = s["exited"].mean() if "exited" in s else np.nan
    print(f"  {label:<35} {len(s):>6,} {fail_r:>8.1%} {exit_r:>8.1%}")

ct2 = pd.crosstab(sub.dropna(subset=["is_failed"])["high_overlap"],
                  sub.dropna(subset=["is_failed"])["is_failed"])
chi2b, p_chi2b, *_ = stats.chi2_contingency(ct2)
m_ovl = smf.ols(f"is_failed ~ high_overlap + {CTRL}", data=sub.dropna(subset=["is_failed"])).fit(cov_type="HC1")
b_ovl = m_ovl.params.get("high_overlap", np.nan)
p_ovl = m_ovl.pvalues.get("high_overlap", 1.0)
print(f"  χ²={chi2b:.3f}  p={p_chi2b:.4f}  {sig(p_chi2b)}")
print(f"  LPM: β(high_overlap→failure) = {b_ovl:>+.4f}  {sig(p_ovl)}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: WITHIN-COMMUNITY CONCENTRATION → FAILURE
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("TEST 5: WITHIN-COMMUNITY CONCENTRATION → FAILURE")
print("="*72)

# For each company: find the modal community among investors, compute:
# - n_modal_community: investors from the most common community
# - pct_modal: fraction from the most common community
# - n_communities: number of distinct communities represented

co_comm_rows = []
for co, investors in co_all_investors.items():
    comms = [inv_to_comm[i] for i in investors if i in inv_to_comm]
    if not comms:
        continue
    comm_counts = pd.Series(comms).value_counts()
    n_investors = len(comms)
    n_comm_distinct = comm_counts.shape[0]
    modal_comm_n = comm_counts.iloc[0]
    pct_modal = modal_comm_n / n_investors

    # "Pure herding" = all investors from same community
    pure_herd = int(n_comm_distinct == 1 and n_investors > 1)

    co_comm_rows.append({
        "co": co,
        "n_investors_with_comm": n_investors,
        "n_communities_repr": n_comm_distinct,
        "n_modal_community": modal_comm_n,
        "pct_modal_community": pct_modal,
        "pure_herd": pure_herd,
    })

comm_df = pd.DataFrame(co_comm_rows)
df2 = cdf.merge(comm_df, on="co", how="left")
sub2 = df2[df2["n_investors_with_comm"].notna() & (df2["n_investors_with_comm"] > 1)].copy()

print(f"\n  Companies with ≥2 mapped investors: {len(sub2):,}")
print(f"  Pure-herd (all same community): {sub2['pure_herd'].sum():,}  ({sub2['pure_herd'].mean():.1%})")
print(f"  Multi-community: {(sub2['pure_herd']==0).sum():,}  ({(sub2['pure_herd']==0).mean():.1%})")

# Pure herd vs not
print(f"\n  Failure by syndicate type:")
for label, mask in [
    ("Pure herd (all same comm)", sub2["pure_herd"]==1),
    ("Multi-community",           sub2["pure_herd"]==0),
]:
    s = sub2[mask].dropna(subset=["is_failed"])
    fail = s["is_failed"].mean()
    n = len(s)
    print(f"    {label:<35} N={n:,}  Fail={fail:.1%}")

# LPM: pct_modal → failure
sub2["nrounds"] = np.log1p(sub2["n_rounds"].fillna(1))
for label, formula in [
    ("M1: pct_modal only",
     f"is_failed ~ pct_modal_community + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
    ("M2: pure_herd",
     f"is_failed ~ pure_herd + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
    ("M3: n_communities_repr (negative = diversity)",
     f"is_failed ~ n_communities_repr + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
    ("M4: horse race with cross_community",
     f"is_failed ~ cross_community + pct_modal_community + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
]:
    try:
        m = smf.ols(formula, data=sub2.dropna(subset=["is_failed"])).fit(cov_type="HC1")
        print(f"\n  [{label}]  R²={m.rsquared:.4f}  N={int(m.nobs):,}")
        for var, lbl in [
            ("cross_community",      "cross_community"),
            ("pct_modal_community",  "pct_modal_community"),
            ("pure_herd",            "pure_herd"),
            ("n_communities_repr",   "n_communities_repr"),
        ]:
            if var in m.params:
                b, p = m.params[var], m.pvalues[var]
                print(f"    {lbl:<28}: β={b:>+.4f}  {sig(p)}")
    except Exception as e:
        print(f"  [{label}] ERR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: ROUND-2 "SECOND OPINION FROM A STRANGER"
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print('TEST 6: ROUND-2 "SECOND OPINION FROM A STRANGER"')
print("="*72)

# Group deals by company×year to define rounds
round_inv = (d_exp.groupby(["co", "Year"])["inv"]
             .apply(set)
             .reset_index()
             .rename(columns={"inv": "investors", "Year": "year"}))
round_inv["year"] = round_inv["year"].astype(int)

stranger_rows = []

for co, grp in round_inv.groupby("co"):
    years = sorted(grp["year"].unique())
    if len(years) < 2:
        continue

    # Round 1: earliest year
    r1_invs = set().union(*grp[grp["year"] == years[0]]["investors"])
    r1_comms = set(inv_to_comm[i] for i in r1_invs if i in inv_to_comm)

    # Round 2: second year
    r2_invs = set().union(*grp[grp["year"] == years[1]]["investors"])
    # NEW investors in round 2 (not in round 1)
    new_r2 = r2_invs - r1_invs

    if not new_r2 or not r1_comms:
        continue

    for inv in new_r2:
        if inv not in inv_to_comm:
            continue
        inv_comm = inv_to_comm[inv]
        is_stranger = int(inv_comm not in r1_comms)
        stranger_rows.append({
            "co":          co,
            "new_investor": inv,
            "is_stranger": is_stranger,
        })

stranger_df = pd.DataFrame(stranger_rows)
print(f"\n  New round-2 investor instances: {len(stranger_df):,}")
print(f"  Companies with round-2 analysis: {stranger_df['co'].nunique():,}")
print(f"  Stranger (diff community): {stranger_df['is_stranger'].sum():,}  ({stranger_df['is_stranger'].mean():.1%})")
print(f"  Friend (same community):   {(~stranger_df['is_stranger'].astype(bool)).sum():,}  ({1-stranger_df['is_stranger'].mean():.1%})")

# For each company: classify as stranger-validated or friend-validated
# Use MAJORITY of new round-2 investors
co_stranger = (stranger_df.groupby("co")["is_stranger"]
               .mean()
               .rename("pct_strangers")
               .reset_index())
co_stranger["majority_stranger"] = (co_stranger["pct_strangers"] >= 0.5).astype(int)
co_stranger["any_stranger"]      = (co_stranger["pct_strangers"] > 0).astype(int)

df3 = cdf.merge(co_stranger, on="co", how="inner")
df3["nrounds"] = np.log1p(df3["n_rounds"].fillna(1))

print(f"\n  Companies merged with outcomes: {len(df3):,}")
print(f"\n  Failure rates by round-2 type:")
print(f"  {'Type':<45} {'N':>6} {'Fail%':>8}")
print(f"  {'-'*62}")
for label, mask in [
    ("Majority strangers (≥50% new investors, diff comm)", df3["majority_stranger"]==1),
    ("Majority friends (≥50% new investors, same comm)",  df3["majority_stranger"]==0),
    ("Any stranger (≥1 new from diff community)",         df3["any_stranger"]==1),
    ("No strangers (all new from same community)",        df3["any_stranger"]==0),
]:
    s = df3[mask].dropna(subset=["is_failed"])
    if len(s) < 20: continue
    fail = s["is_failed"].mean()
    n    = len(s)
    print(f"  {label:<45} {n:>6,} {fail:>8.1%}")

# Mann-Whitney on failure
strang_f = df3[df3["majority_stranger"]==1]["is_failed"].dropna()
friend_f = df3[df3["majority_stranger"]==0]["is_failed"].dropna()
if len(strang_f) > 10 and len(friend_f) > 10:
    _, p_mw = stats.mannwhitneyu(friend_f, strang_f, alternative="greater")
    print(f"\n  Mann-Whitney (friend > stranger failure): p={p_mw:.4f}  {sig(p_mw)}")

# LPM
for label, formula in [
    ("M1: majority_stranger → failure",
     f"is_failed ~ majority_stranger + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
    ("M2: pct_strangers (continuous)",
     f"is_failed ~ pct_strangers + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
    ("M3: +cross_community horse race",
     f"is_failed ~ majority_stranger + cross_community + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"),
]:
    try:
        m = smf.ols(formula, data=df3.dropna(subset=["is_failed"])).fit(cov_type="HC1")
        print(f"\n  [{label}]  R²={m.rsquared:.4f}  N={int(m.nobs):,}")
        for var, lbl in [
            ("majority_stranger","majority_stranger"),
            ("pct_strangers",    "pct_strangers"),
            ("cross_community",  "cross_community"),
        ]:
            if var in m.params:
                b, p = m.params[var], m.pvalues[var]
                print(f"    {lbl:<28}: β={b:>+.4f}  {sig(p)}")
    except Exception as e:
        print(f"  ERR [{label}]: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE COMPANY-LEVEL SIGNAL METRICS
# ═══════════════════════════════════════════════════════════════════════════
out_df = cdf[["co","is_failed","cross_community","log_cap"]].merge(
    co_metrics[["co","signal_independence","mean_prior_coinvest","pct_same_community"]], on="co", how="left"
).merge(
    comm_df[["co","n_communities_repr","pct_modal_community","pure_herd"]], on="co", how="left"
).merge(
    co_stranger[["co","pct_strangers","majority_stranger"]], on="co", how="left"
)
out_df.to_csv(OUT / "company_signal_metrics.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("SUMMARY: IS SIGNAL INDEPENDENCE THE MECHANISM?")
print("="*72)

print(f"""
  TEST 1 — CO-INVESTMENT HERDING (PRIOR CO-INVESTMENTS)
  ─────────────────────────────────────────────────────
  Same-community pairs: {within.mean():.2f} prior co-investments on avg
  Cross-community pairs: {cross.mean():.2f} prior co-investments on avg
  Ratio: {within.mean()/max(cross.mean(),0.001):.1f}×  MW p={pval:.4f}  {sig(pval)}
  → {'Confirmed: within-community investors have significantly higher familiarity.' if pval < 0.10 else 'Weak or no herding signal.'}

  TEST 2 — PORTFOLIO OVERLAP (SIGNAL CORRELATION)
  ─────────────────────────────────────────────────
  Same-community pairs: Jaccard = {w_jac.mean():.4f}
  Cross-community pairs: Jaccard = {c_jac.mean():.4f}
  Ratio: {w_jac.mean()/max(c_jac.mean(),0.0001):.1f}×  MW p={pval_j:.4f}  {sig(pval_j)}
  → {'Confirmed: within-community investors see MORE of the same deals.' if pval_j < 0.10 else 'Portfolio overlap difference is not significant.'}

  TEST 4 — HERDING vs INDEPENDENT SYNDICATE FAILURE
  ─────────────────────────────────────────────────
  High-familiarity (herding) β: {b_fam:>+.4f}  {sig(p_fam)}
  High-overlap (correlated)   β: {b_ovl:>+.4f}  {sig(p_ovl)}
  → {'HERDING syndicates fail more — correlated signals are harmful.' if p_fam < 0.10 and b_fam > 0 else
     'INDEPENDENT syndicates do not significantly outperform.' if p_fam >= 0.10 else
     'High familiarity syndicates do BETTER — check interpretation.'}

  TEST 5 — WITHIN-COMMUNITY CONCENTRATION
  ─────────────────────────────────────────────────
  → See Model 1 above for pct_modal_community → failure.

  VERDICT — SIGNAL INDEPENDENCE AS MECHANISM:
""")

# Pull test-5 modal-community β from earlier
try:
    m_modal = smf.ols(f"is_failed ~ pct_modal_community + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)",
                      data=sub2.dropna(subset=["is_failed"])).fit(cov_type="HC1")
    b_modal = m_modal.params.get("pct_modal_community", np.nan)
    p_modal = m_modal.pvalues.get("pct_modal_community", 1.0)
    m_both = smf.ols(f"is_failed ~ cross_community + pct_modal_community + log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)",
                     data=sub2.dropna(subset=["is_failed"])).fit(cov_type="HC1")
    b_cross_ctrl = m_both.params.get("cross_community", np.nan)
    p_cross_ctrl = m_both.pvalues.get("cross_community", 1.0)

    print(f"  pct_modal_community → failure:  β={b_modal:>+.4f}  {sig(p_modal)}")
    print(f"  cross_community (with pct_modal): β={b_cross_ctrl:>+.4f}  {sig(p_cross_ctrl)}")

    modal_sig     = p_modal < 0.10 and b_modal > 0
    cross_survives = p_cross_ctrl < 0.10

    if modal_sig and cross_survives:
        print("\n  ✅ BOTH within-community herding AND cross-community network position")
        print("     independently predict failure. Signal independence is A mechanism,")
        print("     but cross-community captures additional variance (perhaps deal quality")
        print("     access, information bridging, or resource complementarities).")
    elif modal_sig and not cross_survives:
        print("\n  ⚡ STRONG: Within-community concentration predicts failure and ABSORBS")
        print("     cross-community. The mechanism IS signal independence — cross-community")
        print("     works because it forces exposure to non-correlated evaluators.")
    elif not modal_sig and cross_survives:
        print("\n  ❌ Signal independence (portfolio overlap / herding) does NOT independently")
        print("     predict failure. Cross-community has a different mechanism —")
        print("     possibly information bridging, deal access, or resource complementarities.")
    else:
        print("\n  ❌ Neither herding nor cross-community strongly significant in this subsample.")
        print("     Check power — pair-level data requires ≥2 investors per company.")
except Exception as e:
    print(f"  ERR in verdict: {e}")
