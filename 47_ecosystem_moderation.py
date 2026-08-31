"""
Script 47: Local ecosystem strength as moderator of cross-community effect
==========================================================================

Tests whether the benefit of cross-community investment is stronger when
the home community (local ecosystem) is weaker — substitution vs. complement.

Key prediction: cross_community × log_local_size > 0
  → Large community: cross-community less needed (substitution)
  → Small community: cross-community most beneficial (fill a gap)

Also tests GDP / VC market maturity as country-level moderator.
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

OUT  = Path("output")
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ── Load base data ─────────────────────────────────────────────────────────
print("Loading data…")
cdf = pd.read_csv(OUT / "cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]   = cdf["log_capital"]
cdf["nrounds"]   = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

net = pd.read_csv(OUT / "network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm = net.set_index("investor")["community"].to_dict()

# ── STEP 1: BUILD HOME COMMUNITY METRICS ──────────────────────────────────
print("Building ecosystem strength metrics…")

# Community-level stats from network
comm_stats = net.groupby("community").agg(
    comm_size   = ("investor", "count"),
    mean_degree = ("degree",   "mean"),
).reset_index()
comm_stats["comm_density"] = (
    comm_stats["mean_degree"] / (comm_stats["comm_size"] - 1).clip(lower=1))
comm_size_map   = comm_stats.set_index("community")["comm_size"].to_dict()
comm_density_map = comm_stats.set_index("community")["comm_density"].to_dict()

# For each company: identify home community (first community to invest)
d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))
d_exp["community"] = d_exp["inv"].map(inv_to_comm)
d_exp["Year"] = pd.to_numeric(d_exp["Year"], errors="coerce")
d_exp_c = d_exp.dropna(subset=["community", "Year"])

co_home = {}
for co, grp in d_exp_c.groupby("co"):
    comm_first = {}
    for yr in sorted(grp["Year"].unique()):
        for c in grp[grp["Year"]==yr]["community"].unique():
            if c not in comm_first:
                comm_first[c] = yr
    if comm_first:
        co_home[co] = min(comm_first, key=comm_first.get)  # home = first community

# Metric 1: Home community size
cdf["home_comm"]      = cdf["co"].map(co_home)
cdf["home_comm_size"] = cdf["home_comm"].map(comm_size_map)
cdf["home_comm_density"] = cdf["home_comm"].map(comm_density_map)
cdf["log_home_size"]  = np.log(cdf["home_comm_size"].clip(lower=1))
cdf["log_home_density"] = np.log(cdf["home_comm_density"].clip(lower=1e-6))

# Metric 2: Local deal activity — deals in same region × same year
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")
deals_vc = deals[deals["Deal Type"].isin(
    {"Early Stage VC","Later Stage VC","Seed Round","Angel (individual)","PE Growth/Expansion"})].copy()
region_year_counts = (deals_vc.dropna(subset=["Year"])
                      .merge(cdf[["co","hq_region"]], on="co", how="left")
                      .dropna(subset=["hq_region"])
                      .groupby(["hq_region","Year"])
                      .size()
                      .reset_index(name="regional_deals"))
# Map to each company using its hq_region and deal_year
cdf["deal_year_num"] = pd.to_numeric(cdf["deal_year"], errors="coerce")
cdf = cdf.merge(region_year_counts.rename(columns={"Year":"deal_year_num"}),
                on=["hq_region","deal_year_num"], how="left")
cdf["log_regional_deals"] = np.log1p(cdf["regional_deals"].fillna(0))

print(f"  Companies with home community assigned: {cdf['home_comm'].notna().sum():,}")
print(f"  Home community size range: {cdf['home_comm_size'].min():.0f} – {cdf['home_comm_size'].max():.0f}")
print(f"  Home community density range: {cdf['home_comm_density'].min():.4f} – {cdf['home_comm_density'].max():.4f}")
print(f"  Regional deals range: {cdf['regional_deals'].min():.0f} – {cdf['regional_deals'].max():.0f}")

# Size tertiles for split analysis
cdf["size_tertile"] = pd.qcut(cdf["home_comm_size"], q=3,
                               labels=["Small\n(<Q1)","Medium\n(Q2)","Large\n(>Q3)"])
tertile_cuts = cdf["home_comm_size"].quantile([1/3, 2/3])
print(f"\n  Community size tertile cuts: {tertile_cuts[1/3]:.0f}, {tertile_cuts[2/3]:.0f}")

df = cdf.dropna(subset=["log_cap","deal_year","specter_novelty","home_comm_size"]).copy()

# ── STEP 2: INTERACTION MODELS ────────────────────────────────────────────
print()
print("="*72)
print("MODERATION 1: HOME COMMUNITY SIZE × CROSS-COMMUNITY")
print("="*72)

# Model 1a: main effects only
m1a = smf.ols(
    f"is_failed ~ cross_community + log_home_size + {CTRL}",
    data=df.dropna(subset=["is_failed"])).fit(cov_type="HC1")

# Model 1b: with interaction
m1b = smf.ols(
    f"is_failed ~ cross_community * log_home_size + {CTRL}",
    data=df.dropna(subset=["is_failed"])).fit(cov_type="HC1")

print(f"\n  MODEL 1a — Main effects (no interaction)")
for var, label in [("cross_community","cross_community"),
                   ("log_home_size","log(home community size)")]:
    b = m1a.params.get(var, np.nan)
    p = m1a.pvalues.get(var, 1.0)
    print(f"  {label:<35}  β={b:>+.4f}  {sig(p)}  (p={p:.3f})")
print(f"  R²={m1a.rsquared:.4f}   N={int(m1a.nobs):,}")

print(f"\n  MODEL 1b — With interaction (cross_community × log_home_size)")
for var, label in [("cross_community","cross_community"),
                   ("log_home_size","log(home community size)"),
                   ("cross_community:log_home_size","INTERACTION")]:
    b = m1b.params.get(var, np.nan)
    se = m1b.bse.get(var, np.nan)
    p = m1b.pvalues.get(var, 1.0)
    print(f"  {label:<35}  β={b:>+.4f}  SE={se:.4f}  {sig(p)}  (p={p:.3f})")
print(f"  R²={m1b.rsquared:.4f}   N={int(m1b.nobs):,}")

# Interpret: marginal effect of cross_community at different community sizes
b_cross = m1b.params.get("cross_community", np.nan)
b_inter = m1b.params.get("cross_community:log_home_size", np.nan)
print(f"\n  Marginal effect of cross_community at different home community sizes:")
print(f"  {'Community size':>20}  {'log(size)':>10}  {'β_cross (marginal)':>20}  Interpretation")
print(f"  {'-'*75}")
for size, label in [(10,"Very small (10)"), (50,"Small (50)"), (200,"Medium (200)"),
                    (500,"Large (500)"), (1800,"Very large (1800)")]:
    log_s = np.log(size)
    marg  = b_cross + b_inter * log_s
    print(f"  {label:>20}  {log_s:>10.2f}  {marg:>+20.4f}  {'benefit' if marg<0 else 'no benefit'}")

# ── STEP 3: MODEL WITH COMMUNITY DENSITY ──────────────────────────────────
print()
print("="*72)
print("MODERATION 2: HOME COMMUNITY DENSITY × CROSS-COMMUNITY")
print("="*72)

m2 = smf.ols(
    f"is_failed ~ cross_community * log_home_density + {CTRL}",
    data=df.dropna(subset=["is_failed","log_home_density"])).fit(cov_type="HC1")

for var, label in [("cross_community","cross_community"),
                   ("log_home_density","log(community density)"),
                   ("cross_community:log_home_density","INTERACTION")]:
    b  = m2.params.get(var, np.nan)
    se = m2.bse.get(var, np.nan)
    p  = m2.pvalues.get(var, 1.0)
    print(f"  {label:<38}  β={b:>+.4f}  SE={se:.4f}  {sig(p)}")
print(f"  R²={m2.rsquared:.4f}   N={int(m2.nobs):,}")

# ── STEP 4: REGIONAL DEAL ACTIVITY ────────────────────────────────────────
print()
print("="*72)
print("MODERATION 3: REGIONAL DEAL ACTIVITY × CROSS-COMMUNITY")
print("="*72)

m3 = smf.ols(
    f"is_failed ~ cross_community * log_regional_deals + {CTRL}",
    data=df.dropna(subset=["is_failed","log_regional_deals"])).fit(cov_type="HC1")

for var, label in [("cross_community","cross_community"),
                   ("log_regional_deals","log(regional deal activity)"),
                   ("cross_community:log_regional_deals","INTERACTION")]:
    b  = m3.params.get(var, np.nan)
    se = m3.bse.get(var, np.nan)
    p  = m3.pvalues.get(var, 1.0)
    print(f"  {label:<40}  β={b:>+.4f}  SE={se:.4f}  {sig(p)}")
print(f"  R²={m3.rsquared:.4f}   N={int(m3.nobs):,}")

# ── STEP 5: SPLIT BY COMMUNITY SIZE TERTILE ───────────────────────────────
print()
print("="*72)
print("SPLIT BY HOME COMMUNITY SIZE (tertiles)")
print("="*72)

print(f"\n  {'Tertile':<18}  {'Size range':>18}  {'N companies':>12}  "
      f"{'β_cross':>9}  {'SE':>7}  {'Sig':>5}  {'Fail rate cross':>17}  {'Fail rate within':>18}")
print(f"  {'-'*108}")

for tertile in ["Small\n(<Q1)","Medium\n(Q2)","Large\n(>Q3)"]:
    sub = df[df["size_tertile"]==tertile].dropna(subset=["is_failed"]).copy()
    if len(sub) < 30: continue
    size_range = f"{sub['home_comm_size'].min():.0f}–{sub['home_comm_size'].max():.0f}"
    n = len(sub)
    fail_cross  = sub[sub["cross_community"]==1]["is_failed"].mean()
    fail_within = sub[sub["cross_community"]==0]["is_failed"].mean()
    try:
        m = smf.ols(f"is_failed ~ cross_community + {CTRL}", data=sub).fit(cov_type="HC1")
        b  = m.params.get("cross_community", np.nan)
        se = m.bse.get("cross_community", np.nan)
        p  = m.pvalues.get("cross_community", 1.0)
        label = tertile.replace('\n',' ')
        print(f"  {label:<18}  {size_range:>18}  {n:>12,}  "
              f"{b:>+9.4f}  {se:>7.4f}  {sig(p):>5}  {fail_cross:>17.1%}  {fail_within:>18.1%}")
    except Exception as e:
        print(f"  {tertile:<18}  ERR: {e}")

# ── STEP 6: COUNTRY VC MARKET MATURITY ───────────────────────────────────
print()
print("="*72)
print("MODERATION 4: COUNTRY VC MARKET MATURITY")
print("="*72)

# VC market maturity proxy: N total investors in country from our dataset
country_inv_count = (d_exp_c.dropna(subset=["community"])
                    .merge(cdf[["co","hq_country"]], on="co", how="left")
                    .dropna(subset=["hq_country"])
                    .groupby("hq_country")["inv"]
                    .nunique()
                    .rename("country_n_investors"))
df["country_n_inv"] = df["hq_country"].map(country_inv_count)
df["log_country_inv"] = np.log1p(df["country_n_inv"].fillna(0))

# Also use hq_region as proxy (US/Europe = mature, others = developing)
df["is_mature_market"] = df["hq_region"].isin(["Americas","Europe"]).astype(float)
df["is_developing"]    = (~df["hq_region"].isin(["Americas","Europe"])).astype(float)

print(f"\n  Country VC investor count (top 10):")
top_c = country_inv_count.sort_values(ascending=False).head(10)
for country, n in top_c.items():
    print(f"    {country:<30} {n:>6,} unique investors")

# Model 4a: country VC depth interaction
m4a = smf.ols(
    f"is_failed ~ cross_community * log_country_inv + {CTRL}",
    data=df.dropna(subset=["is_failed","log_country_inv"])).fit(cov_type="HC1")

print(f"\n  Interaction: cross_community × country VC depth (log N investors in country)")
for var, label in [("cross_community","cross_community"),
                   ("log_country_inv","log(N investors in country)"),
                   ("cross_community:log_country_inv","INTERACTION (key)")]:
    b  = m4a.params.get(var, np.nan)
    se = m4a.bse.get(var, np.nan)
    p  = m4a.pvalues.get(var, 1.0)
    star = " ← KEY" if "INTER" in label else ""
    print(f"  {label:<42}  β={b:>+.4f}  SE={se:.4f}  {sig(p)}{star}")
print(f"  R²={m4a.rsquared:.4f}   N={int(m4a.nobs):,}")

# Model 4b: mature vs developing split
print(f"\n  SPLIT: Mature markets (US/Europe) vs Developing (Asia/Africa/ME/Oceania)")
print(f"  {'Market':<25}  {'N':>6}  {'β_cross':>9}  {'SE':>7}  {'Sig':>5}  {'Fail_cross':>12}  {'Fail_within':>13}")
print(f"  {'-'*80}")
for label, mask in [("Mature (US/Europe)", df["is_mature_market"]==1),
                    ("Developing",          df["is_developing"]==1)]:
    sub = df[mask].dropna(subset=["is_failed"]).copy()
    fail_c = sub[sub["cross_community"]==1]["is_failed"].mean()
    fail_w = sub[sub["cross_community"]==0]["is_failed"].mean()
    try:
        m = smf.ols(f"is_failed ~ cross_community + {CTRL}", data=sub).fit(cov_type="HC1")
        b  = m.params.get("cross_community", np.nan)
        se = m.bse.get("cross_community", np.nan)
        p  = m.pvalues.get("cross_community", 1.0)
        print(f"  {label:<25}  {len(sub):>6,}  {b:>+9.4f}  {se:>7.4f}  {sig(p):>5}  {fail_c:>12.1%}  {fail_w:>13.1%}")
    except Exception as e:
        print(f"  {label:<25}  ERR: {e}")

# Wald test: β_mature = β_developing?
# Use interaction model with region dummy
m4c = smf.ols(
    f"is_failed ~ cross_community * is_developing + log_cap + nrounds + specter_novelty + C(deal_year)",
    data=df.dropna(subset=["is_failed"])).fit(cov_type="HC1")
b_int = m4c.params.get("cross_community:is_developing", np.nan)
p_int = m4c.pvalues.get("cross_community:is_developing", 1.0)
print(f"\n  Wald test (β_cross differs by market maturity):")
print(f"  cross_community × is_developing: β={b_int:>+.4f}  {sig(p_int)}  (p={p_int:.3f})")

# ── STEP 7: FULL INTERACTION MODEL ────────────────────────────────────────
print()
print("="*72)
print("FULL INTERACTION MODEL (size + density + maturity together)")
print("="*72)

m_full = smf.ols(
    f"is_failed ~ cross_community * log_home_size + "
    f"cross_community * is_developing + "
    f"log_home_density + {CTRL}",
    data=df.dropna(subset=["is_failed","log_home_size"])).fit(cov_type="HC1")

print(f"\n  {'Variable':<45}  {'β':>9}  {'SE':>8}  {'Sig':>5}")
print(f"  {'-'*75}")
key_vars = [
    ("cross_community","cross_community"),
    ("log_home_size","log(home community size)"),
    ("cross_community:log_home_size","INTERACTION: cross × home size"),
    ("is_developing","is_developing market"),
    ("cross_community:is_developing","INTERACTION: cross × developing"),
    ("log_home_density","log(community density)"),
    ("log_cap","log capital (control)"),
    ("nrounds","log rounds (control)"),
]
for var, label in key_vars:
    b  = m_full.params.get(var, np.nan)
    se = m_full.bse.get(var, np.nan)
    p  = m_full.pvalues.get(var, 1.0)
    flag = " ←" if "INTER" in label and p < 0.10 else ""
    print(f"  {label:<45}  {b:>+9.4f}  {se:>8.4f}  {sig(p):>5}{flag}")
print(f"\n  R²={m_full.rsquared:.4f}   N={int(m_full.nobs):,}")

# ── SUMMARY ───────────────────────────────────────────────────────────────
print()
print("="*72)
print("SUMMARY: DOES LOCAL ECOSYSTEM STRENGTH MODERATE THE EFFECT?")
print("="*72)

b_inter_size = m1b.params.get("cross_community:log_home_size", np.nan)
p_inter_size = m1b.pvalues.get("cross_community:log_home_size", 1.0)
b_inter_dens = m2.params.get("cross_community:log_home_density", np.nan)
p_inter_dens = m2.pvalues.get("cross_community:log_home_density", 1.0)
b_inter_reg  = m3.params.get("cross_community:log_regional_deals", np.nan)
p_inter_reg  = m3.pvalues.get("cross_community:log_regional_deals", 1.0)
b_inter_ctry = m4a.params.get("cross_community:log_country_inv", np.nan)
p_inter_ctry = m4a.pvalues.get("cross_community:log_country_inv", 1.0)

print(f"""
  Interaction summary (all moderation tests):
  {'Moderator':<40}  {'β_interaction':>15}  {'Sig':>5}  {'Direction'}
  {'-'*85}
  {'Home community size':<40}  {b_inter_size:>+15.4f}  {sig(p_inter_size):>5}  {'positive=substitution' if b_inter_size>0 else 'negative=complement'}
  {'Home community density':<40}  {b_inter_dens:>+15.4f}  {sig(p_inter_dens):>5}  {'positive=substitution' if b_inter_dens>0 else 'negative=complement'}
  {'Regional deal activity':<40}  {b_inter_reg:>+15.4f}  {sig(p_inter_reg):>5}  {'positive=substitution' if b_inter_reg>0 else 'negative=complement'}
  {'Country VC depth':<40}  {b_inter_ctry:>+15.4f}  {sig(p_inter_ctry):>5}  {'positive=substitution' if b_inter_ctry>0 else 'negative=complement'}
  {'Developing vs mature market':<40}  {b_int:>+15.4f}  {sig(p_int):>5}  {'cross more helpful in developing' if b_int<0 else 'cross more helpful in mature'}

  PREDICTED PATTERN (substitution hypothesis):
  → Positive interaction: large/dense ecosystem → cross-community less needed
  → Negative β_cross in small communities → effect concentrated where it matters most

  INTERPRETATION:
""")

any_sig = any(p < 0.10 for p in [p_inter_size, p_inter_dens, p_inter_reg, p_inter_ctry, p_int])
if any_sig:
    print("  ✅ At least one ecosystem moderator is significant (p<0.10).")
    print("     The cross-community effect is heterogeneous — it depends on local context.")
    if p_inter_size < 0.10:
        if b_inter_size > 0:
            print(f"     → Community size: SUBSTITUTION confirmed (β={b_inter_size:+.4f} {sig(p_inter_size)})")
            print(f"       Small ecosystems benefit most from cross-community bridges.")
        else:
            print(f"     → Community size: COMPLEMENTARITY (β={b_inter_size:+.4f} {sig(p_inter_size)})")
            print(f"       Unexpected: large ecosystems benefit MORE from bridges.")
    if p_int < 0.10:
        if b_int < 0:
            print(f"     → Developing markets: cross-community MORE protective (β={b_int:+.4f} {sig(p_int)})")
            print(f"       Strong policy finding: bridges matter most where ecosystems are weakest.")
        else:
            print(f"     → Developing markets: cross-community LESS protective (β={b_int:+.4f} {sig(p_int)})")
else:
    print("  ❌ No significant moderation by ecosystem strength (all p>0.10).")
    print("     The cross-community effect is HOMOGENEOUS — equally beneficial")
    print("     regardless of local ecosystem size, density, or market maturity.")
    print("     This is also a finding: network position helps everywhere,")
    print("     not just in underdeveloped ecosystems. The mechanism is universal.")
