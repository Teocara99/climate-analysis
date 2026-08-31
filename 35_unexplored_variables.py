"""
Script 35: Unexplored Variable Analyses
========================================

Five blocks testing whether the cross_community → lower failure effect
survives new controls and extends to positive outcomes.

BLOCK 1  Founder quality controls (CEO bio / education text extraction)
BLOCK 2  Positive outcomes: employees, growth rate, success probability
BLOCK 3  Innovation: patent filing and pipeline
BLOCK 4  Fine-grained industry FE (39 groups vs 7 sectors)
BLOCK 5  Competitor network (exploratory; 29% coverage)

KEY RESULTS:
  Block 1: cross_community survives all founder controls (OLS β=−0.037***)
           2SLS β=−0.045 (p≈0.10) with KP-F=328 — reduced power from smaller N
  Block 2: cross_community positively predicts employees (+11%), growth
           percentile (+4.3pp***), success probability (+9.3pts***)
  Block 3: cross_community predicts having any patents (+3.7pp**), marginal
           on total count; negative (†) on depth among filers
  Block 4: β virtually unchanged adding 39-group industry FE (−0.036→−0.035***)
  Block 5: Competitor data too sparse (29%) for reliable inference; n.s. throughout
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, sys, re
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

sys.path.insert(0, ".")
from load_data import load_companies, load_deals


def sig(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.10 else "n.s.")))


def parse_investors(s):
    if pd.isna(s) or str(s).strip() in ("", "None"): return []
    return [p.strip() for p in re.split(r",\s*", str(s)) if p.strip()]


# ── Load base data ──────────────────────────────────────────────────────────
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]          = cdf["Companies"].str.strip()
cdf["is_failed"]   = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]     = cdf["log_capital"]
cdf["nrounds"]     = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"]  = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x >= 3].index), "Other")

CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

cmp = load_companies(); cmp["co"] = cmp["Companies"].str.strip()
cmp_sub = cmp[cmp["co"].isin(set(cdf["co"]))].copy()
cmp_sub = cmp_sub.rename(columns={
    "Growth Rate":             "growth_rate",
    "Growth Rate Percentile":  "growth_pct",
    "Employees":               "employees",
    "Total Patent Documents":  "total_patents",
    "Active Patent Documents": "active_patents",
    "Pending Patent Documents":"pending_patents",
    "Primary PitchBook Industry Group":  "ind_group",
    "Primary PitchBook Industry Sector": "ind_sector",
})

df = cdf.merge(cmp_sub[["co","growth_rate","growth_pct","employees","total_patents",
                          "active_patents","pending_patents","ind_group","ind_sector",
                          "Competitors"]], on="co", how="left")
mix = pd.read_csv("output/company_investor_mix.csv")
df = df.merge(mix[["Companies","Success Probability"]].rename(
    columns={"Success Probability":"success_prob"}), on="Companies", how="left")

df["log_emp"]     = np.log1p(df["employees"])
df["log_patents"] = np.log1p(df["total_patents"].fillna(0))
df["log_active"]  = np.log1p(df["active_patents"].fillna(0))
df["log_pending"] = np.log1p(df["pending_patents"].fillna(0))
df["has_patent"]  = (df["total_patents"].notna() & (df["total_patents"] > 0)).astype(float)

# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("BLOCK 1: FOUNDER QUALITY CONTROLS")
print("=" * 72)

print("Loading deals file for CEO biography / education...")
d = load_deals()
d["co"] = d["Companies"].str.strip()
d_ceo = (d[d["co"].isin(set(cdf["co"]))]
         .sort_values("Year", ascending=False)
         .drop_duplicates("co")[["co", "CEO Biography", "CEO Education"]])
print(f"  CEO bio coverage: {d_ceo['CEO Biography'].notna().mean():.1%} of {len(d_ceo):,} companies")


def extract_bio(bio):
    if pd.isna(bio):
        return pd.Series([np.nan]*5, index=["serial_founder","prior_exit","is_technical","is_business","yrs_exp"])
    b = str(bio).lower()
    serial = int(bool(re.search(
        r'co[\-\s]?found|previously\s+found|former\s+found|founded\s+\w+(?:\s+\w+)?\s+(?:and|,|in|\.|before)'
        r'|founded\s+(?:a\s+)?(?:company|startup|venture)', b)))
    exit_  = int(bool(re.search(
        r'(?:was\s+)?(?:acquired|ipo|went\s+public|exit(?:ed)?|sold\s+(?:to|for)|merger|listed\s+on|nasdaq|nyse)', b)))
    tech   = int(bool(re.search(
        r'\bph\.?d\b|engineer|scientist|science|physics|chemist|biolog|software\s+develop'
        r'|technical\s+(?:lead|director|co-?founder)|computer\s+science|machine\s+learning|ai\s+research', b)))
    biz    = int(bool(re.search(
        r'\bmba\b|m\.b\.a|financ|consult|invest(?:ment)?\s+bank|private\s+equity|venture\s+capital'
        r'|business\s+develop|mckinsey|bain|bcg|goldman|jp\s+morgan', b)))
    years  = re.findall(r'\b(19\d\d|20[01]\d)\b', str(bio))
    yrs    = int(np.ptp([int(y) for y in years])) if len(years) >= 2 else np.nan
    return pd.Series([serial, exit_, tech, biz, yrs], index=["serial_founder","prior_exit","is_technical","is_business","yrs_exp"])


def extract_edu(edu):
    if pd.isna(edu):
        return pd.Series([np.nan]*3, index=["elite_uni","has_phd","has_mba"])
    e = str(edu).lower()
    elite = int(any(s in e for s in [
        'mit','stanford','harvard','oxford','cambridge','caltech','berkeley',
        'eth zurich','imperial college','columbia','princeton','yale',
        'carnegie mellon','lse','insead','wharton','kellogg','booth',
        'london business school','iit','tsinghua','peking']))
    phd = int(bool(re.search(r'\bph\.?d\b|d\.phil|doctorate|doctoral', e)))
    mba = int(bool(re.search(r'\bmba\b|m\.b\.a|master\s+of\s+business', e)))
    return pd.Series([elite, phd, mba], index=["elite_uni","has_phd","has_mba"])


bio_feats = d_ceo["CEO Biography"].apply(extract_bio)
edu_feats = d_ceo["CEO Education"].apply(extract_edu)
ceo_df = pd.concat([d_ceo["co"].reset_index(drop=True),
                     bio_feats.reset_index(drop=True),
                     edu_feats.reset_index(drop=True)], axis=1)

print("\n  Extraction rates:")
for col in ["serial_founder","prior_exit","is_technical","is_business","elite_uni","has_phd","has_mba"]:
    print(f"    {col:<25} {ceo_df[col].mean()*100:.1f}%")

df1 = cdf.merge(ceo_df, on="co", how="left")
df1_ceo = df1.dropna(subset=["serial_founder","is_technical"]).copy()
df1_full = df1.dropna(subset=["serial_founder","is_technical","elite_uni"]).copy()

FCTRL = "serial_founder + prior_exit + is_technical + is_business"

print(f"\n  OLS: cross_community β with founder controls")
print(f"  {'Specification':<48} {'β':>9}  SE      p")
print(f"  {'-'*68}")
m_base_sub = smf.ols(f"is_failed ~ cross_community + {CTRL}", data=df1_ceo).fit(cov_type="HC1")
m1a = smf.ols(f"is_failed ~ cross_community + {FCTRL} + {CTRL}", data=df1_ceo).fit(cov_type="HC1")
m1b = smf.ols(f"is_failed ~ cross_community + {FCTRL} + elite_uni + has_phd + has_mba + {CTRL}",
              data=df1_full).fit(cov_type="HC1")
for lbl, m in [("Baseline same subsample", m_base_sub),
               ("+ CEO bio (serial,exit,tech,biz)", m1a),
               ("+ CEO bio + education (elite,PhD,MBA)", m1b)]:
    b=m.params["cross_community"]; se=m.bse["cross_community"]; p=m.pvalues["cross_community"]
    print(f"  {lbl:<48} {b:>+9.5f}  ({se:.4f}) {p:.4f} {sig(p):<5}  N={int(m.nobs):,}")

print(f"\n  Individual founder coefficients (bio + edu model):")
for v in ["serial_founder","prior_exit","is_technical","is_business","elite_uni","has_phd","has_mba"]:
    b=m1b.params.get(v,np.nan); se=m1b.bse.get(v,np.nan); p=m1b.pvalues.get(v,np.nan)
    print(f"    {v:<28} β={b:>+.4f} ({se:.4f}) {sig(p)}")

# 2SLS with founder controls
print(f"\n  2SLS with founder controls:")
nm = pd.read_csv("output/network/network_metrics.csv")
nm["inv_key"] = nm["investor"].str.lower().str.strip()
nm = nm.dropna(subset=["community"]); nm["community"] = nm["community"].astype(int)
inv_to_comm = dict(zip(nm["inv_key"], nm["community"]))
comm_sizes  = nm["community"].value_counts().to_dict()

iv_rows = []
for _, row in cmp[cmp["co"].isin(set(df1_full["co"]))].iterrows():
    invs  = parse_investors(row["All Investors"])
    comms = [inv_to_comm.get(i.lower().strip()) for i in invs]
    comms = [c for c in comms if c is not None]
    if not comms: continue
    sizes = [comm_sizes.get(c, 0) for c in comms]
    iv_rows.append({"co": row["co"], "min_comm_size": int(min(sizes))})

iv_df = pd.DataFrame(iv_rows)
df1_iv = df1_full.merge(iv_df, on="co", how="inner")
df1_iv["iv_min_z"] = (df1_iv["min_comm_size"] - df1_iv["min_comm_size"].mean()) / df1_iv["min_comm_size"].std()

df1_iv["const"] = 1.0
yr   = pd.get_dummies(df1_iv["deal_year"].astype(int), prefix="yr",  drop_first=True, dtype=float)
ctry = pd.get_dummies(df1_iv["country_fe"], prefix="ct", drop_first=True, dtype=float)
founder_cols = ["serial_founder","prior_exit","is_technical","is_business","elite_uni","has_phd","has_mba"]
exog = pd.concat([df1_iv[["const","log_cap","nrounds","specter_novelty"] + founder_cols], yr, ctry],
                 axis=1).astype(float).fillna(0)
res_iv = IV2SLS(df1_iv["is_failed"].astype(float),
                exog,
                df1_iv[["cross_community"]].astype(float),
                df1_iv[["iv_min_z"]].astype(float)).fit(cov_type="robust")

b_iv = res_iv.params.get("cross_community", np.nan)
p_iv = res_iv.pvalues.get("cross_community", np.nan)
kpf  = float(res_iv.first_stage.diagnostics["f.stat"].iloc[0])
print(f"    2SLS β = {b_iv:>+.5f}  {sig(p_iv)}  KP-F = {kpf:.1f}  N = {int(res_iv.nobs):,}")
print(f"    Note: N reduction (6,718→{int(res_iv.nobs):,}) driven by education coverage (58%)")
print(f"    Direction consistent with OLS; power reduced by smaller N")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("BLOCK 2: POSITIVE OUTCOMES — GROWTH SIGNALS")
print("=" * 72)
print(f"\n  {'Outcome':<42} {'β_cross':>9}  SE      p        N")
print(f"  {'-'*70}")
for label, col, dropcol in [
    ("A) log(Employees)",          "log_emp",      "employees"),
    ("B) Growth Rate (composite)", "growth_rate",  "growth_rate"),
    ("C) Growth Rate Percentile",  "growth_pct",   "growth_pct"),
    ("D) Success Probability",     "success_prob", "success_prob"),
]:
    sub = df.dropna(subset=[dropcol]).copy()
    m = smf.ols(f"{col} ~ cross_community + {CTRL}", data=sub).fit(cov_type="HC1")
    b=m.params["cross_community"]; se=m.bse["cross_community"]; p=m.pvalues["cross_community"]
    print(f"  {label:<42} {b:>+9.4f}  ({se:.4f}) {p:.4f} {sig(p):<5} {int(m.nobs):>5,}")

print(f"""
  Interpretation:
    Employees:  +11% more staff (exp(0.103)=1.11). Cross-community companies
                scale their teams more — consistent with better deal flow and
                access to talent networks across community boundaries.
    Growth %ile: +4.3 percentile points*** — consistently higher rank regardless
                of absolute growth rate (which has higher noise, hence n.s.).
    Success Prob: +9.3 points*** on a 2–98 scale (mean≈57). PitchBook's ML
                model independently assigns cross-community companies meaningfully
                higher survival odds — external validation of the main finding.
""")

# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("BLOCK 3: INNOVATION — PATENTS")
print("=" * 72)
print(f"\n  Patent filers: {df['has_patent'].sum():.0f} ({df['has_patent'].mean()*100:.1f}% of {len(df):,})")
print(f"\n  {'Outcome':<47} {'β_cross':>9}  SE      p        N")
print(f"  {'-'*72}")
for label, col, filt in [
    ("A) has_patent — binary, full N",      "has_patent",  None),
    ("B) log(1+total_patents) — full N",    "log_patents", None),
    ("C) log(active_patents) — filers only","log_active",  "has_patent == 1"),
    ("D) log(pending_patents) — pipeline",  "log_pending", None),
]:
    sub = df.query(filt).copy() if filt else df.copy()
    m = smf.ols(f"{col} ~ cross_community + {CTRL}", data=sub).fit(cov_type="HC1")
    b=m.params["cross_community"]; se=m.bse["cross_community"]; p=m.pvalues["cross_community"]
    print(f"  {label:<47} {b:>+9.4f}  ({se:.4f}) {p:.4f} {sig(p):<5} {int(m.nobs):>5,}")

print(f"""
  Interpretation:
    has_patent (+3.7pp**): Cross-community companies are more likely to file
    patents — broader information access across communities may accelerate IP
    development (or attract investors who value IP).
    log(active) negative among filers (†): Among those who file, cross-community
    companies do not accumulate deeper portfolios — possibly quality > quantity.
    Pending pipeline (†): Marginally more pending patents — future IP signal.
""")

# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("BLOCK 4: DETAILED SECTOR ANALYSIS — 39-GROUP INDUSTRY FE")
print("=" * 72)
df4 = df.dropna(subset=["ind_group"]).copy()
print(f"\n  {df4['ind_group'].nunique()} industry groups, {df4['ind_sector'].nunique()} sectors, N={len(df4):,}")
print(f"\n  {'Specification':<52} {'β_cross':>9}  SE      p")
print(f"  {'-'*70}")

for label, formula in [
    ("No industry FE (country+year only)",    f"is_failed ~ cross_community + {CTRL}"),
    ("+ 7-sector FE",                         f"is_failed ~ cross_community + C(ind_sector) + {CTRL}"),
    ("+ 39-group FE",                         f"is_failed ~ cross_community + C(ind_group) + {CTRL}"),
]:
    m = smf.ols(formula, data=df4).fit(cov_type="HC1")
    b=m.params["cross_community"]; se=m.bse["cross_community"]; p=m.pvalues["cross_community"]
    print(f"  {label:<52} {b:>+9.5f}  ({se:.4f}) {p:.4f} {sig(p)}")

print(f"\n  Failure and cross-community rates by sector:")
print(f"  {'Sector':<48} {'N':>6}  {'Fail%':>6}  {'Cross%':>7}")
print(f"  {'-'*68}")
for sec, g in df4.groupby("ind_sector"):
    print(f"  {sec:<48} {len(g):>6,}  {g['is_failed'].mean()*100:>5.1f}%  {g['cross_community'].mean()*100:>6.0f}%")

print(f"""
  Verdict: cross_community β moves only 0.001 when adding 39 industry groups
  (−0.036 → −0.035). The effect is sector-invariant. Healthcare (76% cross-
  community, 13.0% failure) and IT (67%, 7.1%) show the highest bridging rates.
  Energy (59%, 5.1%) is the most specialized sector.
""")

# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("BLOCK 5: COMPETITOR NETWORK (Exploratory — 29% coverage)")
print("=" * 72)

df5 = df.copy()
df5["n_competitors"] = df5["Competitors"].apply(
    lambda s: len([c.strip() for c in re.split(r",\s*", str(s)) if c.strip()])
              if not pd.isna(s) and str(s).strip() else 0)
df5["has_comp"]  = (df5["n_competitors"] > 0).astype(float)
df5["log_ncomp"] = np.log1p(df5["n_competitors"])

df5c = df5[df5["has_comp"] == 1].copy()
df5c["cc_x_comp"] = df5c["cross_community"] * df5c["log_ncomp"]
print(f"\n  N with competitors listed: {df5c.shape[0]:,}  mean={df5c['n_competitors'].mean():.1f}  max={df5c['n_competitors'].max()}")

m5a = smf.ols(f"log_ncomp ~ cross_community + {CTRL}", data=df5c).fit(cov_type="HC1")
m5b = smf.ols(f"is_failed ~ cross_community + log_ncomp + {CTRL}", data=df5c).fit(cov_type="HC1")
m5c = smf.ols(f"is_failed ~ cross_community + log_ncomp + cc_x_comp + {CTRL}", data=df5c).fit(cov_type="HC1")

print(f"\n  A) cross_community → log(n_competitors): β={m5a.params['cross_community']:>+.4f} {sig(m5a.pvalues['cross_community'])}")
print(f"  B) failure controlling for competition:   β_cross={m5b.params['cross_community']:>+.4f} {sig(m5b.pvalues['cross_community'])}  "
      f"β_comp={m5b.params['log_ncomp']:>+.4f} {sig(m5b.pvalues['log_ncomp'])}")
print(f"  C) interaction β={m5c.params['cc_x_comp']:>+.4f} {sig(m5c.pvalues['cc_x_comp'])} — no significant moderation")
print(f"\n  Note: 29% coverage creates severe selection bias (PitchBook records competitors")
print(f"  more often for larger/more prominent companies). Results inconclusive.")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("MASTER SUMMARY: cross_community β across all specifications")
print("=" * 72)
print(f"\n  {'Model':<55} {'β':>8}  p        N")
print(f"  {'-'*75}")
specs = [
    ("Baseline (country FE)",                    smf.ols(f"is_failed ~ cross_community + {CTRL}", data=cdf).fit(cov_type="HC1")),
    ("BLOCK 1: + CEO bio (serial,exit,tech,biz)", m1a),
    ("BLOCK 1: + CEO bio + education",            m1b),
    ("BLOCK 2: outcome = log(Employees)",         smf.ols(f"log_emp ~ cross_community + {CTRL}", data=df.dropna(subset=['employees'])).fit(cov_type="HC1")),
    ("BLOCK 2: outcome = Growth Rate %ile",       smf.ols(f"growth_pct ~ cross_community + {CTRL}", data=df.dropna(subset=['growth_pct'])).fit(cov_type="HC1")),
    ("BLOCK 2: outcome = Success Probability",    smf.ols(f"success_prob ~ cross_community + {CTRL}", data=df.dropna(subset=['success_prob'])).fit(cov_type="HC1")),
    ("BLOCK 3: outcome = has_patent",             smf.ols(f"has_patent ~ cross_community + {CTRL}", data=df).fit(cov_type="HC1")),
    ("BLOCK 4: + 7-sector FE",                   smf.ols(f"is_failed ~ cross_community + C(ind_sector) + {CTRL}", data=df4).fit(cov_type="HC1")),
    ("BLOCK 4: + 39-group FE",                   smf.ols(f"is_failed ~ cross_community + C(ind_group) + {CTRL}", data=df4).fit(cov_type="HC1")),
]
for lbl, m in specs:
    b=m.params.get("cross_community",np.nan); p=m.pvalues.get("cross_community",np.nan)
    print(f"  {lbl:<55} {b:>+8.4f}  {p:.4f} {sig(p):<5} {int(m.nobs):>6,}")
print(f"  {'2SLS + founder controls (Block 1)':<55} {b_iv:>+8.4f}  {p_iv:.4f} {sig(p_iv):<5} {int(res_iv.nobs):>6,}")
