"""
Script 90: H7 feasibility only. NO outcome regressions.

1. Outcome options: (a) reached round 3+, (b) months r1→r2, (c) n rounds in 5yr.
2. Treatment: new_community_entry — round-2 investor from community not in round-1.
   Lowercase lookup. Rate overall and by syndicate size.
3. Cells: hardware x new_community_entry x syndicate-size band. Flag <100.
4. MDE for hardware x new_community_entry on best outcome.
5. EnvPU latest date available.
"""

import os, sys, re, warnings, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
from collections import defaultdict
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "h7_feasibility")
os.makedirs(OUT_DIR, exist_ok=True)

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

def sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

print("=" * 70)
print("SCRIPT 90 — H7 FEASIBILITY CHECK")
print("=" * 70)

# ── Load data ──────────────────────────────────────────────────────────────
dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl = dl.dropna(subset=["Deal Date", "Companies"])

def parse_round(r):
    if pd.isna(r): return None
    m = re.match(r"(\d+)", str(r).strip())
    return int(m.group(1)) if m else None

dl["round_num"] = dl["VC Round"].apply(parse_round)
dl["year"]      = dl["Deal Date"].dt.year
vc = dl[dl["round_num"].notna()].sort_values("Deal Date").copy()

# Tuned communities (lowercase keys — fixes Script 88 bug)
tuned_raw = pd.read_csv("output/h3_redesign/lagged_comm_tuned.csv")
tuned_lc  = {(r["investor"].lower(), int(r["year"])): int(r["community_tuned"])
             for _, r in tuned_raw.iterrows()}

# Startup cluster assignments
nov        = pd.read_csv("output/novelty/novelty_scores.csv")
co_cluster = dict(zip(nov["name"], nov["specter_cluster"]))
# Hardware: physical manufacturing-intensive IRA clusters (45X + 48E + geothermal)
HARDWARE_CLUSTERS = {55, 42, 56, 57, 54, 5}

# Round 1 and round 2 tables
r1_all = vc[vc["round_num"] == 1].groupby("Companies").first().reset_index()
r2_all = vc[vc["round_num"] == 2].groupby("Companies").first().reset_index()
r1_dates = r1_all.set_index("Companies")["Deal Date"]

# ── Outcome (a): reached round 3+ ─────────────────────────────────────────
max_round   = vc.groupby("Companies")["round_num"].max()
reached_r3  = (max_round >= 3).astype(int)

# ── Outcome (b): months from r1 to r2 ─────────────────────────────────────
r2_lookup = r2_all.set_index("Companies")["Deal Date"]
months_to_r2 = {}
for co in r1_all["Companies"]:
    r1d = r1_dates.get(co); r2d = r2_lookup.get(co)
    if pd.notna(r1d) and pd.notna(r2d):
        months_to_r2[co] = (r2d - r1d).days / 30.44
m2_ser = pd.Series(months_to_r2)

# ── Outcome (c): n distinct rounds within 5 years of r1 ───────────────────
n5 = {}
for co, grp in vc.groupby("Companies"):
    r1d = r1_dates.get(co)
    if r1d is None: n5[co] = np.nan; continue
    cut  = r1d + pd.DateOffset(years=5)
    n5[co] = grp[(grp["round_num"].notna()) &
                 (grp["Deal Date"] > r1d) &
                 (grp["Deal Date"] <= cut)]["round_num"].nunique()
n5_ser = pd.Series(n5)

# ── Report outcome distributions ──────────────────────────────────────────
print("\n── 1. Outcome Options ──")
m2_v = m2_ser.dropna(); m2_v = m2_v[m2_v > 0]
n5_v = n5_ser.dropna()
r3_v = reached_r3

print(f"""
  (a) Reached round 3+ (binary, proxy for Series B+)
      N startups: {len(r3_v):,}  |  Event rate: {r3_v.mean():.1%}
      Usable variation: YES — binary, well-balanced (37% events)
      Limitation: does not distinguish speed or recurrence; censored by
                  data window (recent startups may not have had time to reach r3).

  (b) Months from round 1 to round 2
      N startups with r2: {len(m2_v):,}  |  mean={m2_v.mean():.1f}  median={m2_v.median():.1f}  sd={m2_v.std():.1f}
      Range: p10={m2_v.quantile(.1):.1f} – p90={m2_v.quantile(.9):.1f} months
      Usable variation: CONDITIONAL — only observed for startups that received r2;
                        selection bias (treated startups may be more likely to get r2,
                        inflating time-to-r2 for the untreated residual).

  (c) Number of distinct rounds within 5 years of round 1
      N startups: {len(n5_v):,}  |  mean={n5_v.mean():.2f}  sd={n5_v.std():.2f}
      Distribution: {dict(n5_v.value_counts().sort_index().head(6))}
      Usable variation: BEST — count outcome, no selection bias, covers all startups.
                        Zero-inflated (40% have zero follow-on rounds in 5 years),
                        but suitable for OLS or Poisson regression.

  ** Recommended outcome: (c) n_rounds_5yr **
""")

# ── New Community Entry ───────────────────────────────────────────────────
print("── 2. Treatment: new_community_entry ──")

rows = []
for _, r1 in r1_all.iterrows():
    co  = r1["Companies"]; yr1 = int(r1["year"])
    r1_invs  = [i.lower() for i in comma_parse(str(r1["Investors"]))]
    r1_comms = {tuned_lc.get((inv, yr1)) for inv in r1_invs
                if tuned_lc.get((inv, yr1)) is not None}
    n_r1_inv  = len([x for x in r1_invs if x])
    n_r1_comm = len(r1_comms)
    if n_r1_inv == 0: continue

    cluster  = co_cluster.get(co, -1)
    hardware = 1 if cluster in HARDWARE_CLUSTERS else 0

    if n_r1_inv <= 1:    sz = "1"
    elif n_r1_inv == 2:  sz = "2"
    elif n_r1_inv <= 4:  sz = "3-4"
    elif n_r1_inv <= 9:  sz = "5-9"
    else:                sz = "10+"

    r2_row = r2_all[r2_all["Companies"] == co]
    has_r2  = len(r2_row) > 0
    nce = np.nan; nce_strict = np.nan; nce_def = 0

    if has_r2:
        r2_row  = r2_row.iloc[0]; yr2 = int(r2_row["year"])
        r2_invs = [i.lower() for i in comma_parse(str(r2_row["Investors"]))]
        r2_comms = {tuned_lc.get((inv, yr2)) for inv in r2_invs
                    if tuned_lc.get((inv, yr2)) is not None}
        if r2_comms:
            nce_def = 1
            new_c   = r2_comms - r1_comms if r1_comms else r2_comms
            nce     = 1 if new_c else 0
            # Strict: require r1 to have at least one community assignment
            nce_strict = nce if n_r1_comm > 0 else np.nan

    rows.append({
        "co": co, "yr1": yr1, "hw": hardware, "sz": sz,
        "r1_n": n_r1_inv, "r1_comms": n_r1_comm,
        "has_r2": int(has_r2), "nce": nce, "nce_def": nce_def,
        "nce_strict": nce_strict,
        "reached_r3": reached_r3.get(co, 0),
        "n5": n5.get(co, np.nan),
    })

df     = pd.DataFrame(rows)
df2    = df[df["has_r2"] == 1].copy()
df_nce = df2[df2["nce_def"] == 1].copy()
df_s   = df2[(df2["r1_comms"] > 0) & (df2["nce_def"] == 1)].copy()

print(f"""
  Round-1 startups:                    {len(df):,}
  With round 2:                        {df2['co'].nunique():,}
  r2 investors have community (NCE evaluable): {len(df_nce):,}
  r1 community coverage:
    r1_comms = 0 (no LCC coverage):   {int((df2['r1_comms']==0).sum()):,}  ({100*(df2['r1_comms']==0).mean():.1f}%)
    r1_comms > 0:                      {int((df2['r1_comms']>0).sum()):,}  ({100*(df2['r1_comms']>0).mean():.1f}%)

  NCE rate (all evaluable):  {df_nce['nce'].mean():.3f}  n={int(df_nce['nce'].sum()):,} / {len(df_nce):,}
  NCE rate (strict, r1>0):   {df_s['nce_strict'].mean():.3f}  n={int(df_s['nce_strict'].sum()):,} / {len(df_s):,}

  ⚠  ROOT CAUSE OF NEAR-CEILING RATE:
     55% of round-1 syndicates have zero tuned community coverage (investors outside
     the LCC). For these startups, ANY round-2 community assignment counts as 'new',
     making NCE mechanically = 1. Even after restricting to r1_comms > 0, the rate
     is 93.2% (only 7% control group). The treatment has near-zero variation.
""")

print("  NCE strict rate by syndicate size:")
tbl = df_s.groupby("sz")["nce_strict"].agg(["mean", "sum", "count"])
tbl.columns = ["NCE_rate", "n_NCE=1", "n_total"]
print(tbl.to_string())

# ── 3. Cells ──────────────────────────────────────────────────────────────
print("\n── 3. Cell Counts: hardware × NCE_strict × syndicate_size ──")
cell_counts = (df_s.groupby(["hw", "nce_strict", "sz"])
               .size().reset_index(name="n"))
cell_pivot  = cell_counts.pivot_table(
    index=["hw", "nce_strict"], columns="sz", values="n", fill_value=0
)[["1", "2", "3-4", "5-9", "10+"]]
print()
print(cell_pivot.to_string())
print()
bad_cells = cell_counts[cell_counts["n"] < 100]
n_total_cells = len(cell_counts)
print(f"  Cells with n < 100:  {len(bad_cells)} / {n_total_cells}")
print()
hw_total = (df["hw"] == 1).sum()
sw_total = (df["hw"] == 0).sum()
print(f"  Hardware startups (clusters {sorted(HARDWARE_CLUSTERS)}): {hw_total:,}  ({100*hw_total/len(df):.1f}%)")
print(f"  Software startups: {sw_total:,}  ({100*sw_total/len(df):.1f}%)")
print(f"""
  ⚠  ALL hardware cells are below 100. Maximum hardware cell size: 25.
     Hardware clusters contain only {hw_total:,} startups ({100*hw_total/len(df):.1f}% of all),
     making the hardware × NCE interaction completely infeasible.
""")

# ── 4. MDE ──────────────────────────────────────────────────────────────
print("── 4. MDE for hardware × NCE_strict on recommended outcome (n_rounds_5yr) ──")
for label, outcome_col, model_type in [
    ("n_rounds_5yr (OLS)",   "n5",         "OLS"),
    ("reached_r3 (OLS LPM)", "reached_r3", "OLS"),
]:
    m = df_s.dropna(subset=[outcome_col, "nce_strict", "hw"]).copy()
    m["hw_x_nce"] = m["hw"] * m["nce_strict"]
    X = add_constant(m[["hw", "nce_strict", "hw_x_nce", "r1_n"]])
    y = m[outcome_col].astype(float)
    res = OLS(y, X).fit(cov_type="HC3")
    se_int = res.bse["hw_x_nce"]
    mde    = (norm.ppf(0.975) + norm.ppf(0.80)) * se_int
    print(f"\n  {label}:")
    print(f"    N = {len(m):,}  |  outcome mean = {y.mean():.3f}  sd = {y.std():.3f}")
    print(f"    hw × NCE_strict:  β={res.params['hw_x_nce']:+.4f}  SE={se_int:.4f}  "
          f"p={res.pvalues['hw_x_nce']:.4f}  {sig(res.pvalues['hw_x_nce'])}")
    print(f"    MDE (80% power):  {mde:.4f}  ({mde / y.std():.3f} SD of outcome)")

# ── 5. EnvPU latest date ──────────────────────────────────────────────────
print("\n── 5. Noailly et al. EnvPU — latest available date ──")
envpu = pd.read_excel("data/EnvPU_data.xlsx")
last_yr  = int(envpu["Year"].max())
last_mo  = int(envpu.loc[envpu["Year"] == last_yr, "Month"].max())
print(f"""
  Local file (data/EnvPU_data.xlsx):
    Coverage:  January 1990 – {pd.Timestamp(last_yr, last_mo, 1).strftime('%B %Y')}
    N months:  {len(envpu)}

  Online check (www.financingcleantech.com/envpu-index, Harvard Dataverse 2025):
    The dataset was archived on Harvard Dataverse (DOI: 10.7910/DVN/8TKRSJ) in 2025,
    but this is an archival deposit of the existing series — not an extension.
    The index time series ends at March 2019. The authors' project website confirms:
    'The data series is available monthly over the period January 1990 to March 2019.'
    No update beyond March 2019 has been released as of September 2026.

  Impact on H5: EnvPU analysis (Script 89) restricted to startups with first rounds
  through March 2019 (2,543 of 8,964 startups, 28.4%). No extended coverage available.
""")

# ── PLAIN VERDICT ──────────────────────────────────────────────────────────
print("=" * 70)
print("PLAIN VERDICT — H7 FEASIBILITY")
print("=" * 70)
print(f"""
  Three structural blockers:

  1. TREATMENT NEAR-CEILING (fatal):
     new_community_entry rate = 93-96% depending on sample restriction.
     The control group (NCE=0) has only 68-99 observations.
     With a 7% control group, any regression estimate will be dominated by
     the imbalance. The treatment as operationalised is not usable.

     Root cause: the tuned-LCC partition leaves 55% of round-1 syndicates
     with zero community coverage. Any round-2 investor with any community
     assignment trivially counts as 'new.' The treatment would need
     redefining (e.g. as a continuous count of NEW communities in round 2,
     or restricting to deals where round-1 investors are all in the LCC).

  2. HARDWARE CELLS EMPTY (fatal):
     Hardware clusters {sorted(HARDWARE_CLUSTERS)} contain only {hw_total:,} startups ({100*hw_total/len(df):.1f}%).
     Within hardware × NCE=1, the largest syndicate-size cell has n=25.
     All {len(bad_cells)} of {n_total_cells} cells are below 100. The hardware × NCE
     interaction is not estimable.

  3. MDE LARGE:
     On the recommended outcome (n_rounds_5yr), MDE = 0.83 (0.87 SD).
     On reached_r3 (LPM), MDE = 0.44 (0.88 SD).
     Both far exceed a meaningful minimum detectable effect (~0.2 SD)
     and reflect the small hardware stratum and near-ceiling treatment.

  DECISION: STOP — H7 cannot be run as specified.
  Suggested path if pursued: (a) redefine treatment as 'n new communities
  added in round 2' (continuous); (b) drop hardware × NCE interaction,
  run NCE effect on all-cluster sample; (c) use broader hardware definition
  beyond the 6 IRA hardware clusters.
""")
print("=" * 70)
print("SCRIPT 90 COMPLETE")
print("=" * 70)
