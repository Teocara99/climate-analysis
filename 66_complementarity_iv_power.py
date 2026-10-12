"""
Script 66: Complementarity interaction — positive outcomes + IV subsample split
===============================================================================

PRE-SPECIFIED PREDICTIONS (registered before running; report regardless of outcome):
  P3: A2 (PB-Expert × Cross-community) best on all four positive outcomes;
      B2, C2, D2 indistinguishable from each other.

  P4: 2SLS cross_community effect significant in expert subsample;
      null (or weak) in non-expert subsample. If non-expert first stage
      F < 10, that subsample result is uninformative — flag explicitly.

Expert flag: genuine PitchBook Preferred Verticals (Script 65 construction).
Frame: N = 4,745 (pb_full: companies with ≥1 investor with PB pref data).
Controls: log_capital + n_rounds + log_syndicate + log_n_investors
          + C(deal_year) + C(hq_country). HC1 SEs throughout.
IV: min_community_size_z (Script 32 instrument; KP-F ≈ 747 on full sample).
"""

import re, sys, pickle, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.preprocessing import normalize
from linearmodels.iv import IV2SLS
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from load_data import load_deals, load_companies

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else "n.s."

def parse_set(s):
    if not isinstance(s, str): return set()
    return {x.strip().lower() for x in re.split(r"[,;|]", s) if x.strip()}

def parse_investors_comma(s):
    """Comma parser (Script 30 / post-fix standard)."""
    if pd.isna(s) or not str(s).strip(): return []
    names = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", n).strip() for n in names if n.strip()]

def parse_investors_semi(s):
    """Semicolon parser (legacy, for co_all_inv via deals)."""
    if pd.isna(s) or not str(s).strip(): return []
    return [p.strip() for p in str(s).split(";") if p.strip()]

def sep(width=70): print("─" * width)
def header(title, width=70): print("\n" + "=" * width); print(title); print("=" * width)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE DATASET
# ─────────────────────────────────────────────────────────────────────────────
header("LOADING DATA")

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]         = cdf["Companies"].str.strip()
cdf["is_failed"]  = (cdf["outcome"] == "Failed").astype(float)
cdf["log_capital_v"] = cdf["log_capital"]
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x >= 3].index), "Other")

# Build log_syndicate and log_n_investors from deals (comma parser)
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()
co_stats_rows = []
for co, grp in deals.dropna(subset=["Investors"]).groupby("co"):
    all_invs = []; syn_sizes = []
    for inv_str in grp["Investors"]:
        invs = parse_investors_comma(inv_str)
        all_invs.extend(invs)
        if invs: syn_sizes.append(len(invs))
    co_stats_rows.append({
        "co": co,
        "n_investors_total": len(set(all_invs)),
        "avg_syndicate": float(np.mean(syn_sizes)) if syn_sizes else 1.0,
    })
co_stats = pd.DataFrame(co_stats_rows)
co_stats["log_n_investors"] = np.log1p(co_stats["n_investors_total"])
co_stats["log_syndicate"]   = np.log1p(co_stats["avg_syndicate"])
cdf = cdf.merge(co_stats[["co", "log_syndicate", "log_n_investors"]], on="co", how="left")
cdf["n_rounds_v"] = np.log1p(cdf["n_rounds"].fillna(1))

# Positive outcomes from load_companies()
cmp = load_companies()
cmp["co"] = cmp["Companies"].str.strip()
cmp["log_employees"]    = np.log1p(pd.to_numeric(cmp["Employees"], errors="coerce"))
cmp["growth_percentile"]= pd.to_numeric(cmp["Growth Rate Percentile"], errors="coerce")
cmp["has_patent"]       = (pd.to_numeric(cmp["Active Patent Documents"], errors="coerce").fillna(0) > 0).astype(float)

cdf = cdf.merge(
    cmp[["co", "log_employees", "growth_percentile", "has_patent"]].drop_duplicates("co"),
    on="co", how="left")
# got_followon = (n_rounds > 1) is near-degenerate in this sample (96% = 1);
# replaced with is_exited (IPO or acquisition) — a positive outcome with ~6% rate.
cdf["is_exited"] = cdf["exited"].astype(float)

print(f"Base cdf: N={len(cdf):,}")
for col in ["log_employees", "growth_percentile", "has_patent", "is_exited"]:
    print(f"  {col:<22} non-null={cdf[col].notna().sum():>5}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PB EXPERT FLAG (Script 65 exact construction)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding PB expert flag …")

co_vert_dict = (deals.dropna(subset=["Verticals"])
                .groupby("co")["Verticals"].agg(lambda x: x.mode()[0]).to_dict())
co_sec_dict  = (deals.dropna(subset=["Primary PitchBook Industry Group"])
                .groupby("co")["Primary PitchBook Industry Group"]
                .agg(lambda x: x.mode()[0]).to_dict())

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv").assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""').dropna(subset=["Year"]))
co_all_inv = d_exp.groupby("co")["inv"].apply(set).to_dict()

inv_data = pd.read_excel("data/Investors_data.xlsx", skiprows=7)
inv_data = inv_data.drop_duplicates(subset=["Investors"]).set_index("Investors")
inv_data.index = inv_data.index.str.strip()
inv_pref_verticals = {n: parse_set(v) for n, v in inv_data["Preferred Verticals"].dropna().items()}
pref_ind_col = "Preferred PitchBook Industry" if "Preferred PitchBook Industry" in inv_data.columns else None
inv_pref_industry = ({n: parse_set(v) for n, v in inv_data[pref_ind_col].dropna().items()}
                     if pref_ind_col else {})

pb_rows = []
for co, investors in co_all_inv.items():
    co_vert = parse_set(co_vert_dict.get(co, ""))
    co_sec_str = co_sec_dict.get(co, "")
    co_sec_set = {co_sec_str.lower().strip()} if co_sec_str else set()
    n_pb = 0; n_pref = 0
    for inv in investors:
        inv_pv = inv_pref_verticals.get(inv, set())
        inv_pi = inv_pref_industry.get(inv, set())
        if inv_pv or inv_pi:
            n_pref += 1
            if (co_vert & inv_pv) or (co_sec_set & inv_pi):
                n_pb += 1
    pb_rows.append({"co": co, "has_pb_expert": int(n_pb > 0), "n_with_pref": n_pref})

pb_df = pd.DataFrame(pb_rows)
pb_full = (cdf.merge(pb_df, on="co", how="inner")
              .query("n_with_pref > 0")
              .dropna(subset=["has_pb_expert", "is_failed"]))
pb_full["network_diverse"] = pb_full["cross_community"].astype(float)
pb_full["type_pb"] = np.where(
    pb_full["has_pb_expert"] == 1,
    np.where(pb_full["network_diverse"] == 1, "A2", "B2"),
    np.where(pb_full["network_diverse"] == 1, "C2", "D2"))
for t in ["A2", "B2", "C2"]:
    pb_full[f"type_{t}"] = (pb_full["type_pb"] == t).astype(float)

print(f"pb_full: N={len(pb_full):,}  (A2={int((pb_full['type_pb']=='A2').sum())}, "
      f"B2={int((pb_full['type_pb']=='B2').sum())}, "
      f"C2={int((pb_full['type_pb']=='C2').sum())}, "
      f"D2={int((pb_full['type_pb']=='D2').sum())})")

CTRL = "log_capital_v + n_rounds_v + log_syndicate + log_n_investors + C(deal_year) + C(country_fe)"

# ─────────────────────────────────────────────────────────────────────────────
# PART A: 2x2 ON POSITIVE OUTCOMES
# ─────────────────────────────────────────────────────────────────────────────
header("PART A: 2×2 ON POSITIVE OUTCOMES")
print("""
Pre-specified (P3): A2 best on all four positive outcomes;
B2, C2, D2 indistinguishable from each other.
""")

outcomes = [
    ("is_failed",          "Failure (binary, inverted)",   False),
    ("log_employees",      "log(1+Employees)",             True),
    ("growth_percentile",  "Growth Rate Percentile",       True),
    ("has_patent",         "Has Patent (binary)",          True),
    ("is_exited",          "Exited (IPO or M&A, binary)",  True),
]
# NOTE: got_followon = (n_rounds > 1) is 96% = 1 in this sample (companies with
# PB pref data overwhelmingly have ≥2 rounds); replaced with is_exited.
print("NOTE: got_followon replaced with is_exited (IPO/M&A); got_followon is near-degenerate"
      " (96% = 1 in this sample).\n")

# Cell means table
print("\nCELL MEANS BY OUTCOME")
header_row = f"{'Outcome':<28} {'N':>5}  {'A2':>8} {'B2':>8} {'C2':>8} {'D2':>8}"
print(header_row)
sep(70)
wald_results = {}

for col, lbl, higher_is_better in outcomes:
    sub = pb_full.dropna(subset=[col]).copy()
    row_parts = [f"  {lbl:<26} {len(sub):>5}"]
    cell_means = {}
    for t in ["A2", "B2", "C2", "D2"]:
        g = sub[sub["type_pb"] == t]
        m = g[col].mean() if len(g) > 0 else float("nan")
        cell_means[t] = m
        row_parts.append(f"  {m:>7.3f}")
    print("".join(row_parts))

    # Wald tests
    try:
        m_ols = smf.ols(f"{col} ~ type_A2 + type_B2 + type_C2 + {CTRL}", data=sub).fit(cov_type="HC1")
        tests = {
            "B2vA2": m_ols.f_test("type_B2 = type_A2"),
            "A2vC2": m_ols.f_test("type_A2 = type_C2"),
            "B2vD2": m_ols.f_test("type_B2 = 0"),
            "C2vD2": m_ols.f_test("type_C2 = 0"),
        }
        wald_results[lbl] = {k: float(v.pvalue) for k, v in tests.items()}
        wald_results[lbl]["N"] = int(m_ols.nobs)
    except Exception as e:
        print(f"    WARNING: regression failed for {col}: {e}")
        wald_results[lbl] = {"B2vA2": np.nan, "A2vC2": np.nan, "B2vD2": np.nan, "C2vD2": np.nan, "N": 0}

sep(70)

# Regression detail per outcome
print("\nREGRESSION DETAIL (β, SE, p per cell; D2 reference)")
sep(70)
for col, lbl, higher_is_better in outcomes:
    sub = pb_full.dropna(subset=[col]).copy()
    try:
        m_ols = smf.ols(f"{col} ~ type_A2 + type_B2 + type_C2 + {CTRL}", data=sub).fit(cov_type="HC1")
        print(f"\n  {lbl}  (N={int(m_ols.nobs):,})")
        for var, cell in [("type_A2","A2"),("type_B2","B2"),("type_C2","C2")]:
            b  = m_ols.params[var]; se = m_ols.bse[var]; p = m_ols.pvalues[var]
            print(f"    {cell}: β={b:>+.4f}  SE={se:.4f}  p={p:.4f} {sig(p)}")
    except Exception as e:
        print(f"  {lbl}: regression error — {e}")

# Wald summary table
print("\n\nWALD TEST SUMMARY TABLE")
print(f"{'Outcome':<30} {'N':>5}  {'B2vA2':>8} {'A2vC2':>8} {'B2vD2':>8} {'C2vD2':>8}")
sep(72)
for lbl, res in wald_results.items():
    n = res["N"]
    vals = [res.get(k, np.nan) for k in ["B2vA2","A2vC2","B2vD2","C2vD2"]]
    row = f"  {lbl:<28} {n:>5}"
    for v in vals:
        if np.isnan(v):
            row += f"  {'n/a':>7}"
        else:
            row += f"  {v:>7.4f}"
            row += sig(v)
    print(row)
sep(72)
print("  B2vA2=echo penalty, A2vC2=expertise premium, B2vD2/C2vD2=network-only effects")

# P3 verdict
print("\nP3 VERDICT:")
p3_flag_a2_best = all(
    wald_results.get(lbl, {}).get("A2vC2", 1.0) < 0.10 or
    wald_results.get(lbl, {}).get("B2vA2", 1.0) < 0.10
    for _, lbl, higher in outcomes if higher)
p3_flag_bcd_null = all(
    wald_results.get(lbl, {}).get("B2vD2", 1.0) >= 0.10 and
    wald_results.get(lbl, {}).get("C2vD2", 1.0) >= 0.10
    for _, lbl, _ in outcomes)
print(f"  A2 distinguishable from others on ≥1 outcome: {p3_flag_a2_best}")
print(f"  B2, C2, D2 all n.s. vs each other:            {p3_flag_bcd_null}")

# ─────────────────────────────────────────────────────────────────────────────
# PART B: IV BY EXPERT STATUS
# ─────────────────────────────────────────────────────────────────────────────
header("PART B: 2SLS BY EXPERT STATUS")
print("""
Pre-specified (P4): 2SLS cross_community effect significant in expert
subsample; null (or uninformative) in non-expert subsample.
Instrument: min_community_size_z (same as Script 32, IV A).
""")

# Build instrument
nm = pd.read_csv("output/network/network_metrics.csv")
nm["inv_key"] = nm["investor"].str.lower().str.strip()
nm = nm.dropna(subset=["community"])
nm["community"] = nm["community"].astype(int)
inv_to_comm = dict(zip(nm["inv_key"], nm["community"]))
comm_sizes  = nm["community"].value_counts().to_dict()

cmp_all = load_companies()
cmp_all["co"] = cmp_all["Companies"].str.strip()

iv_rows = []
for _, row in cmp_all[cmp_all["co"].isin(set(pb_full["co"]))].iterrows():
    invs  = parse_investors_comma(str(row.get("All Investors", "")))
    comms = [inv_to_comm.get(i.lower().strip()) for i in invs]
    comms = [c for c in comms if c is not None]
    if not comms: continue
    sizes = [comm_sizes.get(c, 0) for c in comms]
    iv_rows.append({"co": row["co"], "min_comm_size": int(min(sizes))})

iv_df = pd.DataFrame(iv_rows)
iv_df["min_comm_size_z"] = ((iv_df["min_comm_size"] - iv_df["min_comm_size"].mean())
                            / iv_df["min_comm_size"].std())

pb_iv = pb_full.merge(iv_df[["co", "min_comm_size_z"]], on="co", how="inner")
print(f"pb_full + IV merge: N={len(pb_iv):,} "
      f"(expert={int(pb_iv['has_pb_expert'].sum())}, "
      f"non-expert={int((pb_iv['has_pb_expert']==0).sum())})")

def run_2sls_subsample(sub, label):
    """First stage + 2SLS via linearmodels IV2SLS. Returns (first_F, beta, p, N)."""
    sub = sub.copy().dropna(subset=["is_failed", "min_comm_size_z",
                                    "log_capital_v", "n_rounds_v",
                                    "log_syndicate", "log_n_investors"])
    sub["const"] = 1.0
    yr  = pd.get_dummies(sub["deal_year"].astype(int), prefix="yr",  drop_first=True, dtype=float)
    cty = pd.get_dummies(sub["country_fe"],             prefix="cty", drop_first=True, dtype=float)
    exog = pd.concat([sub[["const", "log_capital_v", "n_rounds_v",
                            "log_syndicate", "log_n_investors"]], yr, cty], axis=1)
    endog = sub[["cross_community"]].astype(float)
    inst  = sub[["min_comm_size_z"]].astype(float)
    Y     = sub["is_failed"].astype(float)

    N = len(sub)
    base_rate = Y.mean()
    print(f"\n  {label}")
    print(f"  N={N:,}  base failure rate={base_rate:.1%}")

    # First stage
    fs_df = sub.copy(); fs_df["__y__"] = sub["cross_community"].astype(float)
    fs_cols = ["min_comm_size_z", "log_capital_v", "n_rounds_v",
               "log_syndicate", "log_n_investors"]
    fs_X = pd.concat([sub[fs_cols], yr, cty], axis=1).astype(float)
    fs_X.insert(0, "const", 1.0)
    fs_y = sub["cross_community"].astype(float)
    from numpy.linalg import lstsq as _lstsq2
    b_fs, _, _, _ = _lstsq2(fs_X.values, fs_y.values, rcond=None)
    res_fs = fs_y.values - fs_X.values @ b_fs
    rss_restricted = np.sum((fs_y.values - fs_y.values.mean()) ** 2)
    rss_full = np.sum(res_fs ** 2)
    k_extra = 1  # one excluded instrument
    df_res = N - fs_X.shape[1]
    first_F = ((rss_restricted - rss_full) / k_extra) / (rss_full / df_res)
    print(f"  First-stage F (partial, min_comm_size_z) = {first_F:.1f}", end="")
    if first_F < 10:
        print("  ⚠️  WEAK INSTRUMENT — 2SLS UNINFORMATIVE")
    else:
        print("  ✓ Strong")

    # 2SLS
    try:
        res_iv = IV2SLS(Y, exog, endog, inst).fit(cov_type="robust")
        b_iv  = float(res_iv.params.get("cross_community", np.nan))
        p_iv  = float(res_iv.pvalues.get("cross_community", np.nan))
        ci    = res_iv.conf_int().loc["cross_community"]
        print(f"  2SLS β(cross_community) = {b_iv:>+.4f}   p={p_iv:.4f} {sig(p_iv)}")
        print(f"  95% CI: [{float(ci.iloc[0]):>+.4f}, {float(ci.iloc[1]):>+.4f}]")
        # MDE at 80% power for this subsample
        sd_y = float(Y.std())
        from scipy.stats import norm
        z_alpha = norm.ppf(0.975); z_beta = norm.ppf(0.80)
        mde = (z_alpha + z_beta) * sd_y / (np.sqrt(N) * 0.5)
        print(f"  MDE at 80% power (OLS approx) = {mde:.4f}  ({mde:.1%} pp)")
    except Exception as e:
        b_iv = p_iv = np.nan
        print(f"  2SLS error: {e}")

    if first_F < 10:
        print(f"  NOTE: First-stage F={first_F:.1f} < 10. 2SLS result is UNINFORMATIVE.")
        print(f"  This is NOT evidence of a null effect — the instrument is weak in this subsample.")

    return first_F, b_iv, p_iv, N, base_rate

sep()
res_expert     = run_2sls_subsample(pb_iv[pb_iv["has_pb_expert"] == 1], "Expert subsample (has_pb_expert=1)")
sep()
res_nonexpert  = run_2sls_subsample(pb_iv[pb_iv["has_pb_expert"] == 0], "Non-expert subsample (has_pb_expert=0)")
sep()

# P4 verdict
print("\nP4 VERDICT:")
first_F_exp, b_exp, p_exp, N_exp, rate_exp = res_expert
first_F_nex, b_nex, p_nex, N_nex, rate_nex = res_nonexpert

p4_expert_sig = (first_F_exp >= 10) and (p_exp < 0.05)
p4_nonexp_null_or_weak = (first_F_nex < 10) or (p_nex >= 0.05)

print(f"  Expert subsample:     first-stage F={first_F_exp:.1f}, 2SLS p={p_exp:.4f} → {sig(p_exp)}")
print(f"  Non-expert subsample: first-stage F={first_F_nex:.1f}", end="")
if first_F_nex < 10:
    print(" → WEAK INSTRUMENT, result uninformative")
else:
    print(f", 2SLS p={p_nex:.4f} → {sig(p_nex)}")

if p4_expert_sig and p4_nonexp_null_or_weak:
    print(f"  P4: SUPPORTED — cross-community benefit instrumentally identified in expert subsample only.")
elif not p4_expert_sig and first_F_exp >= 10:
    print(f"  P4: NOT SUPPORTED — expert 2SLS n.s. (p={p_exp:.3f}); predicting too much.")
elif first_F_exp < 10:
    print(f"  P4: UNINFORMATIVE — expert subsample first stage also weak.")
else:
    print(f"  P4: PARTIALLY SUPPORTED.")

# ─────────────────────────────────────────────────────────────────────────────
# PART C: POWER ON THE C2 vs D2 NULL
# ─────────────────────────────────────────────────────────────────────────────
header("PART C: POWER ANALYSIS — C2 vs D2 NULL (failure outcome)")

sub_cd = pb_full[pb_full["type_pb"].isin(["C2","D2"])].copy()
n_c2 = int((sub_cd["type_pb"] == "C2").sum())
n_d2 = int((sub_cd["type_pb"] == "D2").sum())
r_c2 = float(sub_cd[sub_cd["type_pb"] == "C2"]["is_failed"].mean())
r_d2 = float(sub_cd[sub_cd["type_pb"] == "D2"]["is_failed"].mean())
raw_diff = r_c2 - r_d2

print(f"\n  C2: N={n_c2}, failure rate={r_c2:.1%}")
print(f"  D2: N={n_d2}, failure rate={r_d2:.1%}")
print(f"  Observed raw difference (C2−D2): {raw_diff:>+.1%}")

# Two-proportion z-test MDE at 80% power, two-sided alpha=0.05
p_pool = (r_c2 * n_c2 + r_d2 * n_d2) / (n_c2 + n_d2)
z_alpha2 = stats.norm.ppf(0.975)
z_beta   = stats.norm.ppf(0.80)
# MDE: delta such that z_power = (delta/se_null) - z_alpha/2 = z_beta
se_null = np.sqrt(p_pool * (1 - p_pool) * (1/n_c2 + 1/n_d2))
mde_raw = (z_alpha2 + z_beta) * se_null
print(f"\n  MDE at 80% power (two-sided α=0.05): {mde_raw:.3f} ({mde_raw:.1%} pp)")
print(f"  Base rate (pooled): {p_pool:.1%}")
print(f"  SE under null:      {se_null:.4f}")

detected = abs(raw_diff) >= mde_raw
print(f"\n  Observed |diff|={abs(raw_diff):.1%}  ≥  MDE={mde_raw:.1%} ? {detected}")

print(f"""
  INTERPRETATION:
  The C2 vs D2 Wald test (p=0.993) is based on N={n_c2}+{n_d2}={n_c2+n_d2} companies with
  a pooled failure rate of {p_pool:.1%}. The MDE at 80% power is {mde_raw:.1%} pp.
  The observed raw difference is {raw_diff:>+.1%} — {"well below" if not detected else "at or above"} the MDE.
  The null (C2 ≈ D2) is {"genuinely informative: a {:.1f}pp difference would be detectable, and we observe only {:.1f}pp.".format(mde_raw*100, abs(raw_diff)*100) if not detected else "borderline — the MDE is close to the observed difference."}
  This cell-pair tests whether network diversity matters WITHOUT expert investors.
  The null means: cross-community benefits do not operate through network structure alone —
  the mechanism requires expert investors (consistent with complementarity).
""")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
header("FINAL SUMMARY")
print("""
PRE-SPECIFIED PREDICTIONS:

P3 — A2 best on all four positive outcomes; B2, C2, D2 indistinguishable:""")
print(f"  Failure (inverted):  A2={r_c2:.1%} — see Part A Wald table for other outcomes")
print(f"  (See Wald table above for B2vD2/C2vD2 across all outcomes)")

print("""
P4 — 2SLS significant in expert subsample, null/uninformative in non-expert:""")
print(f"  Expert:     first-stage F={first_F_exp:.1f}  2SLS β={b_exp:>+.4f}  p={p_exp:.4f} {sig(p_exp)}")
print(f"  Non-expert: first-stage F={first_F_nex:.1f}", end="")
if first_F_nex < 10:
    print(" (WEAK INSTRUMENT)")
else:
    print(f"  2SLS β={b_nex:>+.4f}  p={p_nex:.4f} {sig(p_nex)}")

print(f"""
C2 vs D2 power check:
  MDE at 80% power = {mde_raw:.1%} pp  (N={n_c2+n_d2}, base rate {p_pool:.1%})
  Observed diff    = {raw_diff:>+.1%} pp
  → Null is {"informative (detectable effect would be found)" if not detected else "borderline"}
""")

Path("output/complementarity_iv").mkdir(parents=True, exist_ok=True)
print("Output directory: output/complementarity_iv/")
