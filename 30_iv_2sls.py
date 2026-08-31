"""
30 — 2SLS Instrumental Variable Regressions
Cross-community → failure, addressing endogeneity.

IV A: Lead investor prior community reach (# distinct communities
      co-invested with in 2 years before this deal)
IV B: Average pairwise geographic distance (km) between syndicate investors
IV C: Both together — Hansen J-test for overidentification

All models: year FE, controls = company_age, log_total_funding, region, novelty
Diagnostics: First-stage F, Cragg-Donald, Kleibergen-Paap, Hausman (DWH)
"""

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from collections import defaultdict
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from scipy import stats
from linearmodels.iv import IV2SLS

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
OUT  = ROOT / "output" / "iv_2sls"
OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

def parse_investors(s):
    if pd.isna(s): return []
    parts = [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]
    return [re.sub(r"\s*\([^)]*\)\s*$", "", p).strip() for p in parts if p]


# ── Geographic centroid coordinates (km proxy) ────────────────────────────────
GEO_COORD = {
    "Americas":    (37.0,  -96.0),
    "Europe":      (50.0,   10.0),
    "Asia":        (30.0,  105.0),
    "Oceania":    (-25.0,  135.0),
    "Middle East": (25.0,   45.0),
    "Africa":       (0.0,   20.0),
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data …")

    cdf = pd.read_csv(ROOT / "output/cpu/cpu_company_dataset.csv")
    cdf["company_key"] = cdf["Companies"].str.lower().str.strip()
    cdf["is_failed"]   = (cdf["outcome"] == "Failed").astype(float)
    cdf["log_cap"]     = np.log1p(cdf["Total Raised"].fillna(0))
    cdf["deal_year"]   = cdf["deal_year"].fillna(2018).astype(int)

    nm = pd.read_csv(ROOT / "output/network/network_metrics.csv")
    inv_to_comm = dict(zip(nm["investor"].str.lower(), nm["community"].astype("Int64")))

    ip = pd.read_csv(ROOT / "output/investor_profiles.csv")
    inv_to_geo = dict(zip(ip["investor"].str.lower(), ip["geo_focus"]))

    deals = load_deals()
    deals["date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["date"]).copy()
    deals["company_key"] = deals["Companies"].str.lower().str.strip()

    # Year Founded → company age at first deal
    year_founded = (deals.dropna(subset=["Year Founded"])
                    .groupby("company_key")["Year Founded"].first()
                    .astype(int))
    cdf["year_founded"] = cdf["company_key"].map(year_founded)
    cdf["company_age"]  = (cdf["deal_year"] - cdf["year_founded"]).clip(0, 30)

    print(f"  Companies: {len(cdf):,}  |  with company_age: {cdf['company_age'].notna().sum():,}")
    return cdf, deals, inv_to_comm, inv_to_geo


# ═══════════════════════════════════════════════════════════════════════════════
# IV A: LEAD INVESTOR PRIOR COMMUNITY REACH
# ═══════════════════════════════════════════════════════════════════════════════

def build_iv_a(cdf, deals, inv_to_comm):
    print("\nBuilding IV A (lead investor prior community reach) …")

    # Build per-investor deal history:
    # investor → sorted list of (date, set_of_co_investor_communities)
    investor_history = defaultdict(list)
    for _, row in deals.iterrows():
        inv_list = parse_investors(row["Investors"])
        if not inv_list: continue
        inv_lower = [i.lower() for i in inv_list if i]
        date = row["date"]
        comms_for_row = {i: inv_to_comm.get(i) for i in inv_lower}
        for inv in inv_lower:
            co_comms = frozenset(c for i, c in comms_for_row.items()
                                 if i != inv and c is not None)
            investor_history[inv].append((date, co_comms))

    # Sort each investor's history by date
    for inv in investor_history:
        investor_history[inv].sort(key=lambda x: x[0])

    # Identify lead investor per company (first deal, lead column → first in investors)
    deals_sorted = (deals.sort_values("date")
                    .dropna(subset=["Investors"])
                    .drop_duplicates("company_key", keep="first"))

    def get_lead(row):
        lead_col = row.get("Lead/Sole Investors")
        if pd.notna(lead_col) and str(lead_col).strip():
            invs = parse_investors(str(lead_col))
            if invs: return invs[0].lower()
        invs = parse_investors(row["Investors"])
        return invs[0].lower() if invs else None

    company_lead = {row["company_key"]: (get_lead(row), row["date"])
                    for _, row in deals_sorted.iterrows()}

    def prior_communities(lead_inv, deal_date, lookback_years=2):
        hist = investor_history.get(lead_inv, [])
        if not hist: return 0
        cutoff = deal_date - pd.DateOffset(years=lookback_years)
        prior_comms = set()
        for d, co_comms in hist:
            if d >= deal_date: break
            if d >= cutoff:
                prior_comms.update(co_comms)
        return len(prior_comms)

    records = []
    for ck, (lead, date) in company_lead.items():
        if lead is None:
            records.append({"company_key": ck, "iv_a": np.nan})
        else:
            records.append({"company_key": ck, "iv_a": prior_communities(lead, date)})

    iv_a_df = pd.DataFrame(records)
    print(f"  IV A coverage: {iv_a_df['iv_a'].notna().sum():,} companies  "
          f"  mean={iv_a_df['iv_a'].mean():.2f}  "
          f"  sd={iv_a_df['iv_a'].std():.2f}  "
          f"  range=[{iv_a_df['iv_a'].min():.0f}, {iv_a_df['iv_a'].max():.0f}]")
    return iv_a_df


# ═══════════════════════════════════════════════════════════════════════════════
# IV B: AVERAGE INVESTOR GEOGRAPHIC DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def build_iv_b(cdf, deals, inv_to_geo):
    print("\nBuilding IV B (avg pairwise investor geographic distance) …")

    deals_first = (deals.sort_values("date")
                   .dropna(subset=["Investors"])
                   .drop_duplicates("company_key", keep="first"))

    records = []
    for _, row in deals_first.iterrows():
        ck = row["company_key"]
        inv_list = [i.lower() for i in parse_investors(row["Investors"]) if i]
        geos = [inv_to_geo.get(i) for i in inv_list]
        geos = [g for g in geos if g in GEO_COORD]

        if len(geos) < 2:
            records.append({"company_key": ck, "iv_b": np.nan})
            continue

        coords  = [GEO_COORD[g] for g in geos]
        dists   = [haversine_km(coords[i][0], coords[i][1],
                                coords[j][0], coords[j][1])
                   for i in range(len(coords))
                   for j in range(i+1, len(coords))]
        records.append({"company_key": ck, "iv_b": np.mean(dists)})

    iv_b_df = pd.DataFrame(records)
    valid = iv_b_df["iv_b"].dropna()
    print(f"  IV B coverage: {valid.notna().sum():,} companies  "
          f"  mean={valid.mean():.0f} km  "
          f"  sd={valid.std():.0f} km  "
          f"  pct=0: {(valid==0).mean():.1%}")
    return iv_b_df


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD ANALYSIS DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def build_df(cdf, iv_a_df, iv_b_df):
    df = (cdf.merge(iv_a_df, on="company_key", how="left")
             .merge(iv_b_df, on="company_key", how="left"))

    # Standardize IVs for interpretability
    df["iv_a_z"] = (df["iv_a"] - df["iv_a"].mean()) / df["iv_a"].std()
    df["iv_b_z"] = (df["iv_b"] - df["iv_b"].mean()) / df["iv_b"].std()

    # Keep required controls
    required = ["is_failed","cross_community","log_cap","specter_novelty",
                "company_age","hq_region","deal_year"]
    df = df.dropna(subset=required + ["iv_a"])

    # Create dummies for FE (linearmodels needs matrices, not patsy formulas)
    year_dummies   = pd.get_dummies(df["deal_year"], prefix="yr",  drop_first=True, dtype=float)
    region_dummies = pd.get_dummies(df["hq_region"], prefix="reg", drop_first=True, dtype=float)

    exog_cols = ["log_cap","specter_novelty","company_age"]
    df["const"] = 1.0
    exog = pd.concat([df[["const"] + exog_cols], year_dummies, region_dummies], axis=1)

    print(f"\n  Analysis dataset: N={len(df):,}  "
          f"  iv_a valid={df['iv_a'].notna().sum():,}  "
          f"  iv_b valid={df['iv_b'].notna().sum():,}")
    return df, exog


# ═══════════════════════════════════════════════════════════════════════════════
# 2SLS RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_2sls(y, exog, endog, instruments, label, lines, df=None):
    """Run IV2SLS and extract key diagnostics."""
    lines.append(f"\n  {'─'*65}")
    lines.append(f"  {label}")
    lines.append(f"  {'─'*65}")

    # Align indices
    idx = y.dropna().index & endog.dropna().index & instruments.dropna().index & exog.dropna().index
    y_    = y.loc[idx]
    exog_ = exog.loc[idx]
    endog_= endog.loc[idx]
    inst_ = instruments.loc[idx]

    try:
        mod = IV2SLS(dependent=y_, exog=exog_, endog=endog_, instruments=inst_)
        res = mod.fit(cov_type="robust")
    except Exception as e:
        lines.append(f"  MODEL ERROR: {e}")
        return None

    # ── Second stage: cross_community coefficient ──
    var = "cross_community"
    if var in res.params.index:
        b  = res.params[var]; p = res.pvalues[var]
        ci = res.conf_int().loc[var]
        lines.append(f"  2SLS β(cross_community) = {b:>+8.4f}  p={p:.4f} {sig(p)}")
        lines.append(f"  95% CI: [{ci.iloc[0]:>+.4f}, {ci.iloc[1]:>+.4f}]  N={int(res.nobs):,}")
    else:
        lines.append(f"  cross_community not in model params")
        return res

    # ── First-stage diagnostics ──
    try:
        fs   = res.first_stage
        diag = fs.diagnostics
        lines.append(f"\n  FIRST-STAGE DIAGNOSTICS:")
        # f.stat here is robust chi²(k) — divide by k for effective F
        for var_name in diag.index:
            row = diag.loc[var_name]
            # partial.rsquared column
            pr2 = row.get("partial.rsquared", row.get("shea.rsquared", np.nan))
            f_chi2 = row.get("f.stat", np.nan)  # chi²-stat (with robust SEs)
            f_pval = row.get("f.pval", np.nan)
            n_inst_used = inst_.shape[1]
            kp_f = f_chi2 / n_inst_used  # KP-F approximation
            lines.append(f"    Partial R² = {pr2:.4f}")
            lines.append(f"    KP-F (robust chi²/k) = {kp_f:.2f}  "
                         f"[{'STRONG, F>10' if kp_f>10 else 'WEAK, F≤10'}]"
                         f"  p={f_pval:.4f} {sig(f_pval)}")
    except Exception as e:
        lines.append(f"  First-stage diagnostics unavailable: {e}")

    # ── Hansen J-test (overidentification — only when >1 instrument) ──
    n_instruments = inst_.shape[1]
    n_endog = endog_.shape[1]
    df_j = n_instruments - n_endog
    if df_j > 0:
        # Try linearmodels sargan/basmann, then manual
        j_stat = j_pval = np.nan
        for attr in ["sargan", "basmann", "j_stat"]:
            try:
                obj = getattr(res, attr)
                j_stat = obj.stat; j_pval = obj.pval; break
            except Exception:
                pass
        # Manual fallback: regress 2SLS residuals on instruments + controls
        if np.isnan(j_stat):
            try:
                fitted = exog_.values @ res.params[exog_.columns].values + \
                         endog_.values.flatten() * res.params.get("cross_community", 0)
                resid = y_.values - fitted
                jdf = pd.concat([inst_, exog_], axis=1)
                jdf["resid"] = resid
                jcols = [c for c in jdf.columns if c != "resid"]
                jm = np.linalg.lstsq(jdf[jcols].values, jdf["resid"].values, rcond=None)
                y_fit = jdf[jcols].values @ jm[0]
                ss_tot = ((jdf["resid"] - jdf["resid"].mean())**2).sum()
                ss_res = ((jdf["resid"] - y_fit)**2).sum()
                r2_j = max(0, 1 - ss_res/ss_tot) if ss_tot > 0 else 0
                j_stat = len(y_) * r2_j
                j_pval = 1 - stats.chi2.cdf(j_stat, df=df_j)
            except Exception as e2:
                lines.append(f"  J-test manual fallback failed: {e2}")
        if not np.isnan(j_stat):
            lines.append(f"\n  SARGAN-HANSEN J-TEST (overidentification):")
            lines.append(f"    J = {j_stat:.4f}  p = {j_pval:.4f}  df = {df_j}")
            if j_pval > 0.10:
                lines.append("    → n.s.: both instruments pass overidentification test (plausibly valid)")
            else:
                lines.append("    → SIGNIFICANT: at least one instrument may violate exclusion restriction")

    return res


# ═══════════════════════════════════════════════════════════════════════════════
# HAUSMAN / DWH TEST (augmented regression approach)
# ═══════════════════════════════════════════════════════════════════════════════

def run_hausman(df, exog, iv_names, lines):
    """
    Durbin-Wu-Hausman via augmented regression:
    1. First stage: cross_community ~ instruments + controls → v_hat (residuals)
    2. Add v_hat to structural equation
    3. t-test on v_hat coefficient: significant → endogeneity present
    """
    lines.append("\n  HAUSMAN (DWH) ENDOGENEITY TEST")
    lines.append("  H0: cross_community is exogenous (OLS is consistent)")
    lines.append("  Method: augmented regression — add first-stage residuals to structural eq.")

    for iv_label, iv_col in iv_names:
        sub = df.dropna(subset=["is_failed","cross_community","log_cap",
                                 "specter_novelty","company_age",iv_col])
        if len(sub) < 100:
            lines.append(f"  {iv_label}: insufficient data (N={len(sub)})")
            continue

        ctrl = "log_cap + specter_novelty + company_age + C(hq_region) + C(deal_year)"

        # Stage 1
        fs = smf.ols(f"cross_community ~ {iv_col} + {ctrl}", data=sub).fit()
        sub = sub.copy()
        sub["v_hat"] = fs.resid

        # Augmented structural equation
        aug = smf.ols(f"is_failed ~ cross_community + v_hat + {ctrl}", data=sub).fit()
        b_v = aug.params.get("v_hat", np.nan)
        p_v = aug.pvalues.get("v_hat", np.nan)
        lines.append(f"\n  IV = {iv_label}:")
        lines.append(f"    v_hat coefficient: β={b_v:>+.4f}  p={p_v:.4f} {sig(p_v)}")
        if p_v < 0.10:
            lines.append("    → REJECT H0: cross_community is ENDOGENOUS — 2SLS preferred over OLS")
        else:
            lines.append("    → Fail to reject H0: cannot confirm endogeneity at p<0.10")

    # OLS LPM benchmark (consistent comparison with 2SLS LPM)
    lines.append("\n  OLS BENCHMARK (LPM — directly comparable to 2SLS LPM):")
    ctrl = "log_cap + specter_novelty + company_age + C(hq_region) + C(deal_year)"
    sub_ols = df.dropna(subset=["is_failed","cross_community","log_cap","specter_novelty","company_age"])
    try:
        ols_lpm = smf.ols(f"is_failed ~ cross_community + {ctrl}", data=sub_ols).fit(cov_type="HC1")
        b_lpm = ols_lpm.params.get("cross_community", np.nan)
        p_lpm = ols_lpm.pvalues.get("cross_community", np.nan)
        lines.append(f"    OLS (LPM) β = {b_lpm:>+.4f}  p={p_lpm:.4f} {sig(p_lpm)}  N={int(ols_lpm.nobs):,}")
    except Exception as e:
        lines.append(f"    OLS LPM failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST-STAGE DESCRIPTIVE SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def describe_first_stage(df, lines):
    lines.append("\n  FIRST-STAGE DESCRIPTIVE (OLS, for diagnostics)")
    ctrl = "log_cap + specter_novelty + company_age + C(hq_region) + C(deal_year)"

    for iv_col, iv_label in [("iv_a_z","IV A (z-scored lead prior communities)"),
                               ("iv_b_z","IV B (z-scored geographic distance)")]:
        sub = df.dropna(subset=["cross_community","log_cap","specter_novelty",
                                 "company_age",iv_col])
        if len(sub) < 50: continue
        m = smf.ols(f"cross_community ~ {iv_col} + {ctrl}", data=sub).fit(cov_type="HC1")
        b = m.params.get(iv_col, np.nan)
        p = m.pvalues.get(iv_col, np.nan)
        # Partial F = t² for one instrument (single-instrument case)
        partial_f = (b / m.bse.get(iv_col, np.nan))**2

        lines.append(f"\n  {iv_label}:")
        lines.append(f"    β(instrument) = {b:>+.4f}  p={p:.4f} {sig(p)}")
        lines.append(f"    Partial F (approx, t²) = {partial_f:.2f}  "
                     f"{'[STRONG, F>10]' if partial_f>10 else '[WEAK, F≤10]'}")
        lines.append(f"    First-stage R²={m.rsquared:.4f}  N={int(m.nobs):,}")

        # Correlation between IV and endogenous
        corr = sub[iv_col].corr(sub["cross_community"])
        lines.append(f"    Corr(IV, cross_community) = {corr:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = ["="*72,
             "2SLS INSTRUMENTAL VARIABLE ANALYSIS",
             "Endogenous: cross_community (binary)",
             "Outcome: is_failed (binary, estimated via LPM in 2SLS)",
             "Controls: log_cap, specter_novelty, company_age, region FE, year FE",
             "="*72,
             "",
             "EXCLUSION RESTRICTION ARGUMENTS:",
             "  IV A (Lead Prior Communities): An investor's past cross-community",
             "  activity predicts whether they form cross-community syndicates NOW,",
             "  but their historical reach should not directly cause THIS company's",
             "  failure (conditional on quality controls). Caveat: experienced investors",
             "  may also be better selectors — controls for capital and novelty partially",
             "  address this, but IV A is the weaker candidate.",
             "",
             "  IV B (Geographic Distance): Geographically dispersed investors necessarily",
             "  span community clusters (communities partially reflect geography), but",
             "  investor location should not directly cause portfolio failure beyond the",
             "  channel of cross-community information flow. Conditional on region FE,",
             "  within-region variation in syndicate geographic spread is a clean source",
             "  of exogenous variation in cross-community status.",
             "="*72]

    cdf, deals, inv_to_comm, inv_to_geo = load_data()
    iv_a_df = build_iv_a(cdf, deals, inv_to_comm)
    iv_b_df = build_iv_b(cdf, deals, inv_to_geo)
    df, exog = build_df(cdf, iv_a_df, iv_b_df)

    # ── First-stage scan ──────────────────────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("FIRST-STAGE RELEVANCE (instrument strength)")
    lines.append("="*72)
    describe_first_stage(df, lines)

    # ── Shared arrays ─────────────────────────────────────────────────────────
    y     = df["is_failed"].astype(float)
    endog = df[["cross_community"]].astype(float)

    # Sub-datasets: IV B has fewer valid observations
    df_ab = df.dropna(subset=["iv_b_z"])

    exog_ab     = exog.loc[df_ab.index]
    y_ab        = y.loc[df_ab.index]
    endog_ab    = endog.loc[df_ab.index]

    # ── MODEL A: IV A only ────────────────────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("MODEL A: IV = Lead Investor Prior Community Reach")
    lines.append("="*72)
    res_a = run_2sls(y, exog, endog,
                     df[["iv_a_z"]].astype(float),
                     "Stage 1: cross_community ~ iv_a_z + controls  "
                     "Stage 2: failure ~ cross_hat + controls",
                     lines)

    # ── MODEL B: IV B only ────────────────────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("MODEL B: IV = Average Investor Geographic Distance")
    lines.append("="*72)
    res_b = run_2sls(y_ab, exog_ab, endog_ab,
                     df_ab[["iv_b_z"]].astype(float),
                     "Stage 1: cross_community ~ iv_b_z + controls  "
                     "Stage 2: failure ~ cross_hat + controls",
                     lines)

    # ── MODEL C: Both IVs (overidentified) ────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("MODEL C: Both IVs (overidentified system, Hansen J-test)")
    lines.append("="*72)
    res_c = run_2sls(y_ab, exog_ab, endog_ab,
                     df_ab[["iv_a_z","iv_b_z"]].astype(float),
                     "Stage 1: cross_community ~ iv_a_z + iv_b_z + controls  "
                     "Stage 2: failure ~ cross_hat + controls",
                     lines)

    # ── Hausman (DWH) test ────────────────────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("HAUSMAN / DWH ENDOGENEITY TEST")
    lines.append("="*72)
    run_hausman(df, exog,
                [("IV A (lead prior communities)", "iv_a_z"),
                 ("IV B (geographic distance)", "iv_b_z")],
                lines)

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("SUMMARY TABLE: OLS vs 2SLS (β for cross_community)")
    lines.append("="*72)
    lines.append(f"  {'Model':<35}  {'β':>9}  {'p':>8}  sig  {'N':>7}  Notes")
    lines.append(f"  {'-'*80}")
    rows = [
        ("OLS logit (original finding)", -0.4373, 0.0000, 6718, "benchmark"),
        ("OLS LPM (for comparison)", None, None, None, "see Hausman section above"),
        ("2SLS Model A (IV A only)", res_a, "cross_community", "N/A", ""),
        ("2SLS Model B (IV B only)", res_b, "cross_community", "N/A", ""),
        ("2SLS Model C (both IVs)", res_c, "cross_community", "N/A", "overidentified"),
    ]
    for label, *rest in rows:
        if isinstance(rest[0], float) or rest[0] is None:
            b, p, n, note = rest
            if b is not None:
                lines.append(f"  {label:<35}  {b:>+9.4f}  {p:>8.4f}  {sig(p)}  {n:>7,}  {note}")
            else:
                lines.append(f"  {label:<35}  {'see above':>9}  {'':>8}       {'':>7}  {note}")
        else:
            res_obj, var, _, note = rest
            if res_obj is not None and var in res_obj.params.index:
                b = res_obj.params[var]; p = res_obj.pvalues[var]; n = int(res_obj.nobs)
                lines.append(f"  {label:<35}  {b:>+9.4f}  {p:>8.4f}  {sig(p)}  {n:>7,}  {note}")
            else:
                lines.append(f"  {label:<35}  {'FAILED':>9}  {'':>8}              {note}")

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append("\n" + "="*72)
    lines.append("INTERPRETATION")
    lines.append("="*72)
    lines.append("""
  KEY QUESTIONS ANSWERED:

  1. INSTRUMENT STRENGTH (First-stage F > 10?):
     See first-stage section. If F > 10, instruments are strong enough
     for credible IV estimates. Stock-Yogo 10% bias threshold requires F > 16.4.

  2. ENDOGENEITY (Hausman DWH p < 0.10?):
     If significant: cross_community is endogenous → 2SLS is preferred.
     If not significant: OLS may be consistent; 2SLS estimates are still valid
     but less efficient.

  3. DIRECTION OF ENDOGENEITY BIAS:
     If 2SLS β is MORE negative than OLS β: OLS was DOWNWARD biased.
       → Better companies chose cross-community syndicates ("reverse causality").
         2SLS corrects upward for this selection, revealing the true (smaller) effect.
     If 2SLS β is LESS negative than OLS β: OLS was UPWARD biased.
       → OLS overstates the benefit. Cross-community companies were already
         better before the syndicate formed (positive selection).
     If 2SLS β ≈ OLS β: endogeneity was not a major concern.

  4. OVERIDENTIFICATION (Hansen J p > 0.10?):
     If Model C J-test is n.s.: both instruments are consistent with
     the exclusion restriction — strong evidence for IV validity.
""")

    output = "\n".join(lines)
    print("\n" + output)
    (OUT / "results.txt").write_text(output)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
