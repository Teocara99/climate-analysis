"""
25 — Structural Factors Behind the Cross-Community Decline (2014–2025).

The cross-community syndication rate (company level) declined from 79% (2014)
to 45% (2025). This script tests what structural features of the investor
network at the time of funding predict this decline.

PART A: Annual cohort panel — which ecosystem-level structural factor explains
        why later-cohort companies are less likely to receive cross-community funding?
        DV = mean(cross_community) for companies first funded in year Y.
        IVs = ecosystem structural metrics in year Y.

PART B: Company level — what predicts whether a specific company gets
        cross-community funding? (year FE, region controls)

Outputs → output/cross_community_decline/
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from collections import Counter
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
OUT  = ROOT / "output" / "cross_community_decline"
OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_core():
    print("Loading deals and network metrics …")
    deals = load_deals()
    deals["Deal Date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["Deal Date","Investors"])
    deals["Year"] = deals["Deal Date"].dt.year
    deals["YQ"]   = deals["Deal Date"].dt.to_period("Q")
    deals = deals[(deals["Year"] >= 2014) & (deals["Year"] <= 2025)].copy()

    nm = pd.read_csv(ROOT / "output/network/network_metrics.csv")
    inv_to_comm = dict(zip(nm["investor"].str.lower().str.strip(), nm["community"]))
    inv_to_exp  = dict(zip(nm["investor"].str.lower().str.strip(), nm["total_deals"]))

    # Company-level cross_community from ground-truth company dataset
    cdf = pd.read_csv(ROOT / "output/cpu/cpu_company_dataset.csv")
    cdf["deal_year"] = cdf["deal_year"].fillna(2018).astype(int)
    print(f"  {len(deals):,} deals | {len(cdf):,} companies")
    return deals, inv_to_comm, inv_to_exp, cdf


def parse_investors(inv_str):
    if pd.isna(inv_str): return []
    return [n.strip() for n in re.split(r",\s*|\n", str(inv_str)) if n.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: ECOSYSTEM STRUCTURAL METRICS PER YEAR
# (computed from raw deals, not from company-level dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def build_annual_ecosystem(deals, inv_to_comm, inv_to_exp):
    """
    For each calendar year: compute ecosystem structural metrics.
    These describe what the investor network looks like in year Y.
    """
    print("Building annual ecosystem structural metrics …")
    seen_investors = {}  # investor → first year seen
    rows = []

    for year in sorted(deals["Year"].unique()):
        sub = deals[deals["Year"] == year]
        n_deals = len(sub)
        if n_deals < 10: continue

        # All investor names across all deals this year
        all_inv_names = []
        for inv_str in sub["Investors"]:
            all_inv_names.extend(parse_investors(inv_str))

        unique_inv_year = set(all_inv_names)

        # Community assignments
        comms_all = [inv_to_comm.get(n.lower()) for n in all_inv_names]
        comms_all = [c for c in comms_all if c is not None]
        comm_counter = Counter(comms_all)
        n_comms = len(comm_counter)

        if n_comms > 0:
            sizes = list(comm_counter.values())
            avg_comm_size = np.mean(sizes)
            total_classified = sum(sizes)
            hhi = sum((s/total_classified)**2 for s in sizes) if total_classified > 0 else np.nan
        else:
            avg_comm_size = np.nan; hhi = np.nan

        # Syndicate size per deal
        syn_sizes = sub["Investors"].apply(lambda x: len(parse_investors(x)))
        avg_syn = syn_sizes.mean()

        # New vs repeat investors (first time seen in dataset)
        n_new = 0
        for n in unique_inv_year:
            if n not in seen_investors:
                seen_investors[n] = year
                n_new += 1
        pct_new = n_new / len(unique_inv_year) if unique_inv_year else np.nan
        pct_repeat = 1 - pct_new

        # Average investor experience (total_deals from network_metrics)
        exps = [inv_to_exp.get(n.lower(), 1) for n in unique_inv_year]
        avg_exp = np.mean(exps) if exps else np.nan

        # Cumulative investor base (how many unique investors have ever been active)
        n_ever = len(seen_investors)

        rows.append({
            "year":              year,
            "n_deals":           n_deals,
            "n_active_inv":      len(unique_inv_year),
            "n_ever_inv":        n_ever,
            "n_communities":     n_comms,
            "avg_community_size":avg_comm_size,
            "hhi_community":     hhi,
            "avg_syndicate_size":avg_syn,
            "pct_new_investors": pct_new,
            "pct_repeat":        pct_repeat,
            "avg_exp":           avg_exp,
        })

    eco = pd.DataFrame(rows)
    print(f"  Ecosystem panel: {len(eco)} years")
    return eco


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: COHORT PANEL = pct_cross per year + ecosystem metrics
# ═══════════════════════════════════════════════════════════════════════════════

def build_cohort_panel(cdf, eco):
    # Annual cross_community from company dataset (ground truth)
    cohort = (cdf[(cdf["deal_year"]>=2014) & (cdf["deal_year"]<=2025)]
              .groupby("deal_year")["cross_community"]
              .agg(pct_cross="mean", n_companies="count")
              .reset_index()
              .rename(columns={"deal_year":"year"}))
    cohort = cohort[cohort["n_companies"] >= 30]

    panel = cohort.merge(eco, on="year", how="inner")
    panel = panel.sort_values("year").reset_index(drop=True)
    panel["time_trend"] = np.arange(len(panel))

    # Standardise structural predictors
    pred_cols = ["n_active_inv","n_communities","avg_community_size",
                 "hhi_community","avg_syndicate_size","pct_new_investors","avg_exp"]
    for c in pred_cols:
        panel[f"{c}_z"] = (panel[c] - panel[c].mean()) / panel[c].std()

    print(f"  Cohort panel: {len(panel)} years  |  "
          f"pct_cross: {panel['pct_cross'].min():.1%} – {panel['pct_cross'].max():.1%}")
    print(f"  2014: {panel[panel.year==2014]['pct_cross'].values[0]:.1%}  |  "
          f"2025: {panel[panel.year==2025]['pct_cross'].values[0]:.1%}")
    return panel, pred_cols


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: ANNUAL REGRESSIONS — WHAT EXPLAINS THE COHORT DECLINE?
# ═══════════════════════════════════════════════════════════════════════════════

def part_a(panel, pred_cols, lines):
    print("\nPart A: Annual cohort regressions …")
    lines += ["\n"+"="*72,
              "PART A: ANNUAL COHORT PANEL — WHAT EXPLAINS THE DECLINE?",
              "  DV: pct_cross = mean(cross_community) for companies funded in year Y",
              f"  Ground truth: company-level cross_community from cpu_company_dataset.csv",
              f"  N years: {len(panel)}  (2014–2025, min 30 companies/year)",
              f"  Declined from {panel[panel.year==2014]['pct_cross'].values[0]:.1%} (2014) "
              f"to {panel[panel.year==2025]['pct_cross'].values[0]:.1%} (2025)",
              "  Controls: time_trend",
              "  All predictors: ecosystem metrics in year Y (z-scored)",
              "="*72]

    # ── A1: Bivariate regressions (each structural var + trend) ──
    lines.append("\n  [A1] BIVARIATE: pct_cross ~ structural_var + time_trend")
    lines.append(f"  {'Variable':<28}  {'β':>8}  {'p':>7}  {'adj-R²':>7}  verdict")
    lines.append(f"  {'-'*68}")

    bivar = {}
    for col in pred_cols:
        sub = panel.dropna(subset=["pct_cross", f"{col}_z"])
        if len(sub) < 6: continue
        m = smf.ols(f"pct_cross ~ {col}_z + time_trend", data=sub).fit(cov_type="HC3")
        b = m.params[f"{col}_z"]; p = m.pvalues[f"{col}_z"]
        bivar[col] = {"b":b,"p":p,"r2":m.rsquared_adj,"n":len(sub)}
        lines.append(f"  {col:<28}  β={b:>+7.4f}  p={p:.4f}  R²={m.rsquared_adj:.3f}  {sig(p)}")

    # ── A2: Horse race ──
    lines.append("\n  [A2] HORSE RACE: all structural vars + time_trend")
    all_z = " + ".join([f"{c}_z" for c in pred_cols])
    sub_all = panel.dropna(subset=["pct_cross"] + [f"{c}_z" for c in pred_cols])
    lines.append(f"  N = {len(sub_all)}")
    try:
        m_all = smf.ols(f"pct_cross ~ {all_z} + time_trend",
                        data=sub_all).fit(cov_type="HC3")
        lines.append(f"  adj-R² = {m_all.rsquared_adj:.3f}")
        lines.append(f"  {'Variable':<28}  {'β':>8}  {'p':>7}  verdict")
        for col in pred_cols:
            key = f"{col}_z"
            if key in m_all.params.index:
                b = m_all.params[key]; p = m_all.pvalues[key]
                lines.append(f"  {col:<28}  β={b:>+7.4f}  p={p:.4f}  {sig(p)}")

        # VIF
        X_vif = sub_all[[f"{c}_z" for c in pred_cols]].dropna()
        if len(X_vif) > len(X_vif.columns):
            vifs = {X_vif.columns[i]: variance_inflation_factor(X_vif.values, i)
                    for i in range(X_vif.shape[1])}
            high = {k:v for k,v in vifs.items() if v > 5}
            if high:
                lines.append(f"\n  HIGH VIF (>5): {', '.join(f'{k}={v:.1f}' for k,v in high.items())}")
    except Exception as e:
        lines.append(f"  Horse race ERROR: {e}")

    # ── A3: Parsimonious (significant vars only) ──
    sig_vars = [k for k,v in bivar.items() if v["p"] < 0.10]
    if len(sig_vars) >= 2:
        lines.append(f"\n  [A3] PARSIMONIOUS: significant vars ({', '.join(sig_vars)})")
        pars_z = " + ".join([f"{c}_z" for c in sig_vars[:4]])
        sub_p = panel.dropna(subset=["pct_cross"] + [f"{c}_z" for c in sig_vars[:4]])
        try:
            m_p = smf.ols(f"pct_cross ~ {pars_z} + time_trend",
                          data=sub_p).fit(cov_type="HC3")
            lines.append(f"  adj-R² = {m_p.rsquared_adj:.3f}  N={len(sub_p)}")
            for col in sig_vars[:4]:
                key = f"{col}_z"
                if key in m_p.params.index:
                    b = m_p.params[key]; p = m_p.pvalues[key]
                    lines.append(f"    {col:<28}  β={b:>+7.4f}  p={p:.4f}  {sig(p)}")
        except Exception as e:
            lines.append(f"  Parsimonious ERROR: {e}")

    # ── A4: First-difference robustness ──
    lines.append("\n  [A4] FIRST-DIFFERENCE ROBUSTNESS (removes secular trends)")
    lines.append("  Δpct_cross(t) ~ Δstructural(t)  +  Δstructural(t-1)")
    panel_fd = panel.copy()
    panel_fd["d_pct_cross"] = panel_fd["pct_cross"].diff()
    for col in pred_cols:
        panel_fd[f"d_{col}_z"]    = panel_fd[f"{col}_z"].diff()
        panel_fd[f"d_{col}_z_l1"] = panel_fd[f"d_{col}_z"].shift(1)

    lines.append(f"  {'Variable':<28}  {'β(Δt)':>9}  {'p':>7}  {'β(Δt-1)':>9}  {'p':>7}")
    lines.append(f"  {'-'*70}")
    for col in pred_cols:
        sub_fd = panel_fd.dropna(subset=["d_pct_cross",f"d_{col}_z",f"d_{col}_z_l1"])
        if len(sub_fd) < 6: continue
        m_fd = smf.ols(f"d_pct_cross ~ d_{col}_z + d_{col}_z_l1",
                       data=sub_fd).fit(cov_type="HC3")
        b0 = m_fd.params.get(f"d_{col}_z",    np.nan); p0 = m_fd.pvalues.get(f"d_{col}_z",    np.nan)
        b1 = m_fd.params.get(f"d_{col}_z_l1", np.nan); p1 = m_fd.pvalues.get(f"d_{col}_z_l1", np.nan)
        lines.append(f"  {col:<28}  β={b0:>+8.4f}  p={p0:.4f} {sig(p0):4}  "
                     f"β={b1:>+8.4f}  p={p1:.4f} {sig(p1)}")

    # ── A5: Trend decomposition ──
    lines.append("\n  [A5] TREND DECOMPOSITION: how much did each variable change 2014→2025?")
    lines.append(f"  {'Variable':<28}  {'2014':>8}  {'2025':>8}  {'Δ':>8}  {'Δ (σ)':>7}  direction")
    early = panel[panel["year"] <= 2015]
    late  = panel[panel["year"] >= 2024]
    for col in pred_cols + ["pct_cross"]:
        if col not in panel.columns: continue
        e = early[col].mean(); l = late[col].mean(); sd = panel[col].std()
        d = l - e; d_sd = d/sd if sd>0 else np.nan
        direction = ("↑ rises" if d > 0 else "↓ falls")
        lines.append(f"  {col:<28}  {e:>8.3f}  {l:>8.3f}  {d:>+8.3f}  {d_sd:>+7.2f}σ  {direction}")

    # ── A6: THE MECHANISM — connect trend decomp to regression ──
    lines.append("\n  [A6] MECHANISM INTERPRETATION")
    lines.append("  Cross-community DECLINED while ecosystem GREW. Key structural moves:")
    lines.append("  (Highlighted vars that both changed substantially AND predict pct_cross)")

    for col in pred_cols:
        if col not in bivar: continue
        e = early[col].mean(); l = late[col].mean(); sd = panel[col].std()
        d_sd = (l-e)/sd if sd>0 else 0
        b = bivar[col]["b"]; p = bivar[col]["p"]
        if abs(d_sd) > 1.0 and p < 0.10:
            direction = "↑" if l>e else "↓"
            predicted_effect = "→ predicts ↑ cross" if b>0 else "→ predicts ↓ cross"
            consistent = (b<0 and l<e) or (b>0 and l>e)
            match = "CONSISTENT with decline" if not consistent else "OPPOSES decline"
            lines.append(f"  {col:<28}: {direction} {abs(d_sd):.1f}σ  {predicted_effect}  [{match}]")

    return bivar


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: COMPANY-LEVEL PREDICTORS
# ═══════════════════════════════════════════════════════════════════════════════

def part_b(deals, inv_to_comm, inv_to_exp, cdf, lines):
    print("\nPart B: Company-level predictors …")
    lines += ["\n"+"="*72,
              "PART B: COMPANY-LEVEL — WHAT PREDICTS CROSS-COMMUNITY FUNDING?",
              "  DV: is_cross_community (investors span ≥2 Louvain communities)",
              "  IVs: n_investors, total_funding, n_rounds, investor experience, year_FE, region",
              "="*72]

    # Per-company metrics from first deal
    first_deals = (deals.sort_values("Deal Date")
                   .drop_duplicates("Companies", keep="first")
                   [["Companies","Investors","Year"]])

    def company_inv_metrics(inv_str):
        names = parse_investors(inv_str)
        if not names: return np.nan, np.nan
        exps  = [inv_to_exp.get(n.lower(), 1) for n in names]
        return len(names), np.mean(exps)

    first_deals[["n_inv_first","avg_exp_first"]] = first_deals["Investors"].apply(
        lambda x: pd.Series(company_inv_metrics(x)))

    df = cdf.merge(first_deals[["Companies","n_inv_first","avg_exp_first"]],
                   on="Companies", how="left")
    df["log_cap"]   = np.log1p(df["Total Raised"].fillna(0))
    df["nrounds"]   = np.log1p(df["n_rounds"].fillna(1))
    df["n_inv_log"] = np.log1p(df["n_inv_first"].fillna(1))
    df["log_exp"]   = np.log1p(df["avg_exp_first"].fillna(1))
    df["deal_year"] = df["deal_year"].astype(int)
    df = df[df["hq_region"].notna() & df["deal_year"].between(2014,2025)].copy()

    lines.append(f"\n  Company sample: N={len(df):,}")
    lines.append(f"  cross_community=1: {int(df['cross_community'].sum()):,} ({df['cross_community'].mean():.1%})")

    base = "log_cap + nrounds + C(hq_region) + C(deal_year)"

    # ── B1: Bivariate ──
    lines.append("\n  [B1] BIVARIATE PREDICTORS (each + year FE + region)")
    lines.append(f"  {'Variable':<40}  {'β':>8}  {'p':>7}  N  verdict")
    lines.append(f"  {'-'*70}")

    bivar_b = {}
    for var, label in [
        ("n_inv_log",  "log(n_investors in first deal)"),
        ("log_cap",    "log(total funding)"),
        ("nrounds",    "log(n_rounds)"),
        ("log_exp",    "log(avg investor experience)"),
        ("pure_green", "pure_green syndicate (dummy)"),
        ("pure_trad",  "pure_traditional syndicate (dummy)"),
        ("pct_green",  "% green investors"),
    ]:
        sub = df.dropna(subset=["cross_community",var,"log_cap"]).copy()
        try:
            m = smf.logit(f"cross_community ~ {var} + {base}",
                          data=sub).fit(disp=0, maxiter=400)
            b = m.params.get(var, np.nan); p = m.pvalues.get(var, np.nan)
            bivar_b[var] = {"b":b,"p":p,"n":int(m.nobs),"label":label}
            lines.append(f"  {label:<40}  β={b:>+7.4f}  p={p:.4f}  N={int(m.nobs):,}  {sig(p)}")
        except Exception as e:
            lines.append(f"  {label:<40}  ERROR: {e}")

    # ── B2: Full model ──
    lines.append("\n  [B2] FULL MODEL")
    full_vars = "n_inv_log + log_cap + nrounds + log_exp + pct_green"
    sub_f = df.dropna(subset=["cross_community","n_inv_log","log_cap",
                               "nrounds","log_exp","pct_green"]).copy()
    try:
        m_f = smf.logit(f"cross_community ~ {full_vars} + {base}",
                        data=sub_f).fit(disp=0, maxiter=400)
        lines.append(f"  N={int(m_f.nobs):,}  pseudo-R²={m_f.prsquared:.3f}")
        for var in ["n_inv_log","log_cap","nrounds","log_exp","pct_green"]:
            if var in m_f.params.index:
                b = m_f.params[var]; p = m_f.pvalues[var]
                lines.append(f"    {var:<28}  β={b:>+7.4f}  p={p:.4f}  {sig(p)}")
    except Exception as e:
        lines.append(f"  Full model ERROR: {e}")

    # ── B3: Declining trend by region ──
    lines.append("\n  [B3] CROSS-COMMUNITY TREND BY REGION")
    lines.append(f"  {'Region':<12}  {'2014-15':>8}  {'2023-25':>8}  {'Δ':>7}  trend β/yr  verdict")
    for region in ["Americas","Europe","Asia"]:
        sub_r = df[df["hq_region"]==region]
        early = sub_r[sub_r["deal_year"]<=2015]["cross_community"].mean()
        late  = sub_r[sub_r["deal_year"]>=2023]["cross_community"].mean()
        try:
            m_t = smf.ols("cross_community ~ deal_year",
                          data=sub_r.dropna(subset=["cross_community","deal_year"])).fit(cov_type="HC3")
            b_t = m_t.params.get("deal_year", np.nan); p_t = m_t.pvalues.get("deal_year", np.nan)
            lines.append(f"  {region:<12}  {early:.1%}    {late:.1%}  "
                         f"{late-early:>+.1%}  {b_t:>+.4f}/yr {sig(p_t)}")
        except:
            lines.append(f"  {region}: error")

    # ── B4: Syndicate size dose-response ──
    lines.append("\n  [B4] SYNDICATE SIZE → CROSS-COMMUNITY (dose-response)")
    lines.append("  H: Larger syndicates mechanically span more communities")
    lines.append(f"  {'n_investors':>12}  {'% cross-community':>18}  N")
    for n_inv in range(1, 10):
        sub_n = df[df["n_inv_first"].fillna(0)==n_inv]
        if len(sub_n) < 30: continue
        rate = sub_n["cross_community"].mean()
        bar = "█" * int(rate * 20)
        lines.append(f"  n_inv = {n_inv:>2}   {rate:.1%}  {bar}  (N={len(sub_n):,})")

    # ── B5: After controlling for syndicate size ──
    lines.append("\n  [B5] WHAT ELSE PREDICTS CROSS-COMMUNITY? (controlling for syndicate size)")
    for var, label in [("log_cap","log(funding)"),("log_exp","investor experience"),
                        ("pct_green","green investor %"),("pure_trad","pure_traditional")]:
        sub = df.dropna(subset=["cross_community",var,"n_inv_log","log_cap"]).copy()
        try:
            m = smf.logit(f"cross_community ~ n_inv_log + {var} + {base}",
                          data=sub).fit(disp=0, maxiter=400)
            b  = m.params.get(var, np.nan); p  = m.pvalues.get(var, np.nan)
            bn = m.params.get("n_inv_log", np.nan); pn = m.pvalues.get("n_inv_log", np.nan)
            lines.append(f"  {label:<22}  β={b:>+7.4f} {sig(p):4}  "
                         f"[n_inv_log β={bn:>+7.4f} {sig(pn)}]  N={int(m.nobs):,}")
        except Exception as e:
            lines.append(f"  {label}: ERROR {e}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_decline(panel, df_company):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=140, facecolor="white")
    axes = axes.flatten()

    # Panel 1: pct_cross declining trend
    ax = axes[0]
    ax.plot(panel["year"], panel["pct_cross"]*100, "o-", color="#2c3e50", lw=2, ms=7)
    t = np.arange(len(panel)); y = panel["pct_cross"].values*100
    z = np.polyfit(t, y, 1); p = np.poly1d(z)
    ax.plot(panel["year"], p(t), "--", color="#e74c3c", lw=1.5, label=f"Trend {z[0]:+.1f}%/yr")
    ax.set_title("Cross-Community Rate by Founding Cohort", fontsize=10, fontweight="bold")
    ax.set_ylabel("% Companies with Cross-Community Syndicate")
    ax.set_xlabel("Year of First Funding")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Key structural vars vs pct_cross (scatter)
    ax = axes[1]
    sc = ax.scatter(panel["avg_community_size"], panel["pct_cross"]*100,
                    c=panel["year"], cmap="viridis", s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_xlabel("Avg Community Size")
    ax.set_ylabel("% Cross-Community")
    ax.set_title("Ecosystem Community Size vs Cross-Community Rate", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3: pct_new_investors declining
    ax = axes[2]
    ax.plot(panel["year"], panel["pct_new_investors"]*100, "o-", color="#e67e22", lw=2, ms=7)
    ax.set_title("% First-Time Investors by Year", fontsize=10, fontweight="bold")
    ax.set_ylabel("% New Investors (first appearance)")
    ax.set_xlabel("Year")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 4: n_communities growth
    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(panel["year"], panel["n_communities"], "o-", color="#3498db", lw=2, ms=7, label="N communities")
    ax2.plot(panel["year"], panel["avg_community_size"], "s--", color="#e74c3c", lw=1.5, ms=6, label="Avg size")
    ax.set_ylabel("N active communities", color="#3498db")
    ax2.set_ylabel("Avg community size", color="#e74c3c")
    ax.set_title("Community Count & Size Over Time", fontsize=10, fontweight="bold")
    lns = ax.get_lines() + ax2.get_lines()
    ax.legend(lns, [l.get_label() for l in lns], fontsize=8)
    ax.spines["top"].set_visible(False)

    # Panel 5: syndicate size dose-response
    ax = axes[4]
    rates = []
    for n in range(1, 10):
        sub_n = df_company[df_company["n_inv_first"].fillna(0)==n]
        if len(sub_n) >= 30:
            rates.append((n, sub_n["cross_community"].mean(), len(sub_n)))
    if rates:
        ns, rs, cs = zip(*rates)
        bars = ax.bar(ns, [r*100 for r in rs], color="#2980b9", alpha=0.8, edgecolor="white")
        for bar, c in zip(bars, cs):
            if c > 100:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f"N={c}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xlabel("N Investors in First Deal")
        ax.set_ylabel("% Cross-Community")
        ax.set_title("Syndicate Size → Cross-Community Rate", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 6: regional decline
    ax = axes[5]
    for region, color in [("Americas","#3498db"),("Europe","#27ae60"),("Asia","#e67e22")]:
        sub_r = (df_company[df_company["hq_region"]==region]
                 .groupby("deal_year")["cross_community"]
                 .agg(["mean","count"]).reset_index())
        sub_r = sub_r[sub_r["count"]>=20]
        ax.plot(sub_r["deal_year"], sub_r["mean"]*100, "o-",
                color=color, lw=1.8, ms=5, label=region)
    ax.set_xlabel("Year"); ax.set_ylabel("% Cross-Community")
    ax.set_title("Cross-Community Decline by Region", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Structural Drivers of Cross-Community Syndication Decline (2014–2025)\n"
                 "Company-level ground truth + ecosystem structural metrics",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT/"fig1_decline.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved fig1_decline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = [
        "="*72,
        "STRUCTURAL FACTORS BEHIND THE CROSS-COMMUNITY DECLINE (2014–2025)",
        "DV: % of companies in founding cohort with cross-community syndicate",
        "    (ground truth from cpu_company_dataset.csv — 6,718 companies)",
        "="*72,
    ]

    deals, inv_to_comm, inv_to_exp, cdf = load_core()
    eco   = build_annual_ecosystem(deals, inv_to_comm, inv_to_exp)
    panel, pred_cols = build_cohort_panel(cdf, eco)

    eco.to_csv(OUT/"annual_ecosystem.csv", index=False)
    panel.to_csv(OUT/"cohort_panel.csv", index=False)

    bivar = part_a(panel, pred_cols, lines)
    df_co = part_b(deals, inv_to_comm, inv_to_exp, cdf, lines)

    # Summary
    lines += ["\n"+"="*72, "SUMMARY", "="*72]
    sig_a = [(k,v) for k,v in bivar.items() if v["p"]<0.05]
    marg  = [(k,v) for k,v in bivar.items() if 0.05<=v["p"]<0.10]

    lines.append("\n  PART A — Structural variables that predict (bivariate, with trend control):")
    if sig_a:
        for k,v in sig_a:
            lines.append(f"    {k:<28}  β={v['b']:>+.4f}  p={v['p']:.4f} {sig(v['p'])}")
    else:
        lines.append("    None at p<0.05 after trend control")
    if marg:
        for k,v in marg:
            lines.append(f"    {k:<28}  β={v['b']:>+.4f}  p={v['p']:.4f} †  (marginal)")

    lines += [
        "\n  INTERPRETATION:",
        "  Variables with positive β INCREASE pct_cross (if they RISE → counter the decline).",
        "  Variables with negative β DECREASE pct_cross (if they RISE → drive the decline).",
        "  Consistent driver = variable that ROSE over time AND has β in the direction of decline.",
    ]

    output = "\n".join(lines)
    print("\n" + output)
    (OUT/"results.txt").write_text(output)

    plot_decline(panel, df_co)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
