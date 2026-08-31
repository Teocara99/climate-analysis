"""
24 — Regional CPU × Cross-Border Syndication (Company-Level).

For each of the 6,718 companies:
  • Startup region CPU at month of first investment
  • Investors' primary-region CPU at same month
  • CPU_gap = |startup_CPU − investor_CPU|
  • is_cross_border, pct_foreign_investors

All models: YEAR FIXED EFFECTS.
All significant level results: first-difference robustness check.

Outputs → output/regional_cpu/
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from collections import Counter
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT  = ROOT / "output" / "regional_cpu"
OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))


# ═══════════════════════════════════════════════════════════════════════════════
# 0A. MONTHLY CPU SERIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_monthly_cpu():
    """
    Returns monthly DataFrame indexed by year_month string (YYYY-MM).
    Columns: CPU_US, CPU_EU, CPU_gap, CPU_US_z, CPU_EU_z, CPU_gap_z
    CPU_EU ends 2019-12; CPU_US continues to 2026.
    """
    # Gavriilidis US
    cpu_g = pd.read_csv(DATA / "cpu_gavriilidis_us.csv")
    cpu_g = cpu_g[["year_month","cpu_narrow"]].rename(
        columns={"year_month":"ym","cpu_narrow":"CPU_US"})

    # Basaglia multi-country
    cpu_b = pd.read_csv(DATA / "cpu_multicountry.csv")
    cpu_b = cpu_b.rename(columns={"year_month":"ym"})
    eu_cols = ["CPU_UK","CPU_DEU","CPU_FRA","CPU_ESP","CPU_ITA","CPU_IRL"]
    cpu_b["CPU_EU"] = cpu_b[eu_cols].mean(axis=1)
    cpu_b = cpu_b[["ym","CPU_EU"]]

    cpu = cpu_g.merge(cpu_b, on="ym", how="left")
    cpu["CPU_gap"] = (cpu["CPU_US"] - cpu["CPU_EU"]).abs()

    # Z-score over overlapping window only
    both = cpu.dropna(subset=["CPU_US","CPU_EU"])
    for col in ["CPU_US","CPU_EU","CPU_gap"]:
        mu = both[col].mean(); sd = both[col].std()
        cpu[f"{col}_z"] = (cpu[col] - mu) / sd

    cpu = cpu.set_index("ym")
    print(f"  Monthly CPU: CPU_US {cpu['CPU_US'].notna().sum()} months | "
          f"CPU_EU {cpu['CPU_EU'].notna().sum()} months "
          f"(through {cpu.index[cpu['CPU_EU'].notna()][-1]})")
    return cpu


# ═══════════════════════════════════════════════════════════════════════════════
# 0B. INVESTOR MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def build_investor_maps():
    ip = pd.read_csv(ROOT / "output/investor_profiles.csv")
    inv_geo = dict(zip(ip["investor"].str.lower().str.strip(), ip["geo_focus"]))

    nm = pd.read_csv(ROOT / "output/network/network_metrics.csv")
    inv_comm = dict(zip(nm["investor"].str.lower().str.strip(), nm["community"]))

    nm["geo"] = nm["investor"].str.lower().str.strip().map(inv_geo)
    comm_geo = (nm.dropna(subset=["geo","community"])
                .groupby("community")["geo"]
                .agg(lambda x: x.value_counts().index[0])
                .reset_index()
                .rename(columns={"geo":"dominant_geo"}))
    comm_dominant = dict(zip(comm_geo["community"], comm_geo["dominant_geo"]))

    us_comms   = {c for c,g in comm_dominant.items() if g=="Americas"}
    eu_comms   = {c for c,g in comm_dominant.items() if g=="Europe"}
    asia_comms = {c for c,g in comm_dominant.items() if g=="Asia"}
    print(f"  Community geo: US={len(us_comms)} EU={len(eu_comms)} Asia={len(asia_comms)}")
    return inv_geo, inv_comm, us_comms, eu_comms, asia_comms


def region_to_cpu_col(region):
    if not region or pd.isna(region): return None
    r = str(region)
    if "Americ" in r: return "CPU_US"
    if "Europe" in r: return "CPU_EU"
    return None

def geo_focus_to_cpu_col(geo_focus):
    if not geo_focus or pd.isna(geo_focus): return None
    if "Americ" in str(geo_focus): return "CPU_US"
    if "Europe" in str(geo_focus): return "CPU_EU"
    return None

def region_label(region):
    if not region or pd.isna(region): return None
    r = str(region)
    if "Americ" in r: return "Americas"
    if "Europe" in r: return "Europe"
    if "Asia"   in r: return "Asia"
    if "Oceania"in r: return "Oceania"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 0C. BUILD COMPANY-LEVEL GEO DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def build_company_geo(cpu_monthly, inv_geo, inv_comm, us_comms, eu_comms, asia_comms):
    print("  Building company-level geo dataset …")

    # Base company dataset
    cdf = pd.read_csv(ROOT / "output/cpu/cpu_company_dataset.csv")
    cdf["company_key"] = cdf["Companies"].str.lower().str.strip()

    # Raw deals: one investor list per company (first deal by date)
    deals = load_deals()
    deals["Deal Date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["Deal Date","Investors"])
    deals["ym"]       = deals["Deal Date"].dt.strftime("%Y-%m")
    deals["company_key"] = deals["Companies"].str.lower().str.strip()
    first_deals = (deals.sort_values("Deal Date")
                   .drop_duplicates("company_key", keep="first")
                   [["company_key","ym","Investors"]])
    first_inv_map = dict(zip(first_deals["company_key"], first_deals["Investors"]))
    first_ym_map  = dict(zip(first_deals["company_key"], first_deals["ym"]))

    records = []
    for _, row in cdf.iterrows():
        ck        = row["company_key"]
        inv_str   = first_inv_map.get(ck, "")
        ym        = first_ym_map.get(ck, "")
        startup_r = region_label(row.get("hq_region",""))

        names   = re.split(r",\s*|\n", str(inv_str)) if inv_str else []
        geos    = [inv_geo.get(n.strip().lower()) for n in names if n.strip()]
        geos    = [g for g in geos if g]
        comms   = [inv_comm.get(n.strip().lower()) for n in names if n.strip()]
        comms   = [c for c in comms if c is not None]

        has_us   = int(any(c in us_comms   for c in comms))
        has_eu   = int(any(c in eu_comms   for c in comms))
        has_asia = int(any(c in asia_comms for c in comms))

        # Investor majority region
        if geos:
            majority_geo = Counter(geos).most_common(1)[0][0]
            pct_foreign  = (sum(1 for g in geos if g != startup_r) / len(geos)
                            if startup_r else np.nan)
        else:
            majority_geo = None
            pct_foreign  = np.nan

        is_cross = (int(majority_geo != startup_r)
                    if (majority_geo and startup_r) else np.nan)

        # CPU lookups
        startup_cpu_col = region_to_cpu_col(startup_r)
        inv_cpu_col     = geo_focus_to_cpu_col(majority_geo)

        def get_cpu(col, z=False):
            if not col or not ym or ym not in cpu_monthly.index: return np.nan
            key = f"{col}_z" if z else col
            val = cpu_monthly.loc[ym].get(key, np.nan)
            return val if not pd.isna(val) else np.nan

        startup_cpu_val = get_cpu(startup_cpu_col)
        startup_cpu_z   = get_cpu(startup_cpu_col, z=True)
        inv_cpu_val     = get_cpu(inv_cpu_col)
        inv_cpu_z       = get_cpu(inv_cpu_col, z=True)
        cpu_gap_z       = abs(startup_cpu_z - inv_cpu_z) if (
            not pd.isna(startup_cpu_z) and not pd.isna(inv_cpu_z)) else np.nan

        records.append({
            "company_key":    ck,
            "ym":             ym,
            "startup_geo":    startup_r,
            "inv_majority_geo":majority_geo,
            "is_cross_border":is_cross,
            "pct_foreign":    pct_foreign,
            "startup_cpu_val":startup_cpu_val,
            "startup_cpu_z":  startup_cpu_z,
            "inv_cpu_val":    inv_cpu_val,
            "inv_cpu_z":      inv_cpu_z,
            "cpu_gap_z":      cpu_gap_z,
            "has_us_comm":    has_us,
            "has_eu_comm":    has_eu,
            "has_asia_comm":  has_asia,
        })

    geo_df = pd.DataFrame(records)
    df = cdf.merge(geo_df, on="company_key", how="left")

    # Outcomes and controls
    df["is_failed"] = (df["outcome"] == "Failed").astype(float)
    df["is_exit"]   = df["outcome"].isin(["Acquired","IPO / Public"]).astype(float)
    df["log_cap"]   = np.log1p(df["Total Raised"].fillna(0))
    df["nrounds"]   = np.log1p(df["n_rounds"].fillna(1))
    df["deal_year"] = df["deal_year"].fillna(2018).astype(int)

    # Coverage
    print(f"  Coverage:")
    print(f"    Total: {len(df):,} | startup_cpu: {df['startup_cpu_z'].notna().sum():,} | "
          f"inv_cpu: {df['inv_cpu_z'].notna().sum():,} | "
          f"cpu_gap: {df['cpu_gap_z'].notna().sum():,}")
    print(f"    is_cross classifiable: {df['is_cross_border'].notna().sum():,} | "
          f"is_cross=1: {(df['is_cross_border']==1).sum():,}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: STARTUP-REGION CPU → CROSS-BORDER INVESTMENT
# ═══════════════════════════════════════════════════════════════════════════════

def model1(df, lines):
    print("\nModel 1: startup_CPU → is_cross_border / pct_foreign …")
    lines += ["\n"+"="*72,
              "MODEL 1: REGIONAL CPU → CROSS-BORDER INVESTMENT",
              "  H: High startup-region CPU reduces foreign investor participation",
              "  Controls: pure_green + log_cap + nrounds + hq_region + YEAR FE",
              "="*72]

    controls = "pure_green + log_cap + nrounds + C(hq_region) + C(deal_year)"

    for cpu_col, label in [("startup_cpu_z","Startup-region CPU"),
                            ("inv_cpu_z",    "Investor-region CPU")]:
        lines.append(f"\n  [{label}]")
        sub = df.dropna(subset=[cpu_col,"log_cap","hq_region"]).copy()

        # Logit: is_cross_border
        sub_cb = sub.dropna(subset=["is_cross_border"])
        if len(sub_cb) >= 200:
            try:
                m = smf.logit(f"is_cross_border ~ {cpu_col} + {controls}",
                              data=sub_cb).fit(disp=0, maxiter=400)
                b = m.params[cpu_col]; p = m.pvalues[cpu_col]
                lines.append(f"    is_cross_border (logit) β={b:>+8.4f}  p={p:.4f} {sig(p)}  N={int(m.nobs):,}")
            except Exception as e:
                lines.append(f"    is_cross_border: ERROR {e}")

        # OLS: pct_foreign
        sub_pf = sub.dropna(subset=["pct_foreign"])
        if len(sub_pf) >= 200:
            try:
                m2 = smf.ols(f"pct_foreign ~ {cpu_col} + {controls}",
                             data=sub_pf).fit(cov_type="HC3")
                b2 = m2.params[cpu_col]; p2 = m2.pvalues[cpu_col]
                lines.append(f"    pct_foreign (OLS/HC3)    β={b2:>+8.4f}  p={p2:.4f} {sig(p2)}  "
                             f"N={int(m2.nobs):,}  adj-R²={m2.rsquared_adj:.3f}")
            except Exception as e:
                lines.append(f"    pct_foreign: ERROR {e}")

    # First-difference robustness (region × year panel)
    lines.append("\n  [FIRST-DIFFERENCE ROBUSTNESS: Δpct_foreign ~ ΔCPU (region-year panel)]")
    panel = (df.dropna(subset=["pct_foreign","startup_cpu_z","hq_region"])
             .groupby(["hq_region","deal_year"])
             .agg(pct=("pct_foreign","mean"), cpu=("startup_cpu_z","mean"), n=("pct_foreign","count"))
             .reset_index())
    panel = panel[panel["n"]>=5].sort_values(["hq_region","deal_year"])
    panel["d_pct"] = panel.groupby("hq_region")["pct"].diff()
    panel["d_cpu"] = panel.groupby("hq_region")["cpu"].diff()
    fd = panel.dropna(subset=["d_pct","d_cpu"])
    if len(fd) >= 12:
        m_fd = smf.ols("d_pct ~ d_cpu + C(deal_year)", data=fd).fit(cov_type="HC3")
        b = m_fd.params.get("d_cpu", np.nan); p = m_fd.pvalues.get("d_cpu", np.nan)
        lines.append(f"    Δpct_foreign ~ ΔCPU  β={b:>+8.5f}  p={p:.4f} {sig(p)}  N={len(fd)}")
    else:
        lines.append(f"    Too few FD cells: N={len(fd)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: CPU GAP → CROSS-BORDER FLOWS
# ═══════════════════════════════════════════════════════════════════════════════

def model2(df, lines):
    print("\nModel 2: CPU_gap → pct_foreign …")
    lines += ["\n"+"="*72,
              "MODEL 2: CPU DIVERGENCE → CROSS-BORDER FLOWS",
              "  H: |CPU_US − CPU_EU| shifts deals toward the lower-uncertainty region",
              "  Restricted to US/EU startups with deals 2014–2019 (CPU_EU available)",
              "  Year FE absorbs common time trends",
              "="*72]

    controls = "pure_green + log_cap + nrounds + C(hq_region) + C(deal_year)"
    sub = df.dropna(subset=["cpu_gap_z","log_cap","hq_region"])
    sub = sub[sub["hq_region"].isin(["Americas","Europe"])].copy()
    lines.append(f"  US/EU sample: N={len(sub):,}")

    # OLS: pct_foreign ~ cpu_gap
    sub_pf = sub.dropna(subset=["pct_foreign"])
    if len(sub_pf) >= 50:
        try:
            m = smf.ols(f"pct_foreign ~ cpu_gap_z + {controls}", data=sub_pf).fit(cov_type="HC3")
            b = m.params.get("cpu_gap_z", np.nan); p = m.pvalues.get("cpu_gap_z", np.nan)
            lines.append(f"\n  pct_foreign ~ cpu_gap  β={b:>+8.5f}  p={p:.4f} {sig(p)}  adj-R²={m.rsquared_adj:.3f}")
        except Exception as e:
            lines.append(f"  pct_foreign ~ cpu_gap: ERROR {e}")

    # Logit: is_cross_border ~ cpu_gap
    sub_cb = sub.dropna(subset=["is_cross_border"])
    if len(sub_cb) >= 50:
        try:
            m2 = smf.logit(f"is_cross_border ~ cpu_gap_z + {controls}",
                           data=sub_cb).fit(disp=0, maxiter=400)
            b2 = m2.params.get("cpu_gap_z", np.nan); p2 = m2.pvalues.get("cpu_gap_z", np.nan)
            lines.append(f"  is_cross_border ~ cpu_gap β={b2:>+8.5f}  p={p2:.4f} {sig(p2)}  N={int(m2.nobs):,}")
        except Exception as e:
            lines.append(f"  is_cross_border ~ cpu_gap: ERROR {e}")

    # By startup region: startup_cpu vs inv_cpu asymmetry
    lines.append("\n  [By startup region: startup_cpu vs investor_cpu]")
    for region in ["Americas","Europe"]:
        sub_r = sub.dropna(subset=["pct_foreign","startup_cpu_z","inv_cpu_z"])
        sub_r = sub_r[sub_r["hq_region"]==region].copy()
        if len(sub_r) < 50:
            lines.append(f"  {region}: N={len(sub_r)} (too small, skip)")
            continue
        try:
            m_r = smf.ols(
                f"pct_foreign ~ startup_cpu_z + inv_cpu_z + pure_green + log_cap + nrounds + C(deal_year)",
                data=sub_r).fit(cov_type="HC3")
            b_s = m_r.params.get("startup_cpu_z", np.nan); p_s = m_r.pvalues.get("startup_cpu_z", np.nan)
            b_i = m_r.params.get("inv_cpu_z",     np.nan); p_i = m_r.pvalues.get("inv_cpu_z",     np.nan)
            lines.append(f"  {region} (N={int(m_r.nobs):,}): "
                         f"startup_cpu β={b_s:>+.4f} {sig(p_s)}  inv_cpu β={b_i:>+.4f} {sig(p_i)}")
        except Exception as e:
            lines.append(f"  {region}: ERROR {e}")

    # FD robustness
    lines.append("\n  [FIRST-DIFFERENCE ROBUSTNESS: Δpct_foreign ~ Δcpu_gap (region-year)]")
    panel = (sub.dropna(subset=["pct_foreign","cpu_gap_z"])
             .groupby(["hq_region","deal_year"])
             .agg(pct=("pct_foreign","mean"), gap=("cpu_gap_z","mean"), n=("pct_foreign","count"))
             .reset_index())
    panel = panel[panel["n"]>=5].sort_values(["hq_region","deal_year"])
    panel["d_pct"] = panel.groupby("hq_region")["pct"].diff()
    panel["d_gap"] = panel.groupby("hq_region")["gap"].diff()
    fd = panel.dropna(subset=["d_pct","d_gap"])
    if len(fd) >= 8:
        m_fd = smf.ols("d_pct ~ d_gap + C(deal_year)", data=fd).fit(cov_type="HC3")
        b = m_fd.params.get("d_gap", np.nan); p = m_fd.pvalues.get("d_gap", np.nan)
        lines.append(f"    Δpct_foreign ~ Δcpu_gap  β={b:>+8.5f}  p={p:.4f} {sig(p)}  N={len(fd)}")
    else:
        lines.append(f"    Too few FD cells: N={len(fd)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: CROSS-BORDER × STARTUP_CPU → FAILURE
# ═══════════════════════════════════════════════════════════════════════════════

def model3(df, lines):
    print("\nModel 3: is_cross_border × startup_CPU → failure …")
    lines += ["\n"+"="*72,
              "MODEL 3: CROSS-BORDER × STARTUP_CPU → FAILURE",
              "  H: is_cross × startup_CPU < 0 → foreign investors more protective",
              "     when home region has high climate policy uncertainty",
              "  Year FE + controls",
              "="*72]

    controls = "pure_green + log_cap + nrounds + C(hq_region) + C(deal_year)"
    sub = df.dropna(subset=["is_failed","is_cross_border","startup_cpu_z","log_cap","hq_region"]).copy()
    lines.append(f"  N = {len(sub):,}")

    for outcome in ["is_failed","is_exit"]:
        sub_o = df.dropna(subset=[outcome,"is_cross_border","startup_cpu_z",
                                   "log_cap","hq_region"]).copy()
        lines.append(f"\n  Outcome: {outcome}")

        # Main effects
        try:
            m_main = smf.logit(
                f"{outcome} ~ is_cross_border + startup_cpu_z + {controls}",
                data=sub_o).fit(disp=0, maxiter=400)
            b_cb = m_main.params.get("is_cross_border", np.nan)
            p_cb = m_main.pvalues.get("is_cross_border", np.nan)
            b_cp = m_main.params.get("startup_cpu_z",   np.nan)
            p_cp = m_main.pvalues.get("startup_cpu_z",   np.nan)
            lines.append(f"    Main: is_cross β={b_cb:>+.4f} {sig(p_cb)}  startup_cpu β={b_cp:>+.4f} {sig(p_cp)}")
        except Exception as e:
            lines.append(f"    Main effects: ERROR {e}")

        # With interaction
        try:
            m_int = smf.logit(
                f"{outcome} ~ is_cross_border + startup_cpu_z + "
                f"is_cross_border:startup_cpu_z + {controls}",
                data=sub_o).fit(disp=0, maxiter=400)
            int_k = [k for k in m_int.params.index
                     if "is_cross_border:" in k or ":is_cross_border" in k]
            if int_k:
                b_int = m_int.params[int_k[0]]; p_int = m_int.pvalues[int_k[0]]
                verdict = ("CONFIRMED: protective under uncertainty"
                           if (p_int<0.05 and b_int<0 and outcome=="is_failed") else
                           ("MARGINAL" if p_int<0.10 else "n.s."))
                lines.append(f"    Interaction β={b_int:>+.4f}  p={p_int:.4f} {sig(p_int)}  → {verdict}")
        except Exception as e:
            lines.append(f"    Interaction: ERROR {e}")

    # FD robustness (region-year panel)
    lines.append("\n  [FIRST-DIFFERENCE ROBUSTNESS: Δfailure ~ ΔCPU × cross_pct (region-year)]")
    panel = (df.dropna(subset=["is_failed","startup_cpu_z","is_cross_border","hq_region"])
             .groupby(["hq_region","deal_year"])
             .agg(fail=("is_failed","mean"), cpu=("startup_cpu_z","mean"),
                  cross=("is_cross_border","mean"), n=("is_failed","count"))
             .reset_index())
    panel = panel[panel["n"]>=10].sort_values(["hq_region","deal_year"])
    panel["d_fail"]  = panel.groupby("hq_region")["fail"].diff()
    panel["d_cpu"]   = panel.groupby("hq_region")["cpu"].diff()
    panel["cross_l1"]= panel.groupby("hq_region")["cross"].shift(1)
    fd = panel.dropna(subset=["d_fail","d_cpu","cross_l1"])
    if len(fd) >= 10:
        m_fd = smf.ols("d_fail ~ d_cpu + cross_l1 + d_cpu:cross_l1 + C(deal_year)",
                       data=fd).fit(cov_type="HC3")
        int_k = [k for k in m_fd.params.index if "d_cpu:" in k or ":d_cpu" in k]
        if int_k:
            b = m_fd.params[int_k[0]]; p = m_fd.pvalues[int_k[0]]
            lines.append(f"    FD interaction β={b:>+8.5f}  p={p:.4f} {sig(p)}  N={len(fd)}")
    else:
        lines.append(f"    Too few FD cells: N={len(fd)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4: REGIONAL CPU → COMMUNITY ACTIVATION
# ═══════════════════════════════════════════════════════════════════════════════

def model4(df, lines):
    print("\nModel 4: regional CPU → community activation …")
    lines += ["\n"+"="*72,
              "MODEL 4: REGIONAL CPU → GEOGRAPHIC COMMUNITY ACTIVATION",
              "  H: CPU_US↑ → US-dominant communities retreat; EU/Asia communities fill gap",
              "  DV: 1 if company's syndicate included an investor from US/EU/Asia community",
              "  Year FE + controls",
              "="*72]

    controls = "pure_green + log_cap + nrounds + C(hq_region) + C(deal_year)"

    specs = [
        ("has_us_comm",   "startup_cpu_z",  "US-comm ~ startup_CPU",  "H: US CPU↑ → US community retreats (β<0)"),
        ("has_eu_comm",   "startup_cpu_z",  "EU-comm ~ startup_CPU",  "H: US CPU↑ → EU communities substitute"),
        ("has_asia_comm", "startup_cpu_z",  "Asia-comm ~ startup_CPU","H: Asia opportunistic during US uncertainty"),
        ("has_us_comm",   "inv_cpu_z",      "US-comm ~ investor_CPU", "H: high inv-region CPU → US comm less active"),
        ("has_eu_comm",   "inv_cpu_z",      "EU-comm ~ investor_CPU", "H: high inv-region CPU → EU comm less active"),
    ]

    for dv, iv, label, hyp in specs:
        sub = df.dropna(subset=[dv, iv, "log_cap", "hq_region"]).copy()
        lines.append(f"\n  [{label}]  N={len(sub):,}")
        lines.append(f"    {hyp}")
        try:
            m = smf.logit(f"{dv} ~ {iv} + {controls}", data=sub).fit(disp=0, maxiter=400)
            b = m.params.get(iv, np.nan); p = m.pvalues.get(iv, np.nan)
            verdict = "CONFIRMED" if p<0.05 else ("MARGINAL" if p<0.10 else "n.s.")
            lines.append(f"    {iv} β={b:>+8.4f}  p={p:.4f} {sig(p)}  → {verdict}")
        except Exception as e:
            lines.append(f"    ERROR: {e}")

    # FD robustness
    lines.append("\n  [FIRST-DIFFERENCE ROBUSTNESS: Δcomm_share ~ ΔCPU (region-year)]")
    for dv, iv, label in [
        ("has_us_comm",  "startup_cpu_z", "US-comm ~ startup_CPU"),
        ("has_eu_comm",  "startup_cpu_z", "EU-comm ~ startup_CPU"),
    ]:
        panel = (df.dropna(subset=[dv, iv, "hq_region"])
                 .groupby(["hq_region","deal_year"])
                 .agg(dv_m=(dv,"mean"), iv_m=(iv,"mean"), n=(dv,"count"))
                 .reset_index())
        panel = panel[panel["n"]>=10].sort_values(["hq_region","deal_year"])
        panel["d_dv"] = panel.groupby("hq_region")["dv_m"].diff()
        panel["d_iv"] = panel.groupby("hq_region")["iv_m"].diff()
        fd = panel.dropna(subset=["d_dv","d_iv"])
        if len(fd) < 8:
            lines.append(f"  {label}: too few FD cells ({len(fd)})")
            continue
        m_fd = smf.ols("d_dv ~ d_iv + C(deal_year)", data=fd).fit(cov_type="HC3")
        b = m_fd.params.get("d_iv", np.nan); p = m_fd.pvalues.get("d_iv", np.nan)
        lines.append(f"  {label}: β={b:>+8.5f}  p={p:.4f} {sig(p)}  N={len(fd)}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary(df, cpu_monthly):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140, facecolor="white")
    axes = axes.flatten()

    # Panel 1: Cross-border rate by startup region
    ax = axes[0]
    grp = (df[df["hq_region"].isin(["Americas","Europe","Asia"])]
           .dropna(subset=["is_cross_border"])
           .groupby("hq_region")["is_cross_border"]
           .agg(["mean","count"]))
    colors = ["#2980b9","#27ae60","#e67e22"]
    bars = ax.bar(grp.index, grp["mean"]*100, color=colors, alpha=0.8, edgecolor="white")
    for bar, (idx, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"N={int(row['count']):,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("% Cross-Border"); ax.set_ylim(bottom=0)
    ax.set_title("Cross-Border Rate by Startup Region", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: CPU_US and CPU_EU monthly
    ax = axes[1]
    cpu_df = cpu_monthly.reset_index()
    cpu_df["date"] = pd.to_datetime(cpu_df["ym"] + "-01")
    cpu_df = cpu_df[cpu_df["date"] >= "2013-01-01"]
    ax.plot(cpu_df["date"], cpu_df["CPU_US_z"], color="#e74c3c", lw=1.5, label="CPU_US (z)")
    eu_sub = cpu_df.dropna(subset=["CPU_EU_z"])
    ax.plot(eu_sub["date"], eu_sub["CPU_EU_z"], color="#3498db", lw=1.5, ls="--", label="CPU_EU (z)")
    ax.axvline(pd.Timestamp("2020-01-01"), ls=":", color="gray", lw=1)
    ax.text(pd.Timestamp("2020-02-01"), ax.get_ylim()[1]*0.85, "EU ends 2019-12", fontsize=7, color="gray")
    ax.legend(fontsize=8)
    ax.set_title("Regional CPU (monthly, z-scores)", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3: pct_foreign distribution
    ax = axes[2]
    pf = df["pct_foreign"].dropna()
    ax.hist(pf, bins=20, color="#9b59b6", alpha=0.7, edgecolor="white")
    ax.axvline(pf.mean(), color="black", lw=1.5, ls="--", label=f"Mean={pf.mean():.2f}")
    ax.set_xlabel("% Foreign Investors"); ax.set_ylabel("Companies")
    ax.set_title("Distribution of Foreign Investor Share", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 4: pct_foreign by deal year
    ax = axes[3]
    yr = (df.dropna(subset=["pct_foreign","deal_year"])
          .groupby("deal_year")["pct_foreign"]
          .agg(["mean","sem","count"]))
    yr = yr[yr["count"]>=30]
    ax.bar(yr.index, yr["mean"]*100, yerr=yr["sem"]*100*1.96,
           color="#16a085", alpha=0.7, edgecolor="white", capsize=3)
    ax.set_xlabel("Deal Year"); ax.set_ylabel("% Foreign Investors")
    ax.set_title("Foreign Investor Share by Year", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Regional CPU × Cross-Border Syndication — Company Level (N=6,718)\n"
                 "CPU_EU: Basaglia (2014–2019) | CPU_US: Gavriilidis (2013–2026)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT/"fig1_regional_cpu.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved fig1_regional_cpu.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = [
        "="*72,
        "REGIONAL CPU × CROSS-BORDER SYNDICATION — COMPANY LEVEL",
        "Monthly CPU matched to each company's first investment date.",
        "CPU_EU (Basaglia): covers 1990-2019-12 | CPU_US (Gavriilidis): 2013–2026",
        "YEAR FIXED EFFECTS in all models. First-differences as robustness.",
        "="*72,
        "\nDATA NOTES:",
        "  CPU_EU ends 2019-12 → companies with first deal after 2019 have NA for EU CPU.",
        "  CPU_gap defined only for US/EU startups with both series available.",
        "  z-scores computed over 2014-2019 overlapping window.",
        "  is_cross_border: majority investor region ≠ startup HQ region.",
    ]

    print("Loading data …")
    cpu_monthly = build_monthly_cpu()
    inv_geo, inv_comm, us_comms, eu_comms, asia_comms = build_investor_maps()

    print("\nBuilding company geo dataset …")
    df = build_company_geo(cpu_monthly, inv_geo, inv_comm, us_comms, eu_comms, asia_comms)
    df.to_csv(OUT/"company_geo_dataset.csv", index=False)

    lines += [
        f"\nSAMPLE COVERAGE (N={len(df):,}):",
        f"  With startup CPU:             {df['startup_cpu_z'].notna().sum():,}",
        f"  With investor CPU:            {df['inv_cpu_z'].notna().sum():,}",
        f"  With CPU_gap (US/EU, ≤2019): {df['cpu_gap_z'].notna().sum():,}",
        f"  is_cross_border classifiable: {df['is_cross_border'].notna().sum():,}",
        f"  is_cross_border = 1:          {(df['is_cross_border']==1).sum():,}",
        f"  pct_foreign classifiable:     {df['pct_foreign'].notna().sum():,}",
    ]

    model1(df, lines)
    model2(df, lines)
    model3(df, lines)
    model4(df, lines)

    lines += ["\n"+"="*72, "INTERPRETATION GUIDE", "="*72,
              "  Unit: company (cross-section). CPU varies by startup region × month-of-first-deal.",
              "  Year FE removes cohort trends. Cross-sectional CPU variation is limited.",
              "  CPU_gap analyses restricted to US/EU deals 2014–2019 (small N).",
              "  FD robustness uses region-year averages (not company-level differences)."]

    output = "\n".join(lines)
    print("\n" + output)
    (OUT/"results.txt").write_text(output)
    plot_summary(df, cpu_monthly)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
