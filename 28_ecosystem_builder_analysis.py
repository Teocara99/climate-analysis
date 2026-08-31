"""
28 — Ecosystem Builder Deep-Dive.

Top-20 investors by deal count drive the cross-community effect (Block 8 showed
that removing them kills E and F). This script unpacks:

1. Portfolio profile per institution (failure rate, cross-community %, communities bridged)
2. Institution-type comparison (government vs accelerator vs climate VC)
3. Regression: failure ~ has_eco_builder + has_other_cross_comm + controls
   → Is it the INSTITUTION or the cross-community ACCESS that matters?
4. Cross-community bridging map: which communities does each institution link?

Outputs → output/ecosystem_builders/
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from collections import Counter, defaultdict
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
OUT  = ROOT / "output" / "ecosystem_builders"
OUT.mkdir(parents=True, exist_ok=True)

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

def parse_investors(s):
    if pd.isna(s): return []
    return [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]

# ── Institution-type labels ───────────────────────────────────────────────────
INST_TYPE = {
    "Plug and Play Tech Center":    "Accelerator",
    "U.S. Department of Energy":    "Government",
    "Innovate UK":                  "Government",
    "US National Science Foundation": "Government",
    "SOSV":                         "Accelerator",
    "Techstars":                    "Accelerator",
    "European Innovation Council":  "Government",
    "Climate Capital":              "Climate VC",
    "InnoEnergy":                   "Government",
    "Greentown Labs":               "Accelerator",
    "Creative Destruction Lab":     "Accelerator",
    "EIT Climate-KIC":              "Government",
    "Cleantech Open":               "Accelerator",
    "MassChallenge":                "Accelerator",
    "Breakthrough Energy":          "Climate VC",
    "Third Derivative":             "Accelerator",
    "Y Combinator":                 "Accelerator",
    "Lowercarbon Capital":          "Climate VC",
    "EIT Food":                     "Government",
    "Horizon 2020":                 "Government",
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: BUILD DEAL-LEVEL INVESTOR DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data …")
    cdf = pd.read_csv(ROOT / "output/cpu/cpu_company_dataset.csv")
    cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
    cdf["is_exit"]   = cdf["outcome"].isin(["Acquired","IPO / Public"]).astype(float)
    cdf["log_cap"]   = np.log1p(cdf["Total Raised"].fillna(0))
    cdf["nrounds"]   = np.log1p(cdf["n_rounds"].fillna(1))
    cdf["deal_year"] = cdf["deal_year"].fillna(2018).astype(int)
    cdf["company_key"] = cdf["Companies"].str.lower().str.strip()

    nm = pd.read_csv(ROOT / "output/network/network_metrics.csv")
    nm["inv_key"] = nm["investor"].str.lower().str.strip()
    inv_to_comm = dict(zip(nm["inv_key"], nm["community"]))

    top20_names = nm.nlargest(20, "total_deals")["investor"].tolist()
    top20_keys  = {n.lower().strip() for n in top20_names}

    deals = load_deals()
    deals["Deal Date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["Deal Date", "Investors"])
    deals["company_key"] = deals["Companies"].str.lower().str.strip()
    deals["year"] = deals["Deal Date"].dt.year

    # First deal per company
    first_deals = (deals.sort_values("Deal Date")
                   .drop_duplicates("company_key", keep="first")
                   [["company_key", "Investors", "year"]])

    # Per company: which top-20 investors participated?
    def company_flags(inv_str):
        names = parse_investors(inv_str)
        keys  = {n.lower() for n in names}
        comms = {inv_to_comm.get(k) for k in keys if inv_to_comm.get(k) is not None}
        eco_keys = top20_keys & keys
        has_eco  = int(bool(eco_keys))
        eco_names = [n for n in names if n.lower() in top20_keys]
        is_cross  = int(len(comms) >= 2)
        return eco_names, has_eco, is_cross, comms

    records = []
    for _, row in first_deals.iterrows():
        eco_names, has_eco, is_cross, comms = company_flags(row["Investors"])
        records.append({
            "company_key":   row["company_key"],
            "eco_investors": eco_names,
            "has_eco":       has_eco,
            "is_cross_recomp": is_cross,
            "n_comms_recomp":  len(comms),
        })
    eco_df = pd.DataFrame(records)

    df = cdf.merge(eco_df, on="company_key", how="left")
    # Cross-community from original dataset (ground truth)
    # Use has_eco from recomputed + cross_community from original
    df["has_eco"] = df["has_eco"].fillna(0)
    df["is_cross_recomp"] = df["is_cross_recomp"].fillna(0)
    # "other cross-community": cross in original data but no eco builder
    df["has_other_cross"] = ((df["cross_community"] == 1) & (df["has_eco"] == 0)).astype(float)
    # "eco cross-community": has eco builder AND cross in original
    df["eco_and_cross"]   = ((df["cross_community"] == 1) & (df["has_eco"] == 1)).astype(float)

    print(f"  N companies: {len(df):,}")
    print(f"  has_eco=1: {df['has_eco'].sum():,} ({df['has_eco'].mean():.1%})")
    print(f"  has_other_cross=1: {df['has_other_cross'].sum():,} ({df['has_other_cross'].mean():.1%})")
    return df, nm, top20_names, top20_keys, inv_to_comm, deals


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: PER-INSTITUTION PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

def profile_institutions(df, nm, top20_names, top20_keys, inv_to_comm, deals, lines):
    print("\nBuilding per-institution profiles …")
    lines += ["\n"+"="*72,
              "INSTITUTION PROFILES: Top-20 Ecosystem Builders",
              "="*72]

    # For each institution: find its portfolio companies
    deals_first = (deals.sort_values("Deal Date")
                   .drop_duplicates("company_key", keep="first"))

    inst_profiles = []
    for inv_name in top20_names:
        inv_key  = inv_name.lower().strip()
        inst_type = INST_TYPE.get(inv_name, "Other")

        # Companies in portfolio (from all deals, not just first)
        port_mask = deals["Investors"].str.lower().str.contains(
            inv_key, na=False, regex=False)
        port_companies = deals[port_mask]["company_key"].unique()

        # Match to outcomes
        port_df = df[df["company_key"].isin(port_companies)]
        n_port  = len(port_df)
        if n_port == 0:
            inst_profiles.append({"investor": inv_name, "inst_type": inst_type, "n": 0})
            continue

        fail_rate = port_df["is_failed"].mean()
        exit_rate = port_df["is_exit"].mean()
        cross_rate = port_df["cross_community"].mean()

        # Community bridging: for each deal with this investor, what communities are co-investors from?
        port_deals = deals[port_mask].copy()
        co_inv_comms = Counter()
        own_comm = inv_to_comm.get(inv_key)
        for _, row in port_deals.iterrows():
            for co_inv in parse_investors(row["Investors"]):
                if co_inv.lower() == inv_key: continue
                c = inv_to_comm.get(co_inv.lower())
                if c is not None: co_inv_comms[c] += 1
        n_comms_bridged = len(co_inv_comms)
        top_comms = [c for c,_ in co_inv_comms.most_common(4)]

        nm_row = nm[nm["investor"] == inv_name].iloc[0] if (nm["investor"]==inv_name).any() else None
        betweenness = nm_row["betweenness"] if nm_row is not None else np.nan
        own_comm_id = nm_row["community"] if nm_row is not None else np.nan
        total_deals = nm_row["total_deals"] if nm_row is not None else np.nan

        inst_profiles.append({
            "investor":       inv_name,
            "inst_type":      inst_type,
            "n":              n_port,
            "total_deals":    total_deals,
            "fail_rate":      fail_rate,
            "exit_rate":      exit_rate,
            "cross_rate":     cross_rate,
            "n_comms_bridged":n_comms_bridged,
            "own_community":  int(own_comm_id) if not pd.isna(own_comm_id) else None,
            "top_comms":      top_comms,
            "betweenness":    betweenness,
        })

    prof_df = pd.DataFrame(inst_profiles)
    prof_df["inst_type"] = prof_df["investor"].map(INST_TYPE).fillna("Other")
    prof_df.to_csv(OUT / "institution_profiles.csv", index=False)

    # Print table
    eco_avg_fail = df["is_failed"].mean()
    lines.append(f"\n  Ecosystem average failure rate: {eco_avg_fail:.1%}")
    lines.append(f"\n  {'Institution':<35}  {'Type':<13}  {'N':>5}  {'Fail%':>7}  "
                 f"{'vs avg':>7}  {'Exit%':>7}  {'Cross%':>7}  {'N comms':>7}  {'Betw.':>8}")
    lines.append(f"  {'-'*105}")

    for _, r in prof_df.sort_values("inst_type").iterrows():
        if r["n"] == 0: continue
        vs_avg = r["fail_rate"] - eco_avg_fail
        lines.append(
            f"  {r['investor']:<35}  {r['inst_type']:<13}  {int(r['n']):>5}  "
            f"{r['fail_rate']:>6.1%}  {vs_avg:>+6.1%}  {r['exit_rate']:>7.1%}  "
            f"{r['cross_rate']:>7.1%}  {int(r['n_comms_bridged']):>7}  "
            f"{r['betweenness']:>8.4f}"
        )

    # Type-level summary
    lines.append("\n  [BY INSTITUTION TYPE]")
    lines.append(f"  {'Type':<15}  {'N institutions':>15}  {'Avg portfolio N':>15}  "
                 f"{'Avg fail%':>10}  {'Avg cross%':>11}  {'Avg N comms':>12}")
    for itype in ["Government","Accelerator","Climate VC"]:
        sub = prof_df[prof_df["inst_type"]==itype]
        if sub.empty: continue
        lines.append(f"  {itype:<15}  {len(sub):>15}  {sub['n'].mean():>15.0f}  "
                     f"{sub['fail_rate'].mean():>10.1%}  {sub['cross_rate'].mean():>11.1%}  "
                     f"{sub['n_comms_bridged'].mean():>12.1f}")

    return prof_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: KEY INSTITUTION COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_institutions(df, top20_names, inv_to_comm, deals, lines):
    print("\nInstitution-level comparisons …")
    lines += ["\n"+"="*72,
              "INDIVIDUAL INSTITUTION EFFECTS ON FAILURE",
              "  For each top-20: logit(failure) ~ has_this_investor + controls",
              "  Controls: log_cap + nrounds + C(hq_region) + C(deal_year)",
              "="*72,
              f"  Ecosystem average failure: {df['is_failed'].mean():.1%}"]

    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    results = []

    for inv_name in top20_names:
        inv_key = inv_name.lower().strip()
        mask = deals["Investors"].str.lower().str.contains(
            inv_key, na=False, regex=False)
        port_companies = deals[mask]["company_key"].unique()

        df_flag = df.copy()
        df_flag["has_this"] = df_flag["company_key"].isin(port_companies).astype(float)
        n_treated = df_flag["has_this"].sum()
        fail_treated = df_flag[df_flag["has_this"]==1]["is_failed"].mean()
        fail_control = df_flag[df_flag["has_this"]==0]["is_failed"].mean()
        raw_diff = fail_treated - fail_control

        sub = df_flag.dropna(subset=["is_failed","log_cap","hq_region"])
        try:
            m = smf.logit(f"is_failed ~ has_this + {ctrl}", data=sub).fit(disp=0, maxiter=400)
            b = m.params.get("has_this", np.nan); p = m.pvalues.get("has_this", np.nan)
        except:
            b = np.nan; p = np.nan

        results.append({
            "investor":    inv_name,
            "inst_type":   INST_TYPE.get(inv_name,"Other"),
            "n_portfolio": n_treated,
            "fail_treated":fail_treated,
            "fail_control":fail_control,
            "raw_diff":    raw_diff,
            "logit_b":     b,
            "logit_p":     p,
        })

    res_df = pd.DataFrame(results).sort_values("logit_b")
    lines.append(f"\n  {'Institution':<35}  {'Type':<13}  {'N':>5}  "
                 f"{'Fail(own)':>9}  {'Raw diff':>9}  {'Logit β':>9}  {'p':>8}  sig")
    lines.append(f"  {'-'*105}")

    for _, r in res_df.iterrows():
        lines.append(
            f"  {r['investor']:<35}  {r['inst_type']:<13}  {int(r['n_portfolio']):>5}  "
            f"{r['fail_treated']:>9.1%}  {r['raw_diff']:>+8.1%}  "
            f"{r['logit_b']:>+9.4f}  {r['logit_p']:>8.4f}  {sig(r['logit_p'])}"
        )

    return res_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: INSTITUTION vs CROSS-COMMUNITY SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def institution_vs_cross_community(df, lines):
    print("\nSeparating institution effect from cross-community effect …")
    lines += ["\n"+"="*72,
              "INSTITUTION vs CROSS-COMMUNITY: WHAT DRIVES THE EFFECT?",
              "  Decomposition:",
              "    has_eco:          company backed by any top-20 ecosystem builder",
              "    has_other_cross:  cross-community WITHOUT any top-20 builder",
              "    eco_and_cross:    cross-community WITH a top-20 builder",
              "  H: If the effect is pure cross-community access:",
              "     has_eco alone should be n.s.; has_other_cross should be significant.",
              "  H: If the effect is the institution itself:",
              "     has_eco should be significant; has_other_cross should be weaker.",
              "="*72]

    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"

    # Descriptive first
    lines.append(f"\n  SAMPLE SPLIT:")
    for mask, label in [
        (df["has_eco"]==0, "No eco builder"),
        (df["has_eco"]==1, "Has eco builder"),
        ((df["has_eco"]==0) & (df["cross_community"]==1), "Cross-comm, no eco"),
        ((df["has_eco"]==1) & (df["cross_community"]==1), "Cross-comm + eco"),
        ((df["has_eco"]==1) & (df["cross_community"]==0), "Eco, within-community"),
    ]:
        sub = df[mask]
        if len(sub) < 30: continue
        lines.append(f"    {label:<40}  N={len(sub):,}  "
                     f"fail={sub['is_failed'].mean():.1%}  "
                     f"exit={sub['is_exit'].mean():.1%}")

    # Model M1: original cross_community
    lines.append("\n  [M1] BASELINE: failure ~ cross_community (original)")
    sub = df.dropna(subset=["is_failed","cross_community","log_cap","hq_region"])
    m1 = smf.logit(f"is_failed ~ cross_community + {ctrl}", data=sub).fit(disp=0, maxiter=400)
    b1 = m1.params.get("cross_community",np.nan); p1 = m1.pvalues.get("cross_community",np.nan)
    lines.append(f"    cross_community  β={b1:>+7.4f}  p={p1:.4f} {sig(p1)}  N={int(m1.nobs):,}")

    # Model M2: eco builder vs other cross-community (two separate dummies)
    lines.append("\n  [M2] DECOMPOSED: failure ~ has_eco + has_other_cross")
    lines.append("  (has_eco: any top-20 builder; has_other_cross: cross-comm without top-20)")
    sub2 = df.dropna(subset=["is_failed","has_eco","has_other_cross","log_cap","hq_region"])
    try:
        m2 = smf.logit(f"is_failed ~ has_eco + has_other_cross + {ctrl}",
                       data=sub2).fit(disp=0, maxiter=400)
        b_eco = m2.params.get("has_eco",np.nan);  p_eco = m2.pvalues.get("has_eco",np.nan)
        b_oth = m2.params.get("has_other_cross",np.nan); p_oth = m2.pvalues.get("has_other_cross",np.nan)
        lines.append(f"    has_eco          β={b_eco:>+7.4f}  p={p_eco:.4f} {sig(p_eco)}")
        lines.append(f"    has_other_cross  β={b_oth:>+7.4f}  p={p_oth:.4f} {sig(p_oth)}")
        lines.append(f"    N={int(m2.nobs):,}")
    except Exception as e:
        lines.append(f"    ERROR: {e}")

    # Model M3: three-way split
    lines.append("\n  [M3] THREE-WAY: eco_only / eco+cross / other_cross (vs within, no eco)")
    df["group"] = "base"  # within-community, no eco builder
    df.loc[(df["has_eco"]==1) & (df["cross_community"]==0), "group"] = "eco_within"
    df.loc[(df["has_eco"]==1) & (df["cross_community"]==1), "group"] = "eco_cross"
    df.loc[(df["has_eco"]==0) & (df["cross_community"]==1), "group"] = "other_cross"

    sub3 = df.dropna(subset=["is_failed","group","log_cap","hq_region"])
    try:
        m3 = smf.logit(f"is_failed ~ C(group) + {ctrl}", data=sub3).fit(disp=0, maxiter=400)
        for key in m3.params.index:
            if "group" in key:
                b = m3.params[key]; p = m3.pvalues[key]
                label = key.replace("C(group)[T.","").rstrip("]")
                lines.append(f"    {label:<20}  β={b:>+7.4f}  p={p:.4f} {sig(p)}")
        lines.append(f"    N={int(m3.nobs):,}  (ref = within-community, no eco builder)")
    except Exception as e:
        lines.append(f"    ERROR: {e}")

    # Model M4: by institution type
    lines.append("\n  [M4] BY INSTITUTION TYPE: Government vs Accelerator vs Climate VC")
    for itype in ["Government","Accelerator","Climate VC"]:
        investors_of_type = [inv for inv,t in INST_TYPE.items() if t==itype]
        inv_keys = {n.lower() for n in investors_of_type}
        df[f"has_{itype.lower().replace(' ','_')}"] = df["eco_investors"].apply(
            lambda lst: int(any(n.lower() in inv_keys for n in (lst if isinstance(lst,list) else [])))
        )
    sub4 = df.dropna(subset=["is_failed","log_cap","hq_region"])
    try:
        m4 = smf.logit(
            f"is_failed ~ has_government + has_accelerator + has_climate_vc + {ctrl}",
            data=sub4).fit(disp=0, maxiter=400)
        for var in ["has_government","has_accelerator","has_climate_vc"]:
            b = m4.params.get(var,np.nan); p = m4.pvalues.get(var,np.nan)
            lines.append(f"    {var:<25}  β={b:>+7.4f}  p={p:.4f} {sig(p)}")
        lines.append(f"    N={int(m4.nobs):,}")
    except Exception as e:
        lines.append(f"    ERROR: {e}")

    # Model M5: eco builder CONTROLS AWAY the cross-community effect?
    lines.append("\n  [M5] DOES has_eco ABSORB cross_community?")
    lines.append("  Adding has_eco to the cross_community model — does cross_community survive?")
    sub5 = df.dropna(subset=["is_failed","cross_community","has_eco","log_cap","hq_region"])
    try:
        m5 = smf.logit(f"is_failed ~ cross_community + has_eco + {ctrl}",
                       data=sub5).fit(disp=0, maxiter=400)
        b_cc = m5.params.get("cross_community",np.nan); p_cc = m5.pvalues.get("cross_community",np.nan)
        b_ec = m5.params.get("has_eco",np.nan); p_ec = m5.pvalues.get("has_eco",np.nan)
        lines.append(f"    cross_community  β={b_cc:>+7.4f}  p={p_cc:.4f} {sig(p_cc)}")
        lines.append(f"    has_eco          β={b_ec:>+7.4f}  p={p_ec:.4f} {sig(p_ec)}")
        lines.append(f"    N={int(m5.nobs):,}")
        if p_cc < 0.10 and p_ec < 0.10:
            lines.append("    → BOTH significant: eco builder AND cross-community have independent effects")
        elif p_cc < 0.10:
            lines.append("    → cross_community survives: institution does NOT explain the cross-comm effect")
        elif p_ec < 0.10:
            lines.append("    → eco absorbs cross_community: the effect IS the institution, not cross-comm access")
        else:
            lines.append("    → neither significant jointly (collinearity?)")
    except Exception as e:
        lines.append(f"    ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMMUNITY BRIDGING MAP
# ═══════════════════════════════════════════════════════════════════════════════

def community_bridging(nm, top20_names, inv_to_comm, deals, lines):
    print("\nBuilding community bridging map …")
    lines += ["\n"+"="*72,
              "COMMUNITY BRIDGING: Which Communities Does Each Institution Link?",
              "  Own community + top-4 co-investor communities (by co-investment frequency)",
              "="*72]

    n_comms = len(set(inv_to_comm.values()))
    lines.append(f"  Total Louvain communities: {n_comms}")
    lines.append(f"\n  {'Institution':<35}  {'Own comm':>9}  {'Co-inv communities (top 4 by freq)':>40}")
    lines.append(f"  {'-'*90}")

    for inv_name in top20_names:
        inv_key = inv_name.lower().strip()
        own_comm = inv_to_comm.get(inv_key, "?")

        port_mask = deals["Investors"].str.lower().str.contains(
            inv_key, na=False, regex=False)
        port_deals = deals[port_mask]

        co_comms = Counter()
        for _, row in port_deals.iterrows():
            for co in parse_investors(row["Investors"]):
                if co.lower() == inv_key: continue
                c = inv_to_comm.get(co.lower())
                if c is not None and c != own_comm:
                    co_comms[c] += 1

        top4 = [f"C{c}(×{cnt})" for c,cnt in co_comms.most_common(4)]
        lines.append(f"  {inv_name:<35}  C{own_comm:>8}  {', '.join(top4) if top4 else 'within only'}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ecosystem(prof_df, res_df, df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=140, facecolor="white")
    axes = axes.flatten()

    type_colors = {"Government":"#3498db","Accelerator":"#e67e22","Climate VC":"#27ae60","Other":"#95a5a6"}

    # Panel 1: Portfolio failure rate vs ecosystem average
    ax = axes[0]
    eco_avg = df["is_failed"].mean()
    prof_s  = prof_df[prof_df["n"]>0].copy().sort_values("fail_rate")
    colors  = [type_colors.get(t,"#95a5a6") for t in prof_s["inst_type"]]
    bars = ax.barh(range(len(prof_s)), prof_s["fail_rate"]*100, color=colors, alpha=0.8)
    ax.axvline(eco_avg*100, ls="--", color="black", lw=1.5, label=f"Eco avg {eco_avg:.1%}")
    ax.set_yticks(range(len(prof_s)))
    ax.set_yticklabels(prof_s["investor"], fontsize=8)
    ax.set_xlabel("Portfolio Failure Rate (%)")
    ax.set_title("Portfolio Failure Rate by Institution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    handles = [mpatches.Patch(color=c, label=t) for t,c in type_colors.items() if t!="Other"]
    ax.legend(handles=handles + [plt.Line2D([0],[0],color="black",ls="--",label=f"Avg {eco_avg:.1%}")],
              fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 2: Cross-community rate by institution
    ax = axes[1]
    prof_s2 = prof_df[prof_df["n"]>0].copy().sort_values("cross_rate")
    colors2 = [type_colors.get(t,"#95a5a6") for t in prof_s2["inst_type"]]
    ax.barh(range(len(prof_s2)), prof_s2["cross_rate"]*100, color=colors2, alpha=0.8)
    ax.axvline(df["cross_community"].mean()*100, ls="--", color="black", lw=1.5,
               label=f"Avg {df['cross_community'].mean():.1%}")
    ax.set_yticks(range(len(prof_s2)))
    ax.set_yticklabels(prof_s2["investor"], fontsize=8)
    ax.set_xlabel("% Portfolio with Cross-Community Syndicate")
    ax.set_title("Cross-Community Rate by Institution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Panel 3: Logit β (failure effect) per institution
    ax = axes[2]
    res_s = res_df[res_df["n_portfolio"] >= 50].copy().sort_values("logit_b")
    colors3 = [type_colors.get(t,"#95a5a6") for t in res_s["inst_type"]]
    bars3 = ax.barh(range(len(res_s)), res_s["logit_b"], color=colors3, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)
    for i, (_, r) in enumerate(res_s.iterrows()):
        if r["logit_p"] < 0.05:
            ax.text(r["logit_b"] + (0.01 if r["logit_b"]<0 else -0.01),
                    i, sig(r["logit_p"]), va="center",
                    ha="right" if r["logit_b"]<0 else "left", fontsize=8, color="black")
    ax.set_yticks(range(len(res_s)))
    ax.set_yticklabels(res_s["investor"], fontsize=8)
    ax.set_xlabel("Logit β (failure ~ has_institution + controls)")
    ax.set_title("Individual Institution Effect on Failure Risk", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles3 = [mpatches.Patch(color=c, label=t) for t,c in type_colors.items() if t!="Other"]
    ax.legend(handles=handles3, fontsize=7, loc="lower right")

    # Panel 4: Group comparison (cross × eco)
    ax = axes[3]
    grp_labels = ["Within-comm\n(no eco)", "Eco, within-\ncommunity",
                  "Cross-comm\n(no eco)", "Cross-comm\n+ Eco"]
    masks = [
        (df["has_eco"]==0) & (df["cross_community"]==0),
        (df["has_eco"]==1) & (df["cross_community"]==0),
        (df["has_eco"]==0) & (df["cross_community"]==1),
        (df["has_eco"]==1) & (df["cross_community"]==1),
    ]
    grp_colors = ["#e74c3c","#e67e22","#3498db","#27ae60"]
    vals = [df[m]["is_failed"].mean()*100 for m in masks]
    ns   = [df[m].shape[0] for m in masks]
    bars4 = ax.bar(grp_labels, vals, color=grp_colors, alpha=0.8, edgecolor="white")
    for bar, v, n in zip(bars4, vals, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{v:.1f}%\nN={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Failure Rate: Eco Builder × Cross-Community", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Ecosystem Builder Analysis — Top-20 Institutions in Climate VC",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT/"fig1_ecosystem_builders.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved fig1_ecosystem_builders.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = ["="*72,
             "ECOSYSTEM BUILDER DEEP-DIVE",
             "Top-20 investors by deal count (identified as key drivers in Block 8)",
             "="*72]

    df, nm, top20_names, top20_keys, inv_to_comm, deals = load_data()

    prof_df = profile_institutions(df, nm, top20_names, top20_keys, inv_to_comm, deals, lines)
    res_df  = compare_institutions(df, top20_names, inv_to_comm, deals, lines)
    institution_vs_cross_community(df, lines)
    community_bridging(nm, top20_names, inv_to_comm, deals, lines)

    # ── Master interpretation ──
    lines += ["\n"+"="*72, "INTERPRETATION", "="*72]
    lines += [
        "\n  KEY QUESTION: Is the cross-community advantage driven by the INSTITUTION",
        "  (having Techstars/DOE/Y Combinator) or by the CROSS-COMMUNITY ACCESS itself?",
        "\n  Evidence to synthesise:",
        "  M2: has_eco vs has_other_cross → which dominates?",
        "  M5: does has_eco absorb cross_community in same model?",
        "  M4: which institution TYPE drives failure reduction?",
        "\n  Mechanism candidates:",
        "  (A) Institution provides quality signal → better companies get these investors",
        "  (B) Institution provides cross-community access → network information matters",
        "  (C) Both: institutions and network access have independent effects",
    ]

    output = "\n".join(lines)
    print("\n" + output)
    (OUT/"results.txt").write_text(output)

    plot_ecosystem(prof_df, res_df, df)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
