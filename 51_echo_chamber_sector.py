"""
Script 51: Echo Chamber Effect by Sector
=========================================

Tests whether the 2×2 (Expert × Network Diversity) failure pattern
from Script 50 varies by industry sector and technology cluster.

Echo chamber penalty = failure(B: Expert+Same) - failure(A: Expert+Diverse)
Hypothesis: penalty strongest in HIGH-UNCERTAINTY frontier technologies
where expert priors are most likely to be wrong.
"""

import warnings; warnings.filterwarnings("ignore")
import sys, pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

OUT  = Path("output/echo_chamber"); OUT.mkdir(parents=True, exist_ok=True)
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ── Load all data ─────────────────────────────────────────────────────────
print("Loading data…")

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]        = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]   = cdf["log_capital"]
cdf["nrounds"]   = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

net = pd.read_csv("output/network/network_metrics.csv")
net["investor"]  = net["investor"].str.strip()
inv_to_comm      = net.set_index("investor")["community"].to_dict()

inv_clusters = pd.read_csv("output/semantic/investor_clusters.csv")
inv_clusters["investor"] = inv_clusters["investor"].str.strip()
inv_to_semclust = inv_clusters.set_index("investor")["thematic_cluster"].to_dict()

inv_emb_raw  = np.load("data/investor_embeddings.npy")
inv_names    = pd.read_csv("data/investor_names.csv")["investor"].tolist()
inv_emb_norm = normalize(inv_emb_raw, norm="l2")
inv_emb_dict = {n: inv_emb_norm[i] for i, n in enumerate(inv_names)}

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    sp = pickle.load(f)
startup_emb_raw  = sp["embs_raw"]
startup_names_raw = [n.strip() for n in sp["names"]]
startup_emb_norm  = normalize(startup_emb_raw, norm="l2")
startup_emb_dict  = {n: startup_emb_norm[i] for i, n in enumerate(startup_names_raw)}

ns = pd.read_csv("output/novelty/novelty_scores.csv")
ns["co"] = ns["name"].str.strip()

deals = load_deals()
deals["co"]   = deals["Companies"].str.strip()
deals["Year"] = pd.to_numeric(deals["Year"], errors="coerce")

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""')
         .dropna(subset=["Year"]))
co_all_inv = d_exp.groupby("co")["inv"].apply(set).to_dict()

# Sector from PitchBook deal data
co_sector = (deals.dropna(subset=["Primary PitchBook Industry Group"])
             .groupby("co")["Primary PitchBook Industry Group"]
             .agg(lambda x: x.mode()[0])
             .reset_index()
             .rename(columns={"Primary PitchBook Industry Group": "sector"}))

# Also industry code (finer)
co_code = (deals.dropna(subset=["Primary PitchBook Industry Code"])
           .groupby("co")["Primary PitchBook Industry Code"]
           .agg(lambda x: x.mode()[0])
           .reset_index()
           .rename(columns={"Primary PitchBook Industry Code": "industry_code"}))

print(f"  Companies with sector: {len(co_sector):,}")

# ── Startup K-means tech clusters (same as Script 50) ─────────────────────
print("Fitting startup K-means (K=15)…")
km = KMeans(n_clusters=15, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(startup_emb_norm)

tech_cluster_labels = {
    0:  "Carbon / Hydrogen",
    1:  "Solar / Renewable",
    2:  "Data / Monitoring",
    3:  "AgTech / Indoor Farming",
    4:  "EV / Mobility",
    5:  "Grid / Energy Mgmt",
    6:  "Carbon Data / Climate",
    7:  "Wind Power",
    8:  "Battery / Storage",
    9:  "Waste / Water",
    10: "Alt Protein / Food",
    11: "Plastics / Materials",
    12: "Heating / HVAC",
    13: "Climate Services",
    14: "Plant-Based Food",
}
startup_name_to_tc = {n: int(km_labels[i]) for i, n in enumerate(startup_names_raw)}

# Novelty as uncertainty proxy: high novelty = high uncertainty
ns["co"]      = ns["name"].str.strip()
ns["tech_cluster"] = ns["co"].map(startup_name_to_tc)

# ── Build 2×2 type assignment for each company ────────────────────────────
print("Building 2×2 type assignments…")

match_rows = []
for co, investors in co_all_inv.items():
    startup_emb = startup_emb_dict.get(co)
    if startup_emb is None:
        continue
    inv_embs = [inv_emb_dict[i] for i in investors if i in inv_emb_dict]
    if not inv_embs:
        continue
    inv_matrix = np.array(inv_embs)
    sims = inv_matrix @ startup_emb
    match_rows.append({"co": co, "investor_startup_match": float(sims.mean())})

match_df = pd.DataFrame(match_rows)
med_match = match_df["investor_startup_match"].median()

df = (cdf
      .merge(match_df, on="co", how="left")
      .merge(co_sector, on="co", how="left")
      .merge(co_code, on="co", how="left"))
df["tech_cluster"]    = df["co"].map(startup_name_to_tc)
df["tech_cluster_lbl"] = df["tech_cluster"].map(tech_cluster_labels)

# 2×2 type
df["expertise_aligned"] = (df["investor_startup_match"] > med_match).astype(float)
df["network_diverse"]   = df["cross_community"].astype(float)
df["type"] = np.where(
    df["expertise_aligned"]==1,
    np.where(df["network_diverse"]==1, "A_ExDiv", "B_ExSam"),
    np.where(df["network_diverse"]==1, "C_NxDiv", "D_NxSam"),
)
df["type_A"] = (df["type"]=="A_ExDiv").astype(float)
df["type_B"] = (df["type"]=="B_ExSam").astype(float)
df["type_C"] = (df["type"]=="C_NxDiv").astype(float)

type_labels_map = {
    "A_ExDiv": "A: Expert+Diverse (best)",
    "B_ExSam": "B: Expert+Same (echo chamber)",
    "C_NxDiv": "C: Non-expert+Diverse",
    "D_NxSam": "D: Non-expert+Same (ref)",
}

sub = df.dropna(subset=["investor_startup_match", "is_failed"]).copy()
print(f"  Analysis sample N={len(sub):,}")
print(f"  Type distribution: A={sub['type_A'].sum():.0f}  B={sub['type_B'].sum():.0f}  "
      f"C={sub['type_C'].sum():.0f}  D={(sub['type']=='D_NxSam').sum():.0f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: 2×2 BY PITCHBOOK SECTOR
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 1: 2×2 FAILURE TABLE BY SECTOR")
print("="*72)

sector_counts = sub["sector"].value_counts()
major_sectors = sector_counts[sector_counts >= 150].index.tolist()
print(f"  Sectors with N≥150: {len(major_sectors)}")

echo_rows = []

print(f"\n  {'Sector':<30} {'N':>5}  {'A:ExpDiv':>9} {'B:ExpSam':>9} {'C:NxDiv':>9} {'D:NxSam':>9}  "
      f"{'Echo_Pen':>9} {'B-D':>8}")
print(f"  {'-'*100}")

for sector in major_sectors:
    s = sub[sub["sector"] == sector].dropna(subset=["is_failed"])
    n = len(s)
    if n < 100: continue

    fa = s[s["type"]=="A_ExDiv"]["is_failed"].mean()
    fb = s[s["type"]=="B_ExSam"]["is_failed"].mean()
    fc = s[s["type"]=="C_NxDiv"]["is_failed"].mean()
    fd = s[s["type"]=="D_NxSam"]["is_failed"].mean()
    echo_penalty = fb - fa     # how much worse is expert+same vs expert+diverse
    herd_penalty = fb - fd     # how much worse is expert+same vs non-expert+same

    na = (s["type"]=="A_ExDiv").sum()
    nb = (s["type"]=="B_ExSam").sum()
    nc = (s["type"]=="C_NxDiv").sum()
    nd = (s["type"]=="D_NxSam").sum()

    echo_rows.append({
        "sector": sector, "N": n,
        "fail_A": fa, "fail_B": fb, "fail_C": fc, "fail_D": fd,
        "echo_penalty": echo_penalty,
        "herd_penalty": herd_penalty,
        "n_A": na, "n_B": nb, "n_C": nc, "n_D": nd,
    })
    print(f"  {sector:<30} {n:>5}  {fa:>9.1%} {fb:>9.1%} {fc:>9.1%} {fd:>9.1%}  "
          f"{echo_penalty:>+9.1%} {herd_penalty:>+8.1%}")

echo_df = pd.DataFrame(echo_rows).sort_values("echo_penalty", ascending=False)
echo_df.to_csv(OUT / "echo_penalty_by_sector.csv", index=False)

print(f"\n  {'─'*55}")
print(f"  Echo chamber penalty (B−A) ranking:")
for _, row in echo_df.iterrows():
    marker = "🔴 HIGH" if row["echo_penalty"] > 0.05 else ("🟡 MOD" if row["echo_penalty"] > 0 else "🟢 NEG")
    print(f"    {row['sector']:<30} penalty={row['echo_penalty']:>+.1%}  {marker}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: REGRESSION BY SECTOR (top 5)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 2: REGRESSION BY SECTOR — β_B vs β_A (Reference=D)")
print("="*72)

top5_sectors = sector_counts[sector_counts >= 200].index.tolist()[:8]
print(f"\n  {'Sector':<30} {'N':>5}  {'β_A(E+Div)':>12} {'β_B(E+Sam)':>12} {'β_C(N+Div)':>12}  R²")
print(f"  {'-'*85}")

sector_reg_rows = []
for sector in top5_sectors:
    s = sub[sub["sector"]==sector].dropna(subset=["is_failed"]).copy()
    if len(s) < 80: continue
    # Vary controls: drop country_fe if near-constant
    ctrl = "log_cap + nrounds + specter_novelty + C(deal_year)"
    if s["country_fe"].nunique() > 2:
        ctrl = CTRL
    try:
        m = smf.ols(f"is_failed ~ type_A + type_B + type_C + {ctrl}", data=s).fit(cov_type="HC1")
        b_a = m.params.get("type_A", np.nan); p_a = m.pvalues.get("type_A", 1.0)
        b_b = m.params.get("type_B", np.nan); p_b = m.pvalues.get("type_B", 1.0)
        b_c = m.params.get("type_C", np.nan); p_c = m.pvalues.get("type_C", 1.0)
        r2  = m.rsquared
        sector_reg_rows.append({"sector": sector, "N": len(s),
                                 "b_A": b_a, "p_A": p_a,
                                 "b_B": b_b, "p_B": p_b,
                                 "b_C": b_c, "p_C": p_c, "r2": r2})
        print(f"  {sector:<30} {len(s):>5}  "
              f"{b_a:>+10.4f}{sig(p_a):>2}  "
              f"{b_b:>+10.4f}{sig(p_b):>2}  "
              f"{b_c:>+10.4f}{sig(p_c):>2}  {r2:.3f}")
    except Exception as e:
        print(f"  {sector:<30} ERR: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: INTERACTION MODEL — type_B × SECTOR
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 3: INTERACTION MODEL — Expert+Same × Sector")
print("="*72)

# Use sectors with N≥200, recode as dummies, interact with type_B
sub_int = sub[sub["sector"].isin(major_sectors)].copy()
# Reference sector: largest
ref_sector = sector_counts[sector_counts >= 150].index[0]
sub_int["sector_clean"] = sub_int["sector"].str.replace(r"[^a-zA-Z0-9]", "_", regex=True)
ref_clean = ref_sector.replace(" ", "_").replace(r"[^a-zA-Z0-9]", "_")

print(f"  Reference sector: {ref_sector}  (N={sector_counts[ref_sector]})")
print(f"  Sectors in interaction: {len(major_sectors)}")

formula_int = (f"is_failed ~ type_B * C(sector_clean, Treatment('{sub_int['sector_clean'].mode()[0]}')) "
               f"+ type_A + type_C + {CTRL}")
try:
    m_int = smf.ols(formula_int, data=sub_int.dropna(subset=["is_failed"])).fit(cov_type="HC1")
    print(f"\n  Interaction model: R²={m_int.rsquared:.4f}  N={int(m_int.nobs):,}")
    print(f"\n  Significant type_B × sector interactions (p<0.15):")
    int_terms = {k: v for k, v in m_int.params.items()
                 if "type_B" in k and "C(sector" in k}
    int_pvals = {k: m_int.pvalues[k] for k in int_terms}
    if not int_terms:
        print("  (No interaction terms found — check formula)")
    else:
        for term in sorted(int_terms, key=lambda k: m_int.pvalues[k]):
            p = m_int.pvalues[term]
            if p < 0.20:
                b = int_terms[term]
                # Extract sector name from term string
                sec = term.split("[T.")[-1].rstrip("]").replace("_", " ")
                print(f"    {sec:<35}: β={b:>+.4f}  {sig(p)}")
        # Also print the main type_B effect
        b_b_main = m_int.params.get("type_B", np.nan)
        p_b_main = m_int.pvalues.get("type_B", 1.0)
        print(f"\n  Main effect type_B (in reference sector '{ref_sector}'):")
        print(f"    β={b_b_main:>+.4f}  {sig(p_b_main)}")
except Exception as e:
    print(f"  Interaction ERR: {e}")
    # Simpler version: run interaction with top sectors only
    top3 = sector_counts[sector_counts>=300].index.tolist()[:4]
    sub_int3 = sub_int[sub_int["sector"].isin(top3)].copy()
    ref3 = top3[0]
    sub_int3["sector_c"] = sub_int3["sector"].str[:20].str.replace(r"\W","_",regex=True)
    ref3_c = ref3[:20].replace(" ","_")
    try:
        f3 = (f"is_failed ~ type_B * C(sector_c, Treatment('{sub_int3['sector_c'].mode()[0]}')) "
              f"+ type_A + type_C + log_cap + nrounds + specter_novelty + C(deal_year)")
        m3 = smf.ols(f3, data=sub_int3.dropna(subset=["is_failed"])).fit(cov_type="HC1")
        print(f"\n  Simplified (top 4 sectors):  R²={m3.rsquared:.4f}  N={int(m3.nobs):,}")
        for k, b in m3.params.items():
            if "type_B" in k:
                p = m3.pvalues[k]
                if p < 0.20:
                    print(f"    {k:<55}: β={b:>+.4f}  {sig(p)}")
    except Exception as e2:
        print(f"  Simplified ERR: {e2}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: 2×2 BY STARTUP TECH CLUSTER
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 4: 2×2 BY STARTUP TECHNOLOGY CLUSTER")
print("="*72)

# Average novelty per tech cluster (uncertainty proxy)
cluster_novelty = ns.groupby("tech_cluster")["specter_novelty"].mean().to_dict()

print(f"\n  {'Tech Cluster':<28} {'N':>5}  {'A:E+Div':>8} {'B:E+Sam':>8} "
      f"{'C:N+Div':>8} {'D:N+Sam':>8}  {'Echo':>7} {'Novelty':>8}")
print(f"  {'-'*97}")

tc_rows = []
tc_counts = sub["tech_cluster"].value_counts()
major_tc = tc_counts[tc_counts >= 80].index.tolist()

for tc in sorted(major_tc):
    s = sub[sub["tech_cluster"]==tc].dropna(subset=["is_failed"])
    n = len(s)
    if n < 50: continue

    fa = s[s["type"]=="A_ExDiv"]["is_failed"].mean() if (s["type"]=="A_ExDiv").sum() >= 5 else np.nan
    fb = s[s["type"]=="B_ExSam"]["is_failed"].mean() if (s["type"]=="B_ExSam").sum() >= 5 else np.nan
    fc = s[s["type"]=="C_NxDiv"]["is_failed"].mean() if (s["type"]=="C_NxDiv").sum() >= 5 else np.nan
    fd = s[s["type"]=="D_NxSam"]["is_failed"].mean() if (s["type"]=="D_NxSam").sum() >= 5 else np.nan

    echo = fb - fa if not np.isnan(fa) and not np.isnan(fb) else np.nan
    nov  = cluster_novelty.get(tc, np.nan)
    lbl  = tech_cluster_labels.get(tc, f"Cluster {tc}")

    tc_rows.append({
        "tech_cluster": tc, "label": lbl, "N": n,
        "fail_A": fa, "fail_B": fb, "fail_C": fc, "fail_D": fd,
        "echo_penalty": echo, "novelty": nov,
    })
    print(f"  {lbl:<28} {n:>5}  {str(f'{fa:.1%}') if not np.isnan(fa) else '  N/A ':>8} "
          f"{str(f'{fb:.1%}') if not np.isnan(fb) else '  N/A ':>8} "
          f"{str(f'{fc:.1%}') if not np.isnan(fc) else '  N/A ':>8} "
          f"{str(f'{fd:.1%}') if not np.isnan(fd) else '  N/A ':>8}  "
          f"{f'{echo:>+.1%}' if not np.isnan(echo) else '  N/A ':>7} "
          f"{nov:>8.3f}")

tc_df = pd.DataFrame(tc_rows).dropna(subset=["echo_penalty"])
tc_df = tc_df.sort_values("echo_penalty", ascending=False)
tc_df.to_csv(OUT / "echo_penalty_by_tech_cluster.csv", index=False)

# Novelty-echo correlation
pearson_r, pearson_p = None, 1.0
if len(tc_df) >= 5:
    pearson_r, pearson_p = stats.pearsonr(tc_df["novelty"].dropna(), tc_df["echo_penalty"].dropna())
    print(f"\n  Correlation (sector novelty × echo_penalty): r={pearson_r:.3f}  p={pearson_p:.3f}  {sig(pearson_p)}")
    print(f"  Hypothesis: HIGH novelty → LARGER echo penalty")
    if pearson_p < 0.10 and pearson_r > 0:
        print(f"  ✅ Confirmed: frontier technologies suffer most from groupthink")
    elif pearson_p < 0.10 and pearson_r < 0:
        print(f"  ❌ Reversed: frontier technologies benefit from expert consensus")
    else:
        print(f"  ❌ No significant novelty-echo correlation")

print(f"\n  Echo penalty ranking by tech cluster:")
print(f"  {'Tech cluster':<28} {'Echo penalty':>13} {'Novelty':>9} {'Verdict'}")
print(f"  {'-'*75}")
for _, row in tc_df.iterrows():
    if np.isnan(row["echo_penalty"]): continue
    verdict = "HIGH PENALTY" if row["echo_penalty"] > 0.05 else (
              "MOD PENALTY"  if row["echo_penalty"] > 0.01 else (
              "NEUTRAL"      if abs(row["echo_penalty"]) <= 0.01 else "NEGATIVE"))
    print(f"  {row['label']:<28} {row['echo_penalty']:>+13.1%} {row['novelty']:>9.3f}  {verdict}")

# ─── Regression: echo_penalty × novelty (sector-level) ───────────────────
print(f"\n  {'─'*65}")
print(f"  Regression: echo_penalty = f(novelty) across tech clusters")
if len(tc_df) >= 5:
    m_eco = smf.ols("echo_penalty ~ novelty", data=tc_df.dropna()).fit()
    b_nov = m_eco.params.get("novelty", np.nan)
    p_nov = m_eco.pvalues.get("novelty", 1.0)
    print(f"  β(novelty → echo_penalty) = {b_nov:>+.4f}  {sig(p_nov)}")
    print(f"  {'Positive = more novel sectors = larger echo chamber penalty' if b_nov>0 else 'Negative = more novel sectors = SMALLER echo chamber penalty'}")

# ─── Micro-regression within tech clusters ────────────────────────────────
print(f"\n  {'─'*65}")
print(f"  Micro-regression per tech cluster (top 8 by N):")
print(f"  {'Cluster':<28} {'N':>5}  {'β_B(E+Sam)':>12}  {'Sig':>5}")
print(f"  {'-'*60}")
top_tc = [row["tech_cluster"] for _, row in tc_df.head(8).iterrows()]
top_tc += [row["tech_cluster"] for _, row in tc_df.tail(4).iterrows() if row["tech_cluster"] not in top_tc]

for tc in sorted(sub["tech_cluster"].dropna().unique()):
    s = sub[sub["tech_cluster"]==tc].dropna(subset=["is_failed"])
    if len(s) < 60: continue
    ctrl = "log_cap + nrounds + specter_novelty + C(deal_year)"
    try:
        m = smf.ols(f"is_failed ~ type_A + type_B + type_C + {ctrl}", data=s).fit(cov_type="HC1")
        b = m.params.get("type_B", np.nan)
        p = m.pvalues.get("type_B", 1.0)
        lbl = tech_cluster_labels.get(int(tc), f"Cluster {tc}")
        print(f"  {lbl:<28} {len(s):>5}  {b:>+12.4f}  {sig(p):>5}")
    except: pass

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: NOVELTY × ECHO CHAMBER — FULL INDIVIDUAL-LEVEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("STEP 5: INDIVIDUAL-LEVEL NOVELTY × ECHO CHAMBER INTERACTION")
print("="*72)

sub5 = sub.dropna(subset=["specter_novelty", "is_failed"]).copy()
sub5["novelty_std"] = (sub5["specter_novelty"] - sub5["specter_novelty"].mean()) \
                       / sub5["specter_novelty"].std()
sub5["high_novelty"] = (sub5["specter_novelty"] > sub5["specter_novelty"].median()).astype(int)

# type_B × novelty interaction
print(f"\n  N={len(sub5):,}")
for label, formula in [
    ("M1: type_B × novelty_std",
     f"is_failed ~ type_A + type_B * novelty_std + type_C + {CTRL}"),
    ("M2: type_B × high_novelty",
     f"is_failed ~ type_A + type_B * high_novelty + type_C * high_novelty + {CTRL}"),
]:
    try:
        m = smf.ols(formula, data=sub5).fit(cov_type="HC1")
        print(f"\n  [{label}]  R²={m.rsquared:.4f}  N={int(m.nobs):,}")
        for var, lbl in [
            ("type_A",               "A: Expert+Diverse"),
            ("type_B",               "B: Expert+Same (echo chamber)"),
            ("type_C",               "C: Non-expert+Diverse"),
            ("novelty_std",          "novelty(std)"),
            ("high_novelty",         "high_novelty"),
            ("type_B:novelty_std",   "B × novelty (KEY INTERACTION)"),
            ("type_B:high_novelty",  "B × high_novelty (KEY)"),
            ("type_C:high_novelty",  "C × high_novelty"),
        ]:
            if var in m.params:
                b, p = m.params[var], m.pvalues[var]
                flag = " ←" if "KEY" in lbl and p < 0.15 else ""
                print(f"    {lbl:<35}: β={b:>+.4f}  {sig(p)}{flag}")
    except Exception as e:
        print(f"  [{label}] ERR: {e}")

# Failure by type × novelty
print(f"\n  Failure rates by type × novelty quartile:")
sub5["nov_q"] = pd.qcut(sub5["specter_novelty"], q=4, labels=["Low","Mid-low","Mid-high","High"])
print(f"  {'Type':<28} {'Low':>8} {'Mid-low':>9} {'Mid-high':>10} {'High':>8}")
print(f"  {'-'*65}")
for t, lbl in [("A_ExDiv","A: Expert+Diverse"), ("B_ExSam","B: Expert+Same"),
               ("C_NxDiv","C: Non-exp+Diverse"), ("D_NxSam","D: Non-exp+Same")]:
    row_vals = []
    for q in ["Low","Mid-low","Mid-high","High"]:
        s = sub5[(sub5["type"]==t) & (sub5["nov_q"]==q)]["is_failed"]
        row_vals.append(f"{s.mean():.1%}" if len(s)>=5 else " N/A ")
    print(f"  {lbl:<28} " + "  ".join(f"{v:>8}" for v in row_vals))

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("SUMMARY: WHERE IS GROUPTHINK MOST DANGEROUS?")
print("="*72)

# Echo penalty summary
top_echo = echo_df.head(3)
bot_echo = echo_df.tail(3)
print(f"""
  SECTOR ECHO PENALTY RANKING
  ─────────────────────────────────────────────────────────────
  Most dangerous (highest echo penalty = B much worse than A):""")
for _, r in top_echo.iterrows():
    print(f"    {r['sector']:<35}: echo_penalty={r['echo_penalty']:>+.1%}")
print(f"\n  Least dangerous (smallest or negative echo penalty):")
for _, r in bot_echo.iterrows():
    print(f"    {r['sector']:<35}: echo_penalty={r['echo_penalty']:>+.1%}")

print(f"""
  TECH CLUSTER ECHO PENALTY RANKING
  ─────────────────────────────────────────────────────────────
  Most dangerous:""")
for _, r in tc_df.head(4).iterrows():
    if np.isnan(r["echo_penalty"]): continue
    print(f"    {r['label']:<28}: echo_penalty={r['echo_penalty']:>+.1%}  "
          f"novelty={r['novelty']:.3f}")
print(f"\n  Least dangerous:")
for _, r in tc_df.tail(4).iterrows():
    if np.isnan(r["echo_penalty"]): continue
    print(f"    {r['label']:<28}: echo_penalty={r['echo_penalty']:>+.1%}  "
          f"novelty={r['novelty']:.3f}")

b_int = None
try:
    b_int = m.params.get("type_B:novelty_std", None) or m.params.get("type_B:high_novelty", None)
    p_int = m.pvalues.get("type_B:novelty_std", 1.0) if "type_B:novelty_std" in m.pvalues else \
            m.pvalues.get("type_B:high_novelty", 1.0)
except: p_int = 1.0

print(f"""
  NOVELTY × ECHO CHAMBER (INDIVIDUAL LEVEL)
  ─────────────────────────────────────────────────────────────
  Interaction type_B × novelty: {'see models above' if b_int is None else f'β={b_int:+.4f}  {sig(p_int)}'}

  HYPOTHESIS VERDICT:
  {'✅ CONFIRMED: Echo chamber is strongest in high-novelty / frontier sectors.' if (pearson_r is not None and pearson_r>0 and pearson_p<0.10) else
   '❌ NOT CONFIRMED: Echo chamber penalty does not scale with novelty/uncertainty.' if (pearson_r is not None and pearson_p>=0.10) else
   '→ Inconclusive: insufficient tech-cluster variation in echo penalty.'}

  OUTPUT FILES:
    {OUT}/echo_penalty_by_sector.csv
    {OUT}/echo_penalty_by_tech_cluster.csv
""")
