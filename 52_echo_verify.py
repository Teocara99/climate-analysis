"""
Script 52: Statistical Verification of Echo Chamber Effect
==========================================================
A1: Cell sizes (overall + by sector)
A2: Regression with Wald tests
A3: Sector interaction (market-risk vs technical-risk)
"""

import warnings; warnings.filterwarnings("ignore")
import sys, pickle
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.contrast import ContrastResults
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ── Load everything ───────────────────────────────────────────────────────
print("Loading data…")

cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]         = cdf["Companies"].str.strip()
cdf["is_failed"]  = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

net = pd.read_csv("output/network/network_metrics.csv")
inv_to_comm = net.set_index("investor")["community"].to_dict()

inv_emb_raw  = np.load("data/investor_embeddings.npy")
inv_names    = pd.read_csv("data/investor_names.csv")["investor"].tolist()
inv_emb_norm = normalize(inv_emb_raw, norm="l2")
inv_emb_dict = {n: inv_emb_norm[i] for i, n in enumerate(inv_names)}

with open("output/novelty/novelty_specter.pkl", "rb") as f:
    sp = pickle.load(f)
startup_names_raw = [n.strip() for n in sp["names"]]
startup_emb_norm  = normalize(sp["embs_raw"], norm="l2")
startup_emb_dict  = {n: startup_emb_norm[i] for i, n in enumerate(startup_names_raw)}

deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv").assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""').dropna(subset=["Year"]))
co_all_inv = d_exp.groupby("co")["inv"].apply(set).to_dict()

co_sector = (deals.dropna(subset=["Primary PitchBook Industry Group"])
             .groupby("co")["Primary PitchBook Industry Group"]
             .agg(lambda x: x.mode()[0]).reset_index()
             .rename(columns={"Primary PitchBook Industry Group": "sector"}))

# Tech clusters
print("Fitting K-means K=15 on startup embeddings…")
km = KMeans(n_clusters=15, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(startup_emb_norm)
startup_name_to_tc = {n: int(km_labels[i]) for i, n in enumerate(startup_names_raw)}
tc_labels = {0:"Carbon/Hydrogen",1:"Solar/Renewable",2:"Data/Monitoring",
             3:"AgTech",4:"EV/Mobility",5:"Grid/Energy Mgmt",6:"Carbon Data",
             7:"Wind Power",8:"Battery/Storage",9:"Waste/Water",
             10:"Alt Protein",11:"Plastics/Materials",12:"Heating/HVAC",
             13:"Climate Services",14:"Plant-Based Food"}

# ── Build investor-startup cosine match ───────────────────────────────────
print("Computing investor-startup cosine match…")
match_rows = []
for co, investors in co_all_inv.items():
    emb = startup_emb_dict.get(co)
    if emb is None: continue
    inv_embs = [inv_emb_dict[i] for i in investors if i in inv_emb_dict]
    if not inv_embs: continue
    sims = np.array(inv_embs) @ emb
    match_rows.append({"co": co, "investor_startup_match": float(sims.mean())})

match_df = pd.DataFrame(match_rows)
med_match = match_df["investor_startup_match"].median()
print(f"  Median investor-startup match: {med_match:.4f}")

# ── Assemble analysis frame ───────────────────────────────────────────────
df = (cdf
      .merge(match_df, on="co", how="left")
      .merge(co_sector, on="co", how="left"))
df["tech_cluster"]     = df["co"].map(startup_name_to_tc)
df["expertise_aligned"] = (df["investor_startup_match"] > med_match).astype(float)
df["network_diverse"]   = df["cross_community"].astype(float)
df["type"] = np.where(
    df["expertise_aligned"]==1,
    np.where(df["network_diverse"]==1, "A", "B"),
    np.where(df["network_diverse"]==1, "C", "D"),
)
df["type_A"] = (df["type"]=="A").astype(float)
df["type_B"] = (df["type"]=="B").astype(float)
df["type_C"] = (df["type"]=="C").astype(float)
# Reference = D (Non-expert + Same network)

sub = df.dropna(subset=["investor_startup_match","is_failed"]).copy()
print(f"  Full analysis sample: N={len(sub):,}")

# ═══════════════════════════════════════════════════════════════════════════
# A1: CELL SIZES
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A1: CELL SIZES")
print("="*70)

def cell_table(s, label="Overall"):
    counts = s["type"].value_counts()
    fails  = s.groupby("type")["is_failed"].agg(["sum","mean","count"])
    print(f"\n  [{label}]  N={len(s):,}")
    print(f"  {'Cell':<35} {'N':>6} {'Fail N':>7} {'Fail %':>8}  {'Flag'}")
    print(f"  {'-'*70}")
    for t, lbl in [("A","A: Expert+Diverse  ←← BEST"),
                   ("B","B: Expert+Same   ← ECHO CHAMBER"),
                   ("C","C: Non-expert+Diverse"),
                   ("D","D: Non-expert+Same   ← REFERENCE")]:
        if t not in fails.index:
            print(f"  {lbl:<35}   MISSING")
            continue
        n = int(fails.loc[t,"count"]); fn = int(fails.loc[t,"sum"]); fp = fails.loc[t,"mean"]
        flag = " ⚠ FRAGILE (N<50)" if n < 50 else ""
        print(f"  {lbl:<35} {n:>6} {fn:>7} {fp:>8.1%}{flag}")

cell_table(sub)

# Subset sectors
for label, filt in [
    ("Agriculture",         sub["sector"] == "Agriculture"),
    ("Grid/Energy Mgmt",    sub["tech_cluster"] == 5),
    ("Plant-Based Food",    sub["tech_cluster"] == 14),
    ("Battery/Storage",     sub["tech_cluster"] == 8),
    ("Software",            sub["sector"] == "Software"),
    ("Energy Equipment",    sub["sector"] == "Energy Equipment"),
]:
    cell_table(sub[filt], label)

# ═══════════════════════════════════════════════════════════════════════════
# A2: REGRESSION + WALD TESTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A2: REGRESSION + WALD TESTS  (Reference = D: Non-expert + Same)")
print("="*70)

m = smf.ols(f"is_failed ~ type_A + type_B + type_C + {CTRL}", data=sub).fit(cov_type="HC1")

print(f"\n  Model: N={int(m.nobs):,}  R²={m.rsquared:.4f}  AdjR²={m.rsquared_adj:.4f}")
print(f"\n  {'Variable':<40} {'β':>9}  {'SE':>7}  {'t':>6}  {'p':>7}  {'Sig'}")
print(f"  {'-'*80}")
for var, lbl in [
    ("type_A", "A: Expert+Diverse"),
    ("type_B", "B: Expert+Same (echo chamber)"),
    ("type_C", "C: Non-expert+Diverse"),
    ("log_cap", "log_capital"),
    ("nrounds", "log(1+n_rounds)"),
    ("specter_novelty", "specter_novelty"),
]:
    if var in m.params:
        b = m.params[var]; se = m.bse[var]; t = m.tvalues[var]; p = m.pvalues[var]
        print(f"  {lbl:<40} {b:>+9.4f}  {se:>7.4f}  {t:>6.2f}  {p:>7.4f}  {sig(p)}")

# --- Wald tests ---
print(f"\n  WALD TESTS:")
print(f"  {'-'*70}")

def wald(m, term1, term2):
    try:
        r = m.f_test(f"{term1} = {term2}")
        return float(r.statistic), float(r.pvalue)
    except Exception as e:
        return np.nan, np.nan

# Wald 1: β_B ≠ β_A  → echo chamber vs expert-diverse
f1, p1 = wald(m, "type_B", "type_A")
print(f"\n  Wald 1: β_B = β_A?  (Expert-Same vs Expert-Diverse)")
print(f"    H0: being in same-network SAME as diverse-network given expertise")
if not np.isnan(f1):
    print(f"    F={f1:.3f}  p={p1:.4f}  {sig(p1)}")
    delta = m.params.get("type_B",0) - m.params.get("type_A",0)
    print(f"    Δ = β_B − β_A = {delta:>+.4f}")
    if p1 < 0.05:
        print(f"    ✅ ECHO CHAMBER CONFIRMED: Expert+Same significantly worse than Expert+Diverse")
    else:
        print(f"    ❌ Not significant: B and A not distinguishable at p<0.05")

# Wald 2: β_B ≠ β_D (D is reference = 0)
# β_D = 0 by construction; test whether β_B ≠ 0
b_b = m.params.get("type_B", np.nan); p_b = m.pvalues.get("type_B", 1.0)
print(f"\n  Wald 2: β_B = β_D (=0)?  (Expert-Same vs Non-expert-Same)")
print(f"    H0: Expert syndicate in same network = non-expert in same network")
print(f"    t-test β_B: β={b_b:>+.4f}  p={p_b:.4f}  {sig(p_b)}")
if p_b < 0.05:
    if b_b > 0:
        print(f"    ✅ DOMAIN EXPERTISE HURTS within same-network context")
    else:
        print(f"    → Expert-same significantly BETTER than non-expert-same")
else:
    print(f"    ❌ Not significant: expert-same and non-expert-same indistinguishable")

# Wald 3: β_A ≠ β_C  → expertise irrelevant when diverse?
f3, p3 = wald(m, "type_A", "type_C")
print(f"\n  Wald 3: β_A = β_C?  (Expert-Diverse vs Non-expert-Diverse)")
print(f"    H0: Expertise irrelevant when network is diverse")
if not np.isnan(f3):
    print(f"    F={f3:.3f}  p={p3:.4f}  {sig(p3)}")
    delta3 = m.params.get("type_A",0) - m.params.get("type_C",0)
    print(f"    Δ = β_A − β_C = {delta3:>+.4f}")
    if p3 >= 0.05:
        print(f"    ✅ EXPERTISE IRRELEVANT when network is diverse (A ≈ C)")
    else:
        print(f"    → A ≠ C: expertise does matter even with diverse network")

# Summary interpretation
print(f"\n  INTERPRETATION MATRIX:")
print(f"  ─────────────────────────────────────────────────────────────")
ba = m.params.get("type_A", np.nan); bb = m.params.get("type_B", np.nan)
bc = m.params.get("type_C", np.nan)
ref_fail = sub[sub["type"]=="D"]["is_failed"].mean()
print(f"  Baseline (D: Non-expert+Same):     fail rate = {ref_fail:.1%}")
print(f"  A: Expert+Diverse:    β={ba:>+.4f}  implied fail = {ref_fail+ba:.1%}")
print(f"  B: Expert+Same:       β={bb:>+.4f}  implied fail = {ref_fail+bb:.1%}")
print(f"  C: Non-expert+Diverse:β={bc:>+.4f}  implied fail = {ref_fail+bc:.1%}")
print(f"  D: Non-expert+Same:   β= 0.0000  implied fail = {ref_fail:.1%}")

# ═══════════════════════════════════════════════════════════════════════════
# A3: SECTOR INTERACTION — market-risk vs technical-risk
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("A3: SECTOR INTERACTION — Market-risk vs Technical-risk")
print("="*70)

# Map by PitchBook sector
market_risk_sectors = {"Agriculture", "Consumer Non-Durables", "Software"}
tech_risk_sectors   = {"Computer Hardware", "Energy Equipment"}

# Map by tech cluster
market_risk_tc  = {3, 4, 5, 14, 10}   # AgTech, EV, Grid, Plant-Based, Alt Protein
tech_risk_tc    = {8, 9, 2, 7}         # Battery, Waste, Data, Wind

sub3 = sub.reset_index(drop=True).copy()
sub3["is_market_risk"] = (
    sub3["sector"].isin(market_risk_sectors) |
    sub3["tech_cluster"].isin(market_risk_tc)
).astype(int)
sub3["is_tech_risk"] = (
    sub3["sector"].isin(tech_risk_sectors) |
    sub3["tech_cluster"].isin(tech_risk_tc)
).astype(int)
# Clear overlap: if classified as both, keep tech_risk
overlap_mask = (sub3["is_market_risk"] == 1) & (sub3["is_tech_risk"] == 1)
sub3.loc[overlap_mask, "is_market_risk"] = 0

print(f"\n  Market-risk sector companies: {sub3['is_market_risk'].sum():,}")
print(f"  Tech-risk sector companies:   {sub3['is_tech_risk'].sum():,}")
print(f"  Unclassified:                 {(~sub3['is_market_risk'].astype(bool) & ~sub3['is_tech_risk'].astype(bool)).sum():,}")

# Model 1: type_B × market_risk
formula_int = (f"is_failed ~ type_B * is_market_risk + type_A + type_C + {CTRL}")
m_int = smf.ols(formula_int, data=sub3.dropna(subset=["is_failed"])).fit(cov_type="HC1")
print(f"\n  Interaction model: type_B × is_market_risk")
print(f"  N={int(m_int.nobs):,}  R²={m_int.rsquared:.4f}")
print(f"\n  {'Variable':<45} {'β':>9}  {'SE':>7}  {'p':>7}  {'Sig'}")
print(f"  {'-'*75}")
for var, lbl in [
    ("type_A", "A: Expert+Diverse"),
    ("type_B", "B: Expert+Same"),
    ("type_C", "C: Non-expert+Diverse"),
    ("is_market_risk", "is_market_risk"),
    ("type_B:is_market_risk", "B × market_risk   ← KEY INTERACTION"),
]:
    if var in m_int.params:
        b = m_int.params[var]; se = m_int.bse[var]; p = m_int.pvalues[var]
        print(f"  {lbl:<45} {b:>+9.4f}  {se:>7.4f}  {p:>7.4f}  {sig(p)}")

b_int = m_int.params.get("type_B:is_market_risk", np.nan)
p_int = m_int.pvalues.get("type_B:is_market_risk", 1.0)
b_b0  = m_int.params.get("type_B", np.nan)
print(f"\n  Echo penalty in TECH-RISK sector:    β_B = {b_b0:>+.4f}  {sig(m_int.pvalues.get('type_B',1.0))}")
print(f"  Extra echo penalty in MARKET-RISK:   β_int = {b_int:>+.4f}  {sig(p_int)}")
if not np.isnan(b_b0) and not np.isnan(b_int):
    print(f"  Echo penalty in MARKET-RISK sector:  β_B + β_int = {b_b0+b_int:>+.4f}")

if p_int < 0.05:
    print(f"\n  ✅ INTERACTION SIGNIFICANT: Echo chamber is STRONGER in market-risk sectors")
elif p_int < 0.10:
    print(f"\n  † INTERACTION MARGINAL (p={p_int:.3f}): Some evidence echo stronger in market-risk")
else:
    print(f"\n  ❌ INTERACTION NOT SIGNIFICANT (p={p_int:.3f}): Echo penalty similar across sector types")

# Simple failure rates by sector type × echo
print(f"\n  Failure rates by sector type × 2×2 type:")
print(f"  {'Group':<40} {'A:E+Div':>9} {'B:E+Sam':>9} {'D:N+Sam':>9} {'Echo':>8}")
print(f"  {'-'*77}")
for grp_name, grp_mask in [
    ("Market-risk sectors", sub3["is_market_risk"]==1),
    ("Tech-risk sectors",   sub3["is_tech_risk"]==1),
    ("Other / unclassified",(sub3["is_market_risk"]==0)&(sub3["is_tech_risk"]==0)),
]:
    g = sub3[grp_mask].dropna(subset=["is_failed"])
    fa = g[g["type"]=="A"]["is_failed"].mean()
    fb = g[g["type"]=="B"]["is_failed"].mean()
    fd = g[g["type"]=="D"]["is_failed"].mean()
    na = (g["type"]=="A").sum(); nb = (g["type"]=="B").sum(); nd = (g["type"]=="D").sum()
    print(f"  {grp_name:<40} {fa:>8.1%}  {fb:>8.1%}  {fd:>8.1%}  {fb-fa:>+8.1%}")
    print(f"  {'':40} ({na:>4}N)    ({nb:>4}N)    ({nd:>4}N)")

# Model 2: also include type_C × market_risk
print(f"\n  Full 4-type × market_risk interaction:")
formula_full = (f"is_failed ~ (type_A + type_B + type_C) * is_market_risk + {CTRL}")
m_full = smf.ols(formula_full, data=sub3.dropna(subset=["is_failed"])).fit(cov_type="HC1")
print(f"  N={int(m_full.nobs):,}  R²={m_full.rsquared:.4f}")
for var in ["type_A:is_market_risk","type_B:is_market_risk","type_C:is_market_risk"]:
    if var in m_full.params:
        b = m_full.params[var]; p = m_full.pvalues[var]
        print(f"  {var:<40}: β={b:>+.4f}  {sig(p)}")

print(f"\n  ── F-test: all three interactions jointly zero? ──")
try:
    joint = m_full.f_test(["type_A:is_market_risk = 0",
                            "type_B:is_market_risk = 0",
                            "type_C:is_market_risk = 0"])
    print(f"  F({joint.df_num:.0f},{joint.df_denom:.0f}) = {float(joint.statistic):.3f}  p={float(joint.pvalue):.4f}  {sig(float(joint.pvalue))}")
except Exception as e:
    print(f"  F-test ERR: {e}")
