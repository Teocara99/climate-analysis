"""V5 only: Echo chamber with PitchBook-based expert definition."""
import warnings; warnings.filterwarnings("ignore")
import sys, re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

print("Loading…")
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
cdf["co"]         = cdf["Companies"].str.strip()
cdf["is_failed"]  = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x >= 3].index), "Other")

deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv").assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))
co_all_inv = d_exp.groupby("co")["inv"].apply(set).to_dict()

co_sector = (deals.dropna(subset=["Primary PitchBook Industry Group"])
             .groupby("co")["Primary PitchBook Industry Group"]
             .agg(lambda x: x.mode()[0]).reset_index()
             .rename(columns={"Primary PitchBook Industry Group": "sector"}))
co_verticals = (deals.dropna(subset=["Verticals"])
                .groupby("co")["Verticals"]
                .agg(lambda x: x.mode()[0]).reset_index())

print("Loading investor preferred verticals (just 3 columns)…")
inv_data = pd.read_excel("data/Investors_data.xlsx", skiprows=7,
                         usecols=["Investors", "Preferred Verticals", "Preferred PitchBook Industry"])
inv_data["investor"] = inv_data["Investors"].str.strip()
inv_data = inv_data.drop_duplicates(subset=["investor"]).set_index("investor")
print(f"  Records: {len(inv_data):,}  PrefVert non-null: {inv_data['Preferred Verticals'].notna().sum():,}")

def parse_set(s):
    if not isinstance(s, str): return set()
    return {x.strip().lower() for x in re.split(r"[,;|]", s) if x.strip()}

inv_pref_vert = {n: parse_set(v) for n, v in inv_data["Preferred Verticals"].dropna().items()}
inv_pref_ind  = {n: parse_set(v) for n, v in inv_data["Preferred PitchBook Industry"].dropna().items()}

co_vert_dict = co_verticals.set_index("co")["Verticals"].to_dict()
co_sec_dict  = co_sector.set_index("co")["sector"].to_dict()

print("Building PB-expert classification per company…")
pb_expert_rows = []
for co, investors in co_all_inv.items():
    co_vert    = parse_set(co_vert_dict.get(co, ""))
    co_sec_set = {co_sec_dict.get(co, "").lower().strip()}
    n_with_pref = 0; n_pb_expert = 0
    for inv in investors:
        pv = inv_pref_vert.get(inv, set())
        pi = inv_pref_ind.get(inv, set())
        if pv or pi:
            n_with_pref += 1
            if (co_vert & pv) or (co_sec_set & pi):
                n_pb_expert += 1
    pb_expert_rows.append({
        "co": co,
        "n_investors": len(investors),
        "n_with_pref_data": n_with_pref,
        "n_pb_expert": n_pb_expert,
        "has_pb_expert": int(n_pb_expert > 0),
    })

pb_df = pd.DataFrame(pb_expert_rows)
print(f"  Companies: {len(pb_df):,}")
print(f"  With ≥1 pref-data investor: {(pb_df['n_with_pref_data']>0).sum():,}")
print(f"  With ≥1 PB-expert investor: {pb_df['has_pb_expert'].sum():,}")

pb_full = (cdf.merge(pb_df, on="co", how="inner")
           .query("n_with_pref_data > 0")
           .dropna(subset=["is_failed","cross_community"]))

pb_full["pb_expert"]       = pb_full["has_pb_expert"].astype(float)
pb_full["network_diverse"] = pb_full["cross_community"].astype(float)
pb_full["type_pb"] = np.where(
    pb_full["pb_expert"] == 1,
    np.where(pb_full["network_diverse"] == 1, "A2", "B2"),
    np.where(pb_full["network_diverse"] == 1, "C2", "D2"),
)
pb_full["type_A2"] = (pb_full["type_pb"] == "A2").astype(float)
pb_full["type_B2"] = (pb_full["type_pb"] == "B2").astype(float)
pb_full["type_C2"] = (pb_full["type_pb"] == "C2").astype(float)

print("\n" + "="*70)
print("VALIDATION 5: ECHO CHAMBER WITH PITCHBOOK-BASED EXPERT DEFINITION")
print("="*70)
print(f"\n  N={len(pb_full):,}  (companies with ≥1 investor with PB preference data)")
print(f"\n  2×2 CELL SIZES AND FAILURE RATES:")
print(f"  {'Cell':<38} {'N':>6} {'Fail N':>7} {'Fail%':>7}  Flag")
print(f"  {'-'*68}")
labels = {"A2":"A2: PB-Expert+Diverse",
          "B2":"B2: PB-Expert+Same  ← echo?",
          "C2":"C2: Non-expert+Diverse",
          "D2":"D2: Non-expert+Same  ← ref"}
for t, lbl in labels.items():
    s = pb_full[pb_full["type_pb"] == t]
    n = len(s); fn = int(s["is_failed"].sum()) if n > 0 else 0
    fp = s["is_failed"].mean() if n > 0 else np.nan
    flag = " ⚠ FRAGILE" if n < 50 else ""
    print(f"  {lbl:<38} {n:>6} {fn:>7} {fp:>7.1%}{flag}")

fa2 = pb_full[pb_full["type_pb"]=="A2"]["is_failed"].mean()
fb2 = pb_full[pb_full["type_pb"]=="B2"]["is_failed"].mean()
fc2 = pb_full[pb_full["type_pb"]=="C2"]["is_failed"].mean()
fd2 = pb_full[pb_full["type_pb"]=="D2"]["is_failed"].mean()
print(f"\n  Echo penalty B2−A2 = {fb2-fa2:>+.1%}")
print(f"  Ordering: A2={fa2:.1%}  B2={fb2:.1%}  C2={fc2:.1%}  D2={fd2:.1%}")

# Regression
m = smf.ols(f"is_failed ~ type_A2 + type_B2 + type_C2 + {CTRL}",
            data=pb_full).fit(cov_type="HC1")
print(f"\n  Regression: N={int(m.nobs):,}  R²={m.rsquared:.4f}")
for var, lbl in [("type_A2","A2: PB-Expert+Diverse"),
                 ("type_B2","B2: PB-Expert+Same"),
                 ("type_C2","C2: Non-expert+Diverse")]:
    if var in m.params:
        b = m.params[var]; p = m.pvalues[var]
        print(f"    {lbl:<30}: β={b:>+.4f}  {sig(p)}")

w = m.f_test("type_B2 = type_A2")
f_w, p_w = float(w.statistic), float(w.pvalue)
delta = m.params.get("type_B2", 0) - m.params.get("type_A2", 0)
print(f"\n  Wald (B2=A2): F={f_w:.3f}  p={p_w:.4f}  {sig(p_w)}  Δ={delta:>+.4f}")

if p_w < 0.05:
    print(f"  ✅ ECHO CHAMBER REPLICATED with PitchBook expert definition")
elif p_w < 0.10:
    print(f"  † MARGINAL: p={p_w:.3f} — suggestive but not confirmed")
else:
    print(f"  ❌ NOT replicated (p={p_w:.3f}) with PitchBook expert definition")
    if fb2 > fa2:
        print(f"  (Direction is correct: B2 > A2, just not significant at p<0.05)")

# Compare with embedding-based 2×2 on same sample
print(f"\n  NOTE: 79.6% of companies have ≥1 PB-expert investor (base rate very high)")
print(f"  This inflates cell A2+B2 and deflates C2+D2 — power asymmetry.")
