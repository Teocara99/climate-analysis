"""
Script 46: Green investor persistence with Haiku classifications
================================================================

Tests whether GREEN_VC investors (now 8.1% of investors, vs 57.8% Mistral)
still show higher persistence rates (following on across rounds).

1. Persistence rates by Haiku classification + chi-square
2. Down-round persistence (VC Round Up/Down/Flat == 'Down Round')
3. Bridge type analysis with Haiku green labels
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

OUT = Path("output")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data…")
deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

haiku   = pd.read_csv(OUT / "green_classify/investor_green_classifications.csv")
mistral = pd.read_csv(OUT / "investors_classified_output.csv")
haiku["investor"]          = haiku["investor"].str.strip()
mistral["investor_name"]   = mistral["investor_name"].str.strip()

# Merged label: Haiku preferred, Mistral fallback
merged_labels = {}
for _, r in haiku.iterrows():
    merged_labels[r["investor"]] = r["green_focus_haiku"]
for _, r in mistral.iterrows():
    name = r["investor_name"]
    if name not in merged_labels:
        gf = r["green_focus"] if r["green_focus"] in ("GREEN_VC","ESG_ALIGNED","TRADITIONAL") else "TRADITIONAL"
        merged_labels[name] = gf

# Old Mistral labels (for comparison)
mistral_labels = {r["investor_name"]: r["green_focus"] for _, r in mistral.iterrows()}

# ── Build investor × round panel ─────────────────────────────────────────────
print("Building investor × round panel…")

# Restrict to VC-type deals only (exclude grants, accelerators, debt)
VC_TYPES = {"Early Stage VC","Later Stage VC","Seed Round","Angel (individual)",
            "PE Growth/Expansion","Growth/Expansion","PIPE"}
deals_vc = deals[deals["Deal Type"].isin(VC_TYPES)].copy()
deals_vc["Deal Size (USD M)"] = pd.to_numeric(deals_vc["Deal Size"], errors="coerce")

# Explode by investor
d_exp = (deals_vc.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""')
         .dropna(subset=["Deal Date"]))

d_exp["haiku_label"]   = d_exp["inv"].map(merged_labels).fillna("UNKNOWN")
d_exp["mistral_label"] = d_exp["inv"].map(lambda n: mistral_labels.get(n, "UNKNOWN"))
d_exp["Year"] = d_exp["Deal Date"].dt.year

# Round number: use VC Round if available, else rank by date within company
d_exp["vc_round_raw"] = d_exp["VC Round"].str.extract(r"(\d+)").astype(float)
d_exp["round_order"] = d_exp.groupby("co")["Deal Date"].rank(method="dense")

# Down round flag
d_exp["is_down_round"] = (d_exp["VC Round Up/Down/Flat"] == "Down Round").astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PERSISTENCE RATES BY HAIKU CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("1. PERSISTENCE RATES BY HAIKU CLASSIFICATION")
print("="*72)
print("""
  Definition: an investor 'persists' if they invest in both round N and
  round N+1 for the SAME company. Persistence rate = fraction of
  round-N appearances that also appear in round N+1.
""")

# For each company, get sorted deal dates and investor sets per deal
co_rounds = {}
for co, grp in d_exp.groupby("co"):
    dates = sorted(grp["Deal Date"].unique())
    if len(dates) < 2:
        continue
    rounds = []
    for i, dt in enumerate(dates):
        inv_set = set(grp[grp["Deal Date"]==dt]["inv"].tolist())
        rounds.append({"date": dt, "investors": inv_set, "round_n": i+1})
    co_rounds[co] = rounds

# For each investor appearance in round N, check if they appear in round N+1
persist_records = []
for co, rounds in co_rounds.items():
    for i in range(len(rounds)-1):
        r_now  = rounds[i]
        r_next = rounds[i+1]
        for inv in r_now["investors"]:
            label = merged_labels.get(inv, "UNKNOWN")
            m_label = mistral_labels.get(inv, "UNKNOWN")
            persisted = int(inv in r_next["investors"])
            persist_records.append({
                "co": co, "investor": inv,
                "round_n": r_now["round_n"],
                "haiku_label": label,
                "mistral_label": m_label,
                "persisted": persisted,
            })

pr = pd.DataFrame(persist_records)
pr = pr[pr["haiku_label"] != "UNKNOWN"]

print(f"  Investor × round observations: {len(pr):,}")
print(f"  Unique investors: {pr['investor'].nunique():,}")
print()

# Overall persistence rate
overall = pr["persisted"].mean()
print(f"  Overall persistence rate: {overall:.1%}")
print()

# Haiku labels
print("  HAIKU CLASSIFICATIONS:")
print(f"  {'Label':<15}  {'N obs':>8}  {'Persist%':>9}  {'95% CI':>16}")
print(f"  {'-'*52}")
haiku_persist = {}
for label in ["GREEN_VC","ESG_ALIGNED","TRADITIONAL"]:
    sub = pr[pr["haiku_label"]==label]
    n   = len(sub)
    p   = sub["persisted"].mean()
    se  = np.sqrt(p*(1-p)/n)
    ci  = f"[{p-1.96*se:.1%}, {p+1.96*se:.1%}]"
    haiku_persist[label] = (n, p)
    print(f"  {label:<15}  {n:>8,}  {p:>9.1%}  {ci:>16}")

# Chi-square across 3 groups
ct = pd.crosstab(pr["haiku_label"], pr["persisted"])
ct = ct.reindex(["GREEN_VC","ESG_ALIGNED","TRADITIONAL"])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
print(f"\n  Chi-square across 3 groups: χ²={chi2:.2f}  df={dof}  p={p_chi:.4f} {sig(p_chi)}")

# Pairwise comparisons
for a, b in [("GREEN_VC","TRADITIONAL"),("ESG_ALIGNED","TRADITIONAL"),("GREEN_VC","ESG_ALIGNED")]:
    na, pa = haiku_persist[a]
    nb, pb = haiku_persist[b]
    ct2 = np.array([[pa*na, (1-pa)*na],[pb*nb, (1-pb)*nb]])
    chi2_2, p2, _, _ = stats.chi2_contingency(ct2)
    print(f"  {a} vs {b}: diff={pa-pb:+.1%}  p={p2:.4f} {sig(p2)}")

print()
print("  COMPARISON — OLD MISTRAL LABELS:")
print(f"  {'Label':<15}  {'N obs':>8}  {'Persist%':>9}")
print(f"  {'-'*38}")
for label in ["GREEN_VC","ESG_ALIGNED","TRADITIONAL","PARSE_ERROR"]:
    sub = pr[pr["mistral_label"]==label]
    if len(sub) == 0: continue
    print(f"  {label:<15}  {len(sub):>8,}  {sub['persisted'].mean():>9.1%}")
ct_m = pd.crosstab(pr[pr["mistral_label"].isin(["GREEN_VC","TRADITIONAL"])]["mistral_label"],
                   pr[pr["mistral_label"].isin(["GREEN_VC","TRADITIONAL"])]["persisted"])
chi2_m, p_chi_m, _, _ = stats.chi2_contingency(ct_m)
print(f"  Chi-square (Mistral GREEN vs TRAD): χ²={chi2_m:.2f}  p={p_chi_m:.4f} {sig(p_chi_m)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DOWN-ROUND PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("2. DOWN-ROUND PERSISTENCE")
print("="*72)

# Down rounds: companies that have at least one down-round deal
down_cos = set(d_exp[d_exp["is_down_round"]==1]["co"])
print(f"\n  Companies with ≥1 down round: {len(down_cos):,}")

# Rebuild persistence for down-round cohort
# A company is "in a down round period" at round N if round N+1 is a down round
down_records = []
for co, rounds in co_rounds.items():
    if co not in down_cos:
        continue
    for i in range(len(rounds)-1):
        r_now  = rounds[i]
        r_next = rounds[i+1]
        # Check if the NEXT round is a down round
        next_down = d_exp[(d_exp["co"]==co) &
                          (d_exp["Deal Date"]==r_next["date"])]["is_down_round"].max()
        if next_down != 1:
            continue
        for inv in r_now["investors"]:
            label = merged_labels.get(inv, "UNKNOWN")
            persisted = int(inv in r_next["investors"])
            down_records.append({
                "co": co, "investor": inv,
                "haiku_label": label,
                "persisted": persisted,
            })

dr = pd.DataFrame(down_records)
dr = dr[dr["haiku_label"] != "UNKNOWN"]
print(f"  Investor × pre-down-round observations: {len(dr):,}")
print()
print(f"  DOWN-ROUND PERSISTENCE BY HAIKU LABEL:")
print(f"  {'Label':<15}  {'N obs':>8}  {'Persist%':>9}")
print(f"  {'-'*38}")
down_persist = {}
for label in ["GREEN_VC","ESG_ALIGNED","TRADITIONAL"]:
    sub = dr[dr["haiku_label"]==label]
    if len(sub) == 0:
        print(f"  {label:<15}  {'0':>8}  {'N/A':>9}")
        down_persist[label] = (0, np.nan)
        continue
    p = sub["persisted"].mean()
    down_persist[label] = (len(sub), p)
    print(f"  {label:<15}  {len(sub):>8,}  {p:>9.1%}")

if all(v[0]>0 for v in down_persist.values()):
    ct_d = pd.crosstab(dr["haiku_label"], dr["persisted"]).reindex(["GREEN_VC","ESG_ALIGNED","TRADITIONAL"])
    chi2_d, p_d, _, _ = stats.chi2_contingency(ct_d)
    print(f"\n  Chi-square (down rounds): χ²={chi2_d:.2f}  p={p_d:.4f} {sig(p_d)}")
else:
    print("\n  (insufficient observations in one group for chi-square)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BRIDGE TYPE ANALYSIS WITH HAIKU LABELS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("3. BRIDGE TYPE ANALYSIS WITH HAIKU GREEN LABELS")
print("="*72)

cdf = pd.read_csv(OUT / "cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["log_cap"]   = cdf["log_capital"]
cdf["nrounds"]   = np.log1p(cdf["n_rounds"].fillna(1))
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

net = pd.read_csv(OUT / "network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm = net.set_index("investor")["community"].to_dict()

# Community-level Haiku green pct
comm_green_haiku = {}
comm_esg_haiku   = {}
for comm, grp in net.groupby("community"):
    labels = [merged_labels.get(inv,"UNKNOWN") for inv in grp["investor"]]
    n_known = sum(1 for l in labels if l != "UNKNOWN")
    if n_known == 0:
        comm_green_haiku[comm] = 0.0
        comm_esg_haiku[comm]   = 0.0
    else:
        comm_green_haiku[comm] = labels.count("GREEN_VC") / n_known
        comm_esg_haiku[comm]   = labels.count("ESG_ALIGNED") / n_known

# Build bridge timing (from Script 41/42 logic)
d_exp2 = (deals.dropna(subset=["Investors"])
          .assign(inv=lambda x: x["Investors"].str.split(";"))
          .explode("inv")
          .assign(inv=lambda x: x["inv"].str.strip())
          .query('inv != ""'))
d_exp2["community"] = d_exp2["inv"].map(inv_to_comm)
d_exp2["Year"] = pd.to_numeric(d_exp2["Year"], errors="coerce")
d_exp2c = d_exp2.dropna(subset=["community","Year"])

cross_cos = set(cdf[cdf["cross_community"]==1]["co"])
timing_rows = []
for co, grp in d_exp2c.groupby("co"):
    if co not in cross_cos: continue
    comm_first = {}
    for yr in sorted(grp["Year"].unique()):
        for c in grp[grp["Year"]==yr]["community"].unique():
            if c not in comm_first: comm_first[c] = yr
    if len(comm_first) < 2: continue
    sorted_e = sorted(comm_first.items(), key=lambda x: x[1])
    bridge_comm = sorted_e[1][0]
    timing_rows.append({
        "co": co,
        "bridge_green_pct_haiku": comm_green_haiku.get(bridge_comm, 0.0),
        "bridge_esg_pct_haiku":   comm_esg_haiku.get(bridge_comm, 0.0),
    })

timing_df = pd.DataFrame(timing_rows)
timing_df["bridge_is_green_haiku"] = (timing_df["bridge_green_pct_haiku"] >= 0.5).astype(float)
timing_df["bridge_is_trad_haiku"]  = (timing_df["bridge_green_pct_haiku"] <  0.5).astype(float)
timing_df["bridge_is_esg_haiku"]   = (timing_df["bridge_esg_pct_haiku"]  >= 0.3).astype(float)

df_b = cdf.merge(timing_df, on="co", how="left")
for col in ["bridge_is_green_haiku","bridge_is_trad_haiku","bridge_is_esg_haiku"]:
    df_b[col] = df_b[col].fillna(0)
df_b = df_b.dropna(subset=["log_cap","deal_year","specter_novelty"])

print(f"\n  Cross-community companies with bridge identified: {timing_df['co'].nunique():,}")
print(f"  Bridge = GREEN (≥50% green in bridge community): {timing_df['bridge_is_green_haiku'].sum():.0f}")
print(f"  Bridge = TRAD  (<50% green in bridge community): {timing_df['bridge_is_trad_haiku'].sum():.0f}")

# Descriptive
print()
print("  Failure rates by bridge type (Haiku labels):")
wc_fail = df_b[df_b["cross_community"]==0]["is_failed"].mean()
bg_fail = df_b[(df_b["cross_community"]==1)&(df_b["bridge_is_green_haiku"]==1)]["is_failed"].mean()
bt_fail = df_b[(df_b["cross_community"]==1)&(df_b["bridge_is_trad_haiku"]==1)]["is_failed"].mean()
print(f"  Within-community (baseline):       {wc_fail:.1%}")
print(f"  Cross-comm, green bridge:          {bg_fail:.1%}  (N={int(timing_df['bridge_is_green_haiku'].sum())})")
print(f"  Cross-comm, trad bridge:           {bt_fail:.1%}  (N={int(timing_df['bridge_is_trad_haiku'].sum())})")

# Regression: green bridge vs trad bridge → failure
print()
print("  OLS: bridge type → failure (ref = within-community)")
print(f"  {'Variable':<30}  {'β':>9}  {'SE':>8}  {'Sig':>5}  {'Effect':>8}")
print(f"  {'-'*65}")
m = smf.ols(f"is_failed ~ bridge_is_green_haiku + bridge_is_trad_haiku + {CTRL}",
            data=df_b.dropna(subset=["is_failed"])).fit(cov_type="HC1")
for var in ["bridge_is_green_haiku","bridge_is_trad_haiku"]:
    b  = m.params.get(var, np.nan)
    se = m.bse.get(var, np.nan)
    p  = m.pvalues.get(var, 1.0)
    print(f"  {var:<30}  {b:>+9.4f}  {se:>8.4f}  {sig(p):>5}  {b*100:>+.1f}pp")

# Wald test: green bridge = trad bridge?
cov_m = m.cov_params()
if "bridge_is_green_haiku" in cov_m.index and "bridge_is_trad_haiku" in cov_m.index:
    from scipy import stats as st
    b_g = m.params["bridge_is_green_haiku"]
    b_t = m.params["bridge_is_trad_haiku"]
    se_diff = np.sqrt(float(cov_m.loc["bridge_is_green_haiku","bridge_is_green_haiku"]) +
                      float(cov_m.loc["bridge_is_trad_haiku","bridge_is_trad_haiku"]) -
                      2*float(cov_m.loc["bridge_is_green_haiku","bridge_is_trad_haiku"]))
    t_w = (b_g - b_t) / se_diff
    p_w = 2 * st.t.sf(abs(t_w), df=m.df_resid)
    print(f"\n  Wald test (green=trad): t={t_w:.2f}  p={p_w:.4f} {sig(p_w)}")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("VERDICT: DOES THE PATIENT CAPITAL STORY SURVIVE?")
print("="*72)

g_n, g_p = haiku_persist["GREEN_VC"]
e_n, e_p = haiku_persist["ESG_ALIGNED"]
t_n, t_p = haiku_persist["TRADITIONAL"]

print(f"""
  PERSISTENCE (Haiku labels):
    GREEN_VC:    {g_p:.1%}  (N={g_n:,})
    ESG_ALIGNED: {e_p:.1%}  (N={e_n:,})
    TRADITIONAL: {t_p:.1%}  (N={t_n:,})
    Overall chi-square: p={p_chi:.4f} {sig(p_chi)}

  INTERPRETATION:
""")

if p_chi < 0.05:
    if g_p > t_p:
        print("  ✅ GREEN_VC investors DO persist more than TRADITIONAL — patient capital story HOLDS.")
        print("     However, note that GREEN_VC is now only 8.1% of investors (vs 57.8% under Mistral).")
        print("     The effect is real but concentrated in a small, genuinely green-focused minority.")
    elif g_p < t_p:
        print("  ⚠️  GREEN_VC investors actually persist LESS than TRADITIONAL (unexpected).")
        print("     Groups differ significantly but in the opposite direction.")
else:
    print("  ❌ No significant difference in persistence across Haiku groups.")
    print("     The patient capital finding was largely an artifact of Mistral's")
    print("     over-classification (57.8% GREEN vs true 8.1%).")
    print("     RECOMMENDATION: DROP the persistence/patient capital story from the paper.")
    print("     Replace with: 'investor green focus does not predict persistence;")
    print("     network position (cross-community) is the operative mechanism.'")

print(f"""
  BRIDGE TYPE (Haiku labels):
    Traditional bridge → failure: see regression above
    Green bridge → failure: see regression above

  If bridge_trad still significantly reduces failure → bridge type finding is robust.
  If bridge_green is now n.s. → green bridge effect was also artefactual.
""")
