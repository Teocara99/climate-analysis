"""
Script 45: Haiku classification merge, ESG sensitivity, and full re-analysis
=============================================================================

STEP 1 — Merge Haiku + Mistral classifications
STEP 2 — Build three ESG specifications at company level
STEP 3 — Re-run all key models (3 specs × 7 models)
STEP 4 — PitchBook investor type comparison
STEP 5 — Investor quality controls
STEP 6 — Master summary comparison table
"""

import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
from load_data import load_deals, load_companies

def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

def cell(b, p, binary=True):
    effect = f"{b*100:+.1f}pp" if binary else f"{b:+.4f}"
    return f"{effect} {sig(p)}"

OUT  = Path("output")
CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD BASE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading base data…")
cdf = pd.read_csv(OUT / "cpu/cpu_company_dataset.csv")
cdf["co"] = cdf["Companies"].str.strip()
cdf["is_failed"] = (cdf["outcome"] == "Failed").astype(float)
cdf["is_exited"]  = cdf["exited"].astype(float)
cdf["log_cap"]    = cdf["log_capital"]
cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
cdf["got_followon"] = (cdf["n_rounds"].fillna(0) > 1).astype(float)
cdf["country_fe"] = cdf["hq_country"].where(
    cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")

deals = load_deals()
deals["co"] = deals["Companies"].str.strip()

companies_pb = load_companies()
companies_pb["co"] = companies_pb["Companies"].str.strip()

# Outcome extras from PitchBook companies file
co_emp     = companies_pb.dropna(subset=["Employees"]).drop_duplicates("co").set_index("co")["Employees"]
co_sprob   = companies_pb.dropna(subset=["Success Probability"]).drop_duplicates("co").set_index("co")["Success Probability"]
co_patent  = companies_pb.dropna(subset=["Total Patent Documents"]).drop_duplicates("co").set_index("co")["Total Patent Documents"]
cdf["log_employees"] = np.log1p(pd.to_numeric(cdf["co"].map(co_emp), errors="coerce"))
cdf["success_prob"]  = pd.to_numeric(cdf["co"].map(co_sprob), errors="coerce")
cdf["has_patent"]    = (pd.to_numeric(cdf["co"].map(co_patent), errors="coerce").fillna(0) > 0).astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: MERGE CLASSIFICATIONS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("STEP 1: MERGE HAIKU + MISTRAL CLASSIFICATIONS")
print("="*72)

haiku   = pd.read_csv(OUT / "green_classify/investor_green_classifications.csv")
mistral = pd.read_csv(OUT / "investors_classified_output.csv")
all_inv = pd.read_csv(OUT / "all_investors.csv")

haiku["investor"]   = haiku["investor"].str.strip()
mistral["investor_name"] = mistral["investor_name"].str.strip()
all_inv["Investor"] = all_inv["Investor"].str.strip()

haiku_names   = set(haiku["investor"])
mistral_names = set(mistral["investor_name"])
all_names     = set(all_inv["Investor"])

print(f"\n  Total original investors (all_investors.csv): {len(all_names):,}")
print(f"  In Haiku file (PitchBook export):            {len(haiku_names):,}")
matched     = all_names & haiku_names
only_old    = all_names - haiku_names
only_haiku  = haiku_names - all_names
print(f"  Matched (in both):                           {len(matched):,}")
print(f"  Only in original (no PitchBook data):        {len(only_old):,}")
print(f"  Only in PitchBook (new):                     {len(only_haiku):,}")

# Agreement between Mistral and Haiku for matched investors
both = mistral.merge(haiku, left_on="investor_name", right_on="investor", how="inner")
# Normalise Mistral: GREEN_VC vs rest (Mistral had very few ESG_ALIGNED)
both["m_binary"] = both["green_focus"].apply(
    lambda x: "GREEN_VC" if x == "GREEN_VC" else ("ESG_ALIGNED" if x == "ESG_ALIGNED" else "TRADITIONAL"))
agree_3way = (both["m_binary"] == both["green_focus_haiku"]).mean()
# Binary: green vs not-green
both["m_green"] = (both["green_focus"] == "GREEN_VC").astype(int)
both["h_green"] = (both["green_focus_haiku"] == "GREEN_VC").astype(int)
agree_binary = (both["m_green"] == both["h_green"]).mean()
print(f"\n  Mistral vs Haiku agreement (3-way): {agree_3way:.1%}")
print(f"  Mistral vs Haiku agreement (binary GREEN/not): {agree_binary:.1%}")

# Disagreement cases: Mistral=GREEN but Haiku=not-green and vice versa
m_green_h_not = both[(both["m_green"]==1) & (both["h_green"]==0)]
m_not_h_green = both[(both["m_green"]==0) & (both["h_green"]==1)]
print(f"  Mistral=GREEN, Haiku=not-GREEN: {len(m_green_h_not):,}")
print(f"  Mistral=not-GREEN, Haiku=GREEN: {len(m_not_h_green):,}")

# D: Distribution comparison
print("\n  Distribution comparison:")
m_dist = mistral[mistral["green_focus"].isin(["GREEN_VC","ESG_ALIGNED","TRADITIONAL"])]["green_focus"].value_counts(normalize=True)
h_dist = haiku["green_focus_haiku"].value_counts(normalize=True)
print(f"  {'Label':<15} {'Mistral':>10}  {'Haiku':>10}")
print(f"  {'-'*38}")
for label in ["GREEN_VC","ESG_ALIGNED","TRADITIONAL"]:
    print(f"  {label:<15} {m_dist.get(label,0):>10.1%}  {h_dist.get(label,0):>10.1%}")

# Build merged classification: Haiku preferred, Mistral fallback
merged_labels = {}
for _, r in haiku.iterrows():
    merged_labels[r["investor"]] = {
        "green_focus_final": r["green_focus_haiku"],
        "source": "haiku",
        "confidence": r["confidence"],
    }
for _, r in mistral.iterrows():
    name = r["investor_name"]
    if name not in merged_labels:
        gf = r["green_focus"] if r["green_focus"] in ("GREEN_VC","ESG_ALIGNED","TRADITIONAL") else "TRADITIONAL"
        merged_labels[name] = {
            "green_focus_final": gf,
            "source": "mistral_fallback",
            "confidence": "low",
        }

print(f"\n  Final merged labels: {len(merged_labels):,}")
print(f"    Source=haiku:            {sum(1 for v in merged_labels.values() if v['source']=='haiku'):,}")
print(f"    Source=mistral_fallback: {sum(1 for v in merged_labels.values() if v['source']=='mistral_fallback'):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD THREE ESG SPECIFICATIONS AT COMPANY LEVEL
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("STEP 2: BUILD THREE ESG SPECIFICATIONS")
print("="*72)

# Explode deals → investor → label → aggregate per company
d_exp = (deals.dropna(subset=["Investors"])
         .assign(inv=lambda x: x["Investors"].str.split(";"))
         .explode("inv")
         .assign(inv=lambda x: x["inv"].str.strip())
         .query('inv != ""'))

d_exp["green_final"] = d_exp["inv"].map(
    lambda n: merged_labels.get(n, {}).get("green_focus_final", "UNKNOWN"))

# Aggregate per company
co_agg = {}
for co, grp in d_exp.groupby("co"):
    labels = grp["green_final"].tolist()
    n = len(labels)
    n_green  = labels.count("GREEN_VC")
    n_esg    = labels.count("ESG_ALIGNED")
    n_trad   = labels.count("TRADITIONAL")
    n_unk    = labels.count("UNKNOWN")
    n_known  = n - n_unk
    if n_known == 0:
        continue
    co_agg[co] = {
        # Spec A: ESG = Green
        "pct_green_broad":  (n_green + n_esg) / n_known,
        "pct_trad_broad":   n_trad / n_known,
        # Spec B: ESG = Traditional
        "pct_green_strict": n_green / n_known,
        "pct_trad_strict":  (n_esg + n_trad) / n_known,
        # Spec C: three-way
        "pct_green_vc":     n_green / n_known,
        "pct_esg":          n_esg / n_known,
        "pct_trad":         n_trad / n_known,
        # Binaries
        "has_green_haiku":  int(n_green > 0),
        "has_esg_haiku":    int(n_esg > 0),
        "has_trad_haiku":   int(n_trad > 0),
    }

esg_df = pd.DataFrame.from_dict(co_agg, orient="index").reset_index()
esg_df.columns = ["co"] + list(esg_df.columns[1:])

# Syndicate type dummies for each spec
def syndicate_type(row, green_col, trad_col):
    g = row[green_col]
    t = row[trad_col]
    if g == 1.0: return "Pure Green"
    if t == 1.0: return "Pure Trad"
    return "Mixed"

esg_df["syn_A"] = esg_df.apply(lambda r: syndicate_type(r,"pct_green_broad","pct_trad_broad"), axis=1)
esg_df["syn_B"] = esg_df.apply(lambda r: syndicate_type(r,"pct_green_strict","pct_trad_strict"), axis=1)
esg_df["pure_green_A"] = (esg_df["syn_A"]=="Pure Green").astype(float)
esg_df["pure_trad_A"]  = (esg_df["syn_A"]=="Pure Trad").astype(float)
esg_df["pure_green_B"] = (esg_df["syn_B"]=="Pure Green").astype(float)
esg_df["pure_trad_B"]  = (esg_df["syn_B"]=="Pure Trad").astype(float)
esg_df["pure_green_C"] = (esg_df["pct_green_vc"]==1.0).astype(float)
esg_df["pure_trad_C"]  = (esg_df["pct_trad"]==1.0).astype(float)

# Old Mistral labels (from cpu_company_dataset directly)
print(f"\n  Companies with Haiku-based ESG labels: {len(esg_df):,}")
corr_AB = esg_df["pct_green_broad"].corr(esg_df["pct_green_strict"])
print(f"  Correlation pct_green_broad vs pct_green_strict: {corr_AB:.3f}")

# Count syndicate type switches
df_main = cdf.merge(esg_df, on="co", how="left")
switchAB = (df_main["syn_A"] != df_main["syn_B"]).sum()
print(f"  Companies changing syndicate type A→B: {switchAB:,} ({switchAB/len(df_main):.1%})")

print("\n  Spec A distribution (ESG=Green):")
print(f"  {'Type':<15}  N      %")
for t, n in df_main["syn_A"].value_counts().items():
    print(f"  {t:<15}  {n:>5,}  {n/len(df_main):.1%}")
print("\n  Spec B distribution (ESG=Trad):")
for t, n in df_main["syn_B"].value_counts().items():
    print(f"  {t:<15}  {n:>5,}  {n/len(df_main):.1%}")

df_main = df_main.dropna(subset=["log_cap","deal_year","specter_novelty"]).copy()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: MASTER REGRESSION TABLE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("STEP 3: MASTER REGRESSION TABLE")
print("="*72)

# Helper: run OLS and extract one coefficient
def ols_coef(formula, data, var, binary=True):
    try:
        sub = data.dropna(subset=[formula.split("~")[0].strip()] +
                          [v for v in [var] if v in data.columns])
        m = smf.ols(formula, data=sub).fit(cov_type="HC1")
        b = m.params.get(var, np.nan)
        p = m.pvalues.get(var, 1.0)
        return cell(b, p, binary)
    except Exception as e:
        return f"ERR"

# Old Mistral labels already in cdf: pct_green, pct_traditional, pure_green, pure_trad
df_main["has_green_mistral"] = (df_main["pct_green"] > 0).astype(float)
df_main["has_trad_mistral"]  = (df_main["pct_traditional"] > 0).astype(float)

specs = {
    "Old Mistral": {
        "pure_green": "pure_green",
        "pure_trad":  "pure_trad",
        "has_green":  "has_green_mistral",
        "has_trad":   "has_trad_mistral",
        "pct_green":  "pct_green",
        "pct_esg":    None,
        "pct_trad":   "pct_traditional",
    },
    "Spec A (ESG=Green)": {
        "pure_green": "pure_green_A",
        "pure_trad":  "pure_trad_A",
        "has_green":  "has_green_haiku",
        "has_trad":   "has_trad_haiku",
        "pct_green":  "pct_green_broad",
        "pct_esg":    None,
        "pct_trad":   "pct_trad_broad",
    },
    "Spec B (ESG=Trad)": {
        "pure_green": "pure_green_B",
        "pure_trad":  "pure_trad_B",
        "has_green":  "has_green_haiku",
        "has_trad":   "has_trad_haiku",
        "pct_green":  "pct_green_strict",
        "pct_esg":    None,
        "pct_trad":   "pct_trad_strict",
    },
    "Spec C (3-way)": {
        "pure_green": "pure_green_C",
        "pure_trad":  "pure_trad_C",
        "has_green":  "has_green_haiku",
        "has_trad":   "has_trad_haiku",
        "pct_green":  "pct_green_vc",
        "pct_esg":    "pct_esg",
        "pct_trad":   "pct_trad",
    },
}

# Print header
col_w = 22
header = f"  {'Model':<35}"
for spec in specs:
    header += f"  {spec:>{col_w}}"
print("\n" + header)
print(f"  {'-'*120}")

def run_model(label, outcome, formula_fn, var, binary=True):
    row = f"  {label:<35}"
    results = []
    for spec_name, sv in specs.items():
        try:
            f = formula_fn(sv)
            if f is None:
                row += f"  {'—':>{col_w}}"
                results.append(None)
                continue
            sub = df_main.dropna(subset=[outcome]).copy()
            m = smf.ols(f, data=sub).fit(cov_type="HC1")
            b = m.params.get(var(sv), np.nan)
            p = m.pvalues.get(var(sv), 1.0)
            c = cell(b, p, binary)
            row += f"  {c:>{col_w}}"
            results.append((b, p))
        except:
            row += f"  {'ERR':>{col_w}}"
            results.append(None)
    print(row)
    return results

print("\n  MODEL A: Green → Failure (pure_green, ref=Mixed)")
run_model("A. pure_green → failure", "is_failed",
    lambda sv: f"is_failed ~ {sv['pure_green']} + {sv['pure_trad']} + {CTRL}",
    lambda sv: sv["pure_green"])

print()
print("  MODEL B: Traditional → Exit (pure_trad, ref=Mixed)")
run_model("B. pure_trad → exit", "is_exited",
    lambda sv: f"is_exited ~ {sv['pure_green']} + {sv['pure_trad']} + {CTRL}",
    lambda sv: sv["pure_trad"])

print()
print("  MODEL C: Position subsumes composition (cross_community + has_green + has_trad)")
run_model("C. cross_community → failure", "is_failed",
    lambda sv: f"is_failed ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
    lambda sv: "cross_community")
run_model("C. has_green → failure (net ctrl)", "is_failed",
    lambda sv: f"is_failed ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
    lambda sv: sv["has_green"])
run_model("C. has_trad → failure (net ctrl)", "is_failed",
    lambda sv: f"is_failed ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
    lambda sv: sv["has_trad"])

print()
print("  MODEL D: Three-way (Spec C only)")
row_D = f"  {'D. pct_green_vc → failure':<35}"
row_esg = f"  {'D. pct_esg → failure':<35}"
row_trad = f"  {'D. pct_trad → failure':<35}"
for spec_name, sv in specs.items():
    if spec_name != "Spec C (3-way)":
        row_D += f"  {'—':>{col_w}}"
        row_esg += f"  {'—':>{col_w}}"
        row_trad += f"  {'—':>{col_w}}"
        continue
    try:
        f = f"is_failed ~ pct_green_vc + pct_esg + pct_trad + cross_community + {CTRL}"
        sub = df_main.dropna(subset=["is_failed"]).copy()
        m = smf.ols(f, data=sub).fit(cov_type="HC1")
        for var_name, row_ref in [("pct_green_vc",row_D),("pct_esg",row_esg),("pct_trad",row_trad)]:
            b = m.params.get(var_name, np.nan)
            p = m.pvalues.get(var_name, 1.0)
            c = cell(b, p)
            if var_name == "pct_green_vc": row_D += f"  {c:>{col_w}}"
            elif var_name == "pct_esg":   row_esg += f"  {c:>{col_w}}"
            else:                          row_trad += f"  {c:>{col_w}}"
    except Exception as e:
        row_D += f"  ERR"
        row_esg += f"  ERR"
        row_trad += f"  ERR"
print(row_D)
print(row_esg)
print(row_trad)

print()
print("  MODEL E: Bridge type → failure (green vs trad bridge)")
# Need bridge community labels
net = pd.read_csv(OUT / "network/network_metrics.csv")
net["investor"] = net["investor"].str.strip()
inv_to_comm = net.set_index("investor")["community"].to_dict()
# Community dominant green label (using merged labels)
comm_green_pct = {}
for comm, grp in net.groupby("community"):
    greens = [merged_labels.get(inv,{}).get("green_focus_final","UNKNOWN")
              for inv in grp["investor"]]
    n_known = sum(1 for g in greens if g != "UNKNOWN")
    if n_known > 0:
        comm_green_pct[comm] = greens.count("GREEN_VC") / n_known
    else:
        comm_green_pct[comm] = 0.0

# Build bridge timing
d_exp2 = (deals.dropna(subset=["Investors"])
          .assign(inv=lambda x: x["Investors"].str.split(";"))
          .explode("inv")
          .assign(inv=lambda x: x["inv"].str.strip())
          .query('inv != ""'))
d_exp2["community"] = d_exp2["inv"].map(inv_to_comm)
d_exp2["Year"] = pd.to_numeric(d_exp2["Year"], errors="coerce")
d_exp2c = d_exp2.dropna(subset=["community","Year"])

cross_cos = set(df_main[df_main["cross_community"]==1]["co"])
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
    timing_rows.append({"co": co,
                        "bridge_green_pct": comm_green_pct.get(bridge_comm, 0.0)})

timing_df = pd.DataFrame(timing_rows)
timing_df["bridge_is_green_haiku"] = (timing_df["bridge_green_pct"] >= 0.5).astype(float)
timing_df["bridge_is_trad_haiku"]  = (timing_df["bridge_green_pct"] <  0.5).astype(float)
df_bridge = df_main.merge(timing_df, on="co", how="left")
for col in ["bridge_is_green_haiku","bridge_is_trad_haiku"]:
    df_bridge[col] = df_bridge[col].fillna(0)

row_bg = f"  {'E. bridge_green → failure':<35}"
row_bt = f"  {'E. bridge_trad → failure':<35}"
for spec_name in specs:
    try:
        f = f"is_failed ~ bridge_is_green_haiku + bridge_is_trad_haiku + {CTRL}"
        sub = df_bridge.dropna(subset=["is_failed"]).copy()
        m = smf.ols(f, data=sub).fit(cov_type="HC1")
        bg = m.params.get("bridge_is_green_haiku", np.nan)
        pg = m.pvalues.get("bridge_is_green_haiku", 1.0)
        bt = m.params.get("bridge_is_trad_haiku",  np.nan)
        pt = m.pvalues.get("bridge_is_trad_haiku",  1.0)
        row_bg += f"  {cell(bg,pg):>{col_w}}"
        row_bt += f"  {cell(bt,pt):>{col_w}}"
    except:
        row_bg += f"  {'ERR':>{col_w}}"
        row_bt += f"  {'ERR':>{col_w}}"
    break  # same result for all specs (bridge label doesn't change for Old Mistral here)
# Fill remaining specs (bridge labels are Haiku-only, same for A/B/C)
for _ in range(len(specs)-1):
    row_bg += f"  {'(same)':>{col_w}}"
    row_bt += f"  {'(same)':>{col_w}}"
print(row_bg)
print(row_bt)

print()
print("  MODEL F: Persistence rates (green vs trad)")
for spec_name, sv in specs.items():
    g_col = sv["pct_green"]
    t_col = sv["pct_trad"]
    sub = df_main.dropna(subset=["is_failed"]).copy()
    sub["is_green_any"] = (sub.get(g_col, pd.Series(0, index=sub.index)).fillna(0) > 0).astype(int)
    sub["is_trad_any"]  = (sub.get(t_col, pd.Series(0, index=sub.index)).fillna(0) > 0).astype(int)
    n_g = sub["is_green_any"].sum()
    n_t = sub["is_trad_any"].sum()
    f_g = sub[sub["is_green_any"]==1]["is_failed"].mean() if n_g > 0 else np.nan
    f_t = sub[sub["is_trad_any"]==1]["is_failed"].mean() if n_t > 0 else np.nan
    if n_g > 0 and n_t > 0:
        ct = pd.crosstab(sub["is_green_any"] | sub["is_trad_any"],
                         sub["is_failed"])
        ct2 = pd.crosstab(sub[sub["is_green_any"]==1]["is_failed"].rename("f"),
                          sub[sub["is_trad_any"]==1]["is_failed"].rename("f"))
    print(f"    {spec_name}: fail_rate green={f_g:.1%}  trad={f_t:.1%}  diff={f_g-f_t:+.1%}")

print()
print("  MODEL G: Positive outcomes (cross_community + has_green + has_trad)")
for outcome, label, binary in [
    ("log_employees","log_employees",False),
    ("success_prob","success_prob",False),
    ("has_patent","has_patent",True),
    ("got_followon","got_followon",True),
]:
    run_model(f"G. cross_comm → {label}", outcome,
        lambda sv, o=outcome: f"{o} ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
        lambda sv: "cross_community", binary)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: PITCHBOOK INVESTOR TYPE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("STEP 4: PITCHBOOK TYPE vs LLM TYPE COMPARISON")
print("="*72)

pb_inv = pd.read_excel("data/Investors_data.xlsx", sheet_name="Data", header=7)
pb_inv["Investors"] = pb_inv["Investors"].str.strip()
mistral_m = mistral.rename(columns={"investor_name":"Investors","investor_type":"llm_type"})
comp = pb_inv[["Investors","Primary Investor Type"]].merge(
    mistral_m[["Investors","llm_type"]], on="Investors", how="inner")

# Map PitchBook type to our taxonomy
type_map = {
    "Venture Capital": "IVC",
    "Corporate Venture Capital": "CVC",
    "Government": "GVC",
    "Impact Investing": "Impact_VC",
    "Angel (individual)": "Angel_Network",
    "Angel Group": "Angel_Network",
    "University": "University_VC",
    "Asset Manager": "Bank_VC",
    "Hedge Fund": "Bank_VC",
    "Accelerator/Incubator": "OTHER",
    "Corporation": "CVC",
    "PE/Buyout": "IVC",
    "Growth/Expansion": "IVC",
    "Family Office": "IVC",
}
comp["pb_mapped"] = comp["Primary Investor Type"].map(type_map).fillna("OTHER")
agree_type = (comp["pb_mapped"] == comp["llm_type"]).mean()
print(f"\n  PitchBook vs LLM type agreement: {agree_type:.1%}  (N={len(comp):,})")

print("\n  Top 10 disagreements (PitchBook type vs LLM type):")
disagree = comp[comp["pb_mapped"] != comp["llm_type"]].copy()
disagree_pairs = disagree.groupby(["pb_mapped","llm_type"]).size().sort_values(ascending=False).head(10)
for (pb, llm), n in disagree_pairs.items():
    print(f"    PB={pb:<15} LLM={llm:<15}  N={n:>5,}")

print("\n  10 example disagreements (investor names):")
for _, r in disagree.head(10).iterrows():
    print(f"    {r['Investors']:<45} PB={r['pb_mapped']:<12} LLM={r['llm_type']}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: INVESTOR QUALITY CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("STEP 5: INVESTOR QUALITY CONTROLS")
print("="*72)

pb_inv["AUM"] = pd.to_numeric(pb_inv["AUM"], errors="coerce")
pb_inv["Total Investments_pb"] = pd.to_numeric(pb_inv["Total Investments"], errors="coerce")
pb_inv["Total Exits_pb"]       = pd.to_numeric(pb_inv["Total Exits"], errors="coerce")
pb_inv["Year Founded_pb"]      = pd.to_numeric(pb_inv["Year Founded"], errors="coerce")
pb_inv["investor_age"] = 2026 - pb_inv["Year Founded_pb"]

pb_dict = pb_inv.drop_duplicates("Investors").set_index("Investors")[["AUM","Total Investments_pb","Total Exits_pb","investor_age"]].to_dict("index")

# Build per-company investor quality averages
co_quality = {}
for co, grp in d_exp.groupby("co"):
    investors = grp["inv"].unique()
    aums, deals_c, exits_c, ages = [], [], [], []
    for inv in investors:
        d = pb_dict.get(inv, {})
        if not np.isnan(d.get("AUM", np.nan)):        aums.append(d["AUM"])
        if not np.isnan(d.get("Total Investments_pb", np.nan)): deals_c.append(d["Total Investments_pb"])
        if not np.isnan(d.get("Total Exits_pb", np.nan)):       exits_c.append(d["Total Exits_pb"])
        if not np.isnan(d.get("investor_age", np.nan)):         ages.append(d["investor_age"])
    co_quality[co] = {
        "avg_aum":    np.mean(aums)    if aums    else np.nan,
        "avg_deals":  np.mean(deals_c) if deals_c else np.nan,
        "avg_exits":  np.mean(exits_c) if exits_c else np.nan,
        "avg_age":    np.mean(ages)    if ages    else np.nan,
    }

quality_df = pd.DataFrame.from_dict(co_quality, orient="index").reset_index()
quality_df.columns = ["co","avg_aum","avg_deals","avg_exits","avg_age"]
quality_df["log_avg_aum"]   = np.log1p(quality_df["avg_aum"].fillna(0))
quality_df["log_avg_deals"] = np.log1p(quality_df["avg_deals"].fillna(0))
quality_df["log_avg_exits"] = np.log1p(quality_df["avg_exits"].fillna(0))

df_q = df_main.merge(quality_df, on="co", how="left")
print(f"\n  Companies with avg AUM data:   {df_q['avg_aum'].notna().sum():,}")
print(f"  Companies with avg deals data: {df_q['avg_deals'].notna().sum():,}")
print(f"  Companies with avg exits data: {df_q['avg_exits'].notna().sum():,}")
print(f"  Companies with avg age data:   {df_q['avg_age'].notna().sum():,}")

CTRL_Q = CTRL + " + log_avg_deals + log_avg_exits + avg_age"

print("\n  Cross-community → failure WITH investor quality controls:")
print(f"  {'Model':<45}  {'β_cross':>9}  {'SE':>8}  {'Sig':>5}  N")
print(f"  {'-'*75}")
for label, formula in [
    ("Baseline (no quality controls)",
     f"is_failed ~ cross_community + {CTRL}"),
    ("+ avg investor deal count + exits + age",
     f"is_failed ~ cross_community + log_avg_deals + log_avg_exits + avg_age + {CTRL}"),
    ("+ has_green_haiku + has_trad_haiku",
     f"is_failed ~ cross_community + has_green_haiku + has_trad_haiku + {CTRL}"),
    ("Full: quality + green + network",
     f"is_failed ~ cross_community + has_green_haiku + has_trad_haiku + log_avg_deals + log_avg_exits + avg_age + {CTRL}"),
]:
    sub = df_q.dropna(subset=["is_failed"]).copy()
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="HC1")
        b  = m.params.get("cross_community", np.nan)
        se = m.bse.get("cross_community", np.nan)
        p  = m.pvalues.get("cross_community", 1.0)
        print(f"  {label:<45}  {b:>+9.4f}  {se:>8.4f}  {sig(p):>5}  {int(m.nobs):,}")
    except Exception as e:
        print(f"  {label:<45}  ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SUMMARY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*72)
print("STEP 6: SUMMARY — WHAT CHANGED WITH BETTER DATA?")
print("="*72)

summary_models = [
    ("Cross-community → failure", "is_failed",
     lambda sv: f"is_failed ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
     "cross_community"),
    ("Pure green → failure", "is_failed",
     lambda sv: f"is_failed ~ {sv['pure_green']} + {sv['pure_trad']} + {CTRL}",
     None),
    ("has_green → failure (net ctrl)", "is_failed",
     lambda sv: f"is_failed ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
     None),
    ("cross_comm → exit", "is_exited",
     lambda sv: f"is_exited ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
     "cross_community"),
    ("cross_comm → patents", "has_patent",
     lambda sv: f"has_patent ~ cross_community + {sv['has_green']} + {sv['has_trad']} + {CTRL}",
     "cross_community"),
]

print(f"\n  {'Finding':<35} {'Old Mistral':>18} {'Spec A':>18} {'Spec B':>18} {'Spec C':>18}  Robust?")
print(f"  {'-'*115}")

for name, outcome, formula_fn, main_var in summary_models:
    row = f"  {name:<35}"
    sigs_found = []
    for spec_name, sv in specs.items():
        mvar = main_var if main_var else sv["pure_green"]
        if main_var is None and spec_name == "Spec C (3-way)":
            mvar = sv["pct_green"]
        try:
            sub = df_main.dropna(subset=[outcome]).copy()
            f = formula_fn(sv)
            m = smf.ols(f, data=sub).fit(cov_type="HC1")
            b = m.params.get(mvar, np.nan)
            p = m.pvalues.get(mvar, 1.0)
            c = cell(b, p)
            row += f"  {c:>18}"
            sigs_found.append(p < 0.05)
        except:
            row += f"  {'ERR':>18}"
            sigs_found.append(False)
    robust = "✅" if all(sigs_found) else ("⚠️" if any(sigs_found) else "❌")
    row += f"  {robust}"
    print(row)

print(f"""
  LEGEND:
  ✅ = significant (p<0.05) across ALL four specifications
  ⚠️  = significant in SOME but not all specifications
  ❌ = not significant in any specification

  KEY FINDING SUMMARY:
  → Cross-community network position: check if it survives all specs above
  → Pure green: compare Old Mistral (57% green, mostly noise) vs Haiku (8.1% green, precision)
  → ESG_ALIGNED: now 19% of investors — check if they behave like green or traditional
  → Investor quality controls (Step 5): if cross_community survives quality controls,
    the effect is truly about NETWORK POSITION, not just having high-quality investors
""")
