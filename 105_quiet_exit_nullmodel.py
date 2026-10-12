"""
Script 105: Quiet-exit null model and corrections.
Prediction file stays LOCKED (Family I, Q2-Q4 signs unchanged).

1. Fix name rule (single-word on ORIGINAL name, keep HQ-country rule)
2. Null model: expected P(zero climate deals) = (1-p)^n
3. Specialist check (>=5 lifetime deals)
4. Backfill robustness (window ending 2025-06-20)
5. Locked tests Q2, Q3, Q4 + Holm
"""

import os, re, warnings, datetime
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

OUT_DIR = "output/quiet_exit"
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(20261005)

def sep(title=""):
    print(f"\n{'─'*70}")
    if title:
        print(f"  {title}")
        print(f"{'─'*70}")

# ── name handling ───────────────────────────────────────────────────
def clean_name(s):
    s = re.sub(r"\s*\(.*?\)\s*$", "", str(s)).strip()
    suffixes = (r"\b(llc|lp|l\.p\.|llp|ltd|limited|inc|incorporated|corp|"
                r"corporation|plc|gmbh|bv|b\.v\.|sas|sarl|ag|sa|nv|pty|"
                r"fund|partners|capital|ventures|management|group|holdings?"
                r"|advisors?|associates?)\b\.?")
    s = re.sub(suffixes, "", s, flags=re.IGNORECASE).strip().rstrip(",. ")
    return s.lower().strip()

def n_words_original(raw):
    """Word count of the ORIGINAL name (parentheticals already stripped by parser)."""
    return len(str(raw).strip().split())

def country_to_region(c):
    c = str(c).strip()
    americas = {"United States","Canada","Brazil","Mexico","Argentina","Chile",
                "Colombia","Peru","Uruguay","Costa Rica"}
    europe   = {"United Kingdom","Germany","France","Sweden","Netherlands",
                "Switzerland","Spain","Italy","Denmark","Finland","Norway",
                "Belgium","Austria","Portugal","Ireland","Luxembourg",
                "Poland","Czech Republic","Estonia","Latvia","Lithuania"}
    asia     = {"China","India","Japan","South Korea","Singapore","Israel",
                "Hong Kong","Taiwan","Indonesia","Malaysia","Vietnam","Thailand"}
    oceania  = {"Australia","New Zealand"}
    mena     = {"United Arab Emirates","Saudi Arabia","Egypt","Qatar","Kuwait",
                "Bahrain","Jordan","Turkey"}
    africa   = {"South Africa","Kenya","Nigeria","Ghana"}
    if c in americas: return "Americas"
    if c in europe:   return "Europe"
    if c in asia:     return "Asia"
    if c in oceania:  return "Oceania"
    if c in mena:     return "Middle East"
    if c in africa:   return "Africa"
    return "Other"

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip()
            for p in parts if p.strip() and p.strip() != "nan"]

# ── estimation helpers ──────────────────────────────────────────────
def ols_cluster(y, X, clusters=None):
    """OLS; cluster-robust SE if clusters given, else HC3."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    if clusters is not None:
        uniq = pd.unique(clusters)
        meat = np.zeros((p, p))
        for c in uniq:
            m = clusters == c
            s = X[m].T @ resid[m]
            meat += np.outer(s, s)
        G = len(uniq)
        scale = G / (G - 1) * (n - 1) / (n - p)
        V = scale * XtX_inv @ meat @ XtX_inv
        dof = G - 1
    else:
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        w = (resid ** 2) / np.clip((1 - h) ** 2, 1e-12, None)
        meat = X.T @ (X * w[:, None])
        V = XtX_inv @ meat @ XtX_inv
        G = n
        dof = n - p
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return beta, se, n, G, dof

def one_sided_p(t, dof, direction="positive"):
    if direction == "positive":
        return float(stats.t.sf(t, dof))
    return float(stats.t.cdf(t, dof))

def holm(pvals, labels):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(n)
    running = 0.0
    for k, i in enumerate(order):
        a = (n - k) * pvals[i]
        running = max(running, a)
        adj[i] = min(running, 1.0)
    return dict(zip(labels, adj))

def design(df, dv, dummy_cols, fe_col):
    """Build [1, dummies, FE dummies] design matrix."""
    parts = [np.ones((len(df), 1))]
    names = ["const"]
    for c in dummy_cols:
        parts.append(df[[c]].values.astype(float))
        names.append(c)
    fe = pd.get_dummies(df[fe_col].astype(str), prefix="fe", drop_first=True, dtype=float)
    parts.append(fe.values)
    names += list(fe.columns)
    return np.hstack(parts), names, df[dv].values.astype(float)

print("=" * 70)
print("SCRIPT 105 — QUIET EXIT NULL MODEL AND CORRECTIONS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────
sep("Load data")
import sys
sys.path.insert(0, ".")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")
dl = dl.dropna(subset=["Deal Date"]).copy()

rows = []
for _, r in dl.dropna(subset=["Investors"]).iterrows():
    for inv in comma_parse(r["Investors"]):
        rows.append((inv, clean_name(inv), r["Deal Date"], r["Companies"], r["Deal ID"]))
inv_deals = pd.DataFrame(rows, columns=["investor_raw","investor_key","deal_date","company","deal_id"])

PB_COLS = ["Investors","Primary Investor Type","HQ Country/Territory/Region",
           "Last Investment Date","Last Investment Type","Last Investment Company",
           "Total Investments","Total Investments in the last 2 years",
           "Investments","Investments in the last 2 years"]
pb = pd.read_excel("data/Investors_data.xlsx", header=7, usecols=PB_COLS)
pb["_key"] = pb["Investors"].apply(clean_name)
pb = pb[~pb["_key"].duplicated(keep="first")].copy()
pb["Last Investment Date"] = pd.to_datetime(pb["Last Investment Date"], errors="coerce")
pb["pb_region"] = pb["HQ Country/Territory/Region"].apply(country_to_region)
for c in ["Total Investments","Total Investments in the last 2 years",
          "Investments","Investments in the last 2 years"]:
    pb[c] = pd.to_numeric(pb[c], errors="coerce")
pb_idx = pb.set_index("_key")

ip = pd.read_csv("output/investor_profiles.csv")[["investor","geo_focus","community"]].copy()
ip["_key"] = ip["investor"].apply(clean_name)
ip = ip.drop_duplicates("_key")
geo_map  = ip.set_index("_key")["geo_focus"].to_dict()
comm_map = ip.set_index("_key")["community"].to_dict()

ai = pd.read_csv("output/all_investors.csv")[["Investor","green_focus"]].copy()
ai["_key"] = ai["Investor"].apply(clean_name)
gf_map = ai.drop_duplicates("_key").set_index("_key")["green_focus"].to_dict()

nov = pd.read_csv("output/novelty/novelty_scores.csv")[["name","specter_novelty"]]
nov_map = dict(zip(nov["name"], nov["specter_novelty"]))
inv_deals["specter_novelty"] = inv_deals["company"].map(nov_map)

DEALS_MAX = dl["Deal Date"].max()
PB_EXPORT = pd.Timestamp("2026-08-26")
print(f"  Investor-deal pairs: {len(inv_deals):,}")
print(f"  Deals cutoff: {DEALS_MAX.date()} | PB export: {PB_EXPORT.date()}")

# ─────────────────────────────────────────────────────────────────────
# 1. Fixed name rule
# ─────────────────────────────────────────────────────────────────────
sep("1. Name rule fix — single-word test on ORIGINAL name")

uniq = inv_deals[["investor_raw","investor_key"]].drop_duplicates("investor_key").copy()
uniq["in_pb"] = uniq["investor_key"].isin(pb["_key"].values)

# Longest original name observed for each key (a key can map to several raw forms)
raw_words = (inv_deals.assign(w=inv_deals["investor_raw"].apply(n_words_original))
             .groupby("investor_key")["w"].max())
uniq["orig_words"]     = uniq["investor_key"].map(raw_words)
uniq["single_word"]    = uniq["orig_words"] == 1

uniq["our_region"] = uniq["investor_key"].map(geo_map)
uniq["pb_region"]  = uniq["investor_key"].map(pb_idx["pb_region"].to_dict())
both = (uniq["our_region"].notna() & uniq["pb_region"].notna()
        & (uniq["our_region"] != "Other") & (uniq["pb_region"] != "Other"))
uniq["country_conflict"] = both & (uniq["our_region"] != uniq["pb_region"])
uniq["exclude"] = uniq["single_word"] | uniq["country_conflict"]

n_our   = len(uniq)
n_match = int(uniq["in_pb"].sum())
e_sw    = int((uniq["in_pb"] & uniq["single_word"]).sum())
e_ct    = int((uniq["in_pb"] & uniq["country_conflict"] & ~uniq["single_word"]).sum())
n_clean = int((uniq["in_pb"] & ~uniq["exclude"]).sum())

print(f"  Unique investors in deal data          : {n_our:,}")
print(f"  Matched to PitchBook (pre-filter)      : {n_match:,} ({n_match/n_our*100:.1f}%)")
print(f"  Excluded — single-word ORIGINAL name   : {e_sw:,}")
print(f"  Excluded — HQ-country mismatch         : {e_ct:,}")
print(f"  Retained clean matches                 : {n_clean:,} ({n_clean/n_our*100:.1f}%)")
print(f"\n  Script 104 (bug: rule on stripped key) retained 10,300.")
print(f"  Change: {n_clean - 10300:+,} investors recovered.")

ex = uniq[uniq["in_pb"] & uniq["single_word"]].head(10)
print(f"\n  Sample single-word exclusions (original name is genuinely one word):")
print(ex[["investor_raw","orig_words","our_region","pb_region"]].to_string(index=False))

clean_keys = set(uniq.loc[uniq["in_pb"] & ~uniq["exclude"], "investor_key"])
idc = inv_deals[inv_deals["investor_key"].isin(clean_keys)].copy()
for col, src in [("pb_last_inv_date","Last Investment Date"),
                 ("pb_last_type","Last Investment Type"),
                 ("pb_last_co","Last Investment Company"),
                 ("pb_total_inv","Total Investments"),
                 ("pb_tot_2yr","Total Investments in the last 2 years"),
                 ("pb_inv","Investments"),
                 ("pb_inv_2yr","Investments in the last 2 years")]:
    idc[col] = idc["investor_key"].map(pb_idx[src].to_dict())
idc["pb_last_inv_date"] = pd.to_datetime(idc["pb_last_inv_date"], errors="coerce")

# ─────────────────────────────────────────────────────────────────────
# Reusable classifier
# ─────────────────────────────────────────────────────────────────────
def classify(window_end, label=""):
    w2 = window_end - pd.DateOffset(years=2)
    w5 = window_end - pd.DateOffset(years=5)
    inw  = (idc["deal_date"] >= w2) & (idc["deal_date"] <= window_end)
    in5  = (idc["deal_date"] >= w5) & (idc["deal_date"] <= window_end)
    base_keys = set(idc.loc[in5, "investor_key"].unique())
    climate_w = set(idc.loc[inw, "investor_key"].unique())

    per = (idc[idc["investor_key"].isin(base_keys)]
           .groupby("investor_key")
           .agg(pb_last_inv_date=("pb_last_inv_date","first"),
                pb_total_inv=("pb_total_inv","first"),
                pb_tot_2yr=("pb_tot_2yr","first"),
                pb_inv=("pb_inv","first"),
                pb_inv_2yr=("pb_inv_2yr","first"),
                pb_last_type=("pb_last_type","first"),
                pb_last_co=("pb_last_co","first")))

    per["has_climate_w"]  = per.index.isin(climate_w)
    per["active_overall"] = per["pb_last_inv_date"].notna() & (per["pb_last_inv_date"] >= w2)
    per["exit_status"] = np.where(per["has_climate_w"], "stayer",
                          np.where(per["active_overall"], "quiet_exit", "full_exit"))

    # climate deals: prior (before window) and in window
    prior = (idc[idc["investor_key"].isin(base_keys) & (idc["deal_date"] < w2)]
             .groupby("investor_key")["deal_id"].nunique())
    inwin = (idc[idc["investor_key"].isin(base_keys) & inw]
             .groupby("investor_key")["deal_id"].nunique())
    per["prior_climate"]  = per.index.map(prior).fillna(0)
    per["climate_in_win"] = per.index.map(inwin).fillna(0)

    # locked intensity definition
    pos = per["pb_total_inv"] > 0
    per.loc[pos, "intensity"] = (per.loc[pos,"prior_climate"] / per.loc[pos,"pb_total_inv"]).clip(0,1)
    per["intensity_q"] = pd.qcut(per["intensity"].fillna(0), 5,
                                 labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")

    per["green_focus"] = per.index.map(gf_map).fillna("UNKNOWN")
    per["community"]   = per.index.map(comm_map)
    mn = (idc[idc["investor_key"].isin(base_keys) & (idc["deal_date"] < w2)]
          .groupby("investor_key")["specter_novelty"].mean())
    per["portfolio_novelty"] = per.index.map(mn)

    if label:
        c = per["exit_status"].value_counts()
        tot = len(per)
        print(f"\n  [{label}] window {w2.date()} → {window_end.date()} | base N = {tot:,}")
        print(f"    {'status':<12}{'N':>8}{'share':>9}")
        for s in ["stayer","quiet_exit","full_exit"]:
            print(f"    {s:<12}{c.get(s,0):>8,}{c.get(s,0)/tot*100:>8.1f}%")
    return per, w2, w5

print("\n  Reclassifying on corrected sample:")
base, W2, W5 = classify(DEALS_MAX, label="PRIMARY")
TOTAL = len(base)

# ─────────────────────────────────────────────────────────────────────
# 2. Null model
# ─────────────────────────────────────────────────────────────────────
sep("2. Null model — expected P(zero climate deals) = (1-p)^n")

act = base[base["exit_status"].isin(["stayer","quiet_exit"])].copy()
print(f"  Active-overall investors (stayer + quiet_exit): {len(act):,}")

# n coverage
print(f"\n  Coverage of window deal-count (n):")
for nm, col in [("Total Investments in last 2 years","pb_tot_2yr"),
                ("Investments in last 2 years","pb_inv_2yr")]:
    cov = act[col].notna().sum()
    print(f"    {nm:<36}: {cov:,}/{len(act):,} ({cov/len(act)*100:.1f}%)")

def build_null(df, tot_col, win_col, tag):
    d = df.copy()
    d["n_win"] = d[win_col]
    d["prior_overall"] = d[tot_col] - d[win_col]
    # p requires prior overall activity
    ok = d["n_win"].notna() & (d["n_win"] >= 1) & d["prior_overall"].notna() & (d["prior_overall"] >= 1)
    d["p_clim"] = np.nan
    d.loc[ok, "p_clim"] = (d.loc[ok,"prior_climate"] / d.loc[ok,"prior_overall"]).clip(0, 1)
    d["expected_quiet"] = np.where(ok, (1 - d["p_clim"]) ** d["n_win"], np.nan)
    d["valid_null"] = ok & d["expected_quiet"].notna()
    n_clip = int(((d.loc[ok,"prior_climate"] / d.loc[ok,"prior_overall"]) > 1).sum())
    print(f"\n  [{tag}] estimable: {int(d['valid_null'].sum()):,}/{len(d):,} "
          f"({d['valid_null'].mean()*100:.1f}%) | p clipped at 1: {n_clip:,}")
    return d

act = build_null(act, "pb_total_inv", "pb_tot_2yr", "PRIMARY (Total Investments family)")
act_narrow = build_null(
    base[base["exit_status"].isin(["stayer","quiet_exit"])].copy(),
    "pb_inv", "pb_inv_2yr", "SENSITIVITY (Investments family)")

def obs_exp(d, B=999):
    v = d[d["valid_null"]]
    if len(v) < 10:
        return dict(n=len(v), obs=np.nan, exp=np.nan, diff=np.nan,
                    lo=np.nan, hi=np.nan, dlo=np.nan, dhi=np.nan)
    o = float((v["exit_status"] == "quiet_exit").mean())
    e = float(v["expected_quiet"].mean())
    oi = (v["exit_status"] == "quiet_exit").values.astype(float)
    ei = v["expected_quiet"].values
    bo, be, bd = [], [], []
    idxs = np.arange(len(v))
    for _ in range(B):
        s = RNG.choice(idxs, size=len(v), replace=True)
        bo.append(oi[s].mean()); be.append(ei[s].mean())
        bd.append(oi[s].mean() - ei[s].mean())
    return dict(n=len(v), obs=o, exp=e, diff=o - e,
                lo=float(np.percentile(bo,2.5)),  hi=float(np.percentile(bo,97.5)),
                elo=float(np.percentile(be,2.5)), ehi=float(np.percentile(be,97.5)),
                dlo=float(np.percentile(bd,2.5)), dhi=float(np.percentile(bd,97.5)))

ov = obs_exp(act)
print(f"\n  OVERALL (active-overall investors, N={ov['n']:,})")
print(f"    observed quiet-exit share : {ov['obs']:.3f}  [{ov['lo']:.3f}, {ov['hi']:.3f}]")
print(f"    expected under null       : {ov['exp']:.3f}  [{ov['elo']:.3f}, {ov['ehi']:.3f}]")
print(f"    excess (obs - exp)        : {ov['diff']:+.3f} [{ov['dlo']:+.3f}, {ov['dhi']:+.3f}]")
if ov["dlo"] > 0:
    print(f"    → ABOVE null: more quiet exit than chance allocation predicts")
elif ov["dhi"] < 0:
    print(f"    → BELOW null: LESS quiet exit than chance allocation predicts")
    print(f"      (investors are more persistent in climate than random reallocation implies)")
else:
    print(f"    → consistent with null (CI contains 0)")

ovn = obs_exp(act_narrow)
print(f"\n  SENSITIVITY (narrow 'Investments' family, N={ovn['n']:,})")
print(f"    observed {ovn['obs']:.3f} | expected {ovn['exp']:.3f} | "
      f"excess {ovn['diff']:+.3f} [{ovn['dlo']:+.3f}, {ovn['dhi']:+.3f}]")

def fmt_row(level, r, width):
    ci = f"[{r['dlo']:+.3f}, {r['dhi']:+.3f}]"
    return (f"    {level:<{width}}{r['n']:>7,}{r['obs']:>8.3f}{r['exp']:>8.3f}"
            f"{r['diff']:>+9.3f}{ci:>22}")

rows_g = []
print(f"\n  By intensity quintile:")
print(f"    {'q':<5}{'N':>7}{'obs':>8}{'exp':>8}{'excess':>9}{'95% CI (excess)':>22}")
for q in ["Q1","Q2","Q3","Q4","Q5"]:
    r = obs_exp(act[act["intensity_q"] == q])
    if np.isnan(r["obs"]): continue
    print(fmt_row(q, r, 5))
    rows_g.append(dict(group="intensity_q", level=q, **r))

print(f"\n  By green_focus:")
print(f"    {'level':<14}{'N':>7}{'obs':>8}{'exp':>8}{'excess':>9}{'95% CI (excess)':>22}")
for g in ["GREEN_VC","TRADITIONAL","ESG_ALIGNED"]:
    r = obs_exp(act[act["green_focus"] == g])
    if np.isnan(r["obs"]): continue
    print(fmt_row(g, r, 14))
    rows_g.append(dict(group="green_focus", level=g, **r))

pd.DataFrame([dict(group="overall", level="all", **ov)] + rows_g).to_csv(
    f"{OUT_DIR}/null_model.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 3. Specialist check
# ─────────────────────────────────────────────────────────────────────
sep("3. Specialist check — investors with >=5 lifetime deals")

spec = base[base["pb_total_inv"] >= 5].copy()
print(f"  Base population with >=5 lifetime (Total Investments) deals: "
      f"{len(spec):,}/{TOTAL:,} ({len(spec)/TOTAL*100:.1f}%)")

cs = spec["exit_status"].value_counts()
print(f"\n  {'status':<12}{'N':>8}{'share':>9}")
for s in ["stayer","quiet_exit","full_exit"]:
    print(f"  {s:<12}{cs.get(s,0):>8,}{cs.get(s,0)/len(spec)*100:>8.1f}%")

sq = spec.groupby(["intensity_q","exit_status"]).size().unstack(fill_value=0)
sq["N"] = sq.sum(axis=1)
for c in ["stayer","quiet_exit","full_exit"]:
    if c in sq.columns:
        sq[f"sh_{c}"] = (sq[c] / sq["N"]).round(3)
print(f"\n  Exit status by intensity quintile (>=5 lifetime deals):")
print(sq.to_string())
sq.to_csv(f"{OUT_DIR}/specialist_quintiles.csv")

# Null model within specialists
spec_act = build_null(spec[spec["exit_status"].isin(["stayer","quiet_exit"])].copy(),
                      "pb_total_inv", "pb_tot_2yr", "SPECIALISTS")
rs = obs_exp(spec_act)
print(f"  Specialists: observed {rs['obs']:.3f} | expected {rs['exp']:.3f} | "
      f"excess {rs['diff']:+.3f} [{rs['dlo']:+.3f}, {rs['dhi']:+.3f}]")

# ─────────────────────────────────────────────────────────────────────
# 4. Backfill robustness
# ─────────────────────────────────────────────────────────────────────
sep("4. Backfill robustness — window ending 2025-06-20")

print("  Rationale: deal records for the most recent months are incompletely")
print("  backfilled (2026-06 has 158 deals vs ~300 typical), which would")
print("  inflate quiet-exit by making recent climate deals invisible.")

base_bf, W2b, W5b = classify(pd.Timestamp("2025-06-20"), label="BACKFILL")

cmp_rows = []
for nm, d in [("primary (end 2026-06-20)", base), ("backfill (end 2025-06-20)", base_bf)]:
    c = d["exit_status"].value_counts()
    cmp_rows.append(dict(spec=nm, N=len(d),
                         stayer=c.get("stayer",0), quiet_exit=c.get("quiet_exit",0),
                         full_exit=c.get("full_exit",0),
                         qe_share=round(c.get("quiet_exit",0)/len(d),3)))
cmp_df = pd.DataFrame(cmp_rows)
print(f"\n  Comparison:")
print(cmp_df.to_string(index=False))

bq = base_bf.groupby(["intensity_q","exit_status"]).size().unstack(fill_value=0)
bq["N"] = bq.sum(axis=1)
if "quiet_exit" in bq.columns:
    bq["sh_quiet_exit"] = (bq["quiet_exit"] / bq["N"]).round(3)
print(f"\n  Backfill: exit status by intensity quintile:")
print(bq.to_string())

bf_act = build_null(base_bf[base_bf["exit_status"].isin(["stayer","quiet_exit"])].copy(),
                    "pb_total_inv", "pb_tot_2yr", "BACKFILL")
rb = obs_exp(bf_act)
print(f"  Backfill null model: observed {rb['obs']:.3f} | expected {rb['exp']:.3f} | "
      f"excess {rb['diff']:+.3f} [{rb['dlo']:+.3f}, {rb['dhi']:+.3f}]")
print(f"  NOTE: PB 'last 2 years' counts are fixed to the 2026-08-26 export and do")
print(f"  not shift with the backfill window, so the backfill null model is")
print(f"  approximate (n is measured over a later window than p).")
cmp_df.to_csv(f"{OUT_DIR}/backfill_comparison.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 5. Locked tests
# ─────────────────────────────────────────────────────────────────────
sep("5. LOCKED TESTS on corrected sample (Q2, Q3, Q4)")

results = {}

# ---------- Q2 ----------
print("  Q2 — quiet exiters funded MORE novel ventures than stayers (sign: POSITIVE)")
q2 = base[base["exit_status"].isin(["quiet_exit","stayer"])].copy()
q2 = q2[q2["portfolio_novelty"].notna() & q2["intensity_q"].notna()]
q2["quiet"] = (q2["exit_status"] == "quiet_exit").astype(float)

print(f"    N = {len(q2):,} (quiet_exit {int(q2['quiet'].sum()):,} | "
      f"stayer {int((1-q2['quiet']).sum()):,})")
print(f"    raw means — quiet_exit {q2.loc[q2['quiet']==1,'portfolio_novelty'].mean():.4f} | "
      f"stayer {q2.loc[q2['quiet']==0,'portfolio_novelty'].mean():.4f}")

q2c = q2[q2["community"].notna()].copy()
X, nms, y = design(q2c, "portfolio_novelty", ["quiet"], "intensity_q")
b, se, n, G, dof = ols_cluster(y, X, q2c["community"].values)
i = nms.index("quiet"); t = b[i]/se[i]
p_q2 = one_sided_p(t, dof, "positive")
print(f"    clustered by community (N={n:,}, G={G}): beta = {b[i]:+.5f} "
      f"(SE {se[i]:.5f}), t = {t:+.3f}, one-sided p = {p_q2:.4f}")

Xa, nmsa, ya = design(q2, "portfolio_novelty", ["quiet"], "intensity_q")
ba, sea, na, Ga, dofa = ols_cluster(ya, Xa, None)
ia = nmsa.index("quiet"); ta = ba[ia]/sea[ia]
print(f"    HC3 full sample (N={na:,}):              beta = {ba[ia]:+.5f} "
      f"(SE {sea[ia]:.5f}), t = {ta:+.3f}, one-sided p = {one_sided_p(ta,dofa,'positive'):.4f}")
results["Q2"] = dict(beta=b[i], se=se[i], t=t, p=p_q2, n=n,
                     pred="positive", obs_sign="positive" if b[i]>0 else "negative")

# ---------- Q3 ----------
print("\n  Q3 — GREEN_VC quiet-exit MORE than TRADITIONAL, within intensity quintiles")
print("       (sign: POSITIVE; primary DV = null-adjusted excess)")
q3 = act[act["valid_null"] & act["intensity_q"].notna()].copy()
q3 = q3[q3["green_focus"].isin(["GREEN_VC","TRADITIONAL","ESG_ALIGNED"])]
q3["quiet"]  = (q3["exit_status"] == "quiet_exit").astype(float)
q3["excess"] = q3["quiet"] - q3["expected_quiet"]
q3["GREEN_VC"]    = (q3["green_focus"] == "GREEN_VC").astype(float)
q3["ESG_ALIGNED"] = (q3["green_focus"] == "ESG_ALIGNED").astype(float)

print(f"    N = {len(q3):,} | GREEN_VC {int(q3['GREEN_VC'].sum()):,} | "
      f"TRADITIONAL {int((q3['green_focus']=='TRADITIONAL').sum()):,} | "
      f"ESG {int(q3['ESG_ALIGNED'].sum()):,}")
print(f"    mean excess — GREEN_VC {q3.loc[q3['GREEN_VC']==1,'excess'].mean():+.4f} | "
      f"TRADITIONAL {q3.loc[q3['green_focus']=='TRADITIONAL','excess'].mean():+.4f}")

q3c = q3[q3["community"].notna()].copy()
X3, nm3, y3 = design(q3c, "excess", ["GREEN_VC","ESG_ALIGNED"], "intensity_q")
b3, se3, n3, G3, dof3 = ols_cluster(y3, X3, q3c["community"].values)
j = nm3.index("GREEN_VC"); t3 = b3[j]/se3[j]
p_q3 = one_sided_p(t3, dof3, "positive")
print(f"    PRIMARY (null-adjusted, clustered, N={n3:,}, G={G3}): "
      f"beta_GREEN_VC = {b3[j]:+.5f} (SE {se3[j]:.5f}), t = {t3:+.3f}, "
      f"one-sided p = {p_q3:.4f}")

# secondary: raw binary DV (as originally specified before null adjustment)
X3b, nm3b, y3b = design(q3c, "quiet", ["GREEN_VC","ESG_ALIGNED"], "intensity_q")
b3b, se3b, n3b, G3b, dof3b = ols_cluster(y3b, X3b, q3c["community"].values)
jb = nm3b.index("GREEN_VC"); t3b = b3b[jb]/se3b[jb]
print(f"    SECONDARY (raw binary LPM, clustered):  "
      f"beta_GREEN_VC = {b3b[jb]:+.5f} (SE {se3b[jb]:.5f}), t = {t3b:+.3f}, "
      f"one-sided p = {one_sided_p(t3b,dof3b,'positive'):.4f}")
results["Q3"] = dict(beta=b3[j], se=se3[j], t=t3, p=p_q3, n=n3,
                     pred="positive", obs_sign="positive" if b3[j]>0 else "negative")

# ---------- Q4 ----------
print("\n  Q4 — quiet exits concentrated in particular communities (two-sided permutation)")
cc = base["community"].value_counts()
keep = set(cc[cc >= 30].index)
q4 = base[base["community"].isin(keep)].copy()
q4["quiet"] = (q4["exit_status"] == "quiet_exit").astype(float)
overall = q4["quiet"].mean()
obs_stat = (q4.groupby("community")["quiet"].mean() - overall).max()
print(f"    communities with >=30 base investors: {len(keep)} | N = {len(q4):,}")
print(f"    overall quiet-exit share: {overall:.3f}")
print(f"    observed statistic max(community share) - overall = {obs_stat:+.4f}")

lab = q4["quiet"].values.copy()
codes = pd.Categorical(q4["community"]).codes
K = codes.max() + 1
cnt = np.bincount(codes, minlength=K)
B = 999
perm = np.empty(B)
for bi in range(B):
    sh = RNG.permutation(lab)
    perm[bi] = (np.bincount(codes, weights=sh, minlength=K) / cnt - sh.mean()).max()
p_q4 = (1 + int((perm >= obs_stat).sum())) / (B + 1)
print(f"    permutation null: mean {perm.mean():+.4f}, 95th pct {np.percentile(perm,95):+.4f}")
print(f"    p = {p_q4:.4f}  ({int((perm >= obs_stat).sum())}/{B} draws >= observed)")

top = (q4.groupby("community")["quiet"].agg(["mean","size"])
       .sort_values("mean", ascending=False).head(5))
print(f"    highest quiet-exit communities:")
print(top.round(3).to_string())
results["Q4"] = dict(beta=obs_stat, se=np.nan, t=np.nan, p=p_q4, n=len(q4),
                     pred="two-sided", obs_sign="n/a")

# ---------- Holm ----------
sep("Holm correction across Q2, Q3, Q4")
labels = ["Q2","Q3","Q4"]
praw = [results[k]["p"] for k in labels]
padj = holm(praw, labels)

print(f"  {'test':<6}{'estimate':>12}{'p_raw':>10}{'p_Holm':>10}   {'pred':<12}{'verdict':<28}")
print(f"  {'─'*80}")
verdicts = {}
for k in labels:
    r = results[k]
    pa = padj[k]
    if k == "Q4":
        v = "SUPPORTED" if pa < 0.05 else "NOT SUPPORTED"
    else:
        if r["obs_sign"] != r["pred"]:
            v = f"FALSIFIED (sign is {r['obs_sign']})"
        elif pa < 0.05:
            v = "SUPPORTED"
        else:
            v = "NOT SUPPORTED (ns)"
    verdicts[k] = v
    print(f"  {k:<6}{r['beta']:>+12.5f}{r['p']:>10.4f}{pa:>10.4f}   "
          f"{r['pred']:<12}{v:<28}")

pd.DataFrame([dict(test=k, **results[k], p_holm=padj[k], verdict=verdicts[k])
              for k in labels]).to_csv(f"{OUT_DIR}/locked_test_results.csv", index=False)

base.reset_index().to_csv(f"{OUT_DIR}/base_population_v105.csv", index=False)
act.reset_index().to_csv(f"{OUT_DIR}/active_with_null.csv", index=False)

sep("Summary")
print(f"  Corrected base population: {TOTAL:,} "
      f"(was 7,751 under the Script 104 name bug)")
print(f"  Quiet-exit share: {(base['exit_status']=='quiet_exit').mean()*100:.1f}%")
print(f"  Null model: observed {ov['obs']:.3f} vs expected {ov['exp']:.3f} "
      f"(excess {ov['diff']:+.3f} [{ov['dlo']:+.3f}, {ov['dhi']:+.3f}])")
print(f"  Q2 {verdicts['Q2']}")
print(f"  Q3 {verdicts['Q3']}")
print(f"  Q4 {verdicts['Q4']}")
print("\n" + "=" * 70)
print("SCRIPT 105 COMPLETE — predictions were locked before these results")
print("=" * 70)
