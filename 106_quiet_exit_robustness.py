"""
Script 106: Rule out mechanical explanations for the Q5 reversal and Q4.
Locked predictions unchanged. Everything here is declared robustness.

1. Non-overlapping base population (base period strictly before outcome window)
2. Empirical-Bayes shrinkage of the prior climate share p
3. Q4 beyond geography: community 19 identity + within-country permutation
4. Coverage check: deals per year, community 19 vs others
"""

import os, re, warnings
import numpy as np
import pandas as pd
from scipy.special import betaln
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

OUT_DIR = "output/quiet_exit"
os.makedirs(OUT_DIR, exist_ok=True)
RNG = np.random.default_rng(20261006)

def sep(title=""):
    print(f"\n{'─'*70}")
    if title:
        print(f"  {title}")
        print(f"{'─'*70}")

# ── shared helpers (identical to Script 105) ────────────────────────
def clean_name(s):
    s = re.sub(r"\s*\(.*?\)\s*$", "", str(s)).strip()
    suffixes = (r"\b(llc|lp|l\.p\.|llp|ltd|limited|inc|incorporated|corp|"
                r"corporation|plc|gmbh|bv|b\.v\.|sas|sarl|ag|sa|nv|pty|"
                r"fund|partners|capital|ventures|management|group|holdings?"
                r"|advisors?|associates?)\b\.?")
    s = re.sub(suffixes, "", s, flags=re.IGNORECASE).strip().rstrip(",. ")
    return s.lower().strip()

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

def n_words_original(raw):
    return len(str(raw).strip().split())

def boot_obs_exp(d, exp_col="expected_quiet", B=999):
    v = d[d[exp_col].notna()]
    if len(v) < 10:
        return dict(n=len(v), obs=np.nan, exp=np.nan, diff=np.nan,
                    dlo=np.nan, dhi=np.nan)
    oi = (v["exit_status"] == "quiet_exit").values.astype(float)
    ei = v[exp_col].values
    idxs = np.arange(len(v))
    bd = np.empty(B)
    for i in range(B):
        s = RNG.choice(idxs, size=len(v), replace=True)
        bd[i] = oi[s].mean() - ei[s].mean()
    return dict(n=len(v), obs=float(oi.mean()), exp=float(ei.mean()),
                diff=float(oi.mean() - ei.mean()),
                dlo=float(np.percentile(bd, 2.5)),
                dhi=float(np.percentile(bd, 97.5)))

def print_oe_table(title, rows, width=6):
    print(f"\n  {title}")
    print(f"    {'level':<{width}}{'N':>7}{'obs':>8}{'exp':>8}{'excess':>9}"
          f"{'95% CI (excess)':>22}")
    for lvl, r in rows:
        ci = f"[{r['dlo']:+.3f}, {r['dhi']:+.3f}]"
        print(f"    {lvl:<{width}}{r['n']:>7,}{r['obs']:>8.3f}{r['exp']:>8.3f}"
              f"{r['diff']:>+9.3f}{ci:>22}")

print("=" * 70)
print("SCRIPT 106 — MECHANICAL ROBUSTNESS (declared, not pre-registered)")
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
inv_deals = pd.DataFrame(rows, columns=["investor_raw","investor_key",
                                        "deal_date","company","deal_id"])

PB_COLS = ["Investors","Primary Investor Type","HQ Country/Territory/Region",
           "Last Investment Date","Total Investments",
           "Total Investments in the last 2 years"]
pb = pd.read_excel("data/Investors_data.xlsx", header=7, usecols=PB_COLS)
pb["_key"] = pb["Investors"].apply(clean_name)
pb = pb[~pb["_key"].duplicated(keep="first")].copy()
pb["Last Investment Date"] = pd.to_datetime(pb["Last Investment Date"], errors="coerce")
pb["pb_region"] = pb["HQ Country/Territory/Region"].apply(country_to_region)
for c in ["Total Investments","Total Investments in the last 2 years"]:
    pb[c] = pd.to_numeric(pb[c], errors="coerce")
pb_idx = pb.set_index("_key")

ip = pd.read_csv("output/investor_profiles.csv")[["investor","geo_focus","community"]]
ip = ip.copy(); ip["_key"] = ip["investor"].apply(clean_name)
ip = ip.drop_duplicates("_key")
geo_map  = ip.set_index("_key")["geo_focus"].to_dict()
comm_map = ip.set_index("_key")["community"].to_dict()

ai = pd.read_csv("output/all_investors.csv")[["Investor","green_focus"]].copy()
ai["_key"] = ai["Investor"].apply(clean_name)
gf_map = ai.drop_duplicates("_key").set_index("_key")["green_focus"].to_dict()

# ── apply the corrected name filter (Script 105 rule) ───────────────
uniq = inv_deals[["investor_raw","investor_key"]].drop_duplicates("investor_key").copy()
uniq["in_pb"] = uniq["investor_key"].isin(pb["_key"].values)
raw_words = (inv_deals.assign(w=inv_deals["investor_raw"].apply(n_words_original))
             .groupby("investor_key")["w"].max())
uniq["single_word"] = uniq["investor_key"].map(raw_words) == 1
uniq["our_region"]  = uniq["investor_key"].map(geo_map)
uniq["pb_region"]   = uniq["investor_key"].map(pb_idx["pb_region"].to_dict())
both = (uniq["our_region"].notna() & uniq["pb_region"].notna()
        & (uniq["our_region"] != "Other") & (uniq["pb_region"] != "Other"))
uniq["country_conflict"] = both & (uniq["our_region"] != uniq["pb_region"])
clean_keys = set(uniq.loc[uniq["in_pb"] & ~(uniq["single_word"] | uniq["country_conflict"]),
                          "investor_key"])
idc = inv_deals[inv_deals["investor_key"].isin(clean_keys)].copy()
print(f"  Investor-deal pairs: {len(inv_deals):,} | clean-matched investors: {len(clean_keys):,}")

# ─────────────────────────────────────────────────────────────────────
# 1. Non-overlapping base population
# ─────────────────────────────────────────────────────────────────────
sep("1. Non-overlapping base population")

BASE_START = pd.Timestamp("2021-06-20")
CUT        = pd.Timestamp("2024-06-20")   # base ends / outcome begins
OUT_END    = pd.Timestamp("2026-06-20")

print(f"  Base period    : {BASE_START.date()} → {CUT.date()} (exclusive)")
print(f"  Outcome window : {CUT.date()} → {OUT_END.date()}")
print(f"  p estimated from deals strictly before {CUT.date()}")
print(f"\n  Script 105 flaw this fixes: the 5-yr base window OVERLAPPED the")
print(f"  outcome window, so an investor could enter the base population via a")
print(f"  deal inside the outcome window — which forced stayer status and drove")
print(f"  the Q1 excess of -0.744.")

in_base = (idc["deal_date"] >= BASE_START) & (idc["deal_date"] < CUT)
in_out  = (idc["deal_date"] >= CUT) & (idc["deal_date"] <= OUT_END)

base_keys  = set(idc.loc[in_base, "investor_key"].unique())
stayer_set = set(idc.loc[in_out,  "investor_key"].unique())

per = (idc[idc["investor_key"].isin(base_keys)]
       .groupby("investor_key")
       .agg(pb_last_inv_date=("investor_key", lambda s: np.nan)))   # placeholder
per = pd.DataFrame(index=sorted(base_keys))
per["pb_last_inv_date"] = per.index.map(pb_idx["Last Investment Date"].to_dict())
per["pb_total_inv"]     = per.index.map(pb_idx["Total Investments"].to_dict())
per["pb_tot_2yr"]       = per.index.map(pb_idx["Total Investments in the last 2 years"].to_dict())
per["hq_country"]       = per.index.map(pb_idx["HQ Country/Territory/Region"].to_dict())
per["pb_last_inv_date"] = pd.to_datetime(per["pb_last_inv_date"], errors="coerce")

per["has_climate_out"]  = per.index.isin(stayer_set)
per["active_overall"]   = per["pb_last_inv_date"].notna() & (per["pb_last_inv_date"] >= CUT)
per["exit_status"] = np.where(per["has_climate_out"], "stayer",
                      np.where(per["active_overall"], "quiet_exit", "full_exit"))

# prior climate deals: everything strictly before CUT
prior_c = (idc[idc["investor_key"].isin(base_keys) & (idc["deal_date"] < CUT)]
           .groupby("investor_key")["deal_id"].nunique())
per["prior_climate"] = per.index.map(prior_c).fillna(0)

# prior overall deals from PitchBook lifetime minus window
per["prior_overall"] = per["pb_total_inv"] - per["pb_tot_2yr"]
per["prior_overall"] = per[["prior_overall","prior_climate"]].max(axis=1)
per["n_win"]         = per["pb_tot_2yr"]

# locked intensity definition for quintiles
pos = per["pb_total_inv"] > 0
per.loc[pos, "intensity"] = (per.loc[pos,"prior_climate"]
                             / per.loc[pos,"pb_total_inv"]).clip(0,1)
per["intensity_q"] = pd.qcut(per["intensity"].fillna(0), 5,
                             labels=["Q1","Q2","Q3","Q4","Q5"], duplicates="drop")
per["green_focus"] = per.index.map(gf_map).fillna("UNKNOWN")
per["community"]   = per.index.map(comm_map)

cnt = per["exit_status"].value_counts(); TOT = len(per)
print(f"\n  Base population N = {TOT:,}  (Script 105 overlapping base: 10,564)")
print(f"    {'status':<12}{'N':>8}{'share':>9}")
for s in ["stayer","quiet_exit","full_exit"]:
    print(f"    {s:<12}{cnt.get(s,0):>8,}{cnt.get(s,0)/TOT*100:>8.1f}%")

# raw (unshrunk) null
ok = (per["n_win"].notna() & (per["n_win"] >= 1)
      & per["prior_overall"].notna() & (per["prior_overall"] >= 1))
per["p_raw"] = np.nan
per.loc[ok, "p_raw"] = (per.loc[ok,"prior_climate"] / per.loc[ok,"prior_overall"]).clip(0,1)
per["expected_quiet"] = np.where(ok, (1 - per["p_raw"]) ** per["n_win"], np.nan)

act = per[per["exit_status"].isin(["stayer","quiet_exit"])].copy()
print(f"\n  Active-overall investors: {len(act):,}")
print(f"  Null estimable: {int(act['expected_quiet'].notna().sum()):,}"
      f" ({act['expected_quiet'].notna().mean()*100:.1f}%)")
print(f"  Investors with p = 0 (degenerate in Script 105): "
      f"{int((act['p_raw'] == 0).sum()):,} — base period now guarantees p > 0")

ov1 = boot_obs_exp(act)
print(f"\n  OVERALL (non-overlapping base, N={ov1['n']:,})")
print(f"    observed {ov1['obs']:.3f} | expected {ov1['exp']:.3f} | "
      f"excess {ov1['diff']:+.3f} [{ov1['dlo']:+.3f}, {ov1['dhi']:+.3f}]")

rows1 = [(q, boot_obs_exp(act[act["intensity_q"] == q])) for q in ["Q1","Q2","Q3","Q4","Q5"]]
rows1 = [(q, r) for q, r in rows1 if not np.isnan(r["obs"])]
print_oe_table("By intensity quintile (raw p, non-overlapping base):", rows1)
print(f"\n    Script 105 Q1 excess was -0.744; Q5 was +0.252.")

# ─────────────────────────────────────────────────────────────────────
# 2. Empirical-Bayes shrinkage
# ─────────────────────────────────────────────────────────────────────
sep("2. Empirical-Bayes shrinkage of p")

fit = act[ok.reindex(act.index, fill_value=False)].copy()
k = fit["prior_climate"].values.astype(float)
n = fit["prior_overall"].values.astype(float)
k = np.minimum(k, n)

def bb_nll(params, k, n):
    a, b = np.exp(params)
    return -np.sum(betaln(k + a, n - k + b) - betaln(a, b))

res = minimize(bb_nll, x0=np.array([0.0, 0.0]), args=(k, n), method="Nelder-Mead",
               options=dict(maxiter=4000, xatol=1e-7, fatol=1e-7))
alpha, beta_p = np.exp(res.x)
print(f"  Beta-binomial prior fitted by MLE on {len(k):,} investors")
print(f"    alpha = {alpha:.4f}, beta = {beta_p:.4f}")
print(f"    prior mean = {alpha/(alpha+beta_p):.4f}, "
       f"prior 'pseudo-deals' = {alpha+beta_p:.2f}")
print(f"    converged: {res.success}")

per["p_eb"] = np.nan
m = ok
per.loc[m, "p_eb"] = ((per.loc[m,"prior_climate"].clip(upper=per.loc[m,"prior_overall"]) + alpha)
                      / (per.loc[m,"prior_overall"] + alpha + beta_p))
per["expected_quiet_eb"] = np.where(m, (1 - per["p_eb"]) ** per["n_win"], np.nan)

act = per[per["exit_status"].isin(["stayer","quiet_exit"])].copy()

print(f"\n  p shifted by shrinkage: raw mean {act['p_raw'].mean():.4f} → "
      f"EB mean {act['p_eb'].mean():.4f}")

ov2 = boot_obs_exp(act, "expected_quiet_eb")
print(f"\n  OVERALL (EB-shrunk p, N={ov2['n']:,})")
print(f"    observed {ov2['obs']:.3f} | expected {ov2['exp']:.3f} | "
      f"excess {ov2['diff']:+.3f} [{ov2['dlo']:+.3f}, {ov2['dhi']:+.3f}]")

rows2 = [(q, boot_obs_exp(act[act["intensity_q"] == q], "expected_quiet_eb"))
         for q in ["Q1","Q2","Q3","Q4","Q5"]]
rows2 = [(q, r) for q, r in rows2 if not np.isnan(r["obs"])]
print_oe_table("By intensity quintile (EB-shrunk p):", rows2)

# restricted to >=10 prior deals
act10 = act[act["prior_overall"] >= 10].copy()
print(f"\n  Restricted to investors with >=10 prior deals: {len(act10):,}"
      f" / {len(act):,} ({len(act10)/len(act)*100:.1f}%)")
ov3 = boot_obs_exp(act10, "expected_quiet_eb")
print(f"    OVERALL: observed {ov3['obs']:.3f} | expected {ov3['exp']:.3f} | "
      f"excess {ov3['diff']:+.3f} [{ov3['dlo']:+.3f}, {ov3['dhi']:+.3f}]")
rows3 = [(q, boot_obs_exp(act10[act10["intensity_q"] == q], "expected_quiet_eb"))
         for q in ["Q1","Q2","Q3","Q4","Q5"]]
rows3 = [(q, r) for q, r in rows3 if not np.isnan(r["obs"])]
print_oe_table("By intensity quintile (EB p, >=10 prior deals):", rows3)

pd.DataFrame(
    [dict(spec="nonoverlap_raw",  level="overall", **ov1)] +
    [dict(spec="nonoverlap_raw",  level=q, **r) for q, r in rows1] +
    [dict(spec="eb",              level="overall", **ov2)] +
    [dict(spec="eb",              level=q, **r) for q, r in rows2] +
    [dict(spec="eb_ge10",         level="overall", **ov3)] +
    [dict(spec="eb_ge10",         level=q, **r) for q, r in rows3]
).to_csv(f"{OUT_DIR}/null_model_robustness.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 3. Q4 beyond geography
# ─────────────────────────────────────────────────────────────────────
sep("3. Q4 beyond geography — community 19")

try:
    cn = pd.read_csv("output/community_named.csv")
    r19 = cn[cn["community_id"] == 19]
    if len(r19):
        print(f"  community_named.csv label for 19: "
              f"{r19.iloc[0].get('cluster_label','?')} | "
              f"stage={r19.iloc[0].get('top_stage','?')} | "
              f"sector={r19.iloc[0].get('top_sector','?')} | "
              f"geo={r19.iloc[0].get('top_geo','?')}")
except Exception as e:
    print(f"  (community_named.csv unavailable: {e})")

c19 = per[per["community"] == 19].copy()
print(f"\n  Community 19 base-population investors: {len(c19):,}")
print(f"  Quiet-exit rate: {(c19['exit_status']=='quiet_exit').mean():.3f} "
      f"vs {(per['exit_status']=='quiet_exit').mean():.3f} overall")

print(f"\n  HQ country distribution (top 10):")
hq19 = c19["hq_country"].value_counts(dropna=False).head(10)
for cty, n_ in hq19.items():
    print(f"    {str(cty):<28}{n_:>6,}  ({n_/len(c19)*100:>5.1f}%)")

print(f"\n  Overall base population HQ (top 5 for contrast):")
for cty, n_ in per["hq_country"].value_counts(dropna=False).head(5).items():
    print(f"    {str(cty):<28}{n_:>6,}  ({n_/len(per)*100:>5.1f}%)")

deal_counts = (idc[idc["investor_key"].isin(c19.index)]
               .groupby("investor_key")["deal_id"].nunique()
               .sort_values(ascending=False))
print(f"\n  Top 15 community-19 investors by climate deal count:")
raw_lookup = (inv_deals.drop_duplicates("investor_key")
              .set_index("investor_key")["investor_raw"].to_dict())
for key, dc in deal_counts.head(15).items():
    print(f"    {str(raw_lookup.get(key,key))[:40]:<42}{dc:>5} deals  "
          f"{str(c19.loc[key,'hq_country'])[:18]:<20}{c19.loc[key,'exit_status']}")

print(f"\n  green_focus in community 19:")
for g, n_ in c19["green_focus"].value_counts().items():
    print(f"    {g:<16}{n_:>6,}  ({n_/len(c19)*100:>5.1f}%)")

# ── permutation tests ───────────────────────────────────────────────
cc = per["community"].value_counts()
keep = set(cc[cc >= 30].index)
q4 = per[per["community"].isin(keep)].copy()
q4["quiet"] = (q4["exit_status"] == "quiet_exit").astype(float)
codes = pd.Categorical(q4["community"]).codes
K = codes.max() + 1
grp_n = np.bincount(codes, minlength=K)
lab = q4["quiet"].values.astype(float)
overall = lab.mean()
obs_stat = (np.bincount(codes, weights=lab, minlength=K) / grp_n - overall).max()

print(f"\n  Communities with >=30 base investors: {len(keep)} | N = {len(q4):,}")
print(f"  Overall quiet-exit share: {overall:.3f}")
print(f"  Observed statistic: {obs_stat:+.4f}")

B = 999
# (a) unrestricted permutation
perm_u = np.empty(B)
for i in range(B):
    sh = RNG.permutation(lab)
    perm_u[i] = (np.bincount(codes, weights=sh, minlength=K) / grp_n - sh.mean()).max()
p_u = (1 + int((perm_u >= obs_stat).sum())) / (B + 1)
print(f"\n  (a) UNRESTRICTED permutation:")
print(f"      null mean {perm_u.mean():+.4f}, 95th pct {np.percentile(perm_u,95):+.4f}")
print(f"      p = {p_u:.4f}")

# (b) permutation WITHIN HQ country
strata = q4["hq_country"].fillna("UNKNOWN").astype(str).values
scodes = pd.Categorical(strata).codes
base_order = np.argsort(scodes, kind="stable")
lab_sorted = lab[base_order]
n_strata = len(np.unique(scodes))
singleton = int((pd.Series(scodes).value_counts() == 1).sum())

perm_s = np.empty(B)
for i in range(B):
    rnd = RNG.random(len(lab))
    perm_idx = np.lexsort((rnd, scodes))     # group by stratum, random within
    sh = np.empty_like(lab)
    sh[perm_idx] = lab_sorted
    perm_s[i] = (np.bincount(codes, weights=sh, minlength=K) / grp_n - sh.mean()).max()
p_s = (1 + int((perm_s >= obs_stat).sum())) / (B + 1)

print(f"\n  (b) PERMUTATION WITHIN HQ COUNTRY:")
print(f"      strata (countries): {n_strata} | singleton strata: {singleton}")
print(f"      null mean {perm_s.mean():+.4f}, 95th pct {np.percentile(perm_s,95):+.4f}")
print(f"      p = {p_s:.4f}")
print(f"      → community concentration "
      f"{'SURVIVES' if p_s < 0.05 else 'does NOT survive'} conditioning on HQ country")

top5 = (q4.groupby("community")["quiet"].agg(["mean","size"])
        .sort_values("mean", ascending=False).head(5))
print(f"\n  Highest quiet-exit communities:")
print(top5.round(3).to_string())

pd.DataFrame([
    dict(test="unrestricted",   stat=obs_stat, p=p_u,
         null_mean=perm_u.mean(), null_p95=np.percentile(perm_u,95)),
    dict(test="within_country", stat=obs_stat, p=p_s,
         null_mean=perm_s.mean(), null_p95=np.percentile(perm_s,95)),
]).to_csv(f"{OUT_DIR}/q4_permutation_robustness.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
# 4. Coverage check
# ─────────────────────────────────────────────────────────────────────
sep("4. Coverage check — community 19 vs all others")

c19_keys = set(c19.index)
oth_keys = set(per.index) - c19_keys

d = idc[idc["investor_key"].isin(set(per.index))].copy()
d["year"] = d["deal_date"].dt.year
d["grp"]  = np.where(d["investor_key"].isin(c19_keys), "comm19", "others")

yr = (d[(d["year"] >= 2016) & (d["year"] <= 2026)]
      .groupby(["grp","year"])["deal_id"].nunique().unstack(0).fillna(0))
n19, noth = len(c19_keys), len(oth_keys)
yr["c19_per_inv"] = (yr.get("comm19", 0) / n19).round(3)
yr["oth_per_inv"] = (yr.get("others", 0) / noth).round(3)
tot19 = yr.get("comm19", pd.Series(0)).sum()
totoh = yr.get("others", pd.Series(0)).sum()
yr["c19_share"] = (yr.get("comm19", 0) / tot19).round(3)
yr["oth_share"] = (yr.get("others", 0) / totoh).round(3)

print(f"  Community 19 investors: {n19:,} | others: {noth:,}")
print(f"  Deals per year (unique deal_id), per-investor rate and within-group share:")
print(yr.to_string())

# When does each group peak, and what fraction lands in the outcome window?
pk19 = yr["c19_per_inv"].idxmax(); pkoh = yr["oth_per_inv"].idxmax()
print(f"\n  Peak per-investor deal year — community 19: {pk19} | others: {pkoh}")

late19 = yr.loc[2024:2026, "c19_share"].sum()
lateoh = yr.loc[2024:2026, "oth_share"].sum()
print(f"  Share of group's 2016-2026 deals falling in 2024-2026:")
print(f"    community 19: {late19:.3f}")
print(f"    others      : {lateoh:.3f}")
print(f"    ratio       : {late19/lateoh:.2f}x")

if late19 < lateoh:
    print(f"\n  → Community 19 deal recording DOES thin out earlier than other")
    print(f"    communities. Its elevated quiet-exit rate is therefore partly a")
    print(f"    coverage artefact, not purely behavioural. Treat Q4 as confounded")
    print(f"    with recording density.")
else:
    print(f"\n  → Community 19 recording does NOT drop off earlier; its quiet-exit")
    print(f"    rate is not explained by thinner recent coverage.")

yr.to_csv(f"{OUT_DIR}/community19_coverage.csv")
per.reset_index().rename(columns={"index":"investor_key"}).to_csv(
    f"{OUT_DIR}/base_population_v106.csv", index=False)

# ─────────────────────────────────────────────────────────────────────
sep("Summary")
print(f"  1. Non-overlapping base: N={TOT:,}, quiet-exit "
      f"{(per['exit_status']=='quiet_exit').mean()*100:.1f}%")
print(f"     overall excess {ov1['diff']:+.3f} [{ov1['dlo']:+.3f}, {ov1['dhi']:+.3f}]")
print(f"     Q1 excess {rows1[0][1]['diff']:+.3f} (was -0.744 with overlapping base)")
print(f"     Q5 excess {rows1[-1][1]['diff']:+.3f} (was +0.252)")
print(f"  2. EB shrinkage: alpha={alpha:.3f}, beta={beta_p:.3f}; "
      f"overall excess {ov2['diff']:+.3f}")
print(f"     Q5 excess {rows2[-1][1]['diff']:+.3f}; "
      f">=10 prior deals Q5 {rows3[-1][1]['diff']:+.3f}")
print(f"  3. Q4 permutation: unrestricted p={p_u:.4f}, within-country p={p_s:.4f}")
print(f"  4. Community 19 late-window deal share {late19:.3f} vs {lateoh:.3f} others")
print("\n" + "=" * 70)
print("SCRIPT 106 COMPLETE — robustness only, locked predictions untouched")
print("=" * 70)
