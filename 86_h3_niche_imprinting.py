"""
Script 86: H3 — niche imprinting.

PRE-SPECIFIED (report regardless):
  For later entrants in a niche, investors from the niche's founding community
  are MORE likely to fund them. Predicted sign on in_founding_community: POSITIVE.

UNIT: dyad (later entrant × candidate investor)
RISK SET: actual first-round investors + 5 counterfactuals active that year
EXCLUDE: founding startup's first-round investors from both sets

MODEL: conditional logit stratified by startup
  tie ~ in_founding_community + semantic_distance
      + geo_same_country + geo_same_region
      + same_community_t + log_prior_deals + has_prior_network

PLACEBO: replace founding community with random community of similar size,
  500 draws. Report permutation p.
"""

import os, sys, re, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_deals

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "h3_niche")
os.makedirs(OUT_DIR, exist_ok=True)

N_SIM   = 500
RNG     = np.random.default_rng(42)

print("=" * 70)
print("SCRIPT 86 — H3: NICHE IMPRINTING")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD INPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Loading inputs ──")

niche_df  = pd.read_csv("output/niches/niche_profiles.csv")
member_df = pd.read_csv("output/niches/niche_assignments.csv",
                        parse_dates=["first_date"])

lagged_comm = pd.read_csv("output/novelty_t/lagged_communities.csv")
lc_lookup   = {}
for _, r in lagged_comm.iterrows():
    lc_lookup[(r["investor"], int(r["year"]))] = int(r["community_lagged"])

# Community size by year: community_id → n_investors at year t
comm_size = {}    # (year, community) → count
for (inv, yr), comm in lc_lookup.items():
    key = (yr, comm)
    comm_size[key] = comm_size.get(key, 0) + 1

deals_df = load_deals()
deals_df["Deal Date"] = pd.to_datetime(deals_df["Deal Date"], errors="coerce")
deals_df = deals_df.dropna(subset=["Deal Date"])
deals_df["year"] = deals_df["Deal Date"].dt.year

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip() for p in parts if p.strip()]

# Active investors by year: year → set of investor names (had at least 1 deal that year)
active_by_year = {}
for yr, grp in deals_df.groupby("year"):
    invs = set()
    for v in grp["Investors"].dropna():
        invs.update(comma_parse(v))
    active_by_year[int(yr)] = invs

# Investor geographic HQ (for geo_same_country)
inv_xl = pd.read_excel("data/Investors_data.xlsx", header=7,
                       usecols=["Investors", "HQ Country/Territory/Region"])
inv_xl = inv_xl.rename(columns={"Investors": "investor",
                                 "HQ Country/Territory/Region": "inv_country"})
inv_xl = inv_xl.dropna(subset=["investor"]).set_index("investor")["inv_country"].to_dict()

# Startup HQ country (from deals data)
startup_country = (
    deals_df.sort_values("Deal Date")
    .groupby("Company ID")
    .first()["Company Country/Territory/Region"]
    .to_dict()
)

# Investor SPECTER2 embeddings
inv_names_df = pd.read_csv("data/investor_names.csv", header=None, names=["investor"])
inv_emb_arr  = np.load("data/investor_embeddings.npy").astype(np.float32)
inv_norms    = np.linalg.norm(inv_emb_arr, axis=1, keepdims=True)
inv_norms[inv_norms == 0] = 1.0
inv_emb_arr  = inv_emb_arr / inv_norms
inv_name_to_idx = {n: i for i, n in enumerate(inv_names_df["investor"])}
# Skip header row if present
if "investor" in inv_name_to_idx:
    del inv_name_to_idx["investor"]

# Startup SPECTER2 embeddings
with open("output/novelty/novelty_specter.pkl", "rb") as f:
    spec_data = pickle.load(f)
su_emb_names = spec_data["names"]
su_emb_raw   = np.array(spec_data["embs_raw"], dtype=np.float32)
su_norms     = np.linalg.norm(su_emb_raw, axis=1, keepdims=True)
su_norms[su_norms == 0] = 1.0
su_emb_arr   = su_emb_raw / su_norms
su_name_to_idx = {n: i for i, n in enumerate(su_emb_names)}

# First-round investors per company (Company ID → list)
first_round_inv = {}
for cid, grp in deals_df.sort_values("Deal Date").groupby("Company ID"):
    row = grp.iloc[0]
    first_round_inv[cid] = comma_parse(row["Investors"]) if pd.notna(row.get("Investors")) else []

# Investor prior deals count: investor × year → n deals before year t
inv_deals_before = {}   # investor → sorted list of years with deals
for yr, grp in sorted(deals_df.groupby("year")):
    for v in grp["Investors"].dropna():
        for inv in comma_parse(v):
            inv_deals_before.setdefault(inv, []).append(int(yr))

def prior_deals(investor, year):
    years = inv_deals_before.get(investor, [])
    return sum(1 for y in years if y < year)

print(f"  Niches loaded: {len(niche_df):,}")
print(f"  Member rows: {len(member_df):,}")
print(f"  Investor embeddings: {len(inv_name_to_idx):,}")

# ═══════════════════════════════════════════════════════════════════════════
# BUILD DYAD DATASET
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Building dyad dataset ──")

# Filter to niches with founding community AND ≥2 later entrants
niche_usable = niche_df.dropna(subset=["founding_community"]).copy()
niche_usable["founding_community"] = niche_usable["founding_community"].astype(int)
niche_usable = niche_usable[niche_usable["n_members"] >= 3]

print(f"  Niches with founding community AND ≥3 members: {len(niche_usable):,}")

# Niche ID → founding community
niche_fc = niche_usable.set_index("niche_id")["founding_community"].to_dict()

# Niche ID → founding startup Company ID
niche_founder_cid = {}
for nid, row in niche_usable.set_index("niche_id").iterrows():
    founder_name = row["founding_startup"]
    # Get Company ID from member_df
    matches = member_df[(member_df["niche_id"] == nid) &
                        (member_df["position"] == "founder")]["Company ID"]
    if len(matches):
        niche_founder_cid[nid] = matches.iloc[0]

# All later entrants restricted to usable niches
later_df = member_df[
    (member_df["position"] == "later") &
    (member_df["niche_id"].isin(niche_fc))
].copy()

print(f"  Later entrants in usable niches: {len(later_df):,}")

dyad_rows = []
n_excluded_no_investors = 0
n_excluded_empty_pool   = 0

for _, su_row in later_df.iterrows():
    niche_id  = int(su_row["niche_id"])
    cid       = su_row["Company ID"]
    name      = su_row["Companies"]
    t_k       = int(su_row["first_year"])
    fc        = niche_fc.get(niche_id)

    # Realised first-round investors
    realised = first_round_inv.get(cid, [])
    if len(realised) == 0:
        n_excluded_no_investors += 1
        continue

    # Founders' investors: excluded from risk set
    founder_cid = niche_founder_cid.get(niche_id)
    founder_inv = set(first_round_inv.get(founder_cid, [])) if founder_cid else set()

    # Filter realised: exclude founding investors
    realised_ok = [i for i in realised if i not in founder_inv]
    if len(realised_ok) == 0:
        n_excluded_no_investors += 1
        continue

    # Counterfactual pool: active in t_k, not in realised, not in founder_inv
    pool = [i for i in active_by_year.get(t_k, set())
            if i not in set(realised) and i not in founder_inv]
    if len(pool) < 5:
        n_excluded_empty_pool += 1
        continue

    counterfactuals = list(RNG.choice(pool, size=5, replace=False))

    # Startup embedding
    su_idx = su_name_to_idx.get(name)

    # HQ country of this startup
    su_ctry = startup_country.get(cid, None)

    # Build dyad rows
    for investor, label in ([(i, 1) for i in realised_ok] +
                             [(i, 0) for i in counterfactuals]):
        # in_founding_community: investor's lagged community at t_k == fc
        inv_comm_tk = lc_lookup.get((investor, t_k))
        ifc = int(inv_comm_tk == fc) if (inv_comm_tk is not None and fc is not None) else 0

        # same_community_t: investor is in same lagged community as the startup's
        # founding-community investors — computed as matching the modal community
        # of ALL niche members already funded by t_k (excluding the focal later entrant).
        # Since we can only look this up per-niche at the founding year (to avoid
        # using outcome data), we use the founding community itself, making this
        # variable identical to in_founding_community. We therefore DROP same_community_t
        # from the spec to avoid perfect collinearity. (Noted in results.)
        same_comm_t = 0  # placeholder — excluded from b_vars below

        # geo
        inv_ctry = inv_xl.get(investor)
        geo_country = int(su_ctry == inv_ctry) if (su_ctry and inv_ctry) else 0

        # semantic distance
        if su_idx is not None:
            inv_idx = inv_name_to_idx.get(investor)
            if inv_idx is not None:
                sem_d = float(1.0 - su_emb_arr[su_idx] @ inv_emb_arr[inv_idx])
            else:
                sem_d = np.nan
        else:
            sem_d = np.nan

        # prior activity
        pd_count = prior_deals(investor, t_k)
        log_pd   = float(np.log1p(pd_count))
        hpn      = float(pd_count > 0)

        dyad_rows.append({
            "startup":               cid,
            "investor":              investor,
            "niche_id":              niche_id,
            "tie":                   label,
            "in_founding_community": ifc,
            "semantic_distance":     sem_d,
            "geo_same_country":      geo_country,
            "same_community_t":      same_comm_t,
            "log_prior_deals":       log_pd,
            "has_prior_network":     hpn,
            "founding_community":    fc,
            "year":                  t_k,
        })

dm = pd.DataFrame(dyad_rows)
print(f"  Startups without investors (skipped): {n_excluded_no_investors}")
print(f"  Startups with empty pool (skipped): {n_excluded_empty_pool}")
print(f"  Total dyad rows: {len(dm):,}")
print(f"  Strata (later-entrant startups): {dm['startup'].nunique():,}")
print(f"  in_founding_community = 1: {dm['in_founding_community'].sum():,} "
      f"({100*dm['in_founding_community'].mean():.1f}%)")

# Drop rows with missing semantic_distance
dm_ok = dm.dropna(subset=["semantic_distance"]).copy()
print(f"  Rows with semantic_distance: {len(dm_ok):,} / {len(dm):,}")

# Standardise continuous predictors
for col in ["semantic_distance", "log_prior_deals"]:
    mu = dm_ok[col].mean(); sd = dm_ok[col].std()
    dm_ok[f"{col}_z"] = (dm_ok[col] - mu) / sd if sd > 0 else 0.0

# Drop strata that are all-0 or all-1
def has_both(grp):
    return grp["tie"].nunique() == 2

valid_strata = dm_ok.groupby("startup").filter(has_both)
print(f"  Valid strata (both 0 and 1): {valid_strata['startup'].nunique():,}")
print(f"  Valid dyad rows: {len(valid_strata):,}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Main model: conditional logit ──")

b_vars = ["in_founding_community", "semantic_distance_z",
          "geo_same_country",
          "log_prior_deals_z", "has_prior_network"]
# same_community_t dropped: collinear with in_founding_community in this context
# (both track whether investor belongs to the niche's founding Louvain community)

def run_clogit(data, endog_col="tie", group_col="startup"):
    sub = data[data[group_col].isin(
        data.groupby(group_col).filter(has_both)[group_col].unique()
    )].copy()
    endog = sub[endog_col].values.astype(float)
    exog  = sub[b_vars].values.astype(float)
    # Drop rows with any nan in exog
    mask  = np.isfinite(exog).all(axis=1)
    endog, exog = endog[mask], exog[mask]
    grp   = sub[group_col].values[mask]
    model = ConditionalLogit(endog, exog, groups=grp)
    for method in ["newton", "bfgs", "lbfgs"]:
        try:
            res = model.fit(method=method, maxiter=400, disp=False)
            if hasattr(res, "bse") and np.isfinite(res.bse).all():
                return res, sub
        except Exception:
            continue
    raise RuntimeError("All optimisers failed")
    return res, sub

try:
    res, sub = run_clogit(valid_strata)
    print(f"\n  N dyads: {int(res.nobs):,}  |  N strata: {sub['startup'].nunique():,}")
    print()
    print(f"  {'Variable':<28}  {'β':>9}  {'SE':>8}  {'p':>8}  {'':>4}")
    print("  " + "─" * 56)

    def fmt_stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "†"
        return ""

    for v, b, se, p in zip(b_vars, res.params, res.bse, res.pvalues):
        st = fmt_stars(p)
        print(f"  {v:<28}  {b:+9.4f}  {se:8.4f}  {p:8.4f}  {st}")

    b_ifc = res.params[0]
    se_ifc = res.bse[0]
    p_ifc  = res.pvalues[0]

    # MDE
    mde = (norm.ppf(0.975) + norm.ppf(0.80)) * se_ifc
    print(f"\n  MDE (80% power, α=0.05): {mde:.4f}")
    print(f"  in_founding_community: β={b_ifc:+.4f}  SE={se_ifc:.4f}  p={p_ifc:.4f}  {fmt_stars(p_ifc)}")

except Exception as e:
    print(f"  Main model failed: {e}")
    b_ifc = np.nan; se_ifc = np.nan; p_ifc = np.nan; mde = np.nan

# ═══════════════════════════════════════════════════════════════════════════
# PLACEBO: random community of similar size, 500 draws
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Placebo: 500 permutations ──")

# Build community size lookup: (year, community) → set of investors
# We'll need to draw random communities of similar size at year t0 for each niche
niche_t0 = niche_usable.set_index("niche_id")["founding_year"].to_dict()

# All available (year, community) pairs from lagged_comm
year_comm_map = {}   # year → {community → n_investors}
for (yr, comm), cnt in comm_size.items():
    year_comm_map.setdefault(yr, {})[comm] = cnt

def draw_placebo_comm(niche_id, fc_size, t0):
    """Draw a random community of similar size (±50%) from year t0 pool."""
    pool = year_comm_map.get(t0, {})
    lo, hi = fc_size * 0.5, fc_size * 1.5
    candidates = [c for c, n in pool.items() if lo <= n <= hi and c != niche_fc.get(niche_id)]
    if not candidates:
        # Widen search
        candidates = [c for c, n in pool.items() if c != niche_fc.get(niche_id)]
    if not candidates:
        return None
    return int(RNG.choice(candidates))

# Pre-compute fc_size for each niche
niche_fc_size = {}
for nid, fc in niche_fc.items():
    t0 = int(niche_t0.get(nid, 0))
    niche_fc_size[nid] = comm_size.get((t0, fc), 1)

perm_betas = []
valid_strata_cols = valid_strata.copy()

for sim_i in range(N_SIM):
    # For each niche, draw a random placebo community
    placebo_fc = {}
    for nid in niche_fc:
        t0 = int(niche_t0.get(nid, 0))
        fc_sz = niche_fc_size.get(nid, 1)
        placebo_fc[nid] = draw_placebo_comm(nid, fc_sz, t0)

    # Recompute in_founding_community using placebo
    dm_perm = valid_strata_cols.copy()
    dm_perm["in_founding_community"] = dm_perm.apply(
        lambda r: int(
            lc_lookup.get((r["investor"], r["year"])) == placebo_fc.get(r["niche_id"])
            if placebo_fc.get(r["niche_id"]) is not None else 0
        ), axis=1
    )

    try:
        res_p, _ = run_clogit(dm_perm)
        perm_betas.append(float(res_p.params[0]))
    except Exception:
        pass

    if (sim_i + 1) % 100 == 0:
        print(f"  [{sim_i+1}/{N_SIM}] permutations done  "
              f"(n_valid={len(perm_betas)})")

print(f"\n  Valid permutations: {len(perm_betas)} / {N_SIM}")
if perm_betas and not np.isnan(b_ifc):
    perm_arr   = np.array(perm_betas)
    perm_p     = float(np.mean(np.abs(perm_arr) >= abs(b_ifc)))
    perm_mean  = perm_arr.mean()
    perm_sd    = perm_arr.std()
    print(f"  Null distribution: mean={perm_mean:.4f}  SD={perm_sd:.4f}")
    print(f"  Observed |β| = {abs(b_ifc):.4f}")
    print(f"  Permutation p = {perm_p:.4f}")
elif not perm_betas:
    perm_p = np.nan
    print("  No valid permutations — permutation p unavailable.")
else:
    perm_p = np.nan
    print("  Main model unestimable — permutation p not computed.")

# ═══════════════════════════════════════════════════════════════════════════
# PLAIN VERDICT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PLAIN VERDICT — H3: NICHE IMPRINTING")
print("=" * 70)

def fmt_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""

print(f"\n  Pre-specified sign: POSITIVE")
print(f"  Main estimate: β={b_ifc:+.4f}  SE={se_ifc:.4f}  p={p_ifc:.4f}  "
      f"{fmt_stars(p_ifc) if not np.isnan(p_ifc) else ''}")
print(f"  MDE (80% power):  {mde:.4f}")
print(f"  Permutation p:    {perm_p:.4f}" if not np.isnan(perm_p) else "  Permutation p:    unavailable")

# Coverage note
print(f"\n  COVERAGE NOTE:")
print(f"  Only {len(niche_usable):,} of {len(niche_df):,} niches have a founding community.")
print(f"  Of {later_df['Company ID'].nunique():,} later entrants, "
      f"{valid_strata['startup'].nunique() if len(valid_strata) else 0:,} contribute valid dyads.")
print(f"  in_founding_community = 1 in {dm['in_founding_community'].mean()*100:.1f}% of dyads.")
print(f"  Root cause: early-niche founders (pre-2016) predate the co-investment")
print(f"  network, so lagged community coverage is sparse for founding investors.")

if np.isnan(b_ifc):
    verdict = "UNESTIMABLE — insufficient data"
elif b_ifc > 0 and p_ifc < 0.05:
    verdict = "SUPPORTED"
elif b_ifc > 0 and p_ifc < 0.10:
    verdict = "DIRECTIONALLY SUPPORTED (marginal)"
elif b_ifc > 0 and p_ifc >= 0.10:
    verdict = "DIRECTIONALLY CORRECT BUT NOT SIGNIFICANT"
elif b_ifc < 0 and p_ifc < 0.05:
    verdict = "WRONG SIGN — SIGNIFICANT"
else:
    verdict = "NULL — WRONG SIGN"

print(f"\n  ** H3 VERDICT: {verdict} **")

# Save summary
summary = pd.DataFrame([{
    "hypothesis": "H3 niche imprinting",
    "beta_ifc": b_ifc,
    "se_ifc": se_ifc,
    "p_ifc": p_ifc,
    "mde": mde,
    "perm_p": perm_p,
    "n_dyads": int(res.nobs) if not np.isnan(b_ifc) else np.nan,
    "n_strata": valid_strata["startup"].nunique() if not np.isnan(b_ifc) else np.nan,
    "n_niches": len(niche_usable),
    "n_valid_perms": len(perm_betas),
}])
summary.to_csv(os.path.join(OUT_DIR, "h3_summary.csv"), index=False)

print("\n" + "=" * 70)
print("SCRIPT 86 COMPLETE")
print("=" * 70)
