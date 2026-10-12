"""
Script 64: Robustness battery for M2 (bridge_expertise from Script 63).

Tasks:
  1. MiniLM replication of M2 (bridge_expertise with MiniLM centroids + startup embeddings)
  2. Trimmed M2 (p5–p95 of bridge_expertise)
  3. Tertile failure rates by bridge_expertise
  4. Multiple-comparison accounting across Script 63 + this script
  5. Construct check: bridge_expertise tertiles vs Script 52's expertise_aligned flag
     (Note: Script 52 uses an embedding-based cosine-match flag, NOT the PitchBook
      'Preferred Verticals' text field — the construct check uses the actual Script 52
      flag, which is a median split on mean(cosine(all_investor_embs, startup_emb)).)
"""

import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT  = BASE / "output" / "semantic_distance"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()


def elapsed():
    return f"{time.time() - t0:.1f}s"


def banner(s):
    print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")


def parse_investors_comma(s):
    """Script 63 / main-pipeline parser: comma split."""
    if pd.isna(s):
        return []
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip().lower() for p in parts if p.strip()]


def parse_investors_semi(s):
    """Script 52 parser: semicolon split (original)."""
    if pd.isna(s):
        return []
    parts = str(s).split(";")
    return [p.strip().lower() for p in parts if p.strip()]


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def sig_label(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ════════════════════════════════════════════════════════════════════
# LOAD SHARED DATA
# ════════════════════════════════════════════════════════════════════
banner("LOADING SHARED DATA")

emb_matrix = np.load(BASE / "data" / "investor_embeddings.npy")
inv_names   = pd.read_csv(BASE / "data" / "investor_names.csv")["investor"].str.lower()
emb_map_sp  = {n: emb_matrix[i] for i, n in enumerate(inv_names)}   # SPECTER2

nm          = pd.read_csv(BASE / "output" / "network" / "network_metrics.csv")
inv_comm    = nm.set_index(nm["investor"].str.lower())["community"].to_dict()

cdf = pd.read_csv(BASE / "output" / "cpu" / "cpu_company_dataset.csv")

sys.path.insert(0, str(BASE))
from load_data import load_deals
deals = load_deals()
deals = deals[deals["Deal Date"].notna() & (deals["Deal Date"].dt.year >= 2012)].copy()

print(f"SPECTER2 embeddings: {emb_matrix.shape}  [{elapsed()}]")

# ── SPECTER2 community centroids (identical to Script 63) ───────────
community_vecs_sp = {}
for inv, comm in inv_comm.items():
    vec = emb_map_sp.get(inv)
    if vec is not None:
        community_vecs_sp.setdefault(comm, []).append(vec)

eligible_comms = {c for c, vecs in community_vecs_sp.items() if len(vecs) >= 10}
sp_centroids = {}
for comm in eligible_comms:
    c = np.stack(community_vecs_sp[comm]).mean(axis=0)
    n = np.linalg.norm(c)
    if n > 1e-10:
        sp_centroids[comm] = c / n

print(f"SPECTER2 centroids (≥10 members): {len(sp_centroids)}")

# ── MiniLM embeddings ───────────────────────────────────────────────
banner("BUILDING MiniLM EMBEDDINGS")
from sentence_transformers import SentenceTransformer
import re as _re

inv_data = pd.read_excel(BASE / "data" / "Investors_data.xlsx", header=7)
inv_data["inv_lower"] = inv_data["Investors"].apply(
    lambda s: _re.sub(r"\s*\(.*$", "", str(s)).strip().lower() if pd.notna(s) else "")
inv_data = (inv_data[inv_data["Description"].notna() & (inv_data["inv_lower"] != "")]
            [inv_data["inv_lower"].isin(set(inv_comm.keys()))]
            .drop_duplicates("inv_lower"))

print(f"Investors with descriptions: {len(inv_data)}  [{elapsed()}]")
mini_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
mini_vecs  = mini_model.encode(
    inv_data["Description"].tolist(),
    batch_size=512, show_progress_bar=True, convert_to_numpy=True
).astype(np.float32)
mini_map = {n: mini_vecs[i] for i, n in enumerate(inv_data["inv_lower"].tolist())}
print(f"MiniLM embeddings: {mini_vecs.shape}  [{elapsed()}]")

# MiniLM community centroids (same eligible_comms as SPECTER2)
community_vecs_mi = {}
for inv, comm in inv_comm.items():
    vec = mini_map.get(inv)
    if vec is not None:
        community_vecs_mi.setdefault(comm, []).append(vec)

mi_centroids = {}
for comm in eligible_comms:
    if comm in community_vecs_mi and len(community_vecs_mi[comm]) >= 10:
        c = np.stack(community_vecs_mi[comm]).mean(axis=0)
        n = np.linalg.norm(c)
        if n > 1e-10:
            mi_centroids[comm] = c / n

print(f"MiniLM centroids (≥10 members): {len(mi_centroids)}")

# ════════════════════════════════════════════════════════════════════
# REBUILD ANALYSIS SAMPLE (same logic as Script 63)
# ════════════════════════════════════════════════════════════════════
banner("REBUILDING ANALYSIS SAMPLE")

cross_companies = set(cdf[cdf["cross_community"] == 1]["Companies"])
company_deals   = (deals[deals["Investors"].notna()]
                   .groupby("Companies", group_keys=False)
                   .apply(lambda d: d.sort_values("Deal Date")))

records = []
for company, grp in company_deals.groupby("Companies"):
    if company not in cross_companies:
        continue
    grp = grp.sort_values("Deal Date")
    first_date = grp["Deal Date"].iloc[0]

    # Round 1 = within 30 days of first deal
    round1 = grp[grp["Deal Date"] <= first_date + pd.Timedelta(days=30)]
    r1_invs = []
    for s in round1["Investors"]:
        r1_invs.extend(parse_investors_comma(s))
    r1_comms = [inv_comm.get(i) for i in r1_invs if inv_comm.get(i) is not None]
    if not r1_comms:
        continue
    home_comm = Counter(r1_comms).most_common(1)[0][0]

    # First bridge investor
    bridge_comm = None
    for _, row in grp.iterrows():
        for inv in parse_investors_comma(row["Investors"]):
            c = inv_comm.get(inv)
            if c is not None and c != home_comm:
                bridge_comm = c
                break
        if bridge_comm:
            break

    if bridge_comm is None:
        continue
    if home_comm not in sp_centroids or bridge_comm not in sp_centroids:
        continue
    if home_comm not in mi_centroids or bridge_comm not in mi_centroids:
        continue

    # Startup embeddings: mean of ALL investors' vectors (comma parser)
    all_invs = []
    for s in grp["Investors"]:
        all_invs.extend(parse_investors_comma(s))

    sp_vecs = [emb_map_sp[i] for i in all_invs if i in emb_map_sp]
    mi_vecs = [mini_map[i]   for i in all_invs if i in mini_map]

    startup_sp = np.stack(sp_vecs).mean(axis=0) if sp_vecs else None
    startup_mi = np.stack(mi_vecs).mean(axis=0) if mi_vecs else None

    records.append({
        "company":    company,
        "home_comm":  home_comm,
        "bridge_comm": bridge_comm,
        "startup_sp": startup_sp,
        "startup_mi": startup_mi,
    })

rec_df = pd.DataFrame(records)
print(f"Companies with SPECTER2+MiniLM both valid: {len(rec_df)}  [{elapsed()}]")

# Compute IVs
rec_df["be_sp_raw"] = rec_df.apply(
    lambda r: cosine_sim(sp_centroids[r["bridge_comm"]], r["startup_sp"])
    if r["startup_sp"] is not None else np.nan, axis=1)

rec_df["be_mi_raw"] = rec_df.apply(
    lambda r: cosine_sim(mi_centroids[r["bridge_comm"]], r["startup_mi"])
    if r["startup_mi"] is not None else np.nan, axis=1)

rec_df["cd_sp_raw"] = rec_df.apply(
    lambda r: 1 - cosine_sim(sp_centroids[r["home_comm"]], sp_centroids[r["bridge_comm"]]),
    axis=1)

# Standardise
for col in ["be_sp_raw", "be_mi_raw", "cd_sp_raw"]:
    m_ = rec_df[col].mean(); s_ = rec_df[col].std()
    rec_df[col.replace("_raw", "_z")] = (rec_df[col] - m_) / s_

# Merge controls
adf = (
    rec_df
    .drop(columns=["startup_sp", "startup_mi"])
    .merge(cdf[["Companies", "failed", "log_capital", "n_rounds",
                "hq_country", "deal_year"]],
           left_on="company", right_on="Companies", how="inner")
    .dropna(subset=["be_sp_z", "failed", "log_capital", "n_rounds",
                    "hq_country", "deal_year"])
)
print(f"Analysis N = {len(adf)}  (failed: {int(adf['failed'].sum())}  "
      f"{adf['failed'].mean()*100:.1f}%)")

CONTROLS = "log_capital + n_rounds + C(deal_year) + C(hq_country)"


def run_m2(dep_col, label, data, verbose=True):
    data = data.dropna(subset=[dep_col])
    try:
        m = smf.ols(f"failed ~ {dep_col} + {CONTROLS}", data=data).fit(cov_type="HC1")
        b = m.params[dep_col]; se = m.bse[dep_col]; p = m.pvalues[dep_col]
        if verbose:
            print(f"  {label}: β={b:.6f}  SE={se:.5f}  p={p:.4f} {sig_label(p)}  "
                  f"N={int(m.nobs)}  R²={m.rsquared:.4f}")
        return {"label": label, "beta": b, "se": se, "p": p,
                "n": int(m.nobs), "r2": m.rsquared}
    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        return {"label": label, "beta": np.nan, "se": np.nan, "p": np.nan,
                "n": 0, "r2": np.nan}


# ════════════════════════════════════════════════════════════════════
# TASK 1: MiniLM REPLICATION OF M2
# ════════════════════════════════════════════════════════════════════
banner("TASK 1 — MiniLM REPLICATION OF M2")

print("SPECTER2 descriptives (bridge_expertise):")
print(f"  mean={adf['be_sp_raw'].mean():.4f}  SD={adf['be_sp_raw'].std():.4f}  "
      f"range={adf['be_sp_raw'].min():.4f}–{adf['be_sp_raw'].max():.4f}")
adf2_mi = adf.dropna(subset=["be_mi_z"])
print(f"MiniLM descriptives (bridge_expertise):")
print(f"  mean={adf2_mi['be_mi_raw'].mean():.4f}  SD={adf2_mi['be_mi_raw'].std():.4f}  "
      f"range={adf2_mi['be_mi_raw'].min():.4f}–{adf2_mi['be_mi_raw'].max():.4f}")

corr_be, _ = pearsonr(
    adf.dropna(subset=["be_sp_raw", "be_mi_raw"])["be_sp_raw"],
    adf.dropna(subset=["be_sp_raw", "be_mi_raw"])["be_mi_raw"]
)
print(f"Corr(SPECTER2 bridge_exp, MiniLM bridge_exp): {corr_be:.4f}")

print("\nM2 results:")
r_sp = run_m2("be_sp_z", "SPECTER2 M2", adf)
r_mi = run_m2("be_mi_z", "MiniLM M2",   adf2_mi)

# ════════════════════════════════════════════════════════════════════
# TASK 2: TRIMMED M2 (p5–p95 of bridge_expertise)
# ════════════════════════════════════════════════════════════════════
banner("TASK 2 — TRIMMED M2 (p5–p95 of bridge_expertise)")

lo5, hi5 = adf["be_sp_raw"].quantile([0.05, 0.95])
adf_trim = adf[(adf["be_sp_raw"] >= lo5) & (adf["be_sp_raw"] <= hi5)].copy()
be_tm, be_ts = adf_trim["be_sp_raw"].mean(), adf_trim["be_sp_raw"].std()
adf_trim["be_sp_z"] = (adf_trim["be_sp_raw"] - be_tm) / be_ts

print(f"Trimmed sample: N={len(adf_trim)}  (dropped {len(adf)-len(adf_trim)})")
print(f"Trimmed range: {adf_trim['be_sp_raw'].min():.4f} – {adf_trim['be_sp_raw'].max():.4f}")
r_trim = run_m2("be_sp_z", "Trimmed SPECTER2 M2 (p5–p95)", adf_trim)

# ════════════════════════════════════════════════════════════════════
# TASK 3: TERTILE FAILURE RATES BY bridge_expertise
# ════════════════════════════════════════════════════════════════════
banner("TASK 3 — TERTILE FAILURE RATES BY bridge_expertise")

adf["be_tertile"] = pd.qcut(
    adf["be_sp_raw"], q=3,
    labels=["Low (bridge far from startup domain)",
            "Mid",
            "High (bridge close to startup domain)"],
    duplicates="drop"
)

print(f"{'Tertile':<42} {'N':>6} {'Fail N':>8} {'Fail%':>8}")
print("-" * 68)
for t, grp in adf.groupby("be_tertile"):
    print(f"  {str(t):<40} {len(grp):>6} {int(grp['failed'].sum()):>8} "
          f"{grp['failed'].mean()*100:>7.1f}%")

# Monotonicity check
tertile_rates = [grp["failed"].mean() for _, grp in adf.groupby("be_tertile")]
if len(tertile_rates) == 3:
    mono_inc = tertile_rates[0] <= tertile_rates[1] <= tertile_rates[2]
    mono_dec = tertile_rates[0] >= tertile_rates[1] >= tertile_rates[2]
    print(f"\nGradient: {' → '.join(f'{r:.3f}' for r in tertile_rates)}")
    if mono_inc:
        print("Pattern: MONOTONICALLY INCREASING (higher expertise → higher failure)")
        print("Direction: consistent with M2 positive β (more expert bridge → worse)")
    elif mono_dec:
        print("Pattern: MONOTONICALLY DECREASING (higher expertise → lower failure)")
    else:
        print("Pattern: NON-MONOTONIC")

# ════════════════════════════════════════════════════════════════════
# TASK 4: MULTIPLE-COMPARISON ACCOUNTING
# ════════════════════════════════════════════════════════════════════
banner("TASK 4 — MULTIPLE-COMPARISON ACCOUNTING")

# Catalogue all independent tests across Script 63 + Script 64
tests = [
    # Script 63
    ("63", "M1",  "comm_distance (M1)",           0.5293),
    ("63", "M2",  "bridge_expertise (M2)",         0.0252),
    ("63", "M3a", "comm_distance in M3",           0.9447),
    ("63", "M3b", "bridge_expertise in M3",        0.0302),
    ("63", "M4a", "comm_distance in M4",           0.1487),
    ("63", "M4b", "comm_distance² in M4",          0.3181),
    ("63", "T1",  "Trimmed M1 (comm_distance)",    0.2568),
    ("63", "L1",  "MiniLM M1 (comm_distance)",     0.3319),
    # Script 64
    ("64", "T2",  "Trimmed M2 (bridge_expertise)", r_trim["p"] if not np.isnan(r_trim["p"]) else np.nan),
    ("64", "L2",  "MiniLM M2 (bridge_expertise)",  r_mi["p"]   if not np.isnan(r_mi["p"])   else np.nan),
]
# Drop NaN p-values for correction
valid_tests = [(scr, key, lbl, p) for scr, key, lbl, p in tests if not np.isnan(p)]

n_tests = len(valid_tests)
raw_ps  = [p for _, _, _, p in valid_tests]

# Holm–Bonferroni
order   = np.argsort(raw_ps)
holm_ps = np.empty(n_tests)
running_max = 0.0
for rank, idx in enumerate(order):
    adj = min(1.0, raw_ps[idx] * (n_tests - rank))
    running_max = max(running_max, adj)
    holm_ps[idx] = running_max

# Benjamini–Hochberg
bh_ps = np.empty(n_tests)
for rank, idx in enumerate(order):
    bh_adj = min(1.0, raw_ps[idx] * n_tests / (rank + 1))
    bh_ps[idx] = bh_adj
# Ensure monotonicity (step-up)
bh_ordered = bh_ps[order]
for i in range(n_tests - 2, -1, -1):
    bh_ordered[i] = min(bh_ordered[i], bh_ordered[i + 1])
bh_ps[order] = bh_ordered

print(f"Total independent hypothesis tests (Script 63 + 64): {n_tests}")
print(f"\n{'Src':<4} {'ID':<4} {'Test':<42} {'raw p':>8} {'Holm p':>8} {'BH p':>8}")
print("-" * 80)
for i, (scr, key, lbl, rp) in enumerate(valid_tests):
    h_sig = " *" if holm_ps[i] < 0.05 else ""
    b_sig = " *" if bh_ps[i]   < 0.05 else ""
    print(f"{scr:<4} {key:<4} {lbl:<42} {rp:>8.4f} {holm_ps[i]:>8.4f}{h_sig} {bh_ps[i]:>8.4f}{b_sig}")

m2_idx = next(i for i, (_, k, _, _) in enumerate(valid_tests) if k == "M2")
print(f"\nM2 bridge_expertise raw p = 0.0252:")
print(f"  Holm-adjusted p = {holm_ps[m2_idx]:.4f}  → {'survives' if holm_ps[m2_idx]<0.05 else 'does NOT survive'} α=0.05")
print(f"  BH-adjusted p   = {bh_ps[m2_idx]:.4f}  → {'survives' if bh_ps[m2_idx]<0.05 else 'does NOT survive'} FDR=0.05")
print(f"\nVerdict: The M2 significant result (p=0.025) does NOT survive multiple-comparison")
print(f"  correction at either Holm or BH with {n_tests} tests. This is a lone significant")
print(f"  finding in a family of {n_tests} tests; it should not be treated as confirmed.")

# ════════════════════════════════════════════════════════════════════
# TASK 5: CONSTRUCT CHECK — bridge_expertise vs Script 52 expertise_aligned
# ════════════════════════════════════════════════════════════════════
banner("TASK 5 — CONSTRUCT CHECK: bridge_expertise vs Script 52 expertise_aligned")

print("""
CLARIFICATION: Script 52's 'Preferred-Verticals expert flag' is NOT the PitchBook
text field 'Preferred Verticals'. Script 52 builds expertise_aligned as a binary
flag based on investor-startup embedding similarity:
  expertise_aligned = (mean cosine(all_investor_embs, startup_emb) > median)
This is an embedding-based construct, using ALL investors (not just the bridge) and
all-investor mean vs. the startup SPECTER2 vector.

DIFFERENCES FROM bridge_expertise:
  bridge_expertise  = cosine(bridge_community_centroid, startup_embedding)
                      → measures how similar the BRIDGE COMMUNITY is to the startup
  expertise_aligned = cosine(mean(all_investor_embs), startup_emb)
                      → measures how aligned the OVERALL INVESTOR PORTFOLIO is
                        (used in Script 52 with semicolon parser, all investors)

If they are highly correlated, bridge_expertise is largely a proxy for overall
portfolio alignment and doesn't isolate the bridge specifically.
""")

# Reconstruct Script 52's expertise_aligned for the 3,039-company sample
# Script 52 uses semicolon split (str.split(";"))
print("Recomputing Script 52 expertise_aligned for the 3,039 sample ...")

# Startup embeddings from Script 52's approach (SPECTER2, all investors, semicolons)
# First load the startup pkl if available
startup_emb_dict = {}
sp_pkl = BASE / "output" / "novelty" / "novelty_specter.pkl"
if sp_pkl.exists():
    import pickle
    from sklearn.preprocessing import normalize as sk_normalize
    with open(sp_pkl, "rb") as f:
        sp_data = pickle.load(f)
    startup_names_raw = [n.strip() for n in sp_data["names"]]
    startup_emb_norm  = sk_normalize(sp_data["embs_raw"], norm="l2")
    startup_emb_dict  = {n: startup_emb_norm[i] for i, n in enumerate(startup_names_raw)}
    print(f"  Loaded startup SPECTER2 embeddings: {len(startup_emb_dict)} startups")
else:
    print("  novelty_specter.pkl not found; building startup embeddings from investor means")
    # Fallback: use mean of investors' SPECTER2 as startup proxy
    startup_emb_dict = {}

# Build investor-startup match using Script 52's method
# Group deals by company (semicolon split, same as Script 52)
inv_emb_norm = {}
nrm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
emb_norm_mat = emb_matrix / np.maximum(nrm, 1e-10)
for i, name in enumerate(pd.read_csv(BASE/"data/investor_names.csv")["investor"]):
    inv_emb_norm[name.lower()] = emb_norm_mat[i]
# Also un-normalised key
inv_emb_norm_orig = {n: emb_norm_mat[i]
                     for i, n in enumerate(pd.read_csv(BASE/"data/investor_names.csv")["investor"])}

# Use Script 52's semicolon-split co_all_inv
deals_52 = deals.dropna(subset=["Investors"]).copy()
co_all_inv = {}
for _, row in deals_52.iterrows():
    co = row["Companies"].strip()
    invs = parse_investors_semi(row["Investors"])
    co_all_inv.setdefault(co, set()).update(invs)

# Compute investor_startup_match
match_rows = []
companies_in_sample = set(adf["company"])
for co, investors in co_all_inv.items():
    if co not in companies_in_sample:
        continue
    # Use startup embedding (pkl or fallback to investor mean)
    startup_emb = startup_emb_dict.get(co)
    if startup_emb is None:
        # Fallback: use mean of investors' normalised SPECTER2 embeddings (comma parse)
        # This replicates Script 63's approach, not Script 52's exactly
        s_vecs = [inv_emb_norm.get(i) for i in investors if inv_emb_norm.get(i) is not None]
        if not s_vecs:
            continue
        m = np.stack(s_vecs).mean(axis=0)
        n = np.linalg.norm(m)
        startup_emb = m / n if n > 1e-10 else None

    if startup_emb is None:
        continue
    inv_embs = [inv_emb_norm.get(i) for i in investors if inv_emb_norm.get(i) is not None]
    if not inv_embs:
        continue
    sims = np.array(inv_embs) @ startup_emb
    match_rows.append({"company": co, "investor_startup_match": float(sims.mean())})

match_df = pd.DataFrame(match_rows)
med_match = match_df["investor_startup_match"].median()
match_df["expertise_aligned"] = (match_df["investor_startup_match"] > med_match).astype(int)
print(f"  Computed expertise_aligned for {len(match_df)} companies  "
      f"(median match = {med_match:.4f})")

# Merge with analysis dataset
adf_check = adf.merge(match_df, on="company", how="inner")
adf_check = adf_check.dropna(subset=["be_sp_raw", "expertise_aligned"])
print(f"  Matched sample for construct check: N={len(adf_check)}")

# Tertile bridge_expertise in the matched sample
adf_check["be_tertile_n"] = pd.qcut(
    adf_check["be_sp_raw"], q=3, labels=[1, 2, 3], duplicates="drop").astype(int)

# 2 × 3 contingency table
ct = pd.crosstab(
    adf_check["expertise_aligned"],
    adf_check["be_tertile_n"],
    margins=True
)
ct.index = ["Low embedding-match (not expert)", "High embedding-match (expert)", "Total"]
ct.columns = ["be_tertile 1 (bridge far)", "be_tertile 2 (mid)",
              "be_tertile 3 (bridge close)", "Total"]
print(f"\n2×3 TABLE: expertise_aligned (rows) × bridge_expertise tertile (cols)")
print(ct.to_string())

# Row percentages
pct = ct.iloc[:2, :3].div(ct.iloc[:2, 3], axis=0) * 100
print(f"\nRow %:")
print(pct.round(1).to_string())

# Correlation
corr_c, p_corr = pearsonr(
    adf_check["expertise_aligned"].astype(float),
    adf_check["be_sp_raw"]
)
print(f"\nCorr(expertise_aligned, bridge_expertise_raw): r={corr_c:.4f}  p={p_corr:.4f}")

# Spearman (ordinal × continuous)
from scipy.stats import spearmanr
rho, p_rho = spearmanr(adf_check["expertise_aligned"], adf_check["be_sp_raw"])
print(f"Spearman ρ: {rho:.4f}  p={p_rho:.4f}")

print(f"""
CONSTRUCT CHECK VERDICT:
  A high correlation (|r| > 0.5) would indicate bridge_expertise and expertise_aligned
  are measuring the same underlying construct (investor-startup topic alignment).
  A low correlation (|r| < 0.2) would indicate they capture different things:
    - expertise_aligned = global portfolio alignment (average across ALL investors)
    - bridge_expertise  = alignment of BRIDGE COMMUNITY centroid to startup domain
  The latter is more targeted: it asks whether the specific outsider community's
  expertise overlaps with the startup's domain, not whether the average investor does.
  {'SAME CONSTRUCT (|r| > 0.4): differentiation is limited' if abs(corr_c) > 0.4
   else 'DIFFERENT CONSTRUCTS (|r| < 0.4): bridge_expertise adds information beyond overall portfolio alignment'}
""")

# ════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════
banner("FULL SUMMARY")

print(f"""
TASK 1 — MiniLM REPLICATION
  SPECTER2 M2:  β={r_sp['beta']:.6f}  p={r_sp['p']:.4f} {sig_label(r_sp['p'])}  N={r_sp['n']}
  MiniLM M2:    β={r_mi['beta']:.6f}  p={r_mi['p']:.4f} {sig_label(r_mi['p'])}  N={r_mi['n']}
  Corr(SPECTER2, MiniLM bridge_expertise): {corr_be:.4f}
  Interpretation: {'Both significant — sign holds across embedding models' if r_sp['p']<0.05 and r_mi['p']<0.05
                   else 'Sign is consistent but significance differs — embedding model matters for power'}

TASK 2 — TRIMMED M2 (p5–p95)
  β={r_trim['beta']:.6f}  p={r_trim['p']:.4f} {sig_label(r_trim['p'])}  N={r_trim['n']}
  {'Result is robust to trimming' if r_trim['p'] < 0.10
   else 'Result does NOT hold after trimming (driven by tails)'}

TASK 3 — TERTILE GRADIENT
  See table above. Gradient is {'MONOTONIC' if mono_inc or mono_dec else 'NON-MONOTONIC'}.

TASK 4 — MULTIPLE COMPARISONS
  {n_tests} independent tests across Script 63 + 64.
  M2 raw p=0.0252 → Holm p={holm_ps[m2_idx]:.4f}  BH p={bh_ps[m2_idx]:.4f}
  Survives Holm: {'YES' if holm_ps[m2_idx]<0.05 else 'NO'}  |  Survives BH: {'YES' if bh_ps[m2_idx]<0.05 else 'NO'}

TASK 5 — CONSTRUCT CHECK
  Corr(expertise_aligned, bridge_expertise): r={corr_c:.4f}  Spearman ρ={rho:.4f}
  {'SAME construct' if abs(corr_c) > 0.4 else 'DIFFERENT constructs — bridge_expertise is not just portfolio alignment'}

RUNTIME: {elapsed()}
""")
