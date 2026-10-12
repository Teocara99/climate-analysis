"""
Script 77: Robustness Check on Novelty Axes
=============================================
For any regression using overall_sparsity or niche_cohesion, three sample cuts:
  (a) full merged sample
  (b) excluding top 1% of overall_sparsity
  (c) excluding top 2% of overall_sparsity AND bottom 1% of niche_cohesion

Demonstrated on:
  M1: failure ~ overall_sparsity + controls
  M2: failure ~ niche_cohesion   + controls
  M3: failure ~ overall_sparsity + niche_cohesion + controls  [joint]
  M4: failure ~ std_distinct_resid + controls  [within-category axis]

Controls: log_capital + n_rounds + C(deal_year) + C(hq_country)
SE: HC1 sandwich throughout.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

warnings.filterwarnings("ignore")

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

# ──────────────────────────────────────────────────────────────
# HC1 OLS
# ──────────────────────────────────────────────────────────────
def ols_hc1(y, X):
    """Return (betas, SEs, p-values) with HC1 sandwich."""
    N, k = X.shape
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    res  = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = (X * res[:, None]).T @ (X * res[:, None]) * N / (N - k)
    V    = XtX_inv @ meat @ XtX_inv
    ses  = np.sqrt(np.clip(np.diag(V), 0, None))
    ts   = beta / np.where(ses > 1e-20, ses, np.nan)
    ps   = f_dist.sf(ts**2, 1, N - k)
    return beta, ses, ps

def build_Xy(df, focal_vars, outcome="failed"):
    """Build y and design matrix with year/country FE and focal vars."""
    sub = df[focal_vars + [outcome, "log_capital", "n_rounds",
                           "deal_year", "hq_country"]].dropna().copy()
    if len(sub) < 100:
        return None, None, None
    y = sub[outcome].values.astype(float)

    # Dummies for year and country
    yr_dummies  = pd.get_dummies(sub["deal_year"],  prefix="yr",  drop_first=True).astype(float)
    co_dummies  = pd.get_dummies(sub["hq_country"], prefix="co",  drop_first=True).astype(float)

    parts = [np.ones((len(sub), 1)),
             sub[["log_capital", "n_rounds"]].values.astype(float),
             yr_dummies.values,
             co_dummies.values,
             sub[focal_vars].values.astype(float)]
    X = np.column_stack(parts)
    n_focal = len(focal_vars)
    return y, X, n_focal   # focal vars are the LAST n_focal columns

def run_robust(df, focal_vars, model_label, outcome="failed"):
    """Run (a), (b), (c) and return result dict."""
    os_p99 = np.percentile(df["overall_sparsity"], 99)
    os_p98 = np.percentile(df["overall_sparsity"], 98)
    nc_p01 = np.percentile(df["niche_cohesion"],   1)

    samples = {
        "(a) full":        df,
        "(b) excl OS>p99": df[df["overall_sparsity"] <= os_p99],
        "(c) excl OS>p98 & NC<p01": df[(df["overall_sparsity"] <= os_p98) &
                                        (df["niche_cohesion"]   >= nc_p01)],
    }

    results = {}
    for label, sub_df in samples.items():
        y, X, n_focal = build_Xy(sub_df, focal_vars, outcome)
        if y is None:
            results[label] = None
            continue
        out = ols_hc1(y, X)
        if out is None:
            results[label] = None
            continue
        beta, ses, ps = out
        # focal vars are the last n_focal columns
        results[label] = {
            "N":    len(y),
            "coef": beta[-n_focal:],
            "se":   ses[-n_focal:],
            "p":    ps[-n_focal:],
        }
    return results

def print_table(model_label, focal_vars, results):
    cut_labels = list(results.keys())
    print(f"\n  {model_label}")
    # Header
    print(f"  {'Variable':<22}", end="")
    for cl in cut_labels:
        n = results[cl]["N"] if results[cl] else 0
        print(f"  {cl} (N={n:,})", end="")
    print()
    print(f"  {'-'*100}")

    for vi, var in enumerate(focal_vars):
        print(f"  {var:<22}", end="")
        prev_sig = None
        for ci, cl in enumerate(cut_labels):
            r = results[cl]
            if r is None:
                print(f"  {'—':>22}", end="")
                continue
            b, se, p = r["coef"][vi], r["se"][vi], r["p"][vi]
            cell = f"β={b:+.4f} SE={se:.4f} {sig(p)}"
            print(f"  {cell:<26}", end="")
            if ci == 1 and prev_sig is not None:
                # compare (a) sign/sig with (b)
                pass
            prev_sig = (b, p)
        print()

    # Flag sign changes or significance losses (a) → (b)
    for vi, var in enumerate(focal_vars):
        ra = results.get("(a) full")
        rb = results.get("(b) excl OS>p99")
        if ra is None or rb is None: continue
        ba, pa = ra["coef"][vi], ra["p"][vi]
        bb, pb = rb["coef"][vi], rb["p"][vi]
        if np.sign(ba) != np.sign(bb):
            print(f"  !!! SIGN CHANGE (a)→(b) on {var}: "
                  f"β(a)={ba:+.4f} β(b)={bb:+.4f}")
        elif pa < 0.05 and pb >= 0.10:
            print(f"  !!! SIGNIFICANCE LOSS (a)→(b) on {var}: "
                  f"p(a)={pa:.4f} → p(b)={pb:.4f}")
        elif pa < 0.05 and pb >= 0.05:
            print(f"  ↓  WEAKENS (a)→(b) on {var}: "
                  f"p(a)={pa:.4f} → p(b)={pb:.4f}")

# ──────────────────────────────────────────────────────────────
# LOAD AND MERGE
# ──────────────────────────────────────────────────────────────
print("="*72)
print("SCRIPT 77: NOVELTY AXES ROBUSTNESS CHECK")
print("="*72)

nf  = pd.read_csv("output/two_level_novelty/novelty_final.csv")
cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")

cdf["co"] = cdf["Companies"].str.strip()
nf["co"]  = nf["name"].str.strip()

df = cdf.merge(nf, on="co", how="inner", suffixes=("_cpu","_nov"))
# std_distinct comes from novelty_final; rename if needed
if "std_distinct_nov" in df.columns:
    df["std_distinct"] = df["std_distinct_nov"]

print(f"\n  Merged analysis sample: N={len(df):,}")
print(f"  Failure rate: {df['failed'].mean():.3f}")

# Cut-off values
os_p99 = np.percentile(df["overall_sparsity"], 99)
os_p98 = np.percentile(df["overall_sparsity"], 98)
nc_p01 = np.percentile(df["niche_cohesion"],   1)
print(f"\n  Cut-offs:")
print(f"    overall_sparsity p99 = {os_p99:.3f}  → drops {(df['overall_sparsity']>os_p99).sum()} rows")
print(f"    overall_sparsity p98 = {os_p98:.3f}")
print(f"    niche_cohesion   p01 = {nc_p01:.3f}")
print(f"    (c) combined:         drops {((df['overall_sparsity']>os_p98)|(df['niche_cohesion']<nc_p01)).sum()} rows")

# Residualise std_distinct on overall_sparsity (for M4)
# Use only rows with both variables
valid_sd = ~(np.isnan(df["std_distinct"]) | np.isnan(df["overall_sparsity"]))
X_res = np.column_stack([np.ones(valid_sd.sum()),
                          df.loc[valid_sd, "overall_sparsity"].values])
y_res = df.loc[valid_sd, "std_distinct"].values
b_res = np.linalg.lstsq(X_res, y_res, rcond=None)[0]
df.loc[valid_sd, "std_distinct_resid"] = y_res - X_res @ b_res
resid_sd = df["std_distinct_resid"].std()
print(f"\n  std_distinct residualised on overall_sparsity:")
print(f"    SD(resid) = {resid_sd:.4f}")
from scipy.stats import spearmanr
r1, p1 = spearmanr(df["std_distinct_resid"].dropna(), df["overall_sparsity"][df["std_distinct_resid"].notna()])
r2, p2 = spearmanr(df["std_distinct_resid"].dropna(), df["niche_cohesion"][df["std_distinct_resid"].notna()])
print(f"    ρ(resid, overall_sparsity): {r1:+.4f}  {sig(p1)}")
print(f"    ρ(resid, niche_cohesion):   {r2:+.4f}  {sig(p2)}")
# Top 10 on residual
top10_resid = df[["co","std_distinct_resid"]].dropna().nlargest(10,"std_distinct_resid")
print(f"    Top 10 on std_distinct_resid:")
for _, row in top10_resid.iterrows():
    print(f"      [{row['std_distinct_resid']:+.3f}]  {row['co'][:60]}")

print(f"\n{'='*72}")
print("REGRESSION RESULTS: (a) full / (b) excl OS>p99 / (c) excl OS>p98 & NC<p01")
print("="*72)
print("Controls: log_capital + n_rounds + year FE + country FE  |  HC1 SEs")

# M1: overall_sparsity
r1 = run_robust(df, ["overall_sparsity"], "M1")
print_table("M1: failure ~ overall_sparsity + controls", ["overall_sparsity"], r1)

# M2: niche_cohesion
r2 = run_robust(df, ["niche_cohesion"], "M2")
print_table("M2: failure ~ niche_cohesion + controls", ["niche_cohesion"], r2)

# M3: joint
r3 = run_robust(df, ["overall_sparsity", "niche_cohesion"], "M3")
print_table("M3: failure ~ overall_sparsity + niche_cohesion + controls",
            ["overall_sparsity", "niche_cohesion"], r3)

# M4: std_distinct_resid
r4 = run_robust(df, ["std_distinct_resid"], "M4")
print_table("M4: failure ~ std_distinct_resid + controls", ["std_distinct_resid"], r4)

# M5: all three axes
r5 = run_robust(df, ["overall_sparsity","niche_cohesion","std_distinct_resid"], "M5")
print_table("M5: all three axes jointly",
            ["overall_sparsity","niche_cohesion","std_distinct_resid"], r5)

print(f"\n{'='*72}")
print("SUMMARY")
print("="*72)
print("  Models that change sign or lose significance (a)→(b): flagged above.")
print("  Stable coefficients across all three sample cuts → robust to outliers.")
print("  If (b) or (c) differ materially from (a): trim at p98 before reporting.")
