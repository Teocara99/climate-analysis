"""
Script 76: Orthogonal Density Axes
====================================
Rotate (z_local, z_regional) into two uncorrelated components:
  overall_sparsity = -(z_local + z_regional) / 2   [sparse everywhere]
  niche_cohesion   = z_local - z_regional           [tight cluster, sparse field]

These are the two continuous axes for regression use.
Quadrants retained for descriptive labelling only.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
OUT  = BASE / "output" / "two_level_novelty"

def sig(p):
    return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "†" if p<0.10 else "n.s."

def banner(s): print(f"\n{'='*72}\n{s}\n{'='*72}")

# ──────────────────────────────────────────────────────────────
# LOAD CLEANED DATA
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(OUT / "density_novelty_clean.csv")
N  = len(df)
print(f"Loaded cleaned dataset: N={N}")

loc5  = df["local_density_5"].values.astype(float)
reg50 = df["regional_density_50"].values.astype(float)
names = df["name"].values
desc_len    = df["desc_len"].values.astype(float)
is_noise    = df["is_hdbscan_noise"].values.astype(float)
std_dist    = df["std_distinct"].values.astype(float)
specter_nov = df["specter_novelty"].values.astype(float)
reg_spar    = df["regional_sparsity"].values.astype(float)

# Load descriptions from novelty_scores.csv for face-validity display
meta = pd.read_csv(BASE / "output" / "novelty" / "novelty_scores.csv")
meta["name"] = meta["name"].str.strip()
desc_map = dict(zip(meta["name"], meta["Description"].fillna("")))
descs = np.array([str(desc_map.get(n, ""))[:75] for n in names])

# ──────────────────────────────────────────────────────────────
# STEP 1: CORRELATION BETWEEN RAW COMPONENTS
# ──────────────────────────────────────────────────────────────
banner("STEP 1: CORRELATION BETWEEN z_local AND z_regional")

z_loc = (loc5  - loc5.mean())  / loc5.std()
z_reg = (reg50 - reg50.mean()) / reg50.std()

r_pears, p_pears = pearsonr(z_loc, z_reg)
r_spear, p_spear = spearmanr(z_loc, z_reg)
print(f"  Pearson  r(z_local, z_regional) = {r_pears:+.4f}  p={p_pears:.2e} {sig(p_pears)}")
print(f"  Spearman ρ(z_local, z_regional) = {r_spear:+.4f}  p={p_spear:.2e} {sig(p_spear)}")
print(f"\n  This is the collinearity the rotation resolves.")

# ──────────────────────────────────────────────────────────────
# STEP 2: ROTATION INTO ORTHOGONAL AXES
# ──────────────────────────────────────────────────────────────
banner("STEP 2: ORTHOGONAL AXES")

overall_sparsity = -(z_loc + z_reg) / 2   # more negative z = denser → flip sign
niche_cohesion   =  (z_loc - z_reg)        # high local relative to regional

# Verify orthogonality
r_orth_p, p_orth_p = pearsonr(overall_sparsity, niche_cohesion)
r_orth_s, p_orth_s = spearmanr(overall_sparsity, niche_cohesion)

print(f"  overall_sparsity = -(z_local + z_regional) / 2")
print(f"  niche_cohesion   =   z_local - z_regional")
print(f"\n  overall_sparsity:  mean={overall_sparsity.mean():+.4f}  SD={overall_sparsity.std():.4f}  "
      f"range=[{overall_sparsity.min():+.3f}, {overall_sparsity.max():+.3f}]")
print(f"  niche_cohesion:    mean={niche_cohesion.mean():+.4f}  SD={niche_cohesion.std():.4f}  "
      f"range=[{niche_cohesion.min():+.3f}, {niche_cohesion.max():+.3f}]")
print(f"\n  Orthogonality check:")
print(f"    Pearson  r  = {r_orth_p:+.6f}  p={p_orth_p:.4f} {sig(p_orth_p)}")
print(f"    Spearman ρ  = {r_orth_s:+.6f}  p={p_orth_s:.4f} {sig(p_orth_s)}")
if abs(r_orth_p) < 0.001:
    print(f"  CONFIRMED: axes are exactly orthogonal by construction (sum/diff of equal-variance z-scores).")
else:
    print(f"  NOTE: small residual correlation due to non-Gaussianity in the raw densities.")

# ──────────────────────────────────────────────────────────────
# STEP 3: CORRELATIONS WITH EXTERNAL MEASURES
# ──────────────────────────────────────────────────────────────
banner("STEP 3: CORRELATIONS WITH EXTERNAL MEASURES")

ext_measures = [
    ("std_distinct",     std_dist),
    ("specter_novelty",  specter_nov),
    ("is_hdbscan_noise", is_noise),
    ("regional_sparsity",reg_spar),
    ("desc_len",         desc_len),
]

print(f"  {'Measure':<22} {'overall_sparsity':>18} {'niche_cohesion':>16}")
print(f"  {'-'*58}")
for label, arr in ext_measures:
    valid = ~np.isnan(arr)
    rs, ps = spearmanr(overall_sparsity[valid], arr[valid])
    rn, pn = spearmanr(niche_cohesion[valid],   arr[valid])
    ss = f"{rs:+.3f}{sig(ps)}"
    sn = f"{rn:+.3f}{sig(pn)}"
    print(f"  {label:<22} {ss:>18} {sn:>16}")

print(f"\n  Interpretation:")
print(f"    overall_sparsity × std_distinct ≈ ? (same-sign → both capture periphery)")
print(f"    niche_cohesion × specter_novelty should be negative (cohesion ≠ isolation)")

# ──────────────────────────────────────────────────────────────
# STEP 4: TOP 10 ON EACH AXIS
# ──────────────────────────────────────────────────────────────
banner("STEP 4: TOP 10 / BOTTOM 10 ON EACH AXIS")

for axis_name, axis_arr, hi_label, lo_label in [
    ("overall_sparsity", overall_sparsity,
     "HIGH overall_sparsity — sparse everywhere (lonely outliers)",
     "LOW  overall_sparsity — dense everywhere (mainstream)"),
    ("niche_cohesion",   niche_cohesion,
     "HIGH niche_cohesion — tight cluster, sparse field (novel niche)",
     "LOW  niche_cohesion — loose locally but field-dense (misfit)"),
]:
    for ascending, label in [(False, hi_label), (True, lo_label)]:
        order = np.argsort(axis_arr) if ascending else np.argsort(axis_arr)[::-1]
        top   = order[:10]
        print(f"\n  {label}")
        print(f"  {'rank':>4} {'score':>7}  {'loc':>6}  {'reg':>6}  {'name':<48}  description")
        print(f"  {'-'*115}")
        for rank, i in enumerate(top, 1):
            print(f"  {rank:>4} {axis_arr[i]:>+7.3f}  {loc5[i]:.4f}  {reg50[i]:.4f}  "
                  f"{str(names[i])[:47]:<48}  {descs[i]}")

# ──────────────────────────────────────────────────────────────
# SAVE FINAL OUTPUT
# ──────────────────────────────────────────────────────────────
banner("SAVING FINAL NOVELTY OUTPUT")

df_out = df.copy()
df_out["z_local_density"]   = z_loc
df_out["z_regional_density"]= z_reg
df_out["overall_sparsity"]  = overall_sparsity
df_out["niche_cohesion"]    = niche_cohesion

out_path = OUT / "novelty_final.csv"
df_out.to_csv(out_path, index=False)
print(f"  Saved: {out_path}  ({len(df_out):,} rows)")
print(f"  Regression-ready columns: overall_sparsity, niche_cohesion, std_distinct")
print(f"  Descriptive columns: quadrant, is_hdbscan_noise")
