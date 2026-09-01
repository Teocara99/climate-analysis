# Structural Folds vs Structural Holes — Regression Results
*(Vedres & Stark 2010, AJS — adapted for single-assignment Louvain communities)*

## Setup

**Adaptation note:** PitchBook Louvain detection assigns each investor to exactly **one** community (46 communities, N=17,790 investors). "Bridge investors" here are investors whose **co-investors** span 2+ communities — they sit at the boundary but are formally assigned to one. Ego-network density is computed *among the investor's neighbors* within each community they reach.

- **HOLE (Burt-style):** mean within-community ego-density < global median (1.0000)
- **FOLD (Vedres-Stark multiple insider):** mean within-community ego-density ≥ global median

**2SLS spec (replicating Script 30):**
- Endogenous: `cross_community`
- Instrument: IV_A = lead investor's prior community reach (# unique communities touched in prior 2 years)
- Controls: `log_cap`, `specter_novelty`, `company_age`, year FE, region FE
- Outcome: `is_failed`

---

## Step 2–3: Bridge Investor Classification

| Metric | Value |
|---|---|
| Total investors with co-investment data | 25,278 |
| Bridge investors (neighbors in ≥2 communities) | 19,630 (77.7%) |
| Global median within-community ego-density | 1.0000 |
| FOLD (≥ median density) | 11,801 (60.1%) |
| HOLE (< median density) | 7,074 (36.0%) |
| UNCLASSIFIED (all communities degenerate) | 755 (3.8%) |

---

## Step 4: Company-Level Bridge Type

Cross-community companies only (N=4,137):

| Bridge Type | N companies | Failure Rate |
|---|---|---|
| FOLD-dominated | 285 | 5.3% |
| HOLE-dominated | 3,691 | 7.1% |
| MIXED | 161 | — |
| NO_BRIDGE (cross-comm, no classified bridge investor) | 0 | — |
| Non-cross-community (reference) | 2,581 | 10.5% |

---

## Step 5: 2SLS Results

| Model | N | β(cross_community) | SE | p | First-stage F |
|---|---|---|---|---|---|
| M0: Full sample | 6705 | +0.0431 | +0.0400 | 0.2809 | N/A |
| M1: FOLD-bridge companies | 389 | -0.2061 | +0.0608 | 0.0008 | 23.8 |
| M2: HOLE-bridge companies | 5683 | +0.1026 | +0.0514 | 0.0460 | N/A |

**OLS pooled interaction (M4):** `is_failed ~ cross_community + has_fold + has_hole + cross_community×has_fold + cross_community×has_hole + controls`
*(see console output for full M4 coefficients and Wald test)*

---

## Step 6: Specification Flags

1. **UNCLASSIFIED investors** (all communities have <2 neighbors in them): cannot compute ego-density. These are structural singles — by Burt's logic they are HOLES. Conservative robustness check: recode UNCLASSIFIED → HOLE.

2. **Density based on n=2 neighbors** in a community: density is trivially 0 or 1 (one possible edge). Affects 99.7% of bridge investors.

3. **Valid density from only 1 community**: 5,949 investors (30.3%). These cannot satisfy the Vedres-Stark criterion of being an insider in *multiple* communities. Strict FOLD requires valid density in ≥2 communities: N=7,394 (37.7%).

4. **First-stage F < 10** (if observed): weak instruments warning. The IV was validated in Script 30 on the full sample; subgroup F-statistics may be lower due to smaller N.

## Output Files

- `output/structural_folds/bridge_type_folds_holes.csv` — investor-level classification (N=19,630)
- `output/structural_folds/structural_folds_test_results.md` — this file
