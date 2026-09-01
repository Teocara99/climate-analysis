# Policy-Shock Fragility Extension (Script 56)
*(Extends Script 55 — sector-specific policy shocks instead of generic financial uncertainty)*

## Data

| Source | Coverage | Quarters in panel |
|---|---|---|
| Quarterly siloing panel | 2014Q1–2025Q4 | 46 |
| Macro shocks (VIX/rate/GPR) | 2014Q1–2025Q4 | 46 |
| **Noailly et al. (2024) EnvP index** [doi:10.7910/DVN/TCX06Z] | Jan 1981–Mar 2019 | 19 of 46 |
| **Noailly et al. (2022) EnvPU index** [doi:10.7910/DVN/8TKRSJ] | Jan 1990–Mar 2019 | 19 of 46 |

**EnvP/EnvPU limitation:** Both indices end March 2019, so they cannot cover the three most consequential post-2020 policy shocks (Biden 2021, IRA 2022, Trump 2.0 2025). M6/M7 are restricted to 19 quarters (2014Q1–2019Q1).

---

## Policy Event Dummies

| Direction | Quarter | siloing_l1 | Δlog(deals) | EnvP | Event |
|---|---|---|---|---|---|
| POS | 2015Q4 | 0.909 | +0.383 | 184 | Paris Agreement adopted (Dec 2015) |
| NEG | 2016Q4 | 0.750 | +0.051 | 159 | Trump election (Nov 2016) |
| NEG | 2017Q2 | 0.733 | +0.000 | 182 | Trump announces Paris withdrawal (Jun 2017) |
| NEG | 2019Q4 | 0.697 | −0.186 | N/A | Formal US Paris withdrawal notification |
| POS | 2021Q1 | 0.683 | +0.307 | N/A | Biden inauguration + Paris re-entry |
| POS | 2022Q3 | 0.621 | −0.212 | N/A | Inflation Reduction Act signed (Aug 2022) |
| NEG | 2025Q1 | 0.676 | +0.304 | N/A | Trump 2.0 inauguration + IRA repeal threats |

**Observation:** Most negative policy events show positive deal-flow in the shock quarter — the ecosystem does not contract on impact. The IRA signing (2022Q3) is negative because rate hikes dominated that quarter. Trump 2.0 (2025Q1) shows +30% — possibly a rush of deals before anticipated policy squeeze.

---

## Confound Check: Correlation of Siloing_L1 with Each Shock Variable

| Shock variable | r(siloing_l1, shock) | Comment |
|---|---|---|
| policy_any (7 events) | −0.011 n.s. | ✓ Near-zero — no trend confound |
| policy_neg (4 events) | −0.042 n.s. | ✓ Near-zero |
| policy_pos (3 events) | +0.032 n.s. | ✓ Near-zero |
| rate_hike (4 quarters) | **−0.328 \*** | ⚠ Trending confound — see below |
| vix_spike (12 quarters) | −0.238 n.s. | Moderate but not significant |

**Key advantage of policy shocks over rate shocks:** Policy events span 2015–2025 across all seasons, whereas rate shocks cluster in 2022–2023 when siloing also happened to be elevated. This zero correlation is a major identification improvement over Script 55.

---

## Main Regression Results

**Model:** Δlog(deals)_t = α + β₁·siloing_{t−1} + β₂·shock_t + **β₃·(siloing_{t−1} × shock_t)** + trend + Q₂ + Q₃ + Q₄ + ε
**SEs:** Newey-West HAC (lags=4) | **KEY COEFFICIENT:** β₃

| Model | N | N_shock | β(siloing × shock) | SE | Sig | r(sil, shock) |
|---|---|---|---|---|---|---|
| M1: Any policy shock (7 events) | 44 | 7 | **+0.1272** | 0.0752 | **†** | −0.011 |
| M2: Negative policy shocks (4 events) | 44 | 4 | +0.0977 | 0.2643 | n.s. | −0.042 |
| M3: Positive policy shocks (3 events) | 44 | 3 | **+0.1135** | 0.0614 | **†** | +0.032 |
| M4: Rate hike >50bp/Q (Script 55 repl.) | 44 | 4 | +0.4363 | 0.1931 | * | −0.328 ⚠ |
| M5: VIX spike >p75 (Script 55 repl.) | 44 | 12 | −0.0351 | 0.0778 | n.s. | −0.238 |
| **M6: EnvP continuous (N=19, 2014–2019)** | 17 | — | **+0.2263** | 0.0852 | **\*** | −0.196 |
| M7: EnvPU continuous (N=19, 2014–2019) | 17 | — | +0.0473 | 0.1157 | n.s. | −0.245 |

---

## Critical Correction to Script 55: Rate-Hike Sign Flip

Script 55 reported β(siloing × rate_hike) = **−0.303\*\*\*** (without quarter controls).
Script 56 with Q2/Q3/Q4 dummies reports β = **+0.436\*** — the sign REVERSES.

**Why:** All 4 rate-hike quarters (2022Q2–2023Q1) have **below-average siloing** (siloing_l1_z = −0.63 to −1.33). The interaction variable = siloing_l1_z × shock_rate = negative values in all 4 shock quarters. Without quarter controls, the model conflated Q2/Q3 seasonal deal-flow patterns (typically lower) with the rate × siloing interaction. Adding Q2/Q3/Q4 dummies reveals that deal flow in rate-hike quarters was unremarkable once seasonality is removed, reversing the coefficient. **The Script 55 \*\*\* finding for rate hikes was a seasonal confound artifact and should be retracted.**

---

## Comparative Test: Policy vs Financial Interactions (M_comp)

**Single model** estimating both interactions jointly:

| Term | β | SE | Sig |
|---|---|---|---|
| siloing × policy_any | +0.1523 | 0.0653 | * |
| siloing × vix_spike | −0.0672 | 0.0900 | n.s. |

Wald test H₀ (policy = vix): **F = 0.845, p = 0.364 n.s.**
→ Cannot statistically distinguish the two, though the policy interaction is individually significant and the VIX is not.

| Term | β | SE | Sig |
|---|---|---|---|
| siloing × policy_any | +0.1264 | 0.0762 | n.s. |
| siloing × rate_hike | +0.4265 | 0.2158 | † |

Wald test H₀ (policy = rate): **F = 0.304, p = 0.585 n.s.**

---

## Interpretation

### Sign of the interaction: Echo-chamber reinforcement, not AER-2015 cascade

All policy-shock interactions show a **positive sign** (β ≈ +0.10–0.23). Under the Acemoglu et al. (2015) "robust yet fragile" mechanism, we would expect **negative** interactions: larger macro shocks should cascade more severely through siloed networks, reducing deal flow. Instead, we find the opposite.

A positive interaction means: **when the network is more siloed and a policy event occurs, deal flow is higher than expected** — siloing amplifies the response to policy signals upward.

This is consistent with an **echo-chamber reinforcement mechanism**:
1. Siloed investors are embedded in communities of like-minded peers
2. Policy signals (both positive and negative) are interpreted through this shared lens
3. Positive policy signals get amplified within-community → more deals
4. Negative signals are buffered by echo-chamber consensus → less contraction than expected

This is NOT fragility in the cascade sense. It is a different structural effect: the network is **interpretation-robust** to policy shocks — communities converge on a shared reaction rather than dispersing into uncertainty.

### M6 (EnvP): Strongest individual result

β(EnvP_z × siloing) = +0.226\* (N=19, 2014–2019). The Noailly et al. (2024) continuous index shows that when environmental policy salience is high AND the network is more siloed, quarterly deal flow is significantly higher. This is the most precisely identified estimate but is restricted to the pre-2020 period.

### M4 reconsidered

The positive β(rate × siloing) = +0.436 suggests that during rate hikes (all in 2022), networks with lower siloing (more cross-community) experienced steeper deal-flow declines. Cross-community deals are riskier (paper's main finding) and thus more sensitive to tightening financial conditions — consistent with the rest of the paper's narrative but not the AER 2015 cascade story.

### Power constraints

With T ≤ 44 quarters and 3–7 shock events, detecting moderate interactions (β ~ 0.05–0.10) at p < 0.05 requires very strong signal-to-noise. The NW HAC correction (lags=4) further reduces effective sample size. M1 and M3 reach only marginal significance (†). M2 is underpowered with only 4 events. The EnvP continuous measure (M6) offers the best power for the 2014–2019 window.

---

## Summary for Paper

| Key claim | Evidence | Strength |
|---|---|---|
| Script 55 rate-hike *** was a confound | Sign flips with quarter controls | Strong — specification check |
| Policy shocks × siloing: positive direction | M1 † (p=0.09), M3 † (p=0.065) | Marginal |
| EnvP × siloing: positive and significant | M6 * (p=0.017) | Moderate (but N=19 only) |
| Policy ≠ financial interaction | Wald F<1, cannot reject equality | Null result |
| Confound cleared for policy shocks | r(siloing, policy) ≈ 0.01 | Clean identification |

The pattern points toward an **echo-chamber amplification mechanism** rather than a fragility/cascade mechanism. High-siloing networks respond more vigorously to policy signals — amplifying investment within-community. This reframes the contribution away from systemic risk (AER 2015) toward information processing and coordination within siloed investor communities.

## Output Files

- `output/policy_shock_fragility/fragility_panel_with_policy_shocks.csv` — augmented panel (N=46)
- `output/policy_shock_fragility/policy_shocks_fragility_results.md` — this file
