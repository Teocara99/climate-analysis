# Company-level Cox Survival Model (Script 58 Part B)
*(Fixes Script 55 Part B: right-censoring + collinearity issues resolved)*

## Data & Design Decisions

- **N companies:** 5077 | **Events (failures):** 508 (10.0%)
- **Duration:** months from first deal (first_ym) to 2025-12-31
- **Censoring:** non-failed right-censored at observation end (correct) ✓
- **Failure timing:** treated as occurring at observation end (approx., no exact dates) ⚠

### Fixes vs Script 55 Part B

| Issue | Script 55 approach | Script 58 fix |
|---|---|---|
| Right-censoring | LPM, ignored | Cox PH with duration variable |
| siloing constant within cohort | Annual avg → std=0 within cohort year | Quarterly mapping via first_ym |
| policy_any = 92% of sample | — | Use cpu_narrow (continuous, full range) + policy_count |
| Cohort confound | LPM with year FE | entry_year_z as continuous covariate |

## Cox Model 1 — Siloing × CPU Interaction (main)

**Covariates:** siloing_quarterly_z + cpu_narrow_z + sil_x_cpu + log_capital_z + cross_community + n_rounds_z + specter_novelty_z + entry_year_z
**Penalized Cox** (ridge λ=0.01) to stabilize estimation

| Variable | HR | 95% CI | Sig |
|---|---|---|---|
| siloing_quarterly_z | 0.543 | [0.466, 0.632] | ******* |
| cpu_narrow_z | 1.326 | [1.178, 1.493] | ******* |
| sil_x_cpu ★ KEY | 0.998 | [0.883, 1.129] | **n.s.** |
| log_capital_z | 0.693 | [0.624, 0.770] | ******* |
| cross_community | 0.819 | [0.682, 0.985] | ***** |
| n_rounds_z | 0.753 | [0.673, 0.842] | ******* |
| specter_novelty_z | 1.037 | [0.954, 1.128] | **n.s.** |
| entry_year_z | 8.525 | [6.967, 10.432] | ******* |

**HR interpretation:** HR > 1 = higher failure hazard; HR < 1 = lower hazard (protective)

## Ecological Fallacy Check

| Level | Measure | Sil × policy direction | Sig | Interpretation |
|---|---|---|---|---|
| Quarterly aggregate (Script 56 M1) | β(sil × policy_any) | +0.127 | † | Higher siloing + policy → MORE deals |
| Quarterly aggregate (Script 56 M6 EnvP) | β(sil × EnvP) | +0.226 | * | Higher siloing + EnvP → MORE deals |
| Company Cox (this script, Cox 1) | HR(sil × cpu) | see above | — | Check sign vs quarterly |

If Cox HR(sil×cpu) < 1 (protective) while quarterly β > 0: compatible mechanisms —
aggregate: siloed networks generate more deals in policy episodes;
company: startups founded in high-siloing+high-CPU quarters are more embedded and fail less.
Both reinforce the echo-chamber amplification story.

If Cox HR > 1 (harmful) while quarterly β > 0: ecological fallacy —
the aggregate positive interaction is driven by deal count, not company survival.
The ecosystem generates more deals but individual companies are worse off.

## Output Files

- `output/cox_survival/cox_survival_panel.csv` — company-level dataset
- `output/cox_survival/cox_survival_results.md` — this file