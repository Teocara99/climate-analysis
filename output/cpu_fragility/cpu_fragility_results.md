# CPU × Siloing Fragility — Full 2014–2025 Window (Script 57)
*(Extends Script 56 M6/M7 — uses Gavriilidis CPU index instead of Noailly EnvP)*

## Data

| Source | Coverage | Quarters in panel |
|---|---|---|
| Gavriilidis (2021) CPU_narrow | Jan 1985 – Apr 2026 | 46 of 46 |
| Gavriilidis (2021) CPU_broad  | Jan 1985 – Apr 2026 | 46 of 46 |
| *Script 56 EnvP (Noailly 2024)* | *Jan 1981 – Mar 2019* | *~19 of 46* |

**Key advantage over Script 56 M6:** CPU_narrow and CPU_broad cover the **full
2014–2025 panel window** (N≈44) including Biden 2021, IRA 2022, and Trump 2.0 2025 —
the three highest-stakes policy quarters that EnvP misses.

**Reference:** Gavriilidis, K. (2021). Measuring Climate Policy Uncertainty.
SSRN Working Paper 3847388. Data: `data/cpu_gavriilidis_us.csv`.

---

## Main Results

**Model:** Δlog(deals)_t = α + β₁·siloing_{t−1} + β₂·CPU_z_t + **β₃·(siloing_{t−1} × CPU_z_t)** + trend + Q₂ + Q₃ + Q₄ + ε
**SEs:** Newey-West HAC (lags=4) | **KEY COEFFICIENT:** β₃

| Model | N | β(sil × CPU) | SE | Sig | r(sil,CPU) | R² |
|---|---|---|---|---|---|---|
| M6a: CPU_narrow (full 2014–2025) | 44 | **+0.0272** | 0.0408 | **n.s.** | -0.381 ⚠ | 0.406 |
| M6b: CPU_broad  (full 2014–2025) | 44 | **+0.0263** | 0.0434 | **n.s.** | -0.432 ⚠ | 0.406 |
| M6c: CPU_narrow (2014–2019 only, N≈19) | 20 | **+0.0600** | 0.1182 | **n.s.** | -0.351 ⚠ | 0.456 |

---

## Confound Check

| Measure | r(siloing_l1, CPU) | Comment |
|---|---|---|
| CPU_narrow | -0.381 | ⚠ Trending confound |
| CPU_broad  | -0.432 | ⚠ Trending confound |

---

## Interpretation

### Sign direction

M6a (CPU_narrow): β = +0.0272 (n.s.)
M6b (CPU_broad):  β = +0.0263 (n.s.)
M6c (CPU_narrow, 2014–2019): β = +0.0600 (n.s.)

A positive β(sil × CPU) means: when the network is more siloed AND climate
policy uncertainty is higher, deal flow is HIGHER than expected — consistent
with the echo-chamber reinforcement mechanism found in Script 56 for all
policy shock specifications. Siloed networks amplify within-community response
to policy signals rather than cascading into crisis (the AER 2015 prediction).

### Power gain vs Script 56 M6

| | Script 56 M6 (EnvP) | Script 57 M6a (CPU) |
|---|---|---|
| Index | Noailly EnvP | Gavriilidis CPU_narrow |
| Window | 2014–2019 | 2014–2025 |
| N | ~19 | 44 |
| Includes post-2020 shocks | No | Yes |

---

## Output Files

- `output/cpu_fragility/cpu_siloing_panel.csv` — merged panel
- `output/cpu_fragility/cpu_fragility_results.md` — this file