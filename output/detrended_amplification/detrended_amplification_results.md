# Part A: Detrended Ecosystem Amplification (Script 58)
*(Fixes collinearity from Scripts 56–57: r(siloing, CPU) ≈ −0.40)*

## Motivation

Script 57 found β(CPU × siloing) ≈ +0.027 (n.s.) on the full 2014–2025 window.
The null was driven by r(siloing_l1, cpu_narrow) = -0.381 (*) —
both CPU and siloing trend together post-2021, making the interaction collinear
with the individual terms. Detrending removes the shared trend before estimation.

## Detrending — Correlation Reduction

| Method | r(siloing, CPU) before | r(sil_dt, cpu_dt) after | Reduction |
|---|---|---|---|
| Raw | -0.381 (*) | — | baseline |
| HP filter (λ=1600) | -0.381 | +0.220 (n.s.) | ✗ Partial |
| First differences | -0.381 | +0.213 (n.s.) | ✗ Partial |
| Trend residuals | -0.381 | +0.267 (†) | ✗ Partial |

## M8: Detrended CPU × Siloing (continuous interaction)

**Model:** Δlog(deals) ~ sil_dt × cpu_dt + trend + Q2/Q3/Q4 | NW HAC (lags=4)

| Model | N | β(sil_dt × cpu_dt) | SE | Sig | r(sil_dt,cpu_dt) | R² |
|---|---|---|---|---|---|---|
| Script 57 M6a (raw, undetrended) | 44 | +0.0272 | 0.0408 | n.s. | −0.381 ⚠ | 0.406 |
| M8 (HP): sil_dt × cpu_dt  [HP-filtered, N≈44] | 44 | **-0.0083** | 0.0379 | **n.s.** | +0.220 | 0.419 |
| M8 (FD): sil_dt × cpu_dt  [First-differenced, N≈44] | 43 | **-0.0352** | 0.0532 | **n.s.** | +0.213 | 0.483 |
| M8 (TR): sil_dt × cpu_dt  [Trend-residualized, N≈44] | 44 | **-0.0624** | 0.0314 | **†** | +0.267 | 0.422 |

## M9: Policy Dummies × HP-Detrended Siloing

**Diagnostic:** Do the marginal results from Script 56 (p=0.09, p=0.065) survive
when siloing is detrended? If yes → effect is real. If weaker → partly trend-driven.

| Model | N | β(sil_hp × policy) | SE | Sig | Script 56 raw β | Script 56 Sig |
|---|---|---|---|---|---|---|
| M9a: sil_hp × policy_any | 44 | **+0.0908** | 0.0698 | **n.s.** | +0.1272 | † |
| M9b: sil_hp × policy_neg | 44 | **-0.1098** | 0.1859 | **n.s.** | +0.0977 | n.s. |
| M9c: sil_hp × policy_pos | 44 | **+0.1963** | 0.1104 | **†** | +0.1135 | † |