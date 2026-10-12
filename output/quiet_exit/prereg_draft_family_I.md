
## Family I — Quiet Exit (Script 103 feasibility | do not lock)

**Status:** DRAFT — do not lock until review

**Data anchor:** PitchBook Investors_data.xlsx exported 2026-08-26.
Rolling-window columns "Investments in the last N years" are relative to this date.

### Definitions

| Term | Definition |
|---|---|
| Export date | 2026-08-26 (Last Updated Date, all rows identical) |
| Last-2yr window | 2024-08-26 – 2026-08-26 |
| Last-5yr window | 2021-08-26 – 2026-08-26 |
| Base population | Investors matched to Investors_data.xlsx with ≥1 climate deal in 5-yr window |
| stayer | ≥1 climate deal in 2-yr window (our data) |
| quiet_exit | 0 climate deals in 2-yr window AND pb_inv_2yr > 0 (still making deals elsewhere) |
| full_exit | 0 climate deals in 2-yr window AND pb_inv_2yr = 0 |
| green_focus | GREEN_VC / ESG / TRADITIONAL (from all_investors.csv classification) |
| intensity | prior climate deals / pb_total_inv (before 2-yr window) |
| portfolio_novelty | mean specter_novelty of ventures funded by investor (pre-2yr) |

### Feasibility numbers (Script 103, not pre-specified)
Base population: 13,730
 — stayer: 5,707 (41.6%)
 — quiet_exit: 309 (2.3%)
 — full_exit: 7,714 (56.2%)

Communities ≥30 base investors: 27
Investors with portfolio novelty: 13,301 / 13,730

### Q1 (snapshot, descriptive — no test)
**What share of climate-active investors went quiet in the last two years?**
Estimate quiet-exit share = quiet_exit / (quiet_exit + stayer + full_exit).
Report N and share by green_focus and investor type. No inferential test.

### Q2 (pre-registered, directional)
**Did quiet exiters fund more novel ventures (higher specter_novelty) than stayers?**
Prediction: positive (quiet exiters backed more frontier/novel startups).
Unit: investor × portfolio_novelty.
Test: OLS with intensity quintile FE; SE clustered by community where available.
Outcome: portfolio_novelty (mean specter_novelty of pre-2yr climate portfolio).
Min cell: quiet_exit with portfolio_novelty ≈ 302.
Pre-specified sign: positive coefficient on quiet_exit dummy.

### Q3 (pre-registered, directional)
**Within intensity quintiles, do GREEN_VC investors quiet-exit more than TRADITIONAL?**
Prediction: positive (GREEN_VC has higher quiet-exit rate than TRADITIONAL within quintile).
Test: logit quiet_exit ~ green_focus dummies + intensity quintile FE.
Reference: TRADITIONAL. Pre-specified sign: positive coefficient on GREEN_VC.
Min cells: GREEN_VC × stayer/quiet_exit = 3,505; TRADITIONAL × stayer/quiet_exit = 2,373.

### Q4 (pre-registered, two-sided)
**Are quiet exits concentrated in particular investor communities beyond chance?**
Test: permutation test (999 draws) — observed community-level quiet-exit rate vs
random allocation of quiet_exit labels across base population.
Null: community membership does not predict quiet-exit status.
Test statistic: max(community quiet_exit share) − overall quiet_exit share.
Restrict to communities with ≥30 base-population investors (N communities = 27).

### Holm correction
Applied across Q2, Q3, Q4. Q1 is descriptive only.

---
*Do not lock this section. Review cell counts and definitions before locking.*
