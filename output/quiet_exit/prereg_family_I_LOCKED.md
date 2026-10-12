
## Family I — Quiet Exit  [LOCKED 2026-10-05]

**Status: LOCKED**

---

### Data and windows

| Item | Value |
|---|---|
| PitchBook export date | 2026-08-26 (Last Updated Date, all rows) |
| Deals-file cutoff | 2026-06-20 (max Deal Date) |
| Window end used | 2026-06-20 (deals-file cutoff — earlier than PB export) |
| Last-2yr window | 2024-06-20 – 2026-06-20 |
| Last-5yr window | 2021-06-20 – 2026-06-20 |

### Matching and collision filter

| Step | N investors | Share of deal-data |
|---|---|---|
| Unique investors in deal data | 23,838 | 100% |
| Matched to PB (pre-filter) | 18,280 | 76.7% |
| Excluded — single-word name | 6,430 | — |
| Excluded — country mismatch | 1,550 | — |
| Retained (clean matches) | 10,300 | 43.2% |

### Definitions

| Term | Definition |
|---|---|
| active_overall | PB Last Investment Date ≥ 2024-06-20 |
| stayer | ≥1 climate deal in 2yr window (our data) |
| quiet_exit | active_overall AND 0 climate deals in 2yr window |
| full_exit | Last Investment Date < 2024-06-20 |
| Base population | clean-matched investors with ≥1 climate deal in 5yr window |
| intensity | prior_climate_deals / pb_total_inv (deals before 2024-06-20) |
| portfolio_novelty | mean specter_novelty of ventures funded (pre-2yr window) |
| green_focus | GREEN_VC / ESG / TRADITIONAL (all_investors.csv) |

### Feasibility (Script 104, not pre-specified)

Base population: 7,751
 — stayer    : 3,116 (40.2%)
 — quiet_exit: 2,344 (30.2%)
 — full_exit : 2,291 (29.6%)

Quiet-exit validation: 5.6% of quiet exiters have a cleantech keyword
in their PB last investee company but that company is NOT in our sample
(potential missed coverage, not a classification error per se).

Portfolio novelty — stayers: 0.4217 | quiet_exit: 0.4120

---

### Q1 — Snapshot (descriptive, no test)

**What share of climate-active investors went quiet in the last two years?**

Estimand: quiet_exit / base_population.
Report separately by green_focus and PB investor type.
No inferential test. Script 103/104 provide the numbers.

---

### Q2 — Portfolio novelty (pre-registered, directional)

**Did quiet exiters fund more novel ventures than stayers?**

Outcome: portfolio_novelty (mean specter_novelty across investor's pre-2yr climate portfolio).
Unit of observation: investor.
Comparison: quiet_exit vs stayer (exclude full_exit).
Model: OLS portfolio_novelty ~ quiet_exit_dummy + intensity_quintile_FEs.
   SE clustered by community (min-size ≥30) where available; HC3 otherwise.
Pre-specified sign: **positive** coefficient on quiet_exit_dummy.
   (Theoretical prediction: investors who exit climate backed more novel, harder-to-scale
   startups, consistent with the echo-chamber mechanism.)

Min cell: quiet_exit with portfolio_novelty = 2,270.
Feasibility note: observed gap = -0.0097. Power likely low.

---

### Q3 — GREEN_VC quiet-exit rate (pre-registered, directional)

**Within intensity quintiles, do GREEN_VC investors quiet-exit at higher rates than
TRADITIONAL investors?**

Outcome: binary quiet_exit (1) vs stayer (0); exclude full_exit.
Unit: investor.
Model: logit quiet_exit ~ green_focus_dummies + intensity_quintile_FEs.
   Reference: TRADITIONAL.
Pre-specified sign: **positive** coefficient on GREEN_VC dummy
   (GREEN_VC investors are specialised; when they exit climate, they exit entirely,
   whereas TRADITIONAL investors merely reduce exposure).
Comparison is within intensity quintiles (quintile FEs control for baseline commitment).

Min cells: GREEN_VC stayer+quiet_exit = 3,281;
           TRADITIONAL stayer+quiet_exit = 2,085.

---

### Q4 — Community concentration (pre-registered, two-sided)

**Are quiet exits concentrated in particular investor communities beyond chance?**

Test: permutation test (999 draws).
Statistic: max(community quiet_exit share) − overall quiet_exit share,
   across communities with ≥30 base-population investors.
   (N qualifying communities = 26.)
Null: community membership does not predict quiet-exit status.
   p-value = share of permuted statistics ≥ observed.

---

### Holm correction

Applied across Q2, Q3, Q4 (three pre-registered tests).
Q1 is descriptive and excluded from Holm.

---

*Locked by Script 104 on 2026-10-05. No further changes without a dated amendment.*
