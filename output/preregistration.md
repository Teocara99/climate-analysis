# Pre-registration: IRA Natural Experiment — Climate-Tech Novelty & Network Structure

**Original timestamp:** 2026-09-21 14:17 UTC
**Revision timestamp:** 2026-09-22 07:26 UTC
**Previous revision timestamp:** 2026-09-21 14:29 UTC
**Status: LOCKED — no further changes to hypotheses or cluster definitions**
**Study:** How did the Inflation Reduction Act (IRA, signed 2022-08-16) reshape the
novelty profile and investor network structure of climate-tech startups?
**Repository:** Breaking the Echo Chamber (climate_analysis)
**Contact:** teocara98@gmail.com

---

## Cluster validation notes (completed before lock)

**Cluster 4 (EVs/mobility) — partial exclusion:**
184 of 523 startups (35%) in cluster 4 contain 'bike', 'e-bike', 'scooter', or
'micromobility' in their description. These do not qualify for IRA clean vehicle
credits (30D/45W), which apply to highway-capable vehicles with battery capacity
≥7 kWh (passenger) or commercial EVs. Micromobility startups are reclassified as
non-IRA_tech. N moved: 184. Cluster 4 IRA_tech = highway EVs only.
Robustness R4-a: cluster 4 fully included; R4-b: cluster 4 fully excluded.

**Cluster 13 (CCUS) — retained:**
20 random members inspected. All 20 are genuine carbon capture, direct-air capture,
or carbon utilisation companies. Zero false positives detected. Cluster 13 is retained
in full. (< 25% non-CCUS threshold for exclusion was not met.)

---

## Design

**Natural experiment:** IRA signed 2022-08-16. Treated group = IRA_tech startups
(mapped from SPECTER2 cluster IDs below). Control = non-IRA_tech climate-tech startups.

**Pre/post cutoff:** 2022-08-16 (IRA signing date). Startups whose first funding
round occurred on or after this date are "post-IRA".

**Geographic dimension:** US startups (HQ Country = United States) vs. non-US.
US startups are the intended beneficiaries of IRA credits; non-US startups serve
as a geographic comparison arm.

**IRA_tech clusters (locked definitions):**
```
Credit category                    SPECTER2 cluster IDs       Notes
──────────────────────────────────────────────────────────────────────────────
45X  Battery manufacturing          55, 42
Solar  (45Y / 48E)                  40
Wind   (45Y / 48E)                  17
45V  Clean hydrogen                  2
45Q  Carbon capture (CCUS)          13
30D/45W  Clean vehicles & charging   4*, 3     *highway EVs only; 184 micro-
                                               mobility startups reclassified
48E  Standalone energy storage       56, 57, 54
Geothermal  (45Y / 48C)              5
Nuclear  (45Y advanced)              0
```

**Credit sub-families (secondary / exploratory H9):**
- *Manufacturing-intensive credits* (45X, 45V, 45Q): require specific production
  processes, minimum efficiency thresholds, or domestic sourcing. Expected to impose
  tighter technological requirements.
- *Generation credits* (45Y / 48E incl. solar cluster 40, wind cluster 17, storage
  clusters 56/57/54): broader eligibility; lower technological specificity per credit.

**Multiple-testing correction:** Holm sequential within each family (A–D).
**Standard errors:** Clustered by SPECTER2 tech cluster (specter_cluster; 63 levels).
**Software:** statsmodels ≥ 0.14, scikit-learn ≥ 1.3, Python ≥ 3.10.

---

## Hypotheses

### Family A — Entry and scale (DiD: unit = startup)

> Primary estimand: difference-in-differences.
> Post = first round ≥ 2022-08-16. Treated = IRA_tech cluster.

| H# | Hypothesis | Unit | DV | Key variable | Sign | Family |
|---|---|---|---|---|---|---|
| **H1** | IRA accelerates entry. US IRA_tech startups see a larger post-IRA increase in annual entry counts than the comparison cells. | Startup (agg.) | Annual count | US × IRA_tech × post-IRA | **+** | A |
| **H2** | IRA attracts larger rounds. US IRA_tech startups raised more capital post-IRA relative to comparisons. | Startup-round | log_capital | US × IRA_tech × post-IRA | **+** | A |
| **H3** | IRA broadens syndicates. US IRA_tech post-IRA syndicates have more investors per round. Niche definition (mutual 5-NN, ≥3 members, above-median niche_cohesion_t) will be finalised before H3 runs. | Startup-round | log(# investors) | US × IRA_tech × post-IRA | **+** | A |

Holm correction across H1–H3.

---

### Family B — Innovation novelty (DiD: unit = startup at first round)

> **Convergence hypothesis:** IRA credit certainty lowers the cost of imitating
> established approaches and attracts entrants who converge on proven technical
> solutions rather than exploring the frontier. This predicts that US IRA_tech
> startups funded after 2022-08-16 become LESS sparse (more similar to each other),
> relative to (a) pre-IRA US IRA_tech startups, (b) post-IRA US non-IRA_tech
> startups, and (c) post-IRA non-US IRA_tech startups.
>
> Primary outcome: **overall_sparsity_t** (time-aware; higher = more isolated/novel).
> Secondary: **niche_cohesion_t** (higher = niche specialist).

| H# | Hypothesis | Unit | DV | Key variable | Sign | Family |
|---|---|---|---|---|---|---|
| **H4** | IRA convergence. Post-IRA US IRA_tech startups are less novel (lower overall_sparsity_t) relative to all three comparison cells, after controlling for the secular novelty trend via year-quarter and cluster FEs. | Startup | **overall_sparsity_t** | US × IRA_tech × post-IRA | **−** | B |
| **H5** | IRA reduces niche specialisation. Post-IRA US IRA_tech entrants shift toward the mainstream (lower niche_cohesion_t). | Startup | niche_cohesion_t | US × IRA_tech × post-IRA | **−** | B |

Holm correction across H4–H5.

**Secondary / exploratory (H9-sub, not subject to Holm):**
The H4 convergence effect is expected to be *larger* for manufacturing-intensive
credits (45X, 45V, 45Q) — which reward specific, well-defined production methods —
than for generation credits (45Y/48E: solar, wind, storage). Tested by adding a
`mfg_credit` indicator and its triple interaction; exploratory, no sign correction.

---

### Family C — Network structure (DiD: unit = startup)

> Tests whether IRA reshaped the investor network around IRA_tech startups.

| H# | Hypothesis | Unit | DV | Key variable | Sign | Family |
|---|---|---|---|---|---|---|
| **H6** | IRA attracts heterogeneous investor types. US IRA_tech post-IRA startups show a larger increase in pct_ivc (independent VC) relative to comparisons. | Startup | pct_ivc | US × IRA_tech × post-IRA | **+** | C |
| **H7** | IRA reduces community siloing. Post-IRA US IRA_tech syndicates span more Louvain communities (excess_spread) than comparison cells. | Startup | excess_spread | US × IRA_tech × post-IRA | **+** | C |
| **H8** | Hardware concentration. The H7 network effect is larger for hardware-intensive IRA_tech clusters (45X, 48E, geothermal) than for software/service clusters. | Startup | excess_spread | hardware × post-IRA (within IRA_tech) | **+** | C |

Holm correction across H6–H8.

---

### Family D — Heterogeneity and robustness

| H# | Hypothesis | Unit | DV | Key variable | Sign | Family |
|---|---|---|---|---|---|---|
| **H9** | Manufacturing credits converge more. The H4 convergence effect (negative triple interaction) is larger in magnitude for manufacturing-intensive credits (45X, 45V, 45Q) than for generation credits (45Y/48E). | Startup | overall_sparsity_t | mfg_credit × post-IRA (within US IRA_tech) | **−** (more negative) | D |
| **H10** | Non-US placebo. Non-US IRA_tech startups show no convergence effect around 2022-08-16 (geographic placebo). | Startup | overall_sparsity_t | non-US × IRA_tech × post-IRA | **≈0** | D |

Holm correction across H9–H10.

---

## Estimating equation (H4 primary; analogous for H5–H10)

```
overall_sparsity_t = α
    + β₁ · US + β₂ · IRA_tech
    + β₃ · (US × IRA_tech)
    + β₄ · (US × post_IRA)
    + β₅ · (IRA_tech × post_IRA)
    + β₆ · (US × IRA_tech × post_IRA)      ← H4: predicted β₆ < 0
    + γ · log_capital
    + year-quarter FE + tech-cluster FE + country FE
    + ε

SEs clustered by tech cluster (specter_cluster; 63 levels).
```

**Event-study validity check (before reporting β₆):**
Replace post_IRA with year dummies 2017–2025, interact with US × IRA_tech,
reference year = 2021. Pre-2022 coefficients must be flat (no pre-trend).
If they trend, the parallel-trends assumption fails and β₆ is not interpretable
as a causal effect — this will be stated plainly.

**Pre-registered robustness variants:**
- R1: Treatment = founding year ≥ 2023 (pure post-IRA cohort)
- R2: Alternative IRA date 2022-07-27 (Senate passage / Manchin announcement)
- R3: Drop top 1% overall_sparsity_t
- R4-a: Cluster 4 fully included (184 micromobility startups as IRA_tech)
- R4-b: Cluster 4 fully excluded

---

## Variables

| Variable | Source | Notes |
|---|---|---|
| overall_sparsity_t | output/novelty_t/novelty_time_aware.csv | Time-aware; higher = more novel |
| niche_cohesion_t | output/novelty_t/novelty_time_aware.csv | Time-aware; higher = niche |
| IRA_tech | SPECTER2 cluster mapping above | Binary; cluster 4 highway EVs only |
| mfg_credit | Clusters 55, 42, 2, 13 (45X + 45V + 45Q) | Sub-indicator of IRA_tech |
| hardware | Hardware cluster list (Script 83 Step 3b) | Sub-indicator |
| US | HQ Country = "United States" | Binary |
| post_IRA | first_date ≥ 2022-08-16 | Binary |
| log_capital | log(1 + Total Raised USD M) | From cpu_company_dataset |
| specter_cluster | output/novelty/novelty_scores.csv | Tech-cluster FE |
| lagged_community | output/novelty_t/lagged_communities.csv | For H7 excess_spread |

---


---

## Niche definition (finalised 2026-09-21, before H3 runs)

A niche is a connected component of the mutual 3-nearest-neighbour graph on
SPECTER2 startup embeddings satisfying:
- Mean internal cosine similarity ≥ 0.8943 (90th percentile of
  100,000 randomly sampled pairwise similarities in the corpus)
- Size ≤ 30 (backstop; drops the main-stream mega-components)

This yielded **628 niches** with ≥2 dated members (total dated
member rows: 3538).

**Founding startup** = niche member with the earliest first-round date.
**Founding community** = plurality lagged Louvain community of the founding
startup's first-round investors (lagged to deals before year t₀).

### H3 specification (locked)

Unit: dyad (later entrant × candidate investor).
Dependent variable: tie (1 = realised, 0 = counterfactual).
Risk set: per startup — actual first-round investors + 5 counterfactual
  investors active in that year, drawn from the population of investors
  with any deal in that year; founding startup's first-round investors
  are excluded from both realised and counterfactual sets.
Model: conditional logit stratified by startup.

```
tie ~ in_founding_community + semantic_distance
    + geo_same_country + geo_same_region
    + same_community_t + log_prior_deals + has_prior_network
```

Pre-specified sign on in_founding_community: **POSITIVE**.
Placebo: replace founding community with random community of similar size,
  500 draws; report permutation p.


---

## What is NOT pre-registered here

- Survival / failure analyses (separate analysis family)
- Script 79r moderation hypotheses P1–P3 (separately pre-registered in Scripts 79–80)
- Any covariate or functional-form choices made after inspecting results
- Niche rule for H3 (finalised before H3 runs, not before H4)

---

## Script 86 result and H3 redesign rationale (2026-09-22 07:26 UTC)

### Script 86 result (original design)

| Quantity | Value |
|---|---|
| Niches with founding community | 133 / 628 (21%) |
| Valid dyads | 1,701 |
| in_founding_community = 1 | 23 dyads (0.7%) |
| β (in_founding_community) | −1.408 (SE=1.115, p=0.207) |
| Permutation p (500 draws) | 0.180 |
| MDE (80% power) | 3.12 log-odds |
| Verdict | NULL — WRONG SIGN |

**Why H3 failed:** The lagged community approach requires the founding
startup's investors to already have a co-investment history before year t₀.
For niches founded before 2016, almost no investor has prior deal history,
so founding community coverage was only 26.1% of founding investors. This
produced only 23 dyads with in_founding_community=1, making the test
uninformative (MDE >> 1.0).

### Redesign specification

**Changes:**
1. **LCC restriction**: lagged network built on the largest connected
   component only, not the full graph. Isolates the main exchange network.
2. **Tuned resolution**: Louvain resolution parameter tuned per year by
   binary search to yield 40–60 communities. This produces consistent
   community granularity across years rather than letting the resolution
   fall wherever modularity peaks.
3. **Founding-year cutoff**: restrict to niches whose founding startup's
   first round is ≥ 2018. By 2018 the LCC covers 51% of investors, so community coverage is substantially higher.

**Hypothesis prediction:** unchanged. **POSITIVE** sign on
in_founding_community.

**MDE gate:** feasibility check must confirm MDE < 1.0 log-odds before
Script 88 (H3 execution) is authorised.
Script 87 feasibility result:
  - N usable niches (FC assigned, ≥3 members): 64
  - N later entrants: 213
  - N valid dyads: 696
  - in_fc = 1 in valid dyads: 13 (1.9%)
  - SE: 0.7968
  - MDE: 2.2323
  - Decision: STOP


---

## Family E — Environmental Policy Uncertainty (EnvPU) moderation
**Added:** 2026-09-22  **Status: LOCKED**

> Tests whether climate policy uncertainty modulates the structure of
> co-investment matching. Unit: dyad (startup × investor candidate).
> Risk set: same as Script 79r (actual + 5 counterfactual investors).
> Model: conditional logit stratified by startup.
> EnvPU source: Noailly et al. EnvPU Index (US, monthly, 1990–2019).
> Coverage: startups with first rounds through March 2019.
> EnvPU matched by first-round year-quarter.
> SEs clustered by first-round quarter.

| H# | Hypothesis | Term | Sign | Holm family |
|---|---|---|---|---|
| **H5a** | Under higher policy uncertainty investors prefer semantically closer partners (less exploration). | EnvPU × semantic_distance | **−** | E |
| **H5b** | Under higher policy uncertainty investors prefer same-community co-investors (network insularity). | EnvPU × same_community | **+** | E |
| **H5c** | Under higher policy uncertainty investors prefer home-country co-investors (home bias). | EnvPU × geo_same_country | **+** | E |

Holm correction across H5a–H5c.

**Coverage note:** EnvPU index ends March 2019 (351 monthly observations,
1990–2019). Analysis restricted to startups with first rounds ≤ March 2019.

---

## Family F — Technology Readiness Level (TRL) moderation
**Added:** 2026-09-22  **Status: LOCKED — mapping confirmed unchanged 2026-09-22**

**Pilot estimate (Script 91):** β(sem_dist_z × trl_z) = +0.094 (SE=0.034, p=0.006). Directionally consistent with H6. Cluster-to-TRL mapping reviewed and confirmed unchanged after observing this pilot. Execution in Script 92 was fully planned before the pilot.

> Tests whether technology maturity modulates the expertise barrier in co-investment
> matching. Unit: dyad (startup × investor candidate).
> Risk set: same as Script 79r (actual + 5 counterfactual investors active that year).
> Model: conditional logit stratified by startup.
> TRL source: IEA ETP 2023 Clean Energy Technology Guide (>650 technologies, TRL 1–11).
> TRL assigned at the SPECTER2 cluster level (time-invariant).
> Clusters without a clear IEA match (food, software, services, monitoring) excluded.

### Rationale

For technologies at low readiness (TRL 1–6), the investor's own scientific and
engineering expertise is the primary screening criterion — a wrong match cannot be
corrected by market signals or established commercial precedent.
As technology matures (TRL 7–11), performance benchmarks, cost curves, and investment
templates substitute for deep expertise, relaxing the penalty for distant matches.
The effect of semantic distance on co-investment probability should therefore be
steepest at low TRL and attenuate as TRL rises.

### Hypothesis

| H# | Hypothesis | Term | Sign | Holm family |
|---|---|---|---|---|
| **H6** | The negative effect of semantic distance on co-investment probability is weaker for more mature technologies. | sem_dist_z × trl_z | **+** | F |

Holm correction: single hypothesis in family F (no correction required; reported for
consistency).

### Specification

```
tie ~ sem_dist_z + sem_dist_z × trl_z
    + same_community_tuned + geo_same_country + geo_same_region
    + log_prior_deals_z + log_degree_z

SEs clustered by SPECTER2 tech cluster (specter_cluster; 27 matched clusters).
```

Note: the trl_z main effect is absorbed by the startup fixed effect in the conditional
logit (TRL is constant within a startup; startup-level FE eliminates it). Only the
interaction sem_dist_z × trl_z is identified. This is expected and correct.

### TRL measure

- Source: IEA ETP 2023 Energy Technology Perspectives / Clean Energy Technology Guide
- URL: https://www.iea.org/data-and-statistics/data-tools/etp-clean-energy-technology-guide
- Scale: 1–11 (Concept 1–3 / Prototype 4–6 / Demonstration–Early adoption 7–9 / Mature 10–11)
- Assigned at cluster level; time-invariant
- Matched clusters: 27 / 63 (43%): 16 high confidence, 5 medium, 6 low
- Unmatched: 36 clusters (food tech, water, carbon accounting software, sustainability services,
  monitoring/satellite, mining) — excluded from H6

### Coverage and power

| Quantity | Value |
|---|---|
| Matched clusters | 27 / 63 (43%) |
| Startups with TRL | 1,800 / 4,669 (38.6%) |
| Dyads with TRL | 19,932 / 53,628 (37.2%) |
| TRL distinct values | 8 (TRL 4–11 scale) |
| TRL mean (dyad-weighted) | 8.63 |
| TRL SD (dyad-level) | 1.79 |
| MDE (80% power, α=0.05) | 0.095 log-odds per SD of sem_dist_z and trl_z |
| Pilot β sem_dist_z × trl_z | +0.094 (SE=0.034, p=0.006) — directionally correct |

### Robustness variants

- R-i: Exclude low-confidence (L) TRL assignments (6 clusters)
- R-ii: TRL as ordered categorical (4 IEA broad categories: Concept / Prototype / Demonstration / Mature)
- R-iii: Fully-interacted model with year × TRL to control for secular maturation trends

### Cluster–TRL mapping (locked after review)

See `output/h6_trl/cluster_trl_mapping.csv`.

| Cluster | IEA Technology | TRL | Conf |
|---|---|---|---|
| 0  | Advanced nuclear (SMR + fusion) | 4  | M |
| 2  | Green hydrogen via electrolysis | 8  | H |
| 3  | EV charging infrastructure | 9  | H |
| 4  | Battery electric vehicles | 10 | H |
| 5  | Enhanced geothermal systems | 7  | M |
| 13 | Carbon capture (CCUS / DAC) | 7  | H |
| 15 | Ocean wave and tidal energy | 6  | H |
| 16 | Electric aviation (eVTOL + fixed-wing) | 5 | H |
| 17 | Offshore wind | 10 | H |
| 21 | Zero-emission shipping (H₂ / ammonia) | 6 | M |
| 31 | Methane abatement and monitoring | 8 | L |
| 37 | Biomass to energy / Biochar | 8 | M |
| 40 | Solar photovoltaics | 11 | H |
| 42 | Lithium-ion battery recycling | 8 | H |
| 43 | Low-carbon cement and concrete | 7 | H |
| 45 | Bio-based materials (lignin, fibers) | 6 | L |
| 46 | Chemical plastics recycling | 7 | M |
| 47 | Bio-based / compostable packaging | 8 | L |
| 49 | Advanced waste sorting and recycling | 8 | L |
| 51 | Low-carbon construction materials | 7 | L |
| 54 | Thermal energy storage | 8 | H |
| 55 | Li-ion and next-generation batteries | 9 | H |
| 56 | Grid-scale long-duration storage | 8 | H |
| 57 | Battery storage systems (grid-connected) | 8 | H |
| 59 | Smart grid / distributed energy resources | 8 | L |
| 60 | Heat pumps (space heating) | 9 | H |
| 62 | Efficient cooling and HVAC | 9 | H |

## Family G — Investor type and startup novelty (S1)
## Family H — GVC/grant timing (S2a, S2b)

**Locked: 2026-09-22**

### Definitions (fixed on theoretical grounds, pre-data)
- `public_mission` = GVC + Impact (Government + Sovereign Wealth + NFP-VC + Impact Investing)
- `GVC_equity`: GVC investor in a non-Grant deal
- `GVC_grant`:  GVC investor in a Grant deal
- Reference category for S1: IVC (Venture Capital)

### S1 — investor type and startup novelty
- Unit: investor–startup tie (first-round syndicate, startups with ≥75% investors typed)
- Outcome: `overall_sparsity_t` (primary)
- Model: OLS with country FE, year FE, log_capital; SEs clustered by startup
- Pre-specified direction: β_PM > 0 (public_mission > IVC) AND β_CVC < 0 (IVC > CVC)
- N ties: 10,290  |  N startups: 6,066
- MDE for β_PM: 0.1154

### S2a — GVC-equity timing
- Unit: startup with ≥1 GVC-equity and ≥1 private investor (both VC-round dated)
- Outcome: Δ_round = first GVC-equity VC Round − first private VC Round
- Pre-specified direction: Δ_round < 0 (GVC enters earlier)
- N startups: 590  |  MDE: 0.175 round units
- Test: one-sample t-test, H1: μ < 0; robustness: OLS with country + year FE

### S2b — grant timing
- Unit: startup with ≥1 grant and ≥1 private investor (both dated)
- Outcome: Δ_days = first grant date − first private date
- Pre-specified direction: Δ_days < 0 (grant enters earlier)
- N startups: 1,055  |  MDE: 65.4 days
- Test: one-sample t-test, H1: μ < 0; robustness: OLS with country + year FE

### Multiple testing
- Holm correction across 4 primary tests: S1(β_PM), S1(β_CVC), S2a, S2b
- Familywise α = 0.05

**Results (2026-09-22)**:
- S1 β_PM  = +0.0058  (SE_cl=0.0412,  p_Holm=1.0000) — DIRECTIONAL (p≥0.05)
- S1 β_CVC = +0.0837  (SE_cl=0.0649,  p_Holm=1.0000) — FALSIFIED
- S2a mean Δ_round = +1.054 (p_Holm=1.0000) — FALSIFIED
- S2b mean Δ_days  = +341.7 (p_Holm=0.0186) — FALSIFIED

**Status: LOCKED**

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
