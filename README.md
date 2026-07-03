# Climate Tech VC Investment Analysis

End-to-end research pipeline studying **9,755 climate tech companies** and **24,388 investors** from PitchBook (2012–2026). The project connects investor composition (green vs. traditional), company novelty, policy shocks, and startup outcomes.

---

## Research Questions

1. Does **investor green mix** predict company survival, exit success, or failure?
2. Did **policy shocks** (Paris Agreement, EU Green Deal, Trump elections, IRA) change those effects?
3. Did **mixed syndicates** (green + traditional investors) outperform pure-green or pure-traditional during the Trump 1 M&A surge?
4. Does **technological novelty** (BERT-based semantic distance from peers) predict funding and exits — and does investor mix moderate that?

---

## Pipeline Overview

```
PitchBook Excel (47,528 deals, 9,755 companies)
        │
        ├── 01–03   Investment trends over time & by stage
        ├── 04      Unique investor extraction (24,388 investors)
        ├── 05a–e   Investor enrichment + Ollama/Mistral classification
        │           → 8 types (GVC, CVC, IVC, Impact_VC, Bank_VC, ...)
        │           → 3 green-focus labels (GREEN_VC, ESG_ALIGNED, TRADITIONAL)
        ├── 06      Investor type distribution & capital deployed analysis
        ├── 07      Company success × investor mix (survival models 1–3)
        ├── 08      Kaplan–Meier survival analysis (Model 4)
        ├── 09a–g   External data + regression models A/B/C
        │           → FRED macro controls (oil, rates, VIX, GDP)
        │           → Policy shock dummies (geographically scoped)
        │           → Political cycle analysis: Trump 1 → Biden → Trump 2
        │           → EU Green Deal × syndicate interaction
        │           → Fire-sale test: M&A vs IPO split
        ├── 10      Trump 1 M&A deep-dive (sector, acquirer, geography, stage)
        └── 11a–b   Novelty index (BERT embeddings → UMAP → HDBSCAN)
```

---

## Key Findings

### 1. Investor composition predicts exits (highly robust)
Companies backed predominantly by GREEN_VC investors have **significantly lower IPO/M&A exit rates** (logit coef −0.292, p<0.001), controlling for year, region, company age, oil price, interest rate, and VIX. The effect barely moves when macro controls are added (−0.302 → −0.292), confirming it is not a macro story.

### 2. Mixed syndicates are the sweet spot
Companies with a blend of green and traditional investors show:
- Lowest failure rate (8.3% vs 9.5% pure green, 12.3% pure non-green)
- Highest survival rate (91.7%)
- Highest exit rate (6.8%)

### 3. Trump 1 caused a fire-sale M&A surge — captured by Mixed syndicates
During Trump 1 (2016–2021), M&A exit rates jumped from 5.2% → 7.4%. The surge was **exclusively M&A-driven** (not IPO), consistent with strategic acquisitions under policy uncertainty. Mixed syndicates achieved the highest exit rate (11.7% Trump 1 era) vs pure green (8.3%) and pure non-green (10.7%), directionally supporting the fire-sale hypothesis (p~0.20, limited by n=471 Mixed deals in Trump 1).

### 4. Trump 2016 and Trump 2024 have opposite signs
- Trump 2016: +0.429 (p<0.001) — massive exit surge
- Trump 2024: −0.147 (p=0.29, limited signal with 1,537 post-election deals) — exit compression

### 5. EU Green Deal reduced exits, IRA had limited effect so far
Post-EU Green Deal, exit rates fell across all syndicates (Mixed fell most, −3.6pp), suggesting policy certainty encouraged companies to hold and grow rather than sell. The IRA effect is not yet significant (only 13% of deals post-IRA).

### 6. Energy Majors 5× more acquisitive during Trump 1
Shell, BP, Total and peers went from 0.8% → 4.1% of acquirer types — buying cheap climate assets while US policy risk depressed valuations. Early-stage companies (A/B rounds) saw the largest M&A jump (+6.1pp), confirming fire-sale dynamics.

---

## Novelty Index (Scripts 11a–11b)

We compute a **peer novelty score** for each company as the semantic distance from its closest competitor in the embedding space:

```
novelty(i) = 1 − max cosine_similarity(embed(i), embed(j≠i))
```

### Why these choices

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding model (primary) | `BAAI/bge-large-en-v1.5` | #1 on MTEB leaderboard for semantic similarity; validated in prior biodiversity startup analysis |
| Embedding model (comparison) | `allenai/specter2_base` | Trained on scientific citation graphs; better at distinguishing technological mechanisms |
| Preprocessing | ABTT (All-But-The-Top, d=3) | Removes domain-wide discourse bias from embeddings (Mu & Viswanath 2018) |
| Dimensionality reduction | UMAP | Preserves local neighbourhood structure; PCA destroys it (confirmed in biodiversity analysis) |
| Clustering | HDBSCAN | No k to specify; labels genuine outliers as noise; finds natural density-based clusters |
| Cluster interpretation | c-TF-IDF (BERTopic-style) | Extracts class-specific keywords without requiring topic modelling infrastructure |
| Visualisation | UMAP 2D | Same as above, separate pass with min_dist=0.1 for visual spread |

### Why not K-means + PCA (the original `novelty_index.py` approach)
- **K-means** forces every company into a cluster and requires guessing k — dishonest for a corpus where some niches are tight clusters and others sparse outliers
- **PCA** is a global linear projection that compresses most local structure into 2 dimensions — UMAP preserves the neighbourhood geometry that actually matters for interpretation

---

## Setup

```bash
# Install dependencies
pip install fredapi gdeltdoc lifelines statsmodels scikit-learn \
            sentence-transformers umap-learn hdbscan

# Run in order:
python3 01_investment_over_time.py
python3 02_investor_analysis.py
python3 03_capital_by_stage_over_time.py
python3 04_unique_investors.py
python3 05a_enrich_investor_descriptions.py
python3 05b_classify_investors_ollama.py validate   # Phase 1: validate on 100
python3 05b_classify_investors_ollama.py full       # Phase 2: classify all (~40h, local Ollama)
python3 05c_classification_to_csv.py
python3 05d_qa_classification.py                    # QA pass
python3 05e_apply_corrections.py                    # Apply deterministic fixes
python3 06_investor_classification_analysis.py
python3 07_company_success_vs_investor_mix.py
python3 08_survival_models.py
python3 09a_download_external_data.py <FRED_API_KEY>
python3 09b_merge_deal_data.py
python3 09c_regression_models.py
python3 09d_regression_plots.py
python3 09e_political_cycle_analysis.py
python3 09f_us_interaction_model.py
python3 09g_eu_and_exit_type.py
python3 10_trump_ma_analysis.py
python3 11a_novelty_compute.py                      # ~10–20 min (downloads models first run)
python3 11b_novelty_analysis.py
```

### Ollama setup (for investor classification)
```bash
ollama pull mistral          # ~4.4 GB
python3 05b_classify_investors_ollama.py validate
```

### FRED API key
Free at https://fred.stlouisfed.org/docs/api/api_key.html — used to download WTI oil price, federal funds rate, VIX, GDP, jobless claims (2010–2026).

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| PitchBook | Deal-level VC data (47,528 deals) + company profiles (9,755 companies) | Licensed |
| FRED (St. Louis Fed) | Macro controls: oil, rates, VIX, GDP | Free API |
| Policy shocks | Hand-coded timeline of 13 climate policy events | `data/policy_shocks.csv` |

> **Note:** PitchBook Excel files are excluded from this repo (`.gitignore`). The `data/` folder contains only derived CSVs safe for sharing.

---

## Investor Classification Taxonomy

| Type | Definition | Examples |
|------|-----------|---------|
| GVC | Government-backed VC or public agency | US DoE, European Commission, Innovate UK |
| CVC | Corporate venture arm | Google Ventures, Shell Ventures, Aramco |
| IVC | Independent private VC | Sequoia, Lowercarbon Capital, Breakthrough Energy |
| Impact_VC | Mission-driven private fund | Omidyar Network, Third Derivative |
| Bank_VC | Bank or financial institution VC | Goldman Sachs, Wells Fargo Innovation |
| Angel_Network | Angel syndicate or group | Keiretsu Forum, Angels for Impact |
| University_VC | University-affiliated fund | IP Group (Oxford), TechFounders (TU Munich) |
| OTHER | Accelerators, incubators, non-fitting | Y Combinator, Techstars, MassChallenge |

Green focus labels: `GREEN_VC` (primary climate/clean focus) · `ESG_ALIGNED` (mentions ESG, not primary) · `TRADITIONAL` (no green language)

Classification uses local Ollama/Mistral with a QA pass that auto-corrects ~2.6% of labels using self-consistency checks (reasoning text vs. JSON field) and rule-based name-pattern overrides.

---

## Output Files

| File | Description |
|------|-------------|
| `output/investors_classified_output.csv` | 24,388 investors with type + green focus labels |
| `output/company_investor_mix.csv` | Per-company green share, syndicate type, outcome |
| `data/deals_with_external_data.csv` | 32,503 deals with FRED controls + policy dummies |
| `output/novelty/novelty_scores.csv` | Per-company novelty scores (BGE + SPECTER2) |
| `output/regression_tables.csv` | Full coefficient tables for Models A/B/C |
| `output/trump_ma_tables.xlsx` | M&A surge deep-dive tables |

---

## Citation

If you use this pipeline, please cite:

```
Carabelli, M. (2026). Climate Tech VC Investment Analysis Pipeline.
GitHub: https://github.com/Teocara99/climate-analysis
```
