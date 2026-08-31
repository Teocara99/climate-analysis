"""
Script 38: CEO Classification via Claude Haiku API
===================================================

Replaces Mistral/regex with Claude Haiku using a fixed prompt addressing:
  FIX 1 — serial_founder: only True if a PRIOR (different) company was founded
  FIX 2 — technical: judge CEO's personal background, not company sector/products
  FIX 3 — JSON/reasoning consistency enforced via explicit instruction

Three few-shot examples grounded in the actual error patterns observed.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python 38_ceo_haiku_classify.py --phase 1       # 100 bios, standard API
  python 38_ceo_haiku_classify.py --phase 2       # batch API, all remaining
  python 38_ceo_haiku_classify.py --phase results # poll batch + merge results
"""

import argparse, json, os, re, sys, time
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

sys.path.insert(0, ".")
from load_data import load_deals

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 256
TEMP       = 0.3       # deterministic output
MAX_WORKERS = 8        # Haiku rate limit is generous; 8 threads is safe
MAX_BIO    = 900
OUT_PATH   = "output/ceo_llm/ceo_haiku_classifications.csv"
BATCH_PATH = "output/ceo_llm/batch_id.txt"

# ── Prompt ──────────────────────────────────────────────────────────────────
SYSTEM = """You classify startup CEO backgrounds from short biographies.
You are precise, literal, and never infer beyond what the text states.
Your JSON labels must always match your reasoning — never contradict them."""

FEW_SHOT_EXAMPLES = [
    {
        "bio": "Dr. Sarah Chen is the Founder of GreenGrid AI. She holds a PhD in Electrical Engineering from MIT and spent 8 years as a power systems researcher at NREL before starting GreenGrid AI in 2019.",
        "expected": {"technical": True, "serial_founder": False, "green_mission": False, "elite_education": True},
        "reasoning": "PhD in Electrical Engineering (STEM degree) + 8 years as researcher = technical=true. She founded ONLY GreenGrid AI — no prior company mentioned = serial_founder=false. Bio describes her career, no personal climate mission language = green_mission=false. MIT = elite_education=true."
    },
    {
        "bio": "Mr. James Okafor is a Co-Founder and serves as Chief Executive Officer at SolarLeap. He previously served as VP of Sales at Salesforce and Managing Director at Goldman Sachs. He holds an MBA from Wharton.",
        "expected": {"technical": False, "serial_founder": False, "green_mission": False, "elite_education": True},
        "reasoning": "VP Sales + Managing Director at Goldman Sachs = business background, not technical = technical=false. He co-founded only SolarLeap; prior roles were as employee not founder = serial_founder=false. No personal mission language = green_mission=false. Wharton MBA = elite_education=true."
    },
    {
        "bio": "Ms. Priya Mehta is a Co-Founder and CEO at CarbonVault. She previously co-founded and served as CEO at WaterTech Solutions, which was acquired by Veolia in 2018. She is passionate about using business to solve the climate crisis.",
        "expected": {"technical": False, "serial_founder": True, "green_mission": True, "elite_education": False},
        "reasoning": "No STEM degree or engineering career mentioned = technical=false. She co-founded WaterTech Solutions BEFORE CarbonVault, and it was acquired = serial_founder=true. 'Passionate about using business to solve the climate crisis' = personal mission language = green_mission=true. No elite university mentioned = elite_education=false."
    }
]

def build_prompt(bio_text: str) -> str:
    bio_trunc = bio_text[:MAX_BIO] + ("…" if len(bio_text) > MAX_BIO else "")

    examples_str = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        expected_json = json.dumps(ex["expected"])
        examples_str += f"""
EXAMPLE {i}:
Bio: {ex['bio']}
Output: {{"technical": {str(ex['expected']['technical']).lower()}, "serial_founder": {str(ex['expected']['serial_founder']).lower()}, "green_mission": {str(ex['expected']['green_mission']).lower()}, "elite_education": {str(ex['expected']['elite_education']).lower()}, "confidence": "high", "reasoning": "{ex['reasoning']}"}}
"""

    return f"""Classify this CEO biography on four dimensions.

CLASSIFICATION RULES:

1. TECHNICAL (CEO's personal STEM background):
   TRUE only if the CEO personally: holds a science/engineering degree, worked as engineer/scientist/researcher, has a STEM PhD, or personally invented/developed technology.
   FALSE if: CEO runs a technology company but has a business/law/finance background. The COMPANY's products or industry do NOT make the CEO technical.
   RULE: Judge the bio's description of the CEO's own degrees and career roles — not the company's sector.

2. SERIAL_FOUNDER (founded a PRIOR, different company):
   TRUE only if: bio explicitly mentions the CEO founded or co-founded a DIFFERENT company BEFORE the current one. Phrases like "previously co-founded X", "former CEO and co-founder of Y [other company]", or "sold his previous startup" qualify.
   FALSE if: bio only shows the CEO as co-founder of the CURRENT company, even if labeled "Co-Founder and CEO." Working as an employee (VP, Director, Manager) at a prior company does NOT count.
   RULE: The phrase "Co-Founder and CEO at [current company]" alone → serial_founder = false.

3. GREEN_MISSION (personal climate commitment):
   TRUE only if: bio contains explicit personal mission language about climate, sustainability, or environment — e.g. "passionate about solving climate change", "committed to a sustainable future", "driven by environmental impact."
   FALSE if: bio describes the company's green products/sector without attributing personal motivation to the CEO.

4. ELITE_EDUCATION:
   TRUE if: bio or education section mentions MIT, Stanford, Harvard, Oxford, Cambridge, Caltech, UC Berkeley, ETH Zurich, Imperial College, Princeton, Yale, Wharton, INSEAD, LBS, Carnegie Mellon, or equivalent top-10 global university.
   FALSE if: any other university, or education not mentioned.

CRITICAL: Your JSON values must match your reasoning. If you write "no technical background" in reasoning, technical must be false.
{examples_str}
NOW CLASSIFY:
Bio: {bio_trunc}

Respond ONLY with valid JSON, no other text:
{{"technical": true/false, "serial_founder": true/false, "green_mission": true/false, "elite_education": true/false, "confidence": "high/medium/low", "reasoning": "one concise sentence"}}"""


# ── Single classification ────────────────────────────────────────────────────
def classify_one(client: anthropic.Anthropic, row: dict) -> dict:
    co  = row["co"]
    bio = str(row["CEO Biography"])

    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMP,
                system=SYSTEM,
                messages=[{"role": "user", "content": build_prompt(bio)}]
            )
            raw = msg.content[0].text.strip()
            m   = re.search(r'\{.*?\}', raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON: {raw[:120]}")
            result = json.loads(m.group(0))
            return {
                "co":            co,
                "technical":     bool(result.get("technical", False)),
                "serial_founder":bool(result.get("serial_founder", False)),
                "green_mission": bool(result.get("green_mission", False)),
                "elite_education":bool(result.get("elite_education", False)),
                "confidence":    result.get("confidence", "?"),
                "reasoning":     result.get("reasoning", ""),
                "bio_used":      bio[:MAX_BIO],
                "error":         "",
                "input_tokens":  msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
            }
        except anthropic.RateLimitError:
            time.sleep(30)
        except Exception as e:
            if attempt == 2:
                return {"co": co, "technical": None, "serial_founder": None,
                        "green_mission": None, "elite_education": None,
                        "confidence": "error", "reasoning": "", "bio_used": bio[:MAX_BIO],
                        "error": str(e), "input_tokens": 0, "output_tokens": 0}
            time.sleep(2 ** attempt)


# ── Batch of rows via thread pool ────────────────────────────────────────────
def run_phase1(client, rows, checkpoint_every=25):
    results, done = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(classify_one, client, r): r["co"] for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (len(rows) - done) / rate if rate > 0 else 0
            print(f"\r  {done}/{len(rows)}  [{elapsed/60:.1f}min  ETA {eta/60:.1f}min]  ", end="", flush=True)
            if done % checkpoint_every == 0:
                _save(pd.DataFrame(results), OUT_PATH, append=True)
    print()
    return results


def _save(df_new, path, append=False):
    if append and os.path.exists(path):
        existing = pd.read_csv(path)
        df_new   = pd.concat([existing, df_new]).drop_duplicates("co", keep="last")
    df_new.to_csv(path, index=False)


# ── Batch API (Phase 2) ──────────────────────────────────────────────────────
def submit_batch(client, rows):
    requests_list = []
    for row in rows:
        requests_list.append({
            "custom_id": row["co"][:64],   # batch ID, max 64 chars
            "params": {
                "model":      MODEL,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMP,
                "system":     SYSTEM,
                "messages":   [{"role": "user", "content": build_prompt(str(row["CEO Biography"]))}]
            }
        })
    # Submit in chunks of 10,000 (API limit)
    batch_ids = []
    for chunk_start in range(0, len(requests_list), 10_000):
        chunk = requests_list[chunk_start:chunk_start+10_000]
        batch = client.beta.messages.batches.create(requests=chunk)
        batch_ids.append(batch.id)
        print(f"  Submitted batch {len(batch_ids)}: {batch.id}  ({len(chunk):,} requests)")
        with open(BATCH_PATH, "a") as f:
            f.write(batch.id + "\n")
    return batch_ids


def poll_batches(client):
    if not os.path.exists(BATCH_PATH):
        print("No batch IDs found. Run --phase 2 first."); return

    batch_ids = open(BATCH_PATH).read().strip().splitlines()
    all_results = []
    for bid in batch_ids:
        batch = client.beta.messages.batches.retrieve(bid)
        print(f"  Batch {bid}: status={batch.processing_status}  "
              f"succeeded={batch.request_counts.succeeded}  "
              f"errored={batch.request_counts.errored}  "
              f"processing={batch.request_counts.processing}")
        if batch.processing_status != "ended":
            print("  → Not finished yet. Re-run --phase results later.")
            continue
        for result in client.beta.messages.batches.results(bid):
            co  = result.custom_id
            if result.result.type == "succeeded":
                raw = result.result.message.content[0].text.strip()
                try:
                    m   = re.search(r'\{.*?\}', raw, re.DOTALL)
                    parsed = json.loads(m.group(0)) if m else {}
                    all_results.append({
                        "co":            co,
                        "technical":     bool(parsed.get("technical", False)),
                        "serial_founder":bool(parsed.get("serial_founder", False)),
                        "green_mission": bool(parsed.get("green_mission", False)),
                        "elite_education":bool(parsed.get("elite_education", False)),
                        "confidence":    parsed.get("confidence", "?"),
                        "reasoning":     parsed.get("reasoning", ""),
                        "error":         "",
                    })
                except Exception as e:
                    all_results.append({"co": co, "error": str(e)})
            else:
                all_results.append({"co": co, "error": str(result.result.error)})

    if all_results:
        _save(pd.DataFrame(all_results), OUT_PATH, append=True)
        print(f"\n  Saved {len(all_results):,} results to {OUT_PATH}")
    return all_results


# ── Compare to Mistral/regex ─────────────────────────────────────────────────
def compare_to_mistral(df_haiku):
    mistral_path = "output/ceo_llm/ceo_classifications.csv"
    if not os.path.exists(mistral_path):
        return
    df_m = pd.read_csv(mistral_path)
    merged = df_haiku.merge(df_m[["co","technical","serial_founder","green_mission","elite_education"]],
                             on="co", suffixes=("_haiku","_mistral"), how="inner")
    print(f"\n  Agreement (Haiku vs Mistral) on N={len(merged):,} shared companies:")
    for col in ["technical","serial_founder","green_mission","elite_education"]:
        agree = (merged[f"{col}_haiku"] == merged[f"{col}_mistral"]).mean() * 100
        h_rate = merged[f"{col}_haiku"].mean() * 100
        m_rate = merged[f"{col}_mistral"].mean() * 100
        print(f"    {col:<22} agree={agree:.0f}%   Haiku={h_rate:.1f}%   Mistral={m_rate:.1f}%")


def print_summary(df, label=""):
    df_ok = df.dropna(subset=["technical"])
    print(f"\n  {label} Classification rates (N={len(df_ok):,}):")
    for col in ["technical","serial_founder","green_mission","elite_education"]:
        rate = df_ok[col].astype(float).mean() * 100
        regex_rate = {"technical":25.0,"serial_founder":68.5,"green_mission":19.7,"elite_education":20.5}
        print(f"    {col:<22} {rate:.1f}%   (regex: {regex_rate[col]:.1f}%   Δ={rate-regex_rate[col]:>+.1f}pp)")
    tok_in  = df_ok.get("input_tokens",  pd.Series([0])).sum()
    tok_out = df_ok.get("output_tokens", pd.Series([0])).sum()
    cost    = tok_in/1e6*0.25 + tok_out/1e6*1.25
    print(f"    Tokens used:  {tok_in:,} in + {tok_out:,} out  ≈ ${cost:.3f}")
    err = (df_ok["error"] != "").sum() if "error" in df_ok else 0
    print(f"    Errors: {err}")


def show_20_examples(df):
    print("\n" + "="*80)
    print("20 EXAMPLES FOR MANUAL VALIDATION")
    print("="*80)
    # Pick 5 from each quadrant of (technical, serial_founder) and some random
    sample = (pd.concat([
        df[(df["technical"]==True)  & (df["serial_founder"]==True)].head(5),
        df[(df["technical"]==True)  & (df["serial_founder"]==False)].head(5),
        df[(df["technical"]==False) & (df["serial_founder"]==True)].head(5),
        df[(df["technical"]==False) & (df["serial_founder"]==False)].head(5),
    ]).drop_duplicates("co").head(20))

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        bio = str(row.get("bio_used",""))[:300].replace('\n',' ')
        print(f"\n{'─'*78}")
        print(f"{i:02d}. {row['co'][:60]}")
        print(f"    tech={row['technical']}  serial={row['serial_founder']}  "
              f"green={row['green_mission']}  elite={row['elite_education']}  conf={row['confidence']}")
        print(f"    Reason: {str(row['reasoning'])[:160]}")
        print(f"    Bio:    {bio}…")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="1", choices=["1","2","results","rerun"])
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set your API key first:\n  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    # Load bios
    print("Loading CEO bios...")
    d = load_deals()
    d["co"] = d["Companies"].str.strip()
    cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
    study_cos = set(cdf["Companies"].str.strip())
    d_ceo = (d[d["co"].isin(study_cos)]
             .sort_values("Year", ascending=False)
             .drop_duplicates("co")[["co","CEO Biography"]]
             .dropna(subset=["CEO Biography"]))
    print(f"  {len(d_ceo):,} companies with CEO bios")

    if args.phase == "1":
        batch = d_ceo.head(100).to_dict("records")
        print(f"\nPhase 1: classifying {len(batch)} bios with Haiku ({MAX_WORKERS} threads)…")
        results = run_phase1(client, batch)
        df = pd.DataFrame(results)
        _save(df, OUT_PATH)
        print_summary(df, "Haiku Phase 1")
        compare_to_mistral(df)
        show_20_examples(df.dropna(subset=["technical"]))

    elif args.phase == "2":
        if os.path.exists(OUT_PATH):
            existing = pd.read_csv(OUT_PATH)
            # Only skip rows that were successfully classified (no error)
            done = set(existing[existing["error"].isna() | (existing["error"] == "")]["co"])
        else:
            done = set()
        remaining = d_ceo[~d_ceo["co"].isin(done)].to_dict("records")
        print(f"\nPhase 2: classifying {len(remaining):,} remaining bios via standard API ({MAX_WORKERS} threads)…")
        results = run_phase1(client, remaining)
        df_new = pd.DataFrame(results)
        _save(df_new, OUT_PATH, append=True)
        df_all = pd.read_csv(OUT_PATH)
        print_summary(df_all, "Haiku Full")

    elif args.phase == "results":
        print("\nPolling batch results…")
        poll_batches(client)
        df = pd.read_csv(OUT_PATH)
        print_summary(df, "Haiku Full")

    elif args.phase == "rerun":
        # Re-run all founder analyses with Haiku classifications
        if not os.path.exists(OUT_PATH):
            print("No Haiku results yet. Run --phase 1 or --phase 2 first."); return
        import warnings; warnings.filterwarnings("ignore")
        import statsmodels.formula.api as smf
        from scipy import stats

        def sig(p):
            return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

        haiku = pd.read_csv(OUT_PATH)
        cdf   = pd.read_csv("output/cpu/cpu_company_dataset.csv")
        cdf["co"]         = cdf["Companies"].str.strip()
        cdf["is_failed"]  = (cdf["outcome"]=="Failed").astype(float)
        cdf["log_cap"]    = cdf["log_capital"]
        cdf["nrounds"]    = np.log1p(cdf["n_rounds"].fillna(1))
        cdf["country_fe"] = cdf["hq_country"].where(
            cdf["hq_country"].isin(cdf["hq_country"].value_counts()[lambda x: x>=3].index), "Other")
        CTRL = "log_cap + nrounds + specter_novelty + C(country_fe) + C(deal_year)"

        # Only keep successfully classified rows
        haiku_ok = haiku[haiku["error"].isna() | (haiku["error"] == "")].copy()
        for v in ["technical","serial_founder","green_mission","elite_education"]:
            haiku_ok[v] = haiku_ok[v].astype(float)

        df = cdf.merge(haiku_ok[["co","technical","serial_founder","green_mission","elite_education"]],
                       on="co", how="left")
        df_ok = df.dropna(subset=["technical","log_capital","deal_year"]).copy()
        print(f"  Merged: {len(df_ok):,} companies with both CEO and analysis data")

        print("\n" + "="*72)
        print("RE-RUN: FOUNDER CONTROLS WITH HAIKU CLASSIFICATIONS")
        print("="*72)
        bio_ctrl = "technical + serial_founder + green_mission + elite_education"
        m_base = smf.ols(f"is_failed ~ cross_community + {CTRL}", data=df_ok).fit(cov_type="HC1")
        m_full = smf.ols(f"is_failed ~ cross_community + {bio_ctrl} + {CTRL}", data=df_ok).fit(cov_type="HC1")
        print(f"\n  Baseline: β_cross={m_base.params['cross_community']:>+.5f} {sig(m_base.pvalues['cross_community'])}  N={int(m_base.nobs):,}")
        print(f"  + Haiku:  β_cross={m_full.params['cross_community']:>+.5f} {sig(m_full.pvalues['cross_community'])}  N={int(m_full.nobs):,}")
        print(f"\n  Founder coefficients:")
        for v in ["technical","serial_founder","green_mission","elite_education"]:
            b=m_full.params.get(v,np.nan); se=m_full.bse.get(v,np.nan); p=m_full.pvalues.get(v,np.nan)
            print(f"    {v:<22} β={b:>+.5f} ({se:.4f}) {sig(p)}")

        print("\n" + "="*72)
        print("INTERACTIONS: founder × cross_community")
        print("="*72)
        for ivar in ["technical","serial_founder","green_mission"]:
            sub = df_ok.dropna(subset=[ivar]).copy()
            sub["ix"] = sub[ivar].astype(float) * sub["cross_community"]
            m_int = smf.ols(f"is_failed ~ cross_community + {ivar} + ix + {bio_ctrl} + {CTRL}",
                            data=sub).fit(cov_type="HC1")
            b_cc = m_int.params["cross_community"]
            b_ix = m_int.params.get("ix",np.nan)
            p_ix = m_int.pvalues.get("ix",1.0)
            eff_0 = b_cc; eff_1 = b_cc + b_ix
            print(f"\n  {ivar} × cross_community:  β_int={b_ix:>+.4f} {sig(p_ix)}")
            print(f"    {ivar}=0: β_cross={eff_0:>+.4f}   {ivar}=1: β_cross={eff_1:>+.4f}")

        print("\n" + "="*72)
        print("DESCRIPTIVE STATS (Haiku vs Regex)")
        print("="*72)
        regex_rates = {"technical":25.0,"serial_founder":68.5,"green_mission":19.7,"elite_education":20.5}
        print(f"  {'Variable':<22} {'Haiku %':>10}  {'Regex %':>10}  {'Δ':>8}")
        print(f"  {'-'*55}")
        for v in ["technical","serial_founder","green_mission","elite_education"]:
            h = df_ok[v].astype(float).mean()*100
            r = regex_rates[v]
            print(f"  {v:<22} {h:>10.1f}%  {r:>10.1f}%  {h-r:>+7.1f}pp")

if __name__ == "__main__":
    main()
