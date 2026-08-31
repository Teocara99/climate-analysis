"""
Script 37: CEO Background Classification via Ollama (Mistral)
=============================================================

Replaces regex-based classification with LLM inference.
Classifies 4 dimensions from CEO Biography text:
  technical       - personal STEM background (degree/career, not company)
  serial_founder  - founded a PRIOR (different) company
  green_mission   - personal climate/sustainability motivation language
  elite_education - top-tier university attendance

Usage:
  python 37_ceo_llm_classify.py --phase 1   # 100 bios (test)
  python 37_ceo_llm_classify.py --phase 2   # all remaining bios
  python 37_ceo_llm_classify.py --phase all # full run from scratch
"""

import warnings; warnings.filterwarnings("ignore")
import argparse, json, os, re, sys, time
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

sys.path.insert(0, ".")
from load_data import load_deals

OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "mistral"
TEMPERATURE   = 0.3
MAX_WORKERS   = 4
MAX_BIO_CHARS = 900  # truncate very long bios; plenty for classification
OUT_PATH      = "output/ceo_llm/ceo_classifications.csv"

PROMPT_TEMPLATE = """You are classifying a startup CEO based on their biography.

Read the biography carefully and classify on these dimensions:

1. TECHNICAL BACKGROUND (is the CEO personally technical?):
   YES: CEO has a degree in science/engineering, worked as engineer/scientist/researcher, has a PhD/Dr. in a STEM field, personally invented or developed technology
   NO: CEO has a business/finance/law background, or bio is too vague to determine
   IMPORTANT: Look at the CEO's PERSONAL background, not the company's products or industry. A CEO running a tech company is not necessarily technical.

2. SERIAL FOUNDER (founded a PREVIOUS company?):
   YES: CEO founded or co-founded a DIFFERENT company BEFORE the current one. Look for: "previously founded", "prior venture", "former CEO/founder of [other company]", "sold his previous company"
   NO: CEO only co-founded the CURRENT company, or no mention of prior ventures
   IMPORTANT: Co-founding the CURRENT company does NOT count as serial founder.

3. GREEN MISSION (personally motivated by climate/sustainability?):
   YES: Bio mentions personal commitment to climate, sustainability, or environment — mission-driven language about why they do this work
   NO: Bio describes the business sector without personal mission language

4. ELITE EDUCATION:
   YES: Attended MIT, Stanford, Harvard, Oxford, Cambridge, Caltech, Berkeley, ETH Zurich, Imperial College, Princeton, Yale, Wharton, INSEAD, LBS, Carnegie Mellon, or similarly ranked top institution
   NO: Other university or not mentioned

Respond ONLY with valid JSON, nothing else:
{{"technical": true/false, "serial_founder": true/false, "green_mission": true/false, "elite_education": true/false, "confidence": "high/medium/low", "reasoning": "one sentence"}}

CEO Biography: {bio}"""


def classify_one(row):
    co, bio = row["co"], str(row["CEO Biography"])
    bio_trunc = bio[:MAX_BIO_CHARS] + ("…" if len(bio) > MAX_BIO_CHARS else "")
    prompt = PROMPT_TEMPLATE.format(bio=bio_trunc)

    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL,
                json={"model": MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": TEMPERATURE}},
                timeout=90)
            raw = r.json()["response"].strip()

            # Extract JSON — handle occasional preamble text
            m = re.search(r'\{.*?\}', raw, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON found: {raw[:100]}")
            result = json.loads(m.group(0))

            return {
                "co": co,
                "technical":       bool(result.get("technical", False)),
                "serial_founder":  bool(result.get("serial_founder", False)),
                "green_mission":   bool(result.get("green_mission", False)),
                "elite_education": bool(result.get("elite_education", False)),
                "confidence":      result.get("confidence", "?"),
                "reasoning":       result.get("reasoning", ""),
                "bio_used":        bio_trunc,
                "error":           "",
            }
        except Exception as e:
            if attempt == 2:
                return {"co": co, "technical": None, "serial_founder": None,
                        "green_mission": None, "elite_education": None,
                        "confidence": "error", "reasoning": "", "bio_used": bio_trunc,
                        "error": str(e)}
            time.sleep(2)


def run_batch(rows, desc="", checkpoint_every=25):
    results, done = [], 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(classify_one, r): r["co"] for r in rows}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(rows) - done) / rate if rate > 0 else 0
            print(f"\r  {desc}  {done}/{len(rows)}  "
                  f"[{elapsed/60:.1f}min elapsed  ETA {eta/60:.1f}min]   ", end="", flush=True)

            # Checkpoint
            if done % checkpoint_every == 0:
                df_ckpt = pd.DataFrame(results)
                if os.path.exists(OUT_PATH):
                    existing = pd.read_csv(OUT_PATH)
                    df_ckpt = pd.concat([existing, df_ckpt]).drop_duplicates("co")
                df_ckpt.to_csv(OUT_PATH, index=False)

    print()
    return results


def load_bios():
    print("Loading CEO bios from deals file...")
    d = load_deals()
    d["co"] = d["Companies"].str.strip()
    cdf = pd.read_csv("output/cpu/cpu_company_dataset.csv")
    study_cos = set(cdf["Companies"].str.strip())
    d_ceo = (d[d["co"].isin(study_cos)]
             .sort_values("Year", ascending=False)
             .drop_duplicates("co")[["co", "CEO Biography"]])
    d_ceo = d_ceo.dropna(subset=["CEO Biography"])
    print(f"  {len(d_ceo):,} companies with CEO bios")
    return d_ceo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="1", choices=["1","2","all"])
    args = parser.parse_args()

    d_ceo = load_bios()

    # Find already classified
    if os.path.exists(OUT_PATH) and args.phase != "all":
        done_cos = set(pd.read_csv(OUT_PATH)["co"])
        print(f"  Already classified: {len(done_cos):,}")
        remaining = d_ceo[~d_ceo["co"].isin(done_cos)]
    else:
        remaining = d_ceo
        if args.phase == "all" and os.path.exists(OUT_PATH):
            os.remove(OUT_PATH)

    print(f"  Remaining to classify: {len(remaining):,}")

    if args.phase == "1":
        batch = remaining.head(100).to_dict("records")
        print(f"\nPhase 1: classifying {len(batch)} bios (test batch)...")
    else:
        batch = remaining.to_dict("records")
        print(f"\nPhase 2/all: classifying {len(batch)} bios...")

    t0 = time.time()
    results = run_batch(batch, desc=f"Phase {args.phase}")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  ({elapsed/len(batch):.1f}s/bio)")

    # Save
    df_new = pd.DataFrame(results)
    if os.path.exists(OUT_PATH) and args.phase != "all":
        existing = pd.read_csv(OUT_PATH)
        df_new = pd.concat([existing, df_new]).drop_duplicates("co")
    df_new.to_csv(OUT_PATH, index=False)
    print(f"  Saved to {OUT_PATH}  (total: {len(df_new):,} rows)")

    # Summary
    phase1 = df_new.dropna(subset=["technical"])
    if len(phase1):
        print(f"\n  Classification rates (N={len(phase1):,}):")
        for col in ["technical","serial_founder","green_mission","elite_education"]:
            rate = phase1[col].astype(float).mean() * 100
            print(f"    {col:<22} {rate:.1f}%")
        high_conf = (phase1["confidence"] == "high").mean() * 100
        print(f"    confidence=high:       {high_conf:.1f}%")
        errors = (phase1["error"] != "").sum()
        print(f"    errors:                {errors}")


if __name__ == "__main__":
    main()
