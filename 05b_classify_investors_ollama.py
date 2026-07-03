"""
Phase 1/2 — Classify investors via local Ollama (Mistral) into:
  - investor_type: GVC | CVC | IVC | Impact_VC | Bank_VC | Angel_Network | University_VC | OTHER
  - green_focus:   GREEN_VC | ESG_ALIGNED | TRADITIONAL

Usage:
  python3 05b_classify_investors_ollama.py validate   # Phase 1: first 100 -> validation_100_results.json
  python3 05b_classify_investors_ollama.py full       # Phase 2: all investors -> all_investors_classified.json
"""
import json
import re
import sys
import time
import requests
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent / "output"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
REQUEST_TIMEOUT = 30  # generous; "5s" from the spec is unrealistic for a 7B model on CPU

PROMPT_TEMPLATE = """You must classify a venture capital investor. Follow these steps EXACTLY:

1. Read the investor name and description
2. Determine INVESTOR TYPE: Is it GVC, CVC, IVC, Impact_VC, Bank_VC, Angel_Network, University_VC, or OTHER?
3. Determine GREEN FOCUS: Is it GREEN_VC, ESG_ALIGNED, or TRADITIONAL?
4. Rate your confidence on both (high/medium/low)
5. Provide brief reasoning

INVESTOR TYPE RULES — check these FIRST, name patterns are strong signals:
- GVC: ANY government ministry, department, agency, council, commission, public bank, state/national fund,
  or publicly-funded program. This includes things named "Department of X", "National Science Foundation",
  "Innovate UK", "European Commission", "European Innovation Council", "Horizon 2020", state development banks
  (e.g. Bpifrance), sovereign wealth funds (e.g. Temasek), and regional/state economic development agencies
  (e.g. Scottish Enterprise, California Energy Commission). If it is publicly funded or run by a government
  body at any level, it is GVC — even if it funds green/climate projects, GVC always takes priority over
  Impact_VC or IVC.
- CVC: The venture arm of a NON-government company (e.g. Google Ventures, Shell Ventures, Aramco Ventures).
  Only use CVC when the parent organization is a private corporation, not a government body.
- University_VC: Affiliated with a specific university (e.g. Stanford StartX), NOT generic national science
  agencies (those are GVC).
- IVC: An independent PRIVATE firm not owned by government, a corporation, university, or bank.
- Impact_VC: A private, non-government mission-driven fund (e.g. a foundation or private impact fund).
  Do NOT use Impact_VC for government programs even if their mission is climate-related — use GVC instead.
- Bank_VC: Venture arm of a commercial/investment bank (e.g. Goldman Sachs, JPMorgan).
- Angel_Network: Angel syndicates/groups of individual investors.
- OTHER: Accelerators/incubators not clearly tied to one owner type, or anything that truly doesn't fit above.

CRITICAL: You MUST respond with VALID JSON ONLY. No markdown, no extra text, just the JSON object.
The "investor_type" field in your JSON must match the type stated in your own "reasoning" — do not contradict yourself.

INVESTOR DATA:
Name: {investor_name}
Description: {website_description}

RESPOND WITH JSON ONLY:
{{
"investor_type": "GVC|CVC|IVC|Impact_VC|Bank_VC|Angel_Network|University_VC|OTHER",
"green_focus": "GREEN_VC|ESG_ALIGNED|TRADITIONAL",
"confidence_type": "high|medium|low",
"confidence_green": "high|medium|low",
"reasoning": "1-2 sentences explaining classification"
}}"""

VALID_TYPES = {"GVC", "CVC", "IVC", "Impact_VC", "Bank_VC", "Angel_Network", "University_VC", "OTHER"}
VALID_GREEN = {"GREEN_VC", "ESG_ALIGNED", "TRADITIONAL"}
VALID_CONF = {"high", "medium", "low"}


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.3, "num_ctx": 1024}},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate_result(parsed: dict) -> dict:
    return {
        "investor_type": parsed.get("investor_type") if parsed.get("investor_type") in VALID_TYPES else "OTHER",
        "green_focus": parsed.get("green_focus") if parsed.get("green_focus") in VALID_GREEN else "TRADITIONAL",
        "confidence_type": parsed.get("confidence_type") if parsed.get("confidence_type") in VALID_CONF else "low",
        "confidence_green": parsed.get("confidence_green") if parsed.get("confidence_green") in VALID_CONF else "low",
        "reasoning": str(parsed.get("reasoning", ""))[:300],
    }


def classify_one(investor_id, name: str, description: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(investor_name=name, website_description=description or "(no description available)")
    for attempt in range(2):
        try:
            raw = call_ollama(prompt)
        except Exception as e:
            if attempt == 1:
                return _parse_error(investor_id, name, f"request_error: {e}")
            continue
        parsed = extract_json(raw)
        if parsed:
            result = validate_result(parsed)
            result.update({"investor_id": investor_id, "investor_name": name})
            return result
        if attempt == 1:
            return _parse_error(investor_id, name, f"unparseable_response: {raw[:200]}")
    return _parse_error(investor_id, name, "unknown_error")


def _parse_error(investor_id, name, reason) -> dict:
    return {
        "investor_id": investor_id, "investor_name": name,
        "investor_type": "PARSE_ERROR", "green_focus": "PARSE_ERROR",
        "confidence_type": "low", "confidence_green": "low",
        "reasoning": reason,
    }


def load_investors(limit=None) -> pd.DataFrame:
    df = pd.read_csv(OUT / "unique_investors.csv")
    df.insert(0, "investor_id", range(1, len(df) + 1))
    descriptions = json.loads((OUT / "investor_descriptions.json").read_text())
    df["description"] = df["Investor"].map(lambda n: descriptions.get(n, {}).get("description", ""))
    if limit:
        df = df.head(limit)
    return df


def phase1_validate():
    df = load_investors(limit=100)
    results = []
    for i, row in enumerate(df.itertuples(), start=1):
        result = classify_one(row.investor_id, row.Investor, row.description)
        results.append(result)
        print(f"[{i}/100] {row.Investor[:40]:40s} -> {result['investor_type']:14s} {result['green_focus']}")
    (OUT / "validation_100_results.json").write_text(json.dumps(results, indent=1))
    print(f"\nSaved → {OUT / 'validation_100_results.json'}")
    print("\n--- Sample for manual review ---")
    for r in results[::3][:30]:
        print(f"{r['investor_name']:35s} | type={r['investor_type']:14s} ({r['confidence_type']}) "
              f"| green={r['green_focus']:11s} ({r['confidence_green']}) | {r['reasoning']}")


def phase2_full():
    df = load_investors()
    checkpoint_file = OUT / "all_investors_classified.json"
    results = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else []
    done_ids = {r["investor_id"] for r in results}
    todo = df[~df["investor_id"].isin(done_ids)]
    print(f"Total: {len(df):,} | already done: {len(done_ids):,} | remaining: {len(todo):,}")

    for i, row in enumerate(todo.itertuples(), start=1):
        result = classify_one(row.investor_id, row.Investor, row.description)
        results.append(result)
        if i % 500 == 0:
            print(f"  ...processed {i:,}/{len(todo):,}")
        if i % 1000 == 0:
            checkpoint_file.write_text(json.dumps(results, indent=1))
            print(f"  [checkpoint saved at {len(results):,} total]")

    checkpoint_file.write_text(json.dumps(results, indent=1))
    print(f"Done. {len(results):,} investors classified → {checkpoint_file}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        phase1_validate()
    elif mode == "full":
        phase2_full()
    else:
        print("Usage: python3 05b_classify_investors_ollama.py [validate|full]")
