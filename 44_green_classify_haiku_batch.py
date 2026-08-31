"""
Script 44: Green/Traditional investor classification via Claude Haiku Batch API
===============================================================================

Classifies all investors in Investors_data.xlsx into:
  GREEN_VC    — primary focus is green/climate/clean/renewable investing
  ESG_ALIGNED — sustainability is a stated preference but not the primary mandate
  TRADITIONAL — no meaningful green/climate signal

Input fields used per investor:
  - Description (PitchBook official)
  - Primary Investor Type
  - Preferred Verticals
  - Impact Category Preferences
  - Other Stated Preferences
  - Keywords

Usage:
  python3 44_green_classify_haiku_batch.py --phase submit
  python3 44_green_classify_haiku_batch.py --phase status
  python3 44_green_classify_haiku_batch.py --phase results
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import anthropic
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
MODEL        = "claude-haiku-4-5-20251001"
OUT_DIR      = Path("output/green_classify")
BATCH_IDS    = OUT_DIR / "batch_ids.json"
ID_MAP_FILE  = OUT_DIR / "id_to_investor.json"
RESULTS_FILE = OUT_DIR / "investor_green_classifications.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = """You are classifying venture capital and institutional investors by their green/climate focus.

Classify each investor as exactly ONE of:
- GREEN_VC: Green/climate/clean energy is the PRIMARY investment mandate. The investor exists specifically to fund clean tech, climate solutions, renewable energy, sustainability, or environmental impact. Examples: Breakthrough Energy Ventures, Lowercarbon Capital, Energy Impact Partners, a dedicated solar fund.
- ESG_ALIGNED: Sustainability/ESG/impact is a STATED PREFERENCE but the investor has a broader mandate. They invest across sectors but list climate or ESG as a preference. Examples: large PE firms that "seek ESG investments", generalist VCs with a climate vertical, sovereign wealth funds with ESG policies.
- TRADITIONAL: No meaningful green or climate signal. Invests across sectors without sustainability focus. Examples: a generalist angel investor, a sector-agnostic VC with no ESG mention, an accelerator focused on consumer apps.

Rules:
- If "Climate" appears in Impact Category Preferences AND the description/verticals confirm a focus: likely GREEN_VC
- If "Seeks ESG" or "Seeks Impact" is in preferences but description shows broad mandate: ESG_ALIGNED
- Government/GVC investors funding climate programs: GREEN_VC only if climate is their primary stated mandate
- Impact Investing type investors: GREEN_VC if climate-focused, ESG_ALIGNED if broader social impact
- When uncertain between GREEN_VC and ESG_ALIGNED: prefer ESG_ALIGNED
- When uncertain between ESG_ALIGNED and TRADITIONAL: prefer TRADITIONAL

Respond with valid JSON only:
{"green_focus": "GREEN_VC|ESG_ALIGNED|TRADITIONAL", "confidence": "high|medium|low", "reasoning": "1 sentence max"}"""

PROMPT_TEMPLATE = """Investor: {name}
Type: {inv_type}
Description: {description}
Preferred Verticals: {verticals}
Impact Category Preferences: {impact}
Other Preferences: {other_prefs}
Keywords: {keywords}"""


def build_text(row) -> str:
    def clean(v):
        if pd.isna(v) or str(v).strip() in ("nan", ""):
            return "(none)"
        return str(v).strip()[:400]

    return PROMPT_TEMPLATE.format(
        name=clean(row.get("Investors")),
        inv_type=clean(row.get("Primary Investor Type")),
        description=clean(row.get("Description")),
        verticals=clean(row.get("Preferred Verticals")),
        impact=clean(row.get("Impact Category Preferences")),
        other_prefs=clean(row.get("Other Stated Preferences")),
        keywords=clean(row.get("Keywords")),
    )


def load_investors() -> pd.DataFrame:
    df = pd.read_excel("data/Investors_data.xlsx", sheet_name="Data", header=7)
    df["Investors"] = df["Investors"].str.strip()
    df = df.dropna(subset=["Investors"])
    df = df.drop_duplicates(subset=["Investors"])
    print(f"  Loaded {len(df):,} unique investors from Investors_data.xlsx")
    return df


def make_request(row, idx: int) -> dict:
    return {
        "custom_id": f"inv_{idx}",
        "params": {
            "model": MODEL,
            "max_tokens": 150,
            "temperature": 0.1,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": build_text(row)}],
        },
    }


# ── PHASE: SUBMIT ─────────────────────────────────────────────────────────────
def phase_submit():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    df = load_investors()

    # Skip already completed investors if re-running
    done = set()
    if RESULTS_FILE.exists():
        existing = pd.read_csv(RESULTS_FILE)
        done = set(existing[existing["error"].isna() | (existing["error"] == "")]["investor"])
        print(f"  Already classified: {len(done):,} — skipping")

    todo = df[~df["Investors"].isin(done)].copy().reset_index(drop=True)
    print(f"  To classify: {len(todo):,}")

    if len(todo) == 0:
        print("  Nothing to do.")
        return

    # Build id→name mapping (custom_id must be alphanumeric)
    id_map = {f"inv_{i}": name for i, name in enumerate(todo["Investors"])}
    existing_map = json.loads(ID_MAP_FILE.read_text()) if ID_MAP_FILE.exists() else {}
    existing_map.update(id_map)
    ID_MAP_FILE.write_text(json.dumps(existing_map, indent=1))

    # Split into batches of 10,000 (API limit)
    batch_size = 10_000
    chunks = [todo.iloc[i:i+batch_size] for i in range(0, len(todo), batch_size)]
    print(f"  Submitting {len(chunks)} batch(es) of up to {batch_size:,} requests each…")

    batch_ids = json.loads(BATCH_IDS.read_text()) if BATCH_IDS.exists() else []

    for i, chunk in enumerate(chunks):
        offset = i * batch_size
        requests = [make_request(row, offset + j) for j, (_, row) in enumerate(chunk.iterrows())]
        batch = client.beta.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"  Batch {i+1}/{len(chunks)} submitted: {batch.id}  ({len(requests):,} requests)")
        BATCH_IDS.write_text(json.dumps(batch_ids, indent=1))

    print(f"\n  Batch IDs saved → {BATCH_IDS}")
    print(f"  Run with --phase status to check progress.")


# ── PHASE: STATUS ─────────────────────────────────────────────────────────────
def phase_status():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if not BATCH_IDS.exists():
        print("No batch IDs found. Run --phase submit first.")
        return

    ids = json.loads(BATCH_IDS.read_text())
    all_done = True
    for bid in ids:
        b = client.beta.messages.batches.retrieve(bid)
        counts = b.request_counts
        pct = (counts.succeeded + counts.errored) / max(counts.processing + counts.succeeded + counts.errored, 1) * 100
        print(f"  {bid}: {b.processing_status:12s}  "
              f"succeeded={counts.succeeded:,}  errored={counts.errored:,}  "
              f"processing={counts.processing:,}  ({pct:.1f}% done)")
        if b.processing_status != "ended":
            all_done = False

    if all_done:
        print("\n  All batches complete. Run --phase results to download.")
    else:
        print("\n  Still processing. Check again later.")


# ── PHASE: RESULTS ─────────────────────────────────────────────────────────────
def phase_results():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if not BATCH_IDS.exists():
        print("No batch IDs found.")
        return

    ids = json.loads(BATCH_IDS.read_text())
    records = []

    for bid in ids:
        b = client.beta.messages.batches.retrieve(bid)
        if b.processing_status != "ended":
            print(f"  {bid}: still processing — skipping")
            continue

        id_map = json.loads(ID_MAP_FILE.read_text()) if ID_MAP_FILE.exists() else {}
        print(f"  Downloading results for {bid}…")
        for result in client.beta.messages.batches.results(bid):
            inv = id_map.get(result.custom_id, result.custom_id)
            if result.result.type == "succeeded":
                raw = result.result.message.content[0].text.strip()
                # Strip markdown fences if present
                raw = re.sub(r"^```(json)?", "", raw).strip()
                raw = re.sub(r"```$", "", raw).strip()
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                        green_focus = parsed.get("green_focus", "TRADITIONAL")
                        confidence  = parsed.get("confidence", "low")
                        reasoning   = str(parsed.get("reasoning", ""))[:300]
                        # Validate
                        if green_focus not in ("GREEN_VC","ESG_ALIGNED","TRADITIONAL"):
                            green_focus = "TRADITIONAL"
                        records.append({
                            "investor": inv,
                            "green_focus_haiku": green_focus,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "error": "",
                        })
                        continue
                    except json.JSONDecodeError:
                        pass
                records.append({"investor": inv, "green_focus_haiku": "TRADITIONAL",
                                 "confidence": "low", "reasoning": "", "error": f"parse_error: {raw[:100]}"})
            else:
                err = str(result.result.error) if hasattr(result.result, "error") else "unknown"
                records.append({"investor": inv, "green_focus_haiku": "",
                                 "confidence": "", "reasoning": "", "error": err})

    # Merge with any existing partial results
    if RESULTS_FILE.exists():
        existing = pd.read_csv(RESULTS_FILE)
        existing_done = set(existing[existing["error"].isna() | (existing["error"] == "")]["investor"])
        new = pd.DataFrame(records)
        new = new[~new["investor"].isin(existing_done)]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = pd.DataFrame(records)

    combined.to_csv(RESULTS_FILE, index=False)
    ok  = combined[combined["error"].isna() | (combined["error"] == "")]
    err = combined[combined["error"].notna() & (combined["error"] != "")]
    print(f"\n  Results saved → {RESULTS_FILE}")
    print(f"  Classified: {len(ok):,}  Errors: {len(err):,}")
    print(f"\n  GREEN_VC:    {(ok['green_focus_haiku']=='GREEN_VC').sum():,}  "
          f"({(ok['green_focus_haiku']=='GREEN_VC').mean():.1%})")
    print(f"  ESG_ALIGNED: {(ok['green_focus_haiku']=='ESG_ALIGNED').sum():,}  "
          f"({(ok['green_focus_haiku']=='ESG_ALIGNED').mean():.1%})")
    print(f"  TRADITIONAL: {(ok['green_focus_haiku']=='TRADITIONAL').sum():,}  "
          f"({(ok['green_focus_haiku']=='TRADITIONAL').mean():.1%})")

    # Sample for sanity check
    print("\n  Sample GREEN_VC:")
    for _, r in ok[ok["green_focus_haiku"]=="GREEN_VC"].head(5).iterrows():
        print(f"    {r['investor']:<45} {r['confidence']:>6}  {r['reasoning'][:80]}")
    print("\n  Sample TRADITIONAL:")
    for _, r in ok[ok["green_focus_haiku"]=="TRADITIONAL"].head(5).iterrows():
        print(f"    {r['investor']:<45} {r['confidence']:>6}  {r['reasoning'][:80]}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["submit","status","results"], required=True)
    args = parser.parse_args()

    print(f"\n=== Phase: {args.phase.upper()} ===\n")
    if args.phase == "submit":
        phase_submit()
    elif args.phase == "status":
        phase_status()
    elif args.phase == "results":
        phase_results()
