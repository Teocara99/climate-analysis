"""
Script 107: Fill the PitchBook upload template for ALL investors in the deals data.

Template header (exact, all fields quoted):
"Firm Name","Website","HQ City","HQ State/Province","HQ Country","Ticker"

1. Comma-parse Investors -> unique original names
2. Match to Investors_data.xlsx (Script 104/105 rule)
3. Drop obvious individuals
4. Fill fields (matched from PitchBook, unmatched from deals + descriptions JSON)
5. Sort by climate deal count, descending
6. Write quoted CSV, split into equal-size files if over the per-upload limit
"""

import os, re, csv, json, warnings
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = "output/pitchbook_upload"
os.makedirs(OUT_DIR, exist_ok=True)

TEMPLATE_COLS = ["Firm Name", "Website", "HQ City",
                 "HQ State/Province", "HQ Country", "Ticker"]

# PitchBook's bulk-list upload cap. ASSUMPTION — verify against your account.
ROWS_PER_UPLOAD = 1000

def sep(title=""):
    print(f"\n{'─'*70}")
    if title:
        print(f"  {title}")
        print(f"{'─'*70}")

# ── name helpers (identical rule to Scripts 104/105) ────────────────
def clean_name(s):
    s = re.sub(r"\s*\(.*?\)\s*$", "", str(s)).strip()
    suffixes = (r"\b(llc|lp|l\.p\.|llp|ltd|limited|inc|incorporated|corp|"
                r"corporation|plc|gmbh|bv|b\.v\.|sas|sarl|ag|sa|nv|pty|"
                r"fund|partners|capital|ventures|management|group|holdings?"
                r"|advisors?|associates?)\b\.?")
    s = re.sub(suffixes, "", s, flags=re.IGNORECASE).strip().rstrip(",. ")
    return s.lower().strip()

def country_to_region(c):
    c = str(c).strip()
    americas = {"United States","Canada","Brazil","Mexico","Argentina","Chile",
                "Colombia","Peru","Uruguay","Costa Rica"}
    europe   = {"United Kingdom","Germany","France","Sweden","Netherlands",
                "Switzerland","Spain","Italy","Denmark","Finland","Norway",
                "Belgium","Austria","Portugal","Ireland","Luxembourg",
                "Poland","Czech Republic","Estonia","Latvia","Lithuania"}
    asia     = {"China","India","Japan","South Korea","Singapore","Israel",
                "Hong Kong","Taiwan","Indonesia","Malaysia","Vietnam","Thailand"}
    oceania  = {"Australia","New Zealand"}
    mena     = {"United Arab Emirates","Saudi Arabia","Egypt","Qatar","Kuwait",
                "Bahrain","Jordan","Turkey"}
    africa   = {"South Africa","Kenya","Nigeria","Ghana"}
    if c in americas: return "Americas"
    if c in europe:   return "Europe"
    if c in asia:     return "Asia"
    if c in oceania:  return "Oceania"
    if c in mena:     return "Middle East"
    if c in africa:   return "Africa"
    return "Other"

def comma_parse(s):
    parts = re.split(r",\s*|\n", str(s))
    return [re.sub(r"\s*\(.*?\)\s*$", "", p).strip()
            for p in parts if p.strip() and p.strip() != "nan"]

# ── individual detection ────────────────────────────────────────────
FIRM_KEYWORDS = {
    "capital","ventures","venture","partners","partner","fund","funds","llc","lp","llp",
    "inc","ltd","limited","corp","corporation","plc","gmbh","bv","ag","sa","nv","sas",
    "sarl","pty","oy","ab","as","aps","spa","srl","kk","pte","group","holding","holdings",
    "management","managers","advisors","advisers","advisory","associates","investments",
    "investment","investor","investors","equity","bank","banking","bancorp","trust",
    "foundation","technologies","technology","labs","lab","accelerator","incubator",
    "angels","angel","network","office","asset","assets","securities","financial",
    "finance","vc","pe","seed","growth","innovation","innovations","energy","climate",
    "green","sustainable","sustainability","impact","co","company","enterprises",
    "enterprise","industries","industrial","international","global","systems","solutions",
    "development","agency","authority","ministry","department","university","institute",
    "council","association","society","program","programme","initiative","consortium",
    "union","federation","chamber","syndicate","collective","studio","factory","works",
    "house","hub","platform","digital","tech","media","health","bio","pharma","auto",
    "motors","power","electric","grid","solar","wind","water","oil","gas","mining",
    "steel","chemical","chemicals","materials","resources","infrastructure","realty",
    "estate","properties","property","reit","insurance","assurance","mutual","pension",
    "retirement","endowment","family","sovereign","wealth","treasury","exchange",
    "markets","market","trading","brokers","broker","sciences","science","medical",
    "therapeutics","robotics","mobility","transport","logistics","aerospace","defense",
    "telecom","communications","software","data","cloud","cyber","analytics","ai",
    "bancaria","banca","banco","banque","kapital","capitale","fonds","fondo","fondation",
    "stiftung","invest","investering","beteiligungs","vermogens","gestion","gestao",
    "sgr","sicav","spv","trustees","nominees","limitada","holdingen","beteiligung",
    "state","national","federal","municipal","regional","city","county","province",
    "bancorp","savings","credit","loan","lending","debt","mezzanine","buyout","growth",
    "secondary","fof","pension","superannuation","provident","sovereign",
}

PARTICLE = r"(?:van|von|de|den|der|del|della|di|da|das|dos|du|la|le|el|al|bin|ibn|ter|ten|op|af|av)"
NAME_TOK = r"[A-Z][A-Za-z''\-]+"
INITIAL  = r"[A-Z]\."
SUFFIX   = r"(?:Jr\.?|Sr\.?|II|III|IV|PhD|MD)"

PERSON_RE = re.compile(
    rf"^{NAME_TOK}(?:\s+(?:{INITIAL}|{PARTICLE}))?\s+{NAME_TOK}(?:\s+{SUFFIX})?$"
)

# A capitalised two-word name is NOT evidence of a person on its own
# ("Innovate UK", "Cleantech Open", "EIT Food" all match that shape), so the
# surname slot additionally rejects place and institution words.
SURNAME_STOP = {
    "uk","usa","us","eu","europe","europa","america","asia","africa","canada",
    "china","india","japan","korea","germany","france","spain","italy","brazil",
    "mexico","israel","london","paris","berlin","boston","chicago","texas",
    "commission","parliament","government","ministry","republic","kingdom",
    "open","food","foods","cities","city","valley","forum","ocean","sea","lake",
    "years","year","first","second","third","north","south","east","west",
    "derivative","electrons","gamechanger","horizon","circular","nation",
    "nations","world","earth","planet","future","futures","next","now","one",
}

def has_firm_keyword(name):
    toks = re.split(r"[\s,.&/()\-]+", str(name).lower())
    return any(t in FIRM_KEYWORDS for t in toks if t)

def looks_like_person(name, given_names):
    n = str(name).strip()
    if has_firm_keyword(n):
        return False
    if any(ch in n for ch in "&,/0123456789"):
        return False
    if not PERSON_RE.match(n):
        return False
    toks = n.split()
    if toks[0].lower() not in given_names:
        return False
    if toks[-1].lower().strip(".") in SURNAME_STOP:
        return False
    return True

# ── website cleaning ────────────────────────────────────────────────
TRACKING = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content",
            "utm_id","utm_name","fbclid","gclid","gclsrc","dclid","msclkid",
            "yclid","mc_cid","mc_eid","_ga","_gl","ref","referrer","source",
            "campaign","medium","igshid","si","spm","trk","trkCampaign"}
BAD_WEB = {"", "nan", "none", "n/a", "na", "null", "-", "unknown"}

def clean_website(w):
    if w is None:
        return ""
    w = str(w).strip()
    if w.lower() in BAD_WEB:
        return ""
    had_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", w))
    probe = w if had_scheme else "http://" + w
    try:
        sp = urlsplit(probe)
    except ValueError:
        return ""
    if not sp.netloc:
        return ""
    q = [(k, v) for k, v in parse_qsl(sp.query, keep_blank_values=True)
         if k.lower() not in TRACKING]
    path = sp.path.rstrip("/") if sp.path != "/" else ""
    rebuilt = urlunsplit((sp.scheme, sp.netloc, path, urlencode(q), ""))
    if not had_scheme:
        rebuilt = re.sub(r"^https?://", "", rebuilt)
    return rebuilt

print("=" * 70)
print("SCRIPT 107 — PITCHBOOK UPLOAD TEMPLATE (all investors)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. Unique investor names from the deals file
# ─────────────────────────────────────────────────────────────────────
sep("1. Unique investor names from deals file")

import sys
sys.path.insert(0, ".")
from load_data import load_deals

dl = load_deals()
dl["Deal Date"] = pd.to_datetime(dl["Deal Date"], errors="coerce")

rows = []
for _, r in dl.dropna(subset=["Investors"]).iterrows():
    for inv in comma_parse(r["Investors"]):
        rows.append((inv, clean_name(inv), r["Deal ID"]))
pairs = pd.DataFrame(rows, columns=["investor_raw", "investor_key", "deal_id"])

deal_counts = (pairs.groupby("investor_raw")["deal_id"].nunique()
               .rename("climate_deals"))
inv = (pairs[["investor_raw", "investor_key"]]
       .drop_duplicates("investor_raw")
       .set_index("investor_raw")
       .join(deal_counts)
       .reset_index())

n_undated = int(dl["Deal Date"].isna().sum())
print(f"  Investor-deal pairs                 : {len(pairs):,}")
print(f"  Unique investor names (original)    : {len(inv):,}")
print(f"  Unique normalised keys              : {inv['investor_key'].nunique():,}")
print(f"  Total climate deals covered         : {pairs['deal_id'].nunique():,}")
print(f"\n  SCOPE  This includes deals with no Deal Date ({n_undated:,} deals), because")
print(f"         the task is every investor in the deals data. Scripts 104-106")
print(f"         dropped undated deals for the time-window analysis and so reported")
print(f"         72,771 pairs / 24,388 names. The extra rows here are investors")
print(f"         whose deals carry no date — still valid upload targets.")

# ─────────────────────────────────────────────────────────────────────
# 2. Match to Investors_data.xlsx (Script 104/105 rule)
# ─────────────────────────────────────────────────────────────────────
sep("2. Match to Investors_data.xlsx")

PB_COLS = ["Investors", "Primary Investor Type", "HQ City", "HQ State/Province",
           "HQ Country/Territory/Region", "Website", "Exchange"]
pb = pd.read_excel("data/Investors_data.xlsx", header=7, usecols=PB_COLS)
pb["_key"] = pb["Investors"].apply(clean_name)
pb = pb[~pb["_key"].duplicated(keep="first")].copy()
pb["pb_region"] = pb["HQ Country/Territory/Region"].apply(country_to_region)
pb_idx = pb.set_index("_key")

print(f"  PitchBook records (deduped on key)  : {len(pb):,}")
print(f"  Website column non-null             : {pb['Website'].notna().sum():,}"
      f"  <-- EMPTY, see note below")
print(f"  Exchange column non-null            : {pb['Exchange'].notna().sum():,}")

ip = pd.read_csv("output/investor_profiles.csv")[["investor", "geo_focus"]].copy()
ip["_key"] = ip["investor"].apply(clean_name)
geo_map = ip.drop_duplicates("_key").set_index("_key")["geo_focus"].to_dict()

inv["key_in_pb"]   = inv["investor_key"].isin(pb_idx.index)
inv["single_word"] = inv["investor_raw"].apply(lambda s: len(str(s).strip().split()) == 1)
inv["our_region"]  = inv["investor_key"].map(geo_map)
inv["pb_region"]   = inv["investor_key"].map(pb_idx["pb_region"].to_dict())

both = (inv["our_region"].notna() & inv["pb_region"].notna()
        & (inv["our_region"] != "Other") & (inv["pb_region"] != "Other"))
inv["country_conflict"] = both & (inv["our_region"] != inv["pb_region"])
inv["matched"] = inv["key_in_pb"] & ~inv["single_word"] & ~inv["country_conflict"]

n_tot = len(inv)
print(f"\n  Key present in PitchBook            : {int(inv['key_in_pb'].sum()):,}"
      f" ({inv['key_in_pb'].mean()*100:.1f}%)")
print(f"  Rejected — single-word original name: "
      f"{int((inv['key_in_pb'] & inv['single_word']).sum()):,}")
print(f"  Rejected — HQ-country mismatch      : "
      f"{int((inv['key_in_pb'] & inv['country_conflict'] & ~inv['single_word']).sum()):,}")
print(f"  MATCHED                             : {int(inv['matched'].sum()):,}"
      f" ({inv['matched'].mean()*100:.1f}%)")
print(f"  UNMATCHED                           : {int((~inv['matched']).sum()):,}"
      f" ({(~inv['matched']).mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 3. Drop obvious individuals
# ─────────────────────────────────────────────────────────────────────
sep("3. Drop obvious individuals")

pb_type = pb_idx["Primary Investor Type"].to_dict()

# Given-name lexicon built from PitchBook's own 'Angel (individual)' records,
# so the surname-pattern test fires only on plausible personal first names.
angel_names = pb.loc[pb["Primary Investor Type"] == "Angel (individual)", "Investors"]
GIVEN_NAMES = {str(n).strip().split()[0].lower() for n in angel_names
               if str(n).strip() and str(n).strip().split()[0].isalpha()
               and len(str(n).strip().split()[0]) > 1}
print(f"  PitchBook 'Angel (individual)' records : {len(angel_names):,}")
print(f"  Given-name lexicon derived from them   : {len(GIVEN_NAMES):,} first names")

inv["pb_is_angel"] = (inv["investor_key"].map(pb_type) == "Angel (individual)") & inv["matched"]
inv["heur_person"] = (~inv["matched"]) & inv["investor_raw"].apply(
    lambda s: looks_like_person(s, GIVEN_NAMES))
inv["is_individual"] = inv["pb_is_angel"] | inv["heur_person"]

n_ind = int(inv["is_individual"].sum())
print(f"\n  Rule applied in two parts:")
print(f"    MATCHED   -> PitchBook Primary Investor Type == 'Angel (individual)'")
print(f"    UNMATCHED -> no firm keyword AND Firstname-Lastname pattern AND")
print(f"                 first token is a known given name AND surname not a place word")
print(f"\n  Dropped as individuals: {n_ind:,} of {n_tot:,} ({n_ind/n_tot*100:.1f}%)")
print(f"    via PitchBook type (matched)   : {int(inv['pb_is_angel'].sum()):,}")
print(f"    via heuristic (unmatched)      : {int(inv['heur_person'].sum()):,}")

for lbl, msk in [("PitchBook-typed individuals", inv["pb_is_angel"]),
                 ("heuristic individuals", inv["heur_person"])]:
    ex = (inv[msk].sort_values("climate_deals", ascending=False)
          .head(20)[["investor_raw", "climate_deals"]])
    print(f"\n  20 examples — {lbl} (highest deal count first):")
    print(f"    {'name':<40}{'deals':>7}")
    for _, r in ex.iterrows():
        print(f"    {str(r['investor_raw'])[:38]:<40}{r['climate_deals']:>7}")

# Sanity check: organisations that the naive pattern would have caught
naive = inv["investor_raw"].apply(lambda s: bool(PERSON_RE.match(str(s).strip()))
                                  and not has_firm_keyword(s))
rescued = inv[naive & ~inv["is_individual"]].nlargest(10, "climate_deals")
print(f"\n  Organisations rescued from the naive pattern "
      f"({int((naive & ~inv['is_individual']).sum()):,} total), top 10:")
for _, r in rescued.iterrows():
    print(f"    {str(r['investor_raw'])[:38]:<40}{r['climate_deals']:>7} deals")

kept = inv[~inv["is_individual"]].copy()
print(f"\n  Rows retained: {len(kept):,}")

inv.to_csv(f"{OUT_DIR}/investor_match_audit.csv", index=False)
print(f"  Full audit (incl. dropped) -> investor_match_audit.csv")

# ─────────────────────────────────────────────────────────────────────
# 4. Fill template fields
# ─────────────────────────────────────────────────────────────────────
sep("4. Fill template fields")

with open("output/investor_descriptions.json") as f:
    desc = json.load(f)
web_json = {k: clean_website(v.get("website")) for k, v in desc.items()
            if isinstance(v, dict)}
n_web_json = sum(1 for v in web_json.values() if v)
print(f"  investor_descriptions.json entries  : {len(desc):,}")
print(f"  ...with a usable website            : {n_web_json:,}")

pb_name = pb_idx["Investors"].to_dict()
pb_city = pb_idx["HQ City"].to_dict()
pb_st   = pb_idx["HQ State/Province"].to_dict()
pb_ctry = pb_idx["HQ Country/Territory/Region"].to_dict()
pb_web  = pb_idx["Website"].to_dict()

def blank(v):
    if v is None: return ""
    if isinstance(v, float) and np.isnan(v): return ""
    s = str(v).strip()
    return "" if s.lower() in BAD_WEB else s

out_rows = []
web_src = {"pitchbook": 0, "json": 0, "none": 0}

for _, r in kept.iterrows():
    k, raw = r["investor_key"], r["investor_raw"]
    if r["matched"]:
        firm  = blank(pb_name.get(k)) or raw
        city  = blank(pb_city.get(k))
        state = blank(pb_st.get(k))
        ctry  = blank(pb_ctry.get(k))
        site  = clean_website(pb_web.get(k))
        if site:
            web_src["pitchbook"] += 1
        else:
            # PitchBook's Website column is empty for every record; fall back to
            # the same scraped source used for unmatched rows. Not invented.
            site = web_json.get(raw, "")
            web_src["json" if site else "none"] += 1
    else:
        firm  = raw
        city = state = ctry = ""
        site = web_json.get(raw, "")
        web_src["json" if site else "none"] += 1

    out_rows.append({
        "Firm Name": firm,
        "Website": site,
        "HQ City": city,
        "HQ State/Province": state,
        "HQ Country": ctry,
        "Ticker": "",          # no Ticker column exists in Investors_data.xlsx
        "_deals": int(r["climate_deals"]),
        "_matched": bool(r["matched"]),
    })

out = pd.DataFrame(out_rows)
print(f"\n  Website provenance:")
print(f"    from PitchBook Website column     : {web_src['pitchbook']:,}")
print(f"    from investor_descriptions.json    : {web_src['json']:,}")
print(f"    left blank                        : {web_src['none']:,}")
print(f"\n  NOTE  PitchBook's Website column is empty for all {len(pb):,} records, so")
print(f"        matched rows take the website from investor_descriptions.json —")
print(f"        the same source approved for unmatched rows. No values invented.")
n_ticker_in_name = int(pb["Investors"].astype(str)
                       .str.contains(r"\([A-Z]{2,4}:\s*[A-Z0-9.]+\)", regex=True).sum())
print(f"  NOTE  Investors_data.xlsx has no Ticker column (only 'Exchange', the venue")
print(f"        code). Ticker is left blank per instruction. {n_ticker_in_name:,} PitchBook")
print(f"        names embed a ticker like '(NYS: BLK)' and could be parsed on request.")

print(f"\n  HQ field coverage on matched rows:")
mt = out[out["_matched"]]
for c in ["HQ City", "HQ State/Province", "HQ Country"]:
    nn = int((mt[c] != "").sum())
    print(f"    {c:<22}{nn:>7,} / {len(mt):,} ({nn/len(mt)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────
# 5. Sort by climate deal count
# ─────────────────────────────────────────────────────────────────────
sep("5. Sort by climate deal count (descending)")

out = out.sort_values(["_deals", "Firm Name"], ascending=[False, True]).reset_index(drop=True)
print(f"  Top 10 rows:")
print(f"    {'Firm Name':<40}{'deals':>7}  {'matched':<8}website")
for _, r in out.head(10).iterrows():
    print(f"    {r['Firm Name'][:38]:<40}{r['_deals']:>7}  "
          f"{str(r['_matched']):<8}{r['Website'][:30]}")

# ─────────────────────────────────────────────────────────────────────
# 6. Write quoted CSV(s)
# ─────────────────────────────────────────────────────────────────────
sep("6. Write files")

final = out[TEMPLATE_COLS].copy()
total = len(final)

def write_csv(df, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        w.writerow(TEMPLATE_COLS)
        for _, r in df.iterrows():
            w.writerow([r[c] for c in TEMPLATE_COLS])

full_path = f"{OUT_DIR}/pitchbook_upload_all_investors.csv"
write_csv(final, full_path)
print(f"  Unsplit file: {full_path}  ({total:,} rows)")

n_files = int(np.ceil(total / ROWS_PER_UPLOAD))
file_report = []
if n_files > 1:
    # equal-size chunks (sizes differ by at most 1)
    bounds = [int(round(i * total / n_files)) for i in range(n_files + 1)]
    print(f"\n  {total:,} rows exceeds the {ROWS_PER_UPLOAD:,}-row per-upload cap")
    print(f"  -> splitting into {n_files} equal-size files")
    width = len(str(n_files))
    for i in range(n_files):
        lo, hi = bounds[i], bounds[i + 1]
        p = f"{OUT_DIR}/pitchbook_upload_all_{str(i+1).zfill(width)}of{n_files}.csv"
        write_csv(final.iloc[lo:hi], p)
        file_report.append((os.path.basename(p), hi - lo))
else:
    file_report.append((os.path.basename(full_path), total))

n_with_web = int((final["Website"] != "").sum())

sep("REPORT")
print(f"  Total rows written                  : {total:,}")
print(f"  Rows with a website                 : {n_with_web:,} ({n_with_web/total*100:.1f}%)")
print(f"  Rows without a website              : {total-n_with_web:,}")
print(f"  Matched / unmatched                 : {int(out['_matched'].sum()):,} / "
      f"{int((~out['_matched']).sum()):,}")
print(f"  Files                               : {len(file_report)}")
print(f"\n  {'file':<52}{'rows':>8}{'with web':>10}")
start = 0
for name, nrows in file_report:
    chunk = final.iloc[start:start+nrows] if len(file_report) > 1 else final
    nw = int((chunk["Website"] != "").sum())
    print(f"  {name:<52}{nrows:>8,}{nw:>10,}")
    start += nrows

print(f"\n  Encoding UTF-8 | comma-separated | all fields quoted (QUOTE_ALL)")
print(f"  Header: " + ",".join(f'"{c}"' for c in TEMPLATE_COLS))
print(f"\n  ASSUMPTION: {ROWS_PER_UPLOAD:,} rows per upload. Verify against your")
print(f"  PitchBook account limit — if it differs, change ROWS_PER_UPLOAD and rerun.")

print("\n" + "=" * 70)
print("SCRIPT 107 COMPLETE")
print("=" * 70)
