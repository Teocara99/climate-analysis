"""
Phase 0 — Enrich each unique investor with a short text description, used as
the LLM classification signal in 05b_classify_investors_ollama.py.

For investors with a known website (Investors Websites column): fetch the
homepage and pull the meta description / first significant paragraph.
For investors with no known website: search the web by name (DuckDuckGo HTML,
no API key needed) and use the top result's title+snippet as a description,
falling back to fetching that page's meta description if available.

Checkpointed every 500 investors to output/investor_descriptions.json so a
crash doesn't lose progress — re-running resumes from where it left off.
"""
import json
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd

OUT = Path(__file__).parent / "output"
CACHE_FILE = OUT / "investor_descriptions.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
TIMEOUT = 8


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=1))


def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url


def fetch_meta_description(url: str) -> str:
    try:
        resp = requests.get(normalize_url(url), headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        meta = soup.find("meta", attrs={"name": "description"}) or \
            soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return meta["content"].strip()[:500]
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        first_p = ""
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 40:
                first_p = text
                break
        return (title + ". " + first_p).strip()[:500]
    except Exception:
        return ""


def extract_real_url(ddg_href: str) -> str:
    """DuckDuckGo HTML result links are redirects like '//duckduckgo.com/l/?uddg=<encoded_url>&rut=...'."""
    if not ddg_href:
        return ""
    m = re.search(r"uddg=([^&]+)", ddg_href)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1))
    return "https:" + ddg_href if ddg_href.startswith("//") else ddg_href


def web_search_snippet(name: str) -> tuple[str, str]:
    """Return (snippet_text, found_url) from a DuckDuckGo HTML search."""
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": f"{name} venture capital investor"},
        headers=HEADERS, timeout=TIMEOUT,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    result = soup.find("div", class_="result")
    if not result:
        return "", ""
    title_tag = result.find("a", class_="result__a")
    snippet_tag = result.find("a", class_="result__snippet") or result.find("div", class_="result__snippet")
    title = title_tag.get_text(strip=True) if title_tag else ""
    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
    url = extract_real_url(title_tag["href"]) if title_tag and title_tag.get("href") else ""
    return (title + ". " + snippet).strip()[:500], url


def describe_investor(name: str, website: str | None) -> dict:
    if website and isinstance(website, str) and website.strip():
        desc = fetch_meta_description(website)
        return {"description": desc, "source": "website" if desc else "website_empty", "website": website}
    else:
        try:
            desc, found_url = web_search_snippet(name)
        except Exception:
            return {"description": "", "source": "search_failed", "website": None}
        if desc and found_url:
            page_desc = fetch_meta_description(found_url)
            if page_desc:
                desc = page_desc
        return {"description": desc, "source": "search" if desc else "search_empty", "website": found_url or None}


def main(limit: int | None = None, search_delay: float = 5.0, circuit_breaker: int = 8):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    investors = pd.read_csv(OUT / "unique_investors.csv")
    if limit:
        investors = investors.head(limit)
    cache = load_cache()

    todo = [row for _, row in investors.iterrows() if row["Investor"] not in cache]
    with_site = [r for r in todo if pd.notna(r["Website"])]
    without_site = [r for r in todo if pd.isna(r["Website"])]
    print(f"Total investors: {len(investors):,} | already cached: {len(cache):,} | "
          f"remaining: {len(todo):,} (with site: {len(with_site):,}, without: {len(without_site):,})")

    # ── Pass 1: concurrent scraping for investors with a known website ─────
    def scrape_one(row):
        return row["Investor"], describe_investor(row["Investor"], row["Website"])

    done = 0
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(scrape_one, row) for row in with_site]
        for fut in as_completed(futures):
            name, result = fut.result()
            cache[name] = result
            done += 1
            if done % 500 == 0:
                print(f"  [website scrape] {done:,}/{len(with_site):,}")
                save_cache(cache)
    save_cache(cache)
    print(f"Website scraping done: {len(with_site):,} processed.")

    # ── Pass 2: throttled sequential search for investors with no website ─
    consecutive_failures = 0
    search_disabled = False
    for i, row in enumerate(without_site, start=1):
        name = row["Investor"]
        if search_disabled:
            cache[name] = {"description": "", "source": "name_only_circuit_broken", "website": None}
            continue

        result = describe_investor(name, None)
        cache[name] = result
        if result["source"] in ("search_failed", "search_empty"):
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        if consecutive_failures >= circuit_breaker:
            search_disabled = True
            print(f"  [search] {circuit_breaker} consecutive failures — disabling search, "
                  f"remaining {len(without_site) - i:,} investors will be name-only.")

        if i % 100 == 0:
            print(f"  [search] {i:,}/{len(without_site):,}")
        if i % 500 == 0:
            save_cache(cache)
        time.sleep(search_delay)

    save_cache(cache)
    print(f"Done. Descriptions cached for {len(cache):,} investors → {CACHE_FILE}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
