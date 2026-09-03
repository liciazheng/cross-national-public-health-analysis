"""
Pull the Sub-Saharan Africa panel from the World Bank API.

The three-country dataset in data/hiv_indicators.csv was collected by hand and
turned out to carry several errors (see analysis.py and the README). This script
goes to the primary source instead: every Sub-Saharan African country, every
year, straight from the API, with the raw responses cached so a rerun is free
and the numbers are traceable.

    python fetch_worldbank.py            # uses the cache when present
    python fetch_worldbank.py --refresh  # re-download

Writes data/worldbank_panel.csv.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.worldbank.org/v2"
ROOT = Path(__file__).parent
DATA = ROOT / "data"
CACHE = DATA / "worldbank_cache"

YEARS = (2000, 2022)
REGION = "SSF"  # Sub-Saharan Africa

# Every indicator the original three-country dataset tried to cover, mapped to
# the authoritative World Bank series. The mislabelled "GDP_per_capita" column
# in the hand-collected data is replaced by the real per-capita series here.
INDICATORS = {
    "SH.DYN.AIDS.ZS": "hiv_prevalence_pct",        # % of population ages 15-49
    "SH.HIV.ARTC.ZS": "art_coverage_pct",          # % of people living with HIV
    "SH.HIV.PMTC.ZS": "pmtct_coverage_pct",        # % of pregnant women living with HIV
    "SH.DYN.AIDS.DH": "aids_deaths",               # UNAIDS estimate
    "SH.HIV.INCD.TL": "new_infections",            # all ages
    "SH.DYN.AIDS.FE.ZS": "women_share_of_plhiv_pct",
    "SP.POP.TOTL": "population",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",        # current US$, genuinely per capita
    "SL.UEM.TOTL.ZS": "unemployment_pct",
}


def _cache_key(path):
    """
    A short, filesystem-safe cache name. The full request path runs past the
    Windows 260-character limit once 48 country codes are in the URL, so the
    readable part is truncated and a hash of the whole path keeps it unique.
    """
    label = re.sub(r"[^A-Za-z0-9.]+", "_", path.split("?")[0])[:60].strip("_")
    digest = hashlib.sha1(path.encode()).hexdigest()[:10]
    return f"{label}.{digest}.json"


def _get(path, *, refresh=False, attempts=4):
    """GET a JSON path, caching the raw response under data/worldbank_cache/."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / _cache_key(path)

    if cached.exists() and not refresh:
        return json.loads(cached.read_text(encoding="utf-8"))

    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{API}/{path}", timeout=60) as response:
                payload = json.load(response)
            cached.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {path}: {last}")


def fetch_countries(refresh=False):
    """The 48 Sub-Saharan African economies, with income level. Aggregates excluded."""
    payload = _get(f"country?region={REGION}&format=json&per_page=400", refresh=refresh)
    rows = [
        {
            "iso3": c["id"],
            "country": c["name"],
            "income_level": c["incomeLevel"]["value"],
        }
        # adminregion is deliberately not kept: for this region it reads
        # "Sub-Saharan Africa (excluding high income)" for 47 of the 48 and
        # null for the one high-income economy, so it only restates
        # income_level.
        for c in payload[1]
        # A region query already excludes aggregates, but guard anyway: aggregates
        # report their income level as "Aggregates".
        if c["incomeLevel"]["value"] != "Aggregates"
    ]
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def fetch_indicator(code, iso3_codes, refresh=False):
    """One indicator for every country, long format."""
    codes = ";".join(iso3_codes)
    path = (f"country/{codes}/indicator/{code}"
            f"?date={YEARS[0]}:{YEARS[1]}&format=json&per_page=20000")
    payload = _get(path, refresh=refresh)

    meta = payload[0]
    if meta.get("total", 0) > meta.get("per_page", 0):
        raise RuntimeError(f"{code}: {meta['total']} records exceed one page")

    rows = [
        {"iso3": r["countryiso3code"], "year": int(r["date"]), INDICATORS[code]: r["value"]}
        for r in (payload[1] or [])
        if r["countryiso3code"]
    ]
    return pd.DataFrame(rows)


def build_panel(refresh=False):
    countries = fetch_countries(refresh=refresh)
    iso3 = countries["iso3"].tolist()
    print(f"  {len(iso3)} countries, {YEARS[0]}-{YEARS[1]}")

    panel = None
    for code, name in INDICATORS.items():
        frame = fetch_indicator(code, iso3, refresh=refresh)
        got = frame[name].notna().sum()
        print(f"  {code:<18} -> {name:<26} {got:>5} non-null of {len(frame)}")
        panel = frame if panel is None else panel.merge(frame, on=["iso3", "year"], how="outer")

    panel = countries.merge(panel, on="iso3", how="right")

    # Population-normalised measures, so countries of very different size are
    # comparable — the step the original three-country analysis was missing.
    panel["aids_deaths_per_100k"] = panel["aids_deaths"] / panel["population"] * 100_000
    panel["new_infections_per_100k"] = panel["new_infections"] / panel["population"] * 100_000

    return panel.sort_values(["country", "year"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    args = parser.parse_args()

    print("Fetching World Bank indicators for Sub-Saharan Africa")
    print("=" * 68)
    panel = build_panel(refresh=args.refresh)

    out = DATA / "worldbank_panel.csv"
    panel.to_csv(out, index=False)

    print("=" * 68)
    print(f"Wrote {out.relative_to(ROOT)}  "
          f"({len(panel)} rows x {len(panel.columns)} columns)")
    print(f"  countries: {panel['country'].nunique()}")
    print(f"  years:     {panel['year'].min()}-{panel['year'].max()}")
    print(f"  cache:     {CACHE.relative_to(ROOT)} "
          f"({len(list(CACHE.glob('*.json')))} responses)")


if __name__ == "__main__":
    main()
