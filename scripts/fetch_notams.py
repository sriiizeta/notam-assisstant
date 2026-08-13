"""
Pulls real NOTAMs from the FAA public NOTAM Search backend (notams.aim.faa.gov).
This is the public API that backs https://notams.aim.faa.gov/notamSearch/ -- no
API key required. aviationweather.gov's /api/data endpoint does not currently
expose a NOTAM product (only METAR/TAF/PIREP/SIGMET/etc), so this is the
real public source used instead.

Mix of busy and small US airports for variety, per project brief section 6.
"""
import json
import os
import re
import requests

SEARCH_URL = "https://notams.aim.faa.gov/notamSearch/search"
HEADERS = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"}

AIRPORTS = [
    "KJFK", "KATL", "KORD", "KBOS", "KBNA",  # busy
    "KASE", "KOSH", "KACK", "KMVY", "KEGE",  # small
    "KTEB", "KBED", "KHPN", "KFRG", "KISP", "KPWK", "KVNY", "KAPA",  # GA-heavy, added to
    # widen coverage of admin/airspace/informational NOTAMs (see failure_analysis.md --
    # the original 10-airport pull had only 1 admin/1 airspace/0 informational examples)
]

# ARTCC/FIR designators (matching the airports above). The docs/labeling_guide.md
# "Known corpus limitation" note originally claimed a per-airport search
# structurally can't surface TFR/airspace-level NOTAMs -- that assumption
# turned out to be wrong: designatorsForLocation also accepts a bare ARTCC
# code (e.g. "ZNY") and returns genuinely different, area-wide route/airspace
# NOTAMs. Kept as a second list rather than folded into AIRPORTS because the
# labeling rule for these is different (see scripts/apply_labels.py).
FIRS = ["ZNY", "ZBW", "ZAU", "ZTL", "ZME", "ZDV"]

OUTPUT_PATH = "data/raw/notams.json"


def clean_icao_message(raw: str) -> str:
    """A handful of FAA feed records embed HTML tags (<b>, <br>, </B>, ...)
    around the field markers instead of plain text, which breaks the
    parser's field-boundary lookahead (it expects 'A) ... B)' with only
    whitespace between fields, not '</b>KBNA<br> <b>B) '). Strip tags here
    so the parser always sees plain ICAO-format text."""
    text = re.sub(r"</?[a-zA-Z]+\s*/?>", " ", raw)
    return re.sub(r"[ \t]+", " ", text).strip()


def fetch_for_designator(designator: str):
    payload = {"searchType": "0", "designatorsForLocation": designator}
    r = requests.post(SEARCH_URL, data=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("notamList", [])


def fetch_records(designators):
    records = []
    seen_ids = set()
    for designator in designators:
        notam_list = fetch_for_designator(designator)
        print(f"{designator}: fetched {len(notam_list)} NOTAMs")
        for n in notam_list:
            icao_msg = clean_icao_message(n.get("icaoMessage") or "")
            if not icao_msg or icao_msg.upper() == "NOT AVAILABLE":
                continue
            # a handful of FIR-level records render as broken template output
            # (e.g. "A) locationIndicatorICAO ... E) ROUTE undefined,undefined")
            # -- a real FAA-side data-quality issue, not something to parse around
            if "undefined" in icao_msg:
                continue
            notam_number = (n.get("notamNumber") or "").replace("/", "-").replace(" ", "")
            rec_id = f"{designator}-{notam_number}"
            if rec_id in seen_ids:
                continue
            seen_ids.add(rec_id)
            records.append({
                "id": rec_id,
                "route_airports": [designator],
                "text": icao_msg,
            })
    return records


def main():
    all_records = fetch_records(AIRPORTS) + fetch_records(FIRS)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved {len(all_records)} NOTAMs -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
