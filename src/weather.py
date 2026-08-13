import requests

BASE_URL = "https://aviationweather.gov/api/data"


def fetch_metar(icao: str) -> str:
    """Raw METAR text for an airport, or empty string if unavailable.
    aviationweather.gov's /api/data/metar endpoint is confirmed working (no
    API key) -- unlike its NOTAM endpoint, which doesn't exist (see
    README, Section 4)."""
    r = requests.get(f"{BASE_URL}/metar", params={"ids": icao, "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return ""
    return data[0].get("rawOb", "")


def fetch_taf(icao: str) -> str:
    """Raw TAF text for an airport, or empty string if unavailable."""
    r = requests.get(f"{BASE_URL}/taf", params={"ids": icao, "format": "json"}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        return ""
    return data[0].get("rawTAF", "")


def fetch_weather(icao: str) -> dict:
    return {
        "icao": icao,
        "metar": fetch_metar(icao),
        "taf": fetch_taf(icao),
    }
