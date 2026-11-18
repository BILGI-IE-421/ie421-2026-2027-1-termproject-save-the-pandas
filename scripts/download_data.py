import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


"""
download_data.py

Downloads the project's original source data to a local downloads/raw folder.

Raw source files are intentionally NOT saved in data/, because the project
guideline reserves data/ for the data card and small sample/aggregated CSVs.

Sources:
- NOAA Significant Earthquake Database (HazEL API/JSON)
- World Bank GDP per capita, NY.GDP.PCAP.KD (API/JSON)
- FRED CPIAUCNS (CSV export)

Run from the repository root:
    python scripts/download_data.py
"""


# --------------------------------------------------
# 1. PROJECT DIRECTORY
# --------------------------------------------------

def find_project_root() -> Path:
    """
    Find the repository root.

    Supports:
    - Running as a regular Python script
    - Running from a Jupyter Notebook or Google Colab
    """

    # Running as a .py file
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent

    # Running from Jupyter / Google Colab
    current_dir = Path.cwd()

    for folder in [current_dir, *current_dir.parents]:
        if (
            (folder / "scripts").is_dir()
            and (folder / "data").is_dir()
        ):
            return folder

    raise FileNotFoundError(
        "Project root not found. "
        "Run the notebook from the repository root."
    )


ROOT = find_project_root()

RAW_DIR = ROOT / "downloads" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. HTTP SESSION
# --------------------------------------------------

SESSION = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    status=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True
)

adapter = HTTPAdapter(max_retries=retry_strategy)

SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)

SESSION.headers.update({
    "User-Agent": "IE421-Save-the-Pandas/1.0"
})

REQUEST_TIMEOUT = (30, 300)


# --------------------------------------------------
# 3. DATA SOURCE URLS
# --------------------------------------------------

# NOAA HazEL earthquake API
NOAA_URL = (
    "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/earthquakes"
)

# World Bank GDP per capita (constant 2015 US$)
WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/all/"
    "indicator/NY.GDP.PCAP.KD"
)

# World Bank country metadata
WORLD_BANK_COUNTRIES_URL = (
    "https://api.worldbank.org/v2/country"
)

# FRED CPIAUCNS CSV download
FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
)


# --------------------------------------------------
# 4. NOAA DATA
# --------------------------------------------------

def download_noaa() -> None:
    """
    Download NOAA Significant Earthquake records through the HazEL API.

    Save the original API response data as JSON and create CSV/TSV
    versions for easier inspection and preprocessing.
    """

    print("\n[1/3] NOAA Significant Earthquake Database")

    params = {
        "minYear": 1960,
        "maxYear": 2024,
        "page": 1,
        "itemsPerPage": 200
    }

    all_items = []
    page_metadata = []

    while True:
        response = SESSION.get(
            NOAA_URL,
            params=params,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        result = response.json()

        if not isinstance(result, dict) or "items" not in result:
            raise RuntimeError(
                "Unexpected NOAA API response structure. "
                f"Received: {str(result)[:500]}"
            )

        items = result.get("items") or []
        total_items = int(
            result.get("totalItems", len(items))
        )

        all_items.extend(items)

        print(
            f"     Page {params['page']}: "
            f"{len(items)} records downloaded "
            f"({len(all_items)}/{total_items})"
        )

        page_metadata.append({
            "page": params["page"],
            "totalItems": total_items
        })

        if not items or len(all_items) >= total_items:
            break

        params["page"] += 1

    if not all_items:
        raise RuntimeError(
            "NOAA API returned no earthquake records. "
            "Check the endpoint and API parameters."
        )

    # Save API data as JSON.
    json_path = RAW_DIR / "noaa_significant_earthquakes.json"

    json_payload = {
        "source": NOAA_URL,
        "date_range": "1960-2024",
        "page_metadata": page_metadata,
        "observations": all_items
    }

    json_path.write_text(
        json.dumps(
            json_payload,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"[OK] {json_path.relative_to(ROOT)}")

    # Convert records into a DataFrame.
    noaa = pd.json_normalize(all_items)

    if noaa.empty:
        raise RuntimeError(
            "NOAA data could not be converted into a non-empty table."
        )

    # Save CSV.
    csv_path = RAW_DIR / "noaa_significant_earthquakes.csv"
    noaa.to_csv(csv_path, index=False)

    print(f"[OK] {csv_path.relative_to(ROOT)}")

    # Save TSV for tab-separated compatibility.
    tsv_path = RAW_DIR / "noaa_significant_earthquakes.tsv"
    noaa.to_csv(tsv_path, sep="\t", index=False)

    print(f"[OK] {tsv_path.relative_to(ROOT)}")
    print(f"     {len(noaa):,} earthquake records downloaded.")


# --------------------------------------------------
# 5. WORLD BANK COUNTRY METADATA
# --------------------------------------------------

def get_world_bank_country_codes() -> set:
    """
    Retrieve valid country ISO3 codes from the World Bank API.

    Exclude aggregate regions such as WLD and HIC.
    """

    params = {
        "format": "json",
        "per_page": 400
    }

    response = SESSION.get(
        WORLD_BANK_COUNTRIES_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    result = response.json()

    if not isinstance(result, list) or len(result) < 2:
        raise RuntimeError(
            "Unexpected World Bank country metadata response."
        )

    countries = result[1] or []
    valid_codes = set()

    for country in countries:
        iso3 = country.get("id")
        region = country.get("region", {}).get("value")

        if (
            iso3
            and len(iso3) == 3
            and region != "Aggregates"
        ):
            valid_codes.add(iso3.upper())

    return valid_codes


# --------------------------------------------------
# 6. WORLD BANK GDP DATA
# --------------------------------------------------

def download_world_bank() -> None:
    """
    Download all World Bank API pages.

    Preserve the JSON response and create an analysis-friendly CSV.
    """

    print("\n[2/3] World Bank GDP per capita API")

    params = {
        "date": "1960:2024",
        "format": "json",
        "per_page": 20000,
        "page": 1
    }

    first_response = SESSION.get(
        WORLD_BANK_URL,
        params=params,
        timeout=REQUEST_TIMEOUT
    )
    first_response.raise_for_status()

    first_page = first_response.json()

    if (
        not isinstance(first_page, list)
        or len(first_page) < 2
    ):
        raise RuntimeError(
            "Unexpected World Bank API response structure."
        )

    metadata = first_page[0]
    observations = first_page[1] or []
    total_pages = int(metadata.get("pages", 1))

    all_pages = [first_page]

    for page_number in range(2, total_pages + 1):
        page_params = dict(
            params,
            page=page_number
        )

        response = SESSION.get(
            WORLD_BANK_URL,
            params=page_params,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        page_result = response.json()

        if (
            not isinstance(page_result, list)
            or len(page_result) < 2
        ):
            raise RuntimeError(
                f"Unexpected World Bank API response "
                f"on page {page_number}."
            )

        all_pages.append(page_result)
        observations.extend(page_result[1] or [])

    # Preserve the complete API response.
    json_path = RAW_DIR / "world_bank_gdp_api.json"

    json_payload = {
        "source": WORLD_BANK_URL,
        "indicator": "NY.GDP.PCAP.KD",
        "indicator_description": (
            "GDP per capita (constant 2015 US$)"
        ),
        "date_range": "1960:2024",
        "page_metadata": [
            page[0] for page in all_pages
        ],
        "observations": observations
    }

    json_path.write_text(
        json.dumps(
            json_payload,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"[OK] {json_path.relative_to(ROOT)}")

    valid_country_codes = get_world_bank_country_codes()

    rows = []

    for item in observations:
        country_code = item.get("countryiso3code")

        if not country_code:
            continue

        country_code = country_code.strip().upper()

        if country_code not in valid_country_codes:
            continue

        rows.append({
            "Country Name": item["country"]["value"],
            "Country Code": country_code,
            "ISO3": country_code,
            "year": int(item["date"]),
            "GDP_per_capita_2015USD": item.get("value")
        })

    wb = pd.DataFrame(rows)

    if wb.empty:
        raise RuntimeError(
            "World Bank API returned no country-level observations."
        )

    wb["GDP_per_capita_2015USD"] = pd.to_numeric(
        wb["GDP_per_capita_2015USD"],
        errors="coerce"
    )

    if wb.duplicated(["ISO3", "year"]).any():
        raise RuntimeError(
            "Duplicate ISO3-year rows found in World Bank API data."
        )

    csv_path = RAW_DIR / "world_bank_gdp_per_capita_api.csv"
    wb.to_csv(csv_path, index=False)

    print(f"[OK] {csv_path.relative_to(ROOT)}")
    print(
        f"     {len(wb):,} country-year rows; "
        f"{wb['ISO3'].nunique()} ISO3 codes."
    )


# --------------------------------------------------
# 7. FRED CPI DATA
# --------------------------------------------------


def download_fred() -> None:
    """Download and validate monthly CPIAUCNS data for 1960-2024."""

    print("\n[3/3] FRED CPIAUCNS (1960-2024)")

    destination = RAW_DIR / "CPIAUCNS.csv"
    temporary_file = RAW_DIR / "CPIAUCNS_download.tmp"

    # Use a separate session with fewer retries for FRED.
    fred_session = requests.Session()

    retry_strategy = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    fred_session.mount("https://", adapter)

    try:
        response = fred_session.get(
            FRED_URL,
            params={
                "cosd": "1960-01-01",
                "coed": "2024-12-31"
            },
            timeout=(30, 90)
        )
        response.raise_for_status()

        if not response.content:
            raise RuntimeError("FRED returned an empty response.")

        temporary_file.write_bytes(response.content)
        cpi = pd.read_csv(temporary_file)

        required_columns = {"observation_date", "CPIAUCNS"}

        if not required_columns.issubset(cpi.columns):
            raise RuntimeError(
                "FRED CSV does not contain the expected columns."
            )

        if cpi.empty:
            raise RuntimeError("FRED CSV contains no observations.")

        cpi["observation_date"] = pd.to_datetime(
            cpi["observation_date"],
            errors="coerce"
        )
        cpi["CPIAUCNS"] = pd.to_numeric(
            cpi["CPIAUCNS"],
            errors="coerce"
        )

        if cpi["observation_date"].isna().any():
            raise RuntimeError("Invalid dates found in FRED CSV.")

        if cpi["CPIAUCNS"].notna().sum() == 0:
            raise RuntimeError("No valid CPI values found.")

        if cpi["observation_date"].duplicated().any():
            raise RuntimeError("Duplicate dates found in FRED CSV.")

        # Confirm that the requested date range was returned.
        years = cpi["observation_date"].dt.year

        if years.min() < 1960 or years.max() > 2024:
            raise RuntimeError("FRED returned dates outside 1960-2024.")

        temporary_file.replace(destination)

        print(f"[OK] {destination.relative_to(ROOT)}")
        print(f"     Monthly observations: {len(cpi):,}")
        print(f"     Year range: {years.min()}-{years.max()}")

    finally:
        fred_session.close()

        if temporary_file.exists():
            temporary_file.unlink()


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("IE 421 | SAVE THE PANDAS | RAW DATA DOWNLOAD")
    print("=" * 65)

    print(f"Project root: {ROOT}")
    print(f"Raw data folder: {RAW_DIR}")

    download_noaa()
    download_world_bank()
    download_fred()

    print("\n" + "=" * 65)
    print("All three source downloads and basic checks completed.")
    print("Raw files are local and should not be committed to data/.")
    print("=" * 65)


if __name__ == "__main__":
    main()
