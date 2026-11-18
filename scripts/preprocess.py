import pandas as pd
from pathlib import Path

# Project folders - Safe check for Jupyter Notebook / Google Colab
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

RAW_DIR = ROOT / "downloads" / "raw"
OUTPUT_DIR = ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("IE 421 | SAVE THE PANDAS | DATA PREPROCESSING")
print("=" * 55)

# --------------------------------------------------
# STEP 1: LOAD RAW DATA
# --------------------------------------------------

# Check that all required input files exist.
required_files = {
    "NOAA": RAW_DIR / "noaa_significant_earthquakes.csv",
    "World Bank": RAW_DIR / "world_bank_gdp_per_capita_api.csv",
    "FRED CPI": RAW_DIR / "CPIAUCNS.csv"
}

for source, file_path in required_files.items():
    if not file_path.exists():
        raise FileNotFoundError(
            f"{source} input file not found: {file_path}"
        )

# Load raw datasets.
noaa = pd.read_csv(required_files["NOAA"])
world_bank = pd.read_csv(required_files["World Bank"])
cpi = pd.read_csv(required_files["FRED CPI"])

# Validate required columns before preprocessing.
required_columns = {
    "NOAA": {
        "id", "year", "country",
        "eqMagnitude", "tsunamiEventId", "deathsTotal",
        "deathsAmountOrderTotal", "damageMillionsDollarsTotal",
        "damageAmountOrderTotal"
    },
    "World Bank": {
        "Country Name", "Country Code", "ISO3",
        "year", "GDP_per_capita_2015USD"
    },
    "FRED CPI": {"observation_date", "CPIAUCNS"}
}

datasets = {
    "NOAA": noaa,
    "World Bank": world_bank,
    "FRED CPI": cpi
}

for source, required in required_columns.items():
    missing = required - set(datasets[source].columns)

    if missing:
        raise ValueError(
            f"{source} dataset is missing columns: {sorted(missing)}"
        )

    if datasets[source].empty:
        raise ValueError(f"{source} dataset is empty.")

# Optional NOAA columns: used for the event date (month, day) and the
# shallow-earthquake check in RQ2 (eqDepth). Missing ones are created empty.
optional_noaa_columns = ["month", "day", "eqDepth"]
for column in optional_noaa_columns:
    if column not in noaa.columns:
        print(f"WARNING: NOAA column '{column}' not found; filled with NaN.")
        noaa[column] = pd.NA

print("All input file and column checks passed.")

print("\nRaw data loaded.")
print("NOAA:", noaa.shape)
print("World Bank:", world_bank.shape)
print("CPI:", cpi.shape)

# --------------------------------------------------
# STEP 2: PREPARE COUNTRY-ISO3 MAPPING
# --------------------------------------------------

country_mapping = {
    'BOSNIA-HERZEGOVINA': 'Bosnia and Herzegovina',
    'CZECH REPUBLIC': 'Czechia',
    'EGYPT': 'Egypt, Arab Rep.',
    'IRAN': 'Iran, Islamic Rep.',
    'KYRGYZSTAN': 'Kyrgyz Republic',
    'LAOS': 'Lao PDR',
    'MICRONESIA, FED. STATES OF': 'Micronesia, Fed. Sts.',
    'MYANMAR (BURMA)': 'Myanmar',
    'RUSSIA': 'Russian Federation',
    'SLOVAKIA': 'Slovak Republic',
    'SOUTH KOREA': 'Korea, Rep.',
    'SYRIA': 'Syrian Arab Republic',
    'TURKEY': 'Turkiye',
    'VENEZUELA': 'Venezuela, RB',
    'VIETNAM': 'Viet Nam',
    'YEMEN': 'Yemen, Rep.',
    'UK': 'United Kingdom',
    'USA': 'United States',
    'AZORES (PORTUGAL)': 'Portugal',
    'KERMADEC ISLANDS (NEW ZEALAND)': 'New Zealand'
}

wb_iso_lookup = {
    name.strip().casefold(): iso
    for name, iso in
    world_bank[['Country Name', 'ISO3']]
    .drop_duplicates()
    .values
}

noaa_to_iso3 = {}

for country in noaa['country'].dropna().unique():
    wb_name = country_mapping.get(country, country)
    noaa_to_iso3[country] = wb_iso_lookup.get(
        wb_name.strip().casefold()
    )

noaa['ISO3'] = noaa['country'].map(noaa_to_iso3)

print("\nCountry mapping completed.")
print("Matched countries:", sum(v is not None for v in noaa_to_iso3.values()))
print("Unmatched countries:", sum(v is None for v in noaa_to_iso3.values()))

# --------------------------------------------------
# STEP 3: PREPARE CPI DATA
# --------------------------------------------------

cpi["observation_date"] = pd.to_datetime(
    cpi["observation_date"],
    errors="coerce"
)

cpi["CPIAUCNS"] = pd.to_numeric(
    cpi["CPIAUCNS"],
    errors="coerce"
)

if cpi["observation_date"].isna().any():
    raise ValueError("Invalid dates found in CPI data.")

if cpi["observation_date"].duplicated().any():
    raise ValueError("Duplicate observation dates found in CPI data.")

if cpi["CPIAUCNS"].isna().any():
    raise ValueError("Missing or invalid CPI values found.")

if (cpi["CPIAUCNS"] <= 0).any():
    raise ValueError("CPI values must be positive.")

cpi["year"] = cpi["observation_date"].dt.year

cpi_filtered = cpi[
    cpi["year"].between(1960, 2024)
].copy()

cpi_annual = (
    cpi_filtered
    .groupby("year", as_index=False)["CPIAUCNS"]
    .mean()
    .rename(columns={"CPIAUCNS": "CPI_annual"})
)

expected_years = set(range(1960, 2025))
available_years = set(cpi_annual["year"].astype(int))

if available_years != expected_years:
    missing_years = sorted(expected_years - available_years)
    raise ValueError(f"Missing CPI years: {missing_years}")

if cpi_annual["CPI_annual"].isna().any():
    raise ValueError("Annual CPI contains missing values.")

if (cpi_annual["CPI_annual"] <= 0).any():
    raise ValueError("Annual CPI values must be positive.")

cpi_2024_values = cpi_annual.loc[
    cpi_annual["year"] == 2024,
    "CPI_annual"
]

if len(cpi_2024_values) != 1:
    raise ValueError("Could not identify a unique 2024 CPI value.")

cpi_2024 = cpi_2024_values.iloc[0]

print("\nAnnual CPI prepared and validated.")
print("Years:", len(cpi_annual))
print("2024 CPI:", cpi_2024)

# STEP 4: ADJUST DAMAGE TO 2024 USD
# --------------------------------------------------

noaa["year"] = pd.to_numeric(
    noaa["year"], errors="coerce"
)

noaa["damageMillionsDollarsTotal"] = pd.to_numeric(
    noaa["damageMillionsDollarsTotal"],
    errors="coerce"
)

if noaa["year"].isna().any():
    raise ValueError("Invalid or missing earthquake years found.")

if not noaa["year"].between(1960, 2024).all():
    raise ValueError("Earthquake years must be between 1960 and 2024.")

if (noaa["year"] % 1 != 0).any():
    raise ValueError("Non-integer earthquake years found.")

if (noaa["damageMillionsDollarsTotal"].dropna() < 0).any():
    raise ValueError("Negative damage values found.")

noaa["year"] = noaa["year"].astype(int)

# STEP 4a: PARSE EVENT DATE
# NOAA gives year, month and day as separate numbers, so there is no
# day-first ambiguity. Joins stay annual; the full date is kept for
# reference. Events with a missing month or day get NaT but keep their year.
date_parts = pd.DataFrame({
    "year": noaa["year"],
    "month": pd.to_numeric(noaa["month"], errors="coerce"),
    "day": pd.to_numeric(noaa["day"], errors="coerce")
})
noaa["date"] = pd.to_datetime(date_parts, errors="coerce")

print("\nEvent dates parsed.")
print("Events with full date:", int(noaa["date"].notna().sum()), "/", len(noaa))

noaa = noaa.merge(
    cpi_annual,
    on='year',
    how='left',
    validate='many_to_one'
)

noaa['Total_Damage_2024_USD_Millions'] = (
    noaa['damageMillionsDollarsTotal']
    * (cpi_2024 / noaa['CPI_annual'])
)

print("\nDamage adjustment completed.")

# --------------------------------------------------
# STEP 5: MERGE WORLD BANK GDP DATA
# --------------------------------------------------

world_bank["year"] = pd.to_numeric(
    world_bank["year"],
    errors="coerce"
)

world_bank["GDP_per_capita_2015USD"] = pd.to_numeric(
    world_bank["GDP_per_capita_2015USD"],
    errors="coerce"
)

if world_bank["year"].isna().any():
    raise ValueError("Invalid years found in World Bank data.")

if world_bank["ISO3"].isna().any():
    raise ValueError("Missing ISO3 codes found in World Bank data.")

if world_bank.duplicated(["ISO3", "year"]).any():
    raise ValueError("Duplicate ISO3-year rows in World Bank data.")

if (world_bank["GDP_per_capita_2015USD"].dropna() < 0).any():
    raise ValueError("Negative GDP per capita values found.")

gdp_data = world_bank[
    ["ISO3", "year", "GDP_per_capita_2015USD"]
].copy()

original_row_count = len(noaa)

noaa = noaa.merge(
    gdp_data,
    on=["ISO3", "year"],
    how="left",
    validate="many_to_one"
)

if len(noaa) != original_row_count:
    raise RuntimeError(
        "GDP merge changed the number of earthquake records."
    )

print("GDP merge validation passed.")
print("Earthquake rows preserved:", len(noaa))


print("\nGDP data merged.")

# STEP 5a: WHY IS GDP MISSING?
# Separate missing GDP caused by our country mapping (fixable: add the name to
# country_mapping) from missing GDP in the World Bank data (a data limitation).
noaa["gdp_missing_reason"] = "GDP available"
noaa.loc[noaa["country"].isna(), "gdp_missing_reason"] = "No country in NOAA"
noaa.loc[
    noaa["country"].notna() & noaa["ISO3"].isna(),
    "gdp_missing_reason"
] = "No ISO3 match"
noaa.loc[
    noaa["ISO3"].notna() & noaa["GDP_per_capita_2015USD"].isna(),
    "gdp_missing_reason"
] = "No World Bank GDP for this year"

print("\nGDP availability by reason:")
print(noaa["gdp_missing_reason"].value_counts().to_string())

# NOAA country names without an ISO3 code, most frequent first.
unmatched = (
    noaa.loc[noaa["gdp_missing_reason"] == "No ISO3 match", "country"]
    .value_counts()
    .rename_axis("noaa_country")
    .reset_index(name="events")
)
unmatched_path = OUTPUT_DIR / "unmatched_countries.csv"
unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
print(f"\nUnmatched NOAA country names: {len(unmatched)} (saved to {unmatched_path.name})")
print(unmatched.head(15).to_string(index=False))

# --------------------------------------------------
# STEP 5b: TSUNAMI INDICATOR
# --------------------------------------------------

# tsunamiEventId links the earthquake to a NOAA tsunami record.
# Linked = 1, not linked = 0. The ID itself is not needed for the analysis.
noaa["tsunami"] = noaa["tsunamiEventId"].notna().astype(int)

print("\nTsunami indicator created.")
print("Tsunami-associated (1):", int(noaa["tsunami"].sum()))
print("Not tsunami-associated (0):", int((noaa["tsunami"] == 0).sum()))

# --------------------------------------------------
# STEP 5c: LOSS STATUS (missing is not zero)
# --------------------------------------------------

# NOAA description codes: 0 = None, 1-4 = increasing severity.
# An empty number AND an empty description means "unknown", never zero.
def classify_status(number, description, label):
    """Return Reported / No Reported / Unknown / Inconsistent for one event."""
    if pd.isna(number) and pd.isna(description):
        return "Unknown"
    has_pos_num = pd.notna(number) and number > 0
    has_zero_num = pd.notna(number) and number == 0
    has_pos_desc = pd.notna(description) and description in [1, 2, 3, 4]
    has_zero_desc = pd.notna(description) and description == 0
    if (has_pos_num and has_zero_desc) or (has_zero_num and has_pos_desc):
        return "Inconsistent"
    if has_pos_num or has_pos_desc:
        return f"{label} Reported"
    if has_zero_num or has_zero_desc:
        return f"No {label} Reported"
    return "Unknown"


for column in ["deathsTotal", "deathsAmountOrderTotal", "damageAmountOrderTotal", "eqDepth"]:
    noaa[column] = pd.to_numeric(noaa[column], errors="coerce")

noaa["Death_Status"] = [
    classify_status(n, d, "Deaths")
    for n, d in zip(noaa["deathsTotal"], noaa["deathsAmountOrderTotal"])
]
noaa["Economic_Damage_Status"] = [
    classify_status(n, d, "Damage")
    for n, d in zip(noaa["damageMillionsDollarsTotal"], noaa["damageAmountOrderTotal"])
]

print("\nLoss status created.")
print("Death_Status:", noaa["Death_Status"].value_counts().to_dict())
print("Economic_Damage_Status:", noaa["Economic_Damage_Status"].value_counts().to_dict())

# Clearer names for the NOAA description codes.
noaa = noaa.rename(columns={
    "deathsAmountOrderTotal": "death_description",
    "damageAmountOrderTotal": "damage_description"
})

# --------------------------------------------------
# Convert the columns to an integer (Int64) type
# --------------------------------------------------
integer_columns = ['deathsTotal', 'death_description', 'damage_description']
for col in integer_columns:
    if col in noaa.columns:
        noaa[col] = pd.to_numeric(noaa[col], errors='coerce').astype('Int64')

# --------------------------------------------------
# STEP 6: SELECT COMMON DATASET COLUMNS
# --------------------------------------------------

selected_columns = [
    'id',
    'year',
    'date',
    'country',
    'ISO3',
    'eqMagnitude',
    'eqDepth',
    'tsunami',
    'deathsTotal',
    'death_description',
    'Death_Status',
    'damageMillionsDollarsTotal',
    'damage_description',
    'Economic_Damage_Status',
    'Total_Damage_2024_USD_Millions',
    'GDP_per_capita_2015USD',
    'gdp_missing_reason',
    'CPI_annual'
]

cleaned_data = noaa[selected_columns].copy()

# --------------------------------------------------
# STEP 7: DATA QUALITY CHECKS
# --------------------------------------------------

print("\nDATA QUALITY CHECKS")

# Check dataset dimensions and required columns.
if cleaned_data.empty:
    raise ValueError("The cleaned dataset is empty.")

if cleaned_data.columns.duplicated().any():
    raise ValueError("Duplicate column names found.")

if cleaned_data.isna().all(axis=1).any():
    raise ValueError("Completely empty rows found.")

# Check earthquake IDs.
if cleaned_data["id"].isna().any():
    raise ValueError("Missing earthquake IDs found.")

if cleaned_data["id"].duplicated().any():
    raise ValueError("Duplicate earthquake IDs found.")

# Check years.
if not cleaned_data["year"].between(1960, 2024).all():
    raise ValueError("Years outside 1960-2024 found.")

# Check the tsunami indicator.
if not cleaned_data["tsunami"].isin([0, 1]).all():
    raise ValueError("Tsunami indicator must be 0 or 1.")

# Check non-negative numeric variables.
non_negative_columns = [
    "eqMagnitude",
    "deathsTotal",
    "damageMillionsDollarsTotal",
    "Total_Damage_2024_USD_Millions",
    "GDP_per_capita_2015USD",
    "CPI_annual"
]

for column in non_negative_columns:
    values = cleaned_data[column].dropna()

    if (values < 0).any():
        raise ValueError(f"Negative values found in {column}.")

# Check that the CPI adjustment has not created values
# where the original damage amount was missing.
original_missing_damage = (
    cleaned_data["damageMillionsDollarsTotal"].isna()
)

adjusted_present = (
    cleaned_data["Total_Damage_2024_USD_Millions"].notna()
)

if (original_missing_damage & adjusted_present).any():
    raise ValueError(
        "Adjusted damage exists although original damage is missing."
    )

# Confirm that the GDP merge preserved all earthquake records.
if len(cleaned_data) != original_row_count:
    raise RuntimeError(
        "The number of earthquake records changed during preprocessing."
    )

print("All data quality checks passed.")
print("Dataset shape:", cleaned_data.shape)
print("Duplicate earthquake IDs:", cleaned_data["id"].duplicated().sum())
print("Year range:", cleaned_data["year"].min(), "-", cleaned_data["year"].max())
print("Missing GDP:", cleaned_data["GDP_per_capita_2015USD"].isna().sum())
print("Missing deaths:", cleaned_data["deathsTotal"].isna().sum())
print("Missing damage:", cleaned_data["damageMillionsDollarsTotal"].isna().sum())
print("Missing date (month/day unknown):", cleaned_data["date"].isna().sum())
print("Missing depth:", cleaned_data["eqDepth"].isna().sum())

# STEP 8: SAVE CLEANED DATASET
# --------------------------------------------------

output_path = OUTPUT_DIR / "cleaned_data.csv"

cleaned_data.to_csv(
    output_path,
    index=False,
    encoding='utf-8-sig'
)

print("\nCleaned dataset saved:")
print(output_path)
print("Final shape:", cleaned_data.shape)
print("=" * 55)
