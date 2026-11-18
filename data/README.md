# Data card

This project uses three datasets: NOAA Significant Earthquakes, World Bank GDP per capita, and FRED Consumer Price Index (CPI). The datasets are combined and processed using `scripts/preprocess.py` to generate the final cleaned dataset, `data/cleaned_data.csv`

## Dataset 1: NOAA Global Significant Earthquake Database (`noaa_significant_earthquakes.csv`)

- **Source and URL:** NOAA Significant Earthquakes Database, https://www.ngdc.noaa.gov/hazel/view/hazards/earthquake/search
- **License:** Publicly available data provided by the U.S. National Oceanic and Atmospheric Administration (NOAA).
- **Date accessed:** 18 November 2026
- **Time range and granularity:** 1960–2024, one row per significant earthquake event.
- **Size:** 2,666 rows × 48 columns (raw); 2,635 rows retained after preprocessing.
- **Columns used:** `id` (event identifier), `year`, `month`, `day`, `country`, `eqMagnitude` (earthquake magnitude), `tsunamiEventId`, `deathsTotal`, `deathsAmountOrderTotal`, `damageMillionsDollarsTotal`, `damageAmountOrderTotal`.
- **Known issues:** 31 records have missing earthquake magnitude values and are removed. Some country or geographic labels cannot be matched to World Bank country codes. Death and economic damage information is frequently missing. Missing values do not necessarily indicate that no deaths or damage occurred.
- **Cleaning decisions and why:** Rows with missing earthquake magnitude are removed because magnitude is essential to the analysis. Country names are standardized and mapped to ISO3 codes to enable merging with World Bank data. Year, month, and day fields are converted to numeric values, and event dates are constructed. Records are restricted to 1960–2024. Negative damage values are checked and excluded. Missing death and damage values are retained rather than replaced with zero.
- **How to regenerate:** `python scripts/download_data.py && python scripts/preprocess.py`

## Dataset 2: World Bank GDP per capita, constant 2015 US$ (`world_bank_gdp_per_capita_api.csv`)

- **Source and URL:** World Bank Open Data, GDP per capita (constant 2015 US$), https://data.worldbank.org/indicator/NY.GDP.PCAP.KD
- **License:** World Bank data is provided under the World Bank's open data terms. See the source website for licensing details.
- **Date accessed:** 18 November 2026
- **Time range and granularity:** Annual data, 1960–2024; one row per country-year observation.
- **Size:** 14,105 rows × 5 columns (raw); 217 ISO3 country codes.
- **Columns used:** Country identifiers, `year`, and GDP per capita values in constant 2015 US dollars.
- **Known issues:** Some countries do not have GDP observations for every year. Thirteen NOAA country or geographic labels could not be matched to ISO3 codes, affecting 93 earthquake records. Additionally, 164 records had no corresponding World Bank GDP observation for their country and year.
- **Cleaning decisions and why:** Year and GDP values are converted to numeric types. Records with missing ISO3 codes or years are checked, and duplicate country-year combinations are checked to ensure a valid many-to-one merge. Negative GDP values are checked. The dataset is merged with earthquake records using ISO3 and year, with a left join to preserve all earthquake records. A `gdp_missing_reason` column distinguishes missing GDP observations from unmatched country codes.
- **How to regenerate:** `python scripts/download_data.py && python scripts/preprocess.py`

## Dataset 3: FRED – Consumer Price Index (CPI) (`CPIAUCNS.csv`)

- **Source and URL:** Federal Reserve Bank of St. Louis, FRED, Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCNS), https://fred.stlouisfed.org/series/CPIAUCNS
- **License:** FRED data is provided by the Federal Reserve Bank of St. Louis. Refer to the source for information about data usage and attribution.
- **Date accessed:** 18 November 2026
- **Time range and granularity:** January 1960–December 2024, monthly observations in the raw dataset.
- **Size:** 780 rows × 2 columns (raw); 65 annual observations after preprocessing.
- **Columns used:** `observation_date` (date) and `CPIAUCNS` (CPI index).
- **Known issues:** The original dataset contains monthly observations, whereas the earthquake dataset is organized by year. Therefore, the CPI data must be converted to annual values before merging.
- **Cleaning decisions and why:** Dates are parsed using `pandas.to_datetime`, and CPI values are converted to numeric. Missing values, duplicate dates, and non-positive CPI values are checked. The data is restricted to 1960–2024 and monthly observations are aggregated into annual mean CPI values using `groupby`. This produces 65 annual observations. The 2024 annual mean CPI value (313.6888) is used as the reference for converting reported earthquake damage into 2024 US dollars.
- **How to regenerate:** `python scripts/download_data.py && python scripts/preprocess.py`

## Combined dataset: Cleaned Earthquake Dataset (`cleaned_data.csv`)

- **Source and URL:** Combined dataset created by joining NOAA Significant Earthquakes, World Bank GDP per capita, and FRED Consumer Price Index data.
- **Time range and granularity:** 1960–2024, one row per earthquake event.
- **Size:** 2,635 rows × 17 columns.
- **Columns used:** `id`, `year`, `date`, `country`, `ISO3`, `eqMagnitude`, `tsunami`, `deathsTotal`, `death_description`, `Death_Status`, `damageMillionsDollarsTotal`, `damage_description`, `Economic_Damage_Status`, `Total_Damage_2024_USD_Millions`, `GDP_per_capita_2015USD`, `gdp_missing_reason`, `CPI_annual`.
- **Derived columns:** `date` is constructed from the year, month, and day fields. `tsunami` indicates whether an earthquake is associated with a tsunami event. `Death_Status` and `Economic_Damage_Status` classify the reporting status of deaths and economic damage. `Total_Damage_2024_USD_Millions` adjusts reported damage values to 2024 US dollars using annual CPI values.
- **Known issues:** GDP per capita is available for 2,378 earthquake records. It is unavailable for 164 records because no corresponding World Bank observation exists and for 93 records because the country could not be matched to an ISO3 code. Death information is missing for 1,489 records, while economic damage information is missing for 2,074 records. These missing values are not interpreted as zero.
- **Cleaning decisions and why:** The three datasets are merged using year and ISO3 country codes. CPI is first aggregated to annual means and joined to earthquake records using year. World Bank GDP data is then merged using ISO3 and year. Left joins are used to preserve the earthquake records. Event IDs are checked for missing values and uniqueness. Additional checks validate year ranges, non-negative values, tsunami indicators, missing-value handling, and merge row counts.
- **Output:** `data/cleaned_data.csv`
- **How to regenerate:** `python scripts/download_data.py && python scripts/preprocess.py`
