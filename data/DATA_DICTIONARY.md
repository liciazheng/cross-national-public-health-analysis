# Data dictionary

Covers the three raw files and the generated analysis dataset. Ranges and null
counts are from the current data, printed by `analysis.py`.

**Coverage:** South Africa, Kenya, Botswana × 2007–2016 = 30 observations.
**Grain:** one row per country-year. Key: (`Country`, `Year`).
**Source:** [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators).

---

## `analysis_dataset.csv` (generated)

Written by `analysis.py`. Comma-delimited, period decimal separator, UTF-8 —
unlike the raw files. **Do not edit by hand**: it is overwritten on every run.

### Keys

| Column | Type | Nulls | Range | Notes |
|---|---|---|---|---|
| `Country` | text | 0 | 3 values | `South Africa`, `Kenya`, `Botswana` |
| `Year` | integer | 0 | 2007–2016 | Calendar year |

### HIV indicators

| Column | Unit | Type | Nulls | Range | Notes |
|---|---|---|---|---|---|
| `prevalence_total` | % of adults | float | **10** | 4.5–23.6 | Adult HIV prevalence. **South Africa is null by design** — its source values (4.1–4.7%) are far below published national prevalence for the period (~18%), so the indicator behind the column is not the one the header implies. |
| `art_coverage` | % | integer | 0 | 10–76 | Share of people living with HIV receiving antiretroviral therapy. The most reliable series in the dataset. |
| `pmtct_coverage` | % | integer | 0 | 49–100 | Share of HIV-positive pregnant women receiving prophylaxis to prevent mother-to-child transmission. Hits 100 for South Africa 2011–2014. |
| `mtct_rate` | % | float | **9** | 4–20 | Share of exposed infants who acquired HIV. Not reported for 2007–2009 in any country, hence 9 nulls. Header in the raw file is misspelled (`trasmission`). |
| `people_living_with_hiv` | persons | float | **20** | 5.1m–6.9m | **South Africa only.** Kenya and Botswana are null by design: the raw column repeats South Africa's 2016 value (6,900,000) in every year for both, which also exceeds Botswana's entire population. |
| `aids_deaths` | deaths | integer | 0 | 2,800–370,000 | Annual AIDS-related deaths. Absolute counts, so not comparable across countries without normalising — see `aids_deaths_per_100k`. |
| `women_15plus_with_hiv` | persons | integer | 0 | 170,000–4,300,000 | Women aged 15 and over living with HIV. Absolute counts. |

### Economic indicators

| Column | Unit | Type | Nulls | Range | Notes |
|---|---|---|---|---|---|
| `population` | persons | integer | 0 | 1.9m–57.3m | Total population. Botswana is ~25× smaller than South Africa, which is why the derived per-capita columns exist. |
| `gdp_usd` | current USD | float | 0 | $10.1bn–$458bn | **Total national GDP.** Renamed from the raw header `GDP_per_capita`, which was wrong — the values are national aggregates. |
| `unemployment_rate` | % of labour force | float | 0 | 2.7–26.7 | ILO-modelled estimate. Kenya's series is nearly flat (2.7–2.8) across the decade. |

### Derived

Computed in `clean()`. Rounded to 2 decimal places so the written CSV stays
readable; recompute from the base columns if you need full precision.

| Column | Unit | Type | Formula | Range |
|---|---|---|---|---|
| `gdp_per_capita_usd` | current USD | float | `gdp_usd / population` | $840–$8,646 |
| `aids_deaths_per_100k` | deaths per 100k | float | `aids_deaths / population × 100000` | 68.6–732.3 |
| `women_15plus_with_hiv_per_100k` | persons per 100k | float | `women_15plus_with_hiv / population × 100000` | 1,661.8–9,396.9 |

### Dropped before this file is written

| Raw column | Why |
|---|---|
| `HIV_Prevalence_Female` | For South Africa it is a verbatim copy of `PMTCT_Coverage` (49, 63, 79, 93, 100 …). For Kenya and Botswana it sits *below* `prevalence_total` rather than bracketing it, so it is measuring something else — plausibly youth (15–24) prevalence against an adult (15–49) total, but that is inference, not documentation. |
| `HIV_Prevalence_Male` | Same coherence failure, in all three countries (30/30 rows). |
| `HIV per 1000 pop` | Present only in `joined_dataset.csv`. Derived from the broken `people_living_with_hiv` column, giving Botswana 2,459–2,953 per 1,000 — i.e. more than everyone. `aids_deaths_per_100k` replaces it, computed from columns that hold up. |

---

## Raw files

**Format:** semicolon-delimited, **comma decimal separator**, UTF-8 (European
Excel export). Large numbers appear in scientific notation with a comma
mantissa (`3,33077E+11`).

```python
pd.read_csv("data/hiv_indicators.csv", sep=";", decimal=",")
```

### `hiv_indicators.csv`

`Country`, `Year`, `HIV_Prevalence_Female`, `HIV_Prevalence_Male`,
`HIV_Prevalence_Total`, `Antiretroviral_Coverage`, `PMTCT_Coverage`,
`People living with HIV`, `AIDS Deaths`, `Women 15+ with HIV`,
`Mother to child trasmission rate %`

### `economic_indicators.csv`

`Country`, `Year`, `Population`, `GDP_per_capita` (actually total GDP),
`Unemployee` (unemployment rate)

### `joined_dataset.csv`

The original join, done by hand in Excel. Kept for comparison, **not used by
the pipeline**: it disagrees with `hiv_indicators.csv` on
`people_living_with_hiv` in all 30 rows, and adds the broken
`HIV per 1000 pop` column. `analysis.py` re-does the join in code and reports
the discrepancy on every run.

### `hiv_table1.xlsx`, `hiv_table2.xlsx`, `hiv_table_complete.xlsx`

Excel working files from the original pass, kept as provenance. Not read by
the pipeline.

---

## Column name mapping

| Raw header | Name in `analysis_dataset.csv` |
|---|---|
| `HIV_Prevalence_Total` | `prevalence_total` |
| `Antiretroviral_Coverage` | `art_coverage` |
| `PMTCT_Coverage` | `pmtct_coverage` |
| `People living with HIV` | `people_living_with_hiv` |
| `AIDS Deaths` | `aids_deaths` |
| `Women 15+ with HIV` | `women_15plus_with_hiv` |
| `Mother to child trasmission rate %` | `mtct_rate` |
| `Population` | `population` |
| `GDP_per_capita` | `gdp_usd` *(renamed — it is not per capita)* |
| `Unemployee` | `unemployment_rate` |

---

## `worldbank_panel.csv` (generated)

Written by `fetch_worldbank.py` straight from the [World Bank API](https://api.worldbank.org/v2/).
Comma-delimited, period decimal separator, UTF-8. **Do not edit by hand** — it is
overwritten on every run. Raw API responses are cached under
`data/worldbank_cache/` (not committed).

**Coverage:** all 48 Sub-Saharan African economies × 2000–2022 = **1,104 rows**.
**Grain:** one row per country-year. Key: (`iso3`, `year`).

This file replaces the hand-collected three-country data for anything
quantitative. Where the two disagree, this one is right: it comes from the
source, and the fetch asserts the API returned a complete single page rather
than silently analysing page 1 of N.

### Keys and attributes

| Column | Type | Nulls | Range / values | Notes |
|---|---|---|---|---|
| `iso3` | text | 0 | 48 distinct | ISO 3166-1 alpha-3 |
| `country` | text | 0 | 48 distinct | World Bank short name |
| `income_level` | text | 0 | 4 values | Low (21), Lower middle (20), Upper middle (6), High (1 — Seychelles) |
| `year` | integer | 0 | 2000–2022 | Calendar year |

### HIV indicators

| Column | Unit | Nulls | Range | Source indicator | Notes |
|---|---|---|---|---|---|
| `hiv_prevalence_pct` | % of ages 15–49 | 92 | 0.1–29.4 | `SH.DYN.AIDS.ZS` | Adult prevalence. This is the series the hand-collected file got wrong for South Africa: the API gives 17.9–18.4% for 2020–2022 against 4.1–4.7% in the spreadsheet. |
| `art_coverage_pct` | % of people with HIV | 92 | 0–93 | `SH.HIV.ARTC.ZS` | Antiretroviral therapy coverage. Most complete and most reliable series here. |
| `pmtct_coverage_pct` | % of pregnant women with HIV | 161 | 0–100 | `SH.HIV.PMTC.ZS` | Prophylaxis to prevent mother-to-child transmission. Sparsest of the HIV series. |
| `aids_deaths` | people | 92 | 100–280,000 | `SH.DYN.AIDS.DH` | UNAIDS estimate. Rounded at source, which is why the minimum is a round 100. |
| `new_infections` | people | 115 | 100–530,000 | `SH.HIV.INCD.TL` | All ages. Also rounded at source. |
| `women_share_of_plhiv_pct` | % | 23 | 30.5–72.7 | `SH.DYN.AIDS.FE.ZS` | Women's share of those aged 15+ living with HIV. Not a prevalence rate — a share of cases. |

### Economic and demographic

| Column | Unit | Nulls | Range | Source indicator | Notes |
|---|---|---|---|---|---|
| `population` | people | 0 | 81,130–223,200,000 | `SP.POP.TOTL` | Complete for every country-year. |
| `gdp_per_capita_usd` | current US$ | 26 | 110–19,140 | `NY.GDP.PCAP.CD` | **Genuinely per capita**, unlike the `GDP_per_capita` column in the hand-collected file, which held total GDP. |
| `unemployment_pct` | % of labour force | 23 | 0.3–36.5 | `SL.UEM.TOTL.ZS` | Modelled ILO estimate, not a national survey figure. Compare across countries with care. |

### Derived

| Column | Unit | Nulls | Range | Formula | Notes |
|---|---|---|---|---|---|
| `aids_deaths_per_100k` | deaths per 100k | 92 | 2.7–1,261 | `aids_deaths / population × 100000` | The normalisation the original analysis lacked. Without it, South Africa's 25x larger population makes every absolute comparison meaningless. |
| `new_infections_per_100k` | infections per 100k | 115 | 2.8–2,016 | `new_infections / population × 100000` | Same reasoning. |

### Analysis subset

`panel_analysis.py` restricts to `year >= 2005` and to rows with all four of
`art_coverage_pct`, `gdp_per_capita_usd`, `hiv_prevalence_pct` and
`aids_deaths_per_100k`. That leaves **771 country-years across 44 countries** —
the sample every regression in the README is fitted on. Years before 2005 are
dropped because ART coverage is at or near zero for almost every country, so
they carry no signal about treatment scale-up.

`log_deaths` is computed as `log(clip(aids_deaths_per_100k, lower=0.01))`. The
floor matters: a few country-years round to zero deaths per 100k, and `log(0)`
would drop them from every model without warning.
