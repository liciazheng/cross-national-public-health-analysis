# Cross-National Public Health Data Analysis — HIV in Sub-Saharan Africa

A comparative analysis of HIV prevalence, treatment coverage, and prevention-programme effectiveness in **South Africa, Kenya, and Botswana** from **2007 to 2016**, joined against World Bank economic indicators.

## Questions

1. How did HIV prevalence change in these three countries between 2007 and 2016?
2. How far did antiretroviral therapy (ART) coverage expand?
3. How successful were Prevention of Mother-to-Child Transmission (PMTCT) programmes?
4. Is there a relationship between economic development and HIV outcomes?

Sub-Saharan Africa accounts for roughly 70% of global HIV cases. These three countries span a wide range of both epidemic scale and national income, which makes them a useful comparison set: Botswana is small and relatively wealthy with a severe epidemic, Kenya is large and low-income with a milder one, South Africa is large with the biggest absolute burden in the world.

## Data

Two source tables joined on `Country` + `Year` — 10 years × 3 countries = 30 observations.

| File | Contents |
|---|---|
| [`data/hiv_indicators.csv`](data/hiv_indicators.csv) | Prevalence, ART coverage, PMTCT coverage, people living with HIV, AIDS deaths, women 15+ with HIV, MTCT rate |
| [`data/economic_indicators.csv`](data/economic_indicators.csv) | Population, GDP, unemployment rate |
| [`data/joined_dataset.csv`](data/joined_dataset.csv) | The two above joined, plus a derived HIV-per-1000-population column |
| `data/hiv_table*.xlsx` | Excel working files behind the charts |

**Source:** [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators).

**Format note:** the CSVs are semicolon-delimited with comma decimal separators (European Excel export). In pandas:

```python
pd.read_csv("data/joined_dataset.csv", sep=";", decimal=",")
```

## Findings

**Treatment coverage expanded dramatically in all three countries.** ART coverage rose from 10% to 60% in South Africa, 13% to 74% in Kenya, and 29% to 76% in Botswana over the decade — the clearest signal in the dataset.

![ART coverage trends](figures/art-coverage-trends.png)

**PMTCT programmes converged at a high level.** By 2016 all three countries reached 87–93% coverage for preventing mother-to-child transmission, despite very different starting points (South Africa 49%, Kenya 57%, Botswana 78% in 2007). Reported MTCT rates fell to 4–9% accordingly.

**AIDS deaths fell sharply everywhere.** South Africa 370,000 → 130,000; Kenya 64,000 → 33,000; Botswana 11,000 → 2,800.

**Wealth alone does not explain outcomes.** Botswana is by far the richest of the three per capita, yet carries the highest adult prevalence (21.4% in 2016) — the legacy of an epidemic that peaked before treatment scaled up. It also achieved the steepest proportional drop in AIDS deaths, a 75% reduction. Across these three countries, programme commitment and treatment access track outcomes far more closely than national income does.

## Figures

| | |
|---|---|
| [HIV prevalence trends](figures/hiv-prevalence-trends.png) | [ART coverage trends](figures/art-coverage-trends.png) |
| [PMTCT coverage trends](figures/pmtct-coverage-trends.png) | [Mother-to-child transmission rate](figures/mtct-rate.png) |
| [AIDS deaths](figures/aids-deaths.png) | [People living with HIV](figures/people-living-with-hiv.png) |
| [Women 15+ living with HIV](figures/women-15plus-with-hiv.png) | [Population growth](figures/population-growth.png) |
| [Unemployment rate trends](figures/unemployment-rate-trends.png) | |

## Known data limitations

The dataset is published as collected. Several columns do not survive scrutiny and should not be reused without cleaning:

- **`GDP_per_capita` is total GDP, not per capita.** South Africa's 2007 value of `3.33E+11` is $333 billion in national GDP. Divide by the `Population` column to get per capita.
- **`People living with HIV` is wrong for Botswana** — the column repeats South Africa's values (4.7–6.6 million) against a national population of ~2.2 million. The derived `HIV per 1000 pop` column for Botswana (2,459–2,953) is therefore meaningless.
- **`HIV_Prevalence_Female` is wrong for South Africa** — the column duplicates `PMTCT_Coverage` (49, 63, 79, 93, 100 …) rather than a prevalence percentage.
- **`HIV_Prevalence_Total` for South Africa (4.1–4.7%)** sits well below published national adult prevalence for the period (~18%); the indicator behind that column is not the one the header implies.
- **`Mother to child trasmission rate %` is empty for 2007–2009** across all three countries, and the header carries a typo.

The ART coverage, PMTCT coverage, AIDS deaths, population, and unemployment series are internally consistent, and every finding above rests on those.

## Layout

```
data/       source CSVs and Excel working files
figures/    exported charts (PNG)
```
