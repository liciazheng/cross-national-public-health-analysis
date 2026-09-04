"""
Tests for the World Bank fetch and the panel analysis.

No network: every fetch test drives the parsing and assembly code with a
synthetic API payload, so the suite runs the same offline and in CI.

    pip install -r requirements.txt pytest
    pytest
"""

import json

import numpy as np
import pandas as pd
import pytest

import fetch_worldbank as fw
import panel_analysis as pa


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------

def test_cache_key_stays_short_for_a_48_country_url():
    """
    Regression test. The first version of the cache used the request path as the
    filename, which pushed the full path past the Windows 260-character limit
    once 48 ISO codes were in the URL, and fetching died with FileNotFoundError.
    """
    codes = ";".join(f"C{i:02d}" for i in range(48))
    path = f"country/{codes}/indicator/SH.DYN.AIDS.ZS?date=2000:2022&format=json&per_page=20000"

    key = fw._cache_key(path)

    assert len(key) <= 80
    assert not set(key) & set(';:?&/\\*"<>|')


def test_cache_key_is_unique_per_path():
    a = fw._cache_key("country/ZAF/indicator/SH.DYN.AIDS.ZS?date=2000:2022")
    b = fw._cache_key("country/ZAF/indicator/SH.HIV.ARTC.ZS?date=2000:2022")
    assert a != b


def test_cache_key_is_stable():
    path = "country/KEN/indicator/SP.POP.TOTL?date=2000:2022"
    assert fw._cache_key(path) == fw._cache_key(path)


# --------------------------------------------------------------------------
# Indicator parsing
# --------------------------------------------------------------------------

def _payload(rows, total=None):
    """Shape a World Bank style [metadata, records] response."""
    return [{"total": total if total is not None else len(rows), "per_page": 20000}, rows]


def _record(iso3, year, value):
    return {"countryiso3code": iso3, "date": str(year), "value": value}


def test_fetch_indicator_parses_records(monkeypatch):
    rows = [_record("ZAF", 2020, 18.4), _record("KEN", 2020, 3.9)]
    monkeypatch.setattr(fw, "_get", lambda path, refresh=False: _payload(rows))

    frame = fw.fetch_indicator("SH.DYN.AIDS.ZS", ["ZAF", "KEN"])

    assert list(frame.columns) == ["iso3", "year", "hiv_prevalence_pct"]
    assert frame["year"].dtype.kind == "i"
    assert frame.set_index("iso3").loc["ZAF", "hiv_prevalence_pct"] == 18.4


def test_fetch_indicator_drops_rows_without_a_country_code(monkeypatch):
    """Aggregate rows come back with an empty countryiso3code."""
    rows = [_record("ZAF", 2020, 18.4), _record("", 2020, 4.2)]
    monkeypatch.setattr(fw, "_get", lambda path, refresh=False: _payload(rows))

    frame = fw.fetch_indicator("SH.DYN.AIDS.ZS", ["ZAF"])

    assert len(frame) == 1


def test_fetch_indicator_refuses_a_truncated_page(monkeypatch):
    """
    Silently analysing page 1 of 3 would be worse than crashing, so a total
    larger than per_page has to raise.
    """
    rows = [_record("ZAF", 2020, 18.4)]
    monkeypatch.setattr(fw, "_get", lambda path, refresh=False: _payload(rows, total=99999))

    with pytest.raises(RuntimeError, match="exceed one page"):
        fw.fetch_indicator("SH.DYN.AIDS.ZS", ["ZAF"])


def test_fetch_indicator_handles_an_empty_result(monkeypatch):
    monkeypatch.setattr(fw, "_get", lambda path, refresh=False: [{"total": 0, "per_page": 20000}, None])
    assert fw.fetch_indicator("SH.DYN.AIDS.ZS", ["ZAF"]).empty


# --------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------

def test_build_panel_derives_per_100k_measures(monkeypatch):
    countries = pd.DataFrame([
        {"iso3": "ZAF", "country": "South Africa",
         "income_level": "Upper middle income", "sub_region": None},
    ])
    monkeypatch.setattr(fw, "fetch_countries", lambda refresh=False: countries)

    def fake_indicator(code, iso3_codes, refresh=False):
        name = fw.INDICATORS[code]
        value = {"population": 60_000_000, "aids_deaths": 60_000,
                 "new_infections": 120_000}.get(name, 1.0)
        return pd.DataFrame([{"iso3": "ZAF", "year": 2020, name: value}])

    monkeypatch.setattr(fw, "fetch_indicator", fake_indicator)

    panel = fw.build_panel()
    row = panel.iloc[0]

    assert row["aids_deaths_per_100k"] == pytest.approx(100.0)
    assert row["new_infections_per_100k"] == pytest.approx(200.0)
    assert row["country"] == "South Africa"


def test_indicator_names_are_unique():
    """Two indicators mapping to one column name would silently overwrite."""
    names = list(fw.INDICATORS.values())
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# panel_analysis
# --------------------------------------------------------------------------

def _panel(n_countries=6, years=range(2000, 2023), income_effect=0.0, seed=0):
    """
    A synthetic panel with a known between/within structure.

    A country's income is its own level, plus a common trend, plus a per-year
    shock. `income_effect` decides how much of that shock feeds into coverage.
    Coverage never depends on the country's income *level*, so the within
    association is strong while the between association is not — which is the
    thing between_within has to be able to tell apart.

    The shock also keeps log income from being collinear with the year, so the
    fixed-effects specifications are actually identified rather than dropping
    out of a rank-deficient design.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_countries):
        base_income = 6.0 + c * 0.5
        for i, year in enumerate(years):
            shock = rng.normal(0, 0.2)
            log_gdp = base_income + i * 0.03 + shock
            rows.append({
                "country": f"Country {c}",
                "iso3": f"C{c:02d}",
                "income_level": "Low income",
                "year": year,
                "art_coverage_pct": 10 + i * 2 + income_effect * shock * 60
                                    + rng.normal(0, 0.5),
                "gdp_per_capita_usd": float(np.exp(log_gdp)),
                # Must vary within a country: a country-constant regressor is
                # perfectly collinear with the country dummies under country FE.
                "hiv_prevalence_pct": 5.0 + c - i * 0.05 + rng.normal(0, 0.1),
                "aids_deaths_per_100k": 200.0 - i * 5,
            })
    return pd.DataFrame(rows)


def test_load_filters_years_and_derives_logs(tmp_path, monkeypatch):
    panel = _panel()
    monkeypatch.setattr(pa, "DATA", tmp_path)
    panel.to_csv(tmp_path / "worldbank_panel.csv", index=False)

    df, modelled = pa.load()

    assert df["year"].min() == pa.START_YEAR
    assert "log_gdp" in modelled and "log_deaths" in modelled
    expected = np.log(modelled["gdp_per_capita_usd"].iloc[0])
    assert modelled["log_gdp"].iloc[0] == pytest.approx(expected)


def test_load_floors_zero_mortality_before_logging(tmp_path, monkeypatch):
    """log(0) would be -inf and drop the row from every regression."""
    panel = _panel()
    panel.loc[panel.index[0], "aids_deaths_per_100k"] = 0.0
    monkeypatch.setattr(pa, "DATA", tmp_path)
    panel.to_csv(tmp_path / "worldbank_panel.csv", index=False)

    _, modelled = pa.load()

    assert np.isfinite(modelled["log_deaths"]).all()


def test_load_errors_clearly_when_the_panel_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "DATA", tmp_path)
    with pytest.raises(SystemExit, match="fetch_worldbank"):
        pa.load()


def test_between_within_separates_the_two_components(capsys):
    """
    Built so income moves coverage *within* a country but country means are
    ordered against income, the within correlation must come out the stronger
    of the two.
    """
    data = _panel(income_effect=1.0)
    data["log_gdp"] = np.log(data["gdp_per_capita_usd"])

    between, within = pa.between_within(data)

    assert within > between
    assert within > 0.8


def test_regression_table_reports_every_spec():
    data = _panel(income_effect=0.5)
    data["log_gdp"] = np.log(data["gdp_per_capita_usd"])
    data["log_deaths"] = np.log(data["aids_deaths_per_100k"].clip(lower=0.01))

    table = pa.regression_table(pa.ART_SPECS, data, "log_gdp", "test")

    assert list(table["spec"]) == list(pa.ART_SPECS)
    assert (table["n"] == len(data)).all()
    assert (table["ci_low"] <= table["coef"]).all()
    assert (table["coef"] <= table["ci_high"]).all()


# --------------------------------------------------------------------------
# Incidence
# --------------------------------------------------------------------------

def _incidence_panel():
    """
    Two countries built so the weighted and unweighted regional rates diverge
    hard: a big country with a low rate and a small one with a high rate.
    Weighted the region reads ~109 per 100k, unweighted it would read 550.
    """
    return pd.DataFrame([
        {"country": "Big", "year": 2005, "population": 100_000_000,
         "new_infections": 100_000, "new_infections_per_100k": 100.0},
        {"country": "Small", "year": 2005, "population": 1_000_000,
         "new_infections": 10_000, "new_infections_per_100k": 1000.0},
        {"country": "Big", "year": 2022, "population": 100_000_000,
         "new_infections": 50_000, "new_infections_per_100k": 50.0},
        {"country": "Small", "year": 2022, "population": 1_000_000,
         "new_infections": 15_000, "new_infections_per_100k": 1500.0},
    ])


def test_incidence_summary_weights_the_regional_rate_by_population():
    """
    The unweighted mean of country rates would let a country of 100k count as
    much as one of 200m. The regional rate has to be total infections over
    total population.
    """
    regional, _ = pa.incidence_summary(_incidence_panel())

    assert regional.loc[2005, "rate_per_100k"] == pytest.approx(110_000 / 101_000_000 * 1e5)
    assert regional.loc[2005, "rate_per_100k"] < 200  # nowhere near the 550 mean
    assert regional.loc[2005, "median_country_rate"] == pytest.approx(550.0)


def test_incidence_summary_reports_change_over_the_panel_span():
    _, change = pa.incidence_summary(_incidence_panel())

    assert change["Big"] == pytest.approx(-50.0)
    assert change["Small"] == pytest.approx(50.0)
    # Sorted ascending, so the steepest decline is first and risers are last.
    assert list(change.index) == ["Big", "Small"]


def test_incidence_summary_ignores_countries_missing_the_measure():
    panel = _incidence_panel()
    panel.loc[len(panel)] = {"country": "Nodata", "year": 2005, "population": 5_000_000,
                             "new_infections": np.nan, "new_infections_per_100k": np.nan}

    regional, change = pa.incidence_summary(panel)

    assert "Nodata" not in change.index
    assert regional.loc[2005, "countries"] == 2


def test_worse_is_not_one_of_the_series_colours():
    """
    A status colour standing in for a country would make the diverging scale
    unreadable. WORSE has to stay outside the categorical slots.
    """
    assert pa.viz.WORSE not in pa.viz.SERIES
    assert pa.viz.WORSE not in pa.viz.FOCUS.values()


def test_focus_countries_have_a_reserved_colour():
    """Figures across both analyses rely on a country keeping its hue."""
    for country in pa.FOCUS:
        assert country in pa.viz.FOCUS
