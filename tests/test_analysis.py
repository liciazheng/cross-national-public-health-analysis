"""
Tests for the analysis pipeline.

The important ones are the validator tests. Data-quality checks that are
never tested are checks you only find out about when they fail to fire, so
each one is exercised against synthetic data engineered to trip it, and
against clean data to confirm it stays quiet.

    pip install -r requirements.txt pytest
    pytest
"""

import numpy as np
import pandas as pd
import pytest

import analysis
from analysis import (
    COUNTRIES,
    YEARS,
    clean,
    cross_check_manual_join,
    join,
    load_raw,
    validate,
)

YEAR_RANGE = range(YEARS[0], YEARS[1] + 1)


def clean_frame():
    """A joined frame with no data-quality problems, for negative controls."""
    rows = []
    for c, country in enumerate(COUNTRIES):
        for i, year in enumerate(YEAR_RANGE):
            rows.append({
                "Country": country,
                "Year": year,
                "prevalence_female": 8.0 + i * 0.1 + c,
                "prevalence_male": 4.0 + i * 0.1 + c,
                "prevalence_total": 6.0 + i * 0.1 + c,
                "art_coverage": 10 + i * 5,
                "pmtct_coverage": 50 + i * 4,
                "people_living_with_hiv": 100_000 + i * 1_000 + c * 10,
                "aids_deaths": 5_000 - i * 100 + c,
                "women_15plus_with_hiv": 50_000 + i * 500 + c,
                "mtct_rate": 20 - i,
                # Already per capita, so the "this is total GDP" check stays quiet.
                "gdp_usd": 5_000 + i * 100 + c,
                "population": 10_000_000 + i * 100_000 + c * 1_000,
                "unemployment_rate": 20.0 + i * 0.1 + c,
            })
    return pd.DataFrame(rows)


def messages(findings, severity=None):
    return [m for s, m in findings if severity is None or s == severity]


# --- negative control -----------------------------------------------------

def test_clean_data_produces_no_errors():
    findings = validate(clean_frame())
    assert messages(findings, "ERROR") == []


# --- each check fires on data built to trip it ---------------------------

def test_detects_a_filled_down_constant_series():
    df = clean_frame()
    df.loc[df["Country"] == "Kenya", "people_living_with_hiv"] = 6_900_000
    found = messages(validate(df), "ERROR")
    assert any("constant" in m and "Kenya" in m for m in found)


def test_detects_a_column_copied_over_another():
    df = clean_frame()
    mask = df["Country"] == "South Africa"
    df.loc[mask, "prevalence_female"] = df.loc[mask, "pmtct_coverage"].to_numpy()
    found = messages(validate(df), "ERROR")
    assert any("identical to" in m and "South Africa" in m for m in found)


def test_detects_more_cases_than_people():
    df = clean_frame()
    df.loc[df["Country"] == "Botswana", "population"] = 2_200_000
    df.loc[df["Country"] == "Botswana", "people_living_with_hiv"] = 6_900_000
    found = messages(validate(df), "ERROR")
    assert sum("exceeds population" in m for m in found) == len(list(YEAR_RANGE))


def test_detects_total_gdp_masquerading_as_per_capita():
    df = clean_frame()
    df["gdp_usd"] = df["gdp_usd"] * df["population"]
    found = messages(validate(df), "ERROR")
    assert any("holds total GDP" in m for m in found)


def test_detects_incoherent_sex_disaggregated_prevalence():
    df = clean_frame()
    # Total above both the female and male figures, as in the real extract.
    df["prevalence_total"] = df["prevalence_female"] + df["prevalence_male"] + 5
    found = messages(validate(df), "WARN")
    assert sum("outside the female/male range" in m for m in found) == len(COUNTRIES)


def test_detects_coverage_outside_zero_to_one_hundred():
    df = clean_frame()
    df.loc[0, "art_coverage"] = 140
    found = messages(validate(df), "ERROR")
    assert any("art_coverage = 140" in m for m in found)


def test_reports_missing_values():
    df = clean_frame()
    df.loc[df["Year"] == 2007, "mtct_rate"] = np.nan
    found = messages(validate(df), "WARN")
    assert any("mtct_rate" in m and "missing" in m for m in found)


# --- against the real data ------------------------------------------------

@pytest.fixture(scope="module")
def joined():
    return join(*load_raw())


def test_join_is_total_and_correctly_sized(joined):
    assert len(joined) == len(COUNTRIES) * len(list(YEAR_RANGE))
    assert set(joined["Country"]) == set(COUNTRIES)
    for country in COUNTRIES:
        years = sorted(joined[joined["Country"] == country]["Year"])
        assert years == list(YEAR_RANGE)


def test_join_raises_when_keys_do_not_match():
    hiv, economic = load_raw()
    truncated = economic[economic["Country"] != "Kenya"]
    with pytest.raises(ValueError, match="unmatched join keys"):
        join(hiv, truncated)


def test_gdp_column_was_renamed_away_from_its_misleading_header():
    _, economic = load_raw()
    assert "gdp_usd" in economic.columns
    assert "GDP_per_capita" not in economic.columns


def test_the_known_real_errors_are_all_caught(joined):
    found = messages(validate(joined), "ERROR")
    assert any("Kenya" in m and "constant" in m for m in found)
    assert any("South Africa" in m and "identical to" in m for m in found)
    assert any("Botswana" in m and "exceeds population" in m for m in found)
    assert any("holds total GDP" in m for m in found)


def test_manual_join_disagreement_is_reported(joined):
    found = messages(cross_check_manual_join(joined), "ERROR")
    assert any("joined_dataset.csv" in m for m in found)


# --- cleaning -------------------------------------------------------------

@pytest.fixture(scope="module")
def cleaned(joined):
    return clean(joined)


def test_unreliable_cells_are_nulled_not_guessed(cleaned):
    for country in ("Kenya", "Botswana"):
        rows = cleaned[cleaned["Country"] == country]
        assert rows["people_living_with_hiv"].isna().all()

    sa = cleaned[cleaned["Country"] == "South Africa"]
    assert sa["prevalence_total"].isna().all()
    # South Africa's own series is real and must survive.
    assert sa["people_living_with_hiv"].notna().all()


def test_incoherent_columns_are_dropped_entirely(cleaned):
    assert "prevalence_female" not in cleaned.columns
    assert "prevalence_male" not in cleaned.columns


def test_surviving_prevalence_is_only_kenya_and_botswana(cleaned):
    have = cleaned.dropna(subset=["prevalence_total"])
    assert set(have["Country"]) == {"Kenya", "Botswana"}


def test_per_capita_measures_are_derived_correctly(cleaned):
    row = cleaned[(cleaned["Country"] == "South Africa") & (cleaned["Year"] == 2007)].iloc[0]
    # clean() rounds the derived columns to 2dp so the written CSV stays
    # readable, so compare within half of the last retained digit.
    assert row["gdp_per_capita_usd"] == pytest.approx(
        row["gdp_usd"] / row["population"], abs=0.005)
    assert row["aids_deaths_per_100k"] == pytest.approx(
        row["aids_deaths"] / row["population"] * 100_000, abs=0.005)


def test_cleaning_does_not_invent_or_drop_rows(joined, cleaned):
    assert len(cleaned) == len(joined)


def test_cleaned_data_no_longer_trips_the_impossible_value_check(cleaned):
    impossible = cleaned[cleaned["people_living_with_hiv"] > cleaned["population"]]
    assert impossible.empty


# --- findings the README quotes ------------------------------------------

def test_art_coverage_endpoints_match_the_readme(cleaned):
    expected = {"South Africa": (10, 60), "Kenya": (13, 74), "Botswana": (29, 76)}
    for country, (first, last) in expected.items():
        rows = cleaned[cleaned["Country"] == country].set_index("Year")["art_coverage"]
        assert rows[YEARS[0]] == first
        assert rows[YEARS[1]] == last


def test_income_is_uncorrelated_with_treatment_reach(cleaned):
    """The README's headline: r = +0.12, i.e. essentially nothing."""
    r = cleaned[["gdp_per_capita_usd", "art_coverage"]].corr().iloc[0, 1]
    assert r == pytest.approx(0.12, abs=0.01)
    assert abs(r) < 0.2


def test_treatment_tracks_mortality_within_every_country(cleaned):
    for country in COUNTRIES:
        rows = cleaned[cleaned["Country"] == country]
        r = rows[["art_coverage", "aids_deaths_per_100k"]].corr().iloc[0, 1]
        assert r < -0.95, f"{country} r = {r}"


# --- figures --------------------------------------------------------------

def test_every_figure_is_written(tmp_path, monkeypatch, cleaned):
    """Render into a temp directory so the test never touches figures/."""
    monkeypatch.setattr(analysis, "FIGURES", tmp_path)
    written = analysis.plot_all(cleaned)
    assert len(written) == 8
    for name in written:
        assert (tmp_path / name).stat().st_size > 5_000
