"""
The same questions, asked against all of Sub-Saharan Africa.

analysis.py works the three-country dataset that was collected by hand, and
documents what auditing it turned up. This script asks the same questions of
the full panel pulled from the World Bank API by fetch_worldbank.py — 44
countries with usable data, 2005-2022 — where there are enough observations to
actually separate the between-country and within-country stories.

    python fetch_worldbank.py
    python panel_analysis.py

Writes figures/panel-*.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import viz

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

# ART barely existed before the mid-2000s, so the early years carry no signal.
START_YEAR = 2005
FOCUS = ["South Africa", "Kenya", "Botswana"]
INCOME_ORDER = ["Low income", "Lower middle income", "Upper middle income"]

# The claim from Part 1 being re-tested: three countries inside 87-93% by 2016,
# read at the time as evidence that programmes converge.
PART1_YEAR = 2016
PART1_BAND = (87, 93)

MODELLED = ["art_coverage_pct", "gdp_per_capita_usd", "hiv_prevalence_pct",
            "aids_deaths_per_100k"]

# The incidence models drop a few more country-years, so they get their own
# complete-case set rather than shrinking every other regression to match.
INCIDENCE_MODELLED = MODELLED + ["new_infections_per_100k"]


def load():
    path = DATA / "worldbank_panel.csv"
    if not path.exists():
        raise SystemExit("data/worldbank_panel.csv is missing — run fetch_worldbank.py first")

    df = pd.read_csv(path)
    df = df[df["year"] >= START_YEAR].copy()

    modelled = df.dropna(subset=MODELLED).copy()
    modelled["log_gdp"] = np.log(modelled["gdp_per_capita_usd"])
    # Floor before logging: a handful of country-years round to zero deaths per
    # 100k, and log(0) would silently drop them.
    modelled["log_deaths"] = np.log(modelled["aids_deaths_per_100k"].clip(lower=0.01))

    incidence = df.dropna(subset=INCIDENCE_MODELLED).copy()
    incidence["log_gdp"] = np.log(incidence["gdp_per_capita_usd"])
    incidence["log_deaths"] = np.log(incidence["aids_deaths_per_100k"].clip(lower=0.01))
    incidence["log_incidence"] = np.log(incidence["new_infections_per_100k"].clip(lower=0.01))
    # Share of the whole population living with HIV and *not* on treatment.
    # Transmission comes from unsuppressed virus, so this is the quantity the
    # epidemiology says should drive new infections — not coverage on its own.
    incidence["untreated_pct"] = (
        incidence["hiv_prevalence_pct"] * (1 - incidence["art_coverage_pct"] / 100)
    )
    incidence["log_untreated"] = np.log(incidence["untreated_pct"].clip(lower=0.001))

    return df, modelled, incidence


# --------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------

def _fit(formula, data):
    """OLS with standard errors clustered by country."""
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["country"]}
    )


ART_SPECS = {
    "naive": "art_coverage_pct ~ log_gdp",
    "+ prevalence": "art_coverage_pct ~ log_gdp + hiv_prevalence_pct",
    "+ year FE": "art_coverage_pct ~ log_gdp + hiv_prevalence_pct + C(year)",
    "+ country FE": "art_coverage_pct ~ log_gdp + hiv_prevalence_pct + C(year) + C(country)",
}

DEATH_SPECS = {
    "ART only": "log_deaths ~ art_coverage_pct",
    "+ prevalence + income": "log_deaths ~ art_coverage_pct + hiv_prevalence_pct + log_gdp",
    "+ year & country FE": ("log_deaths ~ art_coverage_pct + hiv_prevalence_pct "
                            "+ log_gdp + C(year) + C(country)"),
}

# Same ladder, same regressors, outcome swapped from deaths to new infections.
# Keeping the specifications identical is what makes the two coefficients
# comparable: they are log points of outcome per percentage point of coverage.
INCIDENCE_SPECS = {
    "ART only": "log_incidence ~ art_coverage_pct",
    "+ prevalence + income": "log_incidence ~ art_coverage_pct + hiv_prevalence_pct + log_gdp",
    "+ year & country FE": ("log_incidence ~ art_coverage_pct + hiv_prevalence_pct "
                            "+ log_gdp + C(year) + C(country)"),
}

# The mechanism restated: instead of coverage, the size of the untreated
# reservoir. An elasticity near 1 would mean incidence scales proportionally
# with the number of people carrying unsuppressed virus.
UNTREATED_SPECS = {
    "untreated only": "log_incidence ~ log_untreated",
    "+ income": "log_incidence ~ log_untreated + log_gdp",
    "+ year & country FE": "log_incidence ~ log_untreated + log_gdp + C(year) + C(country)",
}


def regression_table(specs, data, term, label):
    """Fit each spec and collect the coefficient on `term`."""
    rows = []
    for name, formula in specs.items():
        model = _fit(formula, data)
        low, high = model.conf_int().loc[term]
        rows.append({
            "spec": name,
            "coef": model.params[term],
            "se": model.bse[term],
            "p": model.pvalues[term],
            "ci_low": low,
            "ci_high": high,
            "n": int(model.nobs),
            "r2": model.rsquared,
        })
    table = pd.DataFrame(rows)

    print(f"\n{label}")
    print("-" * 76)
    print(f"  {'specification':<24} {'coef':>9} {'se':>8} {'p':>8} "
          f"{'95% CI':>18} {'n':>5} {'R2':>6}")
    for _, r in table.iterrows():
        stars = "***" if r.p < 0.01 else "**" if r.p < 0.05 else "*" if r.p < 0.1 else ""
        print(f"  {r.spec:<24} {r.coef:>9.3f} {r.se:>8.3f} {r.p:>8.3f} "
              f"{f'[{r.ci_low:+.2f}, {r.ci_high:+.2f}]':>18} {r.n:>5} {r.r2:>6.3f} {stars}")
    return table


def incidence_summary(df):
    """
    Describe new HIV infections across the panel.

    Incidence is the measure prevalence cannot stand in for. Prevalence counts
    people living with HIV, so successful treatment pushes it *up* by keeping
    people alive. Incidence counts new infections, so it is the one that says
    whether transmission is actually falling.

    Returns (regional, change): a per-year regional frame and the per-country
    percentage change across the panel's span.
    """
    rows = df.dropna(subset=["new_infections", "new_infections_per_100k", "population"])

    # Population-weighted regional rate, not the mean of the country rates:
    # the region's 43 countries range from ~100k to ~200m people, so an
    # unweighted mean would let Comoros count as much as Nigeria.
    regional = rows.groupby("year").apply(
        lambda g: pd.Series({
            "new_infections": g["new_infections"].sum(),
            "population": g["population"].sum(),
            "rate_per_100k": g["new_infections"].sum() / g["population"].sum() * 1e5,
            "median_country_rate": g["new_infections_per_100k"].median(),
            "countries": g["country"].nunique(),
        }),
        include_groups=False,
    )

    wide = rows.pivot_table(index="country", columns="year",
                            values="new_infections_per_100k")
    first, last = int(rows["year"].min()), int(rows["year"].max())
    change = ((wide[last] - wide[first]) / wide[first] * 100).dropna().sort_values()

    a, b = regional.loc[first], regional.loc[last]
    print(f"\nNew infections, {first}-{last}")
    print("-" * 76)
    print(f"  regional rate      {a.rate_per_100k:>8.1f} -> {b.rate_per_100k:>7.1f} per 100k "
          f"({b.rate_per_100k / a.rate_per_100k - 1:+.0%})")
    print(f"  absolute count   {a.new_infections:>10,.0f} -> {b.new_infections:>9,.0f} "
          f"({b.new_infections / a.new_infections - 1:+.0%})")
    print(f"  median country     {a.median_country_rate:>8.1f} -> "
          f"{b.median_country_rate:>7.1f} per 100k")
    print(f"  population       {a.population:>10,.0f} -> {b.population:>9,.0f} "
          f"({b.population / a.population - 1:+.0%})")
    print(f"\n  fell: {(change < 0).sum()}/{len(change)} countries      "
          f"rose: {(change >= 0).sum()}/{len(change)}")
    print(f"  steepest decline   {change.index[0]} ({change.iloc[0]:+.0f}%)")
    for country, value in change[change >= 0].items():
        print(f"  rose               {country} ({value:+.0f}%)")

    return regional, change


def pmtct_convergence(df):
    """
    Test Part 1's convergence claim against the whole region.

    Part 1 saw three countries land within 87-93% by 2016 and read that as
    programmes converging. Three countries cannot show convergence, only
    agreement among three. The panel can: if programmes really converged, the
    spread across countries has to narrow.

    Returns (spread, latest): dispersion by year, and the last year's values.
    """
    rows = df.dropna(subset=["pmtct_coverage_pct"])

    spread = rows.groupby("year")["pmtct_coverage_pct"].agg(
        n="count", median="median", mean="mean", sd="std",
        p10=lambda s: s.quantile(0.10),
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
        p90=lambda s: s.quantile(0.90),
    )
    spread["iqr"] = spread["p75"] - spread["p25"]

    first, last = int(spread.index.min()), int(spread.index.max())
    latest = rows[rows["year"] == last].set_index("country")["pmtct_coverage_pct"]

    print(f"\nPMTCT coverage: did programmes converge?  ({len(latest)} countries)")
    print("-" * 76)
    print(f"  {'year':<6}{'median':>8}{'sd':>7}{'IQR':>7}{'p10':>6}{'p90':>6}")
    # Milestones drawn from the years actually present, so a shorter panel
    # reports fewer rows rather than raising.
    milestones = sorted({first, *(y for y in (2010, PART1_YEAR) if y in spread.index), last})
    for year in milestones:
        r = spread.loc[year]
        print(f"  {year:<6}{r['median']:>8.0f}{r['sd']:>7.1f}{r['iqr']:>7.0f}"
              f"{r['p10']:>6.0f}{r['p90']:>6.0f}")

    a, b = spread.loc[first], spread.loc[last]
    verdict = "narrowed" if b["sd"] < a["sd"] else "widened"
    print(f"\n  spread {verdict}: sd {a['sd']:.1f} -> {b['sd']:.1f}, "
          f"IQR {a['iqr']:.0f} -> {b['iqr']:.0f}")
    print(f"  NB {first} dispersion is a floor effect - median coverage was "
          f"{a['median']:.0f}%, so countries agreed only in having no programme.")

    if PART1_YEAR in spread.index:
        year_rows = rows[rows["year"] == PART1_YEAR]
        band = year_rows["pmtct_coverage_pct"].between(*PART1_BAND).sum()
        print(f"\n  in Part 1's {PART1_BAND[0]}-{PART1_BAND[1]}% band in {PART1_YEAR}: "
              f"{band}/{len(year_rows)} countries")
        for country in FOCUS:
            value = year_rows[year_rows["country"] == country]["pmtct_coverage_pct"]
            if value.empty:
                continue
            v = value.iloc[0]
            pct = (year_rows["pmtct_coverage_pct"] < v).mean() * 100
            print(f"    {country:<14} {v:>3.0f}% in {PART1_YEAR} = {pct:.0f}th percentile")

    laggards = latest[latest < 50].sort_values()
    print(f"\n  still under 50% in {last}: {len(laggards)} countries")
    for country, value in laggards.items():
        print(f"    {country:<20} {value:>3.0f}%")

    return spread, latest


def between_within(data):
    """
    Split the income/coverage association into its two components.

    BETWEEN asks whether richer countries have higher coverage than poorer ones.
    WITHIN asks whether a country's own coverage rises as its own income does.
    They are different questions and the panel answers them differently.
    """
    means = data.groupby("country")[["art_coverage_pct", "log_gdp"]].mean()
    between = means["art_coverage_pct"].corr(means["log_gdp"])

    demeaned = data.copy()
    for col in ("art_coverage_pct", "log_gdp"):
        demeaned[col] -= demeaned.groupby("country")[col].transform("mean")
    within = demeaned["art_coverage_pct"].corr(demeaned["log_gdp"])

    print("\nBetween- vs within-country association (ART coverage ~ log GDP per capita)")
    print("-" * 76)
    print(f"  BETWEEN  country means, n={len(means):<5}   r = {between:+.3f}")
    print(f"  WITHIN   demeaned, n={len(demeaned):<5}      r = {within:+.3f}")
    return between, within


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_coverage_spaghetti(df):
    """
    Emphasis chart: the whole region in recessive grey, the three original
    countries highlighted. The point is that the scale-up is regional, not
    something special about the three countries first chosen.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    for country, rows in df.dropna(subset=["art_coverage_pct"]).groupby("country"):
        if country in FOCUS:
            continue
        ax.plot(rows["year"], rows["art_coverage_pct"], color=viz.CONTEXT,
                linewidth=1, zorder=1)

    for country in FOCUS:
        rows = df[df["country"] == country].dropna(subset=["art_coverage_pct"])
        ax.plot(rows["year"], rows["art_coverage_pct"], color=viz.FOCUS[country],
                linewidth=2.2, label=country, zorder=3)
        last = rows.iloc[-1]
        ax.annotate(f"  {country}  {last['art_coverage_pct']:.0f}%",
                    xy=(last["year"], last["art_coverage_pct"]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=viz.INK_SECONDARY)

    n = df.dropna(subset=["art_coverage_pct"])["country"].nunique()
    viz.frame(ax)
    viz.titles(ax, "Treatment scale-up was regional, not exceptional",
               f"ART coverage, {n} Sub-Saharan African countries. "
               f"The three from the original analysis are highlighted.")
    ax.set_ylabel("ART coverage (%)")
    last_year = int(df["year"].max())
    ax.set_xlim(START_YEAR - 0.3, last_year + 3.4)
    # Explicit integer ticks: the default locator puts half-years on the axis.
    ax.set_xticks(range(START_YEAR, last_year + 1, 5))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.10), ncol=3,
              fontsize=9, labelcolor=viz.INK_SECONDARY, handlelength=1.6,
              borderaxespad=0, columnspacing=2.4)
    return viz.save(fig, FIGURES / "panel-art-coverage-all.png")


def fig_incidence_spaghetti(df, regional):
    """
    Regional incidence against the spread of individual countries.

    Log scale: country rates span roughly 3 to 1,700 per 100k, and on a linear
    axis the forty countries under 200 collapse onto the floor.
    """
    rows = df.dropna(subset=["new_infections_per_100k"])
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    for country, g in rows.groupby("country"):
        if country in FOCUS:
            continue
        ax.plot(g["year"], g["new_infections_per_100k"], color=viz.CONTEXT,
                linewidth=1, zorder=1)

    for country in FOCUS:
        g = rows[rows["country"] == country]
        ax.plot(g["year"], g["new_infections_per_100k"], color=viz.FOCUS[country],
                linewidth=2.2, label=country, zorder=3)
        last = g.iloc[-1]
        ax.annotate(f"  {country}  {last['new_infections_per_100k']:.0f}",
                    xy=(last["year"], last["new_infections_per_100k"]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=viz.INK_SECONDARY)

    # The regional rate is population-weighted, so it is not the middle of the
    # grey mass and should not be read as one of the countries.
    ax.plot(regional.index, regional["rate_per_100k"], color=viz.INK,
            linewidth=2.6, label="Region (population-weighted)", zorder=4)
    last_year = int(regional.index.max())
    ax.annotate(f"  Region  {regional.loc[last_year, 'rate_per_100k']:.0f}",
                xy=(last_year, regional.loc[last_year, "rate_per_100k"]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9, fontweight="bold", color=viz.INK)

    first_year = int(regional.index.min())
    drop = regional.loc[last_year, "rate_per_100k"] / regional.loc[first_year, "rate_per_100k"] - 1

    viz.frame(ax)
    ax.set_yscale("log")
    viz.titles(ax, "New HIV infections fell across the region",
               f"New infections per 100,000 population, {int(rows['country'].nunique())} "
               f"Sub-Saharan African countries. Regional rate {drop:+.0%} since {first_year}.")
    ax.set_ylabel("New infections per 100,000 (log scale)")
    ax.set_xlim(first_year - 0.3, last_year + 4.2)
    ax.set_xticks(range(first_year, last_year + 1, 5))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.10), ncol=4,
              fontsize=9, labelcolor=viz.INK_SECONDARY, handlelength=1.6,
              borderaxespad=0, columnspacing=2.0)
    return viz.save(fig, FIGURES / "panel-incidence-all.png")


def fig_incidence_change(change):
    """
    Per-country change in incidence, sorted.

    Colour encodes the sign, which is the only thing worth encoding here:
    blue for a fall, red for a rise, against a neutral zero line. The
    highlighted countries are marked by a heavier dot and a darker label rather
    than a fourth hue, so the diverging scale stays intact.
    """
    fig, ax = plt.subplots(figsize=(8.6, 9.6))
    y = np.arange(len(change))

    ax.axvline(0, color=viz.BASELINE, linewidth=1.2, zorder=1)

    for yi, (country, value) in zip(y, change.items()):
        rose = value >= 0
        color = viz.WORSE if rose else viz.SERIES[0]
        focus = country in FOCUS
        ax.plot([0, value], [yi, yi], color=viz.CONTEXT, linewidth=1.4,
                solid_capstyle="butt", zorder=2)
        ax.scatter(value, yi, s=95 if focus else 55, color=color,
                   edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{value:+.0f}%",
                    xy=(value, yi), xytext=(10 if rose else -10, 0),
                    textcoords="offset points", va="center",
                    ha="left" if rose else "right", fontsize=8.5,
                    color=viz.INK if focus else viz.INK_MUTED)

    fell, rose = int((change < 0).sum()), int((change >= 0).sum())
    viz.frame(ax, axis="x")
    viz.titles(ax, f"Incidence fell in {fell} of {fell + rose} countries",
               "Change in new infections per 100,000, first to last year of the panel. "
               "Highlighted countries carry a heavier dot.")
    ax.set_yticks(y)
    ax.set_yticklabels(change.index, fontsize=9)
    for tick, country in zip(ax.get_yticklabels(), change.index):
        tick.set_color(viz.INK if country in FOCUS else viz.INK_SECONDARY)
        if country in FOCUS:
            tick.set_fontweight("bold")
    ax.set_xlabel("Change in incidence (%)")
    ax.set_ylim(-0.8, len(change) - 0.2)
    ax.margins(x=0.16)
    return viz.save(fig, FIGURES / "panel-incidence-change.png")


def fig_outcome_comparison(deaths, incidence):
    """
    The same coefficient, two outcomes.

    Both models regress a log outcome on percentage points of ART coverage
    with identical controls, so the coefficients sit on one scale and can be
    read against each other: how much does a point of coverage buy in lives
    versus in infections averted?
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    specs = list(deaths["spec"])
    y = np.arange(len(specs))[::-1]
    offset = 0.16

    series = [("AIDS deaths", deaths, viz.SERIES[0], +offset),
              ("New infections", incidence, viz.SERIES[1], -offset)]

    ax.axvline(0, color=viz.BASELINE, linewidth=1.2, zorder=1)
    for label, table, color, dy in series:
        for yi, (_, r) in zip(y, table.iterrows()):
            faded = r.p >= 0.05
            ax.plot([r.ci_low, r.ci_high], [yi + dy, yi + dy],
                    color=viz.CONTEXT if faded else color, linewidth=2,
                    solid_capstyle="butt", zorder=2)
            ax.scatter(r.coef, yi + dy, s=85,
                       color=viz.CONTEXT if faded else color,
                       edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"  {r.coef:+.3f}" + ("  (n.s.)" if faded else ""),
                        xy=(r.ci_high, yi + dy), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=viz.INK_SECONDARY)

    viz.frame(ax, axis="x")
    viz.titles(ax, "Treatment tracks fewer deaths more strongly than fewer infections",
               "Effect of one percentage point of ART coverage on each log outcome, "
               "same controls, 95% CIs clustered by country.")
    ax.set_yticks(y)
    ax.set_yticklabels(specs)
    ax.set_xlabel("Log points of outcome per percentage point of coverage")
    ax.set_ylim(-0.6, len(specs) - 0.4)
    ax.margins(x=0.24)
    # Proxy handles: the first row plotted is the non-significant spec, so
    # letting the legend pick up a real marker would show two grey dots.
    handles = [plt.scatter([], [], s=85, color=color, edgecolor=viz.SURFACE,
                           linewidth=1.5, label=label)
               for label, _, color, _ in series]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(0, -0.16), ncol=2, fontsize=9,
              labelcolor=viz.INK_SECONDARY, handlelength=1.2,
              borderaxespad=0, columnspacing=2.4)
    return viz.save(fig, FIGURES / "panel-incidence-coefficient.png")


def fig_prevalence_vs_incidence(df):
    """
    Why prevalence is the wrong scoreboard.

    Both measures indexed to the first year, so the two lines answer the same
    question — how much has this changed? — and disagree. Prevalence is a
    stock that treatment inflates by keeping people alive; incidence is the
    flow that says whether transmission is falling.

    The right panel is South Africa, where the two measures do not merely
    diverge in size but point in opposite directions.
    """
    rows = df.dropna(subset=["hiv_prevalence_pct", "new_infections_per_100k"])
    first = int(rows["year"].min())

    regional = rows.groupby("year")[["hiv_prevalence_pct", "new_infections_per_100k"]].median()
    south_africa = (rows[rows["country"] == "South Africa"]
                    .set_index("year")[["hiv_prevalence_pct", "new_infections_per_100k"]])

    panels = [
        (regional, f"All {rows['country'].nunique()} countries", "median across countries"),
        (south_africa, "South Africa", "the sharpest case"),
    ]
    # Prevalence in neutral ink, incidence in the series hue: the chart argues
    # that one of these lines is the one to watch.
    measures = [
        ("hiv_prevalence_pct", "Prevalence", viz.INK_SECONDARY),
        ("new_infections_per_100k", "New infections", viz.SERIES[0]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    for ax, (data, title, note) in zip(axes, panels):
        ax.axhline(100, color=viz.BASELINE, linewidth=1.2, zorder=1)
        for col, label, color in measures:
            indexed = data[col] / data[col].loc[first] * 100
            ax.plot(indexed.index, indexed, color=color, linewidth=2.2,
                    label=label, zorder=3, clip_on=False)
            last = indexed.index.max()
            ax.annotate(f"  {indexed.loc[last]:.0f}",
                        xy=(last, indexed.loc[last]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        fontweight="bold", color=color)

        viz.frame(ax)
        ax.set_title(title, fontsize=11, pad=26)
        ax.text(0, 1.01, note, transform=ax.transAxes, fontsize=9,
                color=viz.INK_SECONDARY, va="bottom")
        ax.set_xlim(first - 0.3, int(data.index.max()) + 2.6)
        ax.set_xticks(range(first, int(data.index.max()) + 1, 5))
        ax.set_ylim(0, 130)

    axes[0].set_ylabel(f"Index, {first} = 100")
    fig.suptitle("Prevalence understates progress, and can invert it",
                 x=0.008, y=1.13, ha="left", va="bottom",
                 fontsize=14, fontweight="bold", color=viz.INK)
    fig.text(0.008, 1.04,
             "HIV prevalence and new infections, each indexed to its own "
             f"{first} level. A line below 100 has improved.",
             ha="left", va="bottom", fontsize=10, color=viz.INK_SECONDARY)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12),
                   ncol=2, fontsize=9, labelcolor=viz.INK_SECONDARY,
                   handlelength=1.6, borderaxespad=0, columnspacing=2.4)
    return viz.save(fig, FIGURES / "panel-prevalence-vs-incidence.png")


def fig_pmtct_fan(df, spread):
    """
    The distribution behind Part 1's convergence claim.

    Nested quantile bands in one neutral ramp carry the spread, so the three
    categorical hues stay free for the countries Part 1 actually looked at.
    Convergence would show as the bands closing; they do the opposite.
    """
    rows = df.dropna(subset=["pmtct_coverage_pct"])
    years = spread.index

    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    # Sequential, one hue, light to dark - the outer band is the wider claim.
    ax.fill_between(years, spread["p10"], spread["p90"], color="#e4e3dd",
                    linewidth=0, zorder=1, label="10th–90th percentile")
    ax.fill_between(years, spread["p25"], spread["p75"], color="#c9c8c2",
                    linewidth=0, zorder=2, label="Interquartile range")
    ax.plot(years, spread["median"], color=viz.INK, linewidth=2.4,
            zorder=5, label="Median country")

    for country in FOCUS:
        g = rows[rows["country"] == country]
        if g.empty:
            continue
        ax.plot(g["year"], g["pmtct_coverage_pct"], color=viz.FOCUS[country],
                linewidth=2, zorder=6, label=country)
        last = g.iloc[-1]
        ax.annotate(f"  {country}  {last['pmtct_coverage_pct']:.0f}%",
                    xy=(last["year"], last["pmtct_coverage_pct"]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=viz.INK_SECONDARY)

    first, last_year = int(years.min()), int(years.max())
    viz.frame(ax)
    viz.titles(ax, "PMTCT programmes diverged, they did not converge",
               f"Coverage across {int(spread.loc[last_year, 'n'])} Sub-Saharan African "
               f"countries. The spread widens as countries scale up at "
               f"different speeds.")
    ax.set_ylabel("PMTCT coverage (%)")
    ax.set_xlim(first - 0.3, last_year + 4.6)
    ax.set_xticks(range(first, last_year + 1, 5))
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.10), ncol=3,
              fontsize=9, labelcolor=viz.INK_SECONDARY, handlelength=1.6,
              borderaxespad=0, columnspacing=2.0)
    return viz.save(fig, FIGURES / "panel-pmtct-spread.png")


def fig_income_groups(df):
    """
    One dot per country, grouped by World Bank income level, latest year with
    broad coverage. Shows the gradient in the group means AND how far the
    groups overlap — a bar chart of means would hide the second half.
    """
    year = int(df.dropna(subset=["art_coverage_pct"])["year"].max())
    latest = df[(df["year"] == year)].dropna(subset=["art_coverage_pct", "income_level"])
    # INCOME_ORDER omits "High income": the region has exactly one such economy
    # (Seychelles), and a group of one is not a group.
    latest = latest[latest["income_level"].isin(INCOME_ORDER)]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    rng = np.random.default_rng(0)

    for i, group in enumerate(INCOME_ORDER):
        rows = latest[latest["income_level"] == group]
        # Deterministic jitter so points in a column do not overlap.
        jitter = rng.uniform(-0.14, 0.14, len(rows))
        for (_, row), dx in zip(rows.iterrows(), jitter):
            focus = row["country"] in FOCUS
            ax.scatter(i + dx, row["art_coverage_pct"], s=95 if focus else 55,
                       color=viz.FOCUS[row["country"]] if focus else viz.CONTEXT,
                       edgecolor=viz.SURFACE, linewidth=1.5, zorder=4 if focus else 2)
            if focus:
                ax.annotate(f"  {row['country']}",
                            xy=(i + dx, row["art_coverage_pct"]), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=9,
                            color=viz.INK_SECONDARY, zorder=5)

        mean = rows["art_coverage_pct"].mean()
        ax.plot([i - 0.28, i + 0.28], [mean, mean], color=viz.INK,
                linewidth=2, zorder=3, solid_capstyle="butt")
        ax.annotate(f"mean {mean:.0f}%", xy=(i - 0.28, mean), xytext=(0, 8),
                    textcoords="offset points", fontsize=9, color=viz.INK)

    viz.frame(ax)
    viz.titles(ax, "Richer countries do better on average, but the groups overlap heavily",
               f"ART coverage by World Bank income group, {year}. "
               f"Each dot is one country (n={len(latest)}).")
    ax.set_ylabel("ART coverage (%)")
    ax.set_xticks(range(len(INCOME_ORDER)))
    ax.set_xticklabels([g.replace(" income", "") for g in INCOME_ORDER])
    ax.set_xlim(-0.5, len(INCOME_ORDER) - 0.5)
    ax.set_ylim(0, 100)
    return viz.save(fig, FIGURES / "panel-art-by-income.png")


def fig_coefficient_plot(table):
    """
    The income coefficient across four specifications, with 95% CIs. This is
    the whole argument in one chart: the raw association is positive, it
    survives a prevalence control, it vanishes once year is accounted for, and
    it reappears — larger — when the comparison moves inside countries.
    """
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(table))[::-1]

    ax.axvline(0, color=viz.BASELINE, linewidth=1.2, zorder=1)
    for yi, (_, r) in zip(y, table.iterrows()):
        significant = r.p < 0.05
        color = viz.SERIES[0] if significant else viz.CONTEXT
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=color, linewidth=2,
                solid_capstyle="butt", zorder=2)
        ax.scatter(r.coef, yi, s=90, color=color, edgecolor=viz.SURFACE,
                   linewidth=1.5, zorder=3)
        ax.annotate(f"  {r.coef:+.1f}pp" + ("" if significant else "  (n.s.)"),
                    xy=(r.ci_high, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=viz.INK_SECONDARY)

    viz.frame(ax, axis="x")
    viz.titles(ax, "Whether income predicts treatment coverage depends on the comparison",
               "Effect of a one-log-unit rise in GDP per capita on ART coverage, "
               "with 95% CIs clustered by country.")
    ax.set_yticks(y)
    ax.set_yticklabels(table["spec"])
    ax.set_xlabel("Percentage points of ART coverage")
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.margins(x=0.22)
    return viz.save(fig, FIGURES / "panel-income-coefficient.png")


def fig_art_vs_mortality(modelled):
    """
    Why the raw comparison cannot answer the question.

    Across countries, ART coverage and AIDS mortality are *positively*
    associated, because the countries that scaled up hardest are the ones with
    the largest epidemics. Bubble area is HIV prevalence, which is the
    confounder doing the work: the big bubbles sit high on both axes. This is
    the chart that motivates the fixed-effects model rather than contradicting
    it — the negative relationship only appears once prevalence and country are
    held fixed.
    """
    year = int(modelled["year"].max())
    latest = modelled[modelled["year"] == year]

    raw_r = np.log(latest["aids_deaths_per_100k"]).corr(latest["art_coverage_pct"])

    fig, ax = plt.subplots(figsize=(9, 5.6))

    def area(prevalence):
        return 25 + prevalence * 22

    others = latest[~latest["country"].isin(FOCUS)]
    ax.scatter(others["art_coverage_pct"], others["aids_deaths_per_100k"],
               s=area(others["hiv_prevalence_pct"]), color=viz.SERIES[0], alpha=0.5,
               edgecolor=viz.SURFACE, linewidth=1.5, zorder=2)

    for country in FOCUS:
        row = latest[latest["country"] == country]
        if row.empty:
            continue
        row = row.iloc[0]
        ax.scatter(row["art_coverage_pct"], row["aids_deaths_per_100k"],
                   s=area(row["hiv_prevalence_pct"]), color=viz.FOCUS[country],
                   edgecolor=viz.SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(f"  {country}", xy=(row["art_coverage_pct"], row["aids_deaths_per_100k"]),
                    xytext=(11, 0), textcoords="offset points", va="center",
                    fontsize=9, color=viz.INK_SECONDARY, zorder=5)

    viz.frame(ax)
    ax.grid(axis="x", color=viz.GRIDLINE, linewidth=0.8)
    ax.set_yscale("log")
    viz.titles(ax, "The raw cross-country comparison is confounded by epidemic size",
               f"{year}, {len(latest)} countries. Bubble area is HIV prevalence. "
               f"Raw correlation is r = {raw_r:+.2f} — the wrong sign.")
    ax.set_xlabel("ART coverage (%)")
    ax.set_ylabel("AIDS deaths per 100,000 (log scale)")

    # Size legend: the prevalence channel needs a key to be readable.
    handles = [plt.scatter([], [], s=area(p), color=viz.SERIES[0], alpha=0.5,
                           edgecolor=viz.SURFACE, linewidth=1.5, label=f"{p:.0f}%")
               for p in (1, 10, 20)]
    # Below the axes: in-axes placement lands the 20% key bubble on top of a
    # real country at the left edge.
    ax.legend(handles=handles, title="HIV prevalence", frameon=False,
              loc="upper left", bbox_to_anchor=(0, -0.11), ncol=3, fontsize=9,
              title_fontsize=9, labelcolor=viz.INK_SECONDARY, borderaxespad=0,
              columnspacing=2.2, handletextpad=1.0)
    ax.get_legend().get_title().set_color(viz.INK_SECONDARY)
    return viz.save(fig, FIGURES / "panel-art-vs-mortality.png")


# --------------------------------------------------------------------------

def main():
    df, modelled, incidence = load()

    print("SUB-SAHARAN AFRICA PANEL")
    print("=" * 76)
    print(f"  countries in panel:        {df['country'].nunique()}")
    print(f"  years:                     {df['year'].min()}-{df['year'].max()}")
    print(f"  country-years modelled:    {len(modelled)} "
          f"({modelled['country'].nunique()} countries with all four measures)")
    print(f"  with incidence too:        {len(incidence)} "
          f"({incidence['country'].nunique()} countries)")

    art = regression_table(ART_SPECS, modelled, "log_gdp",
                           "Does income predict ART coverage?  (outcome: ART coverage, pp)")
    between_within(modelled)
    regional, change = incidence_summary(df)
    spread, pmtct_latest = pmtct_convergence(df)

    # The two countries whose incidence rose are also the two worst on PMTCT.
    rose = change[change >= 0].index
    linked = pmtct_latest.reindex(rose).dropna()
    if not linked.empty:
        rank = pmtct_latest.rank()
        print(f"\n  countries whose incidence rose, ranked on PMTCT "
              f"(1 = lowest of {len(pmtct_latest)}):")
        for country, value in linked.items():
            print(f"    {country:<20} {value:>3.0f}%   rank {int(rank[country])}")
        aligned = pd.DataFrame({"pmtct": pmtct_latest, "change": change}).dropna()
        print(f"  corr(PMTCT, incidence change) = "
              f"{aligned['pmtct'].corr(aligned['change']):+.2f} over {len(aligned)} countries")

    # Both outcomes on the incidence sample, so the two coefficients are
    # estimated on exactly the same country-years and can be compared.
    deaths_tbl = regression_table(
        DEATH_SPECS, incidence, "art_coverage_pct",
        "Does coverage predict mortality?  (outcome: log AIDS deaths per 100k)")
    incidence_tbl = regression_table(
        INCIDENCE_SPECS, incidence, "art_coverage_pct",
        "Does coverage predict transmission?  (outcome: log new infections per 100k)")
    regression_table(
        UNTREATED_SPECS, incidence, "log_untreated",
        "Does the untreated reservoir predict transmission?  (elasticity)")

    print("\n\nFIGURES")
    print("=" * 76)
    viz.style()
    FIGURES.mkdir(exist_ok=True)
    for name in [
        fig_coverage_spaghetti(df),
        fig_income_groups(df),
        fig_coefficient_plot(art),
        fig_art_vs_mortality(modelled),
        fig_incidence_spaghetti(df, regional),
        fig_incidence_change(change),
        fig_outcome_comparison(deaths_tbl, incidence_tbl),
        fig_prevalence_vs_incidence(df),
        fig_pmtct_fan(df, spread),
    ]:
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
