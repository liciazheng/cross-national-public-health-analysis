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

MODELLED = ["art_coverage_pct", "gdp_per_capita_usd", "hiv_prevalence_pct",
            "aids_deaths_per_100k"]


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

    return df, modelled


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
    df, modelled = load()

    print("SUB-SAHARAN AFRICA PANEL")
    print("=" * 76)
    print(f"  countries in panel:        {df['country'].nunique()}")
    print(f"  years:                     {df['year'].min()}-{df['year'].max()}")
    print(f"  country-years modelled:    {len(modelled)} "
          f"({modelled['country'].nunique()} countries with all four measures)")

    art = regression_table(ART_SPECS, modelled, "log_gdp",
                           "Does income predict ART coverage?  (outcome: ART coverage, pp)")
    regression_table(DEATH_SPECS, modelled, "art_coverage_pct",
                     "Does coverage predict mortality?  (outcome: log AIDS deaths per 100k)")
    between_within(modelled)

    print("\n\nFIGURES")
    print("=" * 76)
    viz.style()
    FIGURES.mkdir(exist_ok=True)
    for name in [
        fig_coverage_spaghetti(df),
        fig_income_groups(df),
        fig_coefficient_plot(art),
        fig_art_vs_mortality(modelled),
    ]:
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
