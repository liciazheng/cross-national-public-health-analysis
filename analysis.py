"""
HIV in Sub-Saharan Africa, 2007-2016 — data pipeline and analysis.

Loads the two raw World Bank extracts, joins them in code, runs a set of
data-quality checks, writes a cleaned analysis dataset, and regenerates
every figure in figures/.

    python analysis.py

The raw CSVs are European Excel exports: semicolon-delimited, comma decimal
separator. They are left untouched — all corrections happen here so that
every adjustment is visible and reproducible.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path(__file__).parent / "data"
FIGURES = Path(__file__).parent / "figures"

COUNTRIES = ["South Africa", "Kenya", "Botswana"]
YEARS = (2007, 2016)

# Categorical slots 1-3 of the reference palette, assigned by entity and never
# reordered: a country keeps its hue in every figure.
COLOR = {
    "South Africa": "#2a78d6",  # blue
    "Kenya": "#eb6834",         # orange
    "Botswana": "#1baf7a",      # aqua
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_raw():
    """Read both raw extracts. Returns (hiv, economic) frames."""
    read = lambda name: pd.read_csv(
        DATA / name, sep=";", decimal=",", encoding="utf-8", encoding_errors="ignore"
    )
    hiv = read("hiv_indicators.csv")
    economic = read("economic_indicators.csv")

    hiv = hiv.rename(columns={
        "HIV_Prevalence_Female": "prevalence_female",
        "HIV_Prevalence_Male": "prevalence_male",
        "HIV_Prevalence_Total": "prevalence_total",
        "Antiretroviral_Coverage": "art_coverage",
        "PMTCT_Coverage": "pmtct_coverage",
        "People living with HIV": "people_living_with_hiv",
        "AIDS Deaths": "aids_deaths",
        "Women 15+ with HIV": "women_15plus_with_hiv",
        "Mother to child trasmission rate %": "mtct_rate",
    })
    economic = economic.rename(columns={
        "Population": "population",
        # Mislabelled at source: the values are total GDP in current USD,
        # not GDP per capita. Renamed to what it actually contains.
        "GDP_per_capita": "gdp_usd",
        "Unemployee": "unemployment_rate",
    })

    hiv.columns = [c.strip() for c in hiv.columns]
    economic.columns = [c.strip() for c in economic.columns]
    return hiv, economic


def join(hiv, economic):
    """Join the two extracts on Country + Year, asserting the join is total."""
    merged = hiv.merge(economic, on=["Country", "Year"], how="outer", indicator=True)

    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(f"unmatched join keys:\n{unmatched[['Country', 'Year', '_merge']]}")

    expected = len(COUNTRIES) * (YEARS[1] - YEARS[0] + 1)
    if len(merged) != expected:
        raise ValueError(f"expected {expected} rows, got {len(merged)}")

    return merged.drop(columns="_merge").sort_values(["Country", "Year"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------

def validate(df):
    """
    Run data-quality checks against the joined frame.

    Returns a list of (severity, message). These checks are what surfaced the
    problems documented in the README — they are kept in the pipeline so the
    same class of error cannot pass silently if the data is ever refreshed.
    """
    findings = []
    indicators = [c for c in df.columns if c not in ("Country", "Year")]

    # 1. A national indicator that never moves across ten years is almost always
    #    a fill-down artifact rather than a real constant.
    for country in COUNTRIES:
        rows = df[df["Country"] == country]
        for col in indicators:
            series = rows[col].dropna()
            if len(series) >= 5 and series.nunique() == 1:
                findings.append(("ERROR",
                    f"{country}: '{col}' is constant at {series.iloc[0]:,g} for all "
                    f"{len(series)} years — likely filled down, not measured"))

    # 2. Two different indicators holding identical values within a country means
    #    a column was copied over its neighbour.
    for country in COUNTRIES:
        rows = df[df["Country"] == country]
        for i, a in enumerate(indicators):
            for b in indicators[i + 1:]:
                # Compare as float: the same values can arrive as int in one
                # column and float in another, and .equals() is dtype-strict.
                pair = rows[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) >= 5 and (pair[a].to_numpy() == pair[b].to_numpy()).all():
                    findings.append(("ERROR",
                        f"{country}: '{a}' is identical to '{b}' across all "
                        f"{len(pair)} years — one column overwrote the other"))

    # 3. More people living with HIV than people is impossible.
    impossible = df[df["people_living_with_hiv"] > df["population"]]
    for _, row in impossible.iterrows():
        findings.append(("ERROR",
            f"{row['Country']} {row['Year']:.0f}: people_living_with_hiv "
            f"({row['people_living_with_hiv']:,.0f}) exceeds population "
            f"({row['population']:,.0f})"))

    # 4. Sex-disaggregated prevalence should bracket the total, not sit under it.
    incoherent = df.dropna(subset=["prevalence_female", "prevalence_male", "prevalence_total"])
    incoherent = incoherent[
        (incoherent["prevalence_total"] > incoherent[["prevalence_female", "prevalence_male"]].max(axis=1))
        | (incoherent["prevalence_total"] < incoherent[["prevalence_female", "prevalence_male"]].min(axis=1))
    ]
    for country in incoherent["Country"].unique():
        n = (incoherent["Country"] == country).sum()
        findings.append(("WARN",
            f"{country}: prevalence_total sits outside the female/male range in "
            f"{n}/10 years — the three columns are not the same indicator"))

    # 5. The column labelled GDP per capita is off by a factor of ~population.
    implied = df["gdp_usd"] / df["population"]
    if (df["gdp_usd"] > 1e9).all() and implied.between(200, 100_000).all():
        findings.append(("ERROR",
            f"'gdp_usd' (raw header: GDP_per_capita) holds total GDP — values run "
            f"${df['gdp_usd'].min():.3g}-${df['gdp_usd'].max():.3g}; dividing by "
            f"population gives a plausible ${implied.min():,.0f}-${implied.max():,.0f}"))

    # 6. Coverage percentages must be within 0-100.
    for col in ("art_coverage", "pmtct_coverage", "mtct_rate"):
        bad = df[(df[col] < 0) | (df[col] > 100)]
        for _, row in bad.iterrows():
            findings.append(("ERROR", f"{row['Country']} {row['Year']:.0f}: {col} = {row[col]}"))

    # 7. Missing values.
    for col in indicators:
        n = df[col].isna().sum()
        if n:
            years = sorted({int(y) for y in df[df[col].isna()]["Year"]})
            findings.append(("WARN", f"'{col}' missing for {n} rows (years {years})"))

    return findings


def cross_check_manual_join(df):
    """
    Compare the script's join against joined_dataset.csv, which was assembled by
    hand in Excel. Reports any cell where the two disagree.
    """
    path = DATA / "joined_dataset.csv"
    if not path.exists():
        return []

    manual = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8", encoding_errors="ignore")
    manual.columns = [c.strip() for c in manual.columns]
    manual = manual.rename(columns={"People living with HIV": "people_living_with_hiv"})

    findings = []
    key = ["Country", "Year"]
    merged = df[key + ["people_living_with_hiv"]].merge(
        manual[key + ["people_living_with_hiv"]], on=key, suffixes=("_script", "_manual")
    )
    diff = merged[merged["people_living_with_hiv_script"] != merged["people_living_with_hiv_manual"]]
    if not diff.empty:
        findings.append(("ERROR",
            f"'people_living_with_hiv' disagrees between the source extract and the "
            f"hand-built joined_dataset.csv in {len(diff)}/{len(merged)} rows — the "
            f"manual join edited the column"))
    return findings


# --------------------------------------------------------------------------
# Clean
# --------------------------------------------------------------------------

def clean(df):
    """
    Apply the corrections the checks justify, and derive the per-capita measures
    the raw data lacks. Every drop below is a documented data error, not a
    convenience.
    """
    out = df.copy()

    # Kenya and Botswana carry South Africa's 2016 value (6,900,000) in every
    # year. Only South Africa's series is real.
    out.loc[out["Country"] != "South Africa", "people_living_with_hiv"] = pd.NA

    # South Africa's prevalence_female duplicates pmtct_coverage, and its
    # prevalence_total (4.1-4.7%) is far below published national adult
    # prevalence for the period (~18%). Neither is usable.
    out.loc[out["Country"] == "South Africa", "prevalence_total"] = pd.NA

    # The sex-disaggregated columns are not the same indicator as the total in
    # any country (check 4) and answer none of the research questions.
    out = out.drop(columns=["prevalence_female", "prevalence_male"])

    # Derived measures. Population-normalising is what makes the three
    # countries comparable at all — Botswana is 25x smaller than South Africa.
    out["gdp_per_capita_usd"] = (out["gdp_usd"] / out["population"]).round(2)
    out["aids_deaths_per_100k"] = (out["aids_deaths"] / out["population"] * 100_000).round(2)
    out["women_15plus_with_hiv_per_100k"] = (
        out["women_15plus_with_hiv"] / out["population"] * 100_000
    ).round(2)

    return out


# --------------------------------------------------------------------------
# Analyse
# --------------------------------------------------------------------------

def analyse(df):
    """Compute the decade-change and correlation figures quoted in the README."""
    first, last = YEARS
    lines = []

    lines.append(f"Change over {first}-{last}")
    lines.append("-" * 64)
    for col, label, unit in [
        ("art_coverage", "ART coverage", "%"),
        ("pmtct_coverage", "PMTCT coverage", "%"),
        ("aids_deaths", "AIDS deaths", ""),
        ("aids_deaths_per_100k", "AIDS deaths per 100k", ""),
        ("prevalence_total", "Adult prevalence", "%"),
        ("gdp_per_capita_usd", "GDP per capita", " USD"),
    ]:
        lines.append(f"\n  {label}")
        for country in COUNTRIES:
            rows = df[df["Country"] == country].set_index("Year")[col]
            a, b = rows.get(first), rows.get(last)
            if pd.isna(a) or pd.isna(b):
                lines.append(f"    {country:<14} — (excluded: unreliable source column)")
                continue
            pct = (b - a) / a * 100
            lines.append(
                f"    {country:<14} {a:>12,.1f}{unit} -> {b:>12,.1f}{unit}   ({pct:+.0f}%)"
            )

    lines.append("\n\nCorrelations (pooled, n=%d)" % len(df.dropna(subset=["art_coverage"])))
    lines.append("-" * 64)
    for x, y, note in [
        ("art_coverage", "aids_deaths_per_100k", "treatment vs mortality"),
        ("gdp_per_capita_usd", "art_coverage", "income vs treatment reach"),
        ("gdp_per_capita_usd", "aids_deaths_per_100k", "income vs mortality"),
    ]:
        r = df[[x, y]].dropna().corr().iloc[0, 1]
        lines.append(f"  {note:<28} r = {r:+.2f}   ({x} ~ {y})")

    lines.append("\n  Within-country, ART coverage vs AIDS deaths per 100k:")
    for country in COUNTRIES:
        rows = df[df["Country"] == country]
        r = rows[["art_coverage", "aids_deaths_per_100k"]].dropna().corr().iloc[0, 1]
        lines.append(f"    {country:<14} r = {r:+.2f}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "font.size": 10,
        "figure.dpi": 140,
    })


def _frame(ax):
    """Hairline, solid, recessive chrome. No dashes."""
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, solid_capstyle="butt")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)


def _titles(ax, title, subtitle):
    """Title above subtitle, both flush left, with room for neither to collide."""
    ax.set_title(title, pad=34)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=10,
                color=INK_SECONDARY, va="bottom")


def _endpoint_labels(ax, df, col, fmt="{:.0f}"):
    """
    Direct-label the last point of each series. Label text stays in ink — the
    coloured line end beside it carries the identity.
    """
    for country in COUNTRIES:
        rows = df[df["Country"] == country].dropna(subset=[col])
        if rows.empty:
            continue
        x, y = rows["Year"].iloc[-1], rows[col].iloc[-1]
        ax.annotate(
            f"  {country}  {fmt.format(y)}",
            xy=(x, y), xytext=(6, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=INK_SECONDARY,
        )


def _lines(df, col, title, subtitle, ylabel, filename, countries=None, fmt="{:.0f}"):
    countries = countries or COUNTRIES
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for country in countries:
        rows = df[df["Country"] == country].dropna(subset=[col])
        ax.plot(rows["Year"], rows[col], color=COLOR[country], linewidth=2,
                marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=country, clip_on=False, zorder=3)

    _frame(ax)
    _endpoint_labels(ax, df[df["Country"].isin(countries)], col, fmt)

    _titles(ax, title, subtitle)
    ax.set_ylabel(ylabel)
    ax.set_xlim(YEARS[0] - 0.2, YEARS[1] + 2.6)
    ax.set_xticks(range(YEARS[0], YEARS[1] + 1))
    # Headroom so a series riding the top of its range is not clipped by the frame.
    ax.margins(y=0.14)
    # Below the plot: with endpoint labels already in the field, an in-axes legend
    # has no quadrant it can sit in without landing on a line.
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.10),
              ncol=len(countries), fontsize=9, labelcolor=INK_SECONDARY,
              handlelength=1.6, borderaxespad=0, columnspacing=2.4)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    return filename


def _small_multiples(df, col, title, subtitle, filename):
    """Absolute AIDS deaths: scales differ 100x, so each country gets its own axis."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True)

    for ax, country in zip(axes, COUNTRIES):
        rows = df[df["Country"] == country].dropna(subset=[col])
        ax.plot(rows["Year"], rows[col], color=COLOR[country], linewidth=2,
                marker="o", markersize=4, markeredgecolor=SURFACE,
                markeredgewidth=1.5, clip_on=False, zorder=3)
        _frame(ax)

        a, b = rows[col].iloc[0], rows[col].iloc[-1]
        ax.set_title(country, fontsize=11, pad=26)
        ax.text(0, 1.01, f"{a:,.0f} -> {b:,.0f}   ({(b - a) / a * 100:+.0f}%)",
                transform=ax.transAxes, fontsize=9, color=INK_SECONDARY, va="bottom")
        ax.set_ylim(0, rows[col].max() * 1.18)
        ax.set_xticks([YEARS[0], 2011, YEARS[1]])
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k" if v >= 1000 else f"{v:,.0f}")

    fig.suptitle(title, x=0.008, y=1.15, ha="left", va="bottom",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.008, 1.05, subtitle, ha="left", va="bottom",
             fontsize=10, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    return filename


def _scatter(df, x, y, title, subtitle, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5.4))

    for country in COUNTRIES:
        rows = df[df["Country"] == country].dropna(subset=[x, y])
        ax.scatter(rows[x], rows[y], s=70, color=COLOR[country],
                   edgecolor=SURFACE, linewidth=1.5, label=country, zorder=3)
        # Year path: the cloud is a decade of movement, not a static sample.
        ax.plot(rows[x], rows[y], color=COLOR[country], linewidth=1, alpha=0.35, zorder=2)

        last = rows.iloc[-1]
        ax.annotate(f"  {country}", xy=(last[x], last[y]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY)

    _frame(ax)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    _titles(ax, title, subtitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    # Lower right: the only quadrant the three trails leave empty.
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK_SECONDARY,
              handlelength=1.2, borderaxespad=1)

    fig.tight_layout()
    fig.savefig(FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_all(df):
    _style()
    FIGURES.mkdir(exist_ok=True)
    written = [
        _lines(df, "art_coverage",
               "Antiretroviral therapy coverage, 2007-2016",
               "Share of people living with HIV receiving ART (%)",
               "Coverage (%)", "art-coverage-trends.png", fmt="{:.0f}%"),

        _lines(df, "pmtct_coverage",
               "PMTCT coverage, 2007-2016",
               "Share of HIV-positive pregnant women receiving prophylaxis (%)",
               "Coverage (%)", "pmtct-coverage-trends.png", fmt="{:.0f}%"),

        _lines(df, "mtct_rate",
               "Mother-to-child transmission rate, 2010-2016",
               "Not reported before 2010",
               "Transmission rate (%)", "mtct-rate.png", fmt="{:.0f}%"),

        _lines(df, "prevalence_total",
               "Adult HIV prevalence, 2007-2016",
               "South Africa excluded: its source column is inconsistent with published estimates",
               "Prevalence (%)", "hiv-prevalence-trends.png",
               countries=["Kenya", "Botswana"], fmt="{:.1f}%"),

        _lines(df, "aids_deaths_per_100k",
               "AIDS deaths per 100,000 population, 2007-2016",
               "Population-normalised, so the three countries are comparable",
               "Deaths per 100k", "aids-deaths-per-100k.png", fmt="{:.0f}"),

        _lines(df, "gdp_per_capita_usd",
               "GDP per capita, 2007-2016",
               "Derived as total GDP / population — the raw column was mislabelled",
               "USD (current)", "gdp-per-capita.png", fmt="${:,.0f}"),

        _small_multiples(df, "aids_deaths",
                         "AIDS deaths, 2007-2016",
                         "Absolute counts; each panel has its own scale",
                         "aids-deaths.png"),

        _scatter(df, "gdp_per_capita_usd", "art_coverage",
                 "Income does not predict treatment reach",
                 "Each point is one country-year; the trail follows 2007 -> 2016",
                 "GDP per capita (USD)", "ART coverage (%)",
                 "art-coverage-vs-gdp.png"),
    ]
    return written


# --------------------------------------------------------------------------

def main():
    hiv, economic = load_raw()
    df = join(hiv, economic)

    findings = validate(df) + cross_check_manual_join(df)

    print("DATA QUALITY")
    print("=" * 64)
    for severity, message in findings:
        print(f"  [{severity:<5}] {message}")
    print(f"\n  {sum(s == 'ERROR' for s, _ in findings)} errors, "
          f"{sum(s == 'WARN' for s, _ in findings)} warnings\n")

    cleaned = clean(df)
    out = DATA / "analysis_dataset.csv"
    cleaned.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(Path(__file__).parent)} "
          f"({len(cleaned)} rows x {len(cleaned.columns)} columns)\n")

    print(analyse(cleaned))

    print("\n\nFIGURES")
    print("=" * 64)
    for name in plot_all(cleaned):
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
