"""Statistical analysis of weather sensitivity in 311 complaints.

All model-fitting logic lives here so the notebook stays readable and the
functions can be reused for any complaint type or weather variable.

Main ideas
----------
* A *sensitivity* is the % change in daily complaints for a fixed weather
  change (e.g. 10°F colder, or 1 inch more rain), estimated with a Poisson
  regression that controls for weekday, month and year. Controlling for month
  means we measure the effect of a cold snap relative to a normal day in the
  same month, not just "winter vs summer".
* The model is fit city-wide and then separately for every ZIP code, which
  gives one sensitivity per ZIP that we can map and compare between neighbors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CONTROLS = "C(day_of_week) + C(month) + C(year)"


@dataclass(frozen=True)
class WeatherSpec:
    """Which complaint type to model, with which weather terms, over which days."""

    complaint_type: str
    weather_terms: tuple[str, ...]   # summed into one combined effect
    unit_change: float               # e.g. -10 = "10°F colder", 1 = "+1 inch"
    label: str                       # human-readable description of unit_change
    months: tuple[int, ...] | None = None   # restrict to these months (None = all)


HEAT_SPEC = WeatherSpec("HEAT/HOT WATER", ("tmin", "tmin_lag1"), -10, "per 10°F colder",
                        months=(10, 11, 12, 1, 2, 3, 4, 5))
SEWER_SPEC = WeatherSpec("Sewer", ("precip", "precip_lag1"), 1, "per +1 inch of rain")


# --------------------------------------------------------------------------
# Data shaping
# --------------------------------------------------------------------------
def subset_panel(panel: pd.DataFrame, spec: WeatherSpec) -> pd.DataFrame:
    """Rows of the ZIP×day panel for one complaint type (and season), with complete weather."""
    df = panel[panel["complaint_type"] == spec.complaint_type]
    if spec.months:
        df = df[df["month"].isin(spec.months)]
    return df.dropna(subset=list(spec.weather_terms)).copy()


def citywide_daily(panel: pd.DataFrame, spec: WeatherSpec) -> pd.DataFrame:
    """Collapse the panel to one row per day (citywide counts plus that day's weather)."""
    df = subset_panel(panel, spec)
    weather_cols = ["tmin", "tmin_lag1", "tmean", "precip", "precip_lag1", "snow", "hdd",
                    "day_of_week", "month", "year"]
    return df.groupby("date").agg(n=("n", "sum"), **{c: (c, "first") for c in weather_cols}).reset_index()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def fit_poisson(df: pd.DataFrame, spec: WeatherSpec, with_weather: bool = True):
    """Quasi-Poisson GLM: n ~ weather terms + weekday + month + year.

    The dispersion is estimated from Pearson chi² (scale='X2'), so standard
    errors account for the overdispersion typical of count data.
    """
    rhs = CONTROLS + (" + " + " + ".join(spec.weather_terms) if with_weather else "")
    model = smf.glm(f"n ~ {rhs}", data=df, family=sm.families.Poisson())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(scale="X2")


def combined_effect(result, spec: WeatherSpec) -> dict:
    """Percent change in complaints for spec.unit_change, with a 95% CI.

    The weather terms are summed (e.g. today's and yesterday's temperature),
    and the variance of the sum uses the full covariance matrix.
    """
    terms = list(spec.weather_terms)
    beta = result.params[terms].sum()
    var = result.cov_params().loc[terms, terms].values.sum()
    se = np.sqrt(var)
    to_pct = lambda b: (np.exp(spec.unit_change * b) - 1) * 100
    lo, hi = sorted((to_pct(beta - 1.96 * se), to_pct(beta + 1.96 * se)))
    return {"pct_effect": to_pct(beta), "ci_low": lo, "ci_high": hi, "beta": beta, "se": se}


def deviance_r2(result) -> float:
    """Share of null deviance explained (a pseudo-R² for GLMs)."""
    return 1 - result.deviance / result.null_deviance


def citywide_summary(panel: pd.DataFrame, spec: WeatherSpec) -> dict:
    """Fit the citywide model with and without weather and report the gain."""
    daily = citywide_daily(panel, spec)
    full = fit_poisson(daily, spec, with_weather=True)
    base = fit_poisson(daily, spec, with_weather=False)
    return {"complaint_type": spec.complaint_type, "effect_label": spec.label, "days": len(daily),
            **combined_effect(full, spec),
            "pseudo_r2_controls_only": deviance_r2(base), "pseudo_r2_with_weather": deviance_r2(full)}


def zip_sensitivities(panel: pd.DataFrame, spec: WeatherSpec, min_complaints: int = 300) -> pd.DataFrame:
    """Fit the model separately for every ZIP with enough complaints.

    Returns one row per ZIP: its sensitivity (% change for the unit change),
    95% CI, total complaints and housing profile.
    """
    df = subset_panel(panel, spec)
    totals = df.groupby("incident_zip")["n"].sum()
    keep = totals[totals >= min_complaints].index
    profile_cols = ["res_units", "res_buildings", "median_year_built", "share_units_pre1940"]
    rows = []
    for z, g in df[df["incident_zip"].isin(keep)].groupby("incident_zip"):
        try:
            eff = combined_effect(fit_poisson(g, spec), spec)
        except (ValueError, np.linalg.LinAlgError):
            continue
        rows.append({"incident_zip": z, "complaints": int(g["n"].sum()),
                     "rate_per_10k_units_per_day": g["n_per_10k_units"].mean(),
                     **eff, **g[profile_cols].iloc[0].to_dict()})
    out = pd.DataFrame(rows).sort_values("pct_effect", ascending=False).reset_index(drop=True)
    out["ci_width"] = out["ci_high"] - out["ci_low"]
    return out


# --------------------------------------------------------------------------
# Geography: are neighboring ZIPs alike?
# --------------------------------------------------------------------------
def zip_centroids(pluto: pd.DataFrame) -> pd.DataFrame:
    """Unit-weighted centroid of each ZIP's residential lots (stands in for a ZIP shapefile)."""
    res = pluto[pluto["is_residential"] & pluto["zipcode"].notna() & pluto["latitude"].notna()]
    w = res["unitsres"]
    g = res.assign(wlat=res["latitude"] * w, wlon=res["longitude"] * w).groupby("zipcode")
    cent = pd.DataFrame({"lat": g["wlat"].sum() / g["unitsres"].sum(),
                         "lon": g["wlon"].sum() / g["unitsres"].sum(),
                         "borough": g["borough"].agg(lambda s: s.mode().iat[0])})
    return cent.rename_axis("incident_zip").reset_index()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; works on scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def neighbor_gap_test(sens: pd.DataFrame, centroids: pd.DataFrame, value: str = "pct_effect",
                      n_perm: int = 2000, seed: int = 0) -> dict:
    """Compare each ZIP with its nearest neighbor against random pairs.

    For every ZIP we find its nearest neighbor (by centroid) and take the
    absolute difference in sensitivity. A permutation test then shuffles the
    values across ZIPs many times. If real neighbors are much more alike than
    shuffled ones, sensitivity is spatially clustered; if not, neighboring
    areas differ as much as random ones (the "100-meter inequality").
    """
    d = sens.merge(centroids, on="incident_zip").reset_index(drop=True)
    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(dist, np.inf)
    nn = dist.argmin(axis=1)
    vals = d[value].to_numpy()
    observed = np.abs(vals - vals[nn]).mean()

    rng = np.random.default_rng(seed)
    perm = np.array([np.abs((v := rng.permutation(vals)) - v[nn]).mean() for _ in range(n_perm)])
    pairs = d[["incident_zip", value]].assign(neighbor_zip=d["incident_zip"].to_numpy()[nn],
                                              neighbor_value=vals[nn], distance_km=dist[np.arange(len(d)), nn])
    pairs["abs_gap"] = (pairs[value] - pairs["neighbor_value"]).abs()
    return {"observed_mean_gap": observed, "random_mean_gap": perm.mean(),
            "p_value": (np.sum(perm <= observed) + 1) / (n_perm + 1),
            "median_neighbor_km": float(np.median(pairs["distance_km"])),
            "permutation_gaps": perm, "pairs": pairs.sort_values("abs_gap", ascending=False)}


# --------------------------------------------------------------------------
# Building level: does building age matter?
# --------------------------------------------------------------------------
def building_rate_model(buildings: pd.DataFrame, min_units: int = 5) -> pd.DataFrame:
    """Quasi-Poisson model of heat complaints per building-season, offset by log(units).

    Returns rate ratios for each building era (vs the 2000+ baseline) and
    borough, i.e. how many times more complaints per apartment.
    """
    df = buildings[(buildings["unitsres"] >= min_units) & buildings["building_era"].notna()
                   & buildings["borough"].notna()].copy()
    df["building_era"] = pd.Categorical(df["building_era"].astype(str),
                                        categories=["2000+", "1980-99", "1960-79", "1940-59", "1920-39", "<1920"])
    df["size_band"] = pd.cut(df["unitsres"], [4, 10, 20, 50, 100, 10_000],
                             labels=["5-10", "11-20", "21-50", "51-100", "100+"])
    model = smf.glm("heat_complaints ~ building_era + C(borough) + C(size_band) + C(heat_season)",
                    data=df, family=sm.families.Poisson(), offset=np.log(df["unitsres"]))
    res = model.fit(scale="X2")
    ci = res.conf_int()
    out = pd.DataFrame({"rate_ratio": np.exp(res.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                        "p_value": res.pvalues})
    out = out.drop(index="Intercept")
    out.attrs["n_obs"] = int(res.nobs)
    return out


def era_summary(buildings: pd.DataFrame, min_units: int = 5) -> pd.DataFrame:
    """Raw descriptive rates by building era (before any model)."""
    df = buildings[(buildings["unitsres"] >= min_units) & buildings["building_era"].notna()]
    g = df.groupby("building_era", observed=True)
    return pd.DataFrame({
        "building_seasons": g.size(),
        "share_with_any_complaint": g["heat_complaints"].apply(lambda s: (s > 0).mean()),
        "complaints_per_100_units": g["heat_complaints"].sum() / g["unitsres"].sum() * 100,
    })
