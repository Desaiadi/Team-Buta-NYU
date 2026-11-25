# Weather × 311 – Feasibility Check (Nov 24, 2026)

**Data used**
- **311:** daily counts for 12 weather-sensitive complaint types, 2023–2025 (1,096 days), from the NYC Open Data API, aggregated on the server.
- **Weather:** daily temperature, rain, snow and wind for Central Park from the **Open-Meteo historical API** (free, no key). NOAA's API was down (503) at the time. NOAA GHCND station `USW00094728` is the official alternative.

## 1. Weather explains 311 volume strongly
Rank correlation (Spearman) of daily counts with weather:

| Complaint type | Temperature | Rain | Snow |
|---|---|---|---|
| HEAT/HOT WATER | **−0.90** | −0.12 | +0.30 |
| Noise – Street/Sidewalk | **+0.81** | −0.12 | −0.33 |
| Dirty Condition | +0.66 | −0.08 | −0.27 |
| Illegal Fireworks | +0.61 | 0.00 | −0.19 |
| Snow or Ice | −0.62 | −0.09 | +0.35 |
| Rodent | +0.55 | +0.03 | −0.20 |
| Sewer | +0.23 | **+0.35** | −0.05 |
| Noise – Residential | **−0.04** | −0.11 | −0.04 |

**Key results**
- **Heat complaints:** in heating season (Oct–Apr) the average day sees ~1,100 complaints when lows are 30–40°F, ~2,500 at 10–20°F and ~4,200 below 10°F.
  - A regression gives **~73% more complaints per 10°F colder** (today plus yesterday), controlling for weekday and year.
  - Temperature adds real information beyond the season itself: R² rises from **0.36** (month and weekday only) to **0.58** (plus temperature).
  - The effect is strongest on the same day (ρ = −0.70) and fades over 1–3 days, so people report cold quickly.
- **Rain and sewer complaints:** a heavy-rain day (>0.5 in) roughly **doubles** sewer complaints compared with a dry day in the same month. Days with >2 in of rain average **593** against 58 on dry days, about 10×.
- **Rain suppresses outdoor complaints:** street noise drops to 0.60× on heavy-rain days and fireworks to 0.44×.
- **Residential noise is not driven by weather:** it's indoors, so it's flat across temperatures and actually falls on very hot days. This contrast is a nice talking point.

**Caveats**
- Reporting grows every year (residential noise rose from 298k to 463k between 2023 and 2025), so year controls or detrending are needed.
- One weather station represents the whole city.
- "Snow or Ice" barely existed as a category in 2023 (196 rows), which looks like a category change.

## 2. What we can build with it
**A. Weather-aware complaint forecast** (plugs into Ismayil's work)
- Add temperature and rain as features or regressors in his time series.
- For 1–7 days ahead, use the real weather forecast (e.g., the NWS API). For 6 months ahead, use normal climate values.
- Headline: "Tomorrow's cold snap → expect ~X heat complaints → staffing need."

**B. Weather-sensitivity map** (standalone, recommended)
- Compute each neighborhood's (ZIP, community board, or grid cell) **sensitivity**: the slope of complaints against temperature or rainfall.
  - Heat sensitivity: which areas spike most in a cold snap? Likely buildings with poor heating. Join `bbl` to PLUTO year-built and unit counts to test whether older buildings are more sensitive.
  - Rain sensitivity: which areas' sewer complaints jump per inch of rain? These are likely flood-vulnerable places.
- This ties back to the original 100-Meter theme: two neighboring blocks get the same weather, so a different response points to building or infrastructure differences.
- Python work: groupby with a regression per area, a PLUTO join, maps, and bootstrap confidence intervals on the slopes.

**C. Does weather slow the city down?** (plugs into Cavid's model)
- Compare closure times on snow, heavy-rain and extreme-cold days against normal days, by agency.
- Add weather as features in his resolution-time model.

**D. "Unexplained spikes" detector**
- Fit expected volume from weather, weekday and season. Flag days or areas where the actual volume is far above the prediction (outages, events, building failures).
- Good EDA and storytelling material.
