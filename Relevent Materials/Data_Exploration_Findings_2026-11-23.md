# 311 Data Exploration – First Look (Nov 23, 2026)

**Sample:** 50,000 requests created in **Oct 2025** (all of Oct 1–31), pulled from the NYC Open Data API (`erm2-nwe9`).
This is one month only. Treat the numbers as indicative, not citywide or full-year figures.

## 1. `resolution_description` is usable
| Check | Result |
|---|---|
| Filled | **97.4%** of rows (100% for NYPD, HPD, DOB, DOT, DHS; low for TLC 54% and EDC 65%) |
| Distinct values | **391**: fixed agency templates, not free text |
| Concentration | The top 10 phrases cover **66%** of rows and the top 30 cover **82%** |

A first regex pass (about 7 rules) mapped about 80% of rows into outcome classes:

| Outcome | Share | Median hours to close |
|---|---|---|
| No violation / nothing found | 26.5% | 1.6 |
| Fixed / action taken | 20.1% | 1.3 |
| Enforcement (summons / violation issued) | 12.7% | 5.7 |
| Gone on arrival | 9.3% | 2.3 |
| No access (couldn't get in) | 6.4% | **160.9** |
| Referred / out of jurisdiction | 3.9% | 36.5 |
| Duplicate | 1.2% | 24.9 |
| Other (needs more rules) | 17.3% | – |

Known issues to fix in the real version:
- The largest "other" bucket is DOB's generic "reviewed this complaint and closed it" (2.3k rows).
- DEP text such as "couldn't issue a summons" is wrongly matched as *enforcement*, so the rules need negation handling.
- Some DOT text has mojibake (encoding errors such as `Transportationâs`).

## 2. Closure time is a tricky target (affects Cavid's model)
- **2,583 requests (≈5%) closed in exactly 0 hours.** DOB's median is 0.0h, which looks like an administrative close, not a resolution.
- 252 requests took more than 1 year. Medians range from 1.1h (NYPD) to 125h (HPD) and 9,239h (EDC).
- A request with outcome "no access" takes about 160h but nothing gets fixed. A short closure time does not mean the problem was solved.
- **Suggestion:** model per agency or per complaint-type group, log-transform the target, drop or flag 0h closures, and consider adding outcome as a feature or a second target.

## 3. Other signals worth using
- **`bbl` is 90% filled.** It joins directly to PLUTO, so no spatial join is needed for building-level denominators. This is good news for the 100 m idea.
- **Repeat locations:** 267 buildings logged 10 or more requests in that month, and these account for **13.7%** of all requests.
- **Reporting channel:** Online 43%, Mobile 25%, Phone 21%, Unknown 10%.
- **Time of day:** 62% of commercial noise and 45–50% of residential and street noise complaints come in between 10pm and 4am. Housing complaints arrive at night only about 10% of the time.
- **Categories:** 152 complaint types and 601 descriptors. Casing is inconsistent (HPD uses `HEAT/HOT WATER`, others use Title Case).
- **Data-quality issues to clean:**
  - `borough == "Unspecified"`
  - Missing ZIP codes (371) and missing city values (2,259)
  - Columns that are almost empty (`taxi_*`, `bridge_*`, `road_ramp`)

## 4. Ideas, updated with this data
1. **Outcome classifier ("Will the city actually do anything?")**, recommended for Aditya. Turn the template text into outcome labels using regex, then predict the outcome from complaint type, agency, borough, hour, channel and building history. The data supports it.
2. **"Gone on arrival" race**, an add-on to #1. For NYPD complaints (noise, parking, driveway), does a slower response raise the chance that the cause is gone when police arrive? Plot P(gone) against response hours by complaint type.
3. **Chronic buildings.** Use `bbl` with PLUTO to find buildings with repeat complaints, normalize by residential units, and predict whether a building will be chronic next month. This ties to brief RQ4 and to Nabi's location comparison.
4. **Data-quality audit module.** Make the shared cleaning step a showcase: 0h and very long closures, encoding fixes, placeholder values, casing, and column sparsity. Report "before" and "after" numbers. This maps directly to the *Program quality* and *Python concepts* parts of the rubric.
5. **Digital-divide lens.** How do reporting channels (phone vs app) vary by neighborhood, and does the channel change the outcome or closure time? It needs no extra dataset.
6. **Night-time noise patterns.** Hour × weekday heatmaps by neighborhood. This fits naturally as EDA or features for Ismayil's forecast.
