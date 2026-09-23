# Two Regimes of Homicide Clearance

Code for the preprint *"Two Regimes of Homicide Clearance: Evidence from Censored Mixture Cure Models Applied to Incident-Level NIBRS Data in Three U.S. States."*

The code builds a sample of homicide incidents (NIBRS offense 09A) from the FBI's state-level NIBRS extracts, measures the time from each offense to its first arrest or exceptional clearance, treats uncleared cases as right-censored at the close of each annual file, and fits three nested models by maximum likelihood:

| Model | Clearance curve F(t) |
| --- | --- |
| 1. Single process | 1 − e^(−kt) |
| 2. Single process with ceiling (mixture cure) | L(1 − e^(−kt)) |
| 3. Two regimes with ceiling | L₁(1 − e^(−k₁t)) + L₂(1 − e^(−k₂t)) |

It also compares fitted curves with Kaplan–Meier estimates, computes bootstrap confidence intervals, runs pooled likelihood ratio tests for common clearance rates across states, and compares incidents with one versus several recorded offenders.

## Data

The data are public and are not included in this repository. Download them from the FBI Crime Data Explorer:

1. Go to https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads
2. In the NIBRS data by state section, select a state and a year and download the ZIP file.
3. Repeat for every state and year you want to analyze.
4. Put all ZIP files, with their original names (e.g. `TN-2021.zip`), in a folder called `data/`.

The preprint uses Tennessee (TN), Virginia (VA) and Michigan (MI), 2021–2025.

## Running

```bash
pip install -r requirements.txt
python src/run_analysis.py --data data --states TN VA MI --years 2021 2022 2023 2024 2025
```

Options: `--out` (output folder, default `results`), `--boot2` and `--boot3` (bootstrap replications for Models 2 and 3, defaults 200 and 100). A full run with the default settings takes a few minutes.

## Outputs (in `results/`)

| File | Content |
| --- | --- |
| `sample_<STATE>.csv` | Incident-level analytic sample |
| `sample_info.json` | Sample sizes and file cutoff dates |
| `table2_model_fits.csv` | Parameter estimates, bootstrap intervals and AIC for the three models |
| `table3_km_vs_models.csv` | Cumulative % cleared: Kaplan–Meier vs fitted models |
| `table4_offenders.csv` | Model 2 by number of recorded offenders |
| `ceiling_by_year.csv` | Model 2 by data year |
| `pooled_tests.json` | Likelihood ratio tests for common rates across states |

## Key definitions

- **Homicide:** incident with at least one offense coded 09A (murder and nonnegligent manslaughter).
- **Clearance date:** earliest arrest date of any arrestee linked to the incident, or the exceptional clearance date, whichever comes first.
- **Censoring:** uncleared incidents are censored at the file cutoff, defined as the latest arrest date recorded in that state's annual file.
- **Same-day clearances** (t = 0) are set to half a day. Time is measured in months (days / 30.44).

## Structure

```
src/build_sample.py   # reads the ZIPs and builds the incident-level sample
src/models.py         # likelihoods, fits, Kaplan–Meier, bootstrap, pooled tests
src/run_analysis.py   # runs everything and writes the result tables
```

## Citation

If you use this code, please cite the preprint (citation to be added after posting).
