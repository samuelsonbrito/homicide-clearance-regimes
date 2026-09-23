"""
Build the analytic sample of homicide incidents from FBI NIBRS state extracts.

Input: the state-level ZIP files downloaded from the FBI Crime Data Explorer,
named <STATE>-<YEAR>.zip (e.g. TN-2021.zip), placed in a single directory.

For each homicide incident (NIBRS offense code 09A) the script computes:
  - incident_date
  - clearance date = earliest arrest date or exceptional clearance date
  - t_days         = days from incident to clearance (NaN if not cleared)
  - censor_days    = days from incident to the cutoff of its annual file
  - n_offenders    = number of offender records in the incident
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HOMICIDE_CODE = "09A"
NOT_APPLICABLE_EXCEPTIONAL = 6  # cleared_except_id meaning "not applicable"
DAYS_PER_MONTH = 30.44


def read_table(zip_path, table, usecols):
    """Read one NIBRS table from a ZIP, whatever folder it sits in."""
    with zipfile.ZipFile(zip_path) as z:
        target = f"{table}.csv".lower()
        names = [n for n in z.namelist() if Path(n).name.lower() == target]
        if not names:
            raise FileNotFoundError(f"{table}.csv not found in {zip_path}")
        with z.open(names[0]) as f:
            return pd.read_csv(f, usecols=usecols, low_memory=False)


def build_state(data_dir, state, years):
    data_dir = Path(data_dir)
    zips = {y: data_dir / f"{state}-{y}.zip" for y in years}

    offenses, incidents, arrestees, offenders, cutoffs = [], [], [], [], {}
    for y, zp in zips.items():
        offenses.append(read_table(zp, "NIBRS_OFFENSE", ["incident_id", "offense_code"]))
        incidents.append(read_table(zp, "NIBRS_incident",
                                    ["data_year", "incident_id", "incident_date",
                                     "cleared_except_id", "cleared_except_date"]))
        arr = read_table(zp, "NIBRS_ARRESTEE", ["arrestee_id", "incident_id", "arrest_date"])
        # Cutoff of the annual file: latest arrest date recorded in it (any offense)
        cutoffs[y] = pd.to_datetime(arr["arrest_date"]).max()
        arrestees.append(arr)
        offenders.append(read_table(zp, "NIBRS_OFFENDER", ["incident_id"]))

    off = pd.concat(offenses)
    hom_ids = set(off.loc[off["offense_code"] == HOMICIDE_CODE, "incident_id"])

    inc = pd.concat(incidents)
    inc = inc[inc["incident_id"].isin(hom_ids)].copy()
    n_repeated = inc["incident_id"].duplicated().sum()
    inc["incident_date"] = pd.to_datetime(inc["incident_date"])
    inc["cleared_except_date"] = pd.to_datetime(inc["cleared_except_date"], errors="coerce")

    arr = pd.concat(arrestees)
    arr = arr[arr["incident_id"].isin(hom_ids)].drop_duplicates("arrestee_id")
    arr["arrest_date"] = pd.to_datetime(arr["arrest_date"])
    first_arrest = arr.groupby("incident_id")["arrest_date"].min()

    g = inc.groupby("incident_id").agg(
        incident_date=("incident_date", "min"),
        data_year=("data_year", "first"),
        exc_id=("cleared_except_id", "first"),
        exc_date=("cleared_except_date", "min"),
    )
    g["exc_date"] = g["exc_date"].where(g["exc_id"] != NOT_APPLICABLE_EXCEPTIONAL)
    g["first_arrest"] = first_arrest.reindex(g.index)
    g["clear_date"] = g[["first_arrest", "exc_date"]].min(axis=1)
    g["t_days"] = (g["clear_date"] - g["incident_date"]).dt.days

    # Keep incidents that occurred in the year of their file
    g = g[g["incident_date"].dt.year == g["data_year"]].copy()
    g["cutoff"] = g["data_year"].map(cutoffs)
    g["censor_days"] = (g["cutoff"] - g["incident_date"]).dt.days

    ofd = pd.concat(offenders)
    g["n_offenders"] = ofd[ofd["incident_id"].isin(g.index)].groupby("incident_id").size() \
        .reindex(g.index).fillna(0).astype(int)

    g["cleared"] = g["t_days"].notna() & (g["t_days"] >= 0)
    obs_days = np.where(g["cleared"], g["t_days"], g["censor_days"])
    # Same-day clearances (t = 0) are set to half a day
    g["t_months"] = np.clip(obs_days / DAYS_PER_MONTH, 0.5 / DAYS_PER_MONTH, None)
    g["state"] = state

    info = {
        "state": state,
        "incidents": len(g),
        "cleared": int(g["cleared"].sum()),
        "incidents_in_more_than_one_file": int(n_repeated),
        "cutoffs": {y: str(c.date()) for y, c in cutoffs.items()},
    }
    return g.reset_index(), info
