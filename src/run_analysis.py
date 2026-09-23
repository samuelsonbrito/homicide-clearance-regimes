"""
Run the full analysis and write result tables to the output directory.

Usage:
    python src/run_analysis.py --data data --states TN VA MI --years 2021 2022 2023 2024 2025
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_sample import build_state
from models import (fit_model1, fit_model2, fit_model3, F1, F2, F3, kaplan_meier,
                    bootstrap_model2, bootstrap_model3, pooled_tests)

KM_POINTS = {"1 day": 1 / 30.44, "1 week": 7 / 30.44, "1 month": 1, "3 months": 3,
             "6 months": 6, "12 months": 12}


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--states", nargs="+", default=["TN", "VA", "MI"])
    ap.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--boot2", type=int, default=200)
    ap.add_argument("--boot3", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    fits, km_rows, off_rows, year_rows, pooled_data, infos = [], [], [], [], {}, []
    for st in args.states:
        g, info = build_state(args.data, st, args.years)
        infos.append(info)
        g.to_csv(out / f"sample_{st}.csv", index=False)
        T, E = g["t_months"].to_numpy(), g["cleared"].to_numpy()
        pooled_data[st] = (T, E)
        print(f"{st}: {info['incidents']} incidents, {info['cleared']} cleared")

        m1, m2, m3 = fit_model1(T, E), fit_model2(T, E), fit_model3(T, E)
        b2 = bootstrap_model2(T, E, m2["x"], n=args.boot2)
        b3 = bootstrap_model3(T, E, m3["x"], n=args.boot3)
        fits.append({
            "state": st, "n": len(T), "cleared": int(E.sum()),
            "m1_k": m1["k"], "m1_aic": m1["aic"],
            "m2_k": m2["k"], "m2_k_ci": ci(b2[:, 0]), "m2_L": m2["L"], "m2_L_ci": ci(b2[:, 1]),
            "m2_aic": m2["aic"],
            "m3_k1": m3["k1"], "m3_k1_ci": ci(b3[:, 0]), "m3_L1": m3["L1"], "m3_L1_ci": ci(b3[:, 1]),
            "m3_k2": m3["k2"], "m3_k2_ci": ci(b3[:, 2]), "m3_L2": m3["L2"], "m3_L2_ci": ci(b3[:, 3]),
            "m3_L": m3["L"], "m3_half_life_fast_days": m3["half_life_fast_days"],
            "m3_half_life_slow_months": m3["half_life_slow_months"], "m3_aic": m3["aic"],
        })

        pts = np.array(list(KM_POINTS.values()))
        for name, vals in [("Kaplan-Meier", kaplan_meier(T, E, pts)), ("Model 1", F1(pts, m1)),
                           ("Model 2", F2(pts, m2)), ("Model 3", F3(pts, m3))]:
            km_rows.append({"state": st, "curve": name,
                            **{k: round(100 * v, 1) for k, v in zip(KM_POINTS, vals)}})

        multi = g["n_offenders"].to_numpy() > 1
        for label, m in [("one offender", ~multi), ("more than one", multi)]:
            p = fit_model2(T[m], E[m])
            b = bootstrap_model2(T[m], E[m], p["x"], n=args.boot2, seed=2)
            off_rows.append({"state": st, "group": label, "n": int(m.sum()),
                             "k": p["k"], "k_ci": ci(b[:, 0]), "L": p["L"], "L_ci": ci(b[:, 1])})

        for y in args.years:
            m = g["data_year"].to_numpy() == y
            p = fit_model2(T[m], E[m], m2["x"])
            year_rows.append({"state": st, "year": y, "n": int(m.sum()), "k": p["k"], "L": p["L"]})

    pd.DataFrame(fits).to_csv(out / "table2_model_fits.csv", index=False)
    pd.DataFrame(km_rows).to_csv(out / "table3_km_vs_models.csv", index=False)
    pd.DataFrame(off_rows).to_csv(out / "table4_offenders.csv", index=False)
    pd.DataFrame(year_rows).to_csv(out / "ceiling_by_year.csv", index=False)
    (out / "sample_info.json").write_text(json.dumps(infos, indent=2))

    if len(pooled_data) > 1:
        pt = pooled_tests(pooled_data)
        (out / "pooled_tests.json").write_text(json.dumps(pt, indent=2, default=float))
        print(json.dumps(pt, indent=2, default=float))
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
