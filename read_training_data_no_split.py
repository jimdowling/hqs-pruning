"""Read training data from the cross-FG feature view as a single
features+labels pair (no train/test split) and save as CSVs.

Calls `feature_view.training_data(statistics_config=False, ...)` so no
descriptive statistics are computed on the materialised dataset, then
writes `X.csv` (and `y.csv` if the FV has labels) to `--csv-dir`.

If `X.csv` already exists, the script reads from disk and skips the
Hopsworks fetch.

Usage
-----
    python read_training_data_no_split.py
    python read_training_data_no_split.py --csv-dir /tmp/td_full
"""

from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",        default="trans_crd_authrztn_fv",
                   help="Feature view name (default: trans_crd_authrztn_fv)")
    p.add_argument("--version",     type=int, default=1,
                   help="Feature view version (default: 1)")
    p.add_argument("--description", default="full training data, no split",
                   help="Training dataset description")
    p.add_argument("--csv-dir",     default="./training_data_full",
                   help="Cache dir for X.csv / y.csv "
                        "(default: ./training_data_full)")
    args = p.parse_args()

    x_path = os.path.join(args.csv_dir, "X.csv")
    y_path = os.path.join(args.csv_dir, "y.csv")

    # ------------------------------------------------------------------
    # cache hit
    # ------------------------------------------------------------------
    if os.path.exists(x_path):
        import pandas as pd
        t0 = time.perf_counter()
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path).squeeze("columns") if os.path.exists(y_path) else None
        elapsed = time.perf_counter() - t0
        print(f"cache hit at {args.csv_dir}/ — loaded in {elapsed:.1f}s "
              f"({len(X):,} rows)")
        print(f"X: {X.shape}")
        if y is not None:
            print(f"y: {y.shape}")
        return

    # ------------------------------------------------------------------
    # cache miss: fetch from Hopsworks
    # ------------------------------------------------------------------
    t_login = time.perf_counter()
    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fv = fs.get_feature_view(name=args.name, version=args.version)
    print(f"feature view: {fv.name} v{fv.version} "
          f"({len(fv.features)} features, "
          f"{sum(1 for f in fv.features if f.label)} labels)  "
          f"[login+lookup {time.perf_counter() - t_login:.1f}s]")

    print("materialising training data (no split, statistics_config=False) …")
    t_read = time.perf_counter()
    X, y = fv.training_data(
        description=args.description,
        dataframe_type="pandas",
        statistics_config=False,
    )
    read_secs = time.perf_counter() - t_read

    n_rows = len(X)
    print(f"read in {read_secs:.1f}s "
          f"({n_rows:,} rows, {n_rows / read_secs:,.0f} rows/s)")
    print(f"X: {X.shape}")
    if y is not None:
        print(f"y: {y.shape}")
    else:
        print("y: None (FV has no labels — set with "
              "create_feature_view.py --label <feature> --recreate)")

    # ------------------------------------------------------------------
    # write CSV cache
    # ------------------------------------------------------------------
    os.makedirs(args.csv_dir, exist_ok=True)
    t_save = time.perf_counter()
    written = []
    if not os.path.exists(x_path):
        X.to_csv(x_path, index=False)
        written.append("X.csv")
    if y is not None and not os.path.exists(y_path):
        y.to_frame().to_csv(y_path, index=False)
        written.append("y.csv")
    print(f"wrote {len(written)} CSV(s) to {args.csv_dir}/ "
          f"({', '.join(written) or 'none'}) [{time.perf_counter() - t_save:.1f}s]")
    print(f"total elapsed: {time.perf_counter() - t_login:.1f}s")


if __name__ == "__main__":
    main()
