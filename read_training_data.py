"""Read training data from the cross-FG feature view as pandas dataframes.

Behaviour:
- If the CSV cache files (`X_train.csv`, `X_test.csv`, optionally
  `y_train.csv`, `y_test.csv`) already exist under `--csv-dir`, the
  script loads them from disk and skips the Hopsworks read entirely.
- Otherwise it calls `feature_view.train_test_split(...)`, prints the
  shapes/timing, and writes the splits to CSV in `--csv-dir` so the
  next invocation can reuse them.

If the feature view was created with `--label final_fraud_flag`, the
labels land in `y_train` / `y_test`; otherwise the call returns
`(X_train, X_test, None, None)`.

Reports wall-clock time for login+metadata, the split read itself,
the CSV save, plus per-split row counts and rows/s throughput.

Usage
-----
    python read_training_data.py
    python read_training_data.py --test-size 0.2 --version 1
    python read_training_data.py --csv-dir /tmp/td_cache
"""

from __future__ import annotations

import argparse
import os
import time


CSV_NAMES = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")


def _csv_paths(csv_dir: str) -> tuple[str, str, str, str]:
    return tuple(os.path.join(csv_dir, n) for n in CSV_NAMES)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",       default="trans_crd_authrztn_fv",
                   help="Feature view name (default: trans_crd_authrztn_fv)")
    p.add_argument("--version",    type=int, default=1,
                   help="Feature view version (default: 1)")
    p.add_argument("--test-size",  type=float, default=0.1,
                   help="Fraction of rows in the test split (default: 0.1)")
    p.add_argument("--description", default="90/10 random train/test split",
                   help="Training dataset description")
    p.add_argument("--csv-dir", default="./training_data",
                   help="Cache dir for X/y CSVs (default: ./training_data). "
                        "Existing CSVs are reused; a fresh FV read writes them.")
    args = p.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit(f"--test-size must be in (0, 1); got {args.test_size}")

    x_tr, x_te, y_tr, y_te = _csv_paths(args.csv_dir)

    # ------------------------------------------------------------------
    # cache hit: load CSVs and stop
    # ------------------------------------------------------------------
    if os.path.exists(x_tr) and os.path.exists(x_te):
        import pandas as pd
        t0 = time.perf_counter()
        X_train = pd.read_csv(x_tr)
        X_test = pd.read_csv(x_te)
        y_train = pd.read_csv(y_tr).squeeze("columns") if os.path.exists(y_tr) else None
        y_test  = pd.read_csv(y_te).squeeze("columns") if os.path.exists(y_te) else None
        elapsed = time.perf_counter() - t0
        n_total = len(X_train) + len(X_test)
        print(f"cache hit at {args.csv_dir}/ — loaded in {elapsed:.1f}s "
              f"({n_total:,} rows total → {len(X_train):,} train + "
              f"{len(X_test):,} test)")
        print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
        if y_train is not None:
            print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")
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

    print(f"materialising train/test split (test_size={args.test_size}) …")
    t_split = time.perf_counter()
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=args.test_size,
        description=args.description,
        dataframe_type="pandas",
        statistics_config=False,
    )
    split_secs = time.perf_counter() - t_split

    n_train = len(X_train)
    n_test  = len(X_test)
    n_total = n_train + n_test
    print(f"split read in {split_secs:.1f}s "
          f"({n_total:,} rows total → {n_train:,} train + {n_test:,} test, "
          f"{n_total / split_secs:,.0f} rows/s)")
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    if y_train is not None:
        print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")
    else:
        print("y_train/y_test: None (FV has no labels — set with "
              "create_feature_view.py --label <feature> --recreate)")

    # ------------------------------------------------------------------
    # write CSV cache (only the ones that don't already exist)
    # ------------------------------------------------------------------
    os.makedirs(args.csv_dir, exist_ok=True)
    t_save = time.perf_counter()
    written = []
    for path, df in [
        (x_tr, X_train),
        (x_te, X_test),
        (y_tr, y_train),
        (y_te, y_test),
    ]:
        if df is None or os.path.exists(path):
            continue
        df.to_csv(path, index=False)
        written.append(os.path.basename(path))
    print(f"wrote {len(written)} CSV(s) to {args.csv_dir}/ "
          f"({', '.join(written) or 'none'}) [{time.perf_counter() - t_save:.1f}s]")

    print(f"total elapsed: {time.perf_counter() - t_login:.1f}s")


if __name__ == "__main__":
    main()
