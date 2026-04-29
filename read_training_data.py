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

Pass `--measure-stats-overhead` to do *two* sequential reads — one
with `statistics_config=False` and one with `statistics_config=True`
— and print the delta as the cost of computing training-dataset
statistics on this feature view.  The cache short-circuit is bypassed
in this mode so the timing reflects a real fetch.

Usage
-----
    python read_training_data.py
    python read_training_data.py --test-size 0.2 --version 1
    python read_training_data.py --csv-dir /tmp/td_cache
    python read_training_data.py --measure-stats-overhead
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
    p.add_argument("--measure-stats-overhead", action="store_true",
                   help="Read twice (statistics_config=False then True) and "
                        "print the delta as the stats-computation overhead. "
                        "Bypasses the CSV cache.")
    args = p.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit(f"--test-size must be in (0, 1); got {args.test_size}")

    x_tr, x_te, y_tr, y_te = _csv_paths(args.csv_dir)

    # ------------------------------------------------------------------
    # cache hit: load CSVs and stop (skipped when measuring overhead)
    # ------------------------------------------------------------------
    if (not args.measure_stats_overhead
            and os.path.exists(x_tr) and os.path.exists(x_te)):
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

    def _read(stats: bool) -> tuple[float, tuple]:
        t = time.perf_counter()
        out = fv.train_test_split(
            test_size=args.test_size,
            description=args.description,
            dataframe_type="pandas",
            statistics_config=stats,
        )
        return time.perf_counter() - t, out

    if args.measure_stats_overhead:
        print("read 1/2 (statistics_config=False) …")
        secs_no_stats, _ = _read(False)
        print(f"  no-stats read: {secs_no_stats:.1f}s")
        print("read 2/2 (statistics_config=True) …")
        secs_with_stats, (X_train, X_test, y_train, y_test) = _read(True)
        print(f"  with-stats read: {secs_with_stats:.1f}s")
        print(f"  stats overhead: {secs_with_stats - secs_no_stats:+.1f}s "
              f"({(secs_with_stats / secs_no_stats - 1) * 100:+.0f}%)")
        split_secs = secs_with_stats
    else:
        print(f"materialising train/test split (test_size={args.test_size}) …")
        split_secs, (X_train, X_test, y_train, y_test) = _read(False)

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
