"""Print the underlying Query for a feature view and execute it.

Pulls the FV's `query` attribute, renders both `repr(query)` (the
hsfs Query object) and `query.to_string()` (the offline SQL Hopsworks
will run), then calls `query.read(dataframe_type="pandas")` to
materialise the result.

By default only a small slice is read (--limit 10000) so this
program does not OOM the Arrow Flight Service the way a full
67-col × 10M-row read does.

Usage
-----
    python show_feature_view_query.py
    python show_feature_view_query.py --limit 100
    python show_feature_view_query.py --no-execute     # SQL only
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",       default="trans_crd_authrztn_fv",
                   help="Feature view name (default: trans_crd_authrztn_fv)")
    p.add_argument("--version",    type=int, default=1,
                   help="Feature view version (default: 1)")
    p.add_argument("--limit",      type=int, default=10_000,
                   help="Row limit for the executed read (default: 10,000). "
                        "Use 0 for no limit.")
    p.add_argument("--no-execute", action="store_true",
                   help="Print the query only; do not execute it")
    args = p.parse_args()

    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    fv = fs.get_feature_view(name=args.name, version=args.version)
    print(f"feature view: {fv.name} v{fv.version} "
          f"({len(fv.features)} features)")

    q = fv.query
    print()
    print("=" * 70)
    print("Query object (repr)")
    print("=" * 70)
    print(repr(q))

    print()
    print("=" * 70)
    print("Query string (offline SQL)")
    print("=" * 70)
    print(q.to_string())

    if args.no_execute:
        return

    print()
    print("=" * 70)
    if args.limit > 0:
        q_to_run = q.show  # placeholder so we use show() below
        print(f"Executing query.show({args.limit})")
    else:
        print("Executing query.read() — full materialisation")
    print("=" * 70)

    t0 = time.perf_counter()
    if args.limit > 0:
        # show() is the cheapest path: returns first N rows as nested lists
        rows = q.show(args.limit)
        elapsed = time.perf_counter() - t0
        print(f"got {len(rows):,} rows in {elapsed:.1f}s "
              f"({len(rows) / elapsed:,.0f} rows/s)")
        print(f"first row: {rows[0] if rows else '(empty)'}")
    else:
        df = q.read(dataframe_type="pandas")
        elapsed = time.perf_counter() - t0
        print(f"got {df.shape[0]:,} × {df.shape[1]} in {elapsed:.1f}s "
              f"({df.shape[0] / elapsed:,.0f} rows/s)")
        print(df.head())


if __name__ == "__main__":
    main()
