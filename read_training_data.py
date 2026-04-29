"""Read training data from the cross-FG feature view as pandas dataframes.

Calls `feature_view.train_test_split(test_size=0.1, dataframe_type="pandas")`,
which materialises a new training-dataset version under the feature view and
returns the four splits in memory.

If the feature view was created with `--label final_fraud_flag`, the labels
land in `y_train` / `y_test`; otherwise the call returns `(X_train, X_test,
None, None)`, so the script just concatenates X+y when y is present.

Usage
-----
    python read_training_data.py
    python read_training_data.py --test-size 0.2 --version 1
    python read_training_data.py --save-dir ./training_data
"""

from __future__ import annotations

import argparse
import os


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
    p.add_argument("--save-dir", default=None,
                   help="If set, also write each split to {save-dir}/{name}.parquet")
    args = p.parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise SystemExit(f"--test-size must be in (0, 1); got {args.test_size}")

    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    fv = fs.get_feature_view(name=args.name, version=args.version)
    print(f"feature view: {fv.name} v{fv.version} "
          f"({len(fv.features)} features, "
          f"{sum(1 for f in fv.features if f.label)} labels)")

    print(f"materialising train/test split (test_size={args.test_size}) …")
    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=args.test_size,
        description=args.description,
        dataframe_type="pandas",
    )

    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    if y_train is not None:
        print(f"y_train: {y_train.shape}  y_test: {y_test.shape}")
    else:
        print("y_train/y_test: None (FV has no labels — set with "
              "create_feature_view.py --label <feature> --recreate)")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        X_train.to_parquet(os.path.join(args.save_dir, "X_train.parquet"))
        X_test.to_parquet(os.path.join(args.save_dir,  "X_test.parquet"))
        if y_train is not None:
            y_train.to_frame().to_parquet(
                os.path.join(args.save_dir, "y_train.parquet"))
            y_test.to_frame().to_parquet(
                os.path.join(args.save_dir, "y_test.parquet"))
        print(f"saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
