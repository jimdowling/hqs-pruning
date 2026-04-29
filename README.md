# hqs-pruning

Synthetic data + feature view + training-data pipeline that reproduces
the cross-FG fraud join in `original-query.sql` against Hopsworks.

## Programs and run order

Run from this directory in order. Steps 1–2 set up the offline data
and the feature view; the rest are read paths.

### 1. Generate synthetic data and load into Hopsworks
```
python generate_synthetic_fgs.py --rows 10000000 --time-travel-format DELTA
```
Creates four feature groups at version 1 and bulk-inserts in 2M-row
chunks:

| FG | name (v1) | rows (default) | join key(s) |
|---|---|---|---|
| fg0 | `trans_crd_authrztn` | 10,000,000 | `trans_key`, `crd_num`, `trans_dt`, `drvd_trans_dt` |
| fg1 | `dbt_auth_sig_aggr` | ~9.5M | `trans_key` |
| fg2 | `dbt_trans_dly_aggr_crd` | ~8.0M | `acct_id`, `drvd_trans_dt` |
| fg3 | `nonmon_change_stats` | ~7.6M | `crd_num`, `trans_dt` |

Use `--mode parquet` for a local-only run, `--time-travel-format HUDI`
if the cluster lacks delta-spark, and `--cards / --days / --rows /
--fg{1,2,3}-coverage` to scale the dataset.

### 2. Create the feature view
```
python create_feature_view.py
```
Builds an hsfs `Query` that LEFT-joins fg0 with fg1, fg2, fg3 on the
keys above, projects exactly the columns named in the original SELECT,
and registers `trans_crd_authrztn_fv v1` (67 features). Use
`--label final_fraud_flag --recreate` to mark the fraud column as a
training label.

### 3. (Optional) Inspect the rendered offline SQL
```
python show_feature_view_query.py --no-execute
```
Prints `repr(fv.query)` and `query.to_string()`. Drop `--no-execute`
to also run `query.show(--limit)` (default 10k rows).

### 4. Read training data
Two paths, both with CSV caching:

#### 90 / 10 train/test split
```
python read_training_data.py
```
Calls `feature_view.train_test_split(test_size=0.1,
statistics_config=False)`, writes
`./training_data/{X,y}_{train,test}.csv`. Subsequent runs short-circuit
to the local CSVs.

Add `--measure-stats-overhead` to time the cost of computing
training-dataset statistics (does two reads, prints the delta).

#### Full dataset, no split
```
python read_training_data_no_split.py
```
Calls `feature_view.training_data(statistics_config=False)`, writes
`./training_data_full/{X,y}.csv`.

## Repository layout
```
generate_synthetic_fgs.py       step 1: synthetic data + Hopsworks load
create_feature_view.py          step 2: feature view
show_feature_view_query.py      step 3 (optional): inspect SQL
read_training_data.py           step 4a: 90/10 split, CSV cache
read_training_data_no_split.py  step 4b: full dataset, CSV cache
original-query.sql              SQL the feature view reproduces
```
