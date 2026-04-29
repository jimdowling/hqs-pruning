"""Synthetic data generator for the fraud-detection feature groups referenced
by the multi-FG join query.

The query joins four offline feature groups:

  fg0 = trans_crd_authrztn_14         (one row per transaction)
  fg1 = dbt_auth_sig_aggr_9           ON  fg0.trans_key = fg1.trans_key
  fg2 = dbt_trans_dly_aggr_crd_4      ON  fg0.crd_num   = fg2.acct_id
                                     AND fg0.drvd_trans_dt = fg2.drvd_trans_dt
  fg3 = nonmon_change_stats_8         ON  fg0.crd_num   = fg3.crd_num
                                     AND fg0.trans_dt   = fg3.trans_dt

Both fg2 and fg3 are LEFT-joined, so generating with imperfect coverage is
realistic.  fg1 joins 1:1 on a per-transaction key.

Run modes
---------
  --mode parquet     write each FG to ./synthetic_fgs/<name>_<v>.parquet
  --mode hopsworks   create the FGs in Hopsworks and insert chunked

Examples
--------
  python generate_synthetic_fgs.py --rows 10_000_000
  python generate_synthetic_fgs.py --rows 1_000_000 --mode parquet
  python generate_synthetic_fgs.py --rows 50_000_000 --cards 5_000_000 --days 180
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------

EPOCH = date(2024, 1, 1)

NETWORK_IDS = ["VISA", "MC", "AMEX", "DISC"]
PROCESSOR_IDS = [f"PRC{i:03d}" for i in range(20)]
TRANS_CODES = ["AUTH", "REVR", "ADJ", "ADVCE", "CHGBK"]
ENTRY_TYPES = ["MAG", "CHIP", "CTLS", "MNL", "ECOM", "CNP"]
ECOM_SEC_LEVELS = ["00", "05", "06", "07", "08"]
MRCH_IND_CODES = [f"{i:04d}" for i in range(5800, 6000)]
CVV_CODES = ["M", "N", "P", "S", "U", " "]
ZONE_TYPES = ["DOM", "INTL", "BORDER", "CARD"]
DECISION_TYP = ["00", "01", "02", "03", "04"]
THREE_DS_VER = ["1.0", "2.0", "2.1", "2.2"]
NM_PRMT_CODES = ["A", "B", "C", "D"]
SIG_PIN = ["S", "P", "X"]
APPROVAL = ["APPROVED", "DECLINED", "REFERRED"]
FLAG_YN = ["Y", "N"]
FLAG_01 = ["0", "1"]
MRCH_NAMES = [
    "AMAZON.COM", "WALMART", "TARGET", "STARBUCKS", "MCDONALDS",
    "SHELL OIL", "EXXON", "UBER", "LYFT", "NETFLIX", "SPOTIFY",
    "COSTCO", "HOME DEPOT", "BEST BUY", "APPLE.COM", "GOOGLE",
    "MICROSOFT", "DELTA AIR", "MARRIOTT", "HILTON",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pick(g: np.random.Generator, pool, n: int) -> np.ndarray:
    arr = np.asarray(pool, dtype=object)
    return arr[g.integers(0, len(arr), size=n)]


def _dates(g: np.random.Generator, n: int, days: int) -> np.ndarray:
    base = np.datetime64(EPOCH)
    return (base + g.integers(0, days, size=n).astype("timedelta64[D]")).astype("datetime64[D]")


def _date_series(arr: np.ndarray) -> pl.Series:
    return pl.Series(arr.astype("datetime64[D]")).cast(pl.Date)


def _card_pool(n_cards: int, g: np.random.Generator) -> np.ndarray:
    nums = g.integers(4_000_000_000_000_000, 5_999_999_999_999_999, size=n_cards, dtype=np.int64)
    return np.array([str(x) for x in nums], dtype=object)


def _formatted_ids(g: np.random.Generator, prefix: str, hi: int, n: int, width: int) -> np.ndarray:
    raw = g.integers(0, hi, size=n)
    fmt = f"{prefix}{{:0{width}d}}"
    return np.array([fmt.format(int(x)) for x in raw], dtype=object)


# ---------------------------------------------------------------------------
# fg0: trans_crd_authrztn_14
# ---------------------------------------------------------------------------

def gen_fg0(n: int, cards: np.ndarray, days: int, g: np.random.Generator) -> pl.DataFrame:
    crd_num = cards[g.integers(0, len(cards), size=n)]
    trans_dt = _dates(g, n, days)
    drvd_trans_dt = trans_dt
    msrmnt_prd_dt = trans_dt
    zdaf_dt = trans_dt

    return pl.DataFrame({
        "trans_key": np.array([f"TK{i:012d}" for i in range(n)], dtype=object),
        "msrmnt_prd_dt": _date_series(msrmnt_prd_dt),
        "crd_num": crd_num,
        "drvd_trans_dt": _date_series(drvd_trans_dt),
        "trans_dt": _date_series(trans_dt),
        "trans_amt": np.round(g.exponential(75.0, size=n) + 1.0, 2).astype(np.float32),
        "trans_auth_num": _formatted_ids(g, "AU", 10**10, n, 10),
        "src_syst_trans_code": _pick(g, TRANS_CODES, n),
        "drvd_crd_prsnt_ind": _pick(g, FLAG_YN, n),
        "trans_cnt_24_hr": g.integers(0, 50, size=n).astype(np.int32),
        "mrch_ctgy_trans_amt_24_hr": np.round(g.exponential(500.0, size=n), 2).astype(np.float32),
        "mrch_ctgy_trans_cnt_24_hr": g.integers(0, 100, size=n).astype(np.int32),
        "trans_auth_ntwrk_id": _pick(g, NETWORK_IDS, n),
        "trans_auth_acqring_prcssr_id": _pick(g, PROCESSOR_IDS, n),
        "mrch_zip_cd_trvl_distance": np.round(g.exponential(50.0, size=n), 2).astype(np.float32),
        "csh_auth_vlcty_amt_48_hr": np.round(g.exponential(200.0, size=n), 2).astype(np.float32),
        "falcon_client_dfn_13_txt": _formatted_ids(g, "DFN13_", 1000, n, 4),
        "falcon_wallet_id": _formatted_ids(g, "WLT", 10**8, n, 8),
        "falcon_client_dfn_15_txt": _formatted_ids(g, "DFN15_", 1000, n, 4),
        "trans_tkn_rqstr_id": _formatted_ids(g, "TKR", 1000, n, 6),
        "dfnse_edge_trans_decsn_typ_code": _pick(g, DECISION_TYP, n),
        "dcfm_drvd_trans_entry_typ_code": _pick(g, ENTRY_TYPES, n),
        "trans_mrch_num": _formatted_ids(g, "MRC", 10**9, n, 9),
        "ecom_trans_sec_lvl_typ_code": _pick(g, ECOM_SEC_LEVELS, n),
        "mrch_ind_clsfcn_code": _pick(g, MRCH_IND_CODES, n),
        "trans_mrch_nm": _pick(g, MRCH_NAMES, n),
        "trans_src_syst_card_acct_age": g.integers(0, 240, size=n).astype(np.int32),
        "frd_strtgy_id": _formatted_ids(g, "STRT", 100, n, 4),
        "vaa_adv_auth_risk_scr": np.round(g.uniform(0, 999, size=n), 2).astype(np.float32),
        "trans_curr_frd_falcon_scr": np.round(g.uniform(0, 999, size=n), 2).astype(np.float32),
        "cvv2_vldn_output_code": _pick(g, CVV_CODES, n),
        "cvv_vldn_output_code": _pick(g, CVV_CODES, n),
        "drvd_mrch_zone_typ_code": _pick(g, ZONE_TYPES, n),
        "trnsnt_data_queue_flg": _pick(g, FLAG_YN, n),
        "prtcl_ver_code_3ds": _pick(g, THREE_DS_VER, n),
        "curr_plstc_iss_dt_to_trans_dt_days_diff": g.integers(0, 1825, size=n).astype(np.int32),
        "lst_mbl_add_dt_to_trans_dt_days_diff": g.integers(0, 1825, size=n).astype(np.int32),
        "frd_detctn_non_montry_trans_prmtd_code": _pick(g, NM_PRMT_CODES, n),
        "sig_pin_trans_ind": _pick(g, SIG_PIN, n),
        "time_on_books": g.integers(0, 240, size=n).astype(np.int32),
        "int_violations": g.integers(0, 5, size=n).astype(np.int32),
        "frad_scor_chng_nr": np.round(g.normal(0, 50, size=n), 2).astype(np.float32),
        "zffsl_total_velocity": np.round(g.exponential(300.0, size=n), 2).astype(np.float32),
        "auth_rec": _formatted_ids(g, "REC", 10**8, n, 8),
        "zdaf_date": _date_series(zdaf_dt),
        "approval_status": _pick(g, APPROVAL, n),
        "zffsl_account_number": _formatted_ids(g, "ACC", 10**10, n, 10),
        "zdaf_crdholder_num_16_clean": crd_num,
    })


# ---------------------------------------------------------------------------
# fg1: dbt_auth_sig_aggr_9 — joined 1:1 on trans_key
# ---------------------------------------------------------------------------

def gen_fg1(fg0: pl.DataFrame, coverage: float, g: np.random.Generator) -> pl.DataFrame:
    keys = fg0["trans_key"]
    if coverage < 1.0:
        keep = g.uniform(size=keys.len()) < coverage
        keys = keys.filter(pl.Series(keep))
    n = keys.len()
    return pl.DataFrame({
        "trans_key": keys,
        "queued_trans_cnt_24hr": g.integers(0, 50, size=n).astype(np.int32),
        "queued_trans_cnt_6hr": g.integers(0, 20, size=n).astype(np.int32),
        "last_hr_sig_decln_trans_amt": np.round(g.exponential(100.0, size=n), 2).astype(np.float32),
        "tot_tkn_dvc_cnt": g.integers(0, 10, size=n).astype(np.int32),
        "tkn_dvc_crd_cnt": g.integers(0, 5, size=n).astype(np.int32),
        "curr_spend_ratio": np.round(g.uniform(0, 5, size=n), 4).astype(np.float32),
        "tot_dispt_cnt_3mnth": g.integers(0, 10, size=n).astype(np.int32),
        "tot_dispt_amt_3mnth": np.round(g.exponential(150.0, size=n), 2).astype(np.float32),
        "dcad_drvd_trans_frd_ind": _pick(g, FLAG_01, n),
    })


# ---------------------------------------------------------------------------
# fg2: dbt_trans_dly_aggr_crd_4 — one row per (acct_id, drvd_trans_dt)
# ---------------------------------------------------------------------------

def gen_fg2(fg0: pl.DataFrame, coverage: float, g: np.random.Generator) -> pl.DataFrame:
    pairs = fg0.select(["crd_num", "drvd_trans_dt"]).unique()
    if coverage < 1.0:
        keep = g.uniform(size=pairs.height) < coverage
        pairs = pairs.filter(pl.Series(keep))
    n = pairs.height
    pairs = pairs.rename({"crd_num": "acct_id"})
    return pairs.with_columns(
        pl.Series("wkly_sig_trans_spnd_ratio",
                  np.round(g.uniform(0, 5, size=n), 4).astype(np.float32)),
    )


# ---------------------------------------------------------------------------
# fg3: nonmon_change_stats_8 — one row per (crd_num, trans_dt)
# ---------------------------------------------------------------------------

def gen_fg3(fg0: pl.DataFrame, coverage: float, days: int, g: np.random.Generator) -> pl.DataFrame:
    pairs = fg0.select(["crd_num", "trans_dt"]).unique()
    if coverage < 1.0:
        keep = g.uniform(size=pairs.height) < coverage
        pairs = pairs.filter(pl.Series(keep))
    n = pairs.height

    return pairs.with_columns([
        pl.Series("final_fraud_flag", (g.uniform(size=n) < 0.005).astype(np.int8)),
        pl.Series("min_report_date", _date_series(_dates(g, n, days))),
        pl.Series("ref_date", _date_series(_dates(g, n, days))),
        pl.Series("cust_src_syst_acct_close_date", _date_series(_dates(g, n, days * 4))),
        pl.Series("nonmon_changes_07d", g.integers(0, 5, size=n).astype(np.int32)),
        pl.Series("nonmon_changes_14d", g.integers(0, 10, size=n).astype(np.int32)),
        pl.Series("nonmon_changes_30d", g.integers(0, 20, size=n).astype(np.int32)),
        pl.Series("phn_change_days", g.integers(0, 730, size=n).astype(np.int32)),
        pl.Series("lostdt_change_flag", _pick(g, FLAG_YN, n)),
        pl.Series("lostdt_change_days", g.integers(0, 730, size=n).astype(np.int32)),
        pl.Series("change_date", _date_series(_dates(g, n, days))),
    ])


# ---------------------------------------------------------------------------
# Hopsworks insertion
# ---------------------------------------------------------------------------

FG_SPECS = [
    # (name, version, primary_key, event_time)
    # All FGs share a single version so repeated runs upsert into the
    # same offline tables instead of accumulating new versions.
    ("trans_crd_authrztn",     1, ["trans_key"],                "trans_dt"),
    ("dbt_auth_sig_aggr",      1, ["trans_key"],                None),
    ("dbt_trans_dly_aggr_crd", 1, ["acct_id", "drvd_trans_dt"], "drvd_trans_dt"),
    ("nonmon_change_stats",    1, ["crd_num", "trans_dt"],      "trans_dt"),
]


def insert_to_hopsworks(df: pl.DataFrame, name: str, version: int,
                        primary_key: list[str], event_time: str | None,
                        fs, chunk_rows: int, time_travel_format: str) -> None:
    print(f"[hopsworks] {name}_{version}: {df.height:,} rows")
    ttf = None if time_travel_format == "NONE" else time_travel_format
    fg = fs.get_or_create_feature_group(
        name=name,
        version=version,
        primary_key=primary_key,
        event_time=event_time,
        online_enabled=False,
        time_travel_format=ttf,
    )
    if chunk_rows <= 0 or df.height <= chunk_rows:
        fg.insert(df, write_options={"wait_for_job": True})
        return
    for start in range(0, df.height, chunk_rows):
        chunk = df.slice(start, chunk_rows)
        print(f"  inserting rows {start:,}–{start + chunk.height:,}")
        fg.insert(chunk, write_options={"wait_for_job": True})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows",         type=int,   default=10_000_000,
                   help="Number of fg0 transaction rows (default: 10,000,000)")
    p.add_argument("--cards",        type=int,   default=1_000_000,
                   help="Size of unique-card pool (default: 1,000,000)")
    p.add_argument("--days",         type=int,   default=90,
                   help="Day-range to scatter trans dates over (default: 90)")
    p.add_argument("--fg1-coverage", type=float, default=0.95,
                   help="Fraction of fg0 rows that have an fg1 match (default: 0.95)")
    p.add_argument("--fg2-coverage", type=float, default=0.85,
                   help="Fraction of (crd, day) pairs covered by fg2 (default: 0.85)")
    p.add_argument("--fg3-coverage", type=float, default=0.80,
                   help="Fraction of (crd, day) pairs covered by fg3 (default: 0.80)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--mode",         choices=["hopsworks", "parquet"], default="hopsworks")
    p.add_argument("--out-dir",      default="./synthetic_fgs",
                   help="Parquet output directory when --mode parquet")
    p.add_argument("--chunk-rows",   type=int,   default=2_000_000,
                   help="Insertion chunk size for hopsworks mode (default: 2,000,000)")
    p.add_argument("--time-travel-format", choices=["HUDI", "DELTA", "NONE"], default="HUDI",
                   help="Feature group time-travel format (default: HUDI; cluster may lack DELTA libs)")
    args = p.parse_args()

    g = np.random.default_rng(args.seed)

    print(f"Card pool: {args.cards:,} unique cards")
    cards = _card_pool(args.cards, g)

    t = time.time()
    print(f"Generating fg0 ({args.rows:,} rows) …")
    fg0 = gen_fg0(args.rows, cards, args.days, g)
    print(f"  fg0 ready in {time.time() - t:.1f}s "
          f"({fg0.height:,} rows × {fg0.width} cols, "
          f"{fg0.estimated_size('mb'):.0f} MB)")

    t = time.time()
    print("Generating fg1 …")
    fg1 = gen_fg1(fg0, args.fg1_coverage, g)
    print(f"  fg1 ready in {time.time() - t:.1f}s ({fg1.height:,} rows)")

    t = time.time()
    print("Generating fg2 …")
    fg2 = gen_fg2(fg0, args.fg2_coverage, g)
    print(f"  fg2 ready in {time.time() - t:.1f}s ({fg2.height:,} rows)")

    t = time.time()
    print("Generating fg3 …")
    fg3 = gen_fg3(fg0, args.fg3_coverage, args.days, g)
    print(f"  fg3 ready in {time.time() - t:.1f}s ({fg3.height:,} rows)")

    fg_dfs = {
        "trans_crd_authrztn":     fg0,
        "dbt_auth_sig_aggr":      fg1,
        "dbt_trans_dly_aggr_crd": fg2,
        "nonmon_change_stats":    fg3,
    }

    if args.mode == "parquet":
        os.makedirs(args.out_dir, exist_ok=True)
        for name, version, *_ in FG_SPECS:
            path = os.path.join(args.out_dir, f"{name}_{version}.parquet")
            print(f"writing {path}")
            fg_dfs[name].write_parquet(path)
        return

    import hopsworks  # local import: parquet mode shouldn't require it
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    for name, version, pk, evt in FG_SPECS:
        insert_to_hopsworks(fg_dfs[name], name, version, pk, evt, fs,
                            args.chunk_rows, args.time_travel_format)


if __name__ == "__main__":
    main()
