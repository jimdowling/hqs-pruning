"""Create a feature view that reproduces the cross-FG fraud query.

The query joins:

  fg0 = trans_crd_authrztn      v1   (driver — one row per transaction)
  fg1 = dbt_auth_sig_aggr       v1   ON  fg0.trans_key = fg1.trans_key
  fg2 = dbt_trans_dly_aggr_crd  v1   ON  fg0.crd_num   = fg2.acct_id
                                    AND fg0.drvd_trans_dt = fg2.drvd_trans_dt
  fg3 = nonmon_change_stats     v1   ON  fg0.crd_num   = fg3.crd_num
                                    AND fg0.trans_dt   = fg3.trans_dt

All joins are LEFT joins.  Only the columns named in the original SELECT are
projected — join keys are pulled in implicitly by hsfs (PKs are auto-available
for joining without being listed in `select(...)`), so they don't appear in
the feature vector unless they were already in the projection list.

No `prefix=` argument is passed on the joins.  The server rejects an
empty string, and leaving it unset means hsfs only auto-prefixes when
there is an actual feature-name collision.  The four projection lists
are disjoint and the right-side join keys (fg2.acct_id, fg3.crd_num,
fg3.trans_dt) are *not* in their projection lists, so no auto-prefixing
fires.

Usage
-----
    python create_feature_view.py
    python create_feature_view.py --name my_fv --version 1
    python create_feature_view.py --label final_fraud_flag
    python create_feature_view.py --recreate            # drop+create
"""

from __future__ import annotations

import argparse


# ---------------------------------------------------------------------------
# Per-FG projection lists (mirror the original SELECT)
# ---------------------------------------------------------------------------

FG0_FEATURES = [
    "msrmnt_prd_dt", "crd_num", "drvd_trans_dt", "trans_amt", "trans_auth_num",
    "src_syst_trans_code", "drvd_crd_prsnt_ind", "trans_cnt_24_hr",
    "mrch_ctgy_trans_amt_24_hr", "mrch_ctgy_trans_cnt_24_hr",
    "trans_auth_ntwrk_id", "trans_auth_acqring_prcssr_id",
    "mrch_zip_cd_trvl_distance", "csh_auth_vlcty_amt_48_hr",
    "falcon_client_dfn_13_txt", "falcon_wallet_id", "falcon_client_dfn_15_txt",
    "trans_tkn_rqstr_id", "dfnse_edge_trans_decsn_typ_code",
    "dcfm_drvd_trans_entry_typ_code", "trans_mrch_num",
    "ecom_trans_sec_lvl_typ_code", "mrch_ind_clsfcn_code", "trans_mrch_nm",
    "trans_src_syst_card_acct_age", "frd_strtgy_id", "vaa_adv_auth_risk_scr",
    "trans_curr_frd_falcon_scr", "cvv2_vldn_output_code", "cvv_vldn_output_code",
    "drvd_mrch_zone_typ_code", "trnsnt_data_queue_flg", "prtcl_ver_code_3ds",
    "curr_plstc_iss_dt_to_trans_dt_days_diff",
    "lst_mbl_add_dt_to_trans_dt_days_diff",
    "frd_detctn_non_montry_trans_prmtd_code", "sig_pin_trans_ind",
    "time_on_books", "int_violations", "frad_scor_chng_nr",
    "zffsl_total_velocity", "auth_rec", "zdaf_date", "approval_status",
    "zffsl_account_number", "zdaf_crdholder_num_16_clean",
]

FG1_FEATURES = [
    "queued_trans_cnt_24hr", "queued_trans_cnt_6hr",
    "last_hr_sig_decln_trans_amt", "tot_tkn_dvc_cnt", "tkn_dvc_crd_cnt",
    "curr_spend_ratio", "tot_dispt_cnt_3mnth", "tot_dispt_amt_3mnth",
    "dcad_drvd_trans_frd_ind",
]

FG2_FEATURES = ["wkly_sig_trans_spnd_ratio"]

FG3_FEATURES = [
    "final_fraud_flag", "min_report_date", "ref_date",
    "cust_src_syst_acct_close_date", "nonmon_changes_07d",
    "nonmon_changes_14d", "nonmon_changes_30d", "phn_change_days",
    "lostdt_change_flag", "lostdt_change_days", "change_date",
]


# ---------------------------------------------------------------------------
# Build query
# ---------------------------------------------------------------------------

def build_query(fs):
    fg0 = fs.get_feature_group("trans_crd_authrztn",     1)
    fg1 = fs.get_feature_group("dbt_auth_sig_aggr",      1)
    fg2 = fs.get_feature_group("dbt_trans_dly_aggr_crd", 1)
    fg3 = fs.get_feature_group("nonmon_change_stats",    1)

    # NB: do not pass prefix="" — the server rejects empty prefixes.
    # Leaving prefix unset means hsfs only auto-prefixes on actual name
    # clashes; the four projection lists are disjoint so no prefixes
    # are added.
    return (
        fg0.select(FG0_FEATURES)
        .join(
            fg1.select(FG1_FEATURES),
            on=["trans_key"],
            join_type="left",
        )
        .join(
            fg2.select(FG2_FEATURES),
            left_on=["crd_num", "drvd_trans_dt"],
            right_on=["acct_id", "drvd_trans_dt"],
            join_type="left",
        )
        .join(
            fg3.select(FG3_FEATURES),
            on=["crd_num", "trans_dt"],
            join_type="left",
        )
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name",     default="trans_crd_authrztn_fv",
                   help="Feature view name (default: trans_crd_authrztn_fv)")
    p.add_argument("--version",  type=int, default=1,
                   help="Feature view version (default: 1)")
    p.add_argument("--label",    action="append", default=[],
                   help="Repeatable. Mark feature(s) as labels (e.g. --label final_fraud_flag)")
    p.add_argument("--recreate", action="store_true",
                   help="Delete an existing FV at (name, version) before creating")
    p.add_argument("--description", default="Cross-FG fraud join: fg0 LEFT fg1, fg2, fg3")
    args = p.parse_args()

    import hopsworks
    proj = hopsworks.login()
    fs = proj.get_feature_store()

    query = build_query(fs)
    print(f"query covers {len(query.features)} features across "
          f"{1 + len(query.joins)} feature groups")

    if args.recreate:
        try:
            existing = fs.get_feature_view(name=args.name, version=args.version)
            print(f"[recreate] deleting existing {args.name} v{args.version}")
            existing.delete()
        except Exception as e:
            print(f"[recreate] no existing FV to delete ({e!s})")

    fv = fs.get_or_create_feature_view(
        name=args.name,
        version=args.version,
        query=query,
        description=args.description,
        labels=args.label or None,
        logging_enabled=False,
    )
    print(f"feature view ready: {fv.name} v{fv.version} (id={fv.id})")


if __name__ == "__main__":
    main()
