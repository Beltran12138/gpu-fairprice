"""First pass on the panel (checkpoint 2026-10-15).

1. Hedonic fit: how much of log list price per GPU do machine features explain?
   Pilot feature set vs extended set, 5-fold CV R^2, latest snapshot.
2. Pre-registered kill test: for spells where a machine becomes rentable,
   predict whether it turns unrentable within 24h. Model AUC vs naive
   baseline (price relative to same-GPU median), time-split out of sample.

Usage: python analysis/first_pass.py [db] [horizon_hours] [any|sustained] [max_starts]
  horizon_hours  24 is the pre-registered test; anything else is exploratory
  sustained      outcome needs two consecutive unrentable observations
  max_starts     drop machines with more spell starts than this in train
"""
import sqlite3
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

DB = sys.argv[1] if len(sys.argv) > 1 else "data/vast.sqlite"
FOCUS = ["RTX 4090", "RTX 5090", "RTX 3090", "RTX 3060", "RTX 5060 Ti", "RTX 5080"]
MAX_GAP_H = 1.0   # consecutive snapshots further apart than this break a spell
HORIZON_H = float(sys.argv[2]) if len(sys.argv) > 2 else 24.0  # pre-registered: 24
LABEL = sys.argv[3] if len(sys.argv) > 3 else "any"            # any | sustained
MAX_STARTS = int(sys.argv[4]) if len(sys.argv) > 4 else 0       # drop machines with more spell starts in train (0 = keep all)
PREV_MAX_H = 1.6  # previous observation may be one missed snapshot back

PILOT = ["dlperf_pg", "reliability2", "verified", "datacenter", "inet_down"]
EXTENDED = PILOT + ["inet_up", "gpu_mem_bw", "pcie_bw", "pci_gen", "cpu_ram_pg",
                    "cpu_cores_pg", "disk_bw", "disk_space", "cuda_max_good",
                    "static_ip", "direct_port_count", "num_gpus", "nvlink",
                    "storage_cost", "inet_down_cost", "region_code"]


def load():
    con = sqlite3.connect(DB)
    df = pd.read_sql_query("select * from offers", con)
    con.close()
    df["ts"] = pd.to_datetime(df["snapshot_ts"])
    # one row per machine per snapshot: the offer covering the most GPUs
    df = df.sort_values("num_gpus").groupby(["snapshot_ts", "machine_id"], as_index=False).agg(
        {**{c: "last" for c in df.columns if c not in ("snapshot_ts", "machine_id", "rentable")},
         "rentable": "max"})
    df["price_pg"] = df["dph_total"] / df["num_gpus"]
    df = df[df["price_pg"] > 0]
    df["lp"] = np.log(df["price_pg"])
    df["dlperf_pg"] = df["dlperf"] / df["num_gpus"]
    df["cpu_ram_pg"] = df["cpu_ram"] / df["num_gpus"]
    df["cpu_cores_pg"] = df["cpu_cores_effective"] / df["num_gpus"]
    df["verified"] = (df["verification"] == "verified").astype(int)
    df["datacenter"] = (df["hosting_type"] == 1).astype(int)
    df["nvlink"] = (df["bw_nvlink"].fillna(0) > 0).astype(int)
    df["static_ip"] = df["static_ip"].fillna(0).astype(float)
    region = df["geolocation"].fillna("").str.split(",").str[-1].str.strip()
    df["region_code"] = region.astype("category").cat.codes
    return df


def cv_r2(X, y, model):
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def hedonic(df):
    last = df[df["snapshot_ts"] == df["snapshot_ts"].max()]
    print(f"\n== 1. Hedonic fit, snapshot {last['snapshot_ts'].iloc[0]} ==")
    print(f"{'gpu':<13}{'n':>5}{'OLS pilot':>11}{'OLS ext':>9}{'GBM ext':>9}")
    for g in FOCUS:
        d = last[last["gpu_name"] == g]
        if len(d) < 60:
            continue
        y = d["lp"].values
        r = [cv_r2(d[f].fillna(0).values, y, m) for f, m in
             ((PILOT, LinearRegression()), (EXTENDED, LinearRegression()),
              (EXTENDED, HistGradientBoostingRegressor(max_iter=200, random_state=0)))]
        print(f"{g:<13}{len(d):>5}{r[0]:>11.2f}{r[1]:>9.2f}{r[2]:>9.2f}")


def spells(df):
    """Rows where a machine becomes rentable, labelled by whether it turns
    unrentable within 24h. Spells straddling a snapshot gap are dropped."""
    # The collector misses ~8% of machines in any one snapshot (present before
    # and after, absent in between), so absence is "not observed", never a state.
    snaps = sorted(df["ts"].unique())
    gap_after = {snaps[i]: (snaps[i + 1] - snaps[i]) / pd.Timedelta(hours=1)
                 for i in range(len(snaps) - 1)}
    end = snaps[-1]
    hour = pd.Timedelta(hours=1)
    by_m = {m: g.set_index("ts")["rentable"].sort_index() for m, g in df.groupby("machine_id")}
    rows = []
    for idx, r in df[df["rentable"] == 1].iterrows():
        t = r["ts"]
        s = by_m[r["machine_id"]]
        past = s[s.index < t]
        if past.empty or (t - past.index[-1]) / hour > PREV_MAX_H:
            continue                      # last observation too old: can't tell if new spell
        p = past.index[-1]
        if any(gap_after[x] > MAX_GAP_H for x in snaps if p <= x < t):
            continue                      # collector gap between the two observations
        if past.iloc[-1] == 1:
            continue                      # already rentable when last seen: not a start
        if t + HORIZON_H * hour > end:
            continue                      # outcome window incomplete
        win = [x for x in snaps if t <= x <= t + HORIZON_H * hour]
        if (win[-1] - t) / hour < HORIZON_H - MAX_GAP_H:
            continue
        if any(gap_after.get(x, 0) > MAX_GAP_H for x in win[:-1]):
            continue                      # whole window must be collected without a gap
        fut = s[(s.index > t) & (s.index <= t + HORIZON_H * hour)]
        if len(fut) < 0.5 * (len(win) - 1):
            continue                      # machine itself mostly unobserved in the window
        z = (fut.values == 0).astype(int)
        if LABEL == "sustained":          # two consecutive observations unrentable
            y = int(((z[1:] + z[:-1]) == 2).any())
        else:
            y = int(z.any())
        rows.append((idx, y))
    out = df.loc[[i for i, _ in rows]].copy()
    out["y"] = [y for _, y in rows]
    return out


def kill_test(df):
    tag = "" if HORIZON_H == 24 else "  [EXPLORATORY: not the pre-registered 24h]"
    print(f"\n== 2. Kill test: becomes unrentable within {HORIZON_H:g}h{tag} ==")
    print(f"label={LABEL}  max spell starts per machine in train={MAX_STARTS or 'no limit'}")
    df = df[df["gpu_name"].isin(FOCUS)].copy()
    med = df.groupby(["snapshot_ts", "gpu_name"])["lp"].transform("median")
    df["rel_price"] = df["lp"] - med
    ev = spells(df)
    print(f"spell starts with complete window: {len(ev)}  "
          f"(positive rate {ev['y'].mean():.2f})")
    if len(ev) < 100 or ev["y"].nunique() < 2:
        print("too few events for a verdict")
        return
    cut = ev["ts"].quantile(0.6)
    tr, te = ev[ev["ts"] <= cut], ev[ev["ts"] > cut]
    if MAX_STARTS:
        # flappy hosts identified from the training period only
        starts = tr.groupby("machine_id").size()
        flappy = set(starts[starts > MAX_STARTS].index)
        tr, te = tr[~tr["machine_id"].isin(flappy)], te[~te["machine_id"].isin(flappy)]
        print(f"dropped {len(flappy)} flappy machines")
    print(f"train n={len(tr)} (<= {cut}), test n={len(te)}")
    # hedonic premium: fair value fitted on training-period rows only
    fit_rows = df[df["ts"] <= cut]
    for part in (tr, te):
        part["premium"] = np.nan
    for g in FOCUS:
        f = fit_rows[fit_rows["gpu_name"] == g]
        if len(f) < 60:
            continue
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(
            f[EXTENDED].fillna(0).values, f["lp"].values)
        for part in (tr, te):
            k = part["gpu_name"] == g
            if k.any():
                part.loc[k, "premium"] = part.loc[k, "lp"] - m.predict(part.loc[k, EXTENDED].fillna(0).values)
    tr, te = tr.dropna(subset=["premium"]), te.dropna(subset=["premium"])
    if te["y"].nunique() < 2:
        print("test set has one class only")
        return
    base = roc_auc_score(te["y"], -te["rel_price"])
    feats = ["premium", "rel_price", "reliability2", "verified", "datacenter", "dlperf_pg", "num_gpus"]
    lr = LogisticRegression(max_iter=1000).fit(tr[feats].fillna(0), tr["y"])
    gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tr[feats].fillna(0), tr["y"])
    prem_only = roc_auc_score(te["y"], -te["premium"])
    print(f"AUC baseline (-price vs same-GPU median): {base:.3f}")
    print(f"AUC hedonic premium only:               {prem_only:.3f}")
    print(f"AUC logistic (premium + features):      {roc_auc_score(te['y'], lr.predict_proba(te[feats].fillna(0))[:, 1]):.3f}")
    print(f"AUC GBM (premium + features):           {roc_auc_score(te['y'], gb.predict_proba(te[feats].fillna(0))[:, 1]):.3f}")
    # bootstrap CI on (GBM - baseline)
    rng = np.random.default_rng(0)
    p_gb = gb.predict_proba(te[feats].fillna(0))[:, 1]
    yv, bv = te["y"].values, -te["rel_price"].values
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(te), len(te))
        if len(set(yv[i])) < 2:
            continue
        d.append(roc_auc_score(yv[i], p_gb[i]) - roc_auc_score(yv[i], bv[i]))
    print(f"GBM - baseline: {np.mean(d):+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
    # Where does any lift come from? The kill criterion is about *price*
    # guidance, so a win carried by machine identity does not count.
    nonprice = [f for f in feats if f not in ("premium", "rel_price")]
    gb_np = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tr[nonprice].fillna(0), tr["y"])
    print(f"AUC GBM non-price features only:        {roc_auc_score(te['y'], gb_np.predict_proba(te[nonprice].fillna(0))[:, 1]):.3f}")
    hist = tr.groupby("machine_id")["y"].mean()
    k = te["machine_id"].isin(hist.index)
    if k.sum() > 50 and te.loc[k, "y"].nunique() == 2:
        print(f"AUC machine's own train-period rate:    "
              f"{roc_auc_score(te.loc[k, 'y'], te.loc[k, 'machine_id'].map(hist)):.3f}  "
              f"(on {k.mean():.0%} of test events)")
    print("\npositive rate by premium quintile (test):")
    te = te.assign(q=pd.qcut(te["premium"], 5, labels=False, duplicates="drop"))
    print(te.groupby("q").agg(n=("y", "size"), y_rate=("y", "mean"),
                              premium=("premium", "median")).round(3).to_string())


if __name__ == "__main__":
    data = load()
    print(f"snapshots={data['snapshot_ts'].nunique()} machine-rows={len(data)} "
          f"machines={data['machine_id'].nunique()}")
    hedonic(data)
    kill_test(data)
