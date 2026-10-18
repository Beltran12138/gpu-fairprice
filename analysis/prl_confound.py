"""Does consumer-GPU occupancy on Vast co-move with the Pearl (PRL) token price?

Pearl mining runs on consumer NVIDIA cards, and miners rent them on Vast
(arXiv 2606.04819). If that is a material share of demand, the occupancy
proxy for consumer cards is partly a crypto-price signal, not AI demand.
Pre-registered in README before the test window's data existed.

Usage: python analysis/prl_confound.py <vast.sqlite> <prl_market_chart.json> [start_utc] [end_utc]
  prl json: CoinGecko /coins/pearl-2/market_chart?vs_currency=usd&days=N (hourly)
"""
import json
import sqlite3
import sys

import numpy as np
import pandas as pd

DB, PJ = sys.argv[1], sys.argv[2]
START = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2026-10-19T00:00:00Z")
END = pd.Timestamp(sys.argv[4] if len(sys.argv) > 4 else "2026-10-26T00:00:00Z")

con = sqlite3.connect(DB)
df = pd.read_sql_query("select snapshot_ts, machine_id, gpu_name, num_gpus, rentable from offers", con)
con.close()
df = df.sort_values("num_gpus").groupby(["snapshot_ts", "machine_id"], as_index=False).agg(
    gpu=("gpu_name", "last"), rent=("rentable", "max"))
df["ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
df = df[(df["ts"] >= START) & (df["ts"] < END)]

g = df["gpu"]
consumer = g.str.match(r"RTX (30|40|50)\d\d") & ~g.str.contains("laptop")
dc = g.str.match(r"(H100|H200|B200|B300|A100|L40S|L40$|A40|RTX 6000Ada|RTX 5880Ada)")
df["seg"] = np.where(consumer, "consumer", np.where(dc, "datacenter", "other"))
occ = df[df["seg"] != "other"].groupby(["ts", "seg"])["rent"].agg(lambda s: 1 - s.mean()).unstack()

p = pd.DataFrame(json.load(open(PJ))["prices"], columns=["ms", "prl"])
p["ts"] = pd.to_datetime(p["ms"], unit="ms", utc=True)
p = p.set_index("ts")["prl"].sort_index()
occ["prl"] = p.reindex(occ.index, method="nearest", tolerance=pd.Timedelta("90min"))
occ = occ.dropna()
print(f"window {START} -> {END}: snapshots={len(occ)}")

H = occ.resample("1h").mean()
H["lp"] = np.log(H["prl"])
ok = H["consumer"].notna() & H["consumer"].shift(1).notna()   # consecutive hours only
D = H.diff()[ok]
print(f"consecutive-hour changes n={len(D)}")
if len(D) < 30:
    print("too few hourly changes for a verdict")
    sys.exit()

r_cons = D["consumer"].corr(D["lp"])
r_dc = D["datacenter"].corr(D["lp"])
r_diff = (D["consumer"] - D["datacenter"]).corr(D["lp"])
null = []
for k in range(24, len(H) - 24):     # circular shifts of at least a day
    sh = pd.Series(np.roll(H["lp"].values, k), index=H.index).diff()[ok]
    null.append(D["consumer"].corr(sh))
null = np.array([x for x in null if not np.isnan(x)])
p95 = np.percentile(null, 95)
print(f"consumer r={r_cons:+.3f}  datacenter r={r_dc:+.3f}  (consumer-datacenter) r={r_diff:+.3f}")
print(f"placebo shifts n={len(null)}: 95th pct={p95:+.3f}, share >= real {np.mean(null >= r_cons):.3f}")
hit = r_cons > p95 and r_diff > 0
print("VERDICT:", "confound supported" if hit else "confound not supported in this window")
