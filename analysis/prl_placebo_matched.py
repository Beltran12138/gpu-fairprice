"""Placebo by shifting PRL timestamps and re-running the whole pipeline: for shift k,
each snapshot is matched to the CoinGecko point nearest (snapshot_time + k hours), within
90 min; then hourly mean, log, diff at the same consecutive-hour pairs as the real stat.
k = 0 is the real statistic. Same alignment as the real stat, n matched by construction.

Usage: python analysis/prl_placebo_matched.py <vast.sqlite> <prl.json> <start_utc> <end_utc>
"""
import json, re, sqlite3, sys
from datetime import datetime, timezone
import numpy as np

DB, PJ, START, END = sys.argv[1:5]
P = lambda s: datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
start, end = P(START), P(END)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = con.execute("select snapshot_ts, machine_id, gpu_name, num_gpus, rentable from offers").fetchall()
con.close()
best = {}
for ts, mid, g, n, r in rows:
    b = best.get((ts, mid))
    if b is None:
        best[(ts, mid)] = [n, g, r]
    else:
        if n is not None and (b[0] is None or n >= b[0]):
            b[0], b[1] = n, g if g is not None else b[1]
        b[2] = max(b[2], r) if r is not None else b[2]
cons_re = re.compile(r"RTX (30|40|50)\d\d")
dc_re = re.compile(r"(H100|H200|B200|B300|A100|L40S|L40$|A40|RTX 6000Ada|RTX 5880Ada)")
acc = {}
for (ts, mid), (n, g, r) in best.items():
    t = P(ts)
    if not (start <= t < end) or g is None:
        continue
    seg = "consumer" if (cons_re.match(g) and "laptop" not in g) else ("datacenter" if dc_re.match(g) else None)
    if seg:
        a = acc.setdefault(t, {}).setdefault(seg, [0.0, 0]); a[0] += r; a[1] += 1
snaps = [(t, 1 - d["consumer"][0] / d["consumer"][1], 1 - d["datacenter"][0] / d["datacenter"][1])
         for t, d in sorted(acc.items()) if "consumer" in d and "datacenter" in d]
prices = json.load(open(PJ))["prices"]
pt = np.array([p[0] / 1000 for p in prices]); pv = np.array([p[1] for p in prices])
h0 = snaps[0][0].replace(minute=0, second=0, microsecond=0)
hidx = np.array([int((t.replace(minute=0, second=0, microsecond=0) - h0).total_seconds() // 3600) for t, _, _ in snaps])
nh = hidx.max() + 1
st = np.array([t.timestamp() for t, _, _ in snaps])
cv = np.array([c for _, c, _ in snaps]); dv = np.array([d for _, _, d in snaps])

def run(k):
    j = np.searchsorted(pt, st + k * 3600).clip(1, len(pt) - 1)
    j = np.where(np.abs(pt[j - 1] - (st + k * 3600)) <= np.abs(pt[j] - (st + k * 3600)), j - 1, j)
    keep = np.abs(pt[j] - (st + k * 3600)) <= 90 * 60
    S = np.zeros((nh, 3)); C = np.zeros(nh)
    np.add.at(S, hidx[keep], np.c_[cv, dv, pv[j]][keep]); np.add.at(C, hidx[keep], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        H = S / C[:, None]
    cons, dc, lp = H[:, 0], H[:, 1], np.log(H[:, 2])
    ok = np.zeros(nh, bool); ok[1:] = ~np.isnan(cons[1:]) & ~np.isnan(cons[:-1])
    d = lambda x: np.diff(x)[ok[1:]]
    return d(cons), d(dc), d(lp), int(keep.sum())

def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[m], b[m])[0, 1], int(m.sum())

Dc, Dd, Dl, kept = run(0)
r_cons, n = corr(Dc, Dl); r_diff, _ = corr(Dc - Dd, Dl)
print(f"window {START} -> {END}: snapshots={len(snaps)} matched={kept} hours={nh} n={n}")
print(f"real (k=0): consumer r={r_cons:+.3f}  (consumer-datacenter) r={r_diff:+.3f}")
for label, ks in [("forward  k=+24..", range(24, nh - 24)), ("backward k=-24..", range(-24, -(nh - 24), -1))]:
    null, ns = [], []
    for k in ks:
        a, _, l, kp = run(k)
        if kp < len(snaps):          # outside CoinGecko coverage
            continue
        r, m = corr(Dc, l); null.append(r); ns.append(m)
    if not null:
        print(f"{label}: no shifts with full price coverage"); continue
    null = np.array(null); p95 = np.percentile(null, 95)
    print(f"{label}: shifts={len(null)} pairs/shift {min(ns)}-{max(ns)}  p95={p95:+.3f}  share>=real={np.mean(null >= r_cons):.3f}"
          f"  VERDICT: {'supported' if (r_cons > p95 and r_diff > 0) else 'not supported'}")
