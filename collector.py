"""Snapshot every public Vast.ai offer into SQLite.

The public search endpoint returns at most 64 offers per query, so the
collector bisects the price axis (dph_total) until every band fits, once for
rentable offers and once for unavailable ones. Each run is one snapshot.

Usage: python collector.py [--db data/vast.sqlite]
"""
import argparse
import json
import random
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

API = "https://console.vast.ai/api/v0/bundles/"
PAGE_CAP = 64
MIN_WIDTH = 0.0005  # $/h; below this a band is accepted even if truncated
PRICE_CEIL = 500.0
NUM_GPUS_SPLIT = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)
WORKERS = 2  # free public endpoint rate-limits (429) above ~2 in flight

# Public endpoint, no auth: never route through an ambient HTTP(S)_PROXY.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

COLUMNS = [
    "id", "machine_id", "host_id", "gpu_name", "num_gpus", "gpu_ram",
    "gpu_total_ram", "gpu_frac", "dph_total", "dph_base", "min_bid",
    "rentable", "rented", "verification", "hosting_type", "reliability2",
    "dlperf", "total_flops", "gpu_max_power", "gpu_mem_bw", "pcie_bw",
    "pci_gen", "cpu_cores_effective", "cpu_ram", "disk_space", "disk_bw",
    "inet_up", "inet_down", "inet_up_cost", "inet_down_cost", "storage_cost",
    "geolocation", "static_ip", "direct_port_count", "cuda_max_good",
    "driver_version", "vms_enabled", "end_date", "duration", "discount_rate",
    "compute_cap", "bw_nvlink", "cluster_id",
]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_ts TEXT PRIMARY KEY,
    n_offers INTEGER,
    n_queries INTEGER,
    n_truncated INTEGER,
    seconds REAL
);
CREATE TABLE IF NOT EXISTS offers (
    snapshot_ts TEXT NOT NULL,
    {", ".join(f"{c} {'TEXT' if c in ('gpu_name','verification','geolocation','driver_version') else 'REAL'}" for c in COLUMNS)},
    PRIMARY KEY (snapshot_ts, id)
);
CREATE INDEX IF NOT EXISTS offers_machine ON offers(machine_id, snapshot_ts);
CREATE INDEX IF NOT EXISTS offers_gpu ON offers(gpu_name, snapshot_ts);
"""


def fetch(lo, hi, rentable, num_gpus=None):
    q = {
        "dph_total": {"gte": lo, "lt": hi},
        "rentable": {"eq": rentable},
        "limit": PAGE_CAP,
    }
    if num_gpus is not None:
        q["num_gpus"] = {"eq": num_gpus}
    url = API + "?q=" + urllib.parse.quote(json.dumps(q))
    for attempt in range(8):
        try:
            with OPENER.open(url, timeout=30) as r:
                return json.load(r).get("offers", [])
        except Exception as e:  # network blips, 429
            retry_after = getattr(e, "headers", None) and e.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(60, 2 ** attempt) + random.random()
            print(f"  retry {attempt+1} [{lo:.4f},{hi:.4f}) {e}; sleep {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed band [{lo},{hi}) rentable={rentable}")


def sweep(rentable, stats):
    """Bisect [0, PRICE_CEIL) until every band returns < PAGE_CAP offers."""
    out = {}
    # Start from a log-spaced grid so we do not waste queries splitting [0, 500).
    edges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6,
             0.7, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, PRICE_CEIL]
    level = list(zip(edges[:-1], edges[1:]))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while level:
            results = pool.map(lambda b: (b, fetch(b[0], b[1], rentable)), level)
            nxt = []
            for (lo, hi), offers in results:
                stats["queries"] += 1
                if len(offers) >= PAGE_CAP and hi - lo > MIN_WIDTH:
                    mid = (lo + hi) / 2
                    nxt += [(lo, mid), (mid, hi)]
                    continue
                if len(offers) >= PAGE_CAP:
                    # Many bundles share one exact price; split by GPU count instead.
                    offers = list(offers)
                    for n in NUM_GPUS_SPLIT:
                        part = fetch(lo, hi, rentable, num_gpus=n)
                        stats["queries"] += 1
                        if len(part) >= PAGE_CAP:
                            stats["truncated"] += 1
                        offers += part
                for o in offers:
                    out[o["id"]] = o
            level = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/vast.sqlite")
    args = ap.parse_args()

    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = {"queries": 0, "truncated": 0}
    offers = {}
    for rentable in (True, False):
        offers.update(sweep(rentable, stats))

    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)
    rows = [(ts, *[o.get(c) for c in COLUMNS]) for o in offers.values()]
    con.executemany(
        f"INSERT OR REPLACE INTO offers VALUES ({','.join('?' * (len(COLUMNS) + 1))})", rows
    )
    con.execute(
        "INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)",
        (ts, len(rows), stats["queries"], stats["truncated"], round(time.time() - t0, 1)),
    )
    con.commit()
    con.close()
    print(f"{ts} offers={len(rows)} queries={stats['queries']} "
          f"truncated={stats['truncated']} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
