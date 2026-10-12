# gpu-fairprice

**What should *this* machine rent for — and how long will it sit at that price?**

Open GPU price indexes already answer "what does an H100-hour cost today"
([CGI](https://github.com/getcomputable/gpu-index),
[gpu-price-index](https://github.com/henryzhangpku/gpu-price-index), Ornn,
Silicon Data). They define one standard good and discard the rest.

This project works on the part they discard: the long tail of heterogeneous
machines (consumer cards, mixed configs, community hosts) where two "RTX 4090"
listings can differ several-fold in measured performance, and individual
sellers have no idea what the market will bear.

Goal: a **machine-level fair value** plus a **price ↔ time-to-rent curve**, so a
seller can see the trade-off between list price and expected occupancy.

## Why this might matter (pilot, one snapshot, 2026-10-12)

On Vast.ai, 1×RTX 4090 offers (n=405): observable machine features
(DLPerf, reliability, verification, datacenter, bandwidth) explain only
**R² ≈ 0.10** of log list price. Sellers are pricing mostly blind.
Whether that blindness costs them occupancy is what the panel data is for.

## Status

| Stage | State |
|---|---|
| 1. Collector: snapshot all public Vast.ai offers to SQLite | running |
| 2. Hedonic fair-value model (quantile regression per GPU) | waiting for data |
| 3. Survival model: time-to-unavailable vs. price premium | needs ≥ several days of panel |
| 4. Seller tool: fair range, power-cost floor, occupancy curve | — |

## Pre-registered kill criterion

Out of sample, predict whether a new listing becomes unavailable within 24h.
If the model's AUC is not clearly above the naive baseline (price relative to
same-GPU median), the "price guidance" idea is not supported and we say so.

## Known measurement caveats

- `rentable = false` is a **proxy** for "rented": a machine can also be
  unavailable because it is offline. Treated as an upper bound on occupancy.
- These are **ask** prices, not transaction prices.
- The public search endpoint caps results at 64 per query; the collector
  bisects the price axis until every band is complete (`snapshots.n_truncated`
  records any band that could not be split further).

## Run

```bash
python collector.py              # one snapshot -> data/vast.sqlite
```

Stdlib only (Python 3.10+). Data source: Vast.ai public search API
(no key). Be polite: the collector keeps ≤2 requests in flight and honours 429s.

## License

MIT
