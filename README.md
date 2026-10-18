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
| 2. Hedonic fair-value model | first pass done (`analysis/first_pass.py`) |
| 3. Price ↔ time-to-rent: does price guidance beat the naive baseline? | pre-registered 24h test due 2026-10-19 |
| 4. Machine rentability score (pre-registered below) | trains on data to 2026-10-19, tested 10-19 → 10-26 |
| 5. Crypto-mining confound check (pre-registered below) | tested 10-19 → 10-26 |

### First pass, 2026-10-16 (exploratory)

- Extended features (network, memory bandwidth, PCIe, CUDA, region…) explain
  more than the pilot suggested: 5-fold CV R² of log price per GPU is
  0.24–0.26 for RTX 4090, ~0.37 for 3090, 0.30 for 5090 (GBM). Most of the
  variance is still unexplained.
- The pre-registered 24h test could not be run yet: laptop sleep left no
  gap-free 24h window (longest was 8.8h).
- On 2h and 4h windows, across four label variants, **price relative to the
  same-GPU median predicts nothing** (AUC 0.46–0.55). Models do beat the
  baseline (AUC 0.61–0.72), but the lift comes entirely from non-price
  features: machine reliability, verification, datacenter tier, and the
  machine's own history. If the 24h test agrees, the price-guidance idea is
  not supported and this README will say so plainly.

## Method sketch: a market-maker's arrival curve, applied to a host

A GPU-hour is perishable inventory: an hour not rented is gone. That makes the
host's problem the same shape as a market maker's quoting problem in
Avellaneda & Stoikov (2008), where fills arrive at intensity

    λ(δ) = A · exp(−k · δ)

for a quote placed δ away from the reference price. Here δ is the listing's
premium over its **hedonic fair value** (stage 2), and "fill" is the listing
becoming unavailable (stage 3). Two consequences:

- `k` is exactly the price ↔ time-to-rent curve this project wants to measure,
  estimated per GPU segment from the snapshot panel.
- With an additive premium (price `p = v + δ`), a risk-neutral host maximising
  expected revenue `p · λ(p − v)` has first-order condition `1 − k·p = 0`, so
  the revenue-maximising list price is `p* = 1/k`, independent of `v` within a
  segment. If the data say the decay is not exponential (e.g. a cliff around
  fair value), the closed form goes and we optimise numerically. Either way the
  output is the "price guidance" a marketplace can show a seller who has no
  idea what to ask.

What does *not* transfer from market making: there is no liquid hedge for a
GPU-hour, no continuous mid price (the fair value is itself an estimate), and
listings are not fungible. The fair-value model carries most of the weight.

## Pre-registered kill criterion

Out of sample, predict whether a new listing becomes unavailable within 24h.
If the model's AUC is not clearly above the naive baseline (price relative to
same-GPU median), the "price guidance" idea is not supported and we say so.

## Pre-registered: machine rentability score (written 2026-10-16, before the data it is tested on exists)

The first pass says *which machine* matters more than *what price*. The next
claim to test is that a score built from a machine's observed history ranks
machines better than the reliability number Vast already publishes.

- **Target**: per machine, the share of observed snapshots in the test window
  where it is unrentable for two or more consecutive observations (an
  occupancy proxy; see caveats). Binary label: above the median of its GPU
  model.
- **Train**: all snapshots up to 2026-10-19 00:00 UTC. **Test**: 2026-10-19
  00:00 → 2026-10-26 00:00 UTC. Machines need ≥ 20 observations in each
  window.
- **Baseline**: Vast's own `reliability2`.
- **Score**: history features from the train window only (past occupancy
  share, count of sub-30-minute unrentable blips, spell starts) plus static
  machine features.
- **Kill criterion**: if the score's test AUC does not beat the baseline by
  at least 0.05, with a 95% bootstrap interval excluding zero, the score adds
  nothing over what Vast already shows, and we say so.

## Pre-registered: crypto-mining confound (written 2026-10-18, before the test window)

Pearl (PRL) mining runs on consumer NVIDIA cards, and miners rent them on
Vast; [arXiv 2606.04819](https://arxiv.org/abs/2606.04819) reports budget-GPU
utilization on Vast rising from 57% to 94% after the miner's release. If that
demand is material, consumer-card occupancy here is partly a token-price
signal, not AI demand.

- **Exploratory, 2026-10-12 → 10-18** (`analysis/prl_confound.py`): hourly
  changes in consumer-card unrentable share vs log PRL price, r = +0.31
  (n = 55); datacenter cards r = +0.03. Placebo (PRL series circularly
  shifted ≥ 24h): 95th percentile +0.36, p ≈ 0.08. Not significant.
- **Test window**: 2026-10-19 00:00 → 2026-10-26 00:00 UTC, same script, PRL
  hourly prices from CoinGecko (`pearl-2`).
- **Supported if** the consumer correlation exceeds the placebo 95th
  percentile **and** the (consumer − datacenter) correlation is positive.
  Otherwise we report the confound as not supported in that window.
- Either way the rentability score result above is reported with this check
  next to it.

## Known measurement caveats

- `rentable = false` is a **proxy** for "rented": a machine can also be
  unavailable because it is offline, or because the host is **renting its own
  machine** (Vast documents self-rentals; they block clients but earn nothing).
  Treated as an upper bound on occupancy. Background "default jobs" do not
  block clients, so they should not contaminate the proxy.
- These are **ask** prices, not transaction prices.
- **The collector misses about 8% of machines in any one snapshot** (present
  in the snapshots before and after, absent in between). Absence is treated as
  "not observed", never as a state. Fixing the collector is scheduled after
  the 2026-10-19 test so the measurement does not change mid-test.
- Unrentable spells are mostly short: 56% last a single snapshot (≤ 30 min),
  which is why the analysis offers a "sustained" label.
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
