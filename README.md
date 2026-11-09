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
| 1. Collector: snapshot all public Vast.ai offers to SQLite | **stopped 2026-10-28** (see note below the table) |
| 2. Hedonic fair-value model | first pass done (`analysis/first_pass.py`) |
| 3. Price ↔ time-to-rent: does price guidance beat the naive baseline? | **no verdict**: the 10-19 run found no complete 24h window (see note) |
| 4. Machine rentability score (pre-registered below) | **not run** (see note) |
| 5. Crypto-mining confound check (pre-registered below) | **not supported** in the 10-19 → 10-26 window (run 2026-10-28) |

**Note, 2026-10-28.** Collection stopped after the pre-registered windows
closed. Vast.ai's terms of service restrict automated collection and the
building of benchmark or market-data products from its data without written
permission; this project has no such permission, so it publishes no further
price statistics. The existing analyses stay up as a record.

Stage 3 has no verdict. The scheduled 10-19 run (`analysis/run_verdict.cmd`)
found zero listings with a complete 24h observation window, for the same
reason as the first pass: collector gaps (laptop sleep, runs up to 31 minutes)
leave no gap-free 24h stretch. The 2h/4h exploratory result above is the only
evidence, and it is exploratory. Stage 4 was never run: no scoring script was
written before collection stopped.

### First pass, 2026-10-16 (exploratory)

- Extended features (network, memory bandwidth, PCIe, CUDA, region…) explain
  more than the pilot suggested: 5-fold CV R² of log price per GPU is
  0.24–0.26 for RTX 4090, ~0.37 for 3090, 0.30 for 5090 (GBM). Most of the
  variance is still unexplained.
  *(Withdrawn 2026-11-09: these folds were not grouped by host, and with
  host-grouped folds the R² mostly disappears. See the second correction
  below.)*
- The pre-registered 24h test could not be run yet: laptop sleep left no
  gap-free 24h window (longest was 8.8h).
- On 2h and 4h windows, across four label variants, **price relative to the
  same-GPU median predicts nothing** (AUC 0.46–0.55). Models do beat the
  baseline (AUC 0.61–0.72), but the lift comes entirely from non-price
  features: machine reliability, verification, datacenter tier, and the
  machine's own history. If the 24h test agrees, the price-guidance idea is
  not supported and this README will say so plainly.
- **Correction (2026-11-09), found by an adversarial code audit.**
  - *The bug.* Machines list several bundles (1 GPU, 2 GPUs, …). Per-GPU price
    differs across a machine's bundles in 98% of multi-bundle machine-snapshots.
    The collapse to one row per machine took price and specs from the largest
    bundle, but took `rentable` as the max over all bundles, column by column.
    So a rentable machine often carried the price of a bundle that could not be
    rented.
  - *The fix.* `first_pass.py` now takes each row whole from one offer: the
    largest rentable bundle if any is rentable, else the largest bundle. The
    outcome definition is unchanged.
  - *Re-run.* The data is cut at the same 10-16 08:32 snapshot (81 snapshots);
    old and new code were run side by side on 2h/4h × any/sustained. Event
    counts barely move (2,635 → 2,622 at 2h).
  - **The baseline result stands:** price relative to the same-GPU median gives
    AUC 0.48–0.53 before and 0.48–0.52 after the fix, and the hedonic premium alone gives
    0.52–0.55.
  - The full models rise slightly: GBM goes from 0.65–0.67 to 0.65–0.74, and
    logistic from 0.67–0.72 to 0.69–0.77.
  - **"Entirely from non-price features" overstated it, before and after the
    fix.** Dropping the two price features lowers GBM AUC by 0.00–0.05 with the
    old code and by 0.02–0.05 with the new code. These are point estimates with
    no interval, from a bootstrap that is not clustered by machine. Read them
    as "price features add little beyond machine features", not "nothing".
    *(Superseded by the intervals in the second correction below.)*
  - The PRL confound check uses only GPU model and rentable state, and a
    machine's bundles differ in GPU model in 0.7% of multi-bundle
    machine-snapshots. So the check is essentially unaffected and was not
    re-run.
- **Second correction (2026-11-09): the remaining five audit findings.** All
  were fixed in `first_pass.py` and re-run on the same 10-16 cut, across
  2h/4h × any/sustained × flappy filter off/on, plus two runs with a stricter
  gap threshold.
  - **Hedonic CV was not grouped by host, and grouping by host removes most of
    the R².** A host prices all its machines alike, so shuffled folds let a
    host's pricing leak from training into test. With folds grouped by
    `host_id`, GBM R² on the extended features (10-16 snapshot) is −0.08 for
    RTX 4090, 0.08 for 5090 and 0.24 for 3090, against 0.25 / 0.33 / 0.41 with
    shuffled folds on the same data. The collapse fix does not explain the
    drop: it moves these numbers by at most 0.01. The specs largely do not
    predict what a *new* host will charge. That weakens stage 2 itself: a
    "fair value" from specs is mostly a fair value for hosts the model has
    already seen.
  - *The old README figures could not be reproduced.* The 10-16 log gives
    4090 0.23, 3090 0.41, 5090 0.31. A scheduled run on the 10-19 snapshot gave
    4090 0.05, so single-snapshot fits were unstable even before grouping.
  - **The hedonic premium was in-sample for training events and out-of-sample
    for test events.** It is now cross-fitted by host: each event is priced by
    a model that never saw its host. On its own, the premium now gives AUC
    0.50–0.62, with no interval.
  - **The bootstrap now resamples whole machines** (188–587 test machines,
    depending on the variant). It now also covers the price-feature increment
    (GBM minus the same GBM without the two price features).
  - **The gap threshold sat on a knife-edge.** Snapshot gaps cluster at about
    0.5h, and at 1.0h after one missed run (0.9997–1.05h); the next gap is
    1.30h. The old threshold of 1.0 split identical one-miss gaps by seconds of
    jitter. It is now 1.1, which allows one missed run. With a stricter 0.9
    (no missed run allowed), events fall by 38–54% and the conclusions below
    are unchanged.
  - **The flappy filter counted labelled events, not spell starts.** It now
    counts every unrentable → rentable transition in the training period.
  - **Results after all fixes:**
    - Price relative to the same-GPU median: AUC 0.44–0.52 across all ten
      runs. **The baseline result stands.**
    - GBM minus that baseline: clustered 95% CIs exclude 0 in every run, with
      lower bounds of +0.07 to +0.13.
    - Price-feature increment: point estimates −0.03 to +0.05. The CI excludes
      0 in 1 of 10 runs, and only just (2h, any label, no filter:
      +0.04 [+0.003, +0.084]).
    - Read: price features add little or nothing detectable beyond machine
      features. The earlier "entirely" was too strong as a statement but holds
      as a finding.
  - The pre-registered 24h test, re-run on the full data with the new
    threshold, still finds 0 listings with a complete window.

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
  *(This placebo was too wide; with a matched-n placebo the exploratory
  correlation clears the threshold. See the 2026-11-09 correction below.)*
- **Test window**: 2026-10-19 00:00 → 2026-10-26 00:00 UTC, same script, PRL
  hourly prices from CoinGecko (`pearl-2`).
- **Supported if** the consumer correlation exceeds the placebo 95th
  percentile **and** the (consumer − datacenter) correlation is positive.
  Otherwise we report the confound as not supported in that window.
- Either way the rentability score result above is reported with this check
  next to it.
- **Result (run 2026-10-28, same script, unchanged):** 111 snapshots in the
  window (collector gaps: 7 on 10-22, none on 10-24, 8 on 10-25), n = 53
  consecutive-hour changes. Consumer r = +0.16, datacenter r = +0.08,
  (consumer − datacenter) r = +0.007. Placebo (119 shifts): 95th percentile
  +0.37, share ≥ real 0.27. **Confound not supported in this window.** With
  n = 53 this only says the exploratory +0.31 did not replicate: the 95%
  interval for the consumer correlation (Fisher z) is −0.11 to +0.41, so a
  true correlation as high as about 0.4 is still consistent with the data.
  (The +0.37 above is the placebo threshold for significance, not a bound on
  the correlation; an earlier version of this paragraph used it as one.) PRL prices: CoinGecko
  `pearl-2`, `days=10`, fetched 2026-10-28 (md5 ac3a814e…).
- **Correction (2026-11-09): the placebo in `prl_confound.py` was too wide.**
  The script shifts the hourly PRL series *with its collector gaps*, so each
  shifted series loses pairs to NaN: 5–24 pairs per shift (median 17) in the
  test window, against n = 53 for the real correlation; 12–36 (median 24)
  against n = 58 in the exploratory window. A null built on ~17 pairs is far
  wider than one built on 53. Found by an adversarial code audit; the
  pre-registered script is left unchanged.
  `analysis/prl_placebo_matched.py` re-runs the whole pipeline with the PRL
  timestamps shifted by k hours instead, so every shift uses the same pairs and
  the same price alignment as the real statistic. Its k = 0 run reproduces the
  test-window statistic above (111 snapshots, n = 53, r = +0.162), and a
  re-run of the original placebo on the same data gives 95th percentile
  +0.368, share 0.269, as reported. So the re-fetched prices match the ones
  used on 10-28.
  - **Test window, verdict unchanged:** consumer r = +0.16. Matched placebo
    95th percentile +0.20 (119 forward shifts) / +0.23 (119 backward); share ≥
    real 0.13 / 0.17. **Confound not supported.** The +0.37 threshold above is
    wrong; the margin is much thinner than it suggested.
  - **Exploratory window, conclusion reversed:** the "not significant" above
    does not hold. The original run (n = 55) cannot be reproduced exactly
    (it ran before 10-18 ended); the two bracketing cut-offs give n = 50,
    r = +0.29 (end 10-18 00:00) and n = 58, r = +0.28 (end 10-19 00:00).
    Against forward shifts both sit above the matched 95th percentile (+0.23,
    share ≥ real 0.04); backward shifts are limited to 25 by price coverage.
    Read this as "the exploratory correlation was not ruled out by the
    placebo", not as evidence of a confound: neighbouring shifts are
    strongly correlated, so a share of 0.04 from ~100 shifts is fragile, and
    this placebo design was chosen after the audit.
  - The test window, which is the pre-registered check, still does not
    support the confound. What changes is that the exploratory signal was
    real by the original rule, so "did not replicate" is now the substantive
    finding rather than "was never there".
  - The per-machine collapse mixes the rentable state of one bundle with the
    specs of another (`groupby.last` vs `max`). Both runs above keep that
    collapse as is. Here it only affects which GPU model a machine is assigned
    to (see the first-pass correction above).
  - PRL prices for the re-run: CoinGecko `pearl-2`, `days=30`, fetched
    2026-11-09 (md5 250737dc…). Hourly points; every snapshot matched within
    90 minutes.

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
