# HMM framework third-wave frozen validation plan

Date frozen: 2026-08-01, before generating third-wave candidate results.

Scope: research only. Production configurations, snapshots, orders, and deployment
are excluded.

## Shared baseline and gates

The baseline is the current selected-order, three-seed Gaussian HMM passed through
the unchanged R38/R39 account simulation. Development is 2015-2021, holdout is
2022-2025, and the long proxy is split into 2006-2014 and 2015-2025. Every candidate
must satisfy all of the following:

- nonnegative CAGR delta in normal development and holdout;
- normal complete maximum drawdown no worse by more than 50 bp;
- nonnegative CAGR delta in proxy early, late, and complete periods;
- nonnegative normal complete CAGR delta with doubled trading costs;
- total scheduled risk-on state switches no higher than baseline.

No neighboring setting is searched. Passing creates only a forward-shadow candidate.

## Direction E: macro/industry input separation

Remove QQQ and SMH from HMM regime inputs, leaving SPX, IEF, GLD, DBC, and UUP.
QQQ/SMH remain investable outcomes and remain fully controlled by existing R38/R39
relative trend, volatility acceleration, gap, and damage rules. This tests whether
the macro state model benefits from not double-counting industry-specific shocks.

## Direction F: slow-state feature separation

Remove one-session return features from the HMM and retain only 60-day volatility
and 20-day mean-return features. Existing daily and one-session fast-risk rules are
unchanged. This tests whether a weekly strategic state model should avoid competing
with the fast layer for the same shock information.

## Direction G: slower parameter refit

Change only HMM refitting from every 21 to every 63 business days, aligned with the
existing order-selection cadence. Filtering and portfolio evaluation remain on the
existing schedule. This tests estimation noise rather than state persistence.

## Direction H: causal feature clipping

Clip every training feature at its own 0.5% and 99.5% training quantiles before
standardization. Apply the same training bounds to validation and current features.
Bounds are recomputed only from the causal training window. Keep Gaussian emissions
and every downstream rule unchanged. This is a fixed, milder robustness treatment
than the rejected Student-t candidate.
