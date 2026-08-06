# HMM framework directions: frozen validation plan

Date frozen: 2026-07-31, before generating any candidate result.

Scope: research only. No production configuration, production snapshot, order,
or deployment may consume these artifacts.

## Shared baseline

- Current R9 three-seed ensemble and its exact data, costs, window, state-order
  selection, Wasserstein templates, portfolio rules, and execution schedule.
- Current R38 and R39 remain the economic comparison benchmark.
- Development: 2015-2021. Holdout: 2022-2025. Post-holdout 2026 is reported
  separately and cannot rescue a failed holdout.
- Normal and long proxy histories are both required where compatible inputs
  exist. Current and 2x transaction costs are reported.

## Direction 1: white-box diagnostics

Add only: normalized posterior entropy, top-two probability margin, one-step
predictive log likelihood, expected nearest-template Wasserstein distance,
template-distance margin, seed vote agreement, and current state duration.
No target weight may change. The baseline passes only if the rerun reproduces
the prior Gaussian return and weight paths to numerical tolerance.

## Direction 2: uncertainty-gated incremental R38 capacity

Keep the qualified R38 1.30 relative multiplier as the floor and 1.375 as the
ceiling. On stable-volatility dates only, use:

`relative multiplier = 1.30 + 0.075 * confidence`

where confidence is the product of the mean top-two posterior margin and the
absolute three-seed risk-on vote margin. No threshold is searched. Existing
trend, one-session shock, cash, SMH overlay, and R39 rules remain unchanged.
The candidate must improve or preserve CAGR in development, holdout, complete,
and long proxy samples; maximum drawdown may not worsen by more than 25 bp
versus R38; 2x-cost CAGR must not deteriorate.

## Direction 3: Student-t emissions

Replace only the diagonal Gaussian emission with a diagonal multivariate
Student-t emission using a fixed five degrees of freedom and its correct latent
scale EM update. Everything downstream stays identical. No neighboring degree
of freedom is searched. Required evidence: Gaussian identity rerun, finite and
causal diagnostics, development and holdout CAGR nonnegative versus Gaussian,
maximum drawdown not worse by more than 50 bp, and fewer extreme negative
one-step predictive log-likelihood observations.

## Direction 4: explicit-duration shadow

Do not alter production actions. Estimate, causally and separately by member,
the empirical survival hazard of the current risk-on/risk-off episode using
only completed prior episodes. Compare its next-session persistence Brier score
with a first-order transition baseline in development and holdout. Duration is
useful only if it improves both periods and does not rely on fewer than 30
mature episodes per class.

## Direction 5: semiconductor residual state

Fit a separate two-state, three-seed causal Gaussian HMM to lagged SMH-minus-QQQ
relative returns and relative volatility. It may predict only the next 21-day
SMH-minus-QQQ outcome and may not change total growth exposure. Compare against
the existing 21-day R39 relative-damage score and 126-day R38 relative-momentum
score. Require holdout Brier or rank improvement, positive block-bootstrap
relative return evidence, and no deterioration in development. Otherwise reject
the extra HMM and retain R39.

## Multiple testing and production rule

These are five declared structural trials. A direction that misses any frozen
gate is rejected. Passing historical gates produces at most a forward-shadow
candidate; it does not authorize a production configuration.
