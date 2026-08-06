# HMM framework second-wave frozen validation plan

Date frozen: 2026-07-31, before generating second-wave results.

Scope: research only. No production configuration, order, snapshot, or deployment
may consume these artifacts.

## Shared design

- Development: 2015-2021. Holdout: 2022-2025. The 2026 partial year is reported
  separately and cannot rescue a failed holdout.
- Long proxy early and late periods are required where compatible inputs exist.
- Current R38/R39 is the economic baseline. “R9” denotes only the underlying
  three-seed HMM output.
- No threshold or neighboring parameter is searched after viewing results.

## Direction A: predictive value of model OOD diagnostics

At each scheduled HMM evaluation, use the causal percentile of the ensemble mean
negative one-step predictive log likelihood and the causal percentile of expected
nearest-template distance. Predict the next 21 sessions' equal-weight QQQ/SMH loss
and 5% peak-to-trough decline. Compare rank/AUC against fixed existing fast-risk
benchmarks: lagged 21/63-day volatility acceleration, lagged 21-day growth weakness,
and the prior one-session growth loss.

An OOD score is action-worthy only if its severe-decline AUC is at least the best
existing benchmark in normal development and holdout plus proxy early and late
periods. Otherwise no OOD allocation rule is tested.

## Direction B: posterior deterioration velocity

Use the negative one-evaluation and four-evaluation change in the three-seed mean
smoothed favorable probability. The same outcomes, benchmarks, periods, and gates
as Direction A apply. Probability level is not added because current actions
already consume it. Failure at the predictive gate stops the direction before any
portfolio mapping.

### Data-contract amendment before Direction B result generation

The current R9 configuration does not enable continuous probability allocation,
so `smoothed_favorable_probability` is structurally empty. The first execution
therefore produced zero Direction B observations and is not a result. Before any
valid Direction B metric was observed, the source was replaced with a causal
template risk-on probability: at each evaluation, every template's risk-on rate is
estimated with a Beta(1,1) prior and only prior evaluations, current template
probabilities are weighted by those prior rates, and the three members are then
averaged. The frozen one- and four-evaluation changes and all gates remain unchanged.

## Direction C: model-order averaging

Replace predictive order selection only with equal-weight predictive moments and
template probabilities from fixed 2-, 3-, and 4-state Gaussian HMMs. Keep features,
1008-day window, three seeds, Wasserstein templates, portfolio logic, R38/R39 rules,
costs, and execution unchanged.

Required gates versus the current selected-order Gaussian baseline:

- nonnegative CAGR delta in normal development and holdout;
- normal maximum drawdown not worse by more than 50 bp;
- nonnegative CAGR delta in long proxy early, late, and complete periods;
- nonnegative complete-period CAGR delta under doubled transaction costs;
- no increase in scheduled risk-on state switches.

Passing all historical gates creates at most a forward-shadow candidate.

## Direction D: fixed sticky transition prior

After Direction C was fully rejected, test a separate fixed Gaussian Sticky HMM.
Add 10 pseudo-observations to every diagonal transition and one to every transition
as the existing neutral prior. Do not search the sticky strength and do not combine
it with Student-t emissions or model-order averaging. Everything else, including
predictive order selection, remains unchanged.

Required gates are identical to Direction C: nonnegative normal development and
holdout CAGR, normal drawdown within 50 bp, nonnegative proxy early/late/complete
CAGR, nonnegative doubled-cost complete CAGR, and no increase in scheduled risk-on
state switches. Passing remains forward-shadow only.
