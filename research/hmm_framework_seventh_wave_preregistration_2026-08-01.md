# HMM framework seventh-wave exploratory plan

Date frozen: 2026-08-01, before seventh-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction O: filtered responsibilities for state return moments

Keep the fitted HMM and live filtering rule unchanged. Estimate each state's return
mean and covariance using one-sided forward-filtered responsibilities instead of
the baseline full-sample smoothed responsibilities. All features, state selection,
template logic, and portfolio rules remain unchanged.

Rationale: the state-to-return mapping should use the same information semantics as
the live filtered posterior. A smoothed historical assignment can use observations
after a historical return to relabel that return, even though no future data enters
the current rolling training window.

## Direction P: condition validation likelihood on the training endpoint

Keep the order candidates and penalty unchanged. Replace the validation likelihood
computed from a fresh stationary start with the conditional block likelihood
`log p(train + validation) - log p(train)`, divided by validation length. Model fits,
features, return moments, templates, and portfolio rules remain unchanged.

Rationale: a contiguous time-series validation block begins in the state
distribution implied by the end of training, not an unrelated stationary reset.

Both directions use the unchanged third-wave R39 account gates: nonnegative CAGR
delta in normal development and holdout, proxy early/late/complete, and complete
double-cost views; normal maximum drawdown within 50 basis points; and no increase
in dominant-template switches. No result changes production, orders, or deployment.
