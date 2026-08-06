# HMM framework sixth-wave exploratory plan

Date frozen: 2026-08-01, before sixth-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction M: update templates only on a new HMM fit

The production baseline refits HMM parameters every 21 anchored business days,
but calls the template mapper on every signal evaluation. Test a single structural
change: keep mapping every time, but move template parameters only when a new HMM
fit was produced. The smoothing coefficient, assignment algorithm, model, features,
and portfolio rules remain unchanged.

Rationale: template smoothing should count independent parameter estimates, not
repeatedly compound the same fitted state distribution.

## Direction N: Wasserstein-consistent diagonal variance update

Keep the baseline update schedule, but replace linear interpolation of feature
variances with the diagonal Gaussian Wasserstein-2 barycenter: interpolate feature
standard deviations and square the result. Means, return moments, smoothing,
assignment, model, features, and portfolio rules remain unchanged.

Rationale: template identity is measured with Gaussian Wasserstein-2 distance, so
its update geometry should use the same metric.

Both directions use the unchanged third-wave R39 account gates: nonnegative CAGR
delta in normal development and holdout, proxy early/late/complete, and complete
double-cost views; normal maximum drawdown within 50 basis points; and no increase
in dominant-template switches. No result changes production, orders, or deployment.
