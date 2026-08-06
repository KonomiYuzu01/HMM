# HMM framework eighth-wave exploratory plan

Date frozen: 2026-08-01, before eighth-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction Q: robust feature scaling

Replace per-feature mean/standard-deviation scaling with median/IQR scaling in both
order selection and final fitting. Do not clip observations. HMM distribution,
windows, state candidates, templates, return moments, and portfolio rules remain
unchanged.

Rationale: one extreme feature observation should not rescale every ordinary
observation in the rolling window. Median/IQR scaling tests this independently from
the rejected Student-t emission and quantile clipping candidates.

## Direction R: empirical-Bayes state return means

Keep HMM fitting and posterior assignments unchanged. For each asset, shrink each
state return mean toward the unconditional training mean using a method-of-moments
between-state variance and the state's posterior effective sample size. There is no
hand-tuned shrinkage coefficient. State covariances and all portfolio rules remain
unchanged.

Rationale: the binary R39 precursor signal uses the sign of raw state-conditioned
growth excess return before the optimizer's ordinary mean shrinkage. Sparse states
therefore need estimation-risk control at the state-to-return mapping itself.

Both directions use the unchanged third-wave R39 account gates: nonnegative CAGR
delta in normal development and holdout, proxy early/late/complete, and complete
double-cost views; normal maximum drawdown within 50 basis points; and no increase
in dominant-template switches. No result changes production, orders, or deployment.
