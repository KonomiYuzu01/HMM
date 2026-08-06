# HMM framework twelfth-wave exploratory plan

Date frozen: 2026-08-01, before twelfth-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction V: tied full feature covariance

Replace the Gaussian HMM's state-specific diagonal emission covariances with one
full covariance matrix shared by all states. Keep standard scaling, candidate state
counts, windows, refit/selection cadence, restarts, return moments, templates,
portfolio rules, R38/R39, and costs unchanged. Update the order-selection complexity
penalty to count the shared `d(d+1)/2` covariance parameters. Template matching uses
the tied covariance's marginal variances because the existing Wasserstein tracker is
explicitly diagonal.

Rationale: cross-asset co-movement is economically meaningful, while sharing the
correlation matrix across states is more parsimonious than a separate full matrix
per state. The test asks whether correlation information adds robust value without
confounding it with state-specific covariance proliferation.

Use the unchanged third-wave R39 account gates. A pass remains research-only and
requires the same annual, rolling, event, numerical, and seed audits.
