# Robust-scaler numerical-equivalence audit

Frozen on 2026-08-01 before five-restart results.

Standard and median/IQR scalers are both feature-wise affine transformations. A
fully optimized diagonal Gaussian HMM with freely estimated per-feature means and
variances should therefore be approximately invariant, apart from initialization,
the covariance floor, and numerical convergence. The two-restart robust candidate
also emitted more small negative likelihood deltas.

Run both the standard and robust scalers with five restarts, holding seeds, windows,
state candidates, penalties, templates, portfolio rules, R39 overlay, and costs
fixed. Compare robust-five against standard-five using the same normal development
and holdout, proxy early/late/complete, double-cost, 50-basis-point drawdown, and
state-switch gates. If the robust candidate fails any gate, or if its advantage
collapses to path identity, classify the earlier result as optimizer-path evidence
rather than a robust-scaling improvement. Passing remains research-only and still
requires forward shadow.
