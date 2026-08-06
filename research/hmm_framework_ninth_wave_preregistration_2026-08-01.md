# HMM framework ninth-wave exploratory plan

Date frozen: 2026-08-01, before ninth-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction S: multi-lookback HMM ensemble

Run the unchanged selected-order Gaussian HMM for every combination of the existing
three random seeds and 756-, 1008-, and 1512-session rolling windows (approximately
3, 4, and 6 trading years). Combine all nine model sleeves with the existing
equal-sleeve, drift-aware, net-member-trade ensemble. When fewer than the requested
sessions exist, use all causal history available, subject to the existing minimum
sample requirement.

The baseline is the existing three-seed 1008-session ensemble. Features, state
candidates, refit and order-selection schedules, two restarts, template logic,
portfolio rules, R38/R39 overlays, and costs remain unchanged.

Rationale: a single rolling window imposes one bias-variance and regime-memory
choice. Averaging structurally different but causal windows can expose model
disagreement and reduce dependence on one historical cutoff without selecting the
best window after seeing returns.

Use the unchanged third-wave R39 account gates: nonnegative CAGR delta in normal
development and holdout, proxy early/late/complete, and complete double-cost views;
normal maximum drawdown within 50 basis points; and no increase in dominant-state
switches. A pass remains research-only and requires the eighth-wave annual,
rolling, event, and seed/window attribution audits before any forward shadow.
