# HMM framework eleventh-wave exploratory plan

Date frozen: 2026-08-01, before eleventh-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction U: robust state-conditioned return location

Keep the baseline Gaussian HMM, standard feature scaling, smoothed state
responsibilities, state selection, windows, templates, and all portfolio rules.
Within each state and asset, replace the posterior-weighted arithmetic return mean
with a posterior-weighted Huber M-location. Initialize at the weighted median, use a
weighted MAD scale (falling back to weighted standard deviation only when MAD is
zero), and fix the conventional 95%-normal-efficiency Huber constant at 1.345.
Estimate the state covariance around the same robust center.

Rationale: prior Student-t work robustified feature emissions, not the separate
state-to-next-return mapping that directly determines the sign of HMM growth excess
return. One extreme asset return should not dominate that mapping.

Use the unchanged third-wave R39 account gates: nonnegative CAGR delta in normal
development and holdout, proxy early/late/complete, and complete double-cost views;
normal maximum drawdown within 50 basis points; and no increase in risk-state
switches. A pass remains research-only and requires annual, rolling, event, and
outlier-attribution audits.
