# HMM framework tenth-wave exploratory plan

Date frozen: 2026-08-01, before tenth-wave results. Prior samples have been
observed; results are exploratory and require forward shadow even if every shared
gate passes.

## Direction T: model-disagreement risk reliability

Use the ninth-wave 756/1008/1512-session, three-seed member set. Before netted
account execution, measure each member's total QQQ plus SMH exposure. Let `m` be
the current sleeve-weighted mean exposure and `v` its sleeve-weighted cross-model
variance. Multiply the aggregate QQQ and SMH weights by `m^2 / (m^2 + v)` and move
the released account weight to CASH. If mean exposure is zero, leave the target
unchanged. Do not alter the QQQ/SMH internal ratio, other assets, state fits, member
targets, leverage, or any R38/R39 rule.

The baseline remains the existing three-seed 1008-session ensemble, not the failed
uncontrolled multi-window candidate.

Rationale: cross-model disagreement is estimation risk. The reliability ratio is
one when models agree and falls continuously without a fitted threshold or a
post-hoc choice of a winning window.

Use the unchanged third-wave R39 account gates: nonnegative CAGR delta in normal
development and holdout, proxy early/late/complete, and complete double-cost views;
normal maximum drawdown within 50 basis points; and no increase in average
per-member risk-state switches. A pass remains research-only and requires annual,
rolling, event, and disagreement-attribution audits.
