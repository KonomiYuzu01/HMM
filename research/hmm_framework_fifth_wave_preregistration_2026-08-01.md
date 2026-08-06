# HMM framework fifth-wave exploratory plan

Date frozen: 2026-08-01, before fifth-wave results. Prior samples have been observed;
results are exploratory and require forward shadow even if all shared gates pass.

## Direction K: event-driven 63-day refit

Use the fixed 63 anchored-business-day refit cadence. At each scheduled HMM
evaluation, also refit once when the causal QQQ/SMH 21-versus-63-day volatility
state differs from the prior evaluation. Do not refit repeatedly while the same
stress state persists. Features, order selection, and all portfolio rules remain
unchanged.

## Direction L: annual model-order selection

Keep parameter refitting every 21 days but change predictive order selection from
every 63 to every 252 anchored business days. Candidate orders, validation window,
and all other settings remain unchanged. This isolates discrete model-dimension
instability from ordinary parameter learning.

Both directions use the third-wave account gates without modification.
