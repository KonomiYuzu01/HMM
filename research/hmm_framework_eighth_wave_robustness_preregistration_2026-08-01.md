# Eighth-wave robustness audit gates

Frozen before reading annual, rolling-window, or event results on 2026-08-01.

A candidate that passed the shared account gates must also satisfy all of:

1. Proxy complete-period maximum-drawdown delta is at least -0.005.
2. Median 756-session rolling CAGR delta is nonnegative and at least half of sampled
   rolling windows have nonnegative CAGR delta. Windows advance by 21 sessions.
3. Median calendar-year CAGR delta is nonnegative and at least half of eligible
   calendar years have nonnegative CAGR delta.
4. In each prespecified stress event, maximum-drawdown delta is at least -0.005, and
   the median event CAGR delta is nonnegative.

Stress events are fixed as: GFC (2007-10-09 through 2009-03-09), euro-area stress
(2011-04-29 through 2011-10-03), 2018 Q4 (2018-09-20 through 2018-12-24), COVID
(2020-02-19 through 2020-03-23), and 2022 tightening (2022-01-03 through
2022-10-14). Worst rolling CAGR delta is reported but has no post-hoc hard threshold.

Passing remains research-only and requires forward shadow; it does not authorize a
production configuration, orders, or deployment.
