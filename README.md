# Executable Wasserstein-HMM Cross-Asset Strategy

This project translates the core ideas of Boukardagha (2026), “Explainable Regime-Aware Investing,” into a strictly causal, backtestable, and executable ETF allocation strategy. It seeks better risk-adjusted returns and lower drawdowns, but **does not guarantee a high Sharpe ratio or future profits**.

The current production configuration is
`config/paper_core_growth_gold20_r11_diversified_financing.yaml` (R11).
Strategy-managed capital initially runs at 25% R11 and 75% R9. R11 retains R9's market-regime assessment, risk
scaling, recovery positions, and conditional VIXY hedge; full R11 uniformly multiplies every non-cash target by 1.065,
with cash absorbing the difference, while retaining the causal SMH overnight hedge. It expresses at most 10% of the
scaled gold target through GDE; GDE remains part of the gold sleeve, not the growth allocation. If the Panel has not
obtained actual holdings, or has not obtained complete closing data for the previous trading day before the next market
open, the orders are not executable and the reference weights must not be treated as brokerage orders.

## 2026-07-27 Staged Rollout of the R11 Diversified-Financing Risk Budget

R11's primary improvement comes from modestly scaling the entire non-cash portfolio rather than relying on a large
GDE allocation. Because only 25% of capital currently uses R11, the effective non-cash target multiplier at the account
level is `75% × 1 + 25% × 1.065 = 1.01625`, only 1.625% above R9. The current central targets are
QQQ 43.5752%, SMH 19.8220%, GLD 22.0215%, GDE 0.4898%, and cash/BIL 14.0915%. The permitted GDE range is
0%–0.9898%; an account with no current GDE position is already within the range and need not trade merely to move
closer to the central target.

Under the current cost assumptions, the 25% R11 allocation increases compound annual growth rates over pure R9 by
0.6196, 0.3117, and 0.7208 percentage points in the 2015-to-present, long-term proxy-data-since-2006, and
post-GDE-listing actual-data windows, respectively; historical maximum drawdowns are -15.58%, -16.83%, and -15.58%,
respectively. When standard transaction costs, GDE costs, and the financing spread all increase simultaneously, the
annualized advantages across the three windows remain 0.4501, 0.2156, and 0.5548 percentage points, with maximum
drawdowns of -16.80%, -17.13%, and -15.77%.

R11 ranks No. 1 among all 1,698 SMH hedge-parameter paths on the standard dataset and all 1,808 paths on the long-term
proxy dataset. Under the three 21-, 63-, and 126-day correlation assumptions, all six multiple-comparison-adjusted
probabilities are 3.74%–4.82%, all below 5%. Across three independent model IDs, both the return uplift and the
risk-adjusted-return uplift are positive under current and stressed costs; the minimum advantage also remains positive
after dropping one year at a time and one hedge event at a time. In 5,000 GDE tracking-error simulations, the 25%,
50%, and 100% rollout levels all satisfy the three objectives of positive returns, risk-adjusted returns no worse than
R9, and historical drawdowns no greater than 18%.

The rigorous testing here covers only the causal SMH hedge-parameter family and the registered R11 candidates; it is
not a unified test of every idea explored in the project's history. The initial 25% rollout is intended to prospectively
validate actual financing rates, execution slippage, market-data timeliness, and the holdings workflow. Expansion to
50% or 100% will not be reviewed until at least 63 post-freeze trading days have been recorded. Expansion is prohibited
if the realized financing spread exceeds 1.50% per year, one-way GDE execution costs exceed 0.75%, or any data-integrity
issue occurs.

Reproduce the current targets and final audit:

```bash
PYTHONPATH=src:.:tools .venv/bin/python tools/refresh_r11_production.py
PYTHONPATH=src:.:tools .venv/bin/python tools/build_r11_production_overlay.py
PYTHONPATH=src:.:tools .venv/bin/python tools/evaluate_r11_final_candidate_audit.py
PYTHONPATH=src:.:tools .venv/bin/python tools/evaluate_r11_final_candidate_robustness.py
```

Use the first command as the single entry point for routine production refreshes, scheduled after 16:15 US Eastern
Time. This workflow refreshes the core assets, GDE, and audited individual stocks; recalculates the R9, R11, and
individual-stock market states; and then automatically exports the Panel snapshot. The program refuses to generate
production signals while the US equity market is open and requires the core, GDE, R9, R11, and individual-stock data
to be aligned to the most recently completed trading-day close. It also fails if the latest five trading days for the
core assets and GDE are not consecutive. If a proxy is used for a missing value in an individual non-core asset, the
date and method must be recorded in the market-data metadata. The missing VIXY value on 2026-07-27 currently uses the
previous close as a flat-price proxy; this proxy is not used for QQQ, SMH, GLD, cash, or the market-regime assessment.

When users select their own growth or semiconductor stocks, the default production overlay configuration is
`config/r9_individual_stock_overlay_v2.yaml`. It does not alter the R9 core or select stocks automatically. Individual
stocks may replace QQQ/SMH only when the market is in a calm/normal uptrend, the selected basket has outperformed its
corresponding ETF over both the past 63 and 126 days, and the 63-day dispersion of stock returns relative to the ETF is
not in the top third of the past three years. Positions are then constrained by three caps: 60-/252-day differential
risk, 5% per stock, and 20% in aggregate. v1 is retained as a high-capture mode that requires supporting records of
actual stock selection. Final validation is available at
`output/individual_stock_regime_final_validation/validation_summary.md`.

## 2026-07-27 Third-Round Validation of the SMH Intraweek Hedge

The R9 production configuration remains unchanged. The intermediate, fully causal “extreme gap on the prior day,
de-risk at the next open” version increased annualized returns by approximately 0.93 percentage points in the
2015-to-present backtest, with an upside capture ratio of approximately 99.65%. However, the family-level
selection-bias-adjusted probability across 2,240 parameter sets was approximately 47%–50%, failing the production
evidence threshold. The newly added actual overnight proxy since 2002 also shows that the unfiltered version made a
negative contribution during the only early event, in 2011. Adding a filter for semiconductor weakness relative to
QQQ reduced the early period to zero triggers, which cannot count as positive independent validation. Historical tail-risk
budgets, rotation into QQQ, and 1–2-day rapid recovery all failed as well. Therefore, both same-open hedging and the
next-day causal rule remain shadow/research paths only and do not generate production orders. The complete rules, data
hashes, cost stresses, and rejected directions are documented in
`research/smh_intraweek_risk_optimization_round3_2026-07-27.md`.

## 2026-07-27 Revalidation of the Individual-Stock/ETF Switching Layer

Four types of improvements were tested further: continuously sizing individual-stock positions by relative strength;
calculating dispersion separately for QQQ and semiconductors; using the higher dispersion of the two groups as a
system-wide warning; and individual-stock no-trade bands of 0.25%–1.5%. No candidate improved the worst 10% of baskets
in both the 2015–2021 development period and the recent 2022–2025 period. The production configuration therefore
continues to use 63-/126-day relative-strength confirmation and a hard 67% dispersion threshold across the full stock
universe, with no trading changes from this research round.

As of 2026-07-24, dispersion in the QQQ growth-stock group was at the 56.75th percentile of the past approximately
three years, while the semiconductor group was at the 87.96th percentile. However, historically, “separate clearance”
worsened the recent tail and maximum drawdown, so these two figures are diagnostic only and cannot replace the current
global gate. The complete audit is available at
`output/individual_stock_regime_optimization_20260727/optimization_summary.md`.

## 2026-07-26 Release of the R9 Individual-Stock/ETF Regime-Switching Layer

The new default mode separates “whether the market is suitable for taking growth risk” from “whether taking
stock-specific risk is worthwhile.” Across 40 random baskets, median annualized relative log return for 2022–2025
improved from -0.42% under the old rule to +0.06%, while the bottom 10% improved from -1.38% to -0.18%. For
2015–2025, median CAGR was 19.57%, the bottom 10% was 19.42%, and median maximum drawdown was -16.31%. The 21-day
block-bootstrap interval still spans zero, so this is a default error-prevention and expression-control mechanism, not
a promise of additional future returns. Current stock dispersion is around the 87th percentile of the past three
years, so the evidence-gated mode uses ETFs. The next month-end review is after the 2026-07-31 close.

## 2026-07-26 Release of the R9 Controlled Individual-Stock Substitution Layer

The old Panel reduced each ETF position using a “risk multiplier” based on the individual stock's total volatility
relative to the ETF, leaving unused risk capacity in cash. This double-reduced the market risk that R9 was already
designed to take, resulting in a median 2015–2025 CAGR of only 16.38% across 40 random stock baskets. The new version
instead controls the differential volatility of stock-basket returns minus the corresponding ETF returns and makes
dollar-for-dollar substitutions for the ETF without retaining additional cash.

Across 40 random baskets, the final rule produced a median CAGR of 19.78% and a bottom-10% result of 18.84%; median
maximum drawdown was -16.36%, and the bottom-10% result was -17.46%. These results validate only the risk framework,
not automated stock selection. During 2022–2025, the median random basket required approximately 4.22% in additional
annualized stock-selection return to match pure R9, while the bottom-10% basket required approximately 14.74%.
Therefore, this overlay is appropriate only if the user genuinely has a stock-selection edge and must not be described
as unconditionally increasing returns.

## 2026-07-25 Release of R9 Staged Re-entry

R9 changes only during the narrow interval when the growth allocation is already near zero, the market is beginning
to recover, and the main model remains defensive. It initially establishes only a 20% QQQ position; only after the
recovery signal has persisted for ten consecutive trading days may it increase to 35% QQQ. This separates the costs
of two kinds of error: first reducing the cost of completely missing a rapid rebound, then using time confirmation to
limit losses during false rebounds.

Following the 2026-07-25 data audit, VIX and VIX3M now use official Cboe history day by day, and rebalance and model-
update dates now use a fixed business-day sequence independent of the data start date. On the corrected common
schedule, CAGR for 2015–2025 is 19.40%, versus 18.85% for R8; maximum drawdown is -16.57%, versus -17.52% for R8.
On the long-term proxy data for 2006–2025, CAGR is 14.56%, versus 14.07% for R8; maximum drawdown is -17.65%, versus
-19.52% for R8. Full sources, file fingerprints, official reconciliations, before/after isolation experiments, and
remaining limitations are documented in `output/r9_data_audit_2026-07-25/数据审计报告.md`.

With execution at the next trading day's open and a cost of 0.15% per unit of rebalanced notional, the corrected R9
CAGR is 17.63%, versus 17.26% for R8. There are still brief rolling 3-year windows of underperformance versus R8, with the
worst annualized difference approximately -1.45 percentage points. Since the scheduling fix, the multiple-comparison
statistical test covering all historical candidates has not yet been rerun, so the corresponding conclusion from the
old version is no longer considered current evidence.

## 2026-07-25 Staged Production Rollout of R8

In the frozen 2015–2025 window, R8 achieved a CAGR of 19.05%, a Sharpe ratio of 1.241, and a maximum drawdown of
-14.23%, versus 16.69%, 1.149, and -13.78%, respectively, for the prior production strategy. At a 15 bps cost, CAGR
was 17.57% versus 15.46%; for 2012–2025, it was 17.38% versus 14.89%; and for the 2006–2025 proxy, it was 13.56%
versus 12.35%. The highest selection-adjusted p-value across the family of 37 candidates was 3.88%. The CAGR advantage
remained +1.26 percentage points after a one-day execution delay, and the incremental results remained positive in
both the standard and proxy samples after excluding the best re-entry event. The 20-year proxy retained a +1.03
percentage-point CAGR advantage at 15 bps, but its MDD was -18.69%, so 18% is not guaranteed under high costs or in
the future. The rolling three-year outperformance rate was 82.7%, with worst annualized relative return of -2.87%;
2008–09 and 2011 are known intervals of relative underperformance.

The current one-way migration from the prior production strategy to R8 is 35.54%, so a complete one-time switch is
not permitted. On each of four consecutive trading days, move 25% of the way toward that day's recalculated target.
One-way model turnover per tranche is approximately 8.88%, and the assumed total explicit cost at 15 bps is
approximately 5.33 bp of NAV. In historical replays from arbitrary launch dates, the 5th-percentile opportunity cost
of the active path was approximately -0.48%, with a worst case of approximately -2.78%. This shows that staging reduces
operational impact but is not free market timing. Official Cboe VIX3M is now used to fill Yahoo gaps exactly by date;
the data source, number of filled values, and final count of remaining missing values are recorded in every
`run_metadata.json`. Complete admission criteria, migration details, and immutable hashes are available at
`output/forward_monitoring/freeze_manifest_2026-07-25-r8-v3.json`.

## 2026-07-23 Production Replacement

The new strategy replaced the old strategy according to pre-specified thresholds rather than being deployed simply
for having the highest CAGR. In the executable-at-open proxy for 2015–2025 (7.5 bps), CAGR was 16.60%, Sharpe ratio
was 1.14, and maximum drawdown was -13.40%, versus 14.56%, 1.10, and -16.96%, respectively, for the old strategy.
CAGR improved by +2.15 pp in the development period, +1.83 pp in the 2022–2025 holdout period, +1.56 pp at a 15 bps
cost, and +1.63 pp from the extended 2012 start date. The family-wise Reality Check p-value among 18 eligible
candidates was 6.34%, passing the 10% threshold. Under 21-/63-/126-day block resampling, the frequencies of breaching
an 18% drawdown were 62.8%/44.7%/17.0%, still showing that 18% is not a guarantee for the future.

The production rules use only a small number of discrete settings with economic meaning: 6-month industry-relative
momentum, 12-month dual-trend scaling, 20-/60-day stressed volatility, maximum exposure of 1.10×, and 4% VIXY when the
term structure is inverted. Ratios and lookback periods were not further fine-tuned around the passing candidates.
New executions from 2026-07-23 onward must be treated as a prospective record and may not be used to revise this
version retroactively.

The project provides multiple frozen configurations with different objectives:

- `config/default.yaml`: prioritizes low volatility and low drawdown.
- `config/cagr_first.yaml`: prioritizes CAGR, with a fixed 80% equal-weighted QQQ/SMH core and a 20% Wasserstein-HMM cross-asset satellite, without leverage.
- `config/paper_core_growth.yaml`: HMM conditional returns control the QQQ/SMH growth core; defensive assets are held at other times.
- `config/paper_core_dd18.yaml`: mechanically raises the forecast-volatility target of the paper-core version to 22% while leaving all other parameters unchanged, to test the outcome after relaxing the drawdown budget to 18%.
- `config/paper_core_sticky.yaml`: a turnover experiment with immediate risk exits and re-entry only on HMM re-estimation dates; eliminated for exceeding the drawdown limit.
- `config/paper_core_hmm_only.yaml`: a pure-HMM experiment with the additional trend filter removed; eliminated for exceeding the drawdown limit.
- `config/paper_core_risk_parity.yaml`: allocates QQQ/SMH by inverse volatility using shrunk HMM conditional volatility estimates; produced no material improvement.
- `config/paper_core_macro_equal.yaml`: uses only the paper's original macro-asset definition to determine HMM states, with fixed equal weights in QQQ/SMH; eliminated for exceeding the drawdown limit.
- `config/paper_core_macro_risk_parity.yaml`: a combined experiment using the macro HMM and inverse-volatility QQQ/SMH allocation; eliminated for exceeding the drawdown limit.
- `config/paper_core_smh_satellite.yaml`: a 20% SMH long-trend/volatility-targeting satellite; improved the holdout period but did not meet the pre-specified complete-calendar-year improvement threshold.
- `config/paper_core_smh_bridge.yaml`: activates the satellite only when the main strategy is risk-off and the long-term SMH trend has turned positive; the effect was too small to promote it to a recommended configuration.
- `config/paper_core_vix_guard.yaml`: immediately prohibits risk-on exposure when VIX/VIX3M is inverted; worsened both return and drawdown and was eliminated.
- `config/paper_core_vix_recovery.yaml`: activates a 20% QQQ/SMH recovery bridge after the inversion ends but before the HMM has re-entered; produced no material improvement.
- `config/paper_core_vrp_feature.yaml`: adds lagged VRP to the HMM; unstable in the development period and eliminated for exceeding the drawdown limit.
- `config/paper_core_bipower_guard.yaml`: uses a daily-frequency bipower proxy to distinguish jumps from persistent volatility; generated too few incremental signals and produced no improvement.
- `config/paper_core_vix_trend_recovery.yaml`: activates the recovery bridge only when the term structure has recovered and the growth trend is positive; produced no improvement.
- `config/paper_core_smh_high_vol_budget.yaml`: switches to an inverse-volatility core when SMH enters the highest 10% of its four-year volatility range; approximately equivalent to the benchmark and was not promoted.
- `config/paper_core_robust_vol_guarded_floor_ensemble.yaml`: a robust comparison baseline for asset-level re-entry optimization; equal-weights three fixed random initializations, caps total risk exposure at 100%, and uses a 20% risk-off growth floor with an acute-volatility guard.
- `config/paper_core_robust_vol_guarded_floor_ensemble_2012.yaml`: the 2012-start stability version of the same rules.
- `config/paper_core_zero_entry_growth_reallocation_ensemble.yaml`: an unlevered growth benchmark; reallocates part of SMH to QQQ only during high-risk re-entry from a zero position, preserving total growth exposure.
- `config/paper_core_growth_gold20_daily_risk_ensemble.yaml`: the previous-generation production strategy; a strategic QQQ/SMH/GLD core, 6-month QQQ/SMH relative momentum, total exposure of up to 1.10× when both assets' 252-day trends are positive, daily jump-aware stressed-volatility control, and a 4% VIXY allocation when VIX/VIX3M is inverted.
- `config/paper_core_growth_gold20_jump_aware_daily_risk_ensemble.yaml`: a mechanism-driven improvement candidate; uses 20-day standard volatility for sudden gaps and 60-day bipower variation to identify persistent volatility, and may replace the operational default only after confirmation in the 2026 paper-trading period.
- `config/paper_core_growth_gold20_lev110_target25_jump_stresscorr_daily_risk_ensemble.yaml`: a high-CAGR shadow candidate after the 2026-07-23 literature integration; allows up to 1.10× total exposure when risk is on and controls joint QQQ/SMH risk using jump-aware volatility and conservative 20-/60-day correlations. It passed the pre-specified point-estimate thresholds, but the family Reality Check has not yet passed, so it does not replace the current Panel.
- `config/paper_core_growth_gold20_lev110_target25_jump_stresscorr_industrymom_daily_risk_ensemble.yaml`: the new best shadow candidate; shifts growth risk between QQQ and SMH using industry-relative momentum over the past 126 trading days without changing the total growth budget. In the 2015–2025 open-price proxy, CAGR was 16.26% and MDD was -16.00%, a 1.70 pp CAGR improvement over production. The Reality Check p-value for the family of 20 candidates was 0.0851, but its block-resampling tail remained weaker than production, so it does not replace the current Panel. The complete audit is available at `output/industry_momentum_optimization_2026-07-23.md`.
- `config/paper_core_zero_entry_growth_gold20_lev110_ensemble.yaml`: a lightly leveraged research control; strong on a close-price basis, but the open-price execution proxy had an MDD of -21.43%, so it was demoted from production-candidate status.
- `config/paper_core_zero_entry_growth_gold20_lev110_accountcap_ensemble.yaml`: a full-account 20% stressed-volatility integration experiment; used to quantify the long-run cost of protecting initial entries from zero positions and does not replace the integrated main candidate.
- `config/paper_core_zero_entry_state_switch_ensemble.yaml`: an exploratory candidate that chooses rotation into QQQ or BIL based on relative SMH/QQQ performance during fully exited periods; has a higher point estimate but depends more heavily on 2024.
- `config/paper_core_vx_futures_guard.yaml`: uses the official Cboe VX1/VX2 settlement curve as a precise hard veto during inversion; materially worsened results and was eliminated.
- `config/paper_core_relative_momentum.yaml`: preserves total growth exposure while tilting within QQQ/SMH based only on 6-/12-month relative momentum; deteriorated in the full-calendar-year results and was eliminated.
- `config/paper_core_growth_floor.yaml`: retains a 20% equal-weighted QQQ/SMH floor during risk-off periods; currently the best research candidate, but it did not meet the pre-specified +1 pp threshold.
- `config/paper_core_growth_gold20_dual_reentry_inverse_momentum_netted_ensemble.yaml`: R8's direct control and promoted predecessor; separates a 20% bridge from zero positions from a 40% floor for existing positions, and switches relative momentum to 60-day inverse volatility during high-volatility periods. The standalone version did not pass multiple-comparison testing, but R8 retains its mechanism and current target.
- `config/paper_core_growth_gold20_dual_reentry_floor45_goodvol_bear20_inverse_momentum_netted_ensemble.yaml`: the R8 fallback configuration; applies the previously tested 45% existing-position guard on top of Dual and permits a 20% bridge from zero only when volatility is high, the 20-day downside-variation share is no greater than 50%, and 252-day excess return is no lower than -20%.
- `config/paper_core_growth_gold20_r9_staged_reentry_netted_ensemble.yaml`: R11's 75% core and fallback configuration; when recovering from a near-zero position, it first allocates 20% to QQQ and permits an increase to 35% only after the signal persists for ten consecutive trading days.
- `config/paper_core_growth_gold20_r10_capital_efficient_guard.yaml`: the previous staged-production configuration; replaced by R11 because the rigorous SMH parameter-family test did not pass.
- `config/paper_core_growth_gold20_r11_diversified_financing.yaml`: the current staged-production configuration; uniformly scales R9's non-cash targets for 25% of strategy-managed capital while retaining a small GDE allocation and the causal SMH overnight hedge, with the remaining 75% retaining R9.
- `config/paper_probability_dd15.yaml`: continuously controls growth exposure using the posterior probability of favorable HMM templates and sets hard risk controls according to a 12%–15% maximum-drawdown budget.

## CAGR-first Version

```bash
python scripts/run_backtest.py --config config/cagr_first.yaml --refresh
```

From 2015-01-02 through 2026-07-21, with transaction costs of 7.5 bps: CAGR 21.96%, annualized volatility 21.63%, zero-rate Sharpe 1.03, Sharpe relative to BIL 0.94, and maximum drawdown -32.46%. Over the same period, SPY had a CAGR of 13.74% and a maximum drawdown of -33.72%. After doubling costs to 15 bps, CAGR was 21.75% and maximum drawdown was -32.56%.

Fixed-window diagnostics: 2015-2021 CAGR 22.02% (SPY 14.84%); 2022-2025 CAGR 16.84% (SPY 11.05%). The strategy did not use a parameter grid search. QQQ/SMH were fixed at equal weights, the 80/20 core-satellite allocation was frozen before the first run, and the results were not used to retroactively modify parameters.

Important: QQQ/SMH were selected after their strong long-term returns were already known, so the asset selection itself is still subject to hindsight selection bias. The 2022-2025 period can only be treated as a fixed-window diagnostic, not as a fully unseen holdout set in the academic sense. Genuine evidence of resistance to overfitting requires freezing the configuration from now on and conducting at least 6-12 months of forward paper trading.

Output is located in `output/cagr_first/`, and the double-cost results are in `output/cagr_first_cost15/`.

## 12%-15% Maximum Drawdown Constraint

```bash
python scripts/run_backtest.py --config config/paper_probability_dd15.yaml --refresh
```

This configuration was fixed before its first run: the HMM favorable-state probability is smoothed with a 0.12 EMA and linearly mapped to 25%-125% equal-weight QQQ/SMH growth exposure; the conditional covariance imposes a 22% forecast-volatility cap; after portfolio drawdown reaches 5%/7.5%/10%/12%, the risky position is multiplied by 75%/50%/25%/0%, respectively, with the remainder allocated to BIL. All state estimates and positions are computed causally.

The first frozen result from 2015-01-02 through 2026-07-21 was: CAGR 6.36%, annualized volatility 10.41%, Sharpe relative to BIL 0.46, and maximum drawdown -14.08%. Over the same period, SPY had a CAGR of 13.74% and a maximum drawdown of -33.72%. This version met the historical drawdown target but **did not meet the return target of substantially outperforming SPY**; it must not be presented as having achieved the objective. The main cost comes from the hard drawdown gate: 37.0% of rebalancing points had zero risk exposure.

The risk-return frontier represented by the existing unscreened versions is shown below. It is evidence of a constraint conflict, not the best result selected from multiple parameter sets:

| Frozen configuration | CAGR | Maximum drawdown | Conclusion |
| --- | ---: | ---: | --- |
| `cagr_first.yaml` | 21.96% | -32.46% | Achieves high CAGR; drawdown target not met |
| `paper_core_growth.yaml` | 15.11% | -16.43% | Close to the drawdown limit; only modestly outperforms SPY |
| `paper_core_dd18.yaml` | 14.35% | -17.71% | Drawdown target met, but CAGR falls after risk is increased |
| `paper_core_sticky.yaml` | 14.26% | -19.06% | Reduces turnover but misses rebounds; rejected |
| `paper_core_hmm_only.yaml` | 14.36% | -22.51% | Drawdown exceeds the limit after removing the auxiliary trend risk control; rejected |
| `paper_core_risk_parity.yaml` | 14.57% | -16.76% | Drawdown target met, but both CAGR and Sharpe are below fixed equal weighting |
| `paper_core_macro_equal.yaml` | 11.04% | -22.04% | Macro regimes react too slowly to technology-sector stress; rejected |
| `paper_core_macro_risk_parity.yaml` | 11.10% | -22.20% | Risk parity does not correct the macro-signal mismatch; rejected |
| `paper_core_smh_satellite.yaml` | 15.46% | -15.73% | Risk-adjusted metrics improve, but the full-year CAGR increase is insufficient |
| `paper_core_smh_bridge.yaml` | 15.14% | -16.43% | Fixes only a few re-entry windows; effect is too small |
| `paper_core_vix_guard.yaml` | 11.71% | -19.27% | Mechanical exits during inversion lead to chasing rebounds; rejected |
| `paper_core_vix_recovery.yaml` | 14.50% | -16.82% | Panic subsiding does not mean the growth trend has recovered; threshold not met |
| `paper_probability_dd15.yaml` | 6.36% | -14.08% | Drawdown target met; CAGR target not met |

Within the scope of long-only ETFs, weekly trading, and the HMM framework described in the paper, the current evidence does not support simultaneously claiming that "CAGR substantially outperforms SPY" and that "maximum drawdown is 12%-15%." Continuing to search state labels, thresholds, and leverage over the same historical interval is more likely to increase overfitting than reproducible returns.

After relaxing the drawdown range to 15%-18%, `paper_core_growth.yaml` is the more robust core-paper candidate among the existing results. The mechanically higher risk budget in `paper_core_dd18.yaml` achieved a 14.35% CAGR and -17.71% maximum drawdown at 7.5 bps costs, but under a 15 bps cost stress it deteriorated to a 12.93% CAGR and -19.24% maximum drawdown. The fixed subperiods likewise do not support stable excess returns: 2015-2021 CAGR 12.83% (SPY 14.84%), and 2022-2025 CAGR 10.00% (SPY 11.05%). This configuration therefore should not be deployed as the preferred version.

Further optimization used structural ablations rather than a parameter grid. The original version's 46 risk-state transitions accounted for 61% of total turnover, so an "exit immediately, wait for HMM refitting before re-entry" rule was tested. Although it reduced the annualized cost drag from 0.94% to 0.86%, CAGR fell to 14.26% and maximum drawdown widened to -19.06%. Next, the trend filter outside the paper was removed, resulting in a 14.36% CAGR and a deterioration in maximum drawdown to -22.51%. Neither change passed the predefined 18% drawdown threshold, so neither proceeded to the double-cost review. The original `paper_core_growth.yaml` therefore remains recommended, rather than being a local optimum selected through continuous parameter search.

QQQ/SMH exposure optimization used a pre-specified 2x2 ablation rather than searching for the historically best ratio. The conditional inverse-volatility version raised QQQ's average core weight from 50% to 57.5%, but because the two assets are highly correlated, portfolio annualized volatility fell only from 16.07% to 15.92%. It also reduced SMH's return contribution and added dynamic-reweighting costs, ultimately producing a 14.57% CAGR, -16.76% maximum drawdown, and 0.81 Sharpe relative to BIL, without outperforming the fixed 50/50 allocation. The risk-on share of the macro-HMM version rose from 70.4% to 79.7%, but it exited technology-specific stress too slowly; both macro versions had maximum drawdowns above 22% and 2022-2025 CAGRs of approximately 5%. None of the three candidates met the ex ante improvement threshold, so they did not proceed to the double-cost review and were not used to reverse-engineer the weighting formula.

Further use of SMH employed an independent, strictly causal 200-day trend and a 60-day volatility target without modifying the HMM thresholds. A fixed 20% account-level satellite scales SMH to 20% annualized volatility when the trend is positive and otherwise holds BIL, while capping the account's target SMH weight at 70%. It raised the 2015-2025 CAGR from 12.83% to 13.22%, improved maximum drawdown from -16.43% to -15.73%, and raised the 2022-2025 CAGR from 10.92% to 12.44%. However, development-period CAGR fell from 13.93% to 13.66%, and the 2015–2025 increase did not reach the ex ante requirement of 1 percentage point, so it was not promoted.

Only one structural review followed: the satellite was activated only when the main strategy remained risk-off but the SMH trend had already turned positive, avoiding dilution of the main strategy during normal periods. The bridge position had an average effective share of only 2.75% at rebalancing points. The 2015-2025 CAGR rose only to 12.93%, the 2022-2025 CAGR rose only to 11.23%, maximum drawdown remained -16.43%, and development-period maximum drawdown instead widened from -15.26% to -16.16%. This result indicates that HMM re-entry lag correctable by a simple SMH trend signal is not the primary return bottleneck; continuing to search trend lengths, satellite shares, or activation thresholds would materially increase overfitting risk. This round of historical analysis therefore continues to retain `paper_core_growth.yaml`.

## Volatility Meta-Survey Extension

*Equity Volatility Regimes: A Meta-Survey* is not a single strategy paper with a fully disclosed backtesting methodology, but a review of realized volatility, the implied-volatility term structure, three-state SWARCH, and market microstructure. Its three most actionable conclusions are: low- and medium-volatility regimes are generally persistent, while high-volatility regimes account for about 10% of history and are abrupt and short-lived; `VIX > VIX3M` backwardation represents acute near-term panic; and rising short-term realized volatility triggers mechanical deleveraging by volatility-control strategies, CTAs, and risk-parity strategies. The existing Wasserstein HMM already captures regime persistence and realized volatility, so this extension tests only the term structure; it does not add the VIX index as a tradable asset or short volatility.

The first frozen experiment immediately vetoed QQQ/SMH risk-on whenever `VIX > VIX3M` at the previous close, without searching a threshold. Daily backwardation occurred 10.22% of the time during 2008-2026, consistent with the article's characterization of it as a rare regime; however, the rule reduced the 2015-2025 CAGR from 12.83% to 10.06% and widened maximum drawdown from -16.43% to -19.27%. The 2022 result was completely unchanged because the original HMM/trend rules had already exited at those times; the losses came mainly from rebounds following sharp declines in 2017, 2020, and 2024-2026. Turnover costs also rose because of repeated exits and re-entries. This is precisely the procyclical volatility-control behavior the article warns about, so the hard gate was rejected.

The second experiment used the conclusion that "high-volatility regimes are short-lived": when the VIX curve recovered from backwardation to contango while the original HMM remained risk-off, it temporarily allocated 20% of the account budget equally to QQQ/SMH; the bridge ended when the HMM became risk-on or backwardation returned. The 20% allocation reused the previously frozen satellite budget; no duration or curve-magnitude parameter was introduced or searched. The bridge was active at 16.35% of rebalancing points, with an average account exposure of 3.27%. It produced a 2015-2025 CAGR of 12.21%, maximum drawdown of -16.82%, development-period CAGR of 13.10%, and 2022-2025 CAGR of 10.66%, none of which exceeded the baseline. Normalization of the term structure indicates only that acute panic has subsided; it does not prove that the technology trend has recovered.

Neither volatility extension passed the ex ante thresholds (a gain of at least 1 percentage point in 2015-2025 CAGR, maximum drawdown no greater than 18%, and no directional deterioration in either the development or holdout period), so neither proceeded to the double-cost test, and no further search was conducted over VIX ratios, confirmation days, or bridge allocations. The second article strengthens the explanation of tail risk and procyclical deleveraging, but the data from this round do not support promoting the VIX term structure to an additional trading switch; `paper_core_growth.yaml` remains the historical baseline for this round.

## Five Frozen Validation Directions

After the two rules above failed, five mutually independent directions, each frozen before execution, were tested further. Every position continued to use only information available as of the previous complete close; there was no threshold grid, post-result tuning, or selection of the best result from adjacent versions:

1. **VRP HMM feature**: Add `VIX² - 21-day annualized realized variance`, lagged by one day before entering the HMM; add no positive or negative threshold.
2. **Jump/persistent volatility**: Use bipower variation of daily returns as a proxy for persistent volatility, retaining the original 20-day, 30% stress threshold. A full bipower jump test requires intraday high-frequency data; Yahoo's 5-minute history covers only the latest 60 days, so this test is explicitly only a reproducible daily-frequency proxy screen since 2015, not a paper-grade intraday validation.
3. **Trend-confirmed VIX recovery bridge**: Activate the frozen 20% recovery bridge only when the curve recovers from backwardation to contango, the baseline remains risk-off, and the existing average multi-horizon trend score for QQQ/SMH is nonnegative.
4. **State-dependent SMH risk budget**: Maintain QQQ/SMH at 50/50 in normal regimes; switch to two-asset inverse volatility only when SMH's 60-day volatility enters the highest 10% of the previous 1008 days. The 10% comes from the article's description of high-volatility-regime frequency and was not selected by backtest search.
5. **Exact VX futures curve**: Construct each trading day's settlement prices for the first two unexpired contracts from Cboe's official CSV files for individual monthly contracts; apply the original hard veto when `VX1 > VX2` to isolate approximation error in VIX/VIX3M. Expiring contracts are excluded from the curve on their expiration date, with zero gaps against ETF trading days during 2015-2025.

The ex ante acceptance thresholds were: full 2015-2025 CAGR at least 1 percentage point above the baseline, i.e. no lower than 13.83%; maximum drawdown no greater than 18%; and CAGRs in both the 2015-2021 development period and 2022-2025 holdout period no lower than the baseline. Results were as follows:

| Frozen direction | 2015-2025 CAGR | Sharpe | Maximum drawdown | Development-period CAGR | Holdout-period CAGR | Conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline `paper_core_growth` | 12.83% | 0.85 | -16.43% | 13.93% | 10.92% | Comparison baseline |
| VRP HMM feature | 11.42% | 0.78 | -23.26% | 9.60% | 14.69% | Unstable across samples and fails the drawdown threshold |
| Daily-frequency bipower proxy | 12.07% | 0.82 | -16.86% | 13.33% | 9.89% | Return deterioration |
| Trend-confirmed recovery bridge | 12.51% | 0.84 | -16.43% | 13.43% | 10.92% | No incremental alpha |
| SMH high-volatility budget | 12.81% | 0.85 | -16.57% | 13.78% | 11.14% | Indistinguishable from the baseline |
| Exact VX backwardation hard veto | 8.62% | 0.66 | -19.37% | 9.05% | 7.87% | Fails on both return and drawdown |

Mechanism statistics explain the failures. VRP changed the number of HMM states selected in 234/581 cases and the final risk decision 47 times. It failed badly in the development period but improved in the holdout period, showing that feeding a directional risk-premium variable directly into unsupervised state clustering rearranges the states without guaranteeing the correct return direction. The daily-frequency bipower proxy changed only 8 stress flags and 5 risk-on decisions, making its incremental signal too sparse. Trend confirmation reduced the recovery bridge to 12 rebalances, but those trades still did not generate positive excess returns. The SMH high-volatility budget triggered 118 times; when triggered, SMH's average core weight fell only from 50% to 43.34%. The portfolio volatility target then raised total growth exposure, offsetting part of the risk reduction, so CAGR and drawdown were almost unchanged.

The exact VX curve instead reinforced the original negative conclusion: among 581 rebalances, exact `VX1 > VX2` occurred 105 times, while the VIX/VIX3M approximation occurred only 48 times, with 59 disagreements between the two measurements. Exact backwardation conflicted with baseline risk-on 42 times, versus only 16 times for the approximate rule. The futures curve indicates front-end stress more often, but still provides no market direction; a mechanical hard veto therefore misses the risk premium and rebound after sharp declines more frequently and raises the full-sample annualized average total cost from 0.97% to 1.16%.

To reduce the risk of incorrectly attributing results to a lucky single historical path, the daily-return differences between each candidate and the baseline were also evaluated with 10,000 paired bootstrap replications using fixed 21-trading-day blocks. The SMH high-volatility budget's 95% interval for annualized relative return was -0.36% to +0.31%, with a 48.0% probability of improvement—effectively indistinguishable from zero effect; the exact VX interval was -6.17% to -1.39%, with only a 0.06% probability of improvement. The intervals for the other three directions also included zero and had negative point estimates. All five failed the first CAGR threshold, so under the ex ante rules they did not receive a second cost test at 15 bps, and adjacent parameters were not searched.

The conclusion is to retain `paper_core_growth.yaml` and not incorporate any of the five directions into the production candidate. Complete results are in `output/five_direction_validation/`; reproduction commands are:

```bash
PYTHONPATH=src .venv/bin/python tools/fetch_vx_curve.py --first-year 2014 --last-year 2027
.venv/bin/python -m regime_strategy.cli --config config/paper_core_vrp_feature.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_bipower_guard.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_vix_trend_recovery.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_smh_high_vol_budget.yaml
.venv/bin/python -m regime_strategy.cli --config config/paper_core_vx_futures_guard.yaml
PYTHONPATH=src .venv/bin/python tools/evaluate_five_directions.py
```

## First-Principles Growth Floor Candidate

After the five volatility directions failed, the strategy design shifted from "predicting more precisely when to exit" to "reducing the risk-premium loss caused by binary exits." First, a QQQ/SMH relative-momentum overlay was frozen and tested: the HMM and 20% volatility target still determine total QQQ+SMH exposure, while 126/252-day relative momentum that skips the most recent 21 days continuously limits SMH's share of the core to 30%-70%. Through 2026-07-21, this version raised full-sample CAGR from 15.11% to 15.49%, but reduced 2015–2025 CAGR from 12.83% to 12.61%, widened maximum drawdown to -17.92%, and deteriorated in both the development and holdout periods. The full-sample improvement came from SMH's strength after 2026 and does not justify promotion; neither the windows nor the tilt range was adjusted in this experiment.

The second, structurally independent frozen candidate is `paper_core_growth_floor.yaml`. It does not predict the relative strength of QQQ and SMH or change the HMM risk state:

- When the HMM is risk-on, it follows the original strategy in full;
- When the HMM is risk-off, 80% of the account continues to hold the original defensive target, while 20% is allocated equally to QQQ/SMH;
- It remains subject to the 20% forecast-volatility cap and the original -10%/-15%/-20% drawdown multipliers;
- It is evaluated every 5 trading days, does not trade when one-way turnover is below 1%, and retains costs of 7.5 bps;
- The 20% allocation reuses the previously frozen account-level satellite budget; 10%/20%/30% was not searched.

It addresses asymmetry in the action space: the original binary strategy completely abandons the growth risk premium at 29.6% of rebalancing points, whereas a permanent floor retains some participation in rebounds. The floor was active at 172 of 581 rebalances; because of the volatility target, drawdown gate, and actual weight drift, the average realized QQQ+SMH position while active was 16.77%. The risk state itself had zero disagreements with the baseline.

| Metric | Original `paper_core_growth` | 20% risk-off growth floor | Difference |
| --- | ---: | ---: | ---: |
| 2015-2025 CAGR | 12.83% | 13.67% | +0.84pp |
| 2015-2025 Sharpe | 0.85 | 0.89 | +0.04 |
| 2015-2025 maximum drawdown | -16.43% | -17.35% | -0.92pp |
| 2015-2021 CAGR | 13.93% | 15.11% | +1.17pp |
| 2022-2025 CAGR | 10.92% | 11.18% | +0.26pp |
| Full-sample CAGR through 2026-07-21 | 15.11% | 15.83% | +0.72pp |
| Full-sample maximum drawdown | -16.43% | -17.35% | -0.92pp |

The candidate outperformed the baseline in 9 of 11 complete calendar years, but returns were not uniform: it underperformed the baseline by 5.02 percentage points in 2022 and outperformed it by 4.06, 4.18, and 3.34 percentage points in 2019, 2023, and 2025, respectively. Ledger attribution by QQQ up and down days shows that it added 10.45% of annualized relative log return on QQQ up days while losing 9.71% on down days, for a net gain of approximately 0.74%. This is the opposite of the failed VX hard-veto direction, but the margin of safety is small. Because the magnitude of risk switching was reduced, annualized average total costs also fell from approximately 0.97% to 0.87%.

The fixed 21-day-block, 10,000-replication paired bootstrap gave an annualized relative-return point estimate of +0.74%, a 95% interval of -0.90% to +2.40%, and an 81.31% probability of positive improvement. The increase in full 2015-2025 CAGR still fell 0.16 percentage points short of the ex ante +1 percentage-point threshold, so `overall_pass=0`: the floor allocation was not searched further, and the 15 bps version was not run under the rules for passing candidates. Under this round's constraint target of "maximum drawdown no greater than 18%," it is the stronger research candidate, but the evidence is not yet sufficient to replace the production baseline at that time.

Reproduction commands and results:

```bash
.venv/bin/python -m regime_strategy.cli --config config/paper_core_relative_momentum.yaml
PYTHONPATH=src .venv/bin/python tools/evaluate_relative_momentum.py
.venv/bin/python -m regime_strategy.cli --config config/paper_core_growth_floor.yaml
PYTHONPATH=src .venv/bin/python tools/evaluate_growth_floor.py
```

The relative-momentum results are in `output/relative_momentum_validation/`, and the growth-floor results are in `output/growth_floor_validation/`. The conclusion of this round is that `paper_core_growth.yaml` remains the baseline and the growth floor is used only for paper tracking; the "Current Conclusion" at the top takes precedence over this historical record.

## First-Principles Robust Portfolio: Multiple Initializations, No Leverage, Volatility-Guarded Floor

Further auditing found that the low-drawdown conclusion from a single HMM changes with EM random initialization. With two random restarts, seeds 7/42/123 can produce maximum drawdowns of approximately -27.6%/-17.0%/-17.4%, respectively; increasing to eight restarts still does not converge to the same trading risk. The cause is not label permutation, but several local solutions with similar likelihoods producing different binary risk-on decisions on critical dates. A small number of erroneous activation days can amplify tail risk when QQQ/SMH exposure simultaneously exceeds 100%, while erroneous deactivation misses recoveries. Continuing to select the "best seed" or increasing the state threshold would constitute result-driven overfitting.

The preferred research candidate for this round, `paper_core_robust_vol_guarded_floor_ensemble.yaml`, therefore uses the following fixed structure:

- Independently run three HMM subaccounts with seeds 7, 42, and 123, each holding one third of the account; the seed set was frozen before robustness testing and was not selected by results.
- Each member uses the posterior mixture return moments from the current HMM fit, a global-calendar-anchored 5/21/63-day trading/refitting/model-order-selection schedule, and a 100% cap on total QQQ+SMH exposure, with no borrowing or leverage.
- When risk-on, hold an equal-weight QQQ/SMH growth core subject to a 20% forecast-volatility target; when risk-off, retain a fixed 20% equal-weight QQQ/SMH floor to reduce the loss of the long-term risk premium from binary exits.
- When QQQ's annualized realized volatility over the previous 20 days exceeds the original strategy's already-frozen 30% stress threshold, close the 20% floor; a negative slow 200-day trend does not by itself close the floor. This prevents the long-term floor from overriding acute-volatility risk controls and adds no new threshold.
- The three subaccounts attempt to restore equal weights every 21 trading days and do not trade when deviation is below 1%; both member-level and outer-level costs are charged at 7.5 bps on two-sided turnover. In the current sample, outer-level deviations always remain below 1%, so outer-level costs are approximately zero, but the code still charges them explicitly.

Net-of-cost results through 2026-07-21 are shown below. The `complete-calendar-year period` is fixed as 2015–2025 to avoid using SMH's 2026 strength to select the model retroactively:

| Metric | Original `paper_core_growth` | Robust ensemble | Difference |
| --- | ---: | ---: | ---: |
| 2015-2025 CAGR | 12.83% | 14.93% | +2.10pp |
| 2015-2025 Sharpe | 0.85 | 1.01 | +0.16 |
| 2015-2025 maximum drawdown | -16.43% | -16.86% | -0.43pp |
| 2015-2021 CAGR | 13.93% | 13.92% | -0.01pp |
| 2022-2025 CAGR | 10.92% | 16.73% | +5.81pp |
| Full-sample CAGR through 2026-07-21 | 15.11% | 16.97% | +1.86pp |
| Full-sample Sharpe through 2026-07-21 | 0.96 | 1.10 | +0.14 |
| 2015-2025 CAGR at 15 bps | 11.44% | 14.16% | +2.72pp |
| Maximum drawdown at 15 bps | -16.83% | -16.88% | -0.05pp |

When run from 2012, the candidate produced a CAGR of 14.31%, Sharpe of 1.00, and maximum drawdown of -16.86% through 2025; the corresponding figures for the original 2012 baseline were 9.82%, 0.71, and -22.51%. The daily-return correlation between the 2012-start and 2015-start runs over their common interval was 0.9967, with identical maximum drawdowns; CAGRs over the common interval were 16.50% and 16.97%, respectively. This materially weakens the original implementation's start-date path dependence.

The statistical evidence still requires conservative interpretation. The 21-day paired-block bootstrap for 2015-2025 gives an annualized relative return of +1.86%, a 95% interval of -1.75% to +5.79%, and an 84.0% probability of positive improvement; the interval for 2012-2025 is +0.65% to +7.77%, with a 99.1% probability of positive improvement. However, among 24 formal candidates with maximum drawdown no greater than 18%, the Reality Check family-wise p-value is 0.450. The candidate also failed the ex ante zero-tolerance Boolean threshold that "no subperiod may deteriorate" because development-period CAGR was approximately 1.1 basis points below the baseline. It is therefore only an operationally executable **paper candidate** with a strong risk-return profile in this round, not proven reproducible alpha.

Using closing data as of 2026-07-21, the theoretical equal-weight member targets for the next trading day are QQQ 37.56%, SMH 37.56%, and BIL 24.88%, with all other weights zero. Actual orders should use the `ensemble_current_sleeve_weight` column in `next_target_weights.csv` to reflect real-time NAV drift across the three subaccounts. Reproduction commands:

```bash
# Generate next-trading-day targets in daily operation; refresh adjusted prices online
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble.yaml --refresh

# Reproduce and validate using the frozen cache
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble.yaml
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble.yaml --cost-bps 15 --output-dir output/paper_core_robust_vol_guarded_floor_ensemble_cost15
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli --config config/paper_core_robust_vol_guarded_floor_ensemble_2012.yaml
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_robust_ensemble.py
```

Complete statistics are in `output/robust_vol_guarded_floor_validation/`.

## Asset-Level Re-entry Risk: SMH Has High Beta, Not Decorrelation from QQQ

The July 2026 event exposed an execution gap in the original robust portfolio. All three members held zero growth exposure from June 30 through July 14, but they simultaneously switched back to risk-on on July 15, raising QQQ+SMH from 0% to approximately 79.3% in a single step. The portfolio lost -4.34% from July 15–17. Using the July 14 complete close, the 20/60-day stressed volatility of QQQ/SMH was approximately 28.7%/60.9%, their 60-day correlation was approximately 0.94, and SMH's beta to QQQ was approximately 2.0. The issue was therefore not a breakdown in correlation, but that the original QQQ stress gate reopened as soon as QQQ volatility fell back below 30%, while SMH's twofold factor sensitivity had not yet normalized.

Three broad volatility-control schemes were tested first: always-on inverse-volatility sizing with a hard 20% cap, applying it only at monthly rebalances, and daily de-risking without re-risking. They reduced full-sample maximum drawdown to approximately -13% to -14%, but lowered CAGR to approximately 14.9%–15.2%. The reason is that high volatility does not imply negative expected returns, so mechanical de-risking gave up the SMH risk premium over the long run; the daily rule also increased the annualized cost drag to approximately 1.12%. All of these versions were rejected.

The narrower production candidate from this round, `paper_core_zero_entry_growth_reallocation_ensemble.yaml`, handles only one state transition:

- It activates only when the previous actual QQQ+SMH weight was zero, the current state switches to risk-on for the first time, and the original target's 20/60-day stressed realized volatility exceeds the existing 20% risk budget;
- It keeps total QQQ+SMH exposure unchanged and does not reduce leverage further;
- On the initial re-entry, it reallocates part of SMH into QQQ using stressed inverse-volatility sizing, bringing the two assets' contributions to account risk close to equal;
- It returns to the normal model target at the next existing 5-day evaluation, without adding confirmation days or optimizing thresholds;
- It is identical to the robust baseline on every other date.

The fixed comparison is shown below. All returns include 7.5 bps in two-sided turnover costs; `July re-entry loss` covers 2026-07-15 through 2026-07-17. The full-sample point estimates in the table are research snapshots and are not a substitute for the live Panel:

| Version | 2015–2025 CAGR | Sharpe | MDD | Development CAGR | Holdout CAGR | Full-sample CAGR | July re-entry loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Robust baseline | 14.93% | 1.011 | -16.86% | 13.92% | 16.73% | 17.01% | -4.34% |
| Growth-first: SMH to QQQ | 14.99% | 1.015 | -16.86% | 13.96% | 16.80% | 17.08% | -3.78% |
| Risk-first: SMH to BIL | 14.96% | 1.016 | -16.86% | 13.92% | 16.81% | 17.10% | -2.77% |
| Relative-state routing | 15.06% | 1.020 | -16.86% | 13.94% | 17.06% | 17.19% | -2.77% |

The growth-first version is the primary candidate from this round because it shows no directional deterioration over the 2015–2025 complete-calendar-year period, development period, holdout period, from the 2012 start, or under doubled 15 bps costs, while its worst relative annualized return in year-by-year leave-one-out tests is only approximately -0.004 percentage points. The 2012–2025 CAGR improves from 14.31% to 14.41%, and the 2015–2025 CAGR at 15 bps improves from 14.16% to 14.20%. When the 20/60-day windows are changed to 10/40 or 40/120, or the 20% activation budget is changed to 18% or 22%, the 2015–2025 CAGR remains approximately 15.04%–15.06%, indicating that the result is not a single-point parameter spike.

`paper_core_zero_entry_state_switch_ensemble.yaml` is an exploratory candidate: when SMH's relative state versus QQQ is negative during the fully exited period, it reallocates the excess SMH risk to BIL; otherwise, it reallocates that risk to QQQ. Its point estimate is higher and its protection against the current event is stronger, but with 2024 excluded, its relative annualized return is approximately -0.02%. It should not replace the simpler primary candidate merely because it performs better in-sample.

The statistical evidence remains insufficient. For the growth-first version, the probability of a positive improvement under a fixed 21-day block bootstrap is approximately 76.6%, and the 95% interval crosses zero; after including the 38 drawdown-feasible candidates in the current directory in a family-wise Reality Check, the p-value is approximately 0.98. The asset-level re-entry rule should be understood as a risk repair with a clear mechanism and very low long-run cost, not as proven new alpha. The complete audit is in `output/zero_entry_tail_relative_risk_validation/`.

## Open-Execution Correction and the Current Executable Candidate

Earlier results applied signals formed at the previous complete close to returns from the previous close to the current close, which is equivalent to trading at the same closing price at which the signal is formed. A real next-day open cannot avoid the overnight gap. The adjusted Open/Close audit splits each day into "overnight return on the old position, rebalance at the open, and intraday return on the new position." It found that the original unlevered growth version's 2015–2025 MDD deteriorated from -16.86% to approximately -22.99%; the 20% GLD + 1.10× version also deteriorated from -16.44% to approximately -21.43%. These close-based candidates can therefore no longer be considered to satisfy the 18% target.

The current executable candidate, `paper_core_growth_gold20_daily_risk_ensemble.yaml`, uses a risk-on core of 40% QQQ, 40% SMH, and 20% GLD, with no leverage. On each trading day, the 20/60-day stressed-volatility rule is allowed only to reduce existing QQQ/SMH positions; it does not add exposure on an unscheduled day merely because volatility has declined. GLD provides structural diversification, while the daily volatility control addresses the risk of the residual growth allocation retained after the 2020 risk-off event. GLD is not a day-by-day crash hedge: on the worst 5% of QQQ/SMH days, GLD averaged approximately +0.02%, and it was positive on only 45.8% of those dates. Its value comes from a median 252-day correlation of approximately 0.03 and rebalancing across market cycles.

The open-execution proxy results, net of 7.5 bps costs, are shown below; they are still not a guarantee of future performance:

| Scope | CAGR | Sharpe | MDD | Concurrent SPY CAGR | SPY MDD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015–2025 | 14.56% | 1.101 | -16.96% | 13.45% | -33.72% |
| 2015–2025, 15 bps costs | 13.80% | 1.050 | -17.09% | 13.45% | -33.72% |
| 2012–2025 | 13.42% | 1.053 | -16.96% | 14.90% | -33.72% |

Only two separate conclusions can therefore be drawn: with a 2015 start and normal costs, the candidate historically outperformed SPY by approximately 1.11 percentage points while keeping MDD below 18%; at 15 bps, the excess return is only approximately 0.35 percentage points, and the estimated cost break-even is approximately 18.5 bps; with a 2012 start, it underperforms SPY by approximately 1.47 percentage points and certainly does not "substantially outperform" QQQ/SMH. Although the high-CAGR, lightly leveraged version raises the 2015–2025 open-execution proxy CAGR to approximately 14.74%, it also raises MDD to approximately -17.30%, gaining only approximately 0.18 percentage points. That is not enough to compensate for the added leverage, financing, and turnover complexity, so it has been downgraded to a research comparator.

Block resampling further limits how the 18% figure should be interpreted. With a fixed 20% stressed-volatility target and 21/63/126-day blocks, the frequencies at which alternative paths breach an 18% drawdown are approximately 63.8%/55.3%/37.7%. The strategy's tail is substantially better than SPY's, but it cannot guarantee that maximum drawdown will not exceed 18%. The historical open-execution CAGRs for the adjacent 18%/20%/22% volatility targets are 14.04%/14.56%/14.81%, with MDDs of -16.11%/-16.96%/-17.70%. This indicates that 20% is not an isolated parameter spike; the 22% target leaves only approximately 0.30 percentage points of headroom to the hard constraint, so it is not used as the default.

The jump-aware version comes from the jump-diffusion decomposition in the volatility review, not from a search over the equity curve: it replaces the standard deviation used for 60-day slow volatility with bipower variation, while retaining the 20-day standard deviation so that exposure can be cut immediately during sudden shocks. Its open-execution CAGR over the frozen window is 14.65%, 13.90% at 15 bps, and 13.50% from the 2012 start, with the same MDD as the benchmark. Across the three block lengths, the probability of a relative CAGR improvement is approximately 93.7%–95.4%, but the 95% interval still crosses zero slightly, and the absolute improvement is only approximately 0.10 percentage points. After including the 12 historically drawdown-feasible candidates from this round, its annualized log excess return relative to SPY is approximately +1.06%, with a nominal one-sided p=0.389, a family-wise Reality Check p=0.438, and a rank of 5. It can therefore be viewed only as an engineering repair with a cleaner mechanism, not as statistically proven alpha. The complete audits are in `output/jump_aware_validation/` and `output/open_family_reality_check/`.

The more conservative risk-capacity frontier also does not eliminate path uncertainty. As the growth stressed-volatility target is lowered from 22% to 20%/18%/16%/14%, frozen-window open-execution CAGR is 14.81%/14.56%/14.04%/13.57%/12.80%, and historical MDD is -17.70%/-16.96%/-16.11%/-14.84%/-13.56%. Even at a 14% target, the frequency of breaching an 18% drawdown under 21-day block resampling remains approximately 37.4%, while CAGR already trails SPY. No point from 14%–22% simultaneously satisfies both "breach probability below 25% for every block length" and outperformance relative to the robust baseline. The complete frontier is in `output/risk_capacity_frontier/`.

Widening the no-trade band does not materially improve the results either. The jump-aware version's 1%/2%/3% no-trade bands produce open-execution CAGRs of 14.65%/14.68%/14.70% and MDDs of -16.96%/-17.10%/-17.19%; turnover declines by only 1.3%/2.3%, while Sharpe decreases slightly. The 1% band is therefore retained, rather than continuing to fine-search the threshold based on the highest in-sample CAGR. The audit is in `output/no_trade_band_validation/`.

The 2026 data have already been used for problem diagnosis and the current Panel, so they can no longer be presented as a completely unseen holdout set. The jump-aware rule was frozen before the close on 2026-07-22; truly forward-looking execution and signal logs begin accumulating on 2026-07-23, and conclusions may be upgraded based on live evidence only after at least 8–12 weeks of observation. Until then, the 2026 report may be used only to check whether the mechanism has failed in an obvious, directionally adverse way, not to claim new alpha.

Release `2026-07-22-v1` records the SHA-256 hashes of six core execution modules, two candidate configurations, the acceptance results, and the current Panel in `output/forward_monitoring/freeze_manifest_2026-07-22-v1.json`; a release with the same name cannot overwrite it. The standard and jump-aware signals for 2026-07-22 have also been written to the immutable `signal_log.csv`, with an execution date of 2026-07-23. Any change to core logic must use a new release name and restart the forward-observation clock; this version must not be rewritten.

Initial entry from a zero position adds another operational safeguard: first, part of SMH is reallocated to QQQ using stressed inverse-volatility sizing; then, a 20% whole-account volatility cap is applied to the 20/60-day joint stressed covariance of QQQ+SMH+GLD. This is a risk budget for the first holding cycle, not a drawdown guarantee; only the next existing evaluation point determines whether exposure should be increased toward the long-run model target. The open-execution evidence is in `output/open_execution_validation/`, and the structural-diversification evidence is in `output/structural_diversification_validation/` and `output/gold_hedge_mechanism_validation/`.

A position-aware panel for a currently uninvested $700,000 account can be generated with:

```bash
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble.yaml \
  --refresh
PYTHONPATH=src .venv/bin/python tools/build_operational_panel.py \
  --account-value 700000 \
  --current-qqq 0 --current-smh 0 --current-gld 0 --current-bil 0
```

Refreshes must occur after 16:15 on a New York trading day; intraday daily bars are discarded, and an old cache written intraday is still treated as incomplete even when read after the close. Operational files are in `output/current_operational_panel/`; use the latest time status in `panel.md` and `order_plan.csv` as authoritative. Only orders with a status of `READY_FOR_OPEN` or `UPCOMING` and `executable=True` may proceed to the next step of manual review. `MISSED`, `DRAFT_*`, or `executable=False` means that the execution window has expired; wait for a new complete close and refresh the calculation, and do not chase the order at an old price.

When the account has a zero position, the model's `HOLD` must not be interpreted as an instruction to remain uninvested. By default, the Panel generates a whole-account onboarding target capped at 20% stressed volatility and also displays a one-week 1σ dollar amount. The latter is not a maximum loss. A user who cannot tolerate the open-execution proxy's historical -16.96% portfolio drawdown—or a still deeper drawdown in resampled paths—should not rely on phased buying to conceal a mismatch in risk tolerance.

Reproduction commands:

```bash
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble.yaml
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble_2012.yaml
PYTHONPATH=src .venv/bin/python -m regime_strategy.ensemble_cli \
  --config config/paper_core_growth_gold20_daily_risk_ensemble.yaml \
  --cost-bps 15 \
  --output-dir output/paper_core_growth_gold20_daily_risk_ensemble_cost15
PYTHONPATH=src .venv/bin/python tools/evaluate_open_execution.py --refresh
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_executable_candidate.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_jump_aware_overlay.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_live_execution.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_risk_capacity_frontier.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_no_trade_bands.py
PYTHONPATH=src:tools .venv/bin/python tools/evaluate_open_family_reality_check.py
PYTHONPATH=src .venv/bin/python tools/snapshot_forward_signal.py
PYTHONPATH=src .venv/bin/python tools/freeze_strategy_release.py
PYTHONPATH=src .venv/bin/python tools/evaluate_structural_diversification.py
PYTHONPATH=src:tools .venv/bin/python \
  tools/evaluate_zero_entry_tail_relative_risk.py
```

## Paper Conclusions and Reproduction Boundaries

The paper reports an OOS Sharpe of 2.18 and a maximum drawdown of -5.43%, but it does not disclose the tickers, train/test split, complete HMM and optimizer hyperparameters, or explicit slippage assumptions. Its formula also writes parameters of a 15-dimensional feature distribution directly into a 5-asset MVO, creating a dimensionality gap. This repository therefore does not claim to "reproduce 2.18." Instead, it uses an auditable implementation: the HMM identifies feature regimes, historical posterior probabilities are used to estimate each regime's **next-period asset-return moments**, and those estimates are smoothed through Wasserstein templates before entering the MVO.

## Default Trading Rules

- Assets: SPY (equities), IEF (U.S. Treasuries), GLD (gold), DBC (commodities/oil-price proxy), and UUP (U.S. dollar), with BIL as a cash substitute.
- Features: previous-trading-day return, lagged 60-day volatility, and lagged 20-day mean; any position on day `t` uses only information available by the close of `t-1`.
- Model: 4-year rolling window; the number of HMM states is selected from 2/3/4 using predictive likelihood over the previous 126 days; the model is re-estimated every 21 trading days, and the state count is reselected every 63 days.
- Identity: diagonal-Gaussian 2-Wasserstein distance maps states to four persistent templates, which are updated with a 0.12 EMA.
- Portfolio: long-only, fully invested, no more than 40% in any single risky asset, conditional mean/covariance shrinkage, and an L1 turnover penalty.
- Robust prior: the final optimized weights are shrunk 50% toward the `1/N` portfolio of the five risky assets, reducing mean-estimation error and model-failure risk.
- Robust return forecast: HMM conditional means receive a 35% weight, while fixed 1/3/6/12-month volatility-adjusted trends receive 65%; this ratio and these horizons are not searched over OOS.
- Risk controls: 9% annualized-volatility target; when SPY's 200-day absolute momentum is negative or its 20-day annualized volatility exceeds 25%, SPY/commodity allocations are reduced to 25%; when the portfolio drawdown from its historical peak reaches 4%/7%/10%, all risky allocations are successively reduced to 75%/50%/25%, with the remainder moved to BIL.
- Execution: evaluate every 5 trading days; do not trade when one-way turnover is below 1%; charge 7.5 bps on total traded notional.

Signals are generated after the close and executed near the next trading day's open. First calculate `target weight - current actual weight`; if total one-way turnover is below 1%, do not trade. Otherwise, use limit orders or a participation-rate algorithm. The live system must write actual fills back as `previous_weights` for the next optimization and must not assume that all theoretical target orders were filled.

## Running the Project

```bash
source .venv/bin/activate
python -m pip install -e .
python scripts/run_backtest.py --refresh
```

On the second run, `--refresh` may be omitted to use the `data/prices.csv` cache. Output is written to `output/backtest_v3/`; `output/backtest/` and `output/backtest_v2/` retain the first two baseline versions:

- `metrics.csv`: zero-rate Sharpe net of costs, `cash_excess_sharpe` relative to BIL, Sortino, maximum drawdown, and other metrics;
- `annual_metrics.csv` / `paper_window_metrics.csv`: robustness breakdowns by year and over the paper's window;
- `daily_returns.csv`: daily returns, costs, turnover, and drawdown;
- `weights.csv`: positions at each daily open;
- `regimes.csv`: HMM state count, template probabilities, and forecast volatility;
- `performance.png`: equity, drawdown, and weight charts.
- `next_target_weights.csv`: the next-trading-day action and weights after the latest complete close. `HOLD` indicates a non-scheduled rebalance day and means that no order should be placed; only `REBALANCE` triggers calculation of target differences.
- `next_signal_diagnostics.csv`: latest HMM state, template probabilities, forecast volatility, and risk-control switches.
- `run_metadata.json`: the complete-close cutoff date, generation time, and configuration path used by the current model run; the Panel rejects stale model output whose cutoff date does not match the price cutoff date.

Tests:

```bash
.venv/bin/python -m pytest -q
```

## Pre-Deployment Gates

Do not deploy based only on the full-sample Sharpe. At minimum, require: Sharpe ratios with the same sign across multiple market subperiods; a positive Calmar ratio after doubling costs; continuous results at adjacent parameter values; and 8–12 weeks of error-free paper-trading signals, fills, and write-backs. The current executable candidate does not use leverage; new accounts should use the Panel's whole-account onboarding target capped at 20% stressed volatility, and chasing orders in `MISSED` or `STALE_MODEL` status is prohibited. Because the strategy still does not outperform SPY from the 2012 start and its excess return is very thin at 15 bps, it should not be marketed as a strategy that reliably generates excess returns over the benchmark.

## Literature-Driven Residual / Dynamic-Correlation Review

Based on Moreira–Muir, Cederburg et al., Engle–Sheppard, Residual Momentum, Risk-Constrained Kelly, PBO, and the Deflated Sharpe Ratio, two candidates with no parameter search were tested under another frozen specification. The 36-month beta + 12–1-month residual-momentum candidate has a 2015–2025 open-execution CAGR of 14.40% and MDD of -16.96%, below the production baseline's 14.56% without improving drawdown; the family-wise p-value across 13 risk-feasible candidates is 0.460. The dynamic factor-risk candidate, which uses 20/60-day stressed correlation and cuts SMH first when over budget, has an open-execution CAGR of 14.46% and MDD of -16.90%; it improves drawdown by only 0.07 pp while sacrificing 0.09 pp of CAGR, so it is not promoted either. The complete paper matrix, mechanism diagnostics, and bootstrap evidence are in `output/literature_integration_validation/decision.md`.
