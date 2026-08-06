from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .features import build_causal_features, build_next_feature, simple_returns
from .model import (
    CausalHMMEstimator,
    RegimeFit,
    WassersteinTemplateTracker,
    equal_weight_predictive_moments,
    hmm_fit_diagnostics,
    probability_weighted_predictive_moments,
)
from .portfolio import (
    apply_account_stress_volatility_cap,
    apply_asset_aware_growth_risk_budget,
    apply_daily_controlled_asset_risk_reduction,
    apply_daily_growth_risk_reduction,
    apply_daily_growth_risk_reallocation_target,
    apply_daily_growth_risk_target,
    apply_trend_cvar_risk_reduction,
    apply_drawdown_overlay,
    apply_asset_multiplier,
    apply_conditional_cash_funded_hedge,
    apply_growth_stress_guard,
    apply_pair_survival_multiplier,
    apply_relative_asset_risk_cap,
    apply_risk_off_growth_floor,
    apply_risk_on_leverage,
    apply_self_financing_overlay,
    apply_volatility_target,
    blend_equal_weight_core,
    blend_satellite_allocation,
    blend_strategic_core,
    cap_asset_standalone_risk_contribution,
    cap_pair_asset_weight,
    cap_pair_standalone_risk_share,
    causal_realized_volatility_state,
    compose_probability_allocation,
    drawdown_conditioned_regime_candidate,
    ensure_minimum_growth_exposure,
    ensure_minimum_growth_group_exposure,
    financing_spread_cost,
    filter_core_weights_by_trend,
    high_volatility_core_weights,
    implied_volatility_term_structure,
    incremental_growth_floor_term_structure_cap,
    incremental_leverage_term_structure_gate,
    idiosyncratic_downside_survival_multiplier,
    inverse_volatility_weights,
    joint_excess_trend_leverage_gate,
    long_only_trend_defensive_weights,
    multi_horizon_trend_forecast,
    paper_regime_candidate,
    reallocate_pair_weights,
    realized_stress_covariance,
    realized_volatility_stress,
    relative_momentum_pair_weight,
    relative_momentum_spread_overlay_weights,
    realized_downside_variation_share,
    realized_jump_variation_share,
    residual_momentum_beta_pair_weight,
    risk_off_growth_floor_is_active,
    solve_allocation,
    satellite_share_for_mode,
    self_financing_overlay_share_for_mode,
    transaction_cost,
    time_series_momentum_overlay_weights,
    trailing_compounded_return,
    trend_volatility_exposure,
    turning_point_cycle,
    update_asymmetric_regime_state,
    update_vix_recovery_bridge,
    volatility_managed_relative_momentum_weight,
    volatility_regime_gated_relative_momentum_weight,
    zero_entry_allocation_method,
)
from .survival import (
    SurvivalOverlayState,
    apply_survival_governor,
    feature_novelty_percentile,
)
from .schedule import anchored_business_day_position


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    regimes: pd.DataFrame
    benchmarks: pd.DataFrame
    asset_returns: pd.DataFrame


def adaptive_hmm_refit_stress(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    assets: list[str],
    fast_days: int,
    slow_days: int,
) -> bool:
    """Return a causal fast-versus-slow volatility stress state."""
    if fast_days <= 1 or slow_days <= fast_days:
        raise ValueError("Adaptive HMM refit windows must satisfy 1 < fast < slow")
    missing = sorted(set(assets).difference(prices.columns))
    if missing:
        raise ValueError(f"Adaptive HMM refit is missing assets: {missing}")
    history = prices.loc[prices.index < date, assets].astype(float)
    returns = np.log(history).diff().dropna(how="any")
    if len(returns) < slow_days:
        return True
    fast = returns.iloc[-fast_days:].std(ddof=1)
    slow = returns.iloc[-slow_days:].std(ddof=1)
    return bool(fast.gt(slow).any())


def adaptive_hmm_refit_policy(
    stressed: bool,
    previous_stressed: bool | None,
    config: dict[str, object],
) -> tuple[int, bool]:
    """Return effective cadence and whether a state transition forces refit."""
    mode = str(config.get("mode", "periodic"))
    calm_days = int(config["calm_refit_days"])
    stress_days = int(config.get("stress_refit_days", calm_days))
    if min(calm_days, stress_days) < 1:
        raise ValueError("Adaptive HMM refit cadences must be positive")
    if mode == "periodic":
        return (stress_days if stressed else calm_days), False
    if mode == "transition":
        transitioned = (
            previous_stressed is not None and stressed != previous_stressed
        )
        return calm_days, transitioned
    raise ValueError(f"Unsupported adaptive HMM refit mode: {mode}")


def should_update_hmm_templates(
    should_refit: bool,
    model_config: dict[str, object],
) -> bool:
    return should_refit if bool(model_config.get("template_update_on_refit_only", False)) else True


class RegimeBacktester:
    def __init__(self, config: dict[str, object]):
        self.config = config
        model_config = config["model"]
        self.estimator = CausalHMMEstimator(model_config)  # type: ignore[arg-type]
        self.tracker = WassersteinTemplateTracker(
            int(model_config["template_count"]),  # type: ignore[index]
            float(model_config["template_smoothing"]),  # type: ignore[index]
            str(model_config.get("template_assignment", "nearest")),  # type: ignore[union-attr]
            str(model_config.get("template_update_geometry", "euclidean_variance")),  # type: ignore[union-attr]
        )
        self.fit: RegimeFit | None = None
        self.ensemble_fits: dict[int, RegimeFit] = {}
        self.ensemble_trackers: dict[int, WassersteinTemplateTracker] = {}
        self.selected_order: int | None = None
        self.last_fit_position = -10**9
        self.last_selection_position = -10**9
        self.last_hmm_refit_stress: bool | None = None
        self.smoothed_favorable_probability: float | None = None
        self.paper_risk_on_state: bool | None = None
        self.previous_vix_backwardation: bool | None = None
        self.vix_recovery_bridge_active = False
        self.zero_growth_relative_price_start: float | None = None
        self.turning_point_reentry_episode_fired = False
        self.turning_point_zero_entry_signal_streak = 0
        self.turning_point_zero_entry_last_signal_date: pd.Timestamp | None = (
            None
        )
        self.last_daily_hmm_signal: bool | None = None
        self.last_daily_hmm_auxiliary_gate: bool | None = None
        self.last_risk_on_target: np.ndarray | None = None
        self.last_risk_off_target: np.ndarray | None = None
        self.daily_hmm_reentry_attempt_active = False
        self.daily_hmm_reentry_blocked = False
        self.last_scheduled_target: np.ndarray | None = None
        self.survival_overlay_state = SurvivalOverlayState()

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        data_config = self.config["data"]
        feature_config = self.config["features"]
        model_config = self.config["model"]
        portfolio_config = self.config["portfolio"]
        backtest_config = self.config["backtest"]
        regime_assets = list(data_config["regime_assets"])  # type: ignore[index]
        cash_asset = str(data_config["cash_asset"])  # type: ignore[index]
        assets = list(data_config["tickers"])  # type: ignore[index]
        cash_index = assets.index(cash_asset)
        annualization = int(backtest_config["annualization"])  # type: ignore[index]

        features = build_causal_features(
            prices,
            regime_assets,
            int(feature_config["volatility_days"]),  # type: ignore[index]
            int(feature_config["momentum_days"]),  # type: ignore[index]
            list(feature_config.get("derived_signals", [])),  # type: ignore[arg-type]
            annualization,
            list(feature_config.get("components", [])) or None,  # type: ignore[arg-type]
        )
        returns = simple_returns(prices[assets]).reindex(features.index)
        dates = features.index[features.index >= pd.Timestamp(backtest_config["oos_start"])]  # type: ignore[index]
        if len(dates) < 2:
            raise ValueError("The configured OOS period contains fewer than two observations")

        weights = np.zeros(len(assets), dtype=float)
        weights[cash_index] = 1.0
        equity = 1.0
        peak = 1.0
        rows: list[dict[str, float | int | pd.Timestamp]] = []
        weight_rows: list[np.ndarray] = []
        regime_rows: list[dict[str, float | int | pd.Timestamp]] = []
        rebalance_every = int(backtest_config["rebalance_every_days"])  # type: ignore[index]

        for oos_position, date in enumerate(dates):
            trading_cost = 0.0
            turnover = 0.0
            drawdown_before = equity / peak - 1.0
            schedule_position = (
                anchored_business_day_position(date)
                if bool(backtest_config.get("calendar_anchored_schedule", False))
                else oos_position
            )
            scheduled_rebalance = schedule_position % rebalance_every == 0
            if scheduled_rebalance:
                target, diagnostics = self._target_weights(
                    date,
                    features,
                    returns,
                    prices,
                    weights,
                    drawdown_before,
                    assets,
                    cash_index,
                    annualization,
                    model_config,  # type: ignore[arg-type]
                    portfolio_config,  # type: ignore[arg-type]
                )
                survival_recovery_enabled = (
                    self._survival_recovery_enabled(portfolio_config)
                )
                if survival_recovery_enabled:
                    self.last_scheduled_target = target.copy()
                    self.survival_overlay_state.begin_alpha_target(target)
                target, scheduled_survival_diagnostics = self._survival_target(
                    date,
                    features,
                    returns,
                    target,
                    assets,
                    cash_index,
                    annualization,
                    portfolio_config,  # type: ignore[arg-type]
                )
                if survival_recovery_enabled:
                    self.survival_overlay_state.observe_result(
                        self.last_scheduled_target,
                        bool(scheduled_survival_diagnostics["survival_active"]),
                    )
                diagnostics.update(scheduled_survival_diagnostics)
                if not survival_recovery_enabled:
                    self.last_scheduled_target = target.copy()
                self._cache_daily_hmm_target(
                    target,
                    portfolio_config,
                    diagnostics,
                )
                proposed_turnover = 0.5 * float(np.abs(target - weights).sum())
                if proposed_turnover >= float(portfolio_config["no_trade_turnover"]):  # type: ignore[index]
                    trading_cost, turnover = transaction_cost(
                        weights,
                        target,
                        float(portfolio_config["cost_bps_per_dollar_traded"]),  # type: ignore[index]
                    )
                    weights = target
                diagnostics["survival_executed"] = int(
                    bool(scheduled_survival_diagnostics["survival_active"])
                    and proposed_turnover
                    >= float(portfolio_config["no_trade_turnover"])  # type: ignore[index]
                )
                regime_rows.append({"date": date, **diagnostics})

            daily_hmm_target = weights.copy()
            daily_hmm_diagnostics = self._empty_daily_hmm_diagnostics()
            if not scheduled_rebalance:
                (
                    daily_hmm_target,
                    daily_hmm_diagnostics,
                ) = self._daily_hmm_filter_target(
                    date,
                    features,
                    returns,
                    prices,
                    weights,
                    assets,
                    cash_index,
                    annualization,
                    portfolio_config,  # type: ignore[arg-type]
                )
            (
                daily_control_target,
                turning_point_diagnostics,
            ) = self._turning_point_reentry_target(
                daily_hmm_target,
                returns.loc[returns.index < date],
                assets,
                cash_index,
                portfolio_config,  # type: ignore[arg-type]
                prices,
                date,
            )
            risk_target = daily_control_target
            daily_risk_volatility = float("nan")
            daily_risk_multiplier = 1.0
            daily_cvar_estimate = float("nan")
            daily_cvar_bear_regime = False
            asset_risk_config = portfolio_config.get("asset_aware_growth_risk")
            if isinstance(asset_risk_config, dict) and bool(
                asset_risk_config.get("enabled", False)
            ) and bool(asset_risk_config.get("daily_reduction", False)):
                trailing_risk_returns = returns.loc[returns.index < date].dropna(
                    how="any"
                )
                daily_recovery = bool(asset_risk_config.get("daily_recovery", False))
                daily_reallocation = bool(
                    asset_risk_config.get("daily_reallocation", False)
                )
                daily_controlled_reduction = bool(
                    asset_risk_config.get("daily_controlled_reduction", False)
                )
                if daily_controlled_reduction:
                    risk_target, daily_risk_volatility, daily_risk_multiplier = (
                        apply_daily_controlled_asset_risk_reduction(
                            risk_target,
                            trailing_risk_returns,
                            assets,
                            str(asset_risk_config["anchor_asset"]),
                            str(asset_risk_config["controlled_asset"]),
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            annualization,
                        )
                    )
                elif daily_reallocation and self.last_scheduled_target is not None:
                    risk_target, daily_risk_volatility, daily_risk_multiplier = (
                        apply_daily_growth_risk_reallocation_target(
                            risk_target,
                            self.last_scheduled_target,
                            trailing_risk_returns,
                            assets,
                            [
                                str(asset)
                                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                            ],
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            {
                                str(asset): float(value)
                                for asset, value in asset_risk_config.get(
                                    "maximum_core_weights", {}
                                ).items()
                            },
                            annualization,
                        )
                    )
                elif daily_recovery and self.last_scheduled_target is not None:
                    risk_target, daily_risk_volatility, daily_risk_multiplier = (
                        apply_daily_growth_risk_target(
                            risk_target,
                            self.last_scheduled_target,
                            trailing_risk_returns,
                            assets,
                            [
                                str(asset)
                                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                            ],
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            annualization,
                        )
                    )
                else:
                    risk_target, daily_risk_volatility, daily_risk_multiplier = (
                        apply_daily_growth_risk_reduction(
                            risk_target,
                            trailing_risk_returns,
                            assets,
                            [
                                str(asset)
                                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                            ],
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            annualization,
                            str(
                                asset_risk_config.get(
                                    "volatility_estimator", "standard"
                                )
                            ),
                            str(
                                asset_risk_config.get(
                                    "correlation_estimator", "slow"
                                )
                            ),
                        )
                    )
                trend_cvar_config = asset_risk_config.get("trend_cvar_switch")
                if isinstance(trend_cvar_config, dict) and bool(
                    trend_cvar_config.get("enabled", False)
                ):
                    (
                        risk_target,
                        daily_cvar_estimate,
                        cvar_multiplier,
                        daily_cvar_bear_regime,
                    ) = apply_trend_cvar_risk_reduction(
                        risk_target,
                        trailing_risk_returns,
                        assets,
                        [
                            str(asset)
                            for asset in asset_risk_config["assets"]
                        ],
                        cash_index,
                        int(trend_cvar_config.get("trend_lookback_days", 200)),
                        int(trend_cvar_config.get("fhs_lookback_days", 1000)),
                        float(trend_cvar_config.get("tail_probability", 0.005)),
                        float(asset_risk_config["target_volatility"]),
                        float(trend_cvar_config.get("ewma_decay", 0.94)),
                        annualization,
                    )
                    daily_risk_multiplier *= cvar_multiplier
            risk_turnover = 0.5 * float(np.abs(risk_target - weights).sum())
            daily_control_executed = risk_turnover >= float(
                portfolio_config["no_trade_turnover"]  # type: ignore[index]
            )
            if daily_control_executed:
                incremental_cost, incremental_turnover = transaction_cost(
                    weights,
                    risk_target,
                    float(portfolio_config["cost_bps_per_dollar_traded"]),  # type: ignore[index]
                )
                trading_cost += incremental_cost
                turnover += incremental_turnover
                weights = risk_target
            if bool(daily_hmm_diagnostics["daily_hmm_transition_active"]) and (
                daily_control_executed
            ):
                daily_signal = bool(
                    daily_hmm_diagnostics["daily_hmm_risk_on_candidate"]
                )
                self._observe_daily_hmm_transition(
                    daily_signal,
                    portfolio_config,  # type: ignore[arg-type]
                )
                self.last_daily_hmm_signal = daily_signal
                if self._daily_hmm_updates_scheduled_state(portfolio_config):
                    self.paper_risk_on_state = daily_signal
                    if daily_signal:
                        self.last_risk_on_target = risk_target.copy()
                    else:
                        self.last_risk_off_target = risk_target.copy()
            daily_hmm_diagnostics.pop("_daily_hmm_candidate_signal")
            turning_point_diagnostics["turning_point_reentry_executed"] = int(
                bool(turning_point_diagnostics["turning_point_reentry_active"])
                and daily_control_executed
            )
            if int(turning_point_diagnostics["turning_point_state"]) != 3:
                self.turning_point_reentry_episode_fired = False
            elif bool(
                turning_point_diagnostics["turning_point_reentry_executed"]
            ):
                self.turning_point_reentry_episode_fired = True
            turning_assets = turning_point_diagnostics.pop(
                "_turning_point_asset_indices"
            )
            turning_point_diagnostics["turning_point_growth_exposure"] = (
                float(weights[turning_assets].sum()) if turning_assets else 0.0
            )

            (
                standalone_risk_target,
                standalone_risk_diagnostics,
            ) = self._standalone_risk_cap_target(
                date,
                returns,
                weights,
                assets,
                cash_index,
                annualization,
                portfolio_config,  # type: ignore[arg-type]
            )
            standalone_risk_turnover = 0.5 * float(
                np.abs(standalone_risk_target - weights).sum()
            )
            if standalone_risk_turnover >= float(
                portfolio_config["no_trade_turnover"]  # type: ignore[index]
            ):
                incremental_cost, incremental_turnover = transaction_cost(
                    weights,
                    standalone_risk_target,
                    float(
                        portfolio_config["cost_bps_per_dollar_traded"]  # type: ignore[index]
                    ),
                )
                trading_cost += incremental_cost
                turnover += incremental_turnover
                weights = standalone_risk_target

            survival_input = weights
            survival_recovery_mode = False
            if self._survival_recovery_enabled(portfolio_config):
                (
                    survival_input,
                    survival_recovery_mode,
                ) = self.survival_overlay_state.target_input(weights)
            survival_target, survival_diagnostics = self._survival_target(
                date,
                features,
                returns,
                survival_input,
                assets,
                cash_index,
                annualization,
                portfolio_config,  # type: ignore[arg-type]
            )
            survival_turnover = 0.5 * float(
                np.abs(survival_target - weights).sum()
            )
            survival_executed = (
                survival_turnover
                >= float(portfolio_config["no_trade_turnover"])  # type: ignore[index]
            )
            if survival_executed:
                incremental_cost, incremental_turnover = transaction_cost(
                    weights,
                    survival_target,
                    float(portfolio_config["cost_bps_per_dollar_traded"]),  # type: ignore[index]
                )
                trading_cost += incremental_cost
                turnover += incremental_turnover
                weights = survival_target
            if self._survival_recovery_enabled(portfolio_config):
                self.survival_overlay_state.observe_result(
                    survival_input,
                    bool(survival_diagnostics["survival_active"]),
                )
            survival_diagnostics["survival_recovery_mode"] = int(
                survival_recovery_mode
            )

            (
                hedge_target,
                conditional_vix_hedge_active,
                conditional_vix_hedge_term_ratio,
            ) = self._conditional_vix_hedge_target(
                date,
                prices,
                weights,
                assets,
                cash_index,
                portfolio_config,  # type: ignore[arg-type]
            )
            hedge_turnover = 0.5 * float(np.abs(hedge_target - weights).sum())
            if hedge_turnover >= float(portfolio_config["no_trade_turnover"]):  # type: ignore[index]
                incremental_cost, incremental_turnover = transaction_cost(
                    weights,
                    hedge_target,
                    float(portfolio_config["cost_bps_per_dollar_traded"]),  # type: ignore[index]
                )
                trading_cost += incremental_cost
                turnover += incremental_turnover
                weights = hedge_target

            day_returns = returns.loc[date].to_numpy(dtype=float)
            gross_return = float(weights @ day_returns)
            financing_cost = financing_spread_cost(
                weights,
                cash_index,
                float(portfolio_config.get("financing_spread_bps", 0.0)),
                annualization,
                float(portfolio_config.get("short_borrow_spread_bps", 0.0)),
            )
            total_cost = trading_cost + financing_cost
            net_return = gross_return - total_cost
            equity *= 1.0 + net_return
            peak = max(peak, equity)
            drawdown = equity / peak - 1.0
            rows.append(
                {
                    "date": date,
                    "gross_return": gross_return,
                    "cost": total_cost,
                    "trading_cost": trading_cost,
                    "financing_cost": financing_cost,
                    "net_return": net_return,
                    "turnover": turnover,
                    "daily_risk_volatility": daily_risk_volatility,
                    "daily_risk_multiplier": daily_risk_multiplier,
                    "daily_cvar_estimate": daily_cvar_estimate,
                    "daily_cvar_bear_regime": int(daily_cvar_bear_regime),
                    **daily_hmm_diagnostics,
                    **turning_point_diagnostics,
                    **standalone_risk_diagnostics,
                    "standalone_risk_cap_executed": int(
                        standalone_risk_turnover
                        >= float(
                            portfolio_config["no_trade_turnover"]  # type: ignore[index]
                        )
                    ),
                    **survival_diagnostics,
                    "survival_executed": int(survival_executed),
                    "conditional_vix_hedge_active": int(
                        conditional_vix_hedge_active
                    ),
                    "conditional_vix_hedge_term_ratio": (
                        conditional_vix_hedge_term_ratio
                    ),
                    "conditional_vix_hedge_share": float(
                        weights[
                            assets.index(
                                str(
                                    portfolio_config.get(
                                        "conditional_vix_hedge", {}
                                    ).get("asset", cash_asset)  # type: ignore[union-attr]
                                )
                            )
                        ]
                        if isinstance(
                            portfolio_config.get("conditional_vix_hedge"), dict
                        )
                        else 0.0
                    ),
                    "equity": equity,
                    "drawdown": drawdown,
                }
            )
            weight_rows.append(weights.copy())

            # Let holdings drift until the next trade.
            gross_growth = 1.0 + gross_return
            if gross_growth > 1e-12:
                weights = weights * (1.0 + day_returns) / gross_growth
                risky = np.ones(len(weights), dtype=bool)
                risky[cash_index] = False
                weights[cash_index] = 1.0 - weights[risky].sum()
                weights /= weights.sum()

        daily = pd.DataFrame(rows).set_index("date")
        weight_frame = pd.DataFrame(weight_rows, index=dates, columns=assets)
        regime_frame = pd.DataFrame(regime_rows).set_index("date")
        benchmarks = self._benchmarks(returns.loc[dates], assets)
        return BacktestResult(daily, weight_frame, regime_frame, benchmarks, returns.loc[dates])

    def recommend_next(
        self,
        prices: pd.DataFrame,
        result: BacktestResult,
    ) -> tuple[pd.Series, dict[str, float | int | str]]:
        """Calculate the next-session target from the latest completed close."""
        data_config = self.config["data"]
        feature_config = self.config["features"]
        model_config = self.config["model"]
        portfolio_config = self.config["portfolio"]
        backtest_config = self.config["backtest"]
        regime_assets = list(data_config["regime_assets"])  # type: ignore[index]
        assets = list(data_config["tickers"])  # type: ignore[index]
        cash_index = assets.index(str(data_config["cash_asset"]))  # type: ignore[index]
        annualization = int(backtest_config["annualization"])  # type: ignore[index]
        all_features = build_causal_features(
            prices,
            regime_assets,
            int(feature_config["volatility_days"]),  # type: ignore[index]
            int(feature_config["momentum_days"]),  # type: ignore[index]
            list(feature_config.get("derived_signals", [])),  # type: ignore[arg-type]
            annualization,
            list(feature_config.get("components", [])) or None,  # type: ignore[arg-type]
        )
        next_feature = build_next_feature(
            prices,
            regime_assets,
            int(feature_config["volatility_days"]),  # type: ignore[index]
            int(feature_config["momentum_days"]),  # type: ignore[index]
            list(feature_config.get("derived_signals", [])),  # type: ignore[arg-type]
            annualization,
            list(feature_config.get("components", [])) or None,  # type: ignore[arg-type]
        )
        all_features = pd.concat([all_features, next_feature.to_frame().T])
        all_returns = simple_returns(prices[assets])

        last_date = result.weights.index[-1]
        start_weights = result.weights.loc[last_date].to_numpy(dtype=float)
        last_returns = all_returns.loc[last_date].to_numpy(dtype=float)
        gross_return = float(start_weights @ last_returns)
        current_weights = start_weights * (1.0 + last_returns) / (1.0 + gross_return)
        risky = np.ones(len(current_weights), dtype=bool)
        risky[cash_index] = False
        current_weights[cash_index] = 1.0 - current_weights[risky].sum()
        current_weights /= current_weights.sum()
        current_drawdown = float(result.daily.iloc[-1]["drawdown"])

        rebalance_every = int(backtest_config["rebalance_every_days"])  # type: ignore[index]
        last_rebalance = result.regimes.index[-1]
        sessions_since_rebalance = int((prices.index > last_rebalance).sum())
        sessions_to_rebalance = max(rebalance_every - sessions_since_rebalance, 1)
        next_rebalance = pd.Timestamp(next_feature.name) + pd.offsets.BDay(
            sessions_to_rebalance - 1
        )
        if sessions_since_rebalance + 1 < rebalance_every:
            cvar_estimate = float("nan")
            cvar_bear_regime = False
            (
                daily_hmm_target,
                daily_hmm_diagnostics,
            ) = self._daily_hmm_filter_target(
                pd.Timestamp(next_feature.name),
                all_features,
                all_returns,
                prices,
                current_weights,
                assets,
                cash_index,
                annualization,
                portfolio_config,  # type: ignore[arg-type]
            )
            (
                risk_target,
                turning_point_diagnostics,
            ) = self._turning_point_reentry_target(
                daily_hmm_target,
                all_returns,
                assets,
                cash_index,
                portfolio_config,  # type: ignore[arg-type]
                prices,
                pd.Timestamp(next_feature.name),
            )
            daily_hmm_diagnostics.pop("_daily_hmm_candidate_signal")
            turning_point_diagnostics.pop("_turning_point_asset_indices")
            risk_volatility = float("nan")
            risk_multiplier = 1.0
            asset_risk_config = portfolio_config.get("asset_aware_growth_risk")
            if isinstance(asset_risk_config, dict) and bool(
                asset_risk_config.get("enabled", False)
            ) and bool(asset_risk_config.get("daily_reduction", False)):
                daily_recovery = bool(asset_risk_config.get("daily_recovery", False))
                daily_reallocation = bool(
                    asset_risk_config.get("daily_reallocation", False)
                )
                daily_controlled_reduction = bool(
                    asset_risk_config.get("daily_controlled_reduction", False)
                )
                if daily_controlled_reduction:
                    risk_target, risk_volatility, risk_multiplier = (
                        apply_daily_controlled_asset_risk_reduction(
                            risk_target,
                            all_returns.dropna(how="any"),
                            assets,
                            str(asset_risk_config["anchor_asset"]),
                            str(asset_risk_config["controlled_asset"]),
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            annualization,
                        )
                    )
                elif daily_reallocation and self.last_scheduled_target is not None:
                    risk_target, risk_volatility, risk_multiplier = (
                        apply_daily_growth_risk_reallocation_target(
                            risk_target,
                            self.last_scheduled_target,
                            all_returns.dropna(how="any"),
                            assets,
                            [
                                str(asset)
                                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                            ],
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            {
                                str(asset): float(value)
                                for asset, value in asset_risk_config.get(
                                    "maximum_core_weights", {}
                                ).items()
                            },
                            annualization,
                        )
                    )
                elif daily_recovery and self.last_scheduled_target is not None:
                    risk_target, risk_volatility, risk_multiplier = (
                        apply_daily_growth_risk_target(
                            risk_target,
                            self.last_scheduled_target,
                            all_returns.dropna(how="any"),
                            assets,
                            [
                                str(asset)
                                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                            ],
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            annualization,
                        )
                    )
                else:
                    risk_target, risk_volatility, risk_multiplier = (
                        apply_daily_growth_risk_reduction(
                            risk_target,
                            all_returns.dropna(how="any"),
                            assets,
                            [
                                str(asset)
                                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                            ],
                            cash_index,
                            int(asset_risk_config["fast_days"]),
                            int(asset_risk_config["slow_days"]),
                            float(asset_risk_config["target_volatility"]),
                            annualization,
                            str(
                                asset_risk_config.get(
                                    "volatility_estimator", "standard"
                                )
                            ),
                            str(
                                asset_risk_config.get(
                                    "correlation_estimator", "slow"
                                )
                            ),
                        )
                    )
                trend_cvar_config = asset_risk_config.get("trend_cvar_switch")
                if isinstance(trend_cvar_config, dict) and bool(
                    trend_cvar_config.get("enabled", False)
                ):
                    (
                        risk_target,
                        cvar_estimate,
                        cvar_multiplier,
                        cvar_bear_regime,
                    ) = apply_trend_cvar_risk_reduction(
                        risk_target,
                        all_returns.dropna(how="any"),
                        assets,
                        [
                            str(asset)
                            for asset in asset_risk_config["assets"]
                        ],
                        cash_index,
                        int(trend_cvar_config.get("trend_lookback_days", 200)),
                        int(trend_cvar_config.get("fhs_lookback_days", 1000)),
                        float(trend_cvar_config.get("tail_probability", 0.005)),
                        float(asset_risk_config["target_volatility"]),
                        float(trend_cvar_config.get("ewma_decay", 0.94)),
                        annualization,
                    )
                    risk_multiplier *= cvar_multiplier
            risk_turnover = 0.5 * float(
                np.abs(risk_target - current_weights).sum()
            )
            if risk_turnover >= float(portfolio_config["no_trade_turnover"]):  # type: ignore[index]
                growth_assets = (
                    [
                        str(asset)
                        for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                    ]
                    if isinstance(asset_risk_config, dict)
                    else [
                        str(asset)
                        for asset in portfolio_config.get(
                            "turning_point_reentry", {}
                        ).get("core_weights", {})  # type: ignore[union-attr]
                    ]
                )
                growth_indices = [assets.index(asset) for asset in growth_assets]
                current_growth = float(current_weights[growth_indices].sum())
                target_growth = float(risk_target[growth_indices].sum())
                diagnostics = {
                    "action": (
                        "DAILY_HMM_REENTRY"
                        if bool(
                            daily_hmm_diagnostics[
                                "daily_hmm_transition_active"
                            ]
                        )
                        and bool(
                            daily_hmm_diagnostics[
                                "daily_hmm_risk_on_candidate"
                            ]
                        )
                        else (
                            "DAILY_HMM_EXIT"
                            if bool(
                                daily_hmm_diagnostics[
                                    "daily_hmm_transition_active"
                                ]
                            )
                            else (
                                "TURNING_POINT_REENTRY"
                                if bool(
                                    turning_point_diagnostics[
                                        "turning_point_reentry_active"
                                    ]
                                )
                                and target_growth > current_growth
                                else (
                                    "RISK_RESTORE"
                                    if target_growth > current_growth
                                    else "RISK_REDUCE"
                                )
                            )
                        )
                    ),
                    "rebalance_due": 0,
                    "sessions_since_rebalance": sessions_since_rebalance,
                    "sessions_to_rebalance": sessions_to_rebalance,
                    "estimated_rebalance_date": next_rebalance.date().isoformat(),
                    "daily_risk_volatility": risk_volatility,
                    "daily_risk_multiplier": risk_multiplier,
                    "daily_cvar_estimate": cvar_estimate,
                    "daily_cvar_bear_regime": int(cvar_bear_regime),
                    **daily_hmm_diagnostics,
                    **turning_point_diagnostics,
                }
                return self._recommend_with_conditional_vix_hedge(
                    risk_target,
                    pd.Timestamp(next_feature.name),
                    prices,
                    assets,
                    cash_index,
                    portfolio_config,  # type: ignore[arg-type]
                    all_features,
                    all_returns,
                    annualization,
                    diagnostics,
                )
            diagnostics: dict[str, float | int | str] = {
                "action": "HOLD",
                "rebalance_due": 0,
                "sessions_since_rebalance": sessions_since_rebalance,
                "sessions_to_rebalance": sessions_to_rebalance,
                "estimated_rebalance_date": next_rebalance.date().isoformat(),
                "daily_cvar_estimate": cvar_estimate,
                "daily_cvar_bear_regime": int(cvar_bear_regime),
                **daily_hmm_diagnostics,
                **turning_point_diagnostics,
            }
            return self._recommend_with_conditional_vix_hedge(
                current_weights,
                pd.Timestamp(next_feature.name),
                prices,
                assets,
                cash_index,
                portfolio_config,  # type: ignore[arg-type]
                all_features,
                all_returns,
                annualization,
                diagnostics,
            )

        target, diagnostics = self._target_weights(
            pd.Timestamp(next_feature.name),
            all_features,
            all_returns,
            prices,
            current_weights,
            current_drawdown,
            assets,
            cash_index,
            int(backtest_config["annualization"]),  # type: ignore[index]
            model_config,  # type: ignore[arg-type]
            portfolio_config,  # type: ignore[arg-type]
        )
        self.last_scheduled_target = target.copy()
        target, turning_point_diagnostics = self._turning_point_reentry_target(
            target,
            all_returns,
            assets,
            cash_index,
            portfolio_config,  # type: ignore[arg-type]
            prices,
            pd.Timestamp(next_feature.name),
        )
        turning_point_diagnostics.pop("_turning_point_asset_indices")
        diagnostics.update(turning_point_diagnostics)
        asset_risk_config = portfolio_config.get("asset_aware_growth_risk")
        if isinstance(asset_risk_config, dict) and bool(
            asset_risk_config.get("enabled", False)
        ) and bool(asset_risk_config.get("daily_reduction", False)):
            daily_recovery = bool(asset_risk_config.get("daily_recovery", False))
            daily_reallocation = bool(
                asset_risk_config.get("daily_reallocation", False)
            )
            daily_controlled_reduction = bool(
                asset_risk_config.get("daily_controlled_reduction", False)
            )
            if daily_controlled_reduction:
                target, risk_volatility, risk_multiplier = (
                    apply_daily_controlled_asset_risk_reduction(
                        target,
                        all_returns.dropna(how="any"),
                        assets,
                        str(asset_risk_config["anchor_asset"]),
                        str(asset_risk_config["controlled_asset"]),
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        float(asset_risk_config["target_volatility"]),
                        annualization,
                    )
                )
            elif daily_reallocation:
                target, risk_volatility, risk_multiplier = (
                    apply_daily_growth_risk_reallocation_target(
                        target,
                        self.last_scheduled_target,
                        all_returns.dropna(how="any"),
                        assets,
                        [
                            str(asset)
                            for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                        ],
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        float(asset_risk_config["target_volatility"]),
                        {
                            str(asset): float(value)
                            for asset, value in asset_risk_config.get(
                                "maximum_core_weights", {}
                            ).items()
                        },
                        annualization,
                    )
                )
            elif daily_recovery:
                target, risk_volatility, risk_multiplier = (
                    apply_daily_growth_risk_target(
                        target,
                        self.last_scheduled_target,
                        all_returns.dropna(how="any"),
                        assets,
                        [
                            str(asset)
                            for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                        ],
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        float(asset_risk_config["target_volatility"]),
                        annualization,
                    )
                )
            else:
                target, risk_volatility, risk_multiplier = (
                    apply_daily_growth_risk_reduction(
                        target,
                        all_returns.dropna(how="any"),
                        assets,
                        [
                            str(asset)
                            for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
                        ],
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        float(asset_risk_config["target_volatility"]),
                        annualization,
                        str(
                            asset_risk_config.get(
                                "volatility_estimator", "standard"
                            )
                        ),
                        str(
                            asset_risk_config.get(
                                "correlation_estimator", "slow"
                            )
                        ),
                )
            )
            trend_cvar_config = asset_risk_config.get("trend_cvar_switch")
            cvar_estimate = float("nan")
            cvar_bear_regime = False
            if isinstance(trend_cvar_config, dict) and bool(
                trend_cvar_config.get("enabled", False)
            ):
                (
                    target,
                    cvar_estimate,
                    cvar_multiplier,
                    cvar_bear_regime,
                ) = apply_trend_cvar_risk_reduction(
                    target,
                    all_returns.dropna(how="any"),
                    assets,
                    [
                        str(asset)
                        for asset in asset_risk_config["assets"]
                    ],
                    cash_index,
                    int(trend_cvar_config.get("trend_lookback_days", 200)),
                    int(trend_cvar_config.get("fhs_lookback_days", 1000)),
                    float(trend_cvar_config.get("tail_probability", 0.005)),
                    float(asset_risk_config["target_volatility"]),
                    float(trend_cvar_config.get("ewma_decay", 0.94)),
                    annualization,
                )
                risk_multiplier *= cvar_multiplier
            diagnostics["daily_risk_volatility"] = risk_volatility
            diagnostics["daily_risk_multiplier"] = risk_multiplier
            diagnostics["daily_cvar_estimate"] = cvar_estimate
            diagnostics["daily_cvar_bear_regime"] = int(cvar_bear_regime)
        diagnostics.update(
            {
                "action": "REBALANCE",
                "rebalance_due": 1,
                "sessions_since_rebalance": sessions_since_rebalance,
                "sessions_to_rebalance": 0,
                "estimated_rebalance_date": pd.Timestamp(next_feature.name).date().isoformat(),
            }
        )
        return self._recommend_with_conditional_vix_hedge(
            target,
            pd.Timestamp(next_feature.name),
            prices,
            assets,
            cash_index,
            portfolio_config,  # type: ignore[arg-type]
            all_features,
            all_returns,
            annualization,
            diagnostics,
        )

    @staticmethod
    def _empty_daily_hmm_diagnostics() -> dict[str, object]:
        return {
            "daily_hmm_filter_enabled": 0,
            "daily_hmm_favorable_probability": float("nan"),
            "daily_hmm_growth_excess_return": float("nan"),
            "daily_hmm_risk_on_candidate": 0,
            "daily_hmm_transition_active": 0,
            "daily_hmm_reentry_mode": 0,
            "daily_hmm_recovery_quality_enabled": 0,
            "daily_hmm_recovery_quality_confirmations": 0,
            "daily_hmm_recovery_quality_required": 0,
            "daily_hmm_recovery_quality_pass": 0,
            "daily_hmm_recovery_quality_gate_applied": 0,
            "daily_hmm_recovery_quality_signal_count": 0,
            "daily_hmm_recovery_breadth_return": float("nan"),
            "daily_hmm_recovery_credit_return": float("nan"),
            "daily_hmm_reentry_blend_fraction": float("nan"),
            "daily_hmm_episode_memory_enabled": 0,
            "daily_hmm_episode_memory_blocked": 0,
            "_daily_hmm_candidate_signal": None,
        }

    def _cache_daily_hmm_target(
        self,
        target: np.ndarray,
        portfolio_config: dict[str, object],
        diagnostics: dict[str, object],
    ) -> None:
        config = portfolio_config.get("daily_hmm_state_filter")
        if not isinstance(config, dict) or not bool(
            config.get("enabled", False)
        ):
            return
        if self.paper_risk_on_state is None:
            return
        leverage_signal_score = float(
            diagnostics.get("leverage_signal_score", float("-inf"))
        )
        self.last_daily_hmm_auxiliary_gate = bool(
            portfolio_config.get("allow_leverage", False)
            and not bool(diagnostics.get("trend_stress", 1))
            and leverage_signal_score
            >= float(portfolio_config.get("leverage_activation_score", 0.0))
            and not (
                bool(
                    portfolio_config.get(
                        "vix_term_structure_guard", {}
                    ).get("hard_veto", True)  # type: ignore[union-attr]
                )
                and bool(diagnostics.get("vix_backwardation", 0))
            )
        )
        signal = bool(self.paper_risk_on_state)
        if self._daily_hmm_episode_memory_enabled(portfolio_config):
            if signal:
                self.daily_hmm_reentry_attempt_active = False
                self.daily_hmm_reentry_blocked = False
            elif (
                self.last_daily_hmm_signal is True
                and self.daily_hmm_reentry_attempt_active
            ):
                self.daily_hmm_reentry_attempt_active = False
                self.daily_hmm_reentry_blocked = True
        self.last_daily_hmm_signal = signal
        if signal:
            self.last_risk_on_target = target.copy()
        else:
            self.last_risk_off_target = target.copy()

    def _daily_hmm_filter_target(
        self,
        date: pd.Timestamp,
        all_features: pd.DataFrame,
        all_returns: pd.DataFrame,
        all_prices: pd.DataFrame,
        weights: np.ndarray,
        assets: list[str],
        cash_index: int,
        annualization: int,
        portfolio_config: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, object]]:
        config = portfolio_config.get("daily_hmm_state_filter")
        diagnostics = self._empty_daily_hmm_diagnostics()
        if not isinstance(config, dict) or not bool(
            config.get("enabled", False)
        ):
            return weights.copy(), diagnostics
        diagnostics["daily_hmm_filter_enabled"] = 1
        if (
            self.fit is None
            or self.last_daily_hmm_signal is None
            or self.last_daily_hmm_auxiliary_gate is None
        ):
            return weights.copy(), diagnostics
        if self.ensemble_fits:
            raise ValueError(
                "Daily HMM state filtering does not support order ensembles"
            )

        excluded_assets = {
            str(asset)
            for asset in portfolio_config.get("optimizer_excluded_assets", [])
        }
        core_indices = [
            index for index, asset in enumerate(assets)
            if asset not in excluded_assets
        ]
        core_assets = [assets[index] for index in core_indices]
        core_cash_index = core_assets.index(assets[cash_index])
        filter_start = self.fit.training_features.index[0]
        filter_data = all_features.loc[
            (all_features.index >= filter_start)
            & (all_features.index <= date)
        ]
        state_probabilities = self.fit.filtered_probabilities(filter_data)
        conditional_mean = (
            state_probabilities @ self.fit.state_return_means
        )
        growth_assets = [
            str(asset)
            for asset in portfolio_config["paper_growth_assets"]  # type: ignore[index]
        ]
        growth_weights = np.asarray(
            portfolio_config.get(
                "paper_growth_weights",
                [1.0 / len(growth_assets)] * len(growth_assets),
            ),
            dtype=float,
        )
        growth_weights /= growth_weights.sum()
        growth_indices = [core_assets.index(asset) for asset in growth_assets]
        growth_state_means = self.fit.state_return_means[:, growth_indices]
        state_growth_mean = growth_state_means @ growth_weights
        state_cash_mean = self.fit.state_return_means[:, core_cash_index]
        favorable_probability = float(
            state_probabilities[state_growth_mean > state_cash_mean].sum()
        )
        growth_expected_return = float(
            annualization
            * (
                growth_weights @ conditional_mean[growth_indices]
                - conditional_mean[core_cash_index]
            )
        )

        base_candidate = bool(
            growth_expected_return > 0.0
            and self.last_daily_hmm_auxiliary_gate
        )
        candidate = base_candidate
        quality_diagnostics: dict[str, object] = {}
        if base_candidate and self.last_daily_hmm_signal is False:
            candidate, quality_diagnostics = (
                self._daily_hmm_recovery_quality(
                    date,
                    all_prices,
                    config,
                )
            )
        diagnostics.update(quality_diagnostics)
        memory_enabled = self._daily_hmm_episode_memory_enabled(
            portfolio_config
        )
        diagnostics["daily_hmm_episode_memory_enabled"] = int(
            memory_enabled
        )
        diagnostics["daily_hmm_episode_memory_blocked"] = int(
            memory_enabled and self.daily_hmm_reentry_blocked
        )
        if (
            candidate
            and self.last_daily_hmm_signal is False
            and memory_enabled
            and self.daily_hmm_reentry_blocked
        ):
            candidate = False

        diagnostics.update(
            {
                "daily_hmm_favorable_probability": favorable_probability,
                "daily_hmm_growth_excess_return": growth_expected_return,
                "daily_hmm_risk_on_candidate": int(candidate),
                "_daily_hmm_candidate_signal": candidate,
            }
        )
        if (
            not candidate
            and not self._daily_hmm_exit_allowed(
                bool(self.paper_risk_on_state),
                portfolio_config,
            )
        ):
            return weights.copy(), diagnostics
        if candidate == self.last_daily_hmm_signal:
            return weights.copy(), diagnostics

        if candidate:
            mode = str(config.get("reentry_mode", "full_cached_target"))
            if mode == "full_cached_target":
                cached_target = self.last_risk_on_target
                diagnostics["daily_hmm_reentry_mode"] = 1
            elif mode == "qqq_first":
                cached_target = self._qqq_first_reentry_target(
                    weights,
                    assets,
                    cash_index,
                    portfolio_config,
                    str(config.get("first_stage_asset", "QQQ")),
                )
                diagnostics["daily_hmm_reentry_mode"] = 2
            elif mode == "confidence_blend":
                cached_target = self.last_risk_on_target
                if cached_target is not None:
                    (
                        cached_target,
                        blend_fraction,
                    ) = self._confidence_blended_reentry_target(
                        weights,
                        cached_target,
                        int(
                            diagnostics.get(
                                "daily_hmm_recovery_quality_confirmations",
                                0,
                            )
                        ),
                        int(
                            diagnostics.get(
                                "daily_hmm_recovery_quality_signal_count",
                                0,
                            )
                        ),
                    )
                    diagnostics[
                        "daily_hmm_reentry_blend_fraction"
                    ] = blend_fraction
                diagnostics["daily_hmm_reentry_mode"] = 3
            else:
                raise ValueError(f"Unknown daily HMM re-entry mode: {mode}")
        else:
            cached_target = self.last_risk_off_target

        if cached_target is None:
            return weights.copy(), diagnostics
        diagnostics["daily_hmm_transition_active"] = int(
            not np.allclose(cached_target, weights)
        )
        return cached_target.copy(), diagnostics

    @staticmethod
    def _daily_hmm_episode_memory_enabled(
        portfolio_config: dict[str, object],
    ) -> bool:
        daily_config = portfolio_config.get("daily_hmm_state_filter")
        if not isinstance(daily_config, dict):
            return False
        memory_config = daily_config.get("episode_memory")
        return isinstance(memory_config, dict) and bool(
            memory_config.get("enabled", False)
        )

    @staticmethod
    def _daily_hmm_exit_allowed(
        scheduled_risk_on: bool,
        portfolio_config: dict[str, object],
    ) -> bool:
        daily_config = portfolio_config.get("daily_hmm_state_filter")
        if not isinstance(daily_config, dict):
            return True
        return not (
            bool(daily_config.get("entry_only", False))
            and scheduled_risk_on
        )

    @staticmethod
    def _daily_hmm_updates_scheduled_state(
        portfolio_config: dict[str, object],
    ) -> bool:
        daily_config = portfolio_config.get("daily_hmm_state_filter")
        if not isinstance(daily_config, dict):
            return True
        update_mode = str(
            daily_config.get("state_update_mode", "daily_and_scheduled")
        )
        if update_mode not in {"daily_and_scheduled", "scheduled_only"}:
            raise ValueError(
                "Daily HMM state-update mode must be "
                "'daily_and_scheduled' or 'scheduled_only'"
            )
        return update_mode == "daily_and_scheduled"

    def _observe_daily_hmm_transition(
        self,
        signal: bool,
        portfolio_config: dict[str, object],
    ) -> None:
        if not self._daily_hmm_episode_memory_enabled(portfolio_config):
            return
        if signal:
            self.daily_hmm_reentry_attempt_active = True
            return
        if self.daily_hmm_reentry_attempt_active:
            self.daily_hmm_reentry_attempt_active = False
            self.daily_hmm_reentry_blocked = True

    @staticmethod
    def _daily_hmm_recovery_quality(
        date: pd.Timestamp,
        all_prices: pd.DataFrame,
        daily_config: dict[str, object],
    ) -> tuple[bool, dict[str, object]]:
        quality_config = daily_config.get("recovery_quality")
        if not isinstance(quality_config, dict) or not bool(
            quality_config.get("enabled", False)
        ):
            return True, {}
        lookback_days = int(quality_config.get("lookback_days", 21))
        if lookback_days <= 0:
            raise ValueError(
                "Daily HMM recovery-quality lookback must be positive"
            )
        signal_specs = quality_config.get("signals", {})
        if not isinstance(signal_specs, dict) or not signal_specs:
            raise ValueError(
                "Daily HMM recovery quality requires named signals"
            )
        history = all_prices.loc[all_prices.index < date]
        confirmations = 0
        diagnostics: dict[str, object] = {
            "daily_hmm_recovery_quality_enabled": 1,
            "daily_hmm_recovery_quality_signal_count": len(signal_specs),
        }
        for name, raw_specification in signal_specs.items():
            if not isinstance(raw_specification, dict):
                raise ValueError(
                    "Daily HMM recovery-quality signals must be mappings"
                )
            numerator = str(raw_specification["numerator"])
            denominator = str(raw_specification["denominator"])
            missing = [
                asset
                for asset in (numerator, denominator)
                if asset not in history
            ]
            if missing:
                raise ValueError(
                    "Daily HMM recovery quality is missing price inputs: "
                    f"{missing}"
                )
            clean = history[[numerator, denominator]].dropna(how="any")
            if len(clean) <= lookback_days:
                relative_return = float("nan")
            else:
                latest_ratio = float(
                    clean[numerator].iloc[-1]
                    / clean[denominator].iloc[-1]
                )
                prior_ratio = float(
                    clean[numerator].iloc[-lookback_days - 1]
                    / clean[denominator].iloc[-lookback_days - 1]
                )
                relative_return = latest_ratio / prior_ratio - 1.0
                confirmations += int(relative_return > 0.0)
            diagnostics[
                f"daily_hmm_recovery_{str(name)}_return"
            ] = relative_return
        required = int(
            quality_config.get("minimum_confirmations", len(signal_specs))
        )
        if not 1 <= required <= len(signal_specs):
            raise ValueError(
                "Daily HMM recovery-quality confirmations are out of range"
            )
        passed = confirmations >= required
        decision_mode = str(quality_config.get("decision_mode", "hard_gate"))
        if decision_mode not in {"hard_gate", "soft"}:
            raise ValueError(
                "Daily HMM recovery-quality decision mode must be "
                "'hard_gate' or 'soft'"
            )
        diagnostics.update(
            {
                "daily_hmm_recovery_quality_confirmations": confirmations,
                "daily_hmm_recovery_quality_required": required,
                "daily_hmm_recovery_quality_pass": int(passed),
                "daily_hmm_recovery_quality_gate_applied": int(
                    decision_mode == "hard_gate"
                ),
            }
        )
        return passed if decision_mode == "hard_gate" else True, diagnostics

    @staticmethod
    def _confidence_blended_reentry_target(
        weights: np.ndarray,
        cached_target: np.ndarray,
        confirmations: int,
        signal_count: int,
    ) -> tuple[np.ndarray, float]:
        """Shrink a daily re-entry toward the last scheduled risk-on target."""
        if weights.shape != cached_target.shape:
            raise ValueError("Re-entry weights and cached target must align")
        if signal_count < 0 or not 0 <= confirmations <= signal_count:
            raise ValueError("Re-entry confirmation counts are invalid")
        directional_votes = 1 + confirmations
        total_votes = 1 + signal_count
        blend_fraction = (1.0 + directional_votes) / (
            2.0 + total_votes
        )
        blended = weights + blend_fraction * (cached_target - weights)
        return blended / blended.sum(), float(blend_fraction)

    @staticmethod
    def _qqq_first_reentry_target(
        weights: np.ndarray,
        assets: list[str],
        cash_index: int,
        portfolio_config: dict[str, object],
        first_stage_asset: str,
    ) -> np.ndarray:
        growth_assets = [
            str(asset)
            for asset in portfolio_config["paper_growth_assets"]  # type: ignore[index]
        ]
        if first_stage_asset not in growth_assets:
            raise ValueError(
                "First-stage asset must belong to paper growth assets"
            )
        strategic_weights = portfolio_config["strategic_core_weights"]
        first_stage_share = float(
            strategic_weights.get(first_stage_asset, 0.0)  # type: ignore[union-attr]
        )
        if not 0.0 < first_stage_share < 1.0:
            raise ValueError(
                "First-stage strategic account weight must be between zero and one"
            )
        defensive = apply_asset_multiplier(
            weights,
            assets,
            growth_assets,
            cash_index,
            0.0,
        )
        first_stage_core = np.zeros(len(assets), dtype=float)
        first_stage_core[assets.index(first_stage_asset)] = 1.0
        return compose_probability_allocation(
            defensive,
            first_stage_core,
            cash_index,
            first_stage_share,
        )

    def _turning_point_reentry_target(
        self,
        weights: np.ndarray,
        trailing_returns: pd.DataFrame,
        assets: list[str],
        cash_index: int,
        portfolio_config: dict[str, object],
        signal_prices: pd.DataFrame | None = None,
        signal_date: pd.Timestamp | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        config = portfolio_config.get("turning_point_reentry")
        default_diagnostics: dict[str, object] = {
            "turning_point_state": -1,
            "turning_point_fast_excess_return": float("nan"),
            "turning_point_slow_excess_return": float("nan"),
            "turning_point_requested_share": 0.0,
            "turning_point_base_floor_gate_pass": 0,
            "turning_point_reentry_active": 0,
            "turning_point_volatility_state": -1,
            "turning_point_realized_volatility": float("nan"),
            "turning_point_volatility_override_gate_pass": 1,
            "turning_point_zero_entry_state": -1,
            "turning_point_zero_entry_requested_share": 0.0,
            "turning_point_zero_entry_active": 0,
            "turning_point_zero_entry_market_signal_active": 0,
            "turning_point_zero_entry_signal_streak": 0,
            "turning_point_zero_entry_path_gate_pass": 1,
            "turning_point_zero_entry_tilt_weight": float("nan"),
            "turning_point_zero_entry_volatility_state": -1,
            "turning_point_zero_entry_downside_variation_share": float("nan"),
            "turning_point_zero_entry_jump_variation_share": float("nan"),
            "turning_point_zero_entry_relief_return": float("nan"),
            "turning_point_zero_entry_jump_gate_pass": 1,
            "turning_point_zero_entry_relief_gate_pass": 1,
            "_turning_point_asset_indices": [],
        }
        if not isinstance(config, dict) or not bool(
            config.get("enabled", False)
        ):
            return weights.copy(), default_diagnostics

        core_weights = {
            str(asset): float(value)
            for asset, value in config["core_weights"].items()  # type: ignore[union-attr]
        }
        unknown_assets = [asset for asset in core_weights if asset not in assets]
        if unknown_assets:
            raise ValueError(
                f"Turning-point re-entry is missing assets: {unknown_assets}"
            )
        if (
            not core_weights
            or any(value < 0.0 for value in core_weights.values())
            or not np.isclose(sum(core_weights.values()), 1.0)
        ):
            raise ValueError("Turning-point core weights must sum to one")
        if assets[cash_index] in core_weights:
            raise ValueError("Turning-point growth core cannot contain cash")

        growth_exposure_assets_value = config.get("growth_exposure_assets")
        growth_exposure_assets = (
            [str(asset) for asset in growth_exposure_assets_value]
            if isinstance(growth_exposure_assets_value, list)
            else list(core_weights)
        )
        if (
            not growth_exposure_assets
            or len(set(growth_exposure_assets)) != len(
                growth_exposure_assets
            )
            or any(asset not in assets for asset in growth_exposure_assets)
            or assets[cash_index] in growth_exposure_assets
        ):
            raise ValueError(
                "Turning-point growth exposure assets must be unique, "
                "known, non-cash assets"
            )
        positive_core_assets = {
            asset for asset, value in core_weights.items() if value > 1e-12
        }
        if not positive_core_assets.issubset(growth_exposure_assets):
            raise ValueError(
                "Turning-point core weights must remain inside the growth "
                "exposure group"
            )
        growth_indices = [
            assets.index(asset) for asset in growth_exposure_assets
        ]
        core = np.zeros(len(assets), dtype=float)
        for asset, value in core_weights.items():
            core[assets.index(asset)] = value

        signal_core_weights = {
            str(asset): float(value)
            for asset, value in config.get(
                "signal_core_weights",
                core_weights,
            ).items()  # type: ignore[union-attr]
        }
        if (
            not signal_core_weights
            or any(asset not in assets for asset in signal_core_weights)
            or any(value < 0.0 for value in signal_core_weights.values())
            or not np.isclose(sum(signal_core_weights.values()), 1.0)
            or assets[cash_index] in signal_core_weights
        ):
            raise ValueError(
                "Turning-point signal core weights must contain known "
                "non-cash assets and sum to one"
            )
        growth_returns = trailing_returns[list(signal_core_weights)].mul(
            pd.Series(signal_core_weights)
        ).sum(axis=1, min_count=len(signal_core_weights))
        state, fast_return, slow_return = turning_point_cycle(
            growth_returns,
            trailing_returns[assets[cash_index]],
            int(config["fast_days"]),
            int(config["slow_days"]),
        )
        state_codes = {
            "insufficient": -1,
            "bull": 0,
            "correction": 1,
            "bear": 2,
            "rebound": 3,
        }
        state_shares = {
            str(name): float(value)
            for name, value in config.get("state_account_shares", {}).items()
        }
        if any(not 0.0 <= value <= 1.0 for value in state_shares.values()):
            raise ValueError(
                "Turning-point state account shares must be between zero and one"
            )
        requested_share = float(state_shares.get(state, 0.0))
        volatility_state = "insufficient"
        realized_volatility = float("nan")
        volatility_config = config.get("volatility_conditioning")
        if isinstance(volatility_config, dict) and bool(
            volatility_config.get("enabled", False)
        ):
            volatility_state, realized_volatility = (
                causal_realized_volatility_state(
                    growth_returns,
                    int(volatility_config.get("window_days", 20)),
                    int(
                        volatility_config.get(
                            "threshold_lookback_days",
                            756,
                        )
                    ),
                    int(
                        volatility_config.get(
                            "minimum_threshold_observations",
                            504,
                        )
                    ),
                    float(volatility_config.get("low_quantile", 0.45)),
                    float(volatility_config.get("high_quantile", 0.90)),
                    estimator=str(
                        volatility_config.get("estimator", "standard")
                    ),
                )
            )
            volatility_shares = volatility_config.get(
                "state_account_shares",
                {},
            )
            if isinstance(volatility_shares, dict):
                state_volatility_shares = volatility_shares.get(state)
                maximum_slow_returns = volatility_config.get(
                    "state_override_maximum_slow_excess_return",
                    {},
                )
                maximum_slow_return = (
                    maximum_slow_returns.get(state)
                    if isinstance(maximum_slow_returns, dict)
                    else None
                )
                volatility_override_gate_pass = (
                    maximum_slow_return is None
                    or slow_return <= float(maximum_slow_return)
                )
                if (
                    isinstance(state_volatility_shares, dict)
                    and volatility_override_gate_pass
                ):
                    requested_share = float(
                        state_volatility_shares.get(
                            volatility_state,
                            requested_share,
                        )
                    )
        current_growth = float(weights[growth_indices].sum())
        minimum_current_growth = float(
            config.get("minimum_current_growth_exposure", 0.0)
        )
        if not 0.0 <= minimum_current_growth <= 1.0:
            raise ValueError(
                "Turning-point minimum current growth exposure must be "
                "between zero and one"
            )
        base_floor_gate_pass = (
            current_growth + 1e-12 >= minimum_current_growth
        )
        risk_off_only = bool(config.get("activate_only_when_risk_off", True))
        signal_active = requested_share > 0.0 and (
            not risk_off_only or self.paper_risk_on_state is False
        ) and base_floor_gate_pass and not (
            bool(config.get("one_shot_per_rebound", False))
            and self.turning_point_reentry_episode_fired
        )
        target = weights.copy()
        applied_share = current_growth
        if signal_active:
            if isinstance(growth_exposure_assets_value, list):
                growth_group_mask = np.zeros(len(assets), dtype=bool)
                growth_group_mask[growth_indices] = True
                target, applied_share = (
                    ensure_minimum_growth_group_exposure(
                        weights,
                        core,
                        growth_group_mask,
                        cash_index,
                        requested_share,
                    )
                )
            else:
                target, applied_share = ensure_minimum_growth_exposure(
                    weights,
                    core,
                    cash_index,
                    requested_share,
                )
        floor_binding = bool(
            signal_active and applied_share > current_growth + 1e-12
        )
        zero_entry_state = "insufficient"
        zero_entry_requested_share = 0.0
        zero_entry_active = False
        zero_entry_market_signal_active = False
        zero_entry_signal_streak = 0
        zero_entry_path_gate_pass = True
        zero_entry_tilt_weight = float("nan")
        zero_entry_volatility_state = "insufficient"
        zero_entry_downside_variation_share = float("nan")
        zero_entry_jump_variation_share = float("nan")
        zero_entry_relief_return = float("nan")
        zero_entry_jump_gate_pass = True
        zero_entry_relief_gate_pass = True
        zero_entry_config = config.get("zero_entry_bridge")
        if isinstance(zero_entry_config, dict) and bool(
            zero_entry_config.get("enabled", False)
        ):
            (
                zero_entry_state,
                _,
                _,
            ) = turning_point_cycle(
                growth_returns,
                trailing_returns[assets[cash_index]],
                int(zero_entry_config.get("fast_days", 20)),
                int(zero_entry_config.get("slow_days", 252)),
            )
            zero_entry_shares = {
                str(name): float(value)
                for name, value in zero_entry_config.get(
                    "state_account_shares",
                    {},
                ).items()
            }
            if any(
                not 0.0 <= value <= 1.0
                for value in zero_entry_shares.values()
            ):
                raise ValueError(
                    "Turning-point zero-entry shares must be between zero "
                    "and one"
                )
            zero_entry_requested_share = float(
                zero_entry_shares.get(zero_entry_state, 0.0)
            )
            zero_entry_volatility_config = zero_entry_config.get(
                "volatility_conditioning"
            )
            if isinstance(zero_entry_volatility_config, dict) and bool(
                zero_entry_volatility_config.get("enabled", False)
            ):
                zero_entry_volatility_state, _ = (
                    causal_realized_volatility_state(
                        growth_returns,
                        int(
                            zero_entry_volatility_config.get(
                                "window_days",
                                20,
                            )
                        ),
                        int(
                            zero_entry_volatility_config.get(
                                "threshold_lookback_days",
                                756,
                            )
                        ),
                        int(
                            zero_entry_volatility_config.get(
                                "minimum_threshold_observations",
                                504,
                            )
                        ),
                        float(
                            zero_entry_volatility_config.get(
                                "low_quantile",
                                0.45,
                            )
                        ),
                        float(
                            zero_entry_volatility_config.get(
                                "high_quantile",
                                0.90,
                            )
                        ),
                        estimator=str(
                            zero_entry_volatility_config.get(
                                "estimator",
                                "standard",
                            )
                        ),
                    )
                )
                zero_entry_volatility_shares = (
                    zero_entry_volatility_config.get(
                        "state_account_shares",
                        {},
                    )
                )
                if isinstance(zero_entry_volatility_shares, dict):
                    zero_entry_state_shares = (
                        zero_entry_volatility_shares.get(zero_entry_state)
                    )
                    if isinstance(zero_entry_state_shares, dict):
                        zero_entry_requested_share = float(
                            zero_entry_state_shares.get(
                                zero_entry_volatility_state,
                                zero_entry_requested_share,
                            )
                        )
            constructive_override = zero_entry_config.get(
                "constructive_high_volatility_override"
            )
            if isinstance(constructive_override, dict) and bool(
                constructive_override.get("enabled", False)
            ):
                override_window_days = int(
                    constructive_override.get("window_days", 20)
                )
                zero_entry_downside_variation_share = (
                    realized_downside_variation_share(
                        growth_returns,
                        override_window_days,
                    )
                )
                maximum_downside_share = float(
                    constructive_override.get(
                        "maximum_downside_variation_share",
                        0.50,
                    )
                )
                minimum_slow_return = float(
                    constructive_override.get(
                        "minimum_slow_excess_return",
                        float("-inf"),
                    )
                )
                minimum_jump_share = constructive_override.get(
                    "minimum_jump_variation_share"
                )
                if minimum_jump_share is not None:
                    zero_entry_jump_variation_share = (
                        realized_jump_variation_share(
                            growth_returns,
                            int(
                                constructive_override.get(
                                    "jump_window_days",
                                    override_window_days,
                                )
                            ),
                        )
                    )
                    zero_entry_jump_gate_pass = bool(
                        np.isfinite(zero_entry_jump_variation_share)
                        and zero_entry_jump_variation_share
                        > float(minimum_jump_share)
                    )
                relief_asset_value = constructive_override.get(
                    "relief_asset"
                )
                relief_signal_value = constructive_override.get(
                    "relief_signal"
                )
                if (
                    relief_asset_value is not None
                    and relief_signal_value is not None
                ):
                    raise ValueError(
                        "Constructive high-volatility relief must use either "
                        "an asset return or a signal level, not both"
                    )
                relief_window_days = int(
                    constructive_override.get(
                        "relief_window_days",
                        5,
                    )
                )
                if relief_asset_value is not None:
                    relief_asset = str(relief_asset_value)
                    if relief_asset not in assets:
                        raise ValueError(
                            "Constructive high-volatility relief asset is "
                            f"missing: {relief_asset}"
                        )
                    zero_entry_relief_return = trailing_compounded_return(
                        trailing_returns[relief_asset],
                        relief_window_days,
                    )
                elif relief_signal_value is not None:
                    relief_signal = str(relief_signal_value)
                    if signal_prices is None or signal_date is None:
                        raise ValueError(
                            "Constructive high-volatility signal relief "
                            "requires causal signal prices and a decision date"
                        )
                    if relief_signal not in signal_prices:
                        raise ValueError(
                            "Constructive high-volatility relief signal is "
                            f"missing: {relief_signal}"
                        )
                    signal_levels = signal_prices.loc[
                        signal_prices.index < signal_date,
                        relief_signal,
                    ].dropna()
                    signal_returns = signal_levels.pct_change(
                        fill_method=None
                    ).dropna()
                    zero_entry_relief_return = trailing_compounded_return(
                        signal_returns,
                        relief_window_days,
                    )
                if (
                    relief_asset_value is not None
                    or relief_signal_value is not None
                ):
                    zero_entry_relief_gate_pass = bool(
                        np.isfinite(zero_entry_relief_return)
                        and zero_entry_relief_return
                        < float(
                            constructive_override.get(
                                "maximum_relief_return",
                                0.0,
                            )
                        )
                    )
                vix_contango_pass = (
                    not bool(
                        constructive_override.get(
                            "require_vix_contango",
                            False,
                        )
                    )
                    or self.previous_vix_backwardation is False
                )
                override_state_shares = {
                    str(name): float(value)
                    for name, value in constructive_override.get(
                        "state_account_shares",
                        {},
                    ).items()
                }
                if not 0.0 <= maximum_downside_share <= 1.0:
                    raise ValueError(
                        "Constructive high-volatility downside share must "
                        "be between zero and one"
                    )
                if any(
                    not 0.0 <= value <= 1.0
                    for value in override_state_shares.values()
                ):
                    raise ValueError(
                        "Constructive high-volatility account shares must "
                        "be between zero and one"
                    )
                if (
                    zero_entry_volatility_state == "high"
                    and np.isfinite(
                        zero_entry_downside_variation_share
                    )
                    and zero_entry_downside_variation_share
                    <= maximum_downside_share
                    and slow_return >= minimum_slow_return
                    and zero_entry_jump_gate_pass
                    and zero_entry_relief_gate_pass
                    and vix_contango_pass
                ):
                    zero_entry_requested_share = float(
                        override_state_shares.get(
                            zero_entry_state,
                            zero_entry_requested_share,
                        )
                    )
            maximum_current_growth = float(
                zero_entry_config.get(
                    "maximum_current_growth_exposure",
                    minimum_current_growth,
                )
            )
            if not 0.0 <= maximum_current_growth <= 1.0:
                raise ValueError(
                    "Turning-point zero-entry maximum current growth must be "
                    "between zero and one"
                )
            zero_entry_risk_off_only = bool(
                zero_entry_config.get(
                    "activate_only_when_risk_off",
                    True,
                )
            )
            disallowed_parent_states = {
                str(value)
                for value in zero_entry_config.get(
                    "disallowed_parent_states",
                    [],
                )
            }
            unknown_parent_states = disallowed_parent_states.difference(
                state_codes
            )
            if unknown_parent_states:
                raise ValueError(
                    "Turning-point zero-entry disallowed parent states "
                    f"contain unknown values: {sorted(unknown_parent_states)}"
                )
            zero_entry_market_signal_active = (
                zero_entry_requested_share > 0.0
                and state not in disallowed_parent_states
                and (
                    not zero_entry_risk_off_only
                    or self.paper_risk_on_state is False
                )
            )
            path_config = zero_entry_config.get("path_conditioning")
            if isinstance(path_config, dict) and bool(
                path_config.get("enabled", False)
            ):
                if signal_date is None:
                    raise ValueError(
                        "Turning-point zero-entry path conditioning requires "
                        "a decision date"
                    )
                minimum_confirmation_days = int(
                    path_config.get("minimum_confirmation_days", 1)
                )
                staging_days = int(path_config.get("staging_days", 0))
                initial_account_share = float(
                    path_config.get(
                        "initial_account_share",
                        zero_entry_requested_share,
                    )
                )
                if minimum_confirmation_days <= 0:
                    raise ValueError(
                        "Turning-point zero-entry confirmation days must be "
                        "positive"
                    )
                if staging_days < 0:
                    raise ValueError(
                        "Turning-point zero-entry staging days cannot be "
                        "negative"
                    )
                if not 0.0 <= initial_account_share <= 1.0:
                    raise ValueError(
                        "Turning-point zero-entry initial account share must "
                        "be between zero and one"
                    )
                current_signal_date = pd.Timestamp(signal_date)
                last_signal_date = getattr(
                    self,
                    "turning_point_zero_entry_last_signal_date",
                    None,
                )
                if (
                    last_signal_date is None
                    or current_signal_date != last_signal_date
                ):
                    prior_streak = int(
                        getattr(
                            self,
                            "turning_point_zero_entry_signal_streak",
                            0,
                        )
                    )
                    self.turning_point_zero_entry_signal_streak = (
                        prior_streak + 1
                        if zero_entry_market_signal_active
                        else 0
                    )
                    self.turning_point_zero_entry_last_signal_date = (
                        current_signal_date
                    )
                zero_entry_signal_streak = int(
                    getattr(
                        self,
                        "turning_point_zero_entry_signal_streak",
                        0,
                    )
                )
                zero_entry_path_gate_pass = bool(
                    zero_entry_signal_streak >= minimum_confirmation_days
                )
                if not zero_entry_path_gate_pass:
                    zero_entry_requested_share = 0.0
                elif (
                    staging_days > 0
                    and zero_entry_signal_streak <= staging_days
                ):
                    zero_entry_requested_share = min(
                        zero_entry_requested_share,
                        initial_account_share,
                    )
            zero_entry_signal_active = (
                zero_entry_market_signal_active
                and zero_entry_path_gate_pass
                and zero_entry_requested_share > 0.0
                and current_growth < maximum_current_growth - 1e-12
            )
            if zero_entry_signal_active:
                allocation_method = str(
                    zero_entry_config.get("allocation_method", "static")
                )
                if allocation_method == "static":
                    zero_entry_core_weights = {
                        str(asset): float(value)
                        for asset, value in zero_entry_config[
                            "core_weights"
                        ].items()  # type: ignore[union-attr]
                    }
                elif allocation_method == "relative_momentum":
                    relative_config = portfolio_config.get(
                        "relative_momentum_core"
                    )
                    if not isinstance(relative_config, dict) or not bool(
                        relative_config.get("enabled", False)
                    ):
                        raise ValueError(
                            "Relative-momentum zero-entry allocation requires "
                            "an enabled relative-momentum core"
                        )
                    anchor_asset = str(relative_config["anchor_asset"])
                    tilt_asset = str(relative_config["tilt_asset"])
                    if (
                        anchor_asset not in assets
                        or tilt_asset not in assets
                        or anchor_asset == tilt_asset
                    ):
                        raise ValueError(
                            "Relative-momentum zero-entry assets must be "
                            "distinct known assets"
                        )
                    zero_entry_tilt_weight, _ = (
                        relative_momentum_pair_weight(
                            trailing_returns[anchor_asset],
                            trailing_returns[tilt_asset],
                            [
                                int(value)
                                for value in relative_config["horizons"]  # type: ignore[union-attr]
                            ],
                            [
                                float(value)
                                for value in relative_config[
                                    "horizon_weights"
                                ]  # type: ignore[union-attr]
                            ],
                            int(relative_config["skip_days"]),
                            float(relative_config["base_tilt_weight"]),
                            float(relative_config["max_tilt"]),
                        )
                    )
                    zero_entry_core_weights = {
                        anchor_asset: 1.0 - zero_entry_tilt_weight,
                        tilt_asset: zero_entry_tilt_weight,
                    }
                else:
                    raise ValueError(
                        "Unknown turning-point zero-entry allocation method: "
                        f"{allocation_method}"
                    )
                if (
                    any(
                        asset not in assets
                        for asset in zero_entry_core_weights
                    )
                    or any(
                        value < 0.0
                        for value in zero_entry_core_weights.values()
                    )
                    or not np.isclose(
                        sum(zero_entry_core_weights.values()),
                        1.0,
                    )
                ):
                    raise ValueError(
                        "Turning-point zero-entry core weights must contain "
                        "known assets and sum to one"
                    )
                zero_entry_core = np.zeros(len(assets), dtype=float)
                for asset, value in zero_entry_core_weights.items():
                    zero_entry_core[assets.index(asset)] = value
                target, zero_entry_applied_share = (
                    ensure_minimum_growth_exposure(
                        target,
                        zero_entry_core,
                        cash_index,
                        zero_entry_requested_share,
                    )
                )
                zero_entry_active = bool(
                    zero_entry_applied_share > current_growth + 1e-12
                )
        any_reentry_active = floor_binding or zero_entry_active
        return target, {
            "turning_point_state": state_codes[state],
            "turning_point_fast_excess_return": fast_return,
            "turning_point_slow_excess_return": slow_return,
            "turning_point_requested_share": (
                requested_share if signal_active else 0.0
            ),
            "turning_point_base_floor_gate_pass": int(
                base_floor_gate_pass
            ),
            "turning_point_reentry_active": int(any_reentry_active),
            "turning_point_volatility_state": {
                "insufficient": -1,
                "low": 0,
                "medium": 1,
                "high": 2,
            }[volatility_state],
            "turning_point_realized_volatility": realized_volatility,
            "turning_point_volatility_override_gate_pass": int(
                volatility_override_gate_pass
                if isinstance(volatility_config, dict)
                and bool(volatility_config.get("enabled", False))
                else True
            ),
            "turning_point_zero_entry_state": state_codes[
                zero_entry_state
            ],
            "turning_point_zero_entry_requested_share": (
                zero_entry_requested_share if zero_entry_active else 0.0
            ),
            "turning_point_zero_entry_active": int(zero_entry_active),
            "turning_point_zero_entry_market_signal_active": int(
                zero_entry_market_signal_active
            ),
            "turning_point_zero_entry_signal_streak": (
                zero_entry_signal_streak
            ),
            "turning_point_zero_entry_path_gate_pass": int(
                zero_entry_path_gate_pass
            ),
            "turning_point_zero_entry_tilt_weight": zero_entry_tilt_weight,
            "turning_point_zero_entry_volatility_state": {
                "insufficient": -1,
                "low": 0,
                "medium": 1,
                "high": 2,
            }[zero_entry_volatility_state],
            "turning_point_zero_entry_downside_variation_share": (
                zero_entry_downside_variation_share
            ),
            "turning_point_zero_entry_jump_variation_share": (
                zero_entry_jump_variation_share
            ),
            "turning_point_zero_entry_relief_return": (
                zero_entry_relief_return
            ),
            "turning_point_zero_entry_jump_gate_pass": int(
                zero_entry_jump_gate_pass
            ),
            "turning_point_zero_entry_relief_gate_pass": int(
                zero_entry_relief_gate_pass
            ),
            "_turning_point_asset_indices": growth_indices,
        }

    def _survival_target(
        self,
        date: pd.Timestamp,
        all_features: pd.DataFrame,
        all_returns: pd.DataFrame,
        weights: np.ndarray,
        assets: list[str],
        cash_index: int,
        annualization: int,
        portfolio_config: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        survival_config = portfolio_config.get("survival_governor")
        if not isinstance(survival_config, dict) or not bool(
            survival_config.get("enabled", False)
        ):
            return weights.copy(), {
                "survival_enabled": 0,
                "survival_active": 0,
                "survival_pre_volatility": float("nan"),
                "survival_post_volatility": float("nan"),
                "survival_long_horizon_volatility": float("nan"),
                "survival_stress_volatility_ratio": float("nan"),
                "survival_target_volatility": float("nan"),
                "survival_risk_multiplier": 1.0,
                "survival_novelty_percentile": float("nan"),
                "survival_novelty_multiplier": 1.0,
                "survival_gross_before": float(
                    np.abs(np.delete(weights, cash_index)).sum()
                ),
                "survival_gross_after": float(
                    np.abs(np.delete(weights, cash_index)).sum()
                ),
                "survival_max_risk_share_before": float("nan"),
                "survival_max_risk_share_after": float("nan"),
                "survival_triggered": 0,
                "survival_recovery_mode": 0,
            }

        novelty_percentile = 0.0
        novelty_config = survival_config.get("novelty")
        if isinstance(novelty_config, dict) and bool(
            novelty_config.get("enabled", False)
        ):
            feature_history = all_features.loc[all_features.index < date]
            if date not in all_features.index:
                raise ValueError("Survival governor is missing the current feature")
            novelty_percentile = feature_novelty_percentile(
                feature_history,
                all_features.loc[date],
                int(novelty_config.get("lookback_days", 1008)),
                float(novelty_config.get("shrinkage", 0.20)),
            )

        trailing_returns = all_returns.loc[all_returns.index < date].dropna(
            how="any"
        )
        risk_assets = [
            str(asset)
            for asset in survival_config["risk_assets"]  # type: ignore[index]
        ]
        result = apply_survival_governor(
            weights,
            trailing_returns,
            assets,
            risk_assets,
            cash_index,
            [
                int(days)
                for days in survival_config.get(
                    "horizons", [20, 60, 126]
                )  # type: ignore[union-attr]
            ],
            float(survival_config["target_volatility"]),
            {
                str(asset): float(value)
                for asset, value in survival_config.get(
                    "maximum_risk_shares", {}
                ).items()  # type: ignore[union-attr]
            },
            novelty_percentile,
            float(
                novelty_config.get("activation_percentile", 0.95)
                if isinstance(novelty_config, dict)
                else 0.95
            ),
            float(
                novelty_config.get("minimum_multiplier", 0.75)
                if isinstance(novelty_config, dict)
                else 0.75
            ),
            float(survival_config.get("maximum_gross_when_novel", 1.0)),
            annualization,
            (
                float(survival_config["activation_stress_ratio"])
                if "activation_stress_ratio" in survival_config
                else None
            ),
            int(survival_config.get("long_horizon_days", 252)),
        )
        return result.weights, {
            "survival_enabled": 1,
            "survival_active": int(result.active),
            "survival_pre_volatility": result.pre_volatility,
            "survival_post_volatility": result.post_volatility,
            "survival_long_horizon_volatility": (
                result.long_horizon_volatility
            ),
            "survival_stress_volatility_ratio": (
                result.stress_volatility_ratio
            ),
            "survival_target_volatility": result.target_volatility,
            "survival_risk_multiplier": result.risk_multiplier,
            "survival_novelty_percentile": result.novelty_percentile,
            "survival_novelty_multiplier": result.novelty_multiplier,
            "survival_gross_before": result.gross_before,
            "survival_gross_after": result.gross_after,
            "survival_max_risk_share_before": result.max_risk_share_before,
            "survival_max_risk_share_after": result.max_risk_share_after,
            "survival_triggered": int(result.triggered),
        }

    @staticmethod
    def _survival_recovery_enabled(
        portfolio_config: dict[str, object],
    ) -> bool:
        survival_config = portfolio_config.get("survival_governor")
        return bool(
            isinstance(survival_config, dict)
            and survival_config.get("enabled", False)
            and survival_config.get("restore_to_alpha_target", False)
        )

    def _conditional_vix_hedge_target(
        self,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        weights: np.ndarray,
        assets: list[str],
        cash_index: int,
        portfolio_config: dict[str, object],
    ) -> tuple[np.ndarray, bool, float]:
        hedge_config = portfolio_config.get("conditional_vix_hedge")
        if not isinstance(hedge_config, dict) or not bool(
            hedge_config.get("enabled", False)
        ):
            return weights.copy(), False, float("nan")
        hedge_asset = str(hedge_config["asset"])
        spot_signal = str(hedge_config["spot_signal"])
        three_month_signal = str(hedge_config["three_month_signal"])
        unknown = [
            name
            for name in (hedge_asset, spot_signal, three_month_signal)
            if name not in (assets if name == hedge_asset else prices.columns)
        ]
        if unknown:
            raise ValueError(f"Conditional VIX hedge is missing inputs: {unknown}")
        implied_history = prices.loc[
            prices.index < date,
            [spot_signal, three_month_signal],
        ].dropna(how="any")
        if implied_history.empty:
            raise ValueError("No causal VIX observation is available for the hedge")
        latest = implied_history.iloc[-1]
        term_ratio, backwardation = implied_volatility_term_structure(
            float(latest[spot_signal]),
            float(latest[three_month_signal]),
        )
        target = apply_conditional_cash_funded_hedge(
            weights,
            assets.index(hedge_asset),
            cash_index,
            float(hedge_config["account_share"]),
            backwardation,
        )
        return target, backwardation, term_ratio

    def _recommend_with_conditional_vix_hedge(
        self,
        target: np.ndarray,
        date: pd.Timestamp,
        prices: pd.DataFrame,
        assets: list[str],
        cash_index: int,
        portfolio_config: dict[str, object],
        all_features: pd.DataFrame,
        all_returns: pd.DataFrame,
        annualization: int,
        diagnostics: dict[str, float | int | str],
    ) -> tuple[pd.Series, dict[str, float | int | str]]:
        standalone_target, standalone_diagnostics = (
            self._standalone_risk_cap_target(
                date,
                all_returns,
                target,
                assets,
                cash_index,
                annualization,
                portfolio_config,
            )
        )
        standalone_turnover = 0.5 * float(
            np.abs(standalone_target - target).sum()
        )
        standalone_executed = (
            standalone_turnover
            >= float(portfolio_config["no_trade_turnover"])
        )
        if standalone_executed:
            target = standalone_target
            if diagnostics.get("action") == "HOLD":
                diagnostics["action"] = "RISK_REBALANCE"
        diagnostics.update(standalone_diagnostics)
        diagnostics["standalone_risk_cap_executed"] = int(standalone_executed)

        recovery_enabled = self._survival_recovery_enabled(portfolio_config)
        if recovery_enabled and diagnostics.get("action") != "HOLD":
            self.survival_overlay_state.begin_alpha_target(target)
        survival_input = target
        survival_recovery_mode = False
        if recovery_enabled:
            (
                survival_input,
                survival_recovery_mode,
            ) = self.survival_overlay_state.target_input(target)
        survival_target, survival_diagnostics = self._survival_target(
            date,
            all_features,
            all_returns,
            survival_input,
            assets,
            cash_index,
            annualization,
            portfolio_config,
        )
        if recovery_enabled:
            self.survival_overlay_state.observe_result(
                survival_input,
                bool(survival_diagnostics["survival_active"]),
            )
        survival_diagnostics["survival_recovery_mode"] = int(
            survival_recovery_mode
        )
        survival_turnover = 0.5 * float(
            np.abs(survival_target - target).sum()
        )
        survival_executed = (
            survival_turnover
            >= float(portfolio_config["no_trade_turnover"])
        )
        action_requires_trade = diagnostics.get("action") != "HOLD"
        if diagnostics.get("action") == "HOLD" and survival_executed:
            diagnostics["action"] = "SURVIVAL_REDUCE"
        diagnostics.update(survival_diagnostics)
        diagnostics["survival_executed"] = int(
            survival_executed or action_requires_trade
        )
        applied_target = (
            survival_target
            if survival_executed or action_requires_trade
            else target
        )

        hedge_target, active, term_ratio = self._conditional_vix_hedge_target(
            date,
            prices,
            applied_target,
            assets,
            cash_index,
            portfolio_config,
        )
        hedge_config = portfolio_config.get("conditional_vix_hedge")
        hedge_share = 0.0
        if isinstance(hedge_config, dict) and bool(
            hedge_config.get("enabled", False)
        ):
            hedge_share = float(
                hedge_target[assets.index(str(hedge_config["asset"]))]
            )
        hedge_turnover = 0.5 * float(
            np.abs(hedge_target - applied_target).sum()
        )
        if (
            diagnostics.get("action") == "HOLD"
            and hedge_turnover
            >= float(portfolio_config["no_trade_turnover"])
        ):
            diagnostics["action"] = "HEDGE_ON" if active else "HEDGE_OFF"
        diagnostics["conditional_vix_hedge_active"] = int(active)
        diagnostics["conditional_vix_hedge_term_ratio"] = term_ratio
        diagnostics["conditional_vix_hedge_share"] = hedge_share
        return pd.Series(hedge_target, index=assets, name=date), diagnostics

    @staticmethod
    def _standalone_risk_cap_target(
        date: pd.Timestamp,
        all_returns: pd.DataFrame,
        weights: np.ndarray,
        assets: list[str],
        cash_index: int,
        annualization: int,
        portfolio_config: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Apply causal pair and strategic-asset standalone-risk constraints."""
        adjusted = weights.copy()
        diagnostics: dict[str, float | int] = {
            "pair_risk_share_cap_enabled": 0,
            "pair_risk_share_before": float("nan"),
            "pair_risk_share_after": float("nan"),
            "pair_risk_share_reallocated": 0.0,
            "pair_risk_share_maximum_controlled_weight": float("nan"),
            "pair_risk_share_anchor_volatility": float("nan"),
            "pair_risk_share_controlled_volatility": float("nan"),
            "idiosyncratic_survival_enabled": 0,
            "idiosyncratic_survival_score": float("nan"),
            "idiosyncratic_survival_beta": float("nan"),
            "idiosyncratic_survival_multiplier": 1.0,
            "idiosyncratic_survival_reallocated": 0.0,
            "strategic_asset_risk_caps_enabled": 0,
        }
        relative_config = portfolio_config.get("relative_momentum_core")
        pair_cap_config = (
            relative_config.get("standalone_risk_share_cap")
            if isinstance(relative_config, dict)
            else None
        )
        if isinstance(pair_cap_config, dict) and bool(
            pair_cap_config.get("enabled", False)
        ):
            anchor_asset = str(relative_config["anchor_asset"])
            controlled_asset = str(relative_config["tilt_asset"])
            trailing_pair_returns = all_returns.loc[
                all_returns.index < date,
                [anchor_asset, controlled_asset],
            ].dropna(how="any")
            _, effective_volatility = realized_stress_covariance(
                trailing_pair_returns,
                int(pair_cap_config["fast_days"]),
                int(pair_cap_config["slow_days"]),
                annualization,
                str(pair_cap_config.get("volatility_estimator", "jump_aware")),
                "stress_max",
            )
            (
                adjusted,
                reallocated,
                before_share,
                after_share,
                maximum_controlled_weight,
            ) = cap_pair_standalone_risk_share(
                adjusted,
                assets,
                anchor_asset,
                controlled_asset,
                float(effective_volatility[0]),
                float(effective_volatility[1]),
                float(pair_cap_config["maximum_tilt_risk_share"]),
            )
            diagnostics.update(
                {
                    "pair_risk_share_cap_enabled": 1,
                    "pair_risk_share_before": before_share,
                    "pair_risk_share_after": after_share,
                    "pair_risk_share_reallocated": reallocated,
                    "pair_risk_share_maximum_controlled_weight": (
                        maximum_controlled_weight
                    ),
                    "pair_risk_share_anchor_volatility": float(
                        effective_volatility[0]
                    ),
                    "pair_risk_share_controlled_volatility": float(
                        effective_volatility[1]
                    ),
                }
            )

        idiosyncratic_config = (
            relative_config.get("idiosyncratic_survival")
            if isinstance(relative_config, dict)
            else None
        )
        if isinstance(idiosyncratic_config, dict) and bool(
            idiosyncratic_config.get("enabled", False)
        ):
            anchor_asset = str(relative_config["anchor_asset"])
            controlled_asset = str(relative_config["tilt_asset"])
            trailing_pair_returns = all_returns.loc[
                all_returns.index < date,
                [anchor_asset, controlled_asset],
            ].dropna(how="any")
            (
                survival_multiplier,
                downside_score,
                beta,
            ) = idiosyncratic_downside_survival_multiplier(
                trailing_pair_returns[anchor_asset],
                trailing_pair_returns[controlled_asset],
                int(idiosyncratic_config["beta_days"]),
                [
                    int(horizon)
                    for horizon in idiosyncratic_config["shock_horizons"]  # type: ignore[union-attr]
                ],
                float(idiosyncratic_config["soft_z_score"]),
                float(idiosyncratic_config["hard_z_score"]),
                float(idiosyncratic_config.get("minimum_multiplier", 0.0)),
            )
            adjusted, reallocated = apply_pair_survival_multiplier(
                adjusted,
                assets,
                anchor_asset,
                controlled_asset,
                survival_multiplier,
            )
            diagnostics.update(
                {
                    "idiosyncratic_survival_enabled": 1,
                    "idiosyncratic_survival_score": downside_score,
                    "idiosyncratic_survival_beta": beta,
                    "idiosyncratic_survival_multiplier": survival_multiplier,
                    "idiosyncratic_survival_reallocated": reallocated,
                }
            )

        strategic_config = portfolio_config.get("strategic_asset_risk_caps")
        if isinstance(strategic_config, dict) and bool(
            strategic_config.get("enabled", False)
        ):
            diagnostics["strategic_asset_risk_caps_enabled"] = 1
            configured_assets = strategic_config.get("assets", {})
            if not isinstance(configured_assets, dict):
                raise ValueError("Strategic asset risk caps must define an asset map")
            unknown = sorted(set(configured_assets).difference(assets))
            if unknown:
                raise ValueError(f"Unknown strategic risk-cap assets: {unknown}")
            for asset, asset_config in configured_assets.items():
                if not isinstance(asset_config, dict):
                    raise ValueError(
                        f"Strategic risk cap for {asset} must be a mapping"
                    )
                trailing_asset_returns = all_returns.loc[
                    all_returns.index < date,
                    [str(asset)],
                ].dropna(how="any")
                _, effective_volatility = realized_stress_covariance(
                    trailing_asset_returns,
                    int(strategic_config["fast_days"]),
                    int(strategic_config["slow_days"]),
                    annualization,
                    str(
                        strategic_config.get(
                            "volatility_estimator",
                            "jump_aware",
                        )
                    ),
                    "slow",
                )
                (
                    adjusted,
                    released,
                    before_contribution,
                    after_contribution,
                ) = cap_asset_standalone_risk_contribution(
                    adjusted,
                    assets.index(str(asset)),
                    cash_index,
                    float(effective_volatility[0]),
                    float(asset_config["maximum_standalone_contribution"]),
                )
                diagnostic_prefix = f"strategic_risk_{asset}"
                diagnostics[f"{diagnostic_prefix}_effective_volatility"] = float(
                    effective_volatility[0]
                )
                diagnostics[f"{diagnostic_prefix}_before_contribution"] = (
                    before_contribution
                )
                diagnostics[f"{diagnostic_prefix}_after_contribution"] = (
                    after_contribution
                )
                diagnostics[f"{diagnostic_prefix}_released"] = released
        return adjusted, diagnostics

    def _target_weights(
        self,
        date: pd.Timestamp,
        all_features: pd.DataFrame,
        all_returns: pd.DataFrame,
        all_prices: pd.DataFrame,
        previous_weights: np.ndarray,
        drawdown: float,
        assets: list[str],
        cash_index: int,
        annualization: int,
        model_config: dict[str, object],
        portfolio_config: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        excluded_assets = {
            str(asset)
            for asset in portfolio_config.get(
                "optimizer_excluded_assets", []
            )  # type: ignore[union-attr]
        }
        if not excluded_assets:
            return self._core_target_weights(
                date,
                all_features,
                all_returns,
                all_prices,
                previous_weights,
                drawdown,
                assets,
                cash_index,
                annualization,
                model_config,
                portfolio_config,
            )
        unknown = sorted(excluded_assets.difference(assets))
        if unknown:
            raise ValueError(f"Unknown optimizer-excluded assets: {unknown}")
        cash_asset = assets[cash_index]
        if cash_asset in excluded_assets:
            raise ValueError("Cash cannot be excluded from the optimizer")
        core_indices = [
            index for index, asset in enumerate(assets) if asset not in excluded_assets
        ]
        core_assets = [assets[index] for index in core_indices]
        core_cash_index = core_assets.index(cash_asset)
        core_target, diagnostics = self._core_target_weights(
            date,
            all_features,
            all_returns[core_assets],
            all_prices,
            previous_weights[core_indices],
            drawdown,
            core_assets,
            core_cash_index,
            annualization,
            model_config,
            portfolio_config,
        )
        target = np.zeros(len(assets), dtype=float)
        target[core_indices] = core_target
        return target, diagnostics

    def _core_target_weights(
        self,
        date: pd.Timestamp,
        all_features: pd.DataFrame,
        all_returns: pd.DataFrame,
        all_prices: pd.DataFrame,
        previous_weights: np.ndarray,
        drawdown: float,
        assets: list[str],
        cash_index: int,
        annualization: int,
        model_config: dict[str, object],
        portfolio_config: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        previous_paper_risk_on_state = self.paper_risk_on_state
        signal_position = all_features.index.get_loc(date)
        anchored_signal_position = anchored_business_day_position(date)
        history = all_features.loc[all_features.index < date]
        lookback = int(model_config["lookback_days"])
        training = history.iloc[-lookback:]
        if len(training) < int(model_config["validation_days"]) + 100:
            raise ValueError(f"Insufficient causal training history at {date.date()}")

        ensemble_orders = tuple(
            sorted({int(order) for order in model_config.get("ensemble_orders", [])})
        )
        ensemble_enabled = bool(ensemble_orders)
        filter_diagnostic_rows: list[dict[str, float]] = []
        template_distance_rows: list[dict[str, float]] = []
        if ensemble_enabled:
            if min(ensemble_orders) < 2:
                raise ValueError("HMM ensemble orders must be at least 2")
            if bool(portfolio_config.get("paper_probability_allocation", False)):
                raise ValueError(
                    "HMM model averaging is not supported with probability allocation"
                )
            self.selected_order = max(ensemble_orders)
            should_select = False
        else:
            selection_days = int(model_config["order_selection_days"])
            anchored_schedule = bool(
                model_config.get("calendar_anchored_schedule", False)
            )
            should_select = self.selected_order is None or (
                anchored_signal_position % selection_days == 0
                if anchored_schedule
                else signal_position - self.last_selection_position >= selection_days
            )
            if should_select:
                self.selected_order, _ = self.estimator.select_order(training)
                self.last_selection_position = signal_position

        refit_days = int(model_config["model_refit_days"])
        refit_stress = False
        refit_state_transition = False
        adaptive_refit_config = model_config.get("adaptive_refit")
        if isinstance(adaptive_refit_config, dict) and bool(
            adaptive_refit_config.get("enabled", False)
        ):
            refit_stress = adaptive_hmm_refit_stress(
                all_prices,
                date,
                [str(asset) for asset in adaptive_refit_config["assets"]],
                int(adaptive_refit_config["fast_days"]),
                int(adaptive_refit_config["slow_days"]),
            )
            refit_days, refit_state_transition = adaptive_hmm_refit_policy(
                refit_stress,
                self.last_hmm_refit_stress,
                adaptive_refit_config,
            )
        anchored_schedule = bool(model_config.get("calendar_anchored_schedule", False))
        should_refit = (
            (not self.ensemble_fits if ensemble_enabled else self.fit is None)
            or (
                anchored_signal_position % refit_days == 0
                if anchored_schedule
                else signal_position - self.last_fit_position >= refit_days
            )
            or refit_state_transition
        )
        if isinstance(adaptive_refit_config, dict) and bool(
            adaptive_refit_config.get("enabled", False)
        ):
            self.last_hmm_refit_stress = refit_stress
        if should_refit:
            assert self.selected_order is not None
            if ensemble_enabled:
                self.ensemble_fits = {
                    order: self.estimator.fit(training, all_returns, order)
                    for order in ensemble_orders
                }
                calibration_order = min(
                    int(model_config["template_count"]),
                    len(training) // 100,
                )
                calibration = self.ensemble_fits.get(calibration_order)
                if calibration is None:
                    calibration = self.estimator.fit(
                        training, all_returns, calibration_order
                    )
                for order in ensemble_orders:
                    if order not in self.ensemble_trackers:
                        tracker = WassersteinTemplateTracker(
                            int(model_config["template_count"]),
                            float(model_config["template_smoothing"]),
                            str(model_config.get("template_assignment", "nearest")),
                            str(
                                model_config.get(
                                    "template_update_geometry", "euclidean_variance"
                                )
                            ),
                        )
                        tracker.initialize(calibration)
                        self.ensemble_trackers[order] = tracker
                self.fit = self.ensemble_fits[self.selected_order]
                self.tracker = self.ensemble_trackers[self.selected_order]
            else:
                self.fit = self.estimator.fit(training, all_returns, self.selected_order)
                if not self.tracker.templates:
                    calibration_order = min(
                        int(model_config["template_count"]),
                        len(training) // 100,
                    )
                    calibration = self.estimator.fit(
                        training, all_returns, calibration_order
                    )
                    self.tracker.initialize(calibration)
            self.last_fit_position = signal_position

        template_update = should_update_hmm_templates(should_refit, model_config)
        if ensemble_enabled:
            template_probability_forecasts: list[np.ndarray] = []
            conditional_means: list[np.ndarray] = []
            conditional_covariances: list[np.ndarray] = []
            for order in ensemble_orders:
                fit = self.ensemble_fits[order]
                tracker = self.ensemble_trackers[order]
                filter_start = fit.training_features.index[0]
                filter_data = all_features.loc[
                    (all_features.index >= filter_start)
                    & (all_features.index <= date)
                ]
                state_probabilities = fit.filtered_probabilities(filter_data)
                filter_diagnostic_rows.append(
                    fit.filter_diagnostics(filter_data, state_probabilities)
                )
                probabilities, _ = tracker.map_and_update(
                    fit, state_probabilities, update=template_update
                )
                template_distance_rows.append(
                    tracker.distance_diagnostics(state_probabilities)
                )
                mean, covariance = tracker.conditional_moments(probabilities)
                template_probability_forecasts.append(probabilities)
                conditional_means.append(mean)
                conditional_covariances.append(covariance)
            template_probabilities = np.mean(
                template_probability_forecasts, axis=0
            )
            conditional_mean, conditional_covariance = (
                equal_weight_predictive_moments(
                    conditional_means, conditional_covariances
                )
            )
        else:
            assert self.fit is not None
            filter_start = self.fit.training_features.index[0]
            filter_data = all_features.loc[
                (all_features.index >= filter_start)
                & (all_features.index <= date)
            ]
            state_probabilities = self.fit.filtered_probabilities(filter_data)
            filter_diagnostic_rows.append(
                self.fit.filter_diagnostics(filter_data, state_probabilities)
            )
            template_probabilities, _ = self.tracker.map_and_update(
                self.fit, state_probabilities, update=template_update
            )
            template_distance_rows.append(
                self.tracker.distance_diagnostics(state_probabilities)
            )
            conditional_mean, conditional_covariance = self.tracker.conditional_moments(
                template_probabilities
            )
            moments_source = str(
                model_config.get("conditional_moments_source", "templates")
            )
            if moments_source == "fit":
                conditional_mean, conditional_covariance = (
                    probability_weighted_predictive_moments(
                        self.fit.state_return_means,
                        self.fit.state_return_covariances,
                        state_probabilities,
                    )
                )
            elif moments_source != "templates":
                raise ValueError(
                    f"Unsupported conditional moments source: {moments_source}"
                )

        hmm_filter_diagnostics = {
            key: float(np.nanmean([row[key] for row in filter_diagnostic_rows]))
            for key in filter_diagnostic_rows[0]
        }
        hmm_filter_diagnostics.update(
            {
                key: float(
                    np.nanmean([row[key] for row in template_distance_rows])
                )
                for key in template_distance_rows[0]
            }
        )
        trailing_returns = all_returns.loc[all_returns.index < date].dropna(how="any")
        vix_spot = float("nan")
        vix_three_month = float("nan")
        vix_term_ratio = float("nan")
        vix_backwardation = False
        vix_hard_veto = False
        vix_guard_config = portfolio_config.get("vix_term_structure_guard")
        if isinstance(vix_guard_config, dict) and bool(
            vix_guard_config.get("enabled", False)
        ):
            spot_signal = str(vix_guard_config["spot_signal"])
            three_month_signal = str(vix_guard_config["three_month_signal"])
            missing_signals = [
                signal
                for signal in (spot_signal, three_month_signal)
                if signal not in all_prices
            ]
            if missing_signals:
                raise ValueError(
                    f"VIX term-structure guard is missing signals: {missing_signals}"
                )
            implied_history = all_prices.loc[
                all_prices.index < date, [spot_signal, three_month_signal]
            ].dropna(how="any")
            if implied_history.empty:
                raise ValueError("No causal VIX term-structure observation is available")
            latest_implied = implied_history.iloc[-1]
            vix_spot = float(latest_implied[spot_signal])
            vix_three_month = float(latest_implied[three_month_signal])
            vix_term_ratio, vix_backwardation = implied_volatility_term_structure(
                vix_spot,
                vix_three_month,
            )
            vix_hard_veto = bool(vix_guard_config.get("hard_veto", True))
        estimation_returns = trailing_returns.iloc[-len(training) :]
        unconditional_mean = estimation_returns.mean().to_numpy()
        unconditional_covariance = LedoitWolf().fit(estimation_returns.to_numpy()).covariance_
        mean_shrink = float(portfolio_config["conditional_mean_shrinkage"])
        covariance_shrink = float(portfolio_config["conditional_covariance_shrinkage"])
        daily_mean = (1.0 - mean_shrink) * conditional_mean + mean_shrink * unconditional_mean
        daily_covariance = (
            (1.0 - covariance_shrink) * conditional_covariance
            + covariance_shrink * unconditional_covariance
        )
        regime_annual_mean = annualization * daily_mean
        trend_forecast, trend_score = multi_horizon_trend_forecast(
            trailing_returns,
            [int(value) for value in portfolio_config["trend_horizons"]],  # type: ignore[union-attr]
            [float(value) for value in portfolio_config["trend_weights"]],  # type: ignore[union-attr]
            float(portfolio_config["trend_forecast_scale"]),
        )
        trend_blend = float(portfolio_config["trend_blend"])
        annual_mean = np.clip(
            (1.0 - trend_blend) * regime_annual_mean + trend_blend * trend_forecast,
            float(portfolio_config["annual_return_floor"]),
            float(portfolio_config["annual_return_cap"]),
        )
        annual_covariance = annualization * daily_covariance
        target = solve_allocation(
            annual_mean,
            annual_covariance,
            previous_weights,
            cash_index,
            float(portfolio_config["risk_aversion"]),
            float(portfolio_config["turnover_penalty"]),
            float(portfolio_config["max_risky_weight"]),
        )
        leverage_enabled = bool(portfolio_config.get("allow_leverage", False))
        leverage_signal_assets = list(
            portfolio_config.get("leverage_signal_assets", ["SPX"])  # type: ignore[arg-type]
        )
        leverage_signal_score = float(
            np.mean([trend_score[assets.index(asset)] for asset in leverage_signal_assets])
        )
        _, trend_stress = apply_growth_stress_guard(
            target,
            trailing_returns,
            assets,
            list(portfolio_config["growth_assets"]),  # type: ignore[arg-type]
            cash_index,
            int(portfolio_config["stress_momentum_days"]),
            int(portfolio_config["stress_volatility_days"]),
            float(portfolio_config["stress_volatility_threshold"]),
            float(portfolio_config["stress_growth_multiplier"]),
            str(portfolio_config.get("stress_signal_asset", "SPX")),
            annualization,
            str(portfolio_config.get("stress_volatility_estimator", "standard")),
        )
        paper_probability_allocation = bool(
            portfolio_config.get("paper_probability_allocation", False)
        )
        paper_regime_switch = bool(portfolio_config.get("paper_regime_switch", False))
        hmm_growth_expected_return = float("nan")
        hmm_growth_volatility = float("nan")
        hmm_growth_excess_return = float("nan")
        favorable_probability = float("nan")
        smoothed_favorable_probability = float("nan")
        target_growth_exposure = float("nan")
        risk_on = False
        paper_risk_on_candidate = False
        paper_growth_assets: list[str] = []
        growth_core_weights: dict[str, float] = {}
        paper_core_high_volatility = False
        paper_core_signal_volatility = float("nan")
        paper_core_high_volatility_threshold = float("nan")
        risk_on_core_trend_filtered_assets = 0
        if paper_probability_allocation or paper_regime_switch:
            paper_growth_assets = list(
                portfolio_config["paper_growth_assets"]  # type: ignore[arg-type]
            )
            paper_growth_weights = np.asarray(
                portfolio_config.get(
                    "paper_growth_weights",
                    [1.0 / len(paper_growth_assets)] * len(paper_growth_assets),
                ),
                dtype=float,
            )
            paper_growth_weights /= paper_growth_weights.sum()
            growth_indices = [assets.index(asset) for asset in paper_growth_assets]
            growth_mean = conditional_mean[growth_indices]
            growth_covariance = conditional_covariance[np.ix_(growth_indices, growth_indices)]
            core_weighting = str(
                portfolio_config.get("paper_growth_core_weighting", "fixed")
            )
            if core_weighting == "fixed":
                growth_core_weights = {
                    str(asset): float(weight)
                    for asset, weight in portfolio_config[
                        "strategic_core_weights"
                    ].items()  # type: ignore[union-attr]
                }
            elif core_weighting == "inverse_volatility":
                allocation_covariance = daily_covariance[
                    np.ix_(growth_indices, growth_indices)
                ]
                dynamic_weights = inverse_volatility_weights(allocation_covariance)
                growth_core_weights = dict(
                    zip(paper_growth_assets, dynamic_weights, strict=True)
                )
            elif core_weighting == "high_vol_inverse":
                allocation_covariance = daily_covariance[
                    np.ix_(growth_indices, growth_indices)
                ]
                strategic_weights = portfolio_config["strategic_core_weights"]
                base_weights = np.asarray(
                    [
                        float(strategic_weights.get(asset, 0.0))  # type: ignore[union-attr]
                        for asset in paper_growth_assets
                    ]
                )
                high_vol_config = portfolio_config["smh_high_volatility_budget"]
                dynamic_weights, paper_core_high_volatility, paper_core_signal_volatility, paper_core_high_volatility_threshold = high_volatility_core_weights(
                    allocation_covariance,
                    base_weights,
                    trailing_returns[str(high_vol_config["signal_asset"])],  # type: ignore[index]
                    int(high_vol_config["volatility_days"]),  # type: ignore[index]
                    int(high_vol_config["lookback_days"]),  # type: ignore[index]
                    float(high_vol_config["high_quantile"]),  # type: ignore[index]
                    annualization,
                )
                growth_core_weights = dict(
                    zip(paper_growth_assets, dynamic_weights, strict=True)
                )
            else:
                raise ValueError(
                    f"Unsupported paper growth core weighting: {core_weighting}"
                )
            core_trend_filter = portfolio_config.get(
                "risk_on_core_trend_filter"
            )
            if isinstance(core_trend_filter, dict) and bool(
                core_trend_filter.get("enabled", False)
            ):
                growth_core_weights, risk_on_core_trend_filtered_assets = (
                    filter_core_weights_by_trend(
                        growth_core_weights,
                        assets,
                        trend_score,
                        [
                            str(asset)
                            for asset in core_trend_filter["assets"]  # type: ignore[union-attr]
                        ],
                        float(core_trend_filter.get("minimum_score", 0.0)),
                        {
                            str(asset): float(weight)
                            for asset, weight in core_trend_filter.get(
                                "weak_weights",
                                {},
                            ).items()  # type: ignore[union-attr]
                        },
                    )
                )
            hmm_growth_expected_return = float(
                annualization * (paper_growth_weights @ growth_mean)
            )
            hmm_growth_volatility = float(
                np.sqrt(
                    max(
                        annualization
                        * (paper_growth_weights @ growth_covariance @ paper_growth_weights),
                        0.0,
                    )
                )
            )
            cash_expected_return = float(annualization * conditional_mean[cash_index])
            hmm_growth_excess_return = hmm_growth_expected_return - cash_expected_return

        if paper_probability_allocation:
            favorable_probability = float(
                sum(
                    probability
                    for probability, template in zip(
                        template_probabilities, self.tracker.templates, strict=True
                    )
                    if float(
                        paper_growth_weights @ template.return_mean[growth_indices]
                    )
                    > float(template.return_mean[cash_index])
                )
            )
            smoothing = float(
                portfolio_config.get(
                    "favorable_probability_smoothing",
                    model_config["template_smoothing"],
                )
            )
            if not 0.0 < smoothing <= 1.0:
                raise ValueError(
                    "Favorable-probability smoothing must be in the interval (0, 1]"
                )
            if self.smoothed_favorable_probability is None:
                self.smoothed_favorable_probability = favorable_probability
            else:
                self.smoothed_favorable_probability = (
                    (1.0 - smoothing) * self.smoothed_favorable_probability
                    + smoothing * favorable_probability
                )
            smoothed_favorable_probability = self.smoothed_favorable_probability
            minimum_exposure = float(portfolio_config["min_growth_exposure"])
            maximum_exposure = float(portfolio_config["max_growth_exposure"])
            if not 0.0 <= minimum_exposure <= maximum_exposure:
                raise ValueError("Growth exposure bounds are invalid")
            target_growth_exposure = minimum_exposure + (
                maximum_exposure - minimum_exposure
            ) * smoothed_favorable_probability
            defensive_target = apply_asset_multiplier(
                target,
                assets,
                list(portfolio_config["growth_assets"]),  # type: ignore[arg-type]
                cash_index,
                0.0,
            )
            growth_core = blend_strategic_core(
                target,
                assets,
                cash_index,
                1.0,
                growth_core_weights,
            )
            target = compose_probability_allocation(
                defensive_target,
                growth_core,
                cash_index,
                target_growth_exposure,
            )
            risk_on = smoothed_favorable_probability >= 0.5
        elif paper_regime_switch:
            paper_risk_on_candidate = paper_regime_candidate(
                leverage_enabled,
                hmm_growth_excess_return,
                bool(
                    portfolio_config.get(
                        "paper_use_auxiliary_trend_filters",
                        True,
                    )
                ),
                trend_stress,
                leverage_signal_score,
                float(portfolio_config.get("leverage_activation_score", 0.0)),
            )
            if vix_hard_veto:
                paper_risk_on_candidate = (
                    paper_risk_on_candidate and not vix_backwardation
                )
            drawdown_conditioned_hmm_veto = bool(
                portfolio_config.get("paper_hmm_veto_drawdown_only", False)
            )
            if drawdown_conditioned_hmm_veto:
                drawdown_tiers = portfolio_config["drawdown_overlay"]
                hmm_guard_drawdown = max(
                    float(tier["threshold"])
                    for tier in drawdown_tiers  # type: ignore[union-attr]
                )
                paper_risk_on_candidate = drawdown_conditioned_regime_candidate(
                    leverage_enabled,
                    hmm_growth_excess_return,
                    trend_stress,
                    leverage_signal_score,
                    float(portfolio_config.get("leverage_activation_score", 0.0)),
                    drawdown,
                    hmm_guard_drawdown,
                )
                risk_on = paper_risk_on_candidate
            elif bool(portfolio_config.get("paper_regime_reentry_on_refit", False)):
                risk_on = update_asymmetric_regime_state(
                    self.paper_risk_on_state,
                    paper_risk_on_candidate,
                    should_refit,
                )
            else:
                risk_on = paper_risk_on_candidate
            self.paper_risk_on_state = risk_on
            if risk_on:
                target = blend_strategic_core(
                    target,
                    assets,
                    cash_index,
                    1.0,
                    growth_core_weights,
                )
            else:
                target = apply_asset_multiplier(
                    target,
                    assets,
                    list(portfolio_config["growth_assets"]),  # type: ignore[arg-type]
                    cash_index,
                    0.0,
                )
        else:
            if "strategic_core_weights" in portfolio_config:
                target = blend_strategic_core(
                    target,
                    assets,
                    cash_index,
                    float(portfolio_config["strategic_core_blend"]),
                    portfolio_config["strategic_core_weights"],  # type: ignore[arg-type]
                )
            else:
                target = blend_equal_weight_core(
                    target,
                    cash_index,
                    float(portfolio_config["strategic_core_blend"]),
                )
            target, trend_stress = apply_growth_stress_guard(
                target,
                trailing_returns,
                assets,
                list(portfolio_config["growth_assets"]),  # type: ignore[arg-type]
                cash_index,
                int(portfolio_config["stress_momentum_days"]),
                int(portfolio_config["stress_volatility_days"]),
                float(portfolio_config["stress_volatility_threshold"]),
                float(portfolio_config["stress_growth_multiplier"]),
                str(portfolio_config.get("stress_signal_asset", "SPX")),
                annualization,
                str(portfolio_config.get("stress_volatility_estimator", "standard")),
            )
            risk_on = (
                leverage_enabled
                and not trend_stress
                and leverage_signal_score
                >= float(portfolio_config.get("leverage_activation_score", 0.0))
            )
        defensive_trend_active = False
        defensive_trend_selected_assets = 0
        defensive_trend_config = portfolio_config.get("risk_off_defensive_trend")
        if (
            isinstance(defensive_trend_config, dict)
            and bool(defensive_trend_config.get("enabled", False))
            and not risk_on
        ):
            target, defensive_trend_selected_assets = long_only_trend_defensive_weights(
                trailing_returns,
                assets,
                [
                    str(asset)
                    for asset in defensive_trend_config["assets"]  # type: ignore[union-attr]
                ],
                cash_index,
                trend_score,
                int(defensive_trend_config["volatility_days"]),
                [
                    str(asset)
                    for asset in defensive_trend_config.get(
                        "fast_trend_assets", []
                    )  # type: ignore[union-attr]
                ],
                (
                    int(defensive_trend_config["fast_trend_days"])
                    if "fast_trend_days" in defensive_trend_config
                    else None
                ),
            )
            defensive_trend_active = True
        risk_off_growth_trend_sleeve_share = 0.0
        risk_off_growth_trend_selected_assets = 0
        growth_trend_sleeve_config = portfolio_config.get(
            "risk_off_growth_trend_sleeve"
        )
        if (
            isinstance(growth_trend_sleeve_config, dict)
            and bool(growth_trend_sleeve_config.get("enabled", False))
            and not risk_on
        ):
            growth_trend_sleeve, risk_off_growth_trend_selected_assets = (
                long_only_trend_defensive_weights(
                    trailing_returns,
                    assets,
                    [
                        str(asset)
                        for asset in growth_trend_sleeve_config["assets"]  # type: ignore[union-attr]
                    ],
                    cash_index,
                    trend_score,
                    int(growth_trend_sleeve_config["volatility_days"]),
                    [
                        str(asset)
                        for asset in growth_trend_sleeve_config["assets"]  # type: ignore[union-attr]
                    ],
                    int(growth_trend_sleeve_config["fast_trend_days"]),
                )
            )
            if risk_off_growth_trend_selected_assets > 0:
                risk_off_growth_trend_sleeve_share = float(
                    growth_trend_sleeve_config["account_share"]
                )
                target = compose_probability_allocation(
                    target,
                    growth_trend_sleeve,
                    cash_index,
                    risk_off_growth_trend_sleeve_share,
                )
        risk_off_growth_floor_share = 0.0
        growth_floor_config = portfolio_config.get("risk_off_growth_floor")
        floor_stress_guard_active = trend_stress
        disable_floor_when_stressed = False
        if isinstance(growth_floor_config, dict):
            disable_floor_when_stressed = bool(
                growth_floor_config.get("disable_when_trend_stressed", False)
            )
            if bool(
                growth_floor_config.get("disable_when_volatility_stressed", False)
            ):
                floor_stress_guard_active = realized_volatility_stress(
                    trailing_returns[str(portfolio_config["stress_signal_asset"])],
                    int(portfolio_config["stress_volatility_days"]),
                    float(portfolio_config["stress_volatility_threshold"]),
                    annualization,
                    str(
                        portfolio_config.get(
                            "stress_volatility_estimator",
                            "standard",
                        )
                    ),
                )
                disable_floor_when_stressed = True
        if (
            isinstance(growth_floor_config, dict)
            and bool(growth_floor_config.get("enabled", False))
            and risk_off_growth_floor_is_active(
                risk_on,
                floor_stress_guard_active,
                disable_floor_when_stressed,
            )
        ):
            floor_core = blend_strategic_core(
                target,
                assets,
                cash_index,
                1.0,
                growth_floor_config["core_weights"],  # type: ignore[arg-type]
            )
            requested_floor_share = float(growth_floor_config["account_share"])
            backwardation_floor_cap = (
                vix_guard_config.get("backwardation_growth_floor_cap")
                if isinstance(vix_guard_config, dict)
                else None
            )
            if backwardation_floor_cap is not None:
                requested_floor_share = (
                    incremental_growth_floor_term_structure_cap(
                        requested_floor_share,
                        float(backwardation_floor_cap),
                        vix_backwardation,
                    )
                )
            target, risk_off_growth_floor_share = apply_risk_off_growth_floor(
                target,
                floor_core,
                cash_index,
                requested_floor_share,
                risk_on,
            )
        diversifier_trend_sleeve_share = 0.0
        diversifier_trend_selected_assets = 0
        diversifier_trend_config = portfolio_config.get(
            "diversifier_trend_sleeve"
        )
        if isinstance(diversifier_trend_config, dict) and bool(
            diversifier_trend_config.get("enabled", False)
        ):
            diversifier_trend_sleeve, diversifier_trend_selected_assets = (
                long_only_trend_defensive_weights(
                    trailing_returns,
                    assets,
                    [
                        str(asset)
                        for asset in diversifier_trend_config["assets"]  # type: ignore[union-attr]
                    ],
                    cash_index,
                    trend_score,
                    int(diversifier_trend_config["volatility_days"]),
                )
            )
            diversifier_trend_sleeve_share = float(
                diversifier_trend_config["account_share"]
            )
            target = compose_probability_allocation(
                target,
                diversifier_trend_sleeve,
                cash_index,
                diversifier_trend_sleeve_share,
            )
        continuous_probability_risk_target = bool(
            portfolio_config.get("continuous_probability_risk_target", False)
        )
        leverage_trend_gate_active = True
        leverage_trend_gate_min_score = float("nan")
        leverage_trend_gate_config = portfolio_config.get("leverage_trend_gate")
        if isinstance(leverage_trend_gate_config, dict) and bool(
            leverage_trend_gate_config.get("enabled", False)
        ):
            (
                leverage_trend_gate_active,
                leverage_trend_gate_scores,
            ) = joint_excess_trend_leverage_gate(
                trailing_returns,
                leverage_signal_assets,
                assets[cash_index],
                leverage_trend_gate_config.get("lookback_days", 252),
            )
            finite_gate_scores = leverage_trend_gate_scores[
                np.isfinite(leverage_trend_gate_scores)
            ]
            if finite_gate_scores.size:
                leverage_trend_gate_min_score = float(
                    finite_gate_scores.min()
                )
        leverage_gate_enabled = (
            risk_on or continuous_probability_risk_target
        ) and leverage_trend_gate_active
        leverage_vix_gate_active = incremental_leverage_term_structure_gate(
            vix_backwardation,
            bool(
                isinstance(vix_guard_config, dict)
                and vix_guard_config.get("incremental_leverage_veto", False)
            ),
        )
        leverage_gate_enabled = (
            leverage_gate_enabled and leverage_vix_gate_active
        )
        if not paper_probability_allocation or continuous_probability_risk_target:
            target, _ = apply_risk_on_leverage(
                target,
                annual_covariance,
                cash_index,
                float(portfolio_config["target_volatility"]),
                float(portfolio_config.get("max_gross_leverage", 1.0)),
                leverage_gate_enabled,
            )
        target = apply_volatility_target(
            target,
            annual_covariance,
            cash_index,
            float(portfolio_config["target_volatility"]),
        )
        relative_momentum_score = float("nan")
        relative_momentum_raw_tilt_weight = float("nan")
        relative_momentum_tilt_weight = float("nan")
        relative_momentum_spread_volatility = float("nan")
        relative_momentum_risk_multiplier = 1.0
        relative_momentum_volatility_state = -1
        relative_momentum_volatility_gate_active = False
        relative_momentum_gate_fallback_weight = float("nan")
        relative_momentum_pair_exposure = 0.0
        relative_momentum_concentration_reallocated = 0.0
        relative_momentum_config = portfolio_config.get("relative_momentum_core")
        if isinstance(relative_momentum_config, dict) and bool(
            relative_momentum_config.get("enabled", False)
        ):
            anchor_asset = str(relative_momentum_config["anchor_asset"])
            tilt_asset = str(relative_momentum_config["tilt_asset"])
            relative_momentum_raw_tilt_weight, relative_momentum_score = (
                relative_momentum_pair_weight(
                    trailing_returns[anchor_asset],
                    trailing_returns[tilt_asset],
                    [
                        int(value)
                        for value in relative_momentum_config["horizons"]  # type: ignore[union-attr]
                    ],
                    [
                        float(value)
                        for value in relative_momentum_config["horizon_weights"]  # type: ignore[union-attr]
                    ],
                    int(relative_momentum_config["skip_days"]),
                    float(relative_momentum_config["base_tilt_weight"]),
                    float(relative_momentum_config["max_tilt"]),
                )
            )
            relative_momentum_tilt_weight = relative_momentum_raw_tilt_weight
            volatility_scaling_config = relative_momentum_config.get(
                "volatility_scaling"
            )
            if isinstance(volatility_scaling_config, dict) and bool(
                volatility_scaling_config.get("enabled", False)
            ):
                (
                    relative_momentum_tilt_weight,
                    relative_momentum_spread_volatility,
                    relative_momentum_risk_multiplier,
                ) = volatility_managed_relative_momentum_weight(
                    relative_momentum_raw_tilt_weight,
                    float(relative_momentum_config["base_tilt_weight"]),
                    trailing_returns[anchor_asset],
                    trailing_returns[tilt_asset],
                    int(volatility_scaling_config["lookback_days"]),
                    float(volatility_scaling_config["target_volatility"]),
                    annualization,
                )
            volatility_gate_config = relative_momentum_config.get(
                "volatility_regime_gate"
            )
            if isinstance(volatility_gate_config, dict) and bool(
                volatility_gate_config.get("enabled", False)
            ):
                pair_returns = trailing_returns[
                    [anchor_asset, tilt_asset]
                ].mean(axis=1)
                volatility_state, _ = causal_realized_volatility_state(
                    pair_returns,
                    int(volatility_gate_config.get("window_days", 20)),
                    int(
                        volatility_gate_config.get(
                            "threshold_lookback_days",
                            756,
                        )
                    ),
                    int(
                        volatility_gate_config.get(
                            "minimum_threshold_observations",
                            504,
                        )
                    ),
                    float(volatility_gate_config.get("low_quantile", 0.45)),
                    float(volatility_gate_config.get("high_quantile", 0.90)),
                )
                relative_momentum_volatility_state = {
                    "insufficient": -1,
                    "low": 0,
                    "medium": 1,
                    "high": 2,
                }[volatility_state]
                gate_fallback_weight = float(
                    relative_momentum_config["base_tilt_weight"]
                )
                fallback_weighting = str(
                    volatility_gate_config.get(
                        "fallback_weighting",
                        "base",
                    )
                )
                if fallback_weighting == "inverse_volatility":
                    fallback_days = int(
                        volatility_gate_config.get(
                            "fallback_lookback_days",
                            60,
                        )
                    )
                    pair_sample = trailing_returns[
                        [anchor_asset, tilt_asset]
                    ].dropna(how="any").iloc[-fallback_days:]
                    if len(pair_sample) >= fallback_days:
                        pair_volatility = pair_sample.std(ddof=1).to_numpy(
                            dtype=float
                        )
                        if np.all(pair_volatility > 0.0):
                            inverse = 1.0 / pair_volatility
                            gate_fallback_weight = float(
                                inverse[1] / inverse.sum()
                            )
                elif fallback_weighting != "base":
                    raise ValueError(
                        "Relative-momentum volatility gate fallback must be "
                        "'base' or 'inverse_volatility'"
                    )
                relative_momentum_gate_fallback_weight = gate_fallback_weight
                (
                    relative_momentum_tilt_weight,
                    relative_momentum_volatility_gate_active,
                ) = volatility_regime_gated_relative_momentum_weight(
                    relative_momentum_tilt_weight,
                    gate_fallback_weight,
                    volatility_state,
                    [
                        str(state)
                        for state in volatility_gate_config.get(
                            "allowed_states",
                            ["insufficient", "low", "medium"],
                        )
                    ],
                )
            target, relative_momentum_pair_exposure = reallocate_pair_weights(
                target,
                assets,
                anchor_asset,
                tilt_asset,
                relative_momentum_tilt_weight,
            )
            maximum_account_weight = relative_momentum_config.get(
                "maximum_tilt_asset_account_weight"
            )
            if maximum_account_weight is not None:
                (
                    target,
                    relative_momentum_concentration_reallocated,
                ) = cap_pair_asset_weight(
                    target,
                    assets,
                    anchor_asset,
                    tilt_asset,
                    float(maximum_account_weight),
                )
        residual_momentum_score = float("nan")
        residual_momentum_beta = float("nan")
        residual_momentum_controlled_weight = float("nan")
        residual_momentum_pair_exposure = 0.0
        residual_momentum_config = portfolio_config.get("residual_momentum_core")
        if isinstance(residual_momentum_config, dict) and bool(
            residual_momentum_config.get("enabled", False)
        ):
            anchor_asset = str(residual_momentum_config["anchor_asset"])
            controlled_asset = str(residual_momentum_config["controlled_asset"])
            (
                residual_momentum_controlled_weight,
                residual_momentum_score,
                residual_momentum_beta,
            ) = residual_momentum_beta_pair_weight(
                trailing_returns[anchor_asset],
                trailing_returns[controlled_asset],
                int(residual_momentum_config["beta_months"]),
                int(residual_momentum_config["formation_months"]),
                int(residual_momentum_config["skip_months"]),
                float(residual_momentum_config["base_controlled_weight"]),
            )
            target, residual_momentum_pair_exposure = reallocate_pair_weights(
                target,
                assets,
                anchor_asset,
                controlled_asset,
                residual_momentum_controlled_weight,
            )
        vix_recovery_bridge_share = 0.0
        vix_recovery_trend_confirmed = False
        recovery_bridge_config = portfolio_config.get("vix_recovery_bridge")
        if isinstance(recovery_bridge_config, dict) and bool(
            recovery_bridge_config.get("enabled", False)
        ):
            if not isinstance(vix_guard_config, dict) or not bool(
                vix_guard_config.get("enabled", False)
            ):
                raise ValueError("VIX recovery bridge requires term-structure signals")
            vix_recovery_trend_confirmed = (
                not bool(recovery_bridge_config.get("require_positive_trend", False))
                or leverage_signal_score
                >= float(portfolio_config.get("leverage_activation_score", 0.0))
            )
            self.vix_recovery_bridge_active = update_vix_recovery_bridge(
                self.vix_recovery_bridge_active,
                self.previous_vix_backwardation,
                vix_backwardation,
                risk_on,
                vix_recovery_trend_confirmed,
            )
            if self.vix_recovery_bridge_active:
                vix_recovery_bridge_share = float(
                    recovery_bridge_config["account_share"]
                )
                recovery_core = blend_strategic_core(
                    target,
                    assets,
                    cash_index,
                    1.0,
                    growth_core_weights,
                )
                target = compose_probability_allocation(
                    target,
                    recovery_core,
                    cash_index,
                    vix_recovery_bridge_share,
                )
        if isinstance(vix_guard_config, dict) and bool(
            vix_guard_config.get("enabled", False)
        ):
            self.previous_vix_backwardation = vix_backwardation
        satellite_active = False
        satellite_internal_exposure = 0.0
        satellite_annual_volatility = float("nan")
        satellite_account_target = 0.0
        satellite_effective_share = 0.0
        satellite_config = portfolio_config.get("smh_satellite")
        if isinstance(satellite_config, dict) and bool(
            satellite_config.get("enabled", False)
        ):
            satellite_asset = str(satellite_config["asset"])
            satellite_asset_index = assets.index(satellite_asset)
            satellite_internal_exposure, satellite_active, satellite_annual_volatility = (
                trend_volatility_exposure(
                    trailing_returns[satellite_asset],
                    int(satellite_config["trend_days"]),
                    int(satellite_config["volatility_days"]),
                    float(satellite_config["target_volatility"]),
                    float(satellite_config["max_internal_exposure"]),
                    annualization,
                )
            )
            satellite_effective_share = satellite_share_for_mode(
                float(satellite_config["account_share"]),
                str(satellite_config.get("activation_mode", "always")),
                satellite_active,
                risk_on,
            )
            satellite_account_target = (
                satellite_effective_share * satellite_internal_exposure
            )
            target = blend_satellite_allocation(
                target,
                satellite_asset_index,
                cash_index,
                satellite_effective_share,
                satellite_internal_exposure,
                float(satellite_config["total_asset_cap"]),
            )
        asset_aware_core_weights = np.asarray([], dtype=float)
        asset_aware_effective_volatility = np.asarray([], dtype=float)
        asset_aware_unit_volatility = float("nan")
        asset_aware_growth_cap = float("nan")
        asset_relative_risk_multiplier = float("nan")
        asset_risk_config = portfolio_config.get("asset_aware_growth_risk")
        asset_risk_assets: list[str] = []
        asset_risk_overlay_active = False
        previous_asset_risk_growth_exposure = float("nan")
        zero_growth_relative_return = float("nan")
        asset_risk_cash_destination = False
        account_risk_volatility = float("nan")
        account_risk_multiplier = 1.0
        if isinstance(asset_risk_config, dict) and bool(
            asset_risk_config.get("enabled", False)
        ) and bool(asset_risk_config.get("scheduled_rebalance", True)):
            asset_risk_assets = [
                str(asset)
                for asset in asset_risk_config["assets"]  # type: ignore[union-attr]
            ]
            previous_asset_risk_growth_exposure = float(
                previous_weights[
                    [assets.index(asset) for asset in asset_risk_assets]
                ].sum()
            )
            proposed_growth_exposure = float(
                target[[assets.index(asset) for asset in asset_risk_assets]].sum()
            )
            prior_close = all_prices.loc[
                all_prices.index < date, asset_risk_assets
            ].iloc[-1]
            relative_price = float(prior_close.iloc[1] / prior_close.iloc[0])
            if proposed_growth_exposure <= 1e-12:
                if self.zero_growth_relative_price_start is None:
                    self.zero_growth_relative_price_start = relative_price
            elif self.zero_growth_relative_price_start is not None:
                zero_growth_relative_return = (
                    relative_price / self.zero_growth_relative_price_start - 1.0
                )
            activation_mode = str(
                asset_risk_config.get("activation_mode", "always")
            )
            if activation_mode not in {
                "always",
                "risk_on_entry",
                "zero_growth_entry",
            }:
                raise ValueError(
                    f"Unknown asset-aware risk activation mode: {activation_mode}"
                )
            asset_risk_overlay_active = (
                activation_mode == "always"
                or (
                    activation_mode == "risk_on_entry"
                    and risk_on
                    and previous_paper_risk_on_state is not True
                )
                or (
                    activation_mode == "zero_growth_entry"
                    and risk_on
                    and previous_asset_risk_growth_exposure <= 1e-12
                )
            )
            if asset_risk_overlay_active:
                allocation_method = str(
                    asset_risk_config.get("allocation_method", "total_volatility")
                )
                dynamic_zero_entry_method = (
                    allocation_method == "zero_episode_relative_switch"
                )
                if dynamic_zero_entry_method:
                    allocation_method = zero_entry_allocation_method(
                        zero_growth_relative_return
                    )
                if allocation_method == "total_volatility":
                    (
                        target,
                        asset_aware_core_weights,
                        asset_aware_effective_volatility,
                        asset_aware_unit_volatility,
                        asset_aware_growth_cap,
                    ) = apply_asset_aware_growth_risk_budget(
                        target,
                        trailing_returns,
                        assets,
                        asset_risk_assets,
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        float(asset_risk_config["target_volatility"]),
                        {
                            str(asset): float(value)
                            for asset, value in asset_risk_config.get(
                                "maximum_core_weights", {}
                            ).items()  # type: ignore[union-attr]
                        },
                        annualization,
                        bool(
                            asset_risk_config.get(
                                "activate_only_when_breached", False
                            )
                        ),
                        (
                            False
                            if dynamic_zero_entry_method
                            else bool(
                                asset_risk_config.get(
                                    "cap_total_volatility", True
                                )
                            )
                        ),
                    )
                elif allocation_method == "relative_asset_cap":
                    asset_risk_cash_destination = True
                    if len(asset_risk_assets) != 2:
                        raise ValueError(
                            "Relative asset risk cap requires exactly two assets"
                        )
                    (
                        target,
                        asset_aware_effective_volatility,
                        asset_relative_risk_multiplier,
                    ) = apply_relative_asset_risk_cap(
                        target,
                        trailing_returns,
                        assets,
                        asset_risk_assets[0],
                        asset_risk_assets[1],
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        annualization,
                        (
                            float(asset_risk_config["target_volatility"])
                            if bool(
                                asset_risk_config.get(
                                    "activate_only_when_breached", False
                                )
                            )
                            else None
                        ),
                    )
                    growth_indices = [
                        assets.index(asset) for asset in asset_risk_assets
                    ]
                    growth_total = float(target[growth_indices].sum())
                    asset_aware_growth_cap = growth_total
                    if growth_total > 1e-12:
                        asset_aware_core_weights = (
                            target[growth_indices] / growth_total
                        )
                else:
                    raise ValueError(
                        f"Unknown asset-aware allocation method: {allocation_method}"
                    )
                if bool(asset_risk_config.get("account_volatility_cap", False)):
                    (
                        target,
                        account_risk_volatility,
                        account_risk_multiplier,
                    ) = apply_account_stress_volatility_cap(
                        target,
                        trailing_returns,
                        assets,
                        [
                            str(asset)
                            for asset in asset_risk_config[
                                "account_volatility_cap_assets"
                            ]  # type: ignore[union-attr]
                        ],
                        cash_index,
                        int(asset_risk_config["fast_days"]),
                        int(asset_risk_config["slow_days"]),
                        float(asset_risk_config["target_volatility"]),
                        annualization,
                    )
            if proposed_growth_exposure > 1e-12:
                self.zero_growth_relative_price_start = None
        target, drawdown_multiplier = apply_drawdown_overlay(
            target,
            cash_index,
            drawdown,
            portfolio_config["drawdown_overlay"],  # type: ignore[arg-type]
        )
        relative_momentum_spread_volatility = float("nan")
        relative_momentum_spread_active_risk = 0.0
        relative_momentum_spread_requested_share = 0.0
        relative_momentum_spread_applied_share = 0.0
        relative_momentum_spread_config = (
            relative_momentum_config.get("spread_overlay")
            if isinstance(relative_momentum_config, dict)
            else None
        )
        if isinstance(relative_momentum_spread_config, dict) and bool(
            relative_momentum_spread_config.get("enabled", False)
        ):
            anchor_asset = str(relative_momentum_config["anchor_asset"])
            tilt_asset = str(relative_momentum_config["tilt_asset"])
            (
                relative_momentum_spread_overlay,
                relative_momentum_spread_volatility,
                relative_momentum_spread_active_risk,
            ) = relative_momentum_spread_overlay_weights(
                trailing_returns[anchor_asset],
                trailing_returns[tilt_asset],
                assets,
                anchor_asset,
                tilt_asset,
                cash_index,
                relative_momentum_score,
                int(relative_momentum_spread_config["lookback_days"]),
                float(relative_momentum_spread_config["target_volatility"]),
                annualization,
            )
            relative_momentum_spread_requested_share = float(
                relative_momentum_spread_config["account_share"]
            )
            (
                target,
                relative_momentum_spread_applied_share,
                _,
            ) = apply_self_financing_overlay(
                target,
                relative_momentum_spread_overlay,
                cash_index,
                relative_momentum_spread_requested_share,
                float(
                    relative_momentum_spread_config.get(
                        "maximum_portfolio_gross",
                        portfolio_config.get("max_gross_leverage", 1.0),
                    )
                ),
            )
        time_series_momentum_signals = np.asarray([], dtype=float)
        time_series_momentum_target_volatility = float("nan")
        time_series_momentum_requested_share = 0.0
        time_series_momentum_applied_share = 0.0
        time_series_momentum_config = portfolio_config.get(
            "time_series_momentum_overlay"
        )
        if isinstance(time_series_momentum_config, dict) and bool(
            time_series_momentum_config.get("enabled", False)
        ):
            time_series_momentum_assets = [
                str(asset)
                for asset in time_series_momentum_config["assets"]  # type: ignore[union-attr]
            ]
            (
                time_series_momentum_overlay,
                time_series_momentum_signals,
                time_series_momentum_target_volatility,
            ) = time_series_momentum_overlay_weights(
                trailing_returns,
                assets,
                time_series_momentum_assets,
                cash_index,
                [
                    int(value)
                    for value in time_series_momentum_config["horizons"]  # type: ignore[union-attr]
                ],
                int(time_series_momentum_config["volatility_lookback_days"]),
                float(time_series_momentum_config["target_volatility"]),
                annualization,
            )
            time_series_momentum_requested_share = float(
                self_financing_overlay_share_for_mode(
                    float(time_series_momentum_config["account_share"]),
                    str(
                        time_series_momentum_config.get(
                            "activation_mode",
                            "always",
                        )
                    ),
                    risk_on,
                )
            )
            (
                target,
                time_series_momentum_applied_share,
                gross_leverage,
            ) = apply_self_financing_overlay(
                target,
                time_series_momentum_overlay,
                cash_index,
                time_series_momentum_requested_share,
                float(
                    time_series_momentum_config.get(
                        "maximum_portfolio_gross",
                        portfolio_config.get("max_gross_leverage", 1.0),
                    )
                ),
            )
        else:
            time_series_momentum_assets = []
            gross_leverage = float(np.abs(np.delete(target, cash_index)).sum())
        target, standalone_risk_diagnostics = self._standalone_risk_cap_target(
            date,
            all_returns,
            target,
            assets,
            cash_index,
            annualization,
            portfolio_config,
        )
        gross_leverage = float(np.abs(np.delete(target, cash_index)).sum())
        dominant = int(np.argmax(template_probabilities))
        predicted_volatility = float(np.sqrt(max(target @ annual_covariance @ target, 0.0)))
        diagnostics: dict[str, float | int] = {
            "hmm_order": int(self.selected_order),
            "hmm_ensemble_count": len(ensemble_orders),
            "hmm_student_t_emission": int(
                str(model_config.get("emission_distribution", "gaussian"))
                == "student_t"
            ),
            "hmm_refit_stress": int(refit_stress),
            "hmm_effective_refit_days": refit_days,
            "hmm_refit_state_transition": int(refit_state_transition),
            "hmm_template_updated": int(template_update),
            "hmm_filtered_return_moments": int(
                str(model_config.get("return_moment_posterior", "smoothed"))
                == "filtered"
            ),
            "hmm_conditioned_order_validation": int(
                bool(model_config.get("condition_order_validation_on_train", False))
            ),
            "hmm_robust_feature_scaler": int(
                str(model_config.get("feature_scaler", "standard")) == "robust"
            ),
            "hmm_empirical_bayes_return_means": int(
                str(model_config.get("state_return_mean_shrinkage", "none"))
                == "empirical_bayes"
            ),
            "hmm_huber_return_location": int(
                str(model_config.get("return_location_estimator", "mean"))
                == "huber"
            ),
            "hmm_tied_covariance": int(
                str(model_config.get("covariance_type", "diag")) == "tied"
            ),
            **hmm_fit_diagnostics(self.fit.model),
            "fit_conditional_moments": int(
                str(model_config.get("conditional_moments_source", "templates"))
                == "fit"
            ),
            "dominant_template": dominant,
            "dominant_probability": float(template_probabilities[dominant]),
            **hmm_filter_diagnostics,
            "predicted_volatility": predicted_volatility,
            "drawdown_multiplier": drawdown_multiplier,
            "trend_stress": int(trend_stress),
            "risk_on_leverage": int(leverage_gate_enabled),
            "leverage_trend_gate_active": int(leverage_trend_gate_active),
            "leverage_trend_gate_min_score": leverage_trend_gate_min_score,
            "leverage_vix_gate_active": int(leverage_vix_gate_active),
            "paper_risk_on_candidate": int(paper_risk_on_candidate),
            "drawdown_conditioned_hmm_veto": int(
                drawdown_conditioned_hmm_veto
                if paper_regime_switch
                else False
            ),
            "gross_leverage": gross_leverage,
            "time_series_momentum_target_volatility": (
                time_series_momentum_target_volatility
            ),
            "time_series_momentum_requested_share": (
                time_series_momentum_requested_share
            ),
            "time_series_momentum_applied_share": (
                time_series_momentum_applied_share
            ),
            "hmm_growth_expected_return": hmm_growth_expected_return,
            "hmm_growth_volatility": hmm_growth_volatility,
            "hmm_growth_excess_return": hmm_growth_excess_return,
            "favorable_probability": favorable_probability,
            "smoothed_favorable_probability": smoothed_favorable_probability,
            "target_growth_exposure": target_growth_exposure,
            "continuous_probability_risk_target": int(
                continuous_probability_risk_target
            ),
            "vix_spot": vix_spot,
            "vix_three_month": vix_three_month,
            "vix_term_ratio": vix_term_ratio,
            "vix_backwardation": int(vix_backwardation),
            "vix_recovery_bridge_active": int(self.vix_recovery_bridge_active),
            "vix_recovery_bridge_share": vix_recovery_bridge_share,
            "vix_recovery_trend_confirmed": int(vix_recovery_trend_confirmed),
            "paper_core_high_volatility": int(paper_core_high_volatility),
            "paper_core_signal_volatility": paper_core_signal_volatility,
            "paper_core_high_volatility_threshold": paper_core_high_volatility_threshold,
            "risk_on_core_trend_filtered_assets": (
                risk_on_core_trend_filtered_assets
            ),
            "relative_momentum_score": relative_momentum_score,
            "relative_momentum_raw_tilt_weight": relative_momentum_raw_tilt_weight,
            "relative_momentum_tilt_weight": relative_momentum_tilt_weight,
            "relative_momentum_spread_volatility": relative_momentum_spread_volatility,
            "relative_momentum_risk_multiplier": relative_momentum_risk_multiplier,
            "relative_momentum_volatility_state": (
                relative_momentum_volatility_state
            ),
            "relative_momentum_volatility_gate_active": int(
                relative_momentum_volatility_gate_active
            ),
            "relative_momentum_gate_fallback_weight": (
                relative_momentum_gate_fallback_weight
            ),
            "relative_momentum_pair_exposure": relative_momentum_pair_exposure,
            "relative_momentum_concentration_reallocated": (
                relative_momentum_concentration_reallocated
            ),
            "relative_momentum_spread_volatility": (
                relative_momentum_spread_volatility
            ),
            "relative_momentum_spread_active_risk": (
                relative_momentum_spread_active_risk
            ),
            "relative_momentum_spread_requested_share": (
                relative_momentum_spread_requested_share
            ),
            "relative_momentum_spread_applied_share": (
                relative_momentum_spread_applied_share
            ),
            "residual_momentum_score": residual_momentum_score,
            "residual_momentum_beta": residual_momentum_beta,
            "residual_momentum_controlled_weight": residual_momentum_controlled_weight,
            "residual_momentum_pair_exposure": residual_momentum_pair_exposure,
            "risk_off_growth_floor_share": risk_off_growth_floor_share,
            "risk_off_growth_trend_sleeve_share": risk_off_growth_trend_sleeve_share,
            "risk_off_growth_trend_selected_assets": risk_off_growth_trend_selected_assets,
            "diversifier_trend_sleeve_share": diversifier_trend_sleeve_share,
            "diversifier_trend_selected_assets": diversifier_trend_selected_assets,
            "defensive_trend_active": int(defensive_trend_active),
            "defensive_trend_selected_assets": defensive_trend_selected_assets,
            "satellite_active": int(satellite_active),
            "satellite_internal_exposure": satellite_internal_exposure,
            "satellite_annual_volatility": satellite_annual_volatility,
            "satellite_account_target": satellite_account_target,
            "satellite_effective_share": satellite_effective_share,
            "asset_aware_unit_volatility": asset_aware_unit_volatility,
            "asset_aware_growth_cap": asset_aware_growth_cap,
            "asset_risk_overlay_active": int(asset_risk_overlay_active),
            "asset_relative_risk_multiplier": asset_relative_risk_multiplier,
            "previous_asset_risk_growth_exposure": previous_asset_risk_growth_exposure,
            "zero_growth_relative_return": zero_growth_relative_return,
            "asset_risk_cash_destination": int(asset_risk_cash_destination),
            "account_risk_volatility": account_risk_volatility,
            "account_risk_multiplier": account_risk_multiplier,
            "spx_trend_score": float(trend_score[assets.index("SPX")]),
            "leverage_signal_score": leverage_signal_score,
            **standalone_risk_diagnostics,
        }
        for index, probability in enumerate(template_probabilities):
            diagnostics[f"template_{index}_probability"] = float(probability)
        for asset, signal in zip(
            time_series_momentum_assets,
            time_series_momentum_signals,
            strict=True,
        ):
            diagnostics[f"time_series_momentum_{asset}_signal"] = float(signal)
        for asset in paper_growth_assets:
            diagnostics[f"paper_core_{asset}_weight"] = float(
                growth_core_weights.get(asset, 0.0)
            )
        for index, asset in enumerate(asset_risk_assets):
            diagnostics[f"asset_aware_{asset}_core_weight"] = (
                float(asset_aware_core_weights[index])
                if index < len(asset_aware_core_weights)
                else float("nan")
            )
            diagnostics[f"asset_aware_{asset}_effective_volatility"] = (
                float(asset_aware_effective_volatility[index])
                if index < len(asset_aware_effective_volatility)
                else float("nan")
            )
        return target, diagnostics

    @staticmethod
    def _benchmarks(returns: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
        benchmark = pd.DataFrame(index=returns.index)
        benchmark["SPX"] = returns["SPX"]
        if "QQQ" in returns:
            benchmark["QQQ"] = returns["QQQ"]
        if "SEMIS" in returns:
            benchmark["SMH"] = returns["SEMIS"]
        if {"QQQ", "SEMIS"}.issubset(returns.columns):
            benchmark["GROWTH_EQUAL"] = returns[["QQQ", "SEMIS"]].mean(axis=1)
        sleeves = [asset for asset in assets if asset != "CASH"]
        benchmark["EQUAL_WEIGHT"] = returns[sleeves].mean(axis=1)
        return benchmark
