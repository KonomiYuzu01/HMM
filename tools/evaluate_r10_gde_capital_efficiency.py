from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics


OUTPUT = Path("output/r10_gde_capital_efficiency")
NORMAL_DIRECTORY = (
    "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble"
)
PROXY_DIRECTORY = "experiment_r9_broad50_stage35_d10_20y_proxy"
NORMAL_OPEN_CLOSE = Path("data/adjusted_open_close_2011_present.csv")
PROXY_OPEN_CLOSE = Path("data/adjusted_open_close_20y_proxy.csv")
LIVE_GDE_OPEN_CLOSE = Path("data/retail_alternatives_open_close.csv")
ASSETS = [
    "SPX",
    "QQQ",
    "SEMIS",
    "BOND",
    "GOLD",
    "OIL",
    "USD",
    "CASH",
    "VIX_HEDGE",
]
ALL_ASSETS = [*ASSETS, "GDE"]

TargetPolicyResult: TypeAlias = tuple[
    pd.Series,
    bool,
    Mapping[str, float | int | bool | str],
]
TargetPolicy: TypeAlias = Callable[
    [pd.Timestamp, pd.Series, float, float],
    TargetPolicyResult,
]


def load_adjusted_open_close(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    opens = frame[[f"open_{asset}" for asset in ASSETS]].copy()
    closes = frame[[f"close_{asset}" for asset in ASSETS]].copy()
    opens.columns = ASSETS
    closes.columns = ASSETS
    return opens.astype(float), closes.astype(float)


def synthetic_gde_interval_return(asset_returns: pd.Series) -> float:
    """Financing-aware 90/90 stock-and-gold product return proxy."""
    return float(
        0.90 * asset_returns["SPX"]
        + 0.90 * asset_returns["GOLD"]
        - 0.80 * asset_returns["CASH"]
    )


def substitution_target(
    base_target: pd.Series,
    substitution_fraction: float,
    gate_mode: str,
    market_gate: bool = True,
    minimum_substitution_fraction: float = 0.0,
) -> tuple[pd.Series, float]:
    if not 0.0 <= substitution_fraction <= 1.0:
        raise ValueError("substitution_fraction must be in [0, 1]")
    if not 0.0 <= minimum_substitution_fraction <= substitution_fraction:
        raise ValueError(
            "minimum_substitution_fraction must be between zero and "
            "substitution_fraction"
        )
    growth_weight = float(
        base_target[["SPX", "QQQ", "SEMIS"]].sum()
    )
    growth_gate = float(np.clip(growth_weight / 0.80, 0.0, 1.0))
    if gate_mode == "always":
        gate = 1.0
    elif gate_mode == "growth_linked":
        gate = growth_gate
    elif gate_mode == "calm_broad_bull":
        gate = growth_gate if market_gate else 0.0
    elif gate_mode == "base_plus_calm":
        effective_fraction = (
            substitution_fraction
            if market_gate
            else minimum_substitution_fraction
        )
        gate = (
            growth_gate
            * effective_fraction
            / substitution_fraction
            if substitution_fraction > 0.0
            else 0.0
        )
    else:
        raise ValueError(f"Unknown GDE gate mode: {gate_mode}")
    gde_weight = (
        substitution_fraction
        * max(float(base_target["GOLD"]), 0.0)
        * gate
    )
    target = pd.Series(0.0, index=ALL_ASSETS)
    target.loc[ASSETS] = base_target.loc[ASSETS].astype(float)
    target["GOLD"] -= gde_weight
    target["GDE"] = gde_weight
    return target, gde_weight


def apply_gde_no_trade_band(
    target: pd.Series,
    current_gde_weight: float,
    no_trade_band: float,
) -> tuple[pd.Series, float]:
    if no_trade_band < 0.0:
        raise ValueError("gde_no_trade_band must be non-negative")
    desired_gde_weight = float(target["GDE"])
    implemented_gde_weight = desired_gde_weight + float(
        np.clip(
            current_gde_weight - desired_gde_weight,
            -no_trade_band,
            no_trade_band,
        )
    )
    adjusted = target.copy()
    adjusted["GOLD"] += desired_gde_weight - implemented_gde_weight
    adjusted["GDE"] = implemented_gde_weight
    return adjusted, implemented_gde_weight


def causal_risk_budgeted_substitution_fraction(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    maximum_fraction: float = 1.0,
    incremental_volatility_budget: float = 0.01,
    lookback_days: int = 60,
    annual_tracking_error: float = 0.091468,
) -> pd.Series:
    if not 0.0 <= maximum_fraction <= 1.0:
        raise ValueError("maximum_fraction must be in [0, 1]")
    if incremental_volatility_budget < 0.0:
        raise ValueError(
            "incremental_volatility_budget must be non-negative"
        )
    if lookback_days < 2:
        raise ValueError("lookback_days must be at least two")
    missing = [
        asset
        for asset in ASSETS
        if asset not in weights.columns or asset not in closes.columns
    ]
    if missing:
        raise ValueError(f"Missing risk-budget assets: {missing}")
    prior_returns = (
        closes[ASSETS]
        .pct_change(fill_method=None)
        .shift(1)
    )
    result = pd.Series(0.0, index=weights.index, dtype=float)
    for date in weights.index.intersection(prior_returns.index):
        history = prior_returns.loc[:date].tail(lookback_days)
        if len(history) < lookback_days or history.isna().any().any():
            continue
        covariance = (
            history.cov().to_numpy(dtype=float) * 252.0
        )
        base = weights.loc[date, ASSETS].to_numpy(dtype=float)
        base_variance = max(float(base @ covariance @ base), 0.0)
        base_volatility = float(np.sqrt(base_variance))
        base_target = weights.loc[date, ASSETS]
        growth_weight = float(
            base_target[["SPX", "QQQ", "SEMIS"]].sum()
        )
        growth_gate = float(
            np.clip(growth_weight / 0.80, 0.0, 1.0)
        )
        maximum_gde_weight = (
            maximum_fraction
            * max(float(base_target["GOLD"]), 0.0)
            * growth_gate
        )
        if maximum_gde_weight <= 0.0:
            continue

        def candidate_volatility(gde_weight: float) -> float:
            economic = base.copy()
            economic[ASSETS.index("SPX")] += 0.90 * gde_weight
            economic[ASSETS.index("GOLD")] -= 0.10 * gde_weight
            economic[ASSETS.index("CASH")] -= 0.80 * gde_weight
            market_variance = max(
                float(economic @ covariance @ economic),
                0.0,
            )
            tracking_variance = (
                gde_weight * annual_tracking_error
            ) ** 2
            return float(
                np.sqrt(market_variance + tracking_variance)
            )

        limit = base_volatility + incremental_volatility_budget
        if candidate_volatility(maximum_gde_weight) <= limit:
            selected_gde_weight = maximum_gde_weight
        else:
            lower = 0.0
            upper = maximum_gde_weight
            for _ in range(40):
                midpoint = 0.5 * (lower + upper)
                if candidate_volatility(midpoint) <= limit:
                    lower = midpoint
                else:
                    upper = midpoint
            selected_gde_weight = lower
        denominator = (
            max(float(base_target["GOLD"]), 0.0) * growth_gate
        )
        if denominator > 0.0:
            result.loc[date] = float(
                np.clip(
                    selected_gde_weight / denominator,
                    0.0,
                    maximum_fraction,
                )
            )
    return result.rename("risk_budgeted_substitution_fraction")


def causal_equity_excess_trend_fraction(
    closes: pd.DataFrame,
    *,
    maximum_fraction: float,
    lookback_days: int = 20,
) -> pd.Series:
    if not 0.0 <= maximum_fraction <= 1.0:
        raise ValueError("maximum_fraction must be in [0, 1]")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    required = ["SPX", "CASH"]
    missing = [asset for asset in required if asset not in closes]
    if missing:
        raise ValueError(f"Missing trend assets: {missing}")
    prior = closes[required].shift(1)
    stock_return = prior["SPX"] / prior["SPX"].shift(lookback_days) - 1.0
    cash_return = (
        prior["CASH"] / prior["CASH"].shift(lookback_days) - 1.0
    )
    active = stock_return.gt(cash_return)
    return (
        active.astype(float) * maximum_fraction
    ).rename("equity_excess_trend_fraction")


def causal_calm_broad_bull_signal(closes: pd.DataFrame) -> pd.Series:
    """Identify a calm broad-equity uptrend using only the prior close."""
    prior = closes[["SPX", "CASH"]].shift(1)
    slow_trend = (
        prior["SPX"]
        > prior["SPX"].rolling(200, min_periods=200).mean()
    )
    stock_return = prior["SPX"] / prior["SPX"].shift(63) - 1.0
    cash_return = prior["CASH"] / prior["CASH"].shift(63) - 1.0
    stock_volatility = (
        closes["SPX"]
        .pct_change(fill_method=None)
        .shift(1)
        .rolling(20, min_periods=20)
        .std(ddof=1)
        * np.sqrt(252.0)
    )
    return (
        slow_trend
        & stock_return.gt(cash_return)
        & stock_volatility.le(0.25)
    ).rename("calm_broad_bull")


def simulate_gde_substitution(
    strategy_directory: str,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    substitution_fraction: float,
    gate_mode: str,
    gde_return_mode: str,
    start_date: str,
    end_date: str | None,
    base_one_way_cost_bps: float = 7.5,
    gde_one_way_cost_bps: float = 30.0,
    financing_spread_bps: float = 100.0,
    minimum_substitution_fraction: float = 0.0,
    weights_override: pd.DataFrame | None = None,
    daily_override: pd.DataFrame | None = None,
    extra_slippage: pd.Series | None = None,
    gde_no_trade_band: float = 0.0,
    substitution_fraction_override: pd.Series | None = None,
    target_policy: TargetPolicy | None = None,
) -> pd.DataFrame:
    if (weights_override is None) != (daily_override is None):
        raise ValueError(
            "weights_override and daily_override must be supplied together"
        )
    if weights_override is None:
        source = Path("output") / strategy_directory
        weights = pd.read_csv(
            source / "weights.csv",
            index_col=0,
            parse_dates=True,
        )
        base_daily = pd.read_csv(
            source / "daily_returns.csv",
            index_col=0,
            parse_dates=True,
        )
    else:
        weights = weights_override.copy()
        base_daily = daily_override.copy()  # type: ignore[union-attr]
    dates = (
        weights.index.intersection(base_daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    dates = dates[dates >= start_date]
    if end_date is not None:
        dates = dates[dates <= end_date]
    if gde_return_mode == "live" and "GDE" not in opens.columns:
        raise ValueError("Live GDE mode requires GDE open and close prices")
    if gde_return_mode not in {"live", "synthetic"}:
        raise ValueError(f"Unknown GDE return mode: {gde_return_mode}")
    if gde_no_trade_band < 0.0:
        raise ValueError("gde_no_trade_band must be non-negative")
    calm_bull = causal_calm_broad_bull_signal(closes).reindex(dates)

    current = np.zeros(len(ALL_ASSETS), dtype=float)
    current[ALL_ASSETS.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    running_equity = 1.0
    running_peak = 1.0
    rows: list[dict[str, float | int | bool | str]] = []
    for date in dates:
        if previous_date is None:
            overnight_return = 0.0
        else:
            base_overnight = (
                opens.loc[date, ASSETS]
                / closes.loc[previous_date, ASSETS]
                - 1.0
            )
            gde_overnight = (
                float(
                    opens.loc[date, "GDE"]
                    / closes.loc[previous_date, "GDE"]
                    - 1.0
                )
                if gde_return_mode == "live"
                else synthetic_gde_interval_return(base_overnight)
            )
            overnight_asset_returns = np.concatenate(
                [base_overnight.to_numpy(dtype=float), [gde_overnight]]
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )

        policy_target = weights.loc[date, ASSETS].astype(float).copy()
        policy_force_trade = False
        policy_metadata: Mapping[str, float | int | bool | str] = {}
        if target_policy is not None:
            policy_target, policy_force_trade, policy_metadata = target_policy(
                date,
                policy_target,
                running_equity,
                running_peak,
            )
            if not policy_target.index.equals(pd.Index(ASSETS)):
                policy_target = policy_target.reindex(ASSETS)
            if policy_target.isna().any() or not np.isfinite(
                policy_target.to_numpy(dtype=float)
            ).all():
                raise ValueError("target_policy returned invalid target weights")
            if abs(float(policy_target.sum()) - 1.0) > 1e-10:
                raise ValueError("target_policy target weights must sum to one")
        base_trade = (
            float(base_daily.loc[date, "turnover"]) > 1e-14
            or policy_force_trade
        )
        trading_cost = 0.0
        current_gde_weight = float(current[ALL_ASSETS.index("GDE")])
        gde_target = current_gde_weight
        gde_desired_target = current_gde_weight
        gde_trade = 0.0
        effective_substitution_fraction = substitution_fraction
        if base_trade:
            if substitution_fraction_override is not None:
                effective_substitution_fraction = float(
                    substitution_fraction_override.reindex(
                        [date],
                        fill_value=0.0,
                    ).iloc[0]
                )
                if not 0.0 <= effective_substitution_fraction <= 1.0:
                    raise ValueError(
                        "substitution_fraction_override must be in [0, 1]"
                    )
            target, gde_desired_target = substitution_target(
                policy_target,
                effective_substitution_fraction,
                gate_mode,
                market_gate=bool(calm_bull.loc[date]),
                minimum_substitution_fraction=(
                    minimum_substitution_fraction
                ),
            )
            target, gde_target = apply_gde_no_trade_band(
                target,
                current_gde_weight,
                gde_no_trade_band,
            )
            target_values = target.to_numpy(dtype=float)
            absolute_trade = np.abs(target_values - current)
            gde_trade = float(
                absolute_trade[ALL_ASSETS.index("GDE")]
            )
            rates = np.full(
                len(ALL_ASSETS),
                base_one_way_cost_bps / 10_000.0,
            )
            rates[ALL_ASSETS.index("GDE")] = (
                gde_one_way_cost_bps / 10_000.0
            )
            trading_cost = float(absolute_trade @ rates)
            current = target_values

        base_intraday = (
            closes.loc[date, ASSETS] / opens.loc[date, ASSETS] - 1.0
        )
        gde_intraday = (
            float(
                closes.loc[date, "GDE"] / opens.loc[date, "GDE"] - 1.0
            )
            if gde_return_mode == "live"
            else synthetic_gde_interval_return(base_intraday)
        )
        intraday_asset_returns = np.concatenate(
            [base_intraday.to_numpy(dtype=float), [gde_intraday]]
        )
        intraday_return = float(current @ intraday_asset_returns)
        financing_cost = (
            max(-float(current[ALL_ASSETS.index("CASH")]), 0.0)
            * financing_spread_bps
            / 10_000.0
            / 252.0
        )
        slippage_cost = (
            0.0
            if extra_slippage is None or date not in extra_slippage.index
            else float(extra_slippage.loc[date])
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - financing_cost
            - slippage_cost
        )
        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth <= 0.0:
            raise RuntimeError("Non-positive intraday portfolio value")
        current = (
            current
            * (1.0 + intraday_asset_returns)
            / gross_intraday_growth
        )
        row: dict[str, float | int | bool | str] = {
                "date": date,
                "net_return": net_return,
                "overnight_return_before_trade": overnight_return,
                "intraday_return_after_trade": intraday_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "slippage_cost": slippage_cost,
                "base_trade": int(base_trade),
                "gde_target": gde_target,
                "gde_desired_target": gde_desired_target,
                "gde_trade": gde_trade,
                "gde_no_trade_band": gde_no_trade_band,
                "effective_substitution_fraction": (
                    effective_substitution_fraction
                ),
                "gross_notional": float(np.abs(current).sum()),
                "policy_force_trade": policy_force_trade,
                "policy_prior_equity": running_equity,
                "policy_prior_peak": running_peak,
                "policy_prior_drawdown": running_equity / running_peak - 1.0,
            }
        row.update(policy_metadata)
        rows.append(row)
        running_equity *= 1.0 + net_return
        running_peak = max(running_peak, running_equity)
        previous_date = date
    frame = pd.DataFrame(rows).set_index("date")
    frame["equity"] = (1.0 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def period_metrics(
    sample: str,
    source: str,
    fraction: float,
    gate_mode: str,
    gde_cost_bps: float,
    baseline: pd.Series,
    candidate: pd.Series,
    periods: dict[str, tuple[str, str]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for period, (start, end) in periods.items():
        base_values = performance_metrics(baseline.loc[start:end])
        candidate_values = performance_metrics(candidate.loc[start:end])
        rows.append(
            {
                "sample": sample,
                "source": source,
                "substitution_fraction": fraction,
                "gate_mode": gate_mode,
                "gde_one_way_cost_bps": gde_cost_bps,
                "period": period,
                **{
                    f"baseline_{key}": value
                    for key, value in base_values.items()
                },
                **{
                    f"candidate_{key}": value
                    for key, value in candidate_values.items()
                },
                "cagr_delta": (
                    candidate_values["cagr"] - base_values["cagr"]
                ),
                "sharpe_delta": (
                    candidate_values["sharpe"] - base_values["sharpe"]
                ),
                "max_drawdown_delta": (
                    candidate_values["max_drawdown"]
                    - base_values["max_drawdown"]
                ),
            }
        )
    return rows


def paired_block_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    block_days: int = 21,
    simulations: int = 10_000,
    seed: int = 20_260_727,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [baseline.rename("baseline"), candidate.rename("candidate")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    count = len(relative)
    blocks = int(np.ceil(count / block_days))
    offsets = np.arange(block_days)
    generator = np.random.default_rng(seed)
    estimates = np.empty(simulations)
    for simulation in range(simulations):
        starts = generator.integers(0, count, size=blocks)
        indices = (starts[:, None] + offsets).ravel()[:count] % count
        estimates[simulation] = float(relative[indices].mean() * 252.0)
    return {
        "observed_annualized_relative_log_return": float(
            relative.mean() * 252.0
        ),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "probability_positive": float(np.mean(estimates > 0.0)),
        "block_days": block_days,
        "simulations": simulations,
    }


def join_live_gde(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    live = pd.read_csv(
        LIVE_GDE_OPEN_CLOSE,
        index_col=0,
        parse_dates=True,
    )
    result_opens = opens.join(live[["open_GDE"]], how="inner")
    result_closes = closes.join(live[["close_GDE"]], how="inner")
    result_opens = result_opens.rename(columns={"open_GDE": "GDE"})
    result_closes = result_closes.rename(columns={"close_GDE": "GDE"})
    return result_opens, result_closes


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    live_opens, live_closes = join_live_gde(normal_opens, normal_closes)
    samples = {
        "normal_synthetic": {
            "directory": NORMAL_DIRECTORY,
            "opens": normal_opens,
            "closes": normal_closes,
            "mode": "synthetic",
            "start": "2015-01-01",
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": ("2015-01-01", "2026-12-31"),
            },
        },
        "proxy_synthetic": {
            "directory": PROXY_DIRECTORY,
            "opens": proxy_opens,
            "closes": proxy_closes,
            "mode": "synthetic",
            "start": "2006-08-01",
            "periods": {
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
                "complete_2006_2026": ("2006-08-01", "2026-12-31"),
            },
        },
        "normal_live": {
            "directory": NORMAL_DIRECTORY,
            "opens": live_opens,
            "closes": live_closes,
            "mode": "live",
            "start": "2022-03-17",
            "periods": {
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
            },
        },
    }
    rows: list[dict[str, float | str]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    for sample, specification in samples.items():
        directory = str(specification["directory"])
        opens = specification["opens"]
        closes = specification["closes"]
        mode = str(specification["mode"])
        start = str(specification["start"])
        periods = specification["periods"]
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)
        for gde_cost in (30.0, 50.0, 75.0):
            baseline = simulate_gde_substitution(
                directory,
                opens,
                closes,
                substitution_fraction=0.0,
                gate_mode="always",
                gde_return_mode=mode,
                start_date=start,
                end_date=None,
                gde_one_way_cost_bps=gde_cost,
            )
            for fraction in (0.25, 0.50, 0.75):
                for gate in (
                    "growth_linked",
                    "calm_broad_bull",
                    "always",
                ):
                    candidate = simulate_gde_substitution(
                        directory,
                        opens,
                        closes,
                        substitution_fraction=fraction,
                        gate_mode=gate,
                        gde_return_mode=mode,
                        start_date=start,
                        end_date=None,
                        gde_one_way_cost_bps=gde_cost,
                    )
                    common = baseline.index.intersection(candidate.index)
                    rows.extend(
                        period_metrics(
                            sample,
                            mode,
                            fraction,
                            gate,
                            gde_cost,
                            baseline.loc[common, "net_return"],
                            candidate.loc[common, "net_return"],
                            periods,
                        )
                    )
                    if (
                        fraction == 0.50
                        and gate in {"growth_linked", "calm_broad_bull"}
                        and gde_cost == 30.0
                    ):
                        path_label = (
                            "primary"
                            if gate == "growth_linked"
                            else "calm_broad_bull"
                        )
                        candidate.to_csv(
                            OUTPUT / f"{sample}_{path_label}_daily.csv",
                            index_label="date",
                        )
                        if gate == "growth_linked":
                            baseline.to_csv(
                                OUTPUT / f"{sample}_baseline_daily.csv",
                                index_label="date",
                            )
                        for block_days in (21, 63, 126):
                            bootstrap_rows.append(
                                {
                                    "sample": sample,
                                    "gate_mode": gate,
                                    **paired_block_bootstrap(
                                        baseline.loc[common, "net_return"],
                                        candidate.loc[common, "net_return"],
                                        block_days=block_days,
                                    ),
                                }
                            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT / "paired_bootstrap.csv",
        index=False,
    )
    primary = metrics[
        metrics["substitution_fraction"].eq(0.50)
        & metrics["gate_mode"].eq("growth_linked")
        & metrics["gde_one_way_cost_bps"].eq(30.0)
    ]
    print(
        primary[
            [
                "sample",
                "period",
                "baseline_cagr",
                "candidate_cagr",
                "cagr_delta",
                "baseline_sharpe",
                "candidate_sharpe",
                "baseline_max_drawdown",
                "candidate_max_drawdown",
            ]
        ].to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
