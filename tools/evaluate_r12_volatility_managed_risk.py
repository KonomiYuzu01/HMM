from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from tools.evaluate_r10_gde_capital_efficiency import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    load_strategy_inputs,
    metric_delta,
    relative_log_return,
    scale_non_cash_weights,
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r11_post_shock_cooldown import sample_definitions


OUTPUT = Path("output/r12_volatility_managed_risk")
BASE_MULTIPLIER = 1.065
TARGET_VOLATILITY = 0.18
MINIMUM_MULTIPLIER = 0.90
MAXIMUM_MULTIPLIER = 1.20
GDE_FRACTION = 0.10
CUMULATIVE_TRIALS = 50
TRADING_EPSILON = 1e-14


@dataclass(frozen=True)
class VolatilityCandidate:
    name: str
    short_window: int
    long_window: int | None = None

    @property
    def minimum_history(self) -> int:
        return self.long_window or self.short_window


CANDIDATES = (
    VolatilityCandidate("vol21", 21),
    VolatilityCandidate("vol63", 63),
    VolatilityCandidate("volmax21_63", 21, 63),
)


def causal_realized_volatility(
    returns: pd.Series,
    candidate: VolatilityCandidate,
) -> pd.Series:
    prior = returns.shift(1)
    short = (
        prior.rolling(
            candidate.short_window,
            min_periods=candidate.short_window,
        ).std(ddof=1)
        * np.sqrt(252.0)
    )
    if candidate.long_window is None:
        return short
    long = (
        prior.rolling(
            candidate.long_window,
            min_periods=candidate.long_window,
        ).std(ddof=1)
        * np.sqrt(252.0)
    )
    result = pd.concat([short, long], axis=1).max(
        axis=1,
        skipna=False,
    )
    return result


def accepted_risk_multipliers(
    baseline_returns: pd.Series,
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    candidate: VolatilityCandidate,
    *,
    target_volatility: float = TARGET_VOLATILITY,
    minimum_multiplier: float = MINIMUM_MULTIPLIER,
    maximum_multiplier: float = MAXIMUM_MULTIPLIER,
) -> tuple[pd.Series, pd.DataFrame]:
    if target_volatility <= 0.0:
        raise ValueError("target_volatility must be positive")
    if not 0.0 < minimum_multiplier <= maximum_multiplier:
        raise ValueError("multiplier bounds are invalid")
    index = weights.index.intersection(base_daily.index)
    realized = causal_realized_volatility(
        baseline_returns.reindex(index),
        candidate,
    )
    desired = (
        BASE_MULTIPLIER * target_volatility / realized
    ).clip(
        lower=minimum_multiplier,
        upper=maximum_multiplier,
    )
    trade_dates = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    accepted = desired.where(trade_dates).ffill().fillna(
        BASE_MULTIPLIER
    )
    diagnostics = pd.DataFrame(
        {
            "realized_volatility": realized,
            "desired_multiplier": desired,
            "base_trade": trade_dates,
            "accepted_multiplier": accepted,
        },
        index=index,
    )
    return accepted, diagnostics


def simulate_fixed_r11(
    settings: dict[str, object],
    scenario: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    fixed = scale_non_cash_weights(weights, BASE_MULTIPLIER)
    guard_daily, guard_weights = simulate_guard(
        fixed,
        daily,
        opens,
        closes,
        guard_for_multiplier(
            BASE_MULTIPLIER,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    result = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=(
            scenario.base_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        gde_one_way_cost_bps=(
            scenario.gde_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        financing_spread_bps=(
            scenario.financing_spread_bps  # type: ignore[attr-defined]
        ),
        weights_override=guard_weights,
        daily_override=guard_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    return result, guard_weights, guard_daily


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    baseline: pd.DataFrame,
    definition: VolatilityCandidate,
    *,
    target_volatility: float = TARGET_VOLATILITY,
    maximum_multiplier: float = MAXIMUM_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    multipliers, diagnostics = accepted_risk_multipliers(
        baseline["net_return"],
        weights,
        daily,
        definition,
        target_volatility=target_volatility,
        maximum_multiplier=maximum_multiplier,
    )
    dynamic = scale_non_cash_weights_by_series(
        weights.loc[multipliers.index],
        multipliers,
    )
    guard_daily, guard_weights = simulate_guard(
        dynamic,
        daily,
        opens,
        closes,
        guard_for_multiplier(
            BASE_MULTIPLIER,
            scenario.emergency_slippage_bps,  # type: ignore[attr-defined]
        ),
        cost_bps=scenario.base_one_way_cost_bps,  # type: ignore[attr-defined]
        start_date=str(settings["start"]),
    )
    trial = simulate_gde_substitution(
        str(settings["directory"]),
        opens,
        closes,
        substitution_fraction=GDE_FRACTION,
        gate_mode="growth_linked",
        gde_return_mode=str(settings["mode"]),
        start_date=str(settings["start"]),
        end_date=None,
        base_one_way_cost_bps=(
            scenario.base_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        gde_one_way_cost_bps=(
            scenario.gde_one_way_cost_bps  # type: ignore[attr-defined]
        ),
        financing_spread_bps=(
            scenario.financing_spread_bps  # type: ignore[attr-defined]
        ),
        weights_override=guard_weights,
        daily_override=guard_daily,
        extra_slippage=guard_daily["slippage_cost"],
        gde_no_trade_band=GDE_NO_TRADE_BAND,
    )
    diagnostics = diagnostics.reindex(trial.index)
    diagnostics["cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, guard_weights, diagnostics


def major_drawdown_events(
    returns: pd.Series,
    threshold: float = -0.10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    events: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_event = False
    start: pd.Timestamp | None = None
    minimum = 0.0
    for date, value in drawdown.items():
        if not in_event and value < 0.0:
            in_event = True
            start = date
            minimum = float(value)
        elif in_event:
            minimum = min(minimum, float(value))
        if in_event and value >= -1e-12:
            assert start is not None
            if minimum <= threshold:
                events.append((start, date))
            in_event = False
            start = None
            minimum = 0.0
    if in_event and start is not None and minimum <= threshold:
        events.append((start, drawdown.index[-1]))
    return events


def robustness_rows(
    sample: str,
    candidate: str,
    baseline: pd.Series,
    trial: pd.Series,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aligned = pd.concat(
        [baseline.rename("baseline"), trial.rename("candidate")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    )
    year_rows: list[dict[str, object]] = []
    for year in sorted(aligned.index.year.unique()):
        kept = relative.loc[relative.index.year != year]
        year_rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "removed_year": int(year),
                "annualized_relative_log_return": float(
                    kept.mean() * 252.0
                ),
            }
        )
    event_rows: list[dict[str, object]] = []
    for number, (start, end) in enumerate(
        major_drawdown_events(aligned["baseline"]),
        start=1,
    ):
        kept = relative.loc[
            (relative.index < start) | (relative.index > end)
        ]
        event_rows.append(
            {
                "sample": sample,
                "candidate": candidate,
                "event": number,
                "removed_start": start,
                "removed_end": end,
                "annualized_relative_log_return": float(
                    kept.mean() * 252.0
                ),
            }
        )
    return year_rows, event_rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(
        PROXY_OPEN_CLOSE
    )
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily = load_strategy_inputs(
        NORMAL_DIRECTORY
    )
    proxy_weights, proxy_daily = load_strategy_inputs(
        PROXY_DIRECTORY
    )
    samples = sample_definitions(
        normal_weights,
        normal_daily,
        proxy_weights,
        proxy_daily,
        normal_opens,
        normal_closes,
        proxy_opens,
        proxy_closes,
        live_opens,
        live_closes,
    )

    metric_rows: list[dict[str, object]] = []
    multiplier_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.Series]] = {}
    baselines: dict[str, pd.Series] = {}
    center_trials: dict[
        tuple[str, str, str], tuple[pd.DataFrame, pd.DataFrame]
    ] = {}

    for sample, settings in samples.items():
        sample_paths: dict[str, pd.Series] = {}
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            if scenario.name == "current_liquidity":
                baselines[sample] = baseline["net_return"]
                baseline.to_csv(
                    OUTPUT / f"{sample}_r11_baseline_daily.csv",
                    index_label="date",
                )
            for definition in CANDIDATES:
                trial, trial_weights, diagnostics = simulate_candidate(
                    settings,
                    scenario,
                    baseline,
                    definition,
                )
                common = baseline.index.intersection(trial.index)
                for period, (period_start, period_end) in periods.items():
                    selected = common[
                        (common >= period_start)
                        & (common <= period_end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "candidate": definition.name,
                            "period": period,
                            **metric_delta(
                                baseline.loc[selected, "net_return"],
                                trial.loc[selected, "net_return"],
                            ),
                        }
                    )
                multiplier_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "candidate": definition.name,
                        "observations": len(diagnostics),
                        "updates": int(
                            diagnostics["base_trade"].fillna(False).sum()
                        ),
                        "mean_multiplier": float(
                            diagnostics["accepted_multiplier"].mean()
                        ),
                        "minimum_multiplier": float(
                            diagnostics["accepted_multiplier"].min()
                        ),
                        "maximum_multiplier": float(
                            diagnostics["accepted_multiplier"].max()
                        ),
                        "minimum_cash_weight": float(
                            diagnostics["cash_weight"].min()
                        ),
                        "guard_triggers": int(
                            diagnostics["guard_triggered"]
                            .fillna(False)
                            .sum()
                        ),
                    }
                )
                if scenario.name == "current_liquidity":
                    sample_paths[definition.name] = trial[
                        "net_return"
                    ]
                    center_trials[
                        (sample, scenario.name, definition.name)
                    ] = (trial, diagnostics)
                    trial.to_csv(
                        OUTPUT
                        / f"{sample}_{definition.name}_daily.csv",
                        index_label="date",
                    )
                    diagnostics.to_csv(
                        OUTPUT
                        / f"{sample}_{definition.name}_diagnostics.csv",
                        index_label="date",
                    )
                    years, events = robustness_rows(
                        sample,
                        definition.name,
                        baseline["net_return"],
                        trial["net_return"],
                    )
                    year_rows.extend(years)
                    event_rows.extend(events)
        paths[sample] = sample_paths

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    pd.DataFrame(multiplier_rows).to_csv(
        OUTPUT / "multiplier_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(year_rows).to_csv(
        OUTPUT / "leave_one_year_out.csv",
        index=False,
    )
    pd.DataFrame(event_rows).to_csv(
        OUTPUT / "leave_one_drawdown_event_out.csv",
        index=False,
    )

    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        names = [definition.name for definition in CANDIDATES]
        matrix = np.column_stack(
            [
                relative_log_return(
                    baselines[sample],
                    paths[sample][name],
                )
                for name in names
            ]
        )
        for selected_index, name in enumerate(names):
            for block_days in (21, 63, 126):
                check = circular_family_reality_check(
                    matrix,
                    selected_index,
                    block_days,
                )
                family_p = float(
                    check["familywise_reality_check_p_value"]
                )
                family_rows.append(
                    {
                        "sample": sample,
                        "candidate": name,
                        "declared_family_size": len(names),
                        "cumulative_trials": CUMULATIVE_TRIALS,
                        **check,
                        "cumulative_trial_adjusted_p_value": min(
                            1.0,
                            family_p * CUMULATIVE_TRIALS / len(names),
                        ),
                    }
                )
    family = pd.DataFrame(family_rows)
    family.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    neighborhood_rows: list[dict[str, object]] = []
    neighborhood = (
        ("target17", 0.17, MAXIMUM_MULTIPLIER),
        ("target19", 0.19, MAXIMUM_MULTIPLIER),
        ("cap115", TARGET_VOLATILITY, 1.15),
        ("cap125", TARGET_VOLATILITY, 1.25),
    )
    settings = samples["normal_synthetic"]
    for scenario in COST_SCENARIOS[:1]:
        baseline, _, _ = simulate_fixed_r11(settings, scenario)
        for definition in CANDIDATES:
            for label, target, cap in neighborhood:
                trial, _, _ = simulate_candidate(
                    settings,
                    scenario,
                    baseline,
                    definition,
                    target_volatility=target,
                    maximum_multiplier=cap,
                )
                delta = metric_delta(
                    baseline["net_return"],
                    trial["net_return"],
                )
                neighborhood_rows.append(
                    {
                        "candidate": definition.name,
                        "neighborhood": label,
                        "target_volatility": target,
                        "maximum_multiplier": cap,
                        **delta,
                        "ridge_pass": bool(
                            delta["cagr_delta"] > 0.0
                            and delta["max_drawdown_delta"] >= 0.0
                        ),
                    }
                )
    pd.DataFrame(neighborhood_rows).to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    print(
        metrics.loc[
            metrics["period"].isin(
                {
                    "complete_2015_2026",
                    "complete_2006_2026",
                    "live_2022_2026",
                }
            ),
            [
                "sample",
                "scenario",
                "candidate",
                "candidate_cagr",
                "candidate_max_drawdown",
                "cagr_delta",
                "max_drawdown_delta",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
