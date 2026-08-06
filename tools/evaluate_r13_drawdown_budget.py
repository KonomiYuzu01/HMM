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
    scale_non_cash_weights_by_series,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r11_post_shock_cooldown import sample_definitions
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    GDE_FRACTION,
    major_drawdown_events,
    robustness_rows,
    simulate_fixed_r11,
)


OUTPUT = Path("output/r13_drawdown_budget")
HIGH_MULTIPLIER = 1.10
CASH_FLOOR = -0.25
CUMULATIVE_TRIALS = 53
TRADING_EPSILON = 1e-14


@dataclass(frozen=True)
class DrawdownCandidate:
    name: str
    mode: str

    def relative_multiplier(
        self,
        drawdown: pd.Series,
        high_multiplier: float = HIGH_MULTIPLIER,
    ) -> pd.Series:
        if self.mode == "linear12":
            return (high_multiplier + 3.0 * drawdown).clip(
                lower=0.70,
                upper=high_multiplier,
            )
        if self.mode == "tier5_10":
            return pd.Series(
                np.select(
                    [drawdown.gt(-0.05), drawdown.gt(-0.10)],
                    [high_multiplier, 1.00],
                    default=0.75,
                ),
                index=drawdown.index,
                dtype=float,
            )
        if self.mode == "tier3_8":
            return pd.Series(
                np.select(
                    [drawdown.gt(-0.03), drawdown.gt(-0.08)],
                    [high_multiplier, 0.95],
                    default=0.70,
                ),
                index=drawdown.index,
                dtype=float,
            )
        raise ValueError(f"Unknown drawdown mode: {self.mode}")

    @property
    def has_discrete_tiers(self) -> bool:
        return self.mode.startswith("tier")


CANDIDATES = (
    DrawdownCandidate("linear12", "linear12"),
    DrawdownCandidate("tier5_10", "tier5_10"),
    DrawdownCandidate("tier3_8", "tier3_8"),
)


def cap_weights_at_cash_floor(
    weights: pd.DataFrame,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.Series]:
    if not -1.0 < cash_floor <= 0.0:
        raise ValueError("cash_floor must be in (-1, 0]")
    adjusted = weights.copy()
    non_cash = [column for column in adjusted if column != "CASH"]
    maximum_non_cash = 1.0 - cash_floor
    total = adjusted[non_cash].sum(axis=1)
    scale = (maximum_non_cash / total).clip(upper=1.0)
    scale = scale.where(total.gt(0.0), 1.0)
    adjusted.loc[:, non_cash] = adjusted.loc[:, non_cash].mul(
        scale,
        axis=0,
    )
    adjusted["CASH"] = 1.0 - adjusted[non_cash].sum(axis=1)
    return adjusted, scale


def drawdown_budget_schedule(
    baseline: pd.DataFrame,
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    definition: DrawdownCandidate,
    *,
    high_multiplier: float = HIGH_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = weights.index.intersection(base_daily.index)
    previous_drawdown = (
        baseline["drawdown"].reindex(index).shift(1).fillna(0.0)
    )
    requested_relative = definition.relative_multiplier(
        previous_drawdown,
        high_multiplier,
    )
    base_trade = (
        base_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    tier_change = (
        requested_relative.ne(requested_relative.shift(1))
        if definition.has_discrete_tiers
        else pd.Series(False, index=index)
    )
    update = base_trade | tier_change
    accepted_relative = (
        requested_relative.where(update)
        .ffill()
        .fillna(high_multiplier)
    )
    absolute_multiplier = BASE_MULTIPLIER * accepted_relative
    requested = scale_non_cash_weights_by_series(
        weights.loc[index],
        absolute_multiplier,
    )
    capped, cap_scale = cap_weights_at_cash_floor(
        requested,
        cash_floor,
    )
    execution_daily = base_daily.loc[index].copy()
    execution_daily.loc[update, "turnover"] = np.maximum(
        execution_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = pd.DataFrame(
        {
            "previous_r11_drawdown": previous_drawdown,
            "requested_relative_multiplier": requested_relative,
            "accepted_relative_multiplier": accepted_relative,
            "requested_absolute_multiplier": absolute_multiplier,
            "cash_cap_scale": cap_scale,
            "implemented_cash_weight": capped["CASH"],
            "base_trade": base_trade,
            "tier_change": tier_change,
            "execution_update": update,
        },
        index=index,
    )
    return capped, execution_daily, diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    baseline: pd.DataFrame,
    definition: DrawdownCandidate,
    *,
    high_multiplier: float = HIGH_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, diagnostics = (
        drawdown_budget_schedule(
            baseline,
            weights,
            daily,
            definition,
            high_multiplier=high_multiplier,
            cash_floor=cash_floor,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        scheduled,
        execution_daily,
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
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    return trial, guard_weights, diagnostics


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
    diagnostic_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    paths: dict[str, dict[str, pd.Series]] = {}
    baselines: dict[str, pd.Series] = {}

    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        sample_paths: dict[str, pd.Series] = {}
        for scenario in COST_SCENARIOS:
            baseline, _, _ = simulate_fixed_r11(settings, scenario)
            if scenario.name == "current_liquidity":
                baselines[sample] = baseline["net_return"]
                baseline.to_csv(
                    OUTPUT / f"{sample}_r11_baseline_daily.csv",
                    index_label="date",
                )
            for definition in CANDIDATES:
                trial, _, diagnostics = simulate_candidate(
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
                diagnostic_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "candidate": definition.name,
                        "mean_relative_multiplier": float(
                            diagnostics[
                                "accepted_relative_multiplier"
                            ].mean()
                        ),
                        "minimum_relative_multiplier": float(
                            diagnostics[
                                "accepted_relative_multiplier"
                            ].min()
                        ),
                        "maximum_relative_multiplier": float(
                            diagnostics[
                                "accepted_relative_multiplier"
                            ].max()
                        ),
                        "cash_cap_days": int(
                            diagnostics["cash_cap_scale"].lt(1.0).sum()
                        ),
                        "minimum_pre_guard_cash": float(
                            diagnostics["implemented_cash_weight"].min()
                        ),
                        "minimum_post_guard_cash": float(
                            diagnostics["post_guard_cash_weight"].min()
                        ),
                        "additional_update_days": int(
                            (
                                diagnostics["execution_update"]
                                & ~diagnostics["base_trade"]
                            ).sum()
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
    pd.DataFrame(diagnostic_rows).to_csv(
        OUTPUT / "risk_budget_diagnostics.csv",
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
    names = [definition.name for definition in CANDIDATES]
    for sample in ("normal_synthetic", "proxy_synthetic"):
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
    pd.DataFrame(family_rows).to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    neighborhood_rows: list[dict[str, object]] = []
    neighborhoods = (
        ("high108", 1.08, CASH_FLOOR),
        ("high112", 1.12, CASH_FLOOR),
        ("cash20", HIGH_MULTIPLIER, -0.20),
        ("cash30", HIGH_MULTIPLIER, -0.30),
    )
    settings = samples["normal_synthetic"]
    scenario = COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    for definition in CANDIDATES:
        for label, high, cash_floor in neighborhoods:
            trial, _, _ = simulate_candidate(
                settings,
                scenario,
                baseline,
                definition,
                high_multiplier=high,
                cash_floor=cash_floor,
            )
            delta = metric_delta(
                baseline["net_return"],
                trial["net_return"],
            )
            neighborhood_rows.append(
                {
                    "candidate": definition.name,
                    "neighborhood": label,
                    "high_multiplier": high,
                    "cash_floor": cash_floor,
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
