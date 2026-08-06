from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.relative_damage import (
    apply_relative_damage_veto,
    causal_relative_damage,
)
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
from tools.evaluate_r12_volatility_managed_risk import simulate_fixed_r11
from tools.evaluate_r38_crowding_fragility_cap import (
    EVENT_WINDOWS,
    _window_metrics,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r38_state_dependent_rollout as rollout


OUTPUT = Path("output/r39_relative_damage_concentration_veto")
PRODUCTION_OUTPUT = Path(
    "output/paper_core_growth_gold20_r38_convex_overlay"
)
CANDIDATE = "r39_relative_damage_concentration_veto"
R38_ROLLOUT_SHARE = 0.25
LOOKBACK_DAYS = 21
ENTRY_ACCOUNT_LOSS_BUDGET = 0.03
MINIMUM_SEMIS_GROWTH_SHARE = 0.60
MAXIMUM_ACTIVE_SEMIS_GROWTH_SHARE = 0.50
MAXIMUM_SHARE_ACTIVATION_MULTIPLE = 1.10
TRADING_EPSILON = 1e-14


def apply_relative_damage_concentration_veto(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    maximum_share_permission: pd.Series | None = None,
    *,
    lookback_days: int = LOOKBACK_DAYS,
    entry_account_loss_budget: float = ENTRY_ACCOUNT_LOSS_BUDGET,
    minimum_semis_growth_share: float = MINIMUM_SEMIS_GROWTH_SHARE,
    maximum_active_semis_growth_share: float = (
        MAXIMUM_ACTIVE_SEMIS_GROWTH_SHARE
    ),
    maximum_share_activation_multiple: float = (
        MAXIMUM_SHARE_ACTIVATION_MULTIPLE
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Veto a concentrated SMH target after material relative damage.

    The signal uses only closes known before the target's execution date.
    When the proposed absolute SMH weight times the trailing SMH-versus-QQQ
    loss exceeds the account budget, SMH is reduced to satisfy both the
    budget and the active growth-sleeve share ceiling. The removed SMH
    weight goes to QQQ, so total growth exposure is unchanged.
    """
    index = (
        weights.index.intersection(execution_daily.index)
        .intersection(closes.index)
    )
    damage = causal_relative_damage(
        closes,
        index,
        lookback_days=lookback_days,
    )
    if maximum_share_permission is None:
        maximum_share_permission = pd.Series(True, index=index)
    adjusted, diagnostics = apply_relative_damage_veto(
        weights.loc[index],
        damage["relative_loss"],
        maximum_share_permission,
        account_loss_budget=entry_account_loss_budget,
        minimum_semis_growth_share=minimum_semis_growth_share,
        maximum_active_semis_growth_share=(
            maximum_active_semis_growth_share
        ),
        maximum_share_activation_multiple=(
            maximum_share_activation_multiple
        ),
    )

    changed = (
        adjusted[["QQQ", "SEMIS"]]
        .sub(weights.loc[index, ["QQQ", "SEMIS"]])
        .abs()
        .max(axis=1)
        .gt(TRADING_EPSILON)
    )
    updated_daily = execution_daily.loc[index].copy()
    updated_daily.loc[changed, "turnover"] = np.maximum(
        updated_daily.loc[changed, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    diagnostics = damage[["prior_relative_log_return"]].join(
        diagnostics,
        how="left",
    )
    diagnostics["relative_damage_target_changed"] = changed
    return adjusted, updated_daily, diagnostics


def _staged_inputs(
    settings: dict[str, object],
    scenario: object,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, r11_weights, r11_daily = simulate_fixed_r11(settings, scenario)
    r38_weights, r38_daily, r38_diagnostics = rollout.build_r38_targets(
        settings,
        scenario,
    )
    index = (
        r11_weights.index.intersection(r11_daily.index)
        .intersection(r38_weights.index)
        .intersection(r38_daily.index)
    )
    staged_weights = (
        (1.0 - R38_ROLLOUT_SHARE) * r11_weights.reindex(index)
        + R38_ROLLOUT_SHARE * r38_weights.reindex(index)
    )
    r11_trade = (
        r11_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    r38_trade = (
        r38_daily.reindex(index)["turnover"].fillna(0.0)
        > TRADING_EPSILON
    )
    staged_daily = r11_daily.reindex(index).copy()
    staged_daily["turnover"] = np.where(
        r11_trade | r38_trade,
        1e-12,
        0.0,
    )
    staged_daily["slippage_cost"] = (
        (1.0 - R38_ROLLOUT_SHARE)
        * r11_daily.reindex(index)["slippage_cost"].fillna(0.0)
        + R38_ROLLOUT_SHARE
        * r38_daily.reindex(index)["slippage_cost"].fillna(0.0)
    )
    return staged_weights, staged_daily, r38_diagnostics.reindex(index)


def _simulate_account(
    settings: dict[str, object],
    scenario: object,
    weights: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    zero_share = pd.Series(0.0, index=weights.index, dtype=float)
    path, _ = rollout.simulate_blended_account(
        settings,
        scenario,
        weights,
        daily,
        weights,
        daily,
        zero_share,
    )
    return path


def simulate_baseline(
    settings: dict[str, object],
    scenario: object,
) -> pd.DataFrame:
    weights, daily, _ = _staged_inputs(settings, scenario)
    return _simulate_account(settings, scenario, weights, daily)


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    entry_account_loss_budget: float = ENTRY_ACCOUNT_LOSS_BUDGET,
    minimum_semis_growth_share: float = MINIMUM_SEMIS_GROWTH_SHARE,
    maximum_active_semis_growth_share: float = (
        MAXIMUM_ACTIVE_SEMIS_GROWTH_SHARE
    ),
    maximum_share_activation_multiple: float = (
        MAXIMUM_SHARE_ACTIVATION_MULTIPLE
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = settings["closes"]
    assert isinstance(closes, pd.DataFrame)
    weights, daily, r38_diagnostics = _staged_inputs(settings, scenario)
    adjusted, adjusted_daily, guard_diagnostics = (
        apply_relative_damage_concentration_veto(
            weights,
            daily,
            closes,
            maximum_share_permission=(
                ~r38_diagnostics[
                    "volatility_acceleration_block"
                ].fillna(False).astype(bool)
            ),
            entry_account_loss_budget=entry_account_loss_budget,
            minimum_semis_growth_share=minimum_semis_growth_share,
            maximum_active_semis_growth_share=(
                maximum_active_semis_growth_share
            ),
            maximum_share_activation_multiple=(
                maximum_share_activation_multiple
            ),
        )
    )
    path = _simulate_account(
        settings,
        scenario,
        adjusted,
        adjusted_daily,
    )
    diagnostics = guard_diagnostics.reindex(path.index).join(
        r38_diagnostics[
            [
                "effective_incremental_permission",
                "volatility_acceleration_block",
            ]
        ],
        how="left",
    )
    return path, diagnostics


def _current_target_diagnostics(
    settings: dict[str, object],
    scenario: object,
    **arguments: float,
) -> tuple[pd.Series, pd.Series]:
    closes = settings["closes"]
    assert isinstance(closes, pd.DataFrame)
    weights, daily, r38_diagnostics = _staged_inputs(settings, scenario)
    targets = pd.read_csv(
        PRODUCTION_OUTPUT / "next_target_weights.csv",
        index_col=0,
    )
    metadata = json.loads(
        (PRODUCTION_OUTPUT / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    next_date = pd.Timestamp(metadata["next_session"])
    production = targets["staged_account_center"].astype(float)
    next_weights = pd.Series(0.0, index=weights.columns, dtype=float)
    for asset in next_weights.index:
        if asset in production.index:
            next_weights.loc[asset] = production.loc[asset]
    if "GDE" not in next_weights.index and "GDE" in production.index:
        next_weights.loc["GOLD"] += production.loc["GDE"]
    if abs(float(next_weights.sum()) - 1.0) > 1e-12:
        raise AssertionError("Current staged target does not sum to one")

    extended_weights = weights.copy()
    extended_weights.loc[next_date] = next_weights
    extended_daily = daily.copy()
    next_daily = daily.iloc[-1].copy()
    next_daily.loc[:] = 0.0
    next_daily["turnover"] = 1.0
    extended_daily.loc[next_date] = next_daily
    extended_closes = closes.copy()
    extended_closes.loc[next_date] = closes.iloc[-1]
    maximum_share_permission = (
        ~r38_diagnostics[
            "volatility_acceleration_block"
        ].fillna(False).astype(bool)
    )
    maximum_share_permission.loc[next_date] = bool(
        maximum_share_permission.iloc[-1]
    )
    adjusted, _, diagnostics = apply_relative_damage_concentration_veto(
        extended_weights,
        extended_daily,
        extended_closes,
        maximum_share_permission=maximum_share_permission,
        **arguments,
    )
    return adjusted.loc[next_date], diagnostics.loc[next_date]


def _definitions() -> list[dict[str, float | str]]:
    return [
        {"label": "central"},
        {"label": "budget250", "entry_account_loss_budget": 0.025},
        {"label": "budget350", "entry_account_loss_budget": 0.035},
        {"label": "share55", "minimum_semis_growth_share": 0.55},
        {"label": "share65", "minimum_semis_growth_share": 0.65},
        {
            "label": "active_cap45",
            "maximum_active_semis_growth_share": 0.45,
        },
        {
            "label": "hard_trigger105",
            "maximum_share_activation_multiple": 1.05,
        },
        {
            "label": "hard_trigger115",
            "maximum_share_activation_multiple": 1.15,
        },
    ]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    metric_rows: list[dict[str, object]] = []
    current_paths: dict[str, dict[str, pd.Series]] = {}
    current_diagnostics: dict[str, pd.DataFrame] = {}
    for sample, settings in samples.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline = simulate_baseline(settings, scenario)
            trial, diagnostics = simulate_candidate(settings, scenario)
            common = baseline.index.intersection(trial.index)
            for period, (start, end) in periods.items():
                selected = common[
                    (common >= start) & (common <= end)
                ]
                metric_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "period": period,
                        **metric_delta(
                            baseline.loc[selected, "net_return"],
                            trial.loc[selected, "net_return"],
                        ),
                    }
                )
            if scenario.name == "current_liquidity":
                current_paths[sample] = {
                    "baseline": baseline["net_return"],
                    "candidate": trial["net_return"],
                }
                current_diagnostics[sample] = diagnostics
                trial.to_csv(
                    OUTPUT / f"{sample}_candidate_daily.csv",
                    index_label="date",
                )
                diagnostics.to_csv(
                    OUTPUT / f"{sample}_diagnostics.csv",
                    index_label="date",
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    normal = current_paths["normal_synthetic"]
    event_rows: list[dict[str, object]] = []
    for event, (start, end) in EVENT_WINDOWS.items():
        index = normal["candidate"].loc[start:end].index
        baseline = _window_metrics(normal["baseline"].reindex(index))
        candidate = _window_metrics(normal["candidate"].reindex(index))
        event_rows.append(
            {
                "event": event,
                "start": start,
                "end": end,
                "baseline_total_return": baseline["total_return"],
                "candidate_total_return": candidate["total_return"],
                "total_return_delta": (
                    candidate["total_return"] - baseline["total_return"]
                ),
                "baseline_max_drawdown": baseline["max_drawdown"],
                "candidate_max_drawdown": candidate["max_drawdown"],
                "max_drawdown_delta": (
                    candidate["max_drawdown"] - baseline["max_drawdown"]
                ),
            }
        )
    pd.DataFrame(event_rows).to_csv(
        OUTPUT / "event_windows.csv",
        index=False,
    )

    central_settings = samples["normal_synthetic"]
    central_scenario = COST_SCENARIOS[0]
    central_baseline = simulate_baseline(
        central_settings,
        central_scenario,
    )
    neighborhood_rows: list[dict[str, object]] = []
    for definition in _definitions():
        label = str(definition["label"])
        arguments = {
            key: float(value)
            for key, value in definition.items()
            if key != "label"
        }
        trial, diagnostics = simulate_candidate(
            central_settings,
            central_scenario,
            **arguments,
        )
        current_weights, current = _current_target_diagnostics(
            central_settings,
            central_scenario,
            **arguments,
        )
        common = central_baseline.index.intersection(trial.index)
        neighborhood_rows.append(
            {
                "neighborhood": label,
                **arguments,
                "active_days": int(
                    diagnostics["relative_damage_guard_active"].sum()
                ),
                "current_active": bool(
                    current["relative_damage_guard_active"]
                ),
                "current_maximum_share_active": bool(
                    current["maximum_share_guard_active"]
                ),
                "current_qqq_weight": float(current_weights["QQQ"]),
                "current_semis_weight": float(current_weights["SEMIS"]),
                "current_semis_growth_share": float(
                    current["implemented_semis_growth_share"]
                ),
                "current_maximum_active_semis_growth_share": float(
                    current["maximum_active_semis_growth_share"]
                ),
                "current_implemented_account_relative_loss": float(
                    current["implemented_account_relative_loss"]
                ),
                "maximum_growth_budget_error": float(
                    diagnostics["growth_budget_error"].max()
                ),
                **metric_delta(
                    central_baseline.loc[common, "net_return"],
                    trial.loc[common, "net_return"],
                ),
            }
        )
    neighborhoods = pd.DataFrame(neighborhood_rows)
    neighborhoods.to_csv(
        OUTPUT / "parameter_neighborhood.csv",
        index=False,
    )

    def row(sample: str, scenario: str, period: str) -> pd.Series:
        return metrics.loc[
            metrics["sample"].eq(sample)
            & metrics["scenario"].eq(scenario)
            & metrics["period"].eq(period)
        ].iloc[0]

    complete = row(
        "normal_synthetic",
        "current_liquidity",
        "complete_2015_2026",
    )
    holdout = row(
        "normal_synthetic",
        "current_liquidity",
        "holdout_2022_2025",
    )
    proxy = row(
        "proxy_synthetic",
        "current_liquidity",
        "complete_2006_2026",
    )
    stress = row(
        "normal_synthetic",
        "cost_stress",
        "complete_2015_2026",
    )
    central_current_weights, central_current = (
        _current_target_diagnostics(
            central_settings,
            central_scenario,
        )
    )
    diagnostics = current_diagnostics["normal_synthetic"]
    gates = {
        "current_damage_veto_active": bool(
            central_current["relative_damage_guard_active"]
        ),
        "current_loss_budget_enforced": bool(
            float(central_current["implemented_account_relative_loss"])
            <= ENTRY_ACCOUNT_LOSS_BUDGET + 1e-12
        ),
        "current_smh_not_above_qqq": bool(
            float(central_current_weights["SEMIS"])
            <= float(central_current_weights["QQQ"]) + 1e-12
        ),
        "current_maximum_share_guard_active": bool(
            central_current["maximum_share_guard_active"]
        ),
        "growth_budget_conserved": bool(
            diagnostics["growth_budget_error"].le(1e-12).all()
        ),
        "complete_cagr_drag_below_50bps": bool(
            complete["candidate_cagr"]
            >= complete["baseline_cagr"] - 0.005
        ),
        "complete_drawdown_within_5bps": bool(
            complete["candidate_max_drawdown"]
            >= complete["baseline_max_drawdown"] - 0.0005
        ),
        "holdout_cagr_drag_below_50bps": bool(
            holdout["candidate_cagr"]
            >= holdout["baseline_cagr"] - 0.005
        ),
        "holdout_drawdown_within_5bps": bool(
            holdout["candidate_max_drawdown"]
            >= holdout["baseline_max_drawdown"] - 0.0005
        ),
        "proxy_cagr_drag_below_50bps": bool(
            proxy["candidate_cagr"]
            >= proxy["baseline_cagr"] - 0.005
        ),
        "stress_drawdown_within_5bps": bool(
            stress["candidate_max_drawdown"]
            >= stress["baseline_max_drawdown"] - 0.0005
        ),
        "parameter_neighborhood_controls_current_damage": bool(
            neighborhoods["current_active"].all()
            and neighborhoods["current_maximum_share_active"].all()
            and (
                neighborhoods[
                    "current_implemented_account_relative_loss"
                ]
                <= neighborhoods["entry_account_loss_budget"]
                .fillna(ENTRY_ACCOUNT_LOSS_BUDGET)
                + 1e-12
            ).all()
            and (
                neighborhoods["current_semis_growth_share"]
                <= neighborhoods[
                    "maximum_active_semis_growth_share"
                ].fillna(MAXIMUM_ACTIVE_SEMIS_GROWTH_SHARE)
                + 1e-12
            ).all()
        ),
    }
    acceptance = pd.DataFrame(
        [{"gate": key, "passed": value} for key, value in gates.items()]
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "initial_research_pass",
        "passed": bool(all(gates.values())),
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    summary = {
        "candidate": CANDIDATE,
        "initial_research_pass": bool(all(gates.values())),
        "requirements": len(gates),
        "requirements_passed": int(sum(gates.values())),
        "complete_cagr_delta": float(
            complete["candidate_cagr"] - complete["baseline_cagr"]
        ),
        "complete_max_drawdown_delta": float(
            complete["candidate_max_drawdown"]
            - complete["baseline_max_drawdown"]
        ),
        "current_proposed_account_relative_loss": float(
            central_current["proposed_account_relative_loss"]
        ),
        "current_implemented_account_relative_loss": float(
            central_current["implemented_account_relative_loss"]
        ),
        "current_qqq_weight": float(central_current_weights["QQQ"]),
        "current_semis_weight": float(central_current_weights["SEMIS"]),
        "current_semis_growth_share": float(
            central_current["implemented_semis_growth_share"]
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(acceptance.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
