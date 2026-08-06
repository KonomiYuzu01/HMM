from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_guard
from regime_strategy.portfolio import apply_daily_growth_risk_reduction
from tools.evaluate_r10_gde_capital_efficiency import (
    simulate_gde_substitution,
)
from tools.evaluate_r11_diversified_capital_grid import metric_delta
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)
from tools.evaluate_r12_volatility_managed_risk import (
    BASE_MULTIPLIER,
    GDE_FRACTION,
    simulate_fixed_r11,
)
from tools.evaluate_r14_incremental_trend_permission import (
    trend_permission_schedule,
)
import tools.evaluate_r21_current_engine_industry_momentum as r21
from tools.evaluate_r21_current_engine_industry_momentum import (
    ACTIVE_MULTIPLIER,
    CASH_FLOOR,
    DEFINITION,
    MAX_TILT,
    REBALANCE_DAYS,
    apply_industry_momentum,
)


OUTPUT = Path("output/r22_post_tilt_growth_risk_cap")
CUMULATIVE_TRIALS = 91
GROWTH_RISK_CAP = 0.20


def apply_post_tilt_growth_risk_cap(
    weights: pd.DataFrame,
    execution_daily: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    target_volatility: float = GROWTH_RISK_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if target_volatility <= 0.0:
        raise ValueError("target_volatility must be positive")
    index = (
        weights.index.intersection(execution_daily.index)
        .intersection(closes.index)
    )
    assets = list(weights.columns)
    for required in ("QQQ", "SEMIS", "CASH"):
        if required not in assets:
            raise ValueError(f"Missing post-tilt risk asset: {required}")
    returns = closes[["QQQ", "SEMIS"]].pct_change(
        fill_method=None
    )
    rows: list[np.ndarray] = []
    diagnostics: list[dict[str, float | bool]] = []
    for date in index:
        original = weights.loc[date].to_numpy(dtype=float)
        history = returns.loc[returns.index < date].dropna(how="any")
        if len(history) < 61:
            adjusted = original.copy()
            realized_volatility = float("nan")
            multiplier = 1.0
        else:
            (
                adjusted,
                realized_volatility,
                multiplier,
            ) = apply_daily_growth_risk_reduction(
                original,
                history,
                assets,
                ["QQQ", "SEMIS"],
                assets.index("CASH"),
                20,
                60,
                target_volatility,
                252,
                "jump_aware",
                "stress_max",
            )
        before_growth = float(
            original[assets.index("QQQ")]
            + original[assets.index("SEMIS")]
        )
        after_growth = float(
            adjusted[assets.index("QQQ")]
            + adjusted[assets.index("SEMIS")]
        )
        released = before_growth - after_growth
        cash_increase = float(
            adjusted[assets.index("CASH")]
            - original[assets.index("CASH")]
        )
        if released < -1e-12:
            raise AssertionError("Post-tilt cap increased growth exposure")
        if abs(released - cash_increase) > 1e-10:
            raise AssertionError("Released growth did not flow to cash")
        rows.append(adjusted)
        diagnostics.append(
            {
                "post_tilt_growth_risk": realized_volatility,
                "post_tilt_growth_risk_multiplier": multiplier,
                "growth_before_post_tilt_cap": before_growth,
                "growth_after_post_tilt_cap": after_growth,
                "growth_released_to_cash": released,
                "post_tilt_cash_increase": cash_increase,
                "post_tilt_risk_cap_active": multiplier < 1.0 - 1e-12,
            }
        )
    capped = pd.DataFrame(rows, index=index, columns=assets)
    risk_diagnostics = pd.DataFrame(diagnostics, index=index)
    updated_daily = execution_daily.loc[index].copy()
    active = risk_diagnostics["post_tilt_risk_cap_active"]
    update = active | active.ne(active.shift(1))
    updated_daily.loc[update, "turnover"] = np.maximum(
        updated_daily.loc[update, "turnover"].to_numpy(dtype=float),
        1e-12,
    )
    risk_diagnostics["post_tilt_risk_execution_update"] = update
    return capped, updated_daily, risk_diagnostics


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    *,
    active_multiplier: float = ACTIVE_MULTIPLIER,
    cash_floor: float = CASH_FLOOR,
    rebalance_days: int = REBALANCE_DAYS,
    max_tilt: float = MAX_TILT,
    growth_risk_cap: float = GROWTH_RISK_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = settings["weights"]
    daily = settings["daily"]
    opens = settings["opens"]
    closes = settings["closes"]
    assert isinstance(weights, pd.DataFrame)
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(opens, pd.DataFrame)
    assert isinstance(closes, pd.DataFrame)
    scheduled, execution_daily, trend_diagnostics = (
        trend_permission_schedule(
            weights,
            daily,
            closes,
            DEFINITION,
            active_multiplier=active_multiplier,
            cash_floor=cash_floor,
        )
    )
    tilted, tilted_daily, momentum_diagnostics = (
        apply_industry_momentum(
            scheduled,
            execution_daily,
            closes,
            rebalance_days=rebalance_days,
            max_tilt=max_tilt,
        )
    )
    capped, capped_daily, risk_diagnostics = (
        apply_post_tilt_growth_risk_cap(
            tilted,
            tilted_daily,
            closes,
            target_volatility=growth_risk_cap,
        )
    )
    guard_daily, guard_weights = simulate_guard(
        capped,
        capped_daily,
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
    diagnostics = trend_diagnostics.reindex(trial.index).join(
        momentum_diagnostics.reindex(trial.index),
        how="left",
        rsuffix="_momentum",
    ).join(
        risk_diagnostics.reindex(trial.index),
        how="left",
    )
    diagnostics["post_guard_cash_weight"] = guard_weights.reindex(
        trial.index
    )["CASH"]
    diagnostics["post_guard_semis_weight"] = guard_weights.reindex(
        trial.index
    )["SEMIS"]
    diagnostics["guard_triggered"] = guard_daily.reindex(
        trial.index
    )["triggered"]
    return trial, diagnostics


def _append_risk_cap_neighborhoods() -> None:
    samples = r21._build_samples()
    settings = samples["normal_synthetic"]
    scenario = COST_SCENARIOS[0]
    baseline, _, _ = simulate_fixed_r11(settings, scenario)
    path = OUTPUT / "parameter_neighborhood.csv"
    neighborhood = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for label, cap in (("growth_risk19", 0.19), ("growth_risk21", 0.21)):
        trial, _ = simulate_candidate(
            settings,
            scenario,
            growth_risk_cap=cap,
        )
        delta = metric_delta(
            baseline["net_return"],
            trial["net_return"],
        )
        rows.append(
            {
                "dimension": "growth_risk_cap",
                "neighborhood": label,
                "active_multiplier": ACTIVE_MULTIPLIER,
                "rebalance_days": REBALANCE_DAYS,
                "max_tilt": MAX_TILT,
                **delta,
                "economic_pass": bool(
                    delta["candidate_cagr"] >= 0.25
                    and delta["candidate_max_drawdown"]
                    >= delta["baseline_max_drawdown"]
                ),
            }
        )
    neighborhood = pd.concat(
        [neighborhood, pd.DataFrame(rows)],
        ignore_index=True,
    )
    neighborhood.to_csv(path, index=False)


def _rewrite_acceptance() -> None:
    acceptance_path = OUTPUT / "acceptance.csv"
    acceptance = pd.read_csv(acceptance_path)
    metrics = pd.read_csv(OUTPUT / "metrics_by_period.csv")
    neighborhood = pd.read_csv(OUTPUT / "parameter_neighborhood.csv")
    diagnostics = pd.read_csv(
        OUTPUT / "normal_synthetic_diagnostics.csv"
    )
    required_dimensions = {
        "active_multiplier",
        "max_tilt",
        "growth_risk_cap",
    }
    selected = neighborhood.loc[
        neighborhood["dimension"].isin(required_dimensions)
    ]
    neighborhood_pass = bool(
        set(selected["dimension"]) == required_dimensions
        and selected.groupby("dimension")["economic_pass"].any().all()
    )
    only_reduces_pass = bool(
        diagnostics["growth_after_post_tilt_cap"]
        .le(
            diagnostics["growth_before_post_tilt_cap"] + 1e-12
        )
        .all()
        and np.allclose(
            diagnostics["growth_released_to_cash"],
            diagnostics["post_tilt_cash_increase"],
            atol=1e-10,
        )
    )

    def row(sample: str, scenario: str, period: str) -> pd.Series:
        chosen = metrics.loc[
            (metrics["sample"] == sample)
            & (metrics["scenario"] == scenario)
            & (metrics["period"] == period)
        ]
        if len(chosen) != 1:
            raise ValueError(f"Expected one {sample}/{scenario}/{period}")
        return chosen.iloc[0]

    complete = row(
        "normal_synthetic", "current_liquidity", "complete_2015_2026"
    )
    development = row(
        "normal_synthetic",
        "current_liquidity",
        "development_2015_2021",
    )
    holdout = row(
        "normal_synthetic",
        "current_liquidity",
        "holdout_2022_2025",
    )
    stress = row(
        "normal_synthetic", "cost_stress", "complete_2015_2026"
    )
    proxy_complete = row(
        "proxy_synthetic",
        "current_liquidity",
        "complete_2006_2026",
    )
    proxy_early = row(
        "proxy_synthetic",
        "current_liquidity",
        "early_2006_2014",
    )
    proxy_late = row(
        "proxy_synthetic",
        "current_liquidity",
        "late_2015_2026",
    )
    existing = acceptance.iloc[0]
    statistical_economic_pass = bool(
        complete["candidate_cagr"] >= 0.25
        and complete["candidate_max_drawdown"]
        >= complete["baseline_max_drawdown"]
        and development["annualized_relative_log_return"] > 0.0
        and development["candidate_max_drawdown"]
        >= development["baseline_max_drawdown"]
        and holdout["annualized_relative_log_return"] > 0.0
        and holdout["candidate_max_drawdown"]
        >= holdout["baseline_max_drawdown"]
        and stress["candidate_cagr"] >= 0.25
        and stress["candidate_max_drawdown"]
        >= stress["baseline_max_drawdown"]
        and proxy_complete["annualized_relative_log_return"] > 0.0
        and proxy_complete["candidate_max_drawdown"]
        >= proxy_complete["baseline_max_drawdown"]
        and proxy_early["annualized_relative_log_return"] > 0.0
        and proxy_late["annualized_relative_log_return"] > 0.0
        and bool(existing["family_multiple_testing_pass"])
        and bool(existing["leave_one_out_pass"])
        and neighborhood_pass
        and bool(existing["year_breadth_pass"])
        and bool(existing["growth_budget_conservation_pass"])
        and only_reduces_pass
    )
    acceptance["parameter_neighborhood_pass"] = neighborhood_pass
    acceptance["post_tilt_risk_only_reduces_pass"] = only_reduces_pass
    acceptance["statistical_economic_pass"] = statistical_economic_pass
    acceptance["production_pass"] = bool(
        statistical_economic_pass
        and bool(existing["declared_cash_hard_limit_pass"])
    )
    acceptance.to_csv(acceptance_path, index=False)
    print(acceptance.round(6).to_string(index=False))


def main() -> None:
    r21.OUTPUT = OUTPUT
    r21.CUMULATIVE_TRIALS = CUMULATIVE_TRIALS
    r21.simulate_candidate = simulate_candidate
    r21.main()
    _append_risk_cap_neighborhoods()
    _rewrite_acceptance()
    print(f"R22 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
