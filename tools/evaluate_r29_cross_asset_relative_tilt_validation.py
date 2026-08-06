from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_strategy.report import performance_metrics
from tools.build_r11_cross_asset_shock_panel import (
    ASSETS,
    OUTPUT as PANEL_PATH,
    validate_panel,
)
from tools.evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
)
from tools.evaluate_r26_pulse_relative_tilt_risk_veto import (
    causal_pulse_gated_industry_momentum,
)


OUTPUT = Path("output/r29_cross_asset_relative_tilt_validation")
CURRENT_COST_BPS = 7.5
STRESS_COST_BPS = 15.0
CUMULATIVE_TRIALS = 102
BLOCK_DAYS = (21, 63, 126)
BOOTSTRAP_SAMPLES = 10_000
MAXIMUM_DRAWDOWN_TOLERANCE = 0.005


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_and_audit_panel() -> tuple[pd.DataFrame, dict[str, object]]:
    metadata_path = PANEL_PATH.with_suffix(
        PANEL_PATH.suffix + ".metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    panel = pd.read_csv(PANEL_PATH, index_col=0, parse_dates=True)
    validate_panel(panel)
    if file_sha256(PANEL_PATH) != metadata["cache_sha256"]:
        raise ValueError("Cross-asset panel fingerprint does not match")
    if int(metadata["row_count"]) != len(panel):
        raise ValueError("Cross-asset panel row count does not match")
    if metadata["price_as_of"] != panel.index.max().date().isoformat():
        raise ValueError("Cross-asset panel last date does not match")
    return panel, metadata


def pair_diagnostics(panel: pd.DataFrame, asset: str) -> pd.DataFrame:
    closes = pd.DataFrame(
        {
            "QQQ": panel["close_QQQ"],
            "SEMIS": panel[f"close_{asset}"],
        },
        index=panel.index,
    )
    return causal_pulse_gated_industry_momentum(
        closes,
        panel.index,
    )


def first_ready_date(diagnostics: pd.DataFrame) -> pd.Timestamp:
    ready = (
        diagnostics["prior_relative_log_return"].notna()
        & diagnostics["prior_relative_volatility"].notna()
        & diagnostics["high_volatility_threshold"].notna()
        & diagnostics["pair_realized_volatility"].notna()
    )
    if not ready.any():
        raise ValueError("Pair never reaches fully causal signal readiness")
    return pd.Timestamp(ready[ready].index[0])


def maximum_true_run(values: pd.Series) -> int:
    state = values.fillna(False).astype(bool)
    if not state.any():
        return 0
    groups = state.ne(state.shift()).cumsum()
    return int(state.groupby(groups).sum().max())


def simulate_open_portfolio(
    opens: pd.DataFrame,
    target_asset_share: pd.Series,
    update: pd.Series,
    *,
    start: pd.Timestamp,
    cost_bps: float,
) -> pd.DataFrame:
    if cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative")
    required = {"QQQ", "ASSET"}
    if set(opens.columns) != required:
        raise ValueError(f"opens must contain exactly {sorted(required)}")
    index = (
        opens.index.intersection(target_asset_share.index)
        .intersection(update.index)
    )
    index = index[index >= start]
    if len(index) < 2:
        raise ValueError("At least two open observations are required")
    prices = opens.loc[index, ["QQQ", "ASSET"]].astype(float)
    targets = target_asset_share.reindex(index).astype(float)
    updates = update.reindex(index).fillna(False).astype(bool)
    if targets.isna().any():
        raise ValueError("Target shares must cover every simulation date")
    if not targets.between(0.0, 1.0).all():
        raise ValueError("Target shares must be between zero and one")

    cost_rate = cost_bps / 10_000.0
    holdings = np.zeros(2, dtype=float)
    previous_post_trade_equity = 1.0
    previous_prices: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    for position, date in enumerate(index):
        current_prices = prices.loc[date].to_numpy(dtype=float)
        if position == 0:
            pre_trade_values = np.zeros(2, dtype=float)
            pre_trade_equity = 1.0
        else:
            assert previous_prices is not None
            pre_trade_values = holdings * (
                current_prices / previous_prices
            )
            pre_trade_equity = float(pre_trade_values.sum())

        should_trade = position == 0 or bool(updates.loc[date])
        requested_asset_share = float(targets.loc[date])
        requested_weights = np.array(
            [1.0 - requested_asset_share, requested_asset_share],
            dtype=float,
        )
        if should_trade:
            if position == 0:
                turnover = 1.0
            else:
                drifted_weights = pre_trade_values / pre_trade_equity
                turnover = float(
                    np.abs(requested_weights - drifted_weights).sum()
                    / 2.0
                )
            transaction_cost = pre_trade_equity * turnover * cost_rate
            post_trade_equity = pre_trade_equity - transaction_cost
            holdings = post_trade_equity * requested_weights
        else:
            turnover = 0.0
            transaction_cost = 0.0
            post_trade_equity = pre_trade_equity
            holdings = pre_trade_values

        implemented_weights = holdings / post_trade_equity
        daily_return = (
            post_trade_equity / previous_post_trade_equity - 1.0
        )
        rows.append(
            {
                "net_return": daily_return,
                "equity": post_trade_equity,
                "trade": should_trade,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "target_asset_share": requested_asset_share,
                "implemented_qqq_weight": implemented_weights[0],
                "implemented_asset_weight": implemented_weights[1],
                "budget_error": abs(implemented_weights.sum() - 1.0),
            }
        )
        previous_post_trade_equity = post_trade_equity
        previous_prices = current_prices
    return pd.DataFrame(rows, index=index)


def simulate_pair(
    panel: pd.DataFrame,
    asset: str,
    diagnostics: pd.DataFrame,
    *,
    start: pd.Timestamp,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens = pd.DataFrame(
        {
            "QQQ": panel["open_QQQ"],
            "ASSET": panel[f"open_{asset}"],
        },
        index=panel.index,
    )
    update = (
        diagnostics["momentum_update"].astype(bool)
        | diagnostics["pulse_veto_state_change"].astype(bool)
    )
    candidate = simulate_open_portfolio(
        opens,
        diagnostics["semis_growth_share"],
        update,
        start=start,
        cost_bps=cost_bps,
    )
    baseline = simulate_open_portfolio(
        opens,
        pd.Series(0.50, index=panel.index),
        update,
        start=start,
        cost_bps=cost_bps,
    )
    return baseline, candidate


def period_metric_row(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    scenario: str,
    period: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> dict[str, object]:
    common = baseline.index.intersection(candidate.index)
    selected = common
    if start is not None:
        selected = selected[selected >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected <= pd.Timestamp(end)]
    base = baseline.loc[selected]
    trial = candidate.loc[selected]
    if len(selected) < 2:
        raise ValueError(f"Period {period} has insufficient observations")
    base_metrics = performance_metrics(base)
    candidate_metrics = performance_metrics(trial)
    relative = np.log1p(trial) - np.log1p(base)
    return {
        "scenario": scenario,
        "period": period,
        "observations": len(selected),
        "first_date": selected.min().date().isoformat(),
        "last_date": selected.max().date().isoformat(),
        "annualized_relative_log_return": float(
            relative.mean() * 252.0
        ),
        "baseline_cagr": base_metrics["cagr"],
        "candidate_cagr": candidate_metrics["cagr"],
        "baseline_max_drawdown": base_metrics["max_drawdown"],
        "candidate_max_drawdown": candidate_metrics["max_drawdown"],
        "max_drawdown_delta": (
            candidate_metrics["max_drawdown"]
            - base_metrics["max_drawdown"]
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel, metadata = load_and_audit_panel()
    diagnostics_by_asset = {
        asset: pair_diagnostics(panel, asset) for asset in ASSETS
    }
    common_start = max(
        first_ready_date(diagnostics)
        for diagnostics in diagnostics_by_asset.values()
    )

    scenarios = {
        "current_liquidity": CURRENT_COST_BPS,
        "cost_stress": STRESS_COST_BPS,
    }
    aggregate_paths: dict[str, dict[str, pd.Series]] = {}
    current_relative: dict[str, pd.Series] = {}
    asset_rows: list[dict[str, object]] = []
    execution_rows: list[pd.DataFrame] = []
    for scenario, cost_bps in scenarios.items():
        baselines: dict[str, pd.Series] = {}
        candidates: dict[str, pd.Series] = {}
        for asset in ASSETS:
            baseline, candidate = simulate_pair(
                panel,
                asset,
                diagnostics_by_asset[asset],
                start=common_start,
                cost_bps=cost_bps,
            )
            baselines[asset] = baseline["net_return"]
            candidates[asset] = candidate["net_return"]
            relative = (
                np.log1p(candidate["net_return"])
                - np.log1p(baseline["net_return"])
            )
            base_metrics = performance_metrics(baseline["net_return"])
            trial_metrics = performance_metrics(candidate["net_return"])
            asset_rows.append(
                {
                    "scenario": scenario,
                    "asset": asset,
                    "observations": len(relative),
                    "annualized_relative_log_return": float(
                        relative.mean() * 252.0
                    ),
                    "baseline_cagr": base_metrics["cagr"],
                    "candidate_cagr": trial_metrics["cagr"],
                    "baseline_max_drawdown": (
                        base_metrics["max_drawdown"]
                    ),
                    "candidate_max_drawdown": (
                        trial_metrics["max_drawdown"]
                    ),
                    "candidate_turnover": float(
                        candidate["turnover"].sum()
                    ),
                    "baseline_turnover": float(
                        baseline["turnover"].sum()
                    ),
                }
            )
            if scenario == "current_liquidity":
                current_relative[asset] = relative
                combined = candidate.add_prefix("candidate_").join(
                    baseline.add_prefix("baseline_")
                )
                combined["asset"] = asset
                execution_rows.append(
                    combined.reset_index(names="date")
                )

        baseline_panel = pd.DataFrame(baselines)
        candidate_panel = pd.DataFrame(candidates)
        aggregate_paths[scenario] = {
            "baseline": baseline_panel.mean(axis=1),
            "candidate": candidate_panel.mean(axis=1),
        }

    asset_metrics = pd.DataFrame(asset_rows)
    asset_metrics.to_csv(OUTPUT / "asset_metrics.csv", index=False)
    pd.concat(execution_rows, ignore_index=True).to_csv(
        OUTPUT / "current_execution_paths.csv",
        index=False,
    )

    periods = (
        ("development", common_start, "2021-12-31"),
        ("holdout", "2022-01-01", None),
        ("complete", common_start, None),
    )
    period_rows: list[dict[str, object]] = []
    for scenario, paths in aggregate_paths.items():
        for period, start, end in periods:
            period_rows.append(
                period_metric_row(
                    paths["baseline"],
                    paths["candidate"],
                    scenario=scenario,
                    period=period,
                    start=start,
                    end=end,
                )
            )
    period_metrics = pd.DataFrame(period_rows)
    period_metrics.to_csv(OUTPUT / "period_metrics.csv", index=False)

    relative_panel = pd.DataFrame(current_relative)
    relative_panel.to_csv(
        OUTPUT / "current_relative_log_return_panel.csv",
        index_label="date",
    )
    pooled_relative = relative_panel.mean(axis=1)
    annual_relative = pooled_relative.groupby(
        pooled_relative.index.year
    ).sum()
    annual_relative.rename(
        "cross_asset_mean_relative_log_return"
    ).to_csv(OUTPUT / "annual_relative_log_return.csv", index_label="year")

    leave_one_out_rows = []
    for asset in ASSETS:
        remaining = relative_panel.drop(columns=asset).mean(axis=1)
        leave_one_out_rows.append(
            {
                "excluded_asset": asset,
                "annualized_relative_log_return": float(
                    remaining.mean() * 252.0
                ),
            }
        )
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    leave_one_out.to_csv(
        OUTPUT / "leave_one_asset_out.csv",
        index=False,
    )

    bootstrap_rows = []
    for block_days in BLOCK_DAYS:
        result = circular_family_reality_check(
            pooled_relative.to_numpy(dtype=float)[:, None],
            0,
            block_days,
            samples=BOOTSTRAP_SAMPLES,
            seed=20_260_729,
        )
        bootstrap_rows.append(
            {
                **result,
                "cumulative_trials": CUMULATIVE_TRIALS,
                "cumulative_adjusted_p_value": min(
                    1.0,
                    float(
                        result[
                            "familywise_reality_check_p_value"
                        ]
                    )
                    * CUMULATIVE_TRIALS,
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUTPUT / "block_bootstrap.csv", index=False)

    diagnostics_rows = []
    for asset, diagnostics in diagnostics_by_asset.items():
        selected = diagnostics.loc[diagnostics.index >= common_start].copy()
        selected["asset"] = asset
        diagnostics_rows.append(selected.reset_index(names="date"))
    all_diagnostics = pd.concat(diagnostics_rows, ignore_index=True)
    all_diagnostics.to_csv(
        OUTPUT / "signal_diagnostics.csv",
        index=False,
    )

    def metric(scenario: str, period: str) -> pd.Series:
        selected = period_metrics.loc[
            period_metrics["scenario"].eq(scenario)
            & period_metrics["period"].eq(period)
        ]
        if len(selected) != 1:
            raise ValueError("Expected one registered period metric")
        return selected.iloc[0]

    current_complete = metric("current_liquidity", "complete")
    current_development = metric("current_liquidity", "development")
    current_holdout = metric("current_liquidity", "holdout")
    stress_complete = metric("cost_stress", "complete")
    current_assets = asset_metrics.loc[
        asset_metrics["scenario"].eq("current_liquidity")
    ]
    development_years = annual_relative.loc[2017:2021]
    holdout_years = annual_relative.loc[2022:2025]
    signal_budget_pass = bool(
        all_diagnostics["semis_growth_share"].between(0.0, 1.0).all()
        and (
            all_diagnostics["semis_growth_share"]
            + all_diagnostics["qqq_growth_share"]
            - 1.0
        )
        .abs()
        .le(1e-12)
        .all()
    )
    implementation_budget_pass = bool(
        pd.concat(execution_rows, ignore_index=True)
        .filter(regex=r"^candidate_budget_error$")
        .le(1e-12)
        .all()
        .all()
    )
    pulse_duration_pass = bool(
        all(
            maximum_true_run(
                diagnostics.loc[
                    diagnostics.index >= common_start,
                    "pulse_veto_active",
                ]
            )
            <= 4
            for diagnostics in diagnostics_by_asset.values()
        )
    )
    gates = {
        "panel_hash_verified": (
            file_sha256(PANEL_PATH) == metadata["cache_sha256"]
        ),
        "panel_no_missing_prices": int(panel.isna().sum().sum()) == 0,
        "panel_no_nonpositive_prices": int(
            panel.le(0.0).sum().sum()
        )
        == 0,
        "panel_no_duplicate_dates": int(
            panel.index.duplicated().sum()
        )
        == 0,
        "development_relative_positive": float(
            current_development["annualized_relative_log_return"]
        )
        > 0.0,
        "holdout_relative_positive": float(
            current_holdout["annualized_relative_log_return"]
        )
        > 0.0,
        "complete_relative_positive": float(
            current_complete["annualized_relative_log_return"]
        )
        > 0.0,
        "stress_complete_relative_positive": float(
            stress_complete["annualized_relative_log_return"]
        )
        > 0.0,
        "at_least_6_of_8_assets_positive": int(
            current_assets["annualized_relative_log_return"].gt(0.0).sum()
        )
        >= 6,
        "median_asset_relative_positive": float(
            current_assets["annualized_relative_log_return"].median()
        )
        > 0.0,
        "all_leave_one_asset_out_positive": bool(
            leave_one_out["annualized_relative_log_return"].gt(0.0).all()
        ),
        "development_year_breadth": int(
            development_years.gt(0.0).sum()
        )
        >= 3,
        "holdout_year_breadth": int(holdout_years.gt(0.0).sum()) >= 3,
        "drawdown_within_0_50pp": float(
            current_complete["candidate_max_drawdown"]
        )
        >= (
            float(current_complete["baseline_max_drawdown"])
            - MAXIMUM_DRAWDOWN_TOLERANCE
        ),
        "all_cumulative_adjusted_p_below_5pct": bool(
            bootstrap["cumulative_adjusted_p_value"].lt(0.05).all()
        ),
        "signal_budget_conservation": signal_budget_pass,
        "implementation_budget_conservation": (
            implementation_budget_pass
        ),
        "pulse_duration_at_most_four_sessions": pulse_duration_pass,
    }
    acceptance = pd.DataFrame(
        [{"gate": gate, "passed": bool(value)} for gate, value in gates.items()]
    )
    acceptance.loc[len(acceptance)] = {
        "gate": "all_gates_pass",
        "passed": bool(all(gates.values())),
    }
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    pd.Series(
        {
            "common_start": common_start.date().isoformat(),
            "last_date": panel.index.max().date().isoformat(),
            "panel_sha256": metadata["cache_sha256"],
            "complete_annualized_relative_log_return": (
                current_complete["annualized_relative_log_return"]
            ),
            "development_annualized_relative_log_return": (
                current_development["annualized_relative_log_return"]
            ),
            "holdout_annualized_relative_log_return": (
                current_holdout["annualized_relative_log_return"]
            ),
            "stress_complete_annualized_relative_log_return": (
                stress_complete["annualized_relative_log_return"]
            ),
            "baseline_complete_max_drawdown": (
                current_complete["baseline_max_drawdown"]
            ),
            "candidate_complete_max_drawdown": (
                current_complete["candidate_max_drawdown"]
            ),
            "positive_assets": int(
                current_assets[
                    "annualized_relative_log_return"
                ].gt(0.0).sum()
            ),
            "median_asset_annualized_relative_log_return": float(
                current_assets[
                    "annualized_relative_log_return"
                ].median()
            ),
            "cumulative_trials": CUMULATIVE_TRIALS,
        },
        name="value",
    ).to_csv(OUTPUT / "summary.csv")

    print("Period metrics:")
    print(period_metrics.round(6).to_string(index=False))
    print("\nCurrent-liquidity asset metrics:")
    print(
        current_assets[
            [
                "asset",
                "annualized_relative_log_return",
                "baseline_cagr",
                "candidate_cagr",
                "baseline_max_drawdown",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nBlock bootstrap:")
    print(bootstrap.round(6).to_string(index=False))
    print("\nAcceptance:")
    print(acceptance.to_string(index=False))
    print(f"\nR29 artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
