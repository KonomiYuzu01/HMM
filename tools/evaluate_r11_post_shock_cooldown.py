from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_smh_dynamic_guard import simulate as simulate_smh_guard
from regime_strategy.report import performance_metrics
from tools.evaluate_r10_gde_capital_efficiency import (
    ASSETS,
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
    relative_log_return,
    scale_non_cash_weights,
)
from tools.evaluate_r11_levered_diversified_strategy import (
    COST_SCENARIOS,
    GDE_NO_TRADE_BAND,
    guard_for_multiplier,
)


OUTPUT = Path("output/r11_post_shock_cooldown")
RISK_MULTIPLIER = 1.065
GDE_FRACTION = 0.10
MINIMUM_PREVIOUS_GROWTH_WEIGHT = 0.50
GROWTH_ASSETS = ("SPX", "QQQ", "SEMIS")
ROLLOUT_SHARES = (0.25, 1.00)


@dataclass(frozen=True)
class CooldownCandidate:
    loss_trigger: float
    hold_sessions: int
    growth_cap: float

    @property
    def name(self) -> str:
        loss = int(round(abs(self.loss_trigger) * 100))
        cap = int(round(self.growth_cap * 100))
        return f"loss{loss}_hold{self.hold_sessions}_cap{cap}"


def candidate_family() -> tuple[CooldownCandidate, ...]:
    return tuple(
        CooldownCandidate(loss, hold, cap)
        for loss in (-0.02, -0.03, -0.04)
        for hold in (5, 10, 15)
        for cap in (0.20, 0.35, 0.50)
    )


def growth_weight(weights: pd.DataFrame) -> pd.Series:
    return weights.loc[:, list(GROWTH_ASSETS)].sum(axis=1)


def causal_growth_shock_signals(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
) -> pd.DataFrame:
    missing_weights = [
        asset for asset in GROWTH_ASSETS if asset not in weights
    ]
    missing_prices = [
        asset for asset in GROWTH_ASSETS if asset not in closes
    ]
    if missing_weights or missing_prices:
        raise ValueError(
            "Growth shock signals require growth weights and prices: "
            f"weights={missing_weights}, prices={missing_prices}"
        )
    aligned = weights.index.intersection(closes.index)
    prior_weights = weights.loc[aligned, list(GROWTH_ASSETS)]
    close_returns = (
        closes.loc[aligned, list(GROWTH_ASSETS)]
        .pct_change(fill_method=None)
    )
    realized_growth_return = (
        prior_weights * close_returns
    ).sum(axis=1, min_count=1)
    signals = pd.DataFrame(index=aligned)
    signals["previous_growth_weight"] = (
        prior_weights.sum(axis=1).shift(1)
    )
    signals["previous_weighted_growth_return"] = (
        realized_growth_return.shift(1)
    )
    return signals


def cap_growth_to_cash(
    weights: pd.Series,
    growth_cap: float,
) -> pd.Series:
    if not 0.0 <= growth_cap <= 1.0:
        raise ValueError("growth_cap must be in [0, 1]")
    adjusted = weights.loc[ASSETS].astype(float).copy()
    current_growth = float(adjusted.loc[list(GROWTH_ASSETS)].sum())
    if current_growth <= growth_cap + 1e-12:
        return adjusted
    scale = growth_cap / current_growth
    adjusted.loc[list(GROWTH_ASSETS)] *= scale
    adjusted["CASH"] += current_growth - growth_cap
    return adjusted


def apply_post_shock_cooldown(
    weights: pd.DataFrame,
    base_daily: pd.DataFrame,
    signals: pd.DataFrame,
    candidate: CooldownCandidate | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = weights.index.intersection(base_daily.index)
    adjusted_weights = weights.loc[dates, ASSETS].copy()
    adjusted_daily = base_daily.loc[dates].copy()
    aligned_signals = signals.reindex(dates)
    diagnostics: list[dict[str, float | int | bool]] = []
    remaining = 0
    overlay_had_effect = False

    for date in dates:
        signal = aligned_signals.loc[date]
        triggered = bool(
            candidate is not None
            and pd.notna(signal["previous_growth_weight"])
            and pd.notna(signal["previous_weighted_growth_return"])
            and float(signal["previous_growth_weight"])
            >= MINIMUM_PREVIOUS_GROWTH_WEIGHT
            and float(signal["previous_weighted_growth_return"])
            <= candidate.loss_trigger
        )
        if triggered:
            assert candidate is not None
            remaining = max(remaining, candidate.hold_sessions)

        active = bool(candidate is not None and remaining > 0)
        baseline_growth = float(
            adjusted_weights.loc[date, list(GROWTH_ASSETS)].sum()
        )
        capped_growth = baseline_growth
        changed = False
        released = False
        if active:
            assert candidate is not None
            capped = cap_growth_to_cash(
                adjusted_weights.loc[date],
                candidate.growth_cap,
            )
            changed = not np.allclose(
                capped.to_numpy(dtype=float),
                adjusted_weights.loc[date].to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
            )
            if changed:
                adjusted_weights.loc[date] = capped
                capped_growth = float(
                    capped.loc[list(GROWTH_ASSETS)].sum()
                )
                overlay_had_effect = True
                base_trade = bool(
                    float(base_daily.loc[date, "turnover"]) > 0.0
                )
                if triggered or base_trade:
                    adjusted_daily.loc[date, "turnover"] = max(
                        float(adjusted_daily.loc[date, "turnover"]),
                        1.0,
                    )
            remaining -= 1
        elif overlay_had_effect:
            adjusted_daily.loc[date, "turnover"] = max(
                float(adjusted_daily.loc[date, "turnover"]),
                1.0,
            )
            overlay_had_effect = False
            released = True

        diagnostics.append(
            {
                "previous_growth_weight": signal[
                    "previous_growth_weight"
                ],
                "previous_weighted_growth_return": signal[
                    "previous_weighted_growth_return"
                ],
                "triggered": triggered,
                "active": active,
                "changed": changed,
                "released": released,
                "remaining_sessions": remaining,
                "baseline_growth_weight": baseline_growth,
                "implemented_growth_weight": capped_growth,
            }
        )

    return (
        adjusted_weights,
        adjusted_daily,
        pd.DataFrame(diagnostics, index=dates),
    )


def metric_delta(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, float]:
    base = performance_metrics(baseline)
    trial = performance_metrics(candidate)
    return {
        **{f"baseline_{key}": value for key, value in base.items()},
        **{f"candidate_{key}": value for key, value in trial.items()},
        "cagr_delta": trial["cagr"] - base["cagr"],
        "sharpe_delta": trial["sharpe"] - base["sharpe"],
        "max_drawdown_delta": (
            trial["max_drawdown"] - base["max_drawdown"]
        ),
    }


def compounded_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def sample_definitions(
    normal_weights: pd.DataFrame,
    normal_daily: pd.DataFrame,
    proxy_weights: pd.DataFrame,
    proxy_daily: pd.DataFrame,
    normal_opens: pd.DataFrame,
    normal_closes: pd.DataFrame,
    proxy_opens: pd.DataFrame,
    proxy_closes: pd.DataFrame,
    live_opens: pd.DataFrame,
    live_closes: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    return {
        "normal_synthetic": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": normal_opens,
            "closes": normal_closes,
            "mode": "synthetic",
            "start": "2015-01-01",
            "periods": {
                "development_2015_2021": (
                    "2015-01-01",
                    "2021-12-31",
                ),
                "holdout_2022_2025": (
                    "2022-01-01",
                    "2025-12-31",
                ),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": (
                    "2015-01-01",
                    "2026-12-31",
                ),
            },
        },
        "proxy_synthetic": {
            "directory": PROXY_DIRECTORY,
            "weights": proxy_weights,
            "daily": proxy_daily,
            "opens": proxy_opens,
            "closes": proxy_closes,
            "mode": "synthetic",
            "start": "2006-08-01",
            "periods": {
                "early_2006_2014": (
                    "2006-08-01",
                    "2014-12-31",
                ),
                "late_2015_2026": (
                    "2015-01-01",
                    "2026-12-31",
                ),
                "complete_2006_2026": (
                    "2006-08-01",
                    "2026-12-31",
                ),
            },
        },
        "normal_live": {
            "directory": NORMAL_DIRECTORY,
            "weights": normal_weights,
            "daily": normal_daily,
            "opens": live_opens,
            "closes": live_closes,
            "mode": "live",
            "start": "2022-03-17",
            "periods": {
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
            },
        },
    }


def select_candidate(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    required = {
        ("normal_synthetic", "development_2015_2021"),
        ("proxy_synthetic", "early_2006_2014"),
    }
    selected = metrics.loc[
        (metrics["scenario"] == "current_liquidity")
        & metrics["rollout_share"].eq(1.0)
        & metrics[["sample", "period"]].apply(tuple, axis=1).isin(required)
    ]
    rows: list[dict[str, object]] = []
    for name, group in selected.groupby("candidate"):
        rows.append(
            {
                "candidate": name,
                "selection_periods": int(group.shape[0]),
                "minimum_cagr_delta": float(group["cagr_delta"].min()),
                "mean_cagr_delta": float(group["cagr_delta"].mean()),
                "minimum_sharpe_delta": float(
                    group["sharpe_delta"].min()
                ),
                "worst_max_drawdown_delta": float(
                    group["max_drawdown_delta"].min()
                ),
                "selection_gate": bool(
                    group.shape[0] == len(required)
                    and group["cagr_delta"].ge(0.0).all()
                    and group["sharpe_delta"].ge(0.0).all()
                    and group["max_drawdown_delta"].ge(-0.005).all()
                ),
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking.loc[ranking["selection_gate"]]
    pool = eligible if not eligible.empty else ranking
    chosen = str(
        pool.sort_values(
            [
                "minimum_cagr_delta",
                "mean_cagr_delta",
                "minimum_sharpe_delta",
                "worst_max_drawdown_delta",
                "candidate",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]["candidate"]
    )
    ranking["selected"] = ranking["candidate"].eq(chosen)
    return chosen, ranking.sort_values("candidate")


def candidate_definition(name: str) -> CooldownCandidate:
    matches = [
        candidate
        for candidate in candidate_family()
        if candidate.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Cooldown candidate is not unique: {name}")
    return matches[0]


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
    normal_weights, normal_daily = load_strategy_inputs(NORMAL_DIRECTORY)
    proxy_weights, proxy_daily = load_strategy_inputs(PROXY_DIRECTORY)
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
    candidates = candidate_family()
    metric_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    current_paths: dict[str, pd.Series] = {}
    candidate_paths: dict[str, dict[str, pd.Series]] = {}
    saved_payloads: dict[
        tuple[str, str, str],
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ] = {}

    for sample, settings in samples.items():
        weights = settings["weights"]
        daily = settings["daily"]
        opens = settings["opens"]
        closes = settings["closes"]
        periods = settings["periods"]
        assert isinstance(weights, pd.DataFrame)
        assert isinstance(daily, pd.DataFrame)
        assert isinstance(opens, pd.DataFrame)
        assert isinstance(closes, pd.DataFrame)
        assert isinstance(periods, dict)

        sample_candidate_paths: dict[str, pd.Series] = {}
        levered = scale_non_cash_weights(weights, RISK_MULTIPLIER)
        for scenario in COST_SCENARIOS:
            r9 = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=0.0,
                gate_mode="always",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
            )
            guard_daily, guard_weights = simulate_smh_guard(
                levered,
                daily,
                opens,
                closes,
                guard_for_multiplier(
                    RISK_MULTIPLIER,
                    scenario.emergency_slippage_bps,
                ),
                cost_bps=scenario.base_one_way_cost_bps,
                start_date=str(settings["start"]),
            )
            current = simulate_gde_substitution(
                str(settings["directory"]),
                opens,
                closes,
                substitution_fraction=GDE_FRACTION,
                gate_mode="growth_linked",
                gde_return_mode=str(settings["mode"]),
                start_date=str(settings["start"]),
                end_date=None,
                base_one_way_cost_bps=scenario.base_one_way_cost_bps,
                gde_one_way_cost_bps=scenario.gde_one_way_cost_bps,
                financing_spread_bps=scenario.financing_spread_bps,
                weights_override=guard_weights,
                daily_override=guard_daily,
                extra_slippage=guard_daily["slippage_cost"],
                gde_no_trade_band=GDE_NO_TRADE_BAND,
            )
            signals = causal_growth_shock_signals(
                guard_weights,
                closes,
            )
            if scenario.name == "current_liquidity":
                current_paths[sample] = current["net_return"]

            for candidate in candidates:
                (
                    cooldown_weights,
                    cooldown_daily,
                    diagnostics,
                ) = apply_post_shock_cooldown(
                    guard_weights,
                    guard_daily,
                    signals,
                    candidate,
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
                        scenario.base_one_way_cost_bps
                    ),
                    gde_one_way_cost_bps=(
                        scenario.gde_one_way_cost_bps
                    ),
                    financing_spread_bps=(
                        scenario.financing_spread_bps
                    ),
                    weights_override=cooldown_weights,
                    daily_override=cooldown_daily,
                    extra_slippage=cooldown_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
                )
                common = (
                    r9.index.intersection(current.index)
                    .intersection(trial.index)
                )
                for rollout_share in ROLLOUT_SHARES:
                    current_staged = (
                        (1.0 - rollout_share)
                        * r9.loc[common, "net_return"]
                        + rollout_share
                        * current.loc[common, "net_return"]
                    )
                    trial_staged = (
                        (1.0 - rollout_share)
                        * r9.loc[common, "net_return"]
                        + rollout_share
                        * trial.loc[common, "net_return"]
                    )
                    for period, (
                        period_start,
                        period_end,
                    ) in periods.items():
                        selected = common[
                            (common >= period_start)
                            & (common <= period_end)
                        ]
                        metric_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "candidate": candidate.name,
                                "loss_trigger": candidate.loss_trigger,
                                "hold_sessions": candidate.hold_sessions,
                                "growth_cap": candidate.growth_cap,
                                "rollout_share": rollout_share,
                                "period": period,
                                **metric_delta(
                                    current_staged.loc[selected],
                                    trial_staged.loc[selected],
                                ),
                            }
                        )
                    for year in (
                        2008,
                        2011,
                        2015,
                        2018,
                        2020,
                        2022,
                        2025,
                        2026,
                    ):
                        selected = common[common.year == year]
                        if selected.empty:
                            continue
                        event_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "candidate": candidate.name,
                                "rollout_share": rollout_share,
                                "year": year,
                                "current_return": compounded_return(
                                    current_staged.loc[selected]
                                ),
                                "candidate_return": compounded_return(
                                    trial_staged.loc[selected]
                                ),
                                "relative_log_return": float(
                                    (
                                        np.log1p(
                                            trial_staged.loc[selected]
                                        )
                                        - np.log1p(
                                            current_staged.loc[selected]
                                        )
                                    ).sum()
                                ),
                            }
                        )
                if scenario.name == "current_liquidity":
                    sample_candidate_paths[candidate.name] = trial[
                        "net_return"
                    ]
                    saved_payloads[
                        (sample, scenario.name, candidate.name)
                    ] = (trial, cooldown_weights, diagnostics)
        candidate_paths[sample] = sample_candidate_paths

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    events.to_csv(OUTPUT / "event_metrics.csv", index=False)
    selected_name, selection = select_candidate(metrics)
    selection.to_csv(OUTPUT / "selection.csv", index=False)

    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        baseline = current_paths[sample]
        names = list(candidate_paths[sample])
        matrix = np.column_stack(
            [
                relative_log_return(
                    baseline,
                    candidate_paths[sample][name],
                )
                for name in names
            ]
        )
        selected_index = names.index(selected_name)
        for block_days in (21, 63, 126):
            family_rows.append(
                {
                    "sample": sample,
                    "candidate": selected_name,
                    "declared_family_size": len(names),
                    "block_days": block_days,
                    **circular_family_reality_check(
                        matrix,
                        selected_index,
                        block_days,
                    ),
                }
            )
    family = pd.DataFrame(family_rows)
    family.to_csv(
        OUTPUT / "declared_family_reality_check.csv",
        index=False,
    )

    for sample in samples:
        payload = saved_payloads[
            (sample, "current_liquidity", selected_name)
        ]
        payload[0].to_csv(
            OUTPUT / f"{sample}_{selected_name}_daily.csv",
            index_label="date",
        )
        payload[1].to_csv(
            OUTPUT / f"{sample}_{selected_name}_weights.csv",
            index_label="date",
        )
        payload[2].to_csv(
            OUTPUT / f"{sample}_{selected_name}_diagnostics.csv",
            index_label="date",
        )

    print(f"Selected candidate: {selected_name}")
    print(
        metrics.loc[
            metrics["candidate"].eq(selected_name)
            & metrics["rollout_share"].eq(1.0)
            & metrics["period"].isin(
                [
                    "complete_2015_2026",
                    "complete_2006_2026",
                    "live_2022_2026",
                ]
            ),
            [
                "sample",
                "scenario",
                "period",
                "cagr_delta",
                "sharpe_delta",
                "max_drawdown_delta",
                "candidate_max_drawdown",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
