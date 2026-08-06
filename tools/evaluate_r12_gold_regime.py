from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from regime_strategy.report import performance_metrics
from evaluate_r10_combined_tail_capital import selected_guard
from evaluate_r10_gde_capital_efficiency import (
    NORMAL_DIRECTORY,
    NORMAL_OPEN_CLOSE,
    PROXY_DIRECTORY,
    PROXY_OPEN_CLOSE,
    join_live_gde,
    load_adjusted_open_close,
    simulate_gde_substitution,
)
from evaluate_r11_diversified_capital_grid import (
    circular_family_reality_check,
    load_strategy_inputs,
    relative_log_return,
    scale_non_cash_weights,
)
from evaluate_smh_dynamic_guard import simulate as simulate_guard


OUTPUT = Path("output/r12_gold_regime_v2")
MACRO_CACHE = Path("data/gold_macro_fred.csv")
R11_PATHS = Path("output/r11_confirmatory_financing_mix")
RISK_MULTIPLIER = 1.065
GDE_FRACTION = 0.10
GDE_NO_TRADE_BAND = 0.02
RISK_ON_GROWTH_THRESHOLD = 0.45
HORIZONS = (63, 126, 252)
FLOOR_FRACTIONS = (0.25, 0.50, 0.75)
MAPPING_MODES = ("graded", "majority", "veto")
HIERARCHICAL_MAPPING_MODES = ("trend_tiered", "trend_confirmed")


@dataclass(frozen=True)
class CostScenario:
    name: str
    base_one_way_cost_bps: float
    emergency_slippage_bps: float
    gde_one_way_cost_bps: float
    financing_spread_bps: float


COST_SCENARIOS = (
    CostScenario("current_liquidity", 7.5, 20.0, 40.0, 100.0),
    CostScenario("cost_stress", 15.0, 50.0, 75.0, 150.0),
)


@dataclass(frozen=True)
class GoldRegimeCandidate:
    horizon_days: int
    floor_fraction: float
    mapping_mode: str

    @property
    def name(self) -> str:
        floor = int(round(self.floor_fraction * 100))
        return f"h{self.horizon_days}_floor{floor}_{self.mapping_mode}"


def candidate_family() -> tuple[GoldRegimeCandidate, ...]:
    return tuple(
        GoldRegimeCandidate(horizon, floor, mode)
        for horizon in HORIZONS
        for floor in FLOOR_FRACTIONS
        for mode in MAPPING_MODES + HIERARCHICAL_MAPPING_MODES
    )


def load_macro_cache(path: Path = MACRO_CACHE) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    required = ["DFII10", "DTWEXBGS", "T10YIE"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Gold macro cache lacks columns: {missing}")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Gold macro cache dates must be unique and increasing")
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    if numeric[["DFII10", "DTWEXBGS"]].dropna().empty:
        raise ValueError("Gold macro cache has no usable joint observations")
    return numeric


def causal_gold_signals(
    closes: pd.DataFrame,
    macro: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    if horizon_days < 2:
        raise ValueError("horizon_days must be at least two")
    required = ["GOLD", "CASH"]
    missing = [asset for asset in required if asset not in closes]
    if missing:
        raise ValueError(f"Gold signal prices lack assets: {missing}")

    prior_closes = closes[required].shift(1)
    gold_excess_trend = (
        np.log(prior_closes["GOLD"] / prior_closes["CASH"])
        .diff(horizon_days)
    )
    known_macro = macro.reindex(closes.index).ffill().shift(1)
    real_yield_change = known_macro["DFII10"].diff(horizon_days)
    dollar_change = np.log(known_macro["DTWEXBGS"]).diff(horizon_days)

    result = pd.DataFrame(index=closes.index)
    result["gold_excess_trend"] = gold_excess_trend
    result["real_yield_change"] = real_yield_change
    result["dollar_change"] = dollar_change
    result["gold_support"] = gold_excess_trend.gt(0.0).astype(float)
    result["real_yield_support"] = real_yield_change.lt(0.0).astype(float)
    result["dollar_support"] = dollar_change.lt(0.0).astype(float)
    result["support_votes"] = result[
        ["gold_support", "real_yield_support", "dollar_support"]
    ].sum(axis=1)
    available = result[
        ["gold_excess_trend", "real_yield_change", "dollar_change"]
    ].notna().all(axis=1)
    result.loc[~available, "support_votes"] = np.nan
    return result


def gold_multiplier(
    support_votes: pd.Series,
    floor_fraction: float,
    mapping_mode: str,
) -> pd.Series:
    if not 0.0 <= floor_fraction <= 1.0:
        raise ValueError("floor_fraction must be in [0, 1]")
    if mapping_mode == "graded":
        multiplier = floor_fraction + (
            1.0 - floor_fraction
        ) * support_votes / 3.0
    elif mapping_mode == "majority":
        multiplier = pd.Series(
            np.where(support_votes >= 2.0, 1.0, floor_fraction),
            index=support_votes.index,
            dtype=float,
        )
    elif mapping_mode == "veto":
        multiplier = pd.Series(
            np.where(support_votes <= 0.0, floor_fraction, 1.0),
            index=support_votes.index,
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown gold mapping mode: {mapping_mode}")
    return multiplier.where(support_votes.notna(), 1.0).clip(0.0, 1.0)


def candidate_gold_multiplier(
    signals: pd.DataFrame,
    candidate: GoldRegimeCandidate,
) -> pd.Series:
    if candidate.mapping_mode in MAPPING_MODES:
        return gold_multiplier(
            signals["support_votes"],
            candidate.floor_fraction,
            candidate.mapping_mode,
        )
    required = [
        "gold_support",
        "real_yield_support",
        "dollar_support",
    ]
    missing = [column for column in required if column not in signals]
    if missing:
        raise ValueError(f"Hierarchical gold signals lack columns: {missing}")
    available = signals["support_votes"].notna()
    trend_adverse = signals["gold_support"].lt(0.5)
    macro_headwinds = (
        2.0
        - signals["real_yield_support"]
        - signals["dollar_support"]
    )
    if candidate.mapping_mode == "trend_tiered":
        mild_fraction = 0.5 * (1.0 + candidate.floor_fraction)
        values = np.where(
            ~trend_adverse,
            1.0,
            np.where(
                macro_headwinds >= 2.0,
                candidate.floor_fraction,
                np.where(macro_headwinds >= 1.0, mild_fraction, 1.0),
            ),
        )
    elif candidate.mapping_mode == "trend_confirmed":
        values = np.where(
            trend_adverse & macro_headwinds.ge(1.0),
            candidate.floor_fraction,
            1.0,
        )
    else:
        raise ValueError(
            f"Unknown hierarchical mapping mode: {candidate.mapping_mode}"
        )
    return pd.Series(
        values,
        index=signals.index,
        dtype=float,
    ).where(available, 1.0).clip(0.0, 1.0)


def apply_gold_regime(
    weights: pd.DataFrame,
    signals: pd.DataFrame,
    candidate: GoldRegimeCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["SPX", "QQQ", "SEMIS", "GOLD", "CASH"]
    missing = [asset for asset in required if asset not in weights]
    if missing:
        raise ValueError(f"Gold regime weights lack assets: {missing}")
    aligned_signals = signals.reindex(weights.index)
    multiplier = candidate_gold_multiplier(aligned_signals, candidate)
    growth_weight = weights[["SPX", "QQQ", "SEMIS"]].sum(axis=1)
    risk_on = growth_weight.ge(RISK_ON_GROWTH_THRESHOLD)
    effective_multiplier = multiplier.where(risk_on, 1.0)

    adjusted = weights.copy()
    original_gold = adjusted["GOLD"].copy()
    adjusted["GOLD"] = original_gold * effective_multiplier
    gold_reduction = original_gold - adjusted["GOLD"]
    adjusted["CASH"] += gold_reduction
    if (adjusted["GOLD"] < -1e-12).any():
        raise ValueError("Gold regime produced a negative gold weight")
    if not np.allclose(
        adjusted.sum(axis=1).to_numpy(dtype=float),
        weights.sum(axis=1).to_numpy(dtype=float),
        atol=1e-10,
    ):
        raise ValueError("Gold regime must preserve total account weight")

    diagnostics = aligned_signals.copy()
    diagnostics["risk_on"] = risk_on.astype(int)
    diagnostics["base_gold_weight"] = original_gold
    diagnostics["gold_multiplier"] = effective_multiplier
    diagnostics["candidate_gold_weight"] = adjusted["GOLD"]
    diagnostics["gold_to_cash"] = gold_reduction
    return adjusted, diagnostics


def guard_for_scenario(scenario: CostScenario):
    return selected_guard(
        scenario.emergency_slippage_bps,
        post_trigger_cap=0.15,
        relative_gap_trigger=-0.005,
        minimum_semis_weight=0.60 * RISK_MULTIPLIER,
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


def gold_drawdown_episodes(
    gold_prices: pd.Series,
    threshold: float = -0.15,
) -> list[dict[str, object]]:
    prices = gold_prices.dropna()
    if prices.empty:
        return []
    peak = float(prices.iloc[0])
    peak_date = prices.index[0]
    trough = peak
    trough_date = peak_date
    episodes: list[dict[str, object]] = []
    for date, value in prices.iloc[1:].items():
        value = float(value)
        if value >= peak:
            drawdown = trough / peak - 1.0
            if drawdown <= threshold:
                episodes.append(
                    {
                        "peak_date": peak_date,
                        "trough_date": trough_date,
                        "recovery_or_end_date": date,
                        "gold_drawdown": drawdown,
                    }
                )
            peak = value
            peak_date = date
            trough = value
            trough_date = date
        elif value < trough:
            trough = value
            trough_date = date
    drawdown = trough / peak - 1.0
    if drawdown <= threshold:
        episodes.append(
            {
                "peak_date": peak_date,
                "trough_date": trough_date,
                "recovery_or_end_date": prices.index[-1],
                "gold_drawdown": drawdown,
            }
        )
    return episodes


def select_candidate(metrics: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    selection_periods = {
        ("proxy_synthetic", "early_2007_2014"),
        ("normal_synthetic", "development_2015_2021"),
    }
    selected = metrics.loc[
        (metrics["scenario"] == "current_liquidity")
        & metrics["mapping_mode"].isin(HIERARCHICAL_MAPPING_MODES)
        & metrics[["sample", "period"]]
        .apply(tuple, axis=1)
        .isin(selection_periods)
    ].copy()
    rows: list[dict[str, object]] = []
    for name, group in selected.groupby("candidate"):
        rows.append(
            {
                "candidate": name,
                "selection_periods": int(group.shape[0]),
                "minimum_cagr_delta": float(group["cagr_delta"].min()),
                "mean_cagr_delta": float(group["cagr_delta"].mean()),
                "minimum_sharpe_delta": float(group["sharpe_delta"].min()),
                "worst_max_drawdown_delta": float(
                    group["max_drawdown_delta"].min()
                ),
                "selection_gate": bool(
                    group.shape[0] == 2
                    and group["cagr_delta"].gt(0.0).all()
                    and group["sharpe_delta"].ge(0.0).all()
                    and group["max_drawdown_delta"].ge(-0.005).all()
                ),
            }
        )
    ranking = pd.DataFrame(rows)
    eligible = ranking.loc[ranking["selection_gate"]].copy()
    pool = eligible if not eligible.empty else ranking
    pool = pool.sort_values(
        [
            "minimum_cagr_delta",
            "mean_cagr_delta",
            "minimum_sharpe_delta",
            "worst_max_drawdown_delta",
            "candidate",
        ],
        ascending=[False, False, False, False, True],
    )
    chosen = str(pool.iloc[0]["candidate"])
    ranking["selected"] = ranking["candidate"].eq(chosen)
    return chosen, ranking.sort_values("candidate")


def selected_candidate_definition(name: str) -> GoldRegimeCandidate:
    matches = [candidate for candidate in candidate_family() if candidate.name == name]
    if len(matches) != 1:
        raise ValueError(f"Selected gold candidate is not unique: {name}")
    return matches[0]


def sample_definitions(
    normal_opens: pd.DataFrame,
    normal_closes: pd.DataFrame,
    proxy_opens: pd.DataFrame,
    proxy_closes: pd.DataFrame,
    live_opens: pd.DataFrame,
    live_closes: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    normal_weights, normal_daily = load_strategy_inputs(NORMAL_DIRECTORY)
    proxy_weights, proxy_daily = load_strategy_inputs(PROXY_DIRECTORY)
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
                "development_2015_2021": ("2015-01-01", "2021-12-31"),
                "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
                "recent_2026": ("2026-01-01", "2026-12-31"),
                "complete_2015_2026": ("2015-01-01", "2026-12-31"),
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
                "early_2007_2014": ("2007-01-01", "2014-12-31"),
                "middle_2015_2021": ("2015-01-01", "2021-12-31"),
                "holdout_2022_2026": ("2022-01-01", "2026-12-31"),
                "complete_2007_2026": ("2007-01-01", "2026-12-31"),
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


def evaluate_production_gate(
    selected_name: str,
    selection: pd.DataFrame,
    metrics: pd.DataFrame,
    family: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    checks: list[dict[str, object]] = []

    selected_selection = selection.loc[
        selection["candidate"] == selected_name
    ].iloc[0]
    checks.append(
        {
            "requirement": "selection_periods",
            "pass": bool(selected_selection["selection_gate"]),
            "detail": (
                f"minimum CAGR delta "
                f"{selected_selection['minimum_cagr_delta']:.4%}; "
                f"minimum Sharpe delta "
                f"{selected_selection['minimum_sharpe_delta']:.4f}"
            ),
        }
    )

    required_holdouts = [
        ("normal_synthetic", "current_liquidity", "holdout_2022_2025"),
        ("normal_synthetic", "current_liquidity", "recent_2026"),
        ("proxy_synthetic", "current_liquidity", "holdout_2022_2026"),
        ("normal_live", "current_liquidity", "live_2022_2026"),
    ]
    for sample, scenario, period in required_holdouts:
        row = metrics.loc[
            (metrics["candidate"] == selected_name)
            & (metrics["sample"] == sample)
            & (metrics["scenario"] == scenario)
            & (metrics["period"] == period)
        ].iloc[0]
        passed = bool(
            row["cagr_delta"] > 0.0
            and row["sharpe_delta"] >= 0.0
            and row["max_drawdown_delta"] >= -0.005
        )
        checks.append(
            {
                "requirement": f"{sample}:{period}",
                "pass": passed,
                "detail": (
                    f"CAGR {row['cagr_delta']:+.4%}; "
                    f"Sharpe {row['sharpe_delta']:+.4f}; "
                    f"MDD delta {row['max_drawdown_delta']:+.4%}"
                ),
            }
        )

    stress_periods = [
        ("normal_synthetic", "complete_2015_2026"),
        ("proxy_synthetic", "complete_2007_2026"),
        ("normal_live", "live_2022_2026"),
    ]
    for sample, period in stress_periods:
        row = metrics.loc[
            (metrics["candidate"] == selected_name)
            & (metrics["sample"] == sample)
            & (metrics["scenario"] == "cost_stress")
            & (metrics["period"] == period)
        ].iloc[0]
        passed = bool(
            row["cagr_delta"] > 0.0
            and row["sharpe_delta"] >= 0.0
            and row["candidate_max_drawdown"] >= -0.18
        )
        checks.append(
            {
                "requirement": f"{sample}:cost_stress",
                "pass": passed,
                "detail": (
                    f"CAGR {row['cagr_delta']:+.4%}; "
                    f"Sharpe {row['sharpe_delta']:+.4f}; "
                    f"MDD {row['candidate_max_drawdown']:.4%}"
                ),
            }
        )

    for sample in ("normal_synthetic", "proxy_synthetic"):
        rows = family.loc[family["sample"] == sample]
        passed = bool(
            rows.shape[0] == 3
            and rows["familywise_reality_check_p_value"].lt(0.05).all()
        )
        checks.append(
            {
                "requirement": f"{sample}:family_reality_check",
                "pass": passed,
                "detail": (
                    "p="
                    + ",".join(
                        f"{value:.4f}"
                        for value in rows[
                            "familywise_reality_check_p_value"
                        ]
                    )
                ),
            }
        )

    material_events = events.loc[
        events["gold_drawdown"].le(-0.20)
        & events["peak_date"].ge("2011-01-01")
    ]
    event_pass = bool(
        not material_events.empty
        and material_events["candidate_minus_baseline_return"].mean() > 0.0
        and material_events[
            "candidate_minus_baseline_return"
        ].ge(-0.005).all()
    )
    checks.append(
        {
            "requirement": "gold_bear_events",
            "pass": event_pass,
            "detail": (
                f"events={material_events.shape[0]}; "
                f"mean relative return "
                f"{material_events['candidate_minus_baseline_return'].mean():+.4%}"
            ),
        }
    )

    audit = pd.DataFrame(checks)
    return bool(audit["pass"].all()), audit


def write_decision_report(
    selected: GoldRegimeCandidate,
    qualified: bool,
    audit: pd.DataFrame,
    metrics: pd.DataFrame,
    events: pd.DataFrame,
    current_target: dict[str, object],
) -> None:
    status = "通过 R12 生产门槛" if qualified else "未通过 R12 生产门槛"
    lines = [
        "# R12 黄金独立市场环境实验",
        "",
        f"结论：候选 `{selected.name}` **{status}**。",
        "",
        "## 候选规则",
        "",
        (
            f"- 观察窗口：{selected.horizon_days} 个交易日；"
            f"黄金最低保留原目标的 {selected.floor_fraction:.0%}；"
            f"映射方式：`{selected.mapping_mode}`。"
        ),
        "- 三个因果信号：黄金相对现金趋势、10年实际利率变化、广义美元指数变化。",
        "- 每个信号至少滞后一日；只在成长仓至少45%时覆盖固定黄金部分。",
        "- 股票防守状态保留原多资产模型；减少的黄金只进入现金。",
        "",
        "## 生产门槛",
        "",
        "| 检查 | 结果 | 说明 |",
        "|---|---:|---|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['requirement']} | "
            f"{'通过' if row['pass'] else '失败'} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            "## 当前目标影响",
            "",
            f"- 数据日期：{current_target['date']}。",
            f"- 三个支持信号：{current_target['support_votes']} / 3。",
            f"- R11 黄金目标：{current_target['base_gold_weight']:.2%}。",
            f"- 候选黄金目标：{current_target['candidate_gold_weight']:.2%}。",
            f"- 转入现金：{current_target['gold_to_cash']:.2%}。",
            "",
            "## 黄金熊市事件",
            "",
            "| 高点 | 低点 | 黄金跌幅 | 候选相对 R11 |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in events.iterrows():
        lines.append(
            f"| {row['peak_date']} | {row['trough_date']} | "
            f"{row['gold_drawdown']:.2%} | "
            f"{row['candidate_minus_baseline_return']:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            (
                "本实验只改变黄金袖套，不重新选择成长资产，也不把黄金减仓资金"
                "自动转入 QQQ/SMH。若未通过全部门槛，R11 继续保持生产版本，"
                "候选只保留为研究结果。"
            ),
            "",
        ]
    )
    (OUTPUT / "decision.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    macro = load_macro_cache()
    normal_opens, normal_closes = load_adjusted_open_close(NORMAL_OPEN_CLOSE)
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    live_opens, live_closes = join_live_gde(normal_opens, normal_closes)
    samples = sample_definitions(
        normal_opens,
        normal_closes,
        proxy_opens,
        proxy_closes,
        live_opens,
        live_closes,
    )
    family_candidates = candidate_family()

    metric_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    paths: dict[tuple[str, str, str], pd.Series] = {}
    baseline_paths: dict[tuple[str, str], pd.Series] = {}
    diagnostics_by_sample: dict[
        tuple[str, str, str], pd.DataFrame
    ] = {}
    reconstruction_rows: list[dict[str, object]] = []

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

        signals_by_horizon = {
            horizon: causal_gold_signals(closes, macro, horizon)
            for horizon in HORIZONS
        }
        levered_weights = scale_non_cash_weights(weights, RISK_MULTIPLIER)
        for scenario in COST_SCENARIOS:
            guard_daily, guard_weights = simulate_guard(
                levered_weights,
                daily,
                opens,
                closes,
                guard_for_scenario(scenario),
                cost_bps=scenario.base_one_way_cost_bps,
                start_date=str(settings["start"]),
            )
            baseline_frame = simulate_gde_substitution(
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
            baseline = baseline_frame["net_return"]
            baseline_paths[(sample, scenario.name)] = baseline

            saved_path = (
                R11_PATHS / f"{sample}_risk1.065_gde10_daily.csv"
            )
            if scenario.name == "current_liquidity" and saved_path.exists():
                saved = pd.read_csv(
                    saved_path,
                    index_col=0,
                    parse_dates=True,
                )["net_return"]
                aligned = pd.concat(
                    [baseline.rename("rebuilt"), saved.rename("saved")],
                    axis=1,
                    join="inner",
                ).dropna()
                reconstruction_rows.append(
                    {
                        "sample": sample,
                        "observations": int(aligned.shape[0]),
                        "maximum_absolute_error": float(
                            (aligned["rebuilt"] - aligned["saved"])
                            .abs()
                            .max()
                        ),
                    }
                )

            for candidate in family_candidates:
                adjusted_weights, diagnostics = apply_gold_regime(
                    guard_weights,
                    signals_by_horizon[candidate.horizon_days],
                    candidate,
                )
                candidate_frame = simulate_gde_substitution(
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
                    weights_override=adjusted_weights,
                    daily_override=guard_daily,
                    extra_slippage=guard_daily["slippage_cost"],
                    gde_no_trade_band=GDE_NO_TRADE_BAND,
                )
                candidate_return = candidate_frame["net_return"]
                paths[(sample, scenario.name, candidate.name)] = candidate_return
                diagnostics_by_sample[
                    (sample, scenario.name, candidate.name)
                ] = diagnostics
                common = baseline.index.intersection(candidate_return.index)
                for period, (start, end) in periods.items():
                    selected_dates = common[
                        (common >= start) & (common <= end)
                    ]
                    metric_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario.name,
                            "candidate": candidate.name,
                            "horizon_days": candidate.horizon_days,
                            "floor_fraction": candidate.floor_fraction,
                            "mapping_mode": candidate.mapping_mode,
                            "period": period,
                            **metric_delta(
                                baseline.loc[selected_dates],
                                candidate_return.loc[selected_dates],
                            ),
                        }
                    )
                used = diagnostics.reindex(candidate_return.index)
                weight_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "candidate": candidate.name,
                        "average_base_gold": float(
                            used["base_gold_weight"].mean()
                        ),
                        "average_candidate_gold": float(
                            used["candidate_gold_weight"].mean()
                        ),
                        "average_gold_to_cash": float(
                            used["gold_to_cash"].mean()
                        ),
                        "fraction_reduced": float(
                            used["gold_to_cash"].gt(1e-10).mean()
                        ),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    weights_summary = pd.DataFrame(weight_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    selected_name, selection = select_candidate(metrics)
    selected = selected_candidate_definition(selected_name)

    family_rows: list[dict[str, object]] = []
    for sample in ("normal_synthetic", "proxy_synthetic"):
        baseline = baseline_paths[(sample, "current_liquidity")]
        matrix = np.column_stack(
            [
                relative_log_return(
                    baseline,
                    paths[(sample, "current_liquidity", candidate.name)],
                )
                for candidate in family_candidates
            ]
        )
        selected_index = [
            candidate.name for candidate in family_candidates
        ].index(selected_name)
        for block_days in (21, 63, 126):
            family_rows.append(
                {
                    "sample": sample,
                    "candidate": selected_name,
                    "declared_family_size": len(family_candidates),
                    **circular_family_reality_check(
                        matrix,
                        selected_index,
                        block_days,
                    ),
                }
            )
    family = pd.DataFrame(family_rows)

    proxy_gold = proxy_closes["GOLD"].loc["2007-01-01":]
    events_rows: list[dict[str, object]] = []
    event_baseline = baseline_paths[
        ("proxy_synthetic", "current_liquidity")
    ]
    event_candidate = paths[
        ("proxy_synthetic", "current_liquidity", selected_name)
    ]
    event_diagnostics = diagnostics_by_sample[
        ("proxy_synthetic", "current_liquidity", selected_name)
    ]
    for episode in gold_drawdown_episodes(proxy_gold):
        peak = episode["peak_date"]
        trough = episode["trough_date"]
        assert isinstance(peak, pd.Timestamp)
        assert isinstance(trough, pd.Timestamp)
        dates = event_baseline.index[
            (event_baseline.index >= peak)
            & (event_baseline.index <= trough)
        ]
        if dates.empty:
            continue
        base_return = compounded_return(event_baseline.loc[dates])
        candidate_return = compounded_return(event_candidate.loc[dates])
        diagnostics = event_diagnostics.reindex(dates)
        events_rows.append(
            {
                "peak_date": peak.date().isoformat(),
                "trough_date": trough.date().isoformat(),
                "recovery_or_end_date": episode[
                    "recovery_or_end_date"
                ].date().isoformat(),
                "gold_drawdown": episode["gold_drawdown"],
                "baseline_return": base_return,
                "candidate_return": candidate_return,
                "candidate_minus_baseline_return": (
                    candidate_return - base_return
                ),
                "average_base_gold": float(
                    diagnostics["base_gold_weight"].mean()
                ),
                "average_candidate_gold": float(
                    diagnostics["candidate_gold_weight"].mean()
                ),
            }
        )
    events = pd.DataFrame(events_rows)

    normal_diagnostics = diagnostics_by_sample[
        ("normal_synthetic", "current_liquidity", selected_name)
    ].dropna(subset=["support_votes"])
    current_row = normal_diagnostics.iloc[-1]
    current_target: dict[str, object] = {
        "date": normal_diagnostics.index[-1].date().isoformat(),
        "candidate": selected_name,
        "support_votes": int(current_row["support_votes"]),
        "gold_excess_trend": float(current_row["gold_excess_trend"]),
        "real_yield_change": float(current_row["real_yield_change"]),
        "dollar_change": float(current_row["dollar_change"]),
        "base_gold_weight": float(current_row["base_gold_weight"]),
        "gold_multiplier": float(current_row["gold_multiplier"]),
        "candidate_gold_weight": float(
            current_row["candidate_gold_weight"]
        ),
        "gold_to_cash": float(current_row["gold_to_cash"]),
    }

    qualified, audit = evaluate_production_gate(
        selected_name,
        selection,
        metrics,
        family,
        events,
    )
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)
    weights_summary.to_csv(OUTPUT / "weight_summary.csv", index=False)
    selection.to_csv(OUTPUT / "candidate_selection.csv", index=False)
    family.to_csv(OUTPUT / "family_reality_check.csv", index=False)
    events.to_csv(OUTPUT / "selected_gold_events.csv", index=False)
    reconstruction.to_csv(
        OUTPUT / "baseline_reconstruction.csv",
        index=False,
    )
    audit.to_csv(OUTPUT / "production_gate.csv", index=False)
    (OUTPUT / "current_target.json").write_text(
        json.dumps(current_target, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for sample in samples:
        for scenario in COST_SCENARIOS:
            selected_path = paths[
                (sample, scenario.name, selected_name)
            ]
            pd.DataFrame(
                {
                    "baseline_return": baseline_paths[
                        (sample, scenario.name)
                    ].reindex(selected_path.index),
                    "candidate_return": selected_path,
                }
            ).to_csv(
                OUTPUT
                / f"{sample}_{scenario.name}_{selected_name}_daily.csv",
                index_label="date",
            )
    write_decision_report(
        selected,
        qualified,
        audit,
        metrics,
        events,
        current_target,
    )

    print(f"Selected: {selected_name}")
    print(f"Qualified for R12: {qualified}")
    print("\nSelection:")
    print(
        selection.loc[selection["selected"]]
        .round(6)
        .to_string(index=False)
    )
    print("\nProduction gate:")
    print(audit.to_string(index=False))
    print("\nCurrent target:")
    print(json.dumps(current_target, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
