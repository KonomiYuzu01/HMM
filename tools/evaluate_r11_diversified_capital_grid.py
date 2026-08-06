from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
OUTPUT = Path("output/r11_diversified_capital_grid")
SEMIS_CAPS = (0.50, 0.60, 0.70)
GDE_FRACTIONS = (0.60, 0.75, 1.00)
GDE_NO_TRADE_BAND = 0.02


@dataclass(frozen=True)
class CostScenario:
    name: str
    base_one_way_cost_bps: float
    gde_one_way_cost_bps: float
    financing_spread_bps: float


COST_SCENARIOS = (
    CostScenario("current_liquidity", 7.5, 40.0, 100.0),
    CostScenario("cost_stress", 15.0, 75.0, 150.0),
)


def load_strategy_inputs(
    strategy_directory: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path("output") / strategy_directory
    weights = pd.read_csv(
        source / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        source / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    return weights, daily


def cap_semis_to_qqq(
    weights: pd.DataFrame,
    semis_cap: float,
) -> pd.DataFrame:
    if not 0.0 <= semis_cap <= 1.0:
        raise ValueError("semis_cap must be in [0, 1]")
    adjusted = weights.loc[:, ASSETS].copy()
    excess = (adjusted["SEMIS"] - semis_cap).clip(lower=0.0)
    adjusted["SEMIS"] -= excess
    adjusted["QQQ"] += excess
    return adjusted


def scale_non_cash_weights(
    weights: pd.DataFrame,
    risk_multiplier: float,
    unscaled_assets: tuple[str, ...] = (),
) -> pd.DataFrame:
    if risk_multiplier < 0.0:
        raise ValueError("risk_multiplier must be non-negative")
    invalid = [
        asset
        for asset in unscaled_assets
        if asset not in ASSETS or asset == "CASH"
    ]
    if invalid:
        raise ValueError(f"Invalid unscaled assets: {invalid}")
    adjusted = weights.loc[:, ASSETS].copy()
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    scaled_assets = [
        asset for asset in non_cash if asset not in unscaled_assets
    ]
    adjusted.loc[:, scaled_assets] *= risk_multiplier
    adjusted["CASH"] = 1.0 - adjusted.loc[:, non_cash].sum(axis=1)
    return adjusted


def scale_non_cash_weights_by_series(
    weights: pd.DataFrame,
    risk_multipliers: pd.Series,
) -> pd.DataFrame:
    aligned = risk_multipliers.reindex(weights.index)
    if aligned.isna().any():
        raise ValueError("risk_multipliers must cover every weight date")
    if aligned.lt(0.0).any():
        raise ValueError("risk_multipliers must be non-negative")
    adjusted = weights.loc[:, ASSETS].copy()
    non_cash = [asset for asset in ASSETS if asset != "CASH"]
    adjusted.loc[:, non_cash] = adjusted.loc[:, non_cash].mul(
        aligned,
        axis=0,
    )
    adjusted["CASH"] = 1.0 - adjusted.loc[:, non_cash].sum(axis=1)
    return adjusted


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


def relative_log_return(
    baseline: pd.Series,
    candidate: pd.Series,
) -> np.ndarray:
    aligned = pd.concat(
        [
            baseline.rename("baseline"),
            candidate.rename("candidate"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    return (
        np.log1p(aligned["candidate"].to_numpy(dtype=float))
        - np.log1p(aligned["baseline"].to_numpy(dtype=float))
    )


def circular_family_reality_check(
    relative_log_returns: np.ndarray,
    selected_index: int,
    block_days: int,
    samples: int = 5_000,
    seed: int = 20_260_727,
    batch_size: int = 50,
) -> dict[str, float | int]:
    if relative_log_returns.ndim != 2:
        raise ValueError("Relative returns must be observations by candidates")
    observations, candidates = relative_log_returns.shape
    if not 0 <= selected_index < candidates:
        raise ValueError("Selected index is outside the candidate matrix")
    if not 1 <= block_days <= observations:
        raise ValueError("Block length must fit inside the sample")

    observed = relative_log_returns.mean(axis=0) * 252.0
    extended = np.vstack(
        [relative_log_returns, relative_log_returns[: block_days - 1]]
    )
    cumulative = np.vstack(
        [
            np.zeros((1, candidates), dtype=float),
            np.cumsum(extended, axis=0),
        ]
    )
    block_sums = (
        cumulative[block_days : block_days + observations]
        - cumulative[:observations]
    )
    full_blocks, remainder = divmod(observations, block_days)
    remainder_sums = (
        cumulative[remainder : remainder + observations]
        - cumulative[:observations]
        if remainder
        else np.empty((0, candidates), dtype=float)
    )
    generator = np.random.default_rng(seed + block_days)
    maximum_statistics = np.empty(samples, dtype=float)
    selected_statistics = np.empty(samples, dtype=float)
    blocks_per_sample = full_blocks + int(remainder > 0)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        starts = generator.integers(
            0,
            observations,
            size=(stop - start, blocks_per_sample),
        )
        totals = block_sums[starts[:, :full_blocks]].sum(axis=1)
        if remainder:
            totals += remainder_sums[starts[:, -1]]
        null_statistics = totals / observations * 252.0 - observed
        maximum_statistics[start:stop] = null_statistics.max(axis=1)
        selected_statistics[start:stop] = null_statistics[:, selected_index]

    selected_observed = float(observed[selected_index])
    return {
        "block_days": block_days,
        "bootstrap_samples": samples,
        "observations": observations,
        "unique_candidate_paths": candidates,
        "selected_annualized_relative_log_return": selected_observed,
        "nominal_one_sided_p_value": float(
            (
                np.count_nonzero(
                    selected_statistics >= selected_observed
                )
                + 1
            )
            / (samples + 1)
        ),
        "familywise_reality_check_p_value": float(
            (
                np.count_nonzero(
                    maximum_statistics >= selected_observed
                )
                + 1
            )
            / (samples + 1)
        ),
        "null_maximum_95pct": float(
            np.quantile(maximum_statistics, 0.95)
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    normal_opens, normal_closes = load_adjusted_open_close(
        NORMAL_OPEN_CLOSE
    )
    proxy_opens, proxy_closes = load_adjusted_open_close(PROXY_OPEN_CLOSE)
    live_opens, live_closes = join_live_gde(
        normal_opens,
        normal_closes,
    )
    normal_weights, normal_daily = load_strategy_inputs(NORMAL_DIRECTORY)
    proxy_weights, proxy_daily = load_strategy_inputs(PROXY_DIRECTORY)
    samples = {
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
                "early_2006_2014": ("2006-08-01", "2014-12-31"),
                "late_2015_2026": ("2015-01-01", "2026-12-31"),
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
                "live_2022": ("2022-03-17", "2022-12-31"),
                "live_2023_2024": ("2023-01-01", "2024-12-31"),
                "live_2025_2026": ("2025-01-01", "2026-12-31"),
                "live_2022_2026": ("2022-03-17", "2026-12-31"),
            },
        },
    }
    rows: list[dict[str, float | str]] = []
    family_rows: list[dict[str, float | int | str]] = []
    current_paths: dict[str, dict[str, pd.Series]] = {}
    current_baselines: dict[str, pd.Series] = {}

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
        sample_paths: dict[str, pd.Series] = {}
        for scenario in COST_SCENARIOS:
            baseline = simulate_gde_substitution(
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
            if scenario.name == "current_liquidity":
                current_baselines[sample] = baseline["net_return"]
            for semis_cap in SEMIS_CAPS:
                adjusted_weights = cap_semis_to_qqq(weights, semis_cap)
                for gde_fraction in GDE_FRACTIONS:
                    candidate_name = (
                        f"cap{int(round(semis_cap * 100))}"
                        f"_gde{int(round(gde_fraction * 100))}"
                    )
                    candidate = simulate_gde_substitution(
                        str(settings["directory"]),
                        opens,
                        closes,
                        substitution_fraction=gde_fraction,
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
                        weights_override=adjusted_weights,
                        daily_override=daily,
                        gde_no_trade_band=GDE_NO_TRADE_BAND,
                    )
                    common = baseline.index.intersection(candidate.index)
                    for period, (period_start, period_end) in periods.items():
                        selected = common[
                            (common >= period_start)
                            & (common <= period_end)
                        ]
                        rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario.name,
                                "candidate": candidate_name,
                                "semis_cap": semis_cap,
                                "gde_fraction": gde_fraction,
                                "gde_no_trade_band": (
                                    GDE_NO_TRADE_BAND
                                ),
                                "period": period,
                                **metric_delta(
                                    baseline.loc[selected, "net_return"],
                                    candidate.loc[selected, "net_return"],
                                ),
                            }
                        )
                    if scenario.name == "current_liquidity":
                        sample_paths[candidate_name] = candidate[
                            "net_return"
                        ]
                        candidate.to_csv(
                            OUTPUT
                            / f"{sample}_{candidate_name}_daily.csv",
                            index_label="date",
                        )
        current_paths[sample] = sample_paths

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    for sample in ("normal_synthetic", "proxy_synthetic"):
        baseline = current_baselines[sample]
        names = list(current_paths[sample])
        matrix = np.column_stack(
            [
                relative_log_return(
                    baseline,
                    current_paths[sample][name],
                )
                for name in names
            ]
        )
        for selected_index, name in enumerate(names):
            for block_days in (21, 63, 126):
                family_rows.append(
                    {
                        "sample": sample,
                        "candidate": name,
                        "declared_family_size": len(names),
                        **circular_family_reality_check(
                            matrix,
                            selected_index,
                            block_days,
                        ),
                    }
                )
    family = pd.DataFrame(family_rows)
    family.to_csv(OUTPUT / "declared_family_reality_check.csv", index=False)

    full_periods = {
        "complete_2015_2026",
        "complete_2006_2026",
        "live_2022_2026",
    }
    print("Full-period metrics:")
    print(
        metrics.loc[metrics["period"].isin(full_periods)]
        [
            [
                "sample",
                "scenario",
                "candidate",
                "cagr_delta",
                "sharpe_delta",
                "candidate_max_drawdown",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nDeclared-family Reality Check:")
    print(
        family[
            [
                "sample",
                "candidate",
                "block_days",
                "selected_annualized_relative_log_return",
                "familywise_reality_check_p_value",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nArtifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
