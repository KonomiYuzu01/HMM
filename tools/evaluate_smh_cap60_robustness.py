from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_open_execution import (
    load_open_close,
    simulate_open_execution,
)
from evaluate_smh_shock_guard import (
    ASSETS,
    Guard,
    SAMPLES,
    five_day_compound,
    load_sample,
    simulate,
)
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/smh_shock_guard")
STRATEGY = str(SAMPLES["normal"]["strategy"])
PERIODS = {
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_2025": ("2022-01-01", "2025-12-31"),
    "complete_2015_present": ("2015-01-01", None),
    "recent_2024_present": ("2024-01-01", None),
}
BASELINE = Guard("baseline")
CAP60 = Guard(
    "diversified_cap60_toQQQ",
    scheduled_cap=0.60,
    overflow_asset="QQQ",
)


def simulate_open_frames(
    weights: pd.DataFrame,
    daily: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    cost_bps: float,
    financing_spread_bps: float = 100.0,
    short_borrow_spread_bps: float = 100.0,
) -> pd.DataFrame:
    dates = (
        weights.index.intersection(daily.index)
        .intersection(opens.index)
        .intersection(closes.index)
    )
    assets = list(weights.columns)
    current = np.zeros(len(assets), dtype=float)
    current[assets.index("CASH")] = 1.0
    previous_date: pd.Timestamp | None = None
    rows: list[dict[str, float | int]] = []

    for date in dates:
        overnight_return = 0.0
        if previous_date is not None:
            overnight_asset_returns = (
                opens.loc[date, assets].to_numpy(dtype=float)
                / closes.loc[previous_date, assets].to_numpy(dtype=float)
                - 1.0
            )
            overnight_return = float(current @ overnight_asset_returns)
            current = current * (1.0 + overnight_asset_returns) / (
                1.0 + overnight_return
            )

        traded = float(daily.loc[date, "turnover"]) > 1e-14
        open_turnover = 0.0
        if traded:
            target = weights.loc[date, assets].to_numpy(dtype=float)
            open_turnover = 0.5 * float(np.abs(target - current).sum())
            current = target

        intraday_asset_returns = (
            closes.loc[date, assets].to_numpy(dtype=float)
            / opens.loc[date, assets].to_numpy(dtype=float)
            - 1.0
        )
        intraday_return = float(current @ intraday_asset_returns)
        trading_cost = 2.0 * open_turnover * cost_bps / 10_000.0
        financing_cost = (
            max(-float(current[assets.index("CASH")]), 0.0)
            * financing_spread_bps
            / 10_000.0
            / 252.0
        )
        risky = np.ones(len(current), dtype=bool)
        risky[assets.index("CASH")] = False
        financing_cost += (
            float(np.maximum(-current[risky], 0.0).sum())
            * short_borrow_spread_bps
            / 10_000.0
            / 252.0
        )
        net_return = (
            (1.0 + overnight_return) * (1.0 + intraday_return)
            - 1.0
            - trading_cost
            - financing_cost
        )

        gross_intraday_growth = 1.0 + intraday_return
        if gross_intraday_growth <= 1e-12:
            raise ValueError("Portfolio lost all capital intraday")
        current = (
            current
            * (1.0 + intraday_asset_returns)
            / gross_intraday_growth
        )
        rows.append(
            {
                "date": date,
                "net_return": net_return,
                "overnight_return_before_trade": overnight_return,
                "intraday_return_after_trade": intraday_return,
                "trading_cost": trading_cost,
                "financing_cost": financing_cost,
                "open_turnover": open_turnover,
                "traded_at_open": int(traded),
            }
        )
        previous_date = date

    result = pd.DataFrame(rows).set_index("date")
    result["equity"] = (1.0 + result["net_return"]).cumprod()
    result["drawdown"] = (
        result["equity"] / result["equity"].cummax() - 1.0
    )
    return result


def annualized_relative_log_return(
    candidate: pd.Series,
    baseline: pd.Series,
) -> float:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    return float(
        (
            np.log1p(aligned["candidate"])
            - np.log1p(aligned["baseline"])
        ).mean()
        * 252.0
    )


def circular_block_relative_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    block_days: int,
    samples: int = 20_000,
    seed: int = 20_260_727,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    relative = (
        np.log1p(aligned["candidate"])
        - np.log1p(aligned["baseline"])
    ).to_numpy(dtype=float)
    observations = len(relative)
    rng = np.random.default_rng(seed + block_days)
    block_count = int(np.ceil(observations / block_days))
    starts = rng.integers(
        0,
        observations,
        size=(samples, block_count),
    )
    offsets = np.arange(block_days)
    indices = (starts[:, :, None] + offsets) % observations
    totals = relative[indices].reshape(samples, -1)[:, :observations].sum(axis=1)
    annualized = totals / observations * 252.0
    return {
        "block_days": block_days,
        "samples": samples,
        "observed_annualized_relative_log_return": float(
            relative.mean() * 252.0
        ),
        "probability_positive": float((annualized > 0.0).mean()),
        "relative_log_return_5pct": float(np.quantile(annualized, 0.05)),
        "relative_log_return_median": float(np.median(annualized)),
        "relative_log_return_95pct": float(np.quantile(annualized, 0.95)),
    }


def metric_row(
    execution: str,
    cost_bps: float,
    period: str,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, float | str]:
    start, end = PERIODS[period]
    selected = candidate.loc[start:end]
    reference = baseline.loc[start:end]
    candidate_metrics = performance_metrics(selected["net_return"])
    baseline_metrics = performance_metrics(reference["net_return"])
    return {
        "execution": execution,
        "cost_bps": cost_bps,
        "period": period,
        **candidate_metrics,
        "baseline_cagr": baseline_metrics["cagr"],
        "baseline_max_drawdown": baseline_metrics["max_drawdown"],
        "cagr_delta_vs_baseline": (
            candidate_metrics["cagr"] - baseline_metrics["cagr"]
        ),
        "max_drawdown_delta_vs_baseline": (
            candidate_metrics["max_drawdown"]
            - baseline_metrics["max_drawdown"]
        ),
        "worst_day_delta_vs_baseline": float(
            selected["net_return"].min()
            - reference["net_return"].min()
        ),
        "worst_five_days_delta_vs_baseline": float(
            five_day_compound(selected["net_return"]).min()
            - five_day_compound(reference["net_return"]).min()
        ),
        "annualized_relative_log_return": annualized_relative_log_return(
            selected["net_return"],
            reference["net_return"],
        ),
    }


def load_member(
    seed: int,
    asset_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path("output") / STRATEGY / "members" / f"seed_{seed}"
    weights = pd.read_csv(
        root / "weights.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        root / "daily_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    return weights, daily, asset_returns.reindex(weights.index)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    base_weights, base_daily, asset_returns = load_sample("normal")
    opens, closes = load_open_close(refresh=False)
    reconstruction = simulate_open_frames(
        base_weights.loc[:, ASSETS],
        base_daily,
        opens,
        closes,
        cost_bps=7.5,
    )
    reference = simulate_open_execution(
        STRATEGY,
        opens,
        closes,
        start_date="2015-01-01",
        end_date=None,
        cost_bps=7.5,
    )
    common = reconstruction.index.intersection(reference.index)
    reconstruction_check = pd.DataFrame(
        [
            {
                "maximum_absolute_return_difference": float(
                    (
                        reconstruction.loc[common, "net_return"]
                        - reference.loc[common, "net_return"]
                    )
                    .abs()
                    .max()
                ),
                "maximum_absolute_turnover_difference": float(
                    (
                        reconstruction.loc[common, "open_turnover"]
                        - reference.loc[common, "open_turnover"]
                    )
                    .abs()
                    .max()
                ),
            }
        ]
    )

    metric_rows: list[dict[str, float | str]] = []
    standard_open: dict[str, pd.DataFrame] = {}
    standard_close: dict[str, pd.DataFrame] = {}
    standard_weights: dict[str, pd.DataFrame] = {}
    for cost_bps in (7.5, 15.0, 25.0):
        baseline_close, baseline_weights = simulate(
            base_weights,
            base_daily,
            asset_returns,
            BASELINE,
            cost_bps=cost_bps,
        )
        candidate_close, candidate_weights = simulate(
            base_weights,
            base_daily,
            asset_returns,
            CAP60,
            cost_bps=cost_bps,
        )
        baseline_open = simulate_open_frames(
            baseline_weights,
            baseline_close,
            opens,
            closes,
            cost_bps=cost_bps,
        )
        candidate_open = simulate_open_frames(
            candidate_weights,
            candidate_close,
            opens,
            closes,
            cost_bps=cost_bps,
        )
        for execution, candidate, baseline in (
            ("close_to_close", candidate_close, baseline_close),
            ("next_open_proxy", candidate_open, baseline_open),
        ):
            for period in PERIODS:
                metric_rows.append(
                    metric_row(
                        execution,
                        cost_bps,
                        period,
                        candidate,
                        baseline,
                    )
                )
        if cost_bps == 7.5:
            standard_open = {
                "baseline": baseline_open,
                "cap60": candidate_open,
            }
            standard_close = {
                "baseline": baseline_close,
                "cap60": candidate_close,
            }
            standard_weights = {
                "baseline": baseline_weights,
                "cap60": candidate_weights,
            }

    member_rows: list[dict[str, float | int | str]] = []
    for seed in (7, 42, 123):
        member_weights, member_daily, member_returns = load_member(
            seed,
            asset_returns,
        )
        baseline_close, baseline_weights = simulate(
            member_weights,
            member_daily,
            member_returns,
            BASELINE,
        )
        candidate_close, candidate_weights = simulate(
            member_weights,
            member_daily,
            member_returns,
            CAP60,
        )
        baseline_open = simulate_open_frames(
            baseline_weights,
            baseline_close,
            opens,
            closes,
            cost_bps=7.5,
        )
        candidate_open = simulate_open_frames(
            candidate_weights,
            candidate_close,
            opens,
            closes,
            cost_bps=7.5,
        )
        for period in PERIODS:
            row = metric_row(
                "next_open_proxy",
                7.5,
                period,
                candidate_open,
                baseline_open,
            )
            member_rows.append({"seed": seed, **row})

    bootstrap = pd.DataFrame(
        [
            circular_block_relative_bootstrap(
                standard_open["cap60"]["net_return"],
                standard_open["baseline"]["net_return"],
                block_days,
            )
            for block_days in (21, 63, 126)
        ]
    )
    cap_events = standard_weights["baseline"].loc[
        standard_weights["baseline"]["SEMIS"] > 0.60,
        ["QQQ", "SEMIS"],
    ].copy()
    cap_events["cap_excess"] = cap_events["SEMIS"] - 0.60
    cap_events["candidate_semis"] = standard_weights["cap60"].loc[
        cap_events.index,
        "SEMIS",
    ]
    cap_events["candidate_qqq"] = standard_weights["cap60"].loc[
        cap_events.index,
        "QQQ",
    ]

    metrics = pd.DataFrame(metric_rows)
    members = pd.DataFrame(member_rows)
    reconstruction_check.to_csv(
        DESTINATION / "cap60_open_reconstruction.csv",
        index=False,
    )
    metrics.to_csv(
        DESTINATION / "cap60_execution_metrics.csv",
        index=False,
    )
    members.to_csv(
        DESTINATION / "cap60_seed_robustness.csv",
        index=False,
    )
    bootstrap.to_csv(
        DESTINATION / "cap60_block_bootstrap.csv",
        index=False,
    )
    cap_events.to_csv(DESTINATION / "cap60_binding_dates.csv")
    standard_open["baseline"].to_csv(
        DESTINATION / "cap60_baseline_open_daily.csv"
    )
    standard_open["cap60"].to_csv(
        DESTINATION / "cap60_candidate_open_daily.csv"
    )
    standard_close["cap60"].to_csv(
        DESTINATION / "cap60_candidate_close_daily.csv"
    )

    summary = metrics.loc[
        (metrics["cost_bps"].isin([7.5, 15.0]))
        & (
            metrics["period"].isin(
                [
                    "development_2015_2021",
                    "holdout_2022_2025",
                    "complete_2015_present",
                    "recent_2024_present",
                ]
            )
        ),
        [
            "execution",
            "cost_bps",
            "period",
            "cagr_delta_vs_baseline",
            "max_drawdown_delta_vs_baseline",
            "worst_day_delta_vs_baseline",
            "worst_five_days_delta_vs_baseline",
        ],
    ]
    print("Reconstruction:")
    print(reconstruction_check.round(12).to_string(index=False))
    print("\nCap60 execution robustness:")
    print(summary.round(6).to_string(index=False))
    print("\nSeed robustness, complete period:")
    print(
        members.loc[
            members["period"] == "complete_2015_present",
            [
                "seed",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nBlock bootstrap:")
    print(bootstrap.round(6).to_string(index=False))
    print(f"\nBinding sessions: {len(cap_events)}")
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
