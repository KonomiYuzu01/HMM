from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

from evaluate_smh_dynamic_guard import (
    ASSETS,
    OpenGapGuard,
    simulate,
    variants,
)
from evaluate_smh_dynamic_guard_robustness import event_clusters
from regime_strategy.report import performance_metrics


DESTINATION = Path("output/smh_long_proxy_validation")
ROOT = Path(
    "output/"
    "paper_core_growth_gold20_r9_staged_reentry_netted_ensemble_20y_proxy"
)
OPEN_CLOSE = Path("data/adjusted_open_close_20y_proxy.csv")
PROXY_CLOSE = Path("data/prices_20y_proxy.csv")
PERIODS = {
    "early_2002_2014": ("2002-07-30", "2014-12-31"),
    "gfc_2007_2009": ("2007-01-01", "2009-12-31"),
    "post_gfc_2010_2014": ("2010-01-01", "2014-12-31"),
    "development_2015_2021": ("2015-01-01", "2021-12-31"),
    "holdout_2022_present": ("2022-01-01", None),
    "complete_2002_present": ("2002-07-30", None),
}


def causal_rule(
    name: str,
    minimum_semis_weight: float = 0.60,
    relative_gap_trigger: float | None = None,
    overflow_asset: str = "CASH",
    maximum_hold_days: int = 4,
) -> OpenGapGuard:
    return OpenGapGuard(
        name=name,
        absolute_gap_trigger=-0.025,
        relative_gap_trigger=relative_gap_trigger,
        post_trigger_cap=0.25,
        trigger_delay_days=1,
        minimum_semis_weight=minimum_semis_weight,
        recovery_signal="relative",
        recovery_confirmations=2,
        maximum_hold_days=maximum_hold_days,
        overflow_asset=overflow_asset,
    )


def load_inputs(
    root: Path = ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    prices = pd.read_csv(OPEN_CLOSE, index_col=0, parse_dates=True)
    opens = prices[[f"open_{asset}" for asset in ASSETS]].copy()
    closes = prices[[f"close_{asset}" for asset in ASSETS]].copy()
    opens.columns = ASSETS
    closes.columns = ASSETS
    return weights, daily, opens.astype(float), closes.astype(float)


def metric_rows(
    name: str,
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for period, (start, end) in PERIODS.items():
        selected = candidate.loc[start:end, "net_return"]
        reference = baseline.loc[start:end, "net_return"]
        candidate_metrics = performance_metrics(selected)
        baseline_metrics = performance_metrics(reference)
        rows.append(
            {
                "variant": name,
                "period": period,
                **candidate_metrics,
                "baseline_cagr": baseline_metrics["cagr"],
                "baseline_max_drawdown": baseline_metrics["max_drawdown"],
                "cagr_delta_vs_baseline": (
                    candidate_metrics["cagr"]
                    - baseline_metrics["cagr"]
                ),
                "max_drawdown_delta_vs_baseline": (
                    candidate_metrics["max_drawdown"]
                    - baseline_metrics["max_drawdown"]
                ),
                "worst_day_delta_vs_baseline": float(
                    selected.min() - reference.min()
                ),
                "trigger_count": int(
                    candidate.loc[start:end, "triggered"].sum()
                ),
            }
        )
    return rows


def nonoverlapping_signal_positions(
    signal: pd.Series,
    maximum_gap_sessions: int = 5,
) -> list[int]:
    raw_positions = np.flatnonzero(signal.to_numpy(dtype=bool))
    selected: list[int] = []
    for position in raw_positions:
        if (
            not selected
            or position - selected[-1] > maximum_gap_sessions
        ):
            selected.append(int(position))
    return selected


def asset_event_study() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(OPEN_CLOSE, index_col=0, parse_dates=True)
    semis_gap = (
        prices["open_SEMIS"] / prices["close_SEMIS"].shift(1) - 1.0
    )
    qqq_gap = (
        prices["open_QQQ"] / prices["close_QQQ"].shift(1) - 1.0
    )
    signal = (semis_gap <= -0.025) & (
        semis_gap - qqq_gap <= -0.005
    )
    positions = nonoverlapping_signal_positions(signal)
    event_rows: list[dict[str, object]] = []
    for position in positions:
        if position + 5 >= len(prices):
            continue
        row: dict[str, object] = {
            "signal_date": prices.index[position],
            "execution_date": prices.index[position + 1],
            "semis_gap": float(semis_gap.iloc[position]),
            "relative_gap": float(
                semis_gap.iloc[position] - qqq_gap.iloc[position]
            ),
        }
        for horizon in range(1, 6):
            end = position + horizon
            semis_return = float(
                prices["close_SEMIS"].iloc[end]
                / prices["open_SEMIS"].iloc[position + 1]
                - 1.0
            )
            qqq_return = float(
                prices["close_QQQ"].iloc[end]
                / prices["open_QQQ"].iloc[position + 1]
                - 1.0
            )
            cash_return = float(
                prices["close_CASH"].iloc[end]
                / prices["open_CASH"].iloc[position + 1]
                - 1.0
            )
            row[f"cash_minus_semis_h{horizon}"] = (
                cash_return - semis_return
            )
            row[f"qqq_minus_semis_h{horizon}"] = (
                qqq_return - semis_return
            )
        event_rows.append(row)
    events = pd.DataFrame(event_rows)

    study_periods = {
        "early_2002_2014": ("2002-01-01", "2014-12-31"),
        "development_2015_2021": ("2015-01-01", "2021-12-31"),
        "holdout_2022_present": ("2022-01-01", None),
        "complete_2002_present": ("2002-01-01", None),
    }
    summary_rows: list[dict[str, object]] = []
    for period, (start, end) in study_periods.items():
        selected = events.loc[
            (events["signal_date"] >= start)
            & (
                True
                if end is None
                else events["signal_date"].le(end)
            )
        ]
        for horizon in range(1, 6):
            values = selected[
                f"cash_minus_semis_h{horizon}"
            ].to_numpy(dtype=float)
            positive = int((values > 0.0).sum())
            summary_rows.append(
                {
                    "period": period,
                    "horizon_days": horizon,
                    "events": len(values),
                    "mean_cash_minus_semis": float(values.mean()),
                    "median_cash_minus_semis": float(
                        np.median(values)
                    ),
                    "positive_share": float((values > 0.0).mean()),
                    "one_sided_sign_p_value": float(
                        binomtest(
                            positive,
                            len(values),
                            0.5,
                            alternative="greater",
                        ).pvalue
                    ),
                    "one_sided_mean_t_test_p_value": float(
                        ttest_1samp(
                            values,
                            0.0,
                            alternative="greater",
                        ).pvalue
                    ),
                    "mean_after_110bp_roundtrip": float(
                        values.mean() - 0.011
                    ),
                }
            )
    return events, pd.DataFrame(summary_rows)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    weights, base_daily, opens, closes = load_inputs()
    baseline, _ = simulate(
        weights,
        base_daily,
        opens,
        closes,
        OpenGapGuard("baseline"),
        start_date="2002-07-30",
    )
    guards = [
        causal_rule("causal_unfiltered"),
        causal_rule(
            "causal_filtered",
            relative_gap_trigger=-0.005,
        ),
        causal_rule(
            "causal_filtered_to_qqq",
            relative_gap_trigger=-0.005,
            overflow_asset="QQQ",
        ),
        causal_rule(
            "causal_filtered_hold1",
            relative_gap_trigger=-0.005,
            maximum_hold_days=1,
        ),
        causal_rule(
            "causal_filtered_hold2",
            relative_gap_trigger=-0.005,
            maximum_hold_days=2,
        ),
    ]
    for minimum_weight in (0.45, 0.50, 0.55):
        guards.append(
            causal_rule(
                f"causal_filtered_min{int(minimum_weight * 100)}",
                minimum_semis_weight=minimum_weight,
                relative_gap_trigger=-0.005,
            )
        )
    existing = {guard.name: guard for guard in variants()}
    guards.extend(
        [
            existing[
                "accountloss200bp_rel2_or220bp_cap25"
                "_continuegap1_deep15_hold5"
            ],
            existing["accountloss220bp_cap25_hold5"],
        ]
    )

    metric_frames: list[dict[str, float | int | str]] = []
    cycle_frames: list[pd.DataFrame] = []
    for guard in guards:
        candidate, _ = simulate(
            weights,
            base_daily,
            opens,
            closes,
            guard,
            start_date="2002-07-30",
        )
        metric_frames.extend(
            metric_rows(guard.name, candidate, baseline)
        )
        cycles = event_clusters(candidate, baseline, guard.name)
        if not cycles.empty:
            cycle_frames.append(cycles)
    metrics = pd.DataFrame(metric_frames)
    cycles = pd.concat(cycle_frames, ignore_index=True)
    metrics.to_csv(DESTINATION / "strategy_metrics.csv", index=False)
    cycles.to_csv(DESTINATION / "event_clusters.csv", index=False)

    seed_rows: list[dict[str, float | int | str]] = []
    filtered = causal_rule(
        "causal_filtered",
        relative_gap_trigger=-0.005,
    )
    for member_root in sorted((ROOT / "members").glob("seed_*")):
        member_weights, member_daily, _, _ = load_inputs(member_root)
        member_baseline, _ = simulate(
            member_weights,
            member_daily,
            opens,
            closes,
            OpenGapGuard("baseline"),
            start_date="2002-07-30",
        )
        member_candidate, _ = simulate(
            member_weights,
            member_daily,
            opens,
            closes,
            filtered,
            start_date="2002-07-30",
        )
        for row in metric_rows(
            "causal_filtered",
            member_candidate,
            member_baseline,
        ):
            seed_rows.append(
                {"member": member_root.name, **row}
            )
    pd.DataFrame(seed_rows).to_csv(
        DESTINATION / "seed_metrics.csv",
        index=False,
    )

    asset_events, asset_summary = asset_event_study()
    asset_events.to_csv(
        DESTINATION / "asset_event_study_events.csv",
        index=False,
    )
    asset_summary.to_csv(
        DESTINATION / "asset_event_study_summary.csv",
        index=False,
    )

    print("Fixed-rule strategy metrics:")
    print(
        metrics.loc[
            metrics["variant"].isin(
                ["causal_unfiltered", "causal_filtered"]
            ),
            [
                "variant",
                "period",
                "cagr_delta_vs_baseline",
                "max_drawdown_delta_vs_baseline",
                "trigger_count",
            ],
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nAsset event study, complete period:")
    print(
        asset_summary.loc[
            asset_summary["period"] == "complete_2002_present"
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nArtifacts:", DESTINATION.resolve())


if __name__ == "__main__":
    main()
