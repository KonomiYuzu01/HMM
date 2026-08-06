from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluate_smh_dynamic_guard import (
    OpenGapGuard,
    add_deltas,
    load_inputs,
    metrics,
    simulate,
)


DESTINATION = Path("output/smh_causal_delayed_grid")


def variants() -> list[OpenGapGuard]:
    candidates = [OpenGapGuard("baseline")]
    for absolute_gap in (
        -0.0200,
        -0.0225,
        -0.0250,
        -0.0275,
        -0.0300,
        -0.0325,
        -0.0350,
    ):
        gap_label = int(round(abs(absolute_gap) * 10_000))
        for minimum_semis_weight in (0.45, 0.50, 0.55, 0.60):
            weight_label = int(round(minimum_semis_weight * 100))
            for post_trigger_cap in (0.15, 0.20, 0.25, 0.30, 0.35):
                cap_label = int(round(post_trigger_cap * 100))
                for overflow_asset in ("CASH", "QQQ"):
                    overflow_label = (
                        "" if overflow_asset == "CASH" else "_toQQQ"
                    )
                    for recovery_confirmations in (1, 2):
                        for maximum_hold_days in (3, 4, 5, 6):
                            candidates.append(
                                OpenGapGuard(
                                    name=(
                                        f"causal_gap{gap_label}bp"
                                        f"_min{weight_label}"
                                        f"_cap{cap_label}"
                                        f"_confirm{recovery_confirmations}"
                                        f"_hold{maximum_hold_days}"
                                        f"{overflow_label}"
                                    ),
                                    absolute_gap_trigger=absolute_gap,
                                    post_trigger_cap=post_trigger_cap,
                                    trigger_delay_days=1,
                                    minimum_semis_weight=(
                                        minimum_semis_weight
                                    ),
                                    recovery_signal="relative",
                                    recovery_confirmations=(
                                        recovery_confirmations
                                    ),
                                    maximum_hold_days=maximum_hold_days,
                                    overflow_asset=overflow_asset,
                                )
                            )
    return candidates


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    weights, base_daily, opens, closes = load_inputs()
    all_rows: list[dict[str, float | int | str]] = []
    baseline_returns: pd.Series | None = None

    for guard in variants():
        daily, candidate_weights = simulate(
            weights,
            base_daily,
            opens,
            closes,
            guard,
        )
        if guard.name == "baseline":
            baseline_returns = daily["net_return"].copy()
        if baseline_returns is None:
            raise RuntimeError("Baseline must be evaluated first")
        all_rows.extend(
            metrics(
                guard,
                daily,
                candidate_weights,
                baseline_returns,
            )
        )

    result = add_deltas(pd.DataFrame(all_rows))
    result.to_csv(DESTINATION / "metrics.csv", index=False)
    pivot = result.pivot(
        index="variant",
        columns="period",
        values="cagr_delta_vs_baseline",
    )
    full = result.loc[
        result["period"] == "complete_2015_present"
    ].set_index("variant")
    summary = pivot.join(
        full[
            [
                "positive_capture",
                "max_drawdown_delta_vs_baseline",
                "worst_day_delta_vs_baseline",
                "trigger_count",
                "annualized_turnover",
            ]
        ]
    )
    periods = [
        "development_2015_2021",
        "holdout_2022_2025",
        "complete_2015_present",
        "recent_2024_present",
    ]
    accepted = summary.loc[
        (summary[periods] > 0.0).all(axis=1)
    ].sort_values(
        [
            "positive_capture",
            "complete_2015_present",
        ],
        ascending=False,
    )
    accepted.to_csv(DESTINATION / "accepted.csv")
    development = result.loc[
        result["period"] == "development_2015_2021"
    ].copy()
    development_eligible = development.loc[
        (development["variant"] != "baseline")
        & (development["cagr_delta_vs_baseline"] > 0.0)
        & (development["positive_capture"] >= 0.995)
        & (development["max_drawdown_delta_vs_baseline"] >= -1e-12)
        & (development["worst_day_delta_vs_baseline"] >= -1e-12)
    ].sort_values(
        [
            "cagr_delta_vs_baseline",
            "positive_capture",
        ],
        ascending=False,
    )
    selected_by_development = (
        result.loc[
            result["variant"].isin(
                development_eligible.head(10)["variant"]
            )
        ]
        .sort_values(["variant", "period"])
        .copy()
    )
    selected_by_development.to_csv(
        DESTINATION / "selected_by_development.csv",
        index=False,
    )
    print("Highest-capture all-period-positive candidates:")
    print(accepted.head(50).round(6).to_string())
    print("\nTop candidates selected using development period only:")
    print(
        development_eligible.head(10)[
            [
                "variant",
                "cagr_delta_vs_baseline",
                "positive_capture",
                "max_drawdown_delta_vs_baseline",
                "worst_day_delta_vs_baseline",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nAccepted: {len(accepted)} / {len(summary) - 1}")
    print(f"Artifacts: {DESTINATION.resolve()}")


if __name__ == "__main__":
    main()
