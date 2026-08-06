from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.evaluate_hmm_whitebox_diagnostics import ensemble_diagnostics, load_member_regimes
from tools.evaluate_r11_diversified_capital_grid import load_strategy_inputs, metric_delta
from tools.evaluate_r11_levered_diversified_strategy import COST_SCENARIOS
import tools.evaluate_r21_current_engine_industry_momentum as r21
import tools.evaluate_r38_accelerating_volatility_capacity_fill as accel
import tools.evaluate_r38_accelerating_volatility_capacity_fill_1375 as r38_baseline
import tools.evaluate_r38_convex_semiconductor_overlay as r38


OUTPUT = Path("output/r40_uncertainty_gated_capacity")
NORMAL_DIRECTORY = Path("output/research_hmm_gaussian_diagnostics")
PROXY_DIRECTORY = Path("output/research_hmm_gaussian_diagnostics_20y_proxy")
BASE_ACTIVE_MULTIPLIER = 1.30
INCREMENTAL_MULTIPLIER = 0.075


def confidence_from_directory(directory: Path) -> pd.Series:
    diagnostics = ensemble_diagnostics(load_member_regimes(directory))
    confidence = diagnostics["uncertainty_capacity_confidence"].clip(0.0, 1.0)
    confidence.name = "capacity_confidence"
    return confidence


def with_research_strategy_inputs(
    settings: dict[str, object],
    directory: Path,
) -> dict[str, object]:
    source = (
        str(directory.relative_to("output"))
        if directory.parts and directory.parts[0] == "output"
        else str(directory)
    )
    weights, daily = load_strategy_inputs(source)
    updated = deepcopy(settings)
    updated["weights"] = weights
    updated["daily"] = daily
    updated["directory"] = directory.name
    return updated


def simulate_candidate(
    settings: dict[str, object],
    scenario: object,
    confidence: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active_multiplier = (
        BASE_ACTIVE_MULTIPLIER
        + INCREMENTAL_MULTIPLIER * confidence.astype(float)
    )
    original = r38.r37.r33.one_session_shock_schedule

    def schedule(
        weights: pd.DataFrame,
        base_daily: pd.DataFrame,
        closes: pd.DataFrame,
        *,
        cash_floor: float = -0.20,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index = weights.index.intersection(closes.index)
        causal_multiplier = active_multiplier.reindex(index).ffill().fillna(
            BASE_ACTIVE_MULTIPLIER
        )
        return accel.accelerating_volatility_schedule(
            weights,
            base_daily,
            closes,
            cash_floor=cash_floor,
            low_vol_active_multiplier=causal_multiplier,
            high_vol_active_multiplier=BASE_ACTIVE_MULTIPLIER,
        )

    r38.r37.r33.one_session_shock_schedule = schedule
    try:
        path, diagnostics = r38.simulate_candidate(
            settings,
            scenario,
            base_multiplier=1.070,
            active_multiplier=1.375,
            shock_multiplier=1.00,
            cash_floor=-0.20,
            overlay_fraction=0.10,
        )
    finally:
        r38.r37.r33.one_session_shock_schedule = original
    diagnostics = diagnostics.copy()
    diagnostics["capacity_confidence"] = confidence.reindex(
        diagnostics.index
    ).ffill().fillna(0.0)
    diagnostics["confidence_active_multiplier"] = (
        BASE_ACTIVE_MULTIPLIER
        + INCREMENTAL_MULTIPLIER * diagnostics["capacity_confidence"]
    )
    return path, diagnostics


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = r21._build_samples()
    definitions = {
        "normal_synthetic": (
            with_research_strategy_inputs(
                samples["normal_synthetic"], NORMAL_DIRECTORY
            ),
            confidence_from_directory(NORMAL_DIRECTORY),
        ),
        "proxy_synthetic": (
            with_research_strategy_inputs(
                samples["proxy_synthetic"], PROXY_DIRECTORY
            ),
            confidence_from_directory(PROXY_DIRECTORY),
        ),
    }
    metric_rows: list[dict[str, object]] = []
    current_paths: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for sample, (settings, confidence) in definitions.items():
        periods = settings["periods"]
        assert isinstance(periods, dict)
        for scenario in COST_SCENARIOS:
            baseline, baseline_diagnostics = r38_baseline.simulate_candidate(
                settings,
                scenario,
            )
            candidate, candidate_diagnostics = simulate_candidate(
                settings,
                scenario,
                confidence,
            )
            common = baseline.index.intersection(candidate.index)
            for period, (start, end) in periods.items():
                selected = common[(common >= start) & (common <= end)]
                metric_rows.append(
                    {
                        "sample": sample,
                        "scenario": scenario.name,
                        "period": period,
                        **metric_delta(
                            baseline.loc[selected, "net_return"],
                            candidate.loc[selected, "net_return"],
                        ),
                    }
                )
            if scenario.name == "current_liquidity":
                current_paths[sample] = (
                    baseline,
                    candidate,
                    candidate_diagnostics,
                )
                baseline.to_csv(OUTPUT / f"{sample}_baseline_daily.csv")
                candidate.to_csv(OUTPUT / f"{sample}_candidate_daily.csv")
                candidate_diagnostics.to_csv(
                    OUTPUT / f"{sample}_diagnostics.csv"
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_by_period.csv", index=False)

    def selected_delta(
        sample: str,
        scenario: str,
        period: str,
        field: str,
    ) -> float:
        row = metrics.loc[
            metrics["sample"].eq(sample)
            & metrics["scenario"].eq(scenario)
            & metrics["period"].eq(period)
        ].iloc[0]
        return float(row[field])

    gates = {
        "development_cagr_nonnegative": selected_delta(
            "normal_synthetic", "current_liquidity", "development_2015_2021", "cagr_delta"
        ) >= 0.0,
        "holdout_cagr_nonnegative": selected_delta(
            "normal_synthetic", "current_liquidity", "holdout_2022_2025", "cagr_delta"
        ) >= 0.0,
        "complete_cagr_nonnegative": selected_delta(
            "normal_synthetic", "current_liquidity", "complete_2015_2026", "cagr_delta"
        ) >= 0.0,
        "complete_drawdown_within_25bp": selected_delta(
            "normal_synthetic", "current_liquidity", "complete_2015_2026", "max_drawdown_delta"
        ) >= -0.0025,
        "proxy_cagr_nonnegative": selected_delta(
            "proxy_synthetic", "current_liquidity", "complete_2006_2026", "cagr_delta"
        ) >= 0.0,
        "stress_cagr_nonnegative": selected_delta(
            "normal_synthetic", "cost_stress", "complete_2015_2026", "cagr_delta"
        ) >= 0.0,
    }
    acceptance = pd.DataFrame(
        [{"gate": gate, "passed": passed} for gate, passed in gates.items()]
    )
    acceptance.to_csv(OUTPUT / "acceptance.csv", index=False)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "research_only",
        "production_changed": False,
        "orders_generated": False,
        "central_definition": (
            "1.30 + 0.075 * mean posterior margin * absolute three-seed vote margin"
        ),
        "gates": gates,
        "research_pass": bool(all(gates.values())),
        "decision": (
            "continue_forward_shadow" if all(gates.values()) else "reject_current_specification"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
